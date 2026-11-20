"""HEALPix stereo entry point. All trainable layers come from OmniDS."""
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from module_cnn_only.network import OmniDS
from module_cnn_only.similarity_context import GEVModule, groupwise_correlation_volume

from .geometry import at_depth, at_parent_depth
from .ops import (
    LevelOps, ParentAlignedGEV, aggregate_profile, clear, convex_children, packed,
    restore, sequential_spatial, update_state,
)


def build_stereo_model(options=None, *, healpix_geometry=None, grid_package=None, local_cameras=None):
    options = dict(options or {})
    if options.get("use_healpix", False):
        if options.get("use_yinyang", False) or grid_package is not None:
            raise ValueError("use_healpix and use_yinyang are mutually exclusive")
        return HealpixOmniDS(healpix_geometry, options)
    if healpix_geometry is not None:
        raise ValueError("Set use_healpix to pass HEALPix geometry")
    if local_cameras is None:
        return OmniDS(options, grid_package=grid_package)
    return OmniDS(options, grid_package=grid_package, local_cameras=local_cameras)


def transfer_options(saved, use_dav2=False, **overrides):
    """Explicitly transfer a baseline configuration to the new topology."""
    return {**dict(saved or {}), "use_healpix": True, "use_yinyang": False,
            "use_local_spherical_encoder": False, "use_dav2_geometry": use_dav2,
            "reference_grid": "left_fisheye", **overrides}


def sample_static(feature, grid):
    b, channels = feature.shape[:2]
    faces, h, w, _ = grid.shape
    with torch.autocast(device_type=feature.device.type, enabled=False):
        grid = grid.reshape(1, faces*h, w, 2).expand(b, -1, -1, -1)
        value = F.grid_sample(feature.float(), grid, align_corners=True)
    return value.reshape(b, channels, faces, h, w).transpose(1, 2).to(feature.dtype)


def sample_target(feature, grid):
    b, channels = feature.shape[:2]
    d, faces, h, w, _ = grid.shape
    with torch.autocast(device_type=feature.device.type, enabled=False):
        grid = grid.permute(1, 2, 3, 0, 4).reshape(1, faces*h*w, d, 2).expand(b, -1, -1, -1)
        value = F.grid_sample(feature.float(), grid, align_corners=True)
    value = value.reshape(b, channels, faces, h, w, d).permute(0, 2, 1, 5, 3, 4)
    return value.to(feature.dtype)


def raw_matching(left, right, valid):
    b, faces = left.shape[:2]
    # Mask before multiplying: even a zero output mask cannot recover from
    # NaN or Inf in a product's backward pass.
    a, z = [packed(clear(t, valid)).permute(0, 1, 3, 4, 2) for t in (left, right)]
    gwc = groupwise_correlation_volume(a, z, 8)
    gwc = restore(gwc, b, faces).permute(0, 1, 2, 5, 3, 4)
    # The stock two-camera branch of SimilarityProfileEncoder uses precisely
    # this full-channel normalized correlation on the same descriptors.
    profile = (a * z).sum(dim=1) / a.new_tensor(a.shape[1]).sqrt()
    profile = restore(profile, b, faces).permute(0, 1, 4, 2, 3)
    return clear(gwc, valid), clear(profile, valid.squeeze(2))


class HealpixOmniDS(OmniDS):
    """Dedicated topology with direct transfer of the trained stereo submodules."""
    def __init__(self, geometry, options=None):
        options = dict(options or {})
        if geometry is None:
            raise ValueError("A calibrated HealpixGeometry is required")
        if options.get("use_yinyang", False) or options.get("use_local_spherical_encoder", False):
            raise ValueError("HEALPix cannot use the Yin–Yang or Global–Local branch")
        for key, expected in (("num_cams", 2), ("base_channel", 32), ("num_invdepth", 96),
                              ("num_depth_samples", 96), ("gev_num_groups", 8), ("num_downsample", 1)):
            if options.get(key, expected) != expected:
                raise ValueError(f"HEALPix requires {key}={expected}")
        if options.get("use_group_variance", False) or options.get("encoder_downsample_twice", False):
            raise ValueError("HEALPix requires stereo GWC and the original stride-2 encoder")
        if options.get("spatial_agg_type", "conv") != "conv":
            raise ValueError("HEALPix only supports standard SpatialProfileAggregator Conv2D")
        if options.get("similarity_type", "correlation") != "correlation":
            raise ValueError("HEALPix expects the trained stereo correlation")
        if options.get("triton", False) or options.get("use_triton", False):
            raise ValueError("The legacy Triton path is unavailable in HEALPix stereo")
        if options.get("healpix_nside", geometry.nside) != geometry.nside:
            raise ValueError("Configured HEALPix Nside and sweep geometry differ")
        options.update(use_healpix=True, use_yinyang=False, use_geometry_refiner=False, reference_grid="left_fisheye",
                       healpix_nside=geometry.nside)
        super().__init__(options)
        from earth2grid import healpix as earth

        self.geometry = geometry
        self.ops = LevelOps(earth)
        self.register_buffer("inverse_depth_candidates", geometry.axis.clone())
        self.depth_chunk = int(self.opts.get("depth_chunk", 8))
        if self.depth_chunk < 1:
            raise ValueError("depth_chunk must be positive")

    def load_pretrained(self, state):
        """Strictly transfer matching tensors and account for topology-only keys."""
        source = {name.removeprefix("module."): tensor for name, tensor in state.items()}
        current = self.state_dict()
        allowed_drop = ("enhancer.", "warp.", "fusion.", "training_geometry.", "local_spherical.")
        if not self.opts.use_dav2_geometry:
            allowed_drop += ("dav2_geometry.", "geometry_fusion.")
        unexpected = set(source) - set(current)
        unknown = sorted(key for key in unexpected if not key.startswith(allowed_drop))
        if unknown:
            raise ValueError(f"Unexpected pretrained keys: {unknown}")
        axis = "inverse_depth_candidates"
        if axis in source and not torch.equal(source[axis].cpu(), current[axis].cpu()):
            raise ValueError("Pretrained checkpoint and calibrated HEALPix depth axes differ")
        missing = set(current) - set(source) - {axis}
        if missing:
            raise ValueError(f"Missing pretrained keys: {sorted(missing)}")
        inherited = set(source) & set(current)
        self.load_state_dict({**current, **{key: source[key] for key in inherited}}, strict=True)
        return {"inherited": len(inherited), "discarded": sorted(unexpected), "initialized": []}

    def _maybe_checkpoint(self, function, *args):
        if self.opts.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def _chunk(self, static, target, grid, valid):
        right = sample_target(target, grid)
        left = static.unsqueeze(3).expand_as(right)
        return raw_matching(left, right, valid)

    def _matching(self, features, static, valid):
        depth = self.opts.num_invdepth
        costs, profiles = [], []
        for start in range(0, depth, self.depth_chunk):
            stop = min(start + self.depth_chunk, depth)
            cost, profile = self._maybe_checkpoint(self._chunk, static, features[1],
                                                   self.geometry.right_grid[start:stop], valid[:, :, :, start:stop])
            costs.append(cost)
            profiles.append(profile)
        profile = aggregate_profile(self.similarity_context.spatial_aggregator, self.ops,
                                    torch.cat(profiles, dim=2), valid.squeeze(2))
        # The raw left feature is constant along D. Its zero-masked mean is
        # the static feature times the fraction of supported candidates.
        fraction = (valid.sum(dim=3, dtype=torch.float32) / depth).to(static.dtype)
        guide = clear(static, valid.any(dim=3)) * fraction
        return torch.cat(costs, dim=3), profile, guide

    def _context(self, static, left, pixels, features, dav2_rgb):
        if self.opts.use_dav2_geometry:
            if dav2_rgb is None:
                raise ValueError("The DA-V2 branch requires officially preprocessed dav2_rgb")
            from module_cnn_only.dav2_geometry import vit_sampling_grid

            b, faces, _, h, w = static.shape
            patch = self.dav2_geometry.extract_features(dav2_rgb)
            grid = vit_sampling_grid(self.geometry.left_grid, (768, 800), features[0].shape[-2:],
                                     dav2_rgb.shape[-2:]).reshape(1, faces*h, w, 2)
            mapped = F.grid_sample(patch, grid.to(patch.dtype).expand(b, -1, -1, -1),
                                   align_corners=False, padding_mode="border")
            mapped = clear(mapped.reshape(b, -1, faces, h, w).transpose(1, 2), left)
            refine = sequential_spatial(self.dav2_geometry.refine.convs, self.ops, mapped, left)
            pair = torch.cat((static, (mapped + refine).relu()), dim=2)
            residual = sequential_spatial(self.geometry_fusion.residual, self.ops, pair, left)
            static = clear(static + self.geometry_fusion.alpha * residual, left)

        def projected(layer):
            value = self.ops.conv2d(static, left, layer.conv)
            if layer.opts.bn:
                value = restore(layer.bn(packed(value)), *static.shape[:2])
            return value

        return clear(projected(self.context_proj).relu(), pixels), clear(projected(self.state_proj).tanh(), pixels)

    def _step(self, state, sim, context, gev, depth, active):
        return update_state(self.update_block, self.ops, state, sim, context, gev, depth, active)

    def _render(self, depth, state, active, candidates):
        logits = .25 * sequential_spatial(self.update_block.mask, self.ops, state, active)
        allowed = active & at_depth(candidates.squeeze(2), depth, dim=2)
        child, support = convex_children(self.ops, depth, logits, allowed)
        mask = self.geometry.child_candidates[None].expand(child.shape[0], -1, -1, -1, -1)
        support = support & at_depth(mask, child, dim=2)
        support = support & at_parent_depth(candidates.squeeze(2), child)
        return self.geometry.project(child, support)

    def forward(self, imgs, *, iters=None, test_mode=False, dav2_rgb=None, return_details=False):
        iters = self.opts.gru_iters if iters is None else iters
        if iters < 1 or len(imgs) != 2 or imgs[0].shape != imgs[1].shape or imgs[0].ndim != 4:
            raise ValueError("Two equal [B,C,768,800] inputs and at least one iteration are required")
        if imgs[0].shape[-2:] != (768, 800) or imgs[0].shape[1] != (3 if self.opts.use_rgb else 1):
            raise ValueError("Inputs must use the original image dimensions and checkpoint channel mode")
        b = imgs[0].shape[0]
        with torch.autocast(device_type=imgs[0].device.type,
                            enabled=self.opts.mixed_precision or torch.is_autocast_enabled(imgs[0].device.type)):
            features = self.encoder(imgs)
            if features[0].shape[1:] != (32, 384, 400):
                raise ValueError("Incorrect CNN feature shape for calibrated stride-2 lookup")
            left = self.geometry.left_valid[None, :, None].expand(b, -1, -1, -1, -1)
            valid = self.geometry.stereo_valid.permute(1, 0, 2, 3)[None, :, None].expand(b, -1, -1, -1, -1, -1)
            static = clear(sample_static(features[0], self.geometry.left_grid), left)
            cost, profile, guide = self._matching(features, static, valid)
            adapter = ParentAlignedGEV(self.gev_module, self.ops)
            pyramid, audit = self._maybe_checkpoint(adapter, cost, guide, valid, return_details)
            candidates = pyramid[0][1] & valid
            active = left & candidates.any(dim=3)
            context, state = self._context(static, left, active, features, dav2_rgb)
            current = state.new_full((b, 12, 1, *state.shape[-2:]), 48.)
            lookup = SimpleNamespace(_gev_pyramid=[packed(v) for v, _ in pyramid],
                                     _valid_pyramid=[packed(m) for _, m in pyramid])
            predictions, output_masks = [], []
            for i in range(iters):
                current = current.detach()
                sim = self.similarity_context.lookup(packed(profile), packed(current), packed(valid))
                gev = GEVModule.lookup(lookup, packed(current), radius=self._gev_radius)
                state, delta = self._maybe_checkpoint(self._step, state, restore(sim, b, 12), context,
                                                      restore(gev, b, 12), current, active)
                current = clear((current + delta).clamp(0., 95.), active)
                if not test_mode or i == iters - 1:
                    estimate, observed = self._render(current, state, active, candidates)
                    predictions.append(estimate)
                    output_masks.append(observed)
        self.output_valid = output_masks[-1]
        if return_details:
            inverse, radial = self.index_to_depth(predictions[-1], output_masks[-1])
            return {"predictions": predictions, "output_masks": output_masks, "index": predictions[-1],
                    "inverse_depth": inverse, "radial_depth": radial, "valid": output_masks[-1],
                    "native_index": current, "hidden": state, "native_valid": active,
                    "candidate_valid": candidates, "pyramid_audit": audit, "profile_shape": list(profile.shape),
                    "cost_shape": list(cost.shape)}
        return predictions[-1] if test_mode else predictions
