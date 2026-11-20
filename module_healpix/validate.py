"""Calibrated direct-GWC HEALPix checks and backward without an optimizer."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.healpix_probe import geometry_probe
# Stage 1's executable probe imports its sibling geometry_probe by filename.
# Bind that location only in the validation entry point.
sys.path.insert(0, str(Path(geometry_probe.__file__).resolve().parent))
try:
    from experiments.healpix_probe.volumes import correlate, sample_volumes
finally:
    sys.path.pop(0)
from experiments.local_spherical_grid.camera_geometry import load_pair
from module_cnn_only.network import OmniDS
from module_cnn_only.similarity_context import DownBlock3D
from utils.image import normalizeImage, rgb2gray

from .geometry import HealpixGeometry, pad_xy_ids
from .network import build_stereo_model, raw_matching, sample_static, sample_target, transfer_options
from .ops import ParentAlignedGEV, clear, convex_children, packed, update_state
from experiments.healpix_probe.stage2.face_gev import FaceOps


def paired_images(cameras, images, rgb, device):
    values = []
    for cam, image in zip(cameras, images):
        pixels = image if rgb else rgb2gray(image, channel_wise_mean=True)
        values.append(normalizeImage(pixels, cam.invalid_mask))
    return [torch.from_numpy(value.transpose(2, 0, 1) if rgb else value[None])[None].to(device)
            for value in values]


def phase_checks(model, device):
    # PAD-XY blocks must map to actual NESTED children at both GEV descents.
    hierarchy = {}
    for n in (model.geometry.nside, model.geometry.nside // 2, model.geometry.nside // 4):
        ids, sub = pad_xy_ids(n), pad_xy_ids(n // 2)
        np.testing.assert_array_equal(ids // 4, np.repeat(np.repeat(sub, 2, 1), 2, 2))
        hierarchy[str(n)] = int(ids.size)

    down = DownBlock3D(1, 1).to(device).eval()
    for wrapped in (down.down, down.conv1, down.conv2):
        wrapped.conv.weight.zero_()
        wrapped.conv.weight[0, 0, 1, 1, 1] = 1
        wrapped.bn.running_var.fill_(1 - wrapped.bn.eps)
    coords = torch.arange(8, device=device, dtype=torch.float32)
    field = (coords[:, None] + coords).expand(1, 12, 1, 4, 8, 8).contiguous()
    valid = torch.ones_like(field, dtype=torch.bool)
    aligned, _ = ParentAlignedGEV(model.gev_module, model.ops).down(field, valid, down)
    expected = F.avg_pool3d(packed(field), (2, 2, 2))
    stock = down(packed(field))
    torch.testing.assert_close(packed(aligned), expected)
    assert torch.equal(expected - stock, torch.ones_like(expected))
    up = nn.ConvTranspose3d(1, 1, 3, stride=2, padding=1, output_padding=1,
                            bias=False, device=device)
    up.weight.zero_()
    up.weight[0, 0, 1, 1, 1] = 1
    impulse = field.new_zeros((1, 12, 1, 2, 4, 4))
    impulse[0, 0, 0, 0, 2, 2] = 1
    unpooled = model.ops.transpose3d(impulse, torch.ones_like(impulse, dtype=torch.bool), up, valid)
    torch.testing.assert_close(unpooled[0, 0, 0, 0, 4:6, 4:6], torch.full((2, 2), .25, device=device))
    assert not unpooled[:, :, :, 1:].any()
    assert unpooled[0, 0, 0, 0].sum().item() == 1
    return {"checked_children": hierarchy, "stock_spatial_phase_offset_cells_per_axis": 0.5,
            "corrected_down_max_error": (packed(aligned) - expected).abs().max().item(),
            "corrected_up_impulse_2x2_weights": [0.25] * 4}


def component_checks(model, imgs, grids):
    device = imgs[0].device
    # Use the Stage 1 Nside 8 artifact for a fixed-camera exact match without
    # conflating weights learned by the refiner with raw sweep differences.
    small = grids / "grids_n8.npz"
    with np.load(small, allow_pickle=False) as a:
        lg, rg, joint = [torch.as_tensor(a[key], device=device)
                         for key in ("left_grid", "right_grid", "stereo_valid")]
    features = model.encoder(imgs)
    reference = sample_volumes(features, (lg, rg), joint, 8, torch.float32)
    mask = joint.permute(1, 0, 2, 3)[None, :, None]
    right = clear(sample_target(features[1], rg), mask)
    left = clear(sample_static(features[0], lg).unsqueeze(3).expand_as(right), mask)
    for candidate, expected in zip((left, right), reference):
        torch.testing.assert_close(candidate, expected)
    gwc, raw = raw_matching(left, right, mask)
    torch.testing.assert_close(gwc, correlate(*reference, 8))
    chunk_gwc, chunk_profile = model._chunk(sample_static(features[0], lg), features[1], rg, mask)
    torch.testing.assert_close(chunk_gwc, gwc, rtol=0, atol=0)
    torch.testing.assert_close(chunk_profile, raw, rtol=0, atol=0)
    old_volumes = tuple(packed(v).permute(0, 1, 3, 4, 2) for v in reference)
    grid = rg[:, 0].permute(1, 2, 0, 3)
    profile = model.similarity_context.encoder(None, [grid, grid], 96, spherical_volumes=old_volumes)
    torch.testing.assert_close(packed(raw), profile)

    # The first GWC group has four channels and scales dot products by sqrt(4).
    plen = 8
    all_valid = torch.ones((1, 12, 1, 4, plen, plen), dtype=torch.bool, device=device)
    tensor = torch.randn((1, 12, 32, 4, plen, plen), device=device)
    unit_cost, unit_profile = raw_matching(torch.ones_like(tensor), torch.ones_like(tensor), all_valid)
    torch.testing.assert_close(unit_cost, torch.full_like(unit_cost, 2))
    torch.testing.assert_close(unit_profile, torch.full_like(unit_profile, 32 ** .5))
    muted = all_valid.clone()
    muted[:, 1, :, :, :4] = False
    clean = raw_matching(tensor.masked_fill(~muted, 0), tensor.masked_fill(~muted, 0), muted)
    with torch.enable_grad():
        bad_left = tensor.masked_fill(~muted, float("nan")).requires_grad_()
        bad_right = tensor.masked_fill(~muted, float("inf")).requires_grad_()
        bad_cost, bad_profile = raw_matching(bad_left, bad_right, muted)
        torch.testing.assert_close((bad_cost, bad_profile), clean, rtol=0, atol=0)
        assert not bad_cost.masked_select(~muted).any()
        assert not bad_profile.masked_select(~muted.squeeze(2)).any()
        for grad in torch.autograd.grad(bad_cost.sum() + bad_profile.sum(), (bad_left, bad_right)):
            assert torch.isfinite(grad).all() and not grad.masked_select(~muted).any()
            assert grad.masked_select(muted).abs().max() > 0

    # Exercise a real FaceGEV layer with an isolated spatial tap.
    pixel_support = all_valid[:, :, :, 0]
    ids = torch.arange(plen*plen*12, device=device, dtype=torch.float32).reshape(1, 12, 1, plen, plen)
    halo, _ = model.ops.halo(ids, pixel_support, 1)
    n = int(halo[0,0,0,0,plen//2 + 1])
    f, p = divmod(n, plen*plen)
    y, x = divmod(p, plen)
    assert f != 0
    surface = torch.zeros_like(all_valid, dtype=torch.float32)
    surface[0, f, 0, 0, y, x] = 1
    layer = DownBlock3D(1, 1).to(device).eval().conv1
    layer.conv.weight.zero_()
    layer.conv.weight[0, 0, 1, 0, 1] = 1
    layer.bn.running_var.fill_(1 - layer.bn.eps)
    adapter = ParentAlignedGEV(model.gev_module, model.ops)
    observed = adapter.layer(surface, all_valid, layer, all_valid)
    index = (0, 0, 0, 0, 0, plen//2)
    torch.testing.assert_close(observed[index], observed.new_tensor(1.))
    assert not observed[:, :, :, 1:].any()
    planar = ParentAlignedGEV(model.gev_module, FaceOps(None, False)).layer(surface, all_valid, layer, all_valid)
    assert planar[index] == 0
    blocked = all_valid.clone()
    blocked[0, f, 0, 0, y, x] = False
    hidden = adapter.layer(surface.masked_fill(~blocked, float("nan")), blocked, layer, blocked)
    assert torch.isfinite(hidden).all() and not hidden.any()

    # Full face-local update equals stock update when given the same padding.
    state = torch.randn((1,12,64,plen,plen), device=device)
    context = tensor[:, :, :, 0]
    sim = torch.randn((1,12,model.similarity_context.output_dim,plen,plen), device=device)
    gev = torch.randn((1,12,model.gev_module.lookup_dim(model._gev_radius),plen,plen), device=device)
    depth = torch.full_like(pixel_support, 48., dtype=torch.float32)
    stock_state, stock_delta, _ = model.update_block._update(
        packed(state), packed(sim), packed(context), packed(gev), packed(depth), True)
    face_state, face_delta = update_state(model.update_block, FaceOps(None, False), state, sim, context,
                                          gev, depth, pixel_support)
    torch.testing.assert_close(packed(face_state), stock_state)
    torch.testing.assert_close(packed(face_delta), stock_delta, atol=5e-5, rtol=1e-5)

    # Direct recurrence test: a hidden-state impulse can cross a face edge
    # through the candidate gate with the same stock GRU equations.
    gate_fixture = type(model.update_block)(
        model.similarity_context.output_dim, 32, model.gev_module.lookup_dim(model._gev_radius), 64,
    ).to(device)
    for gate in (gate_fixture.gru.convz, gate_fixture.gru.convr, gate_fixture.gru.convq):
        gate.weight.zero_()
        gate.bias.zero_()
    gate_fixture.gru.convq.weight[0, 0, 0, 1] = 1
    pulse = torch.zeros_like(state)
    pulse[0, f, 0, y, x] = 1
    crossed, _ = update_state(gate_fixture, model.ops, pulse, sim, context, gev, depth, pixel_support)
    target = crossed[0, 0, 0, 0, plen//2]
    torch.testing.assert_close(target, .5 * crossed.new_tensor(.5).tanh())
    pixel_mask = pixel_support.clone()
    pixel_mask[0, f, 0, y, x] = False
    rejected, _ = update_state(gate_fixture, model.ops, pulse.masked_fill(~pixel_mask, float("nan")),
                               sim, context, gev, depth, pixel_mask)
    assert not rejected.any()

    logits = depth.new_zeros((1, 12, 36, plen, plen))
    children, child_valid = convex_children(model.ops, depth, logits, pixel_mask)
    torch.testing.assert_close(children[child_valid], depth.new_full((int(child_valid.sum()),), 48.))
    assert not child_valid[0, f, 0, 2*y:2*y+2, 2*x:2*x+2].any()
    return {"left_sweep_max_abs_error": (left-reference[0]).abs().max().item(),
            "right_sweep_max_abs_error": (right-reference[1]).abs().max().item(),
            "stage1_feature_volume_shape": list(left.shape),
            "gwc_max_abs_error": (gwc-correlate(*reference, 8)).abs().max().item(),
            "similarity_max_abs_error": (packed(raw)-profile).abs().max().item(),
            "gwc_all_ones_score": unit_cost.flatten()[0].item(),
            "masked_matching_nan_inf_forward_and_backward": True,
            "gev_cross_face_response": observed[index].item(), "gev_planar_control": planar[index].item(),
            "invalid_gev_impulse_leaked": False,
            "update_delta_max_abs_error": (packed(face_delta)-stock_delta).abs().max().item(),
            "gru_cross_face_impulse_response": target.item(), "invalid_hidden_input_leaked": False,
            "convex_children_preserve_constant_and_parent_mask": True}


def gradient_audit(model, output):
    terms = []
    for predicted, valid in zip(output["predictions"], output["output_masks"]):
        terms.append((predicted[valid] / 96).square().mean())
    torch.stack(terms).mean().backward()
    names = ["encoder.layers.0.conv.weight", "similarity_context.spatial_aggregator.conv.0.weight",
             "gev_module.regularization.stem.0.conv.weight", "gev_module.regularization.down1.down.conv.weight",
             "gev_module.regularization.up1.up.weight", "context_proj.conv.weight", "state_proj.conv.weight",
             "update_block.encoder.conv_sim2.weight", "update_block.encoder.conv_gev2.weight",
             "update_block.gru.convz.weight", "update_block.gru.convr.weight", "update_block.gru.convq.weight",
             "update_block.depth_head.conv2.weight", "update_block.mask.2.weight"]
    if model.opts.use_dav2_geometry:
        names.extend(["dav2_geometry.projections.0.weight", "dav2_geometry.refine.convs.0.weight",
                      "geometry_fusion.alpha"])
        assert not any(p.grad is not None for p in model.dav2_geometry.backbone.parameters())
    gradients = {}
    parameters = dict(model.named_parameters())
    for name in names:
        grad = parameters[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0, name
        gradients[name] = float(grad.norm())
    assert all(torch.isfinite(p.grad).all() for p in parameters.values() if p.grad is not None)
    return gradients


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--grids", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--frame", type=int, default=4887)
    p.add_argument("--iters", type=int)
    p.add_argument("--depth-chunk", type=int, default=8)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--skip-component-checks", action="store_true")
    p.add_argument("--dav2", action="store_true")
    p.add_argument("--dav2-source")
    p.add_argument("--dav2-checkpoint")
    args = p.parse_args()
    if args.dav2 and (not args.dav2_source or not args.dav2_checkpoint):
        p.error("--dav2 requires --dav2-source and --dav2-checkpoint")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    # Different halo sizes can select different cuDNN algorithms. Full FP32
    # accumulation makes stock-vs-face parity a comparison of the operators.
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    args.out.mkdir(parents=True, exist_ok=True)
    cameras, images, config, _ = load_pair(args.data, frame=args.frame)
    axis = geometry_probe.candidate_axis(args.selection, config)
    cache = args.grids / f"grids_n{args.nside}.npz"
    geometry = HealpixGeometry(cameras, axis, nside=args.nside, stage1_grid=cache if cache.exists() else None)
    saved = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    opts = transfer_options(saved.get("net_opts"), use_dav2=args.dav2, mixed_precision=args.amp,
                            depth_chunk=args.depth_chunk, gradient_checkpointing=args.backward)
    if args.dav2:
        opts.update(dav2_model_repo=args.dav2_source, dav2_checkpoint=args.dav2_checkpoint)
    with torch.random.fork_rng(devices=[]):
        legacy = build_stereo_model()
        assert type(legacy) is OmniDS
    del legacy
    try:
        build_stereo_model({"use_healpix": True, "use_yinyang": True}, healpix_geometry=geometry)
    except ValueError:
        exclusivity = True
    else:
        raise AssertionError("Conflicting topology flags accepted")
    model = build_stereo_model(opts, healpix_geometry=geometry).eval()
    loaded = model.load_pretrained(saved["net_state_dict"])
    assert not any(name.startswith("enhancer.") for name in model.state_dict())
    del saved
    model.to(device)
    imgs = paired_images(cameras, images, opts.get("use_rgb", False), device)
    dav2_rgb = None
    if args.dav2:
        from module_cnn_only.dav2_geometry import prepare_dav2_rgb

        dav2_rgb = prepare_dav2_rgb(images[0][None], args.dav2_source, opts.get("dav2_input_size", 518)).to(device)
    report = {"nside": args.nside, "frame": args.frame, "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": geometry_probe.hash_file(args.checkpoint), "device": str(device), "torch": torch.__version__,
              "input_shapes": [list(x.shape) for x in imgs],
              "iters": model.opts.gru_iters if args.iters is None else args.iters, "dav2": args.dav2,
              "backward": args.backward, "amp": args.amp, "legacy_factory": True, "mutual_exclusion": exclusivity,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "cudnn_tf32": torch.backends.cudnn.allow_tf32 if device.type == "cuda" else None,
              "stage1_exact_grid_parity": cache.exists(), "weight_transfer": loaded}
    with torch.no_grad():
        report["gev_phase"] = phase_checks(model, device)
        if not args.skip_component_checks:
            report["components"] = component_checks(model, imgs, args.grids)
    print("Component and geometry checks completed", flush=True)

    matching = model._matching

    def audit_matching(features, static, valid):
        cost, profile, guide = matching(features, static, valid)
        b, faces, _, depth, h, w = valid.shape
        assert cost.shape == (b, faces, 8, depth, h, w)
        assert profile.shape == (b, faces, depth, h, w)
        assert guide.shape == (b, faces, 32, h, w)
        pixel = valid.any(dim=3)
        with torch.no_grad():
            for tensor, mask in ((cost, valid), (profile, valid.squeeze(2)), (guide, pixel)):
                assert torch.isfinite(tensor).all() and not tensor.masked_select(~mask).any()
            expected_guide = clear(static, pixel) * valid.to(static.dtype).mean(dim=3)
            torch.testing.assert_close(guide, expected_guide)
        report["direct_matching"] = {"cnn_feature_shape": list(features[0].shape),
                                     "scan_chunk_shape": [b, faces, 32, min(depth, model.depth_chunk), h, w],
                                     "candidate_mask_shape": list(valid.shape), "guide_shape": list(guide.shape),
                                     "valid_candidates": int(valid.sum()), "total_candidates": valid.numel(),
                                     "nonfinite_or_nonzero_invalid": False}
        return cost, profile, guide

    model._matching = audit_matching
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = perf_counter()
    with torch.set_grad_enabled(args.backward):
        result = model(imgs, iters=args.iters, test_mode=not args.backward, dav2_rgb=dav2_rgb, return_details=True)
        assert result["index"].shape == (1, 1, 320, 480) == result["radial_depth"].shape
        assert result["hidden"].shape == (1, 12, 64, args.nside, args.nside)
        assert result["valid"].any() and all(v.any() for v in result["output_masks"])
        assert not result["hidden"].masked_select(~result["native_valid"]).any()
        lo = result["index"].floor().long()
        hi = result["index"].ceil().long()
        physical = model.geometry.output_candidates[None]
        assert not (result["valid"] & ~(physical.gather(1, lo) & physical.gather(1, hi))).any()
        for k in ("index", "inverse_depth", "radial_depth"):
            assert torch.isfinite(result[k]).all() and not result[k].masked_select(~result["valid"]).any()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report["forward_ms"] = (perf_counter() - t0) * 1000
        print("End-to-end forward complete", flush=True)
        if args.backward:
            t1 = perf_counter()
            report["gradient_norms"] = gradient_audit(model, result)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            report["backward_ms"] = (perf_counter() - t1) * 1000
        if device.type == "cuda":
            report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    report["output_valid_pixels"] = int(result["valid"].sum())
    report["native_valid_cells"] = int(result["native_valid"].sum())
    report["output_shape"] = list(result["index"].shape)
    for key in ("pyramid_audit", "profile_shape", "cost_shape"):
        report[key] = result[key]
    np.savez_compressed(args.out / "prediction.npz",
                        **{key: result[key].detach().cpu().numpy()
                           for key in ("index", "inverse_depth", "radial_depth", "valid")})
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Verified standalone HEALPix", args.nside, "backward", args.backward, "valid", report["output_valid_pixels"], flush=True)


if __name__ == "__main__":
    main()
