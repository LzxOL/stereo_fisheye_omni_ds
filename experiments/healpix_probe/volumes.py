"""Frozen fish-eye encoder, HEALPix sampling, and stock groupwise correlation."""
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from geometry_probe import hash_file
from module_cnn_only.featurelayer import FeatureLayers
from module_cnn_only.similarity_context import groupwise_correlation_volume
from module_cnn_only.sweep import spherical_sweep
from utils.image import normalizeImage, rgb2gray


def timed(operation, device, repeats=5):
    # Measure steady-state wall time, including launch overhead, without any
    # host-to-device transfers or one-off CUDA initialization in the samples.
    sync = torch.cuda.synchronize if device.type == "cuda" else lambda _: None
    output = operation()
    sync(device)
    del output
    times = []
    baseline_mib = peak_delta_mib = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_mib = torch.cuda.memory_allocated(device) / 2**20
    for trial in range(repeats):
        sync(device)
        start = perf_counter()
        output = operation()
        sync(device)
        times.append((perf_counter() - start) * 1000)
        if trial < repeats - 1:
            del output
    if baseline_mib is not None:
        peak_delta_mib = torch.cuda.max_memory_allocated(device) / 2**20 - baseline_mib
    return output, dict(ms_median=float(np.median(times)), ms_samples=times,
                        gpu_baseline_allocated_mib=baseline_mib, gpu_peak_delta_mib=peak_delta_mib)


def frozen_features(checkpoint, cameras, images, device):
    # Only load the trusted project checkpoint and instantiate its encoder.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    opts = saved["net_opts"]
    assert opts.get("base_channel", 32) == 32
    assert not opts.get("encoder_downsample_twice", False)
    use_rgb = opts.get("use_rgb", False)
    encoder = FeatureLayers(32, use_rgb, downsample_twice=False)
    state = {}
    for key, value in saved["net_state_dict"].items():
        key = key.removeprefix("module.")
        if key.startswith("encoder."):
            state[key.removeprefix("encoder.")] = value
    encoder.load_state_dict(state, strict=True)
    del state, saved
    encoder = encoder.requires_grad_(False).eval().to(device)
    inputs = []
    for camera, image in zip(cameras, images):
        source = image if use_rgb else rgb2gray(image, channel_wise_mean=True)
        normalized = normalizeImage(source, camera.invalid_mask)
        chw = normalized.transpose(2, 0, 1) if use_rgb else normalized[None]
        inputs.append(torch.from_numpy(chw.astype(np.float32))[None].to(device))
    features, timing = timed(lambda: encoder(inputs), device)
    assert all(tuple(features[i].shape[1:]) == (32, (camera.height + 1) // 2,
                                                (camera.width + 1) // 2)
               for i, camera in enumerate(cameras))
    assert all(torch.isfinite(f).all() for f in features)
    return features, {"checkpoint": str(checkpoint), "sha256": hash_file(checkpoint),
                      "parameter_tensors": len(encoder.state_dict()), "use_rgb": bool(use_rgb),
                      "input_shapes": [list(image.shape) for image in inputs],
                      "output_shapes": [list(feature.shape) for feature in features], **timing}


def sample_volumes(features, grids, joint, chunk, dtype):
    b, c = features[0].shape[:2]
    dcount, faces, nside, _ = joint.shape
    lg, rg = grids
    shape = (b, faces, c, dcount, nside, nside)
    left, right = [torch.empty(shape, device=features[0].device, dtype=dtype) for _ in range(2)]
    left_slice = F.grid_sample(features[0], lg.reshape(1, faces * nside, nside, 2).expand(b,-1,-1,-1),
                               align_corners=True).reshape(b, c, faces, nside, nside)
    left_slice = left_slice.transpose(1, 2).unsqueeze(3).to(dtype)
    for start in range(0, dcount, chunk):
        stop = min(start + chunk, dcount)
        grid = rg[start:stop].permute(1, 2, 3, 0, 4).reshape(1, faces * nside * nside, stop - start, 2)
        samples = F.grid_sample(features[1], grid.expand(b, -1, -1, -1), align_corners=True)
        samples = samples.reshape(b, c, faces, nside, nside, stop - start).permute(0,2,1,5,3,4)
        mask = joint[start:stop].permute(1, 0, 2, 3)[None, :, None]
        left[:, :, :, start:stop] = left_slice.masked_fill(~mask, 0)
        right[:, :, :, start:stop] = samples.to(dtype).masked_fill(~mask, 0)
    return left, right


def correlate(left, right, chunk):
    b, faces, c, dcount, h, w = left.shape
    volume = left.new_empty((b, faces, 8, dcount, h, w))
    for start in range(0, dcount, chunk):
        stop = min(start + chunk, dcount)
        a, z = [v[:, :, :, start:stop].reshape(b * faces, c, stop - start, h, w)
                .permute(0, 1, 3, 4, 2) for v in (left, right)]
        gwc = groupwise_correlation_volume(a, z, 8)
        volume[:, :, :, start:stop] = gwc.reshape(b, faces, 8, h, w, stop - start).permute(0,1,2,5,3,4)
    return volume


def sampling_coordinates_check(sweep, rg, joint, features):
    _, faces, nside, _ = joint.shape
    # Independent original 5-D sampler over three real right grids. Use a
    # small decoded subset of feature channels to avoid irrelevant storage.
    picks = sweep.selected
    shaped = rg[picks].permute(1, 2, 3, 0, 4)
    shaped = shaped.reshape(faces * nside, nside, len(picks), 2)
    feature = features[1][:, :2]
    old = spherical_sweep([feature, feature], [shaped, shaped])[0]
    flat = shaped.reshape(1, faces * nside * nside, len(picks), 2)
    new = F.grid_sample(feature, flat, align_corners=True).reshape_as(old)
    error = (new - old).abs().max().item()
    torch.testing.assert_close(new, old, atol=1e-6, rtol=1e-5)

    # Linear pixel ramp recovers pixel centers exactly (up to float32 roundoff).
    torch_device = feature.device
    h, w = feature.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=torch_device),
                            torch.arange(w, device=torch_device), indexing="ij")
    ramp = torch.stack((2 * xx, 2 * yy)).float()[None]
    projected = F.grid_sample(ramp, flat, align_corners=True)[0]
    projected = projected.permute(2, 1, 0).reshape(len(picks), faces, nside, nside, 2)
    valid = sweep.joint[picks]
    uv = sweep.right_uv_selected
    residual = np.linalg.norm(projected.cpu().numpy()[valid] - uv[valid], axis=-1)
    np.testing.assert_array_less(residual, 2e-4)

    lg = torch.as_tensor(sweep.left_grid, device=torch_device)
    sampled_left = F.grid_sample(ramp, lg.reshape(1, faces * nside, nside, 2), align_corners=True)
    sampled_left = sampled_left[0].permute(1, 2, 0).reshape(faces, nside, nside, 2).cpu().numpy()
    left_error = np.linalg.norm(sampled_left[sweep.left_valid] - sweep.left_uv[sweep.left_valid], axis=-1)
    np.testing.assert_array_less(left_error, 2e-4)
    return dict(original_sampler_max_abs_difference=error, right_pixel_ramp_max_px=float(residual.max()),
                left_pixel_ramp_max_px=float(left_error.max()))


def run_volumes(features, sweep, device, chunk, dtype, repeats):
    lg = torch.as_tensor(sweep.left_grid, device=device)
    rg = torch.as_tensor(sweep.right_grid, device=device)
    joint = torch.as_tensor(sweep.joint, device=device)
    sampler = sampling_coordinates_check(sweep, rg, joint, features)
    (left, right), sweep_timing = timed(lambda: sample_volumes(features, (lg, rg), joint, chunk, dtype),
                                      device, repeats)
    gwc, gwc_timing = timed(lambda: correlate(left, right, chunk), device, repeats)
    nside = sweep.nside
    assert left.shape == right.shape == (1, 12, 32, 96, nside, nside)
    assert gwc.shape == (1, 12, 8, 96, nside, nside)
    invalid_counts = []
    for volume in (left, right, gwc):
        assert torch.isfinite(volume).all()
        invalid = ~joint.permute(1, 0, 2, 3)[None, :, None]
        invalid_counts.append(torch.count_nonzero(volume.masked_fill(~invalid, 0)).item())
    assert invalid_counts == [0, 0, 0]

    # Compare stock GWC with a scalar definition at real valid sites and
    # on face boundaries. Retain the half-precision error rather than hiding it.
    valid_ids = np.argwhere(sweep.joint)
    selected = valid_ids[np.linspace(0, len(valid_ids) - 1, 512).astype(int)]
    edge = valid_ids[np.isin(valid_ids[:, 2:], [0, nside - 1]).any(axis=1)]
    assert len(edge) > 0
    selected = np.concatenate((selected, edge[np.linspace(0, len(edge) - 1, 256).astype(int)]))
    d, f, y, x = torch.as_tensor(selected, device=device).T
    lhs = left[0, f, :, d, y, x].double().reshape(-1,8,4)
    rhs = right[0, f, :, d, y, x].double().reshape(-1,8,4)
    ref = (lhs * rhs).sum(-1) / 2
    observed = gwc[0, f, :, d, y, x].double()
    torch.testing.assert_close(observed, ref, atol=0.02 if dtype == torch.float16 else 5e-5,
                               rtol=0.003 if dtype == torch.float16 else 2e-6)
    valid_cost = gwc[0, f, :, d, y, x]
    return {
        "dtype": str(dtype), "left_shape": list(left.shape), "right_shape": list(right.shape),
        "gwc_shape": list(gwc.shape), "valid_candidates": int(joint.sum().item()),
        "sampled_valid_gwc_std": float(valid_cost.float().std().item()),
        "resident_volumes_mib": sum(volume.nbytes for volume in (left, right, gwc)) / 2**20,
        "grids_and_mask_mib": sum(item.nbytes for item in (lg, rg, joint)) / 2**20,
        "invalid_nonzero_counts_left_right_gwc": invalid_counts,
        "fp64_gwc_reference_sites": len(selected),
        "fp64_gwc_reference_edge_sites": 256,
        "fp64_gwc_max_abs_error": (observed - ref).abs().max().item(),
        "sampling_checks": sampler,
        "sweep": sweep_timing,
        "gwc": gwc_timing,
    }
