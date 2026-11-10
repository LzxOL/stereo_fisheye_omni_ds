"""CNN-only bounded smoke test using the original Dataset's calibrated LUT builder.

Small test: python tools/smoke_cnn_only.py --db_root /data
Four-camera real batch: add --real_sample --device cuda --mixed_precision.
Stereo: add --num_cams 2; --full_size uses D/H/W=96/160/240 and 768x800 images.
Stereo --real_sample reads real images only; its loss is synthetic, never Rig GT.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import Dataset
from module_cnn_only.network import OmniDS
from module_cnn_only.loss_functions import sequence_loss
from module_cnn_only.sweep import spherical_sweep
from utils.common import Edict
from utils.geometry import applyTransform


def stereo_grids(data, height, width, depth):
    """Test fixture only: use real OCam calibration, left cam4 and right cam3.

    The network itself never knows these camera IDs. No Dataset/GT adaptation is
    implemented here. Normalised grids address the original stride-2 CNN centres.
    """
    reference, target = data.ocams[3], data.ocams[2]
    axis = data.invdepths[::2][:depth]

    def project(camera, points):
        pixels = camera.rayToPixel(points)
        u, v = pixels
        valid = np.isfinite(pixels).all(0) & (u >= 0) & (u <= camera.width - 1)
        valid &= (v >= 0) & (v <= camera.height - 1)
        x = np.nan_to_num(u).clip(0, camera.width - 1)
        y = np.nan_to_num(v).clip(0, camera.height - 1)
        x0, x1, y0, y1 = np.floor(x).astype(int), np.ceil(x).astype(int), np.floor(y).astype(int), np.ceil(y).astype(int)
        invalid = camera.invalid_mask
        valid &= ~(invalid[y0, x0] | invalid[y0, x1] | invalid[y1, x0] | invalid[y1, x1])
        feature_size = np.array([(camera.width + 1) // 2, (camera.height + 1) // 2])
        grid = pixels.T / (feature_size - 1) - 1.0  # pixels/2 then 2*x/(Wf-1)-1
        valid &= np.isfinite(grid).all(1) & (np.abs(grid) <= 1).all(1)
        return np.where(valid[:, None], grid, 2.).astype(np.float32), valid

    def rays(azimuth, elevation):
        az, el = np.meshgrid(azimuth, elevation)
        return np.stack([np.cos(el) * np.sin(az), -np.sin(el), np.cos(el) * np.cos(az)]).reshape(3, -1)

    # Infer an overlap-centred window from this calibration; no invented camera poses.
    az = np.deg2rad(np.linspace(-180, 180, 181, endpoint=False))
    probe = rays(az, np.deg2rad(np.linspace(45, -45, 31)))
    _, left_valid = project(reference, probe)
    points = applyTransform(target.rig2cam, applyTransform(reference.cam2rig, probe / axis[depth // 2]))
    _, right_valid = project(target, points)
    common = left_valid & right_valid
    assert common.any(), 'No calibrated overlap for the smoke-test camera pair'
    azimuth = np.arctan2(probe[0, common], probe[2, common])
    center = np.angle(np.exp(1j * azimuth).mean())
    directions = rays(np.linspace(center - np.deg2rad(65), center + np.deg2rad(65), width),
                      np.deg2rad(np.linspace(45, -45, height)))
    left, left_valid = project(reference, directions)
    right = np.empty((height, width, depth, 2), dtype=np.float32)
    valid = np.zeros((depth, height, width), dtype=bool)
    for d, rho in enumerate(axis):
        points = directions / rho
        rig = applyTransform(reference.cam2rig, points)
        if d == depth // 2:
            np.testing.assert_allclose(applyTransform(reference.rig2cam, rig), points, atol=1e-6)
        grid, visible = project(target, applyTransform(target.rig2cam, rig))
        right[:, :, d] = grid.reshape(height, width, 2)
        valid[d] = (left_valid & visible).reshape(height, width)
    left = torch.from_numpy(left.reshape(height, width, 1, 2)).expand(-1, -1, depth, -1)
    right = torch.from_numpy(right)
    assert valid.any()
    assert not torch.equal(right[:, :, 0], right[:, :, -1]), 'Right projection must depend on depth'
    return [left, right], torch.from_numpy(valid)[None, None]


def stereo_mask_checks(model, imgs, grids, candidate, iterations):
    """Check visibility fallbacks, unsupported lookup bins and all-invalid backward."""
    b, _, d, h, w = candidate.shape
    index = torch.full((b, 1, h, w), d / 2, device=imgs[0].device)
    features = torch.stack([torch.ones((b, 32, *imgs[0].shape[-2:]), device=index.device),
                            torch.full((b, 32, *imgs[0].shape[-2:]), 2., device=index.device)], 1)
    # Keep some stereo candidates but remove the current right projection.
    partial = candidate.clone()
    partial[:, :, d // 2:] = False
    sampled = model._sample_cnn_erp(features, grids, index, partial)
    supported_left = partial.any(2) & (grids[0][:, :, 0].abs() <= 1).all(-1)[None, None]
    assert supported_left.any()
    torch.testing.assert_close(sampled[supported_left.expand_as(sampled)], torch.ones_like(sampled[supported_left.expand_as(sampled)]))
    empty = torch.zeros_like(candidate)
    assert not model._sample_cnn_erp(features, grids, index, empty).any()
    constant = torch.full_like(index, 3.25)
    mask = torch.zeros(b, 36, h, w, device=index.device)
    torch.testing.assert_close(model.upsample_depth(constant, mask),
                               torch.full((b, 1, 2 * h, 2 * w), 3.25, device=index.device))
    isolated = torch.zeros_like(index, dtype=torch.bool)
    isolated[:, :, h // 2, w // 2] = True
    extreme_mask = mask.clone().view(b, 1, 9, 2, 2, h, w)
    extreme_mask[:, :, 4] = -1000
    isolated_up = model.upsample_depth(constant, extreme_mask.reshape_as(mask), isolated)
    torch.testing.assert_close(isolated_up[..., h:h+2, w:w+2],
                               torch.full((b, 1, 2, 2), 3.25, device=index.device))
    # Poison unsupported bins to ensure lookup masks are applied, not just pooled.
    profile = torch.full((b, d, h, w), 100., device=index.device)
    assert not model.similarity_context.lookup(profile, index, empty).any()
    volumes = spherical_sweep([features[:, 0], features[:, 1]], grids)
    model.gev_module.build(*volumes, volumes, valid_mask=empty)
    assert not model.gev_module.lookup(index, radius=model._gev_radius).any()
    assert all(not level.any() for level in model.gev_module._valid_pyramid)
    model.gev_module.clear_cache()
    for label, valid in [('partial', partial), ('all_invalid', empty)]:
        model.zero_grad(set_to_none=True)
        predictions = model(imgs, grids, iters=iterations, valid_mask=valid)
        assert all(torch.isfinite(p).all() for p in predictions)
        output_valid = F.interpolate(valid.any(2).float(), scale_factor=2, mode='nearest').bool()
        assert all(not p[~output_valid].any() for p in predictions)
        sum(p.mean() for p in predictions).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if label == 'all_invalid':
            assert all(not p.any() for p in predictions)
    # A caller may leave NaN placeholders in invisible target grids: sanitise before sampling.
    y, x = torch.nonzero(candidate.any(2)[0, 0], as_tuple=True)
    bad_grids = [grids[0], grids[1].clone()]
    bad_grids[1][y[0], x[0]] = torch.nan
    model.zero_grad(set_to_none=True)
    bad_output = model(imgs, bad_grids, iters=iterations, valid_mask=candidate)
    assert all(torch.isfinite(p).all() and not p[..., 2*y[0]:2*y[0]+2, 2*x[0]:2*x[0]+2].any() for p in bad_output)
    sum(p.mean() for p in bad_output).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if not model.opts.use_dynamic_context:
        # Static context has its own left visibility, even where no stereo match exists.
        was_training = model.training
        model.eval()
        captured = []
        hook = model.update_block.register_forward_pre_hook(lambda m, i: captured.append(i[2].detach().clone()))
        with torch.no_grad():
            left = model.encoder(imgs)[0]
            grid = grids[0][:, :, 0][None].expand(b, -1, -1, -1)
            visible = (torch.isfinite(grid).all(-1) & (grid.abs() <= 1).all(-1))[:, None]
            sampled_left = F.grid_sample(left, grid, align_corners=True)
            expected = torch.relu(model.context_proj(torch.where(visible, sampled_left, 0.)))
            expected = torch.where(visible, expected, 0.)
            model(imgs, grids, iters=2, valid_mask=empty)
            torch.testing.assert_close(captured[0], expected)
            assert expected.abs().sum() > 0
            captured.clear()
            model(imgs, grids, iters=2, valid_mask=candidate,
                  left_valid_mask=torch.zeros_like(index, dtype=torch.bool))
            assert all(not context.any() for context in captured)
        hook.remove()
        model.train(was_training)
    return dict(left_fallback=True, zero_context=True, constant_index_upsampling=True,
                invalid_lookup=True, partial_forward_backward=True, all_invalid_forward_backward=True,
                nonfinite_grid_forward_backward=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db_root', required=True, help='parent of omnithings/')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--real_sample', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--num_cams', type=int, choices=[2, 4], default=4)
    parser.add_argument('--use_group_variance', action='store_true')
    parser.add_argument('--use_dynamic_context', action='store_true')
    parser.add_argument('--full_size', action='store_true', help='stereo D/H/W=96/160/240, images 768x800')
    parser.add_argument('--num_invdepth', type=int)
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--output', default='/tmp/omnids_cnn_only_smoke.json')
    args = parser.parse_args()
    if args.num_invdepth is None:
        args.num_invdepth = 96 if args.num_cams == 2 and (args.full_size or args.real_sample) else (16 if args.num_cams == 2 else 32)
    if args.iters < 2 or args.num_invdepth < 16 or args.num_invdepth % 8:
        parser.error('Use >=2 iterations and num_invdepth divisible by 8 and >=16')
    if args.mixed_precision and not args.device.startswith('cuda'):
        parser.error('The original CUDA autocast requires CUDA for --mixed_precision')
    torch.manual_seed(0)
    torch.set_num_threads(4)
    stereo = args.num_cams == 2
    full = args.real_sample or args.full_size
    size = ([320, 480] if full else [32, 48]) if stereo else ([160, 640] if args.real_sample else [32, 64])
    options = Edict(equirect_size=size, num_invdepth=args.num_invdepth, num_downsample=1,
                   equi_h=size[0], equi_w=size[1],
                   use_rgb=False, phi_deg=45., num_cams=args.num_cams,
                   num_depth_samples=args.num_invdepth if stereo else args.num_invdepth // 2,
                   sim_levels=4, gev_num_pyramid_levels=2, gev_radius=4,
                   mixed_precision=args.mixed_precision, cnn_only=True,
                   use_group_variance=(not stereo or args.use_group_variance),
                   use_dynamic_context=(not stereo or args.use_dynamic_context))
    # No fabricated grids and no new LUT cache in the dataset: use the exact author geometry.
    data_options = Edict(options)
    if stereo:
        data_options.num_invdepth = 2 * args.num_invdepth
    data = Dataset('omnithings', data_options, load_lut=False, db_root=args.db_root)
    candidate = None
    if stereo:
        grids, candidate = stereo_grids(data, size[0] // 2, size[1] // 2, args.num_invdepth)
        grids = [g.to(args.device) for g in grids]
        candidate = candidate.to(args.device)
        if args.real_sample:
            images, _ = data.loadImages(data.train_idx[0], True, use_rgb=False)
            imgs = [torch.as_tensor(images[i], device=args.device)[None] for i in (3, 2)]
        else:
            fish_size = (768, 800) if full else (64, 80)
            imgs = [torch.randn(1, 1, *fish_size, device=args.device) for _ in range(2)]
        # Network-only: no Rig GT is used as a left-centred label.
        gt = torch.full((1, 1, *size), args.num_invdepth * .3, device=args.device)
        valid = F.interpolate(candidate.any(2).float(), scale_factor=2, mode='nearest').bool()
        imgs = [img.requires_grad_() for img in imgs]
    elif args.real_sample:
        grids = [torch.as_tensor(g, device=args.device) for g in data.buildLookupTable()]
        imgs, gt, valid, _ = data.loadSample(data.train_idx[0])
        imgs = [torch.as_tensor(img, device=args.device)[None] for img in imgs]
        gt = torch.as_tensor(gt, device=args.device)[None, None].float()
        valid = torch.as_tensor(valid, device=args.device)[None, None].bool()
    else:
        grids = [torch.as_tensor(g, device=args.device) for g in data.buildLookupTable()]
        # Random images are only for tensor/gradient checks; grids still use actual calibration.
        imgs = [torch.randn(1, 1, 64, 80, device=args.device) for _ in range(4)]
        gt = torch.full((1, 1, *size), args.num_invdepth * .3, device=args.device)
        valid = torch.ones_like(gt, dtype=torch.bool)

    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    model = OmniDS(options).to(args.device).train()
    assert not hasattr(model, 'dino_extractor') and not hasattr(model, 'context_fuser')
    shapes, counts, hooks = {}, {}, []
    index_ranges = []
    contexts, hidden_states = [], []

    def record(name, value):
        counts[name] = counts.get(name, 0) + 1
        shapes[name] = [list(v.shape) for v in value] if isinstance(value, (list, tuple)) else list(value.shape)

    for name, layer in [('cnn', model.encoder), ('profile', model.similarity_context.spatial_aggregator),
                        ('gev_before_spatial_upsample', model.gev_module.regularization),
                        ('context', model.context_proj), ('hidden_init', model.state_proj),
                        ('delta_index', model.update_block.depth_head), ('up_mask', model.update_block.mask)]:
        hooks.append(layer.register_forward_hook(lambda m, i, o, name=name: record(name, o)))
    for name, layer in [('similarity_lookup', model.update_block.encoder.conv_sim1),
                        ('gev_lookup', model.update_block.encoder.conv_gev1)]:
        hooks.append(layer.register_forward_pre_hook(lambda m, i, name=name: record(name, i[0])))
    def record_gru(module, inputs):
        record('recurrent_hidden', inputs[0])
        record('gru_input', inputs[1])
        hidden_states.append(inputs[0].detach().clone())

    hooks.append(model.update_block.gru.register_forward_pre_hook(record_gru))
    hooks.append(model.update_block.encoder.conv_depth1.register_forward_pre_hook(
        lambda m, i: index_ranges.append([float(i[0].detach().min()), float(i[0].detach().max())])))

    def record_context(module, inputs):
        contexts.append(inputs[2].detach().clone())

    hooks.append(model.update_block.register_forward_pre_hook(record_context))
    hooks.append(model.gev_module.regularization.register_forward_pre_hook(
        lambda m, i: record('gev_input', i[0])))

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5, eps=1e-8)
    # Prove GWC-only does not execute Python's original Tensor.var branch.
    guard = patch.object(torch.Tensor, 'var', side_effect=AssertionError('Variance executed')) if not model.opts.use_group_variance else nullcontext()
    context_guard = patch.object(model, '_sample_cnn_erp', side_effect=AssertionError('Dynamic context executed')) if not model.opts.use_dynamic_context else nullcontext()
    with guard, context_guard:
        predictions = model(imgs, grids, iters=args.iters, valid_mask=candidate)
    assert len(predictions) == args.iters
    assert all(list(p.shape) == [1, 1, *size] and torch.isfinite(p).all() for p in predictions)
    loss = sequence_loss(predictions, gt, valid)
    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision, init_scale=128.)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = {}
    for name, layer in [('cnn', model.encoder), ('context', model.context_proj), ('state', model.state_proj),
                        ('view_weights', model.view_weight_head), ('similarity', model.similarity_context),
                        ('gev', model.gev_module), ('gru', model.update_block.gru),
                        ('depth_head', model.update_block.depth_head), ('upsample_mask', model.update_block.mask)]:
        grads = [p.grad for p in layer.parameters() if p.grad is not None]
        if name == 'view_weights' and not model.opts.use_dynamic_context:
            assert not grads
            continue
        assert grads and all(torch.isfinite(g).all() for g in grads), name
        gradients[name] = sum(float(g.abs().sum()) for g in grads)
        assert gradients[name] > 0, name
    assert counts['cnn'] == counts['hidden_init'] == counts['profile'] == counts['gev_before_spatial_upsample'] == 1
    assert counts['context'] == (args.iters if model.opts.use_dynamic_context else 1)
    assert shapes['gev_input'][1] == (16 if model.opts.use_group_variance else 8)
    if not model.opts.use_dynamic_context:
        assert all(torch.equal(contexts[0], c) for c in contexts)
    assert not torch.equal(hidden_states[0], hidden_states[-1])
    assert index_ranges[0] != index_ranges[-1]
    image_gradients = [float(img.grad.abs().sum()) for img in imgs] if stereo else []
    if stereo:
        assert all(torch.isfinite(img.grad).all() and img.grad.abs().sum() > 0 for img in imgs)
        assert all((p >= 0).all() and (p <= args.num_invdepth - 1).all() for p in predictions)
        assert index_ranges[0] == [args.num_invdepth / 2, args.num_invdepth / 2]
        assert all(0 <= lo <= hi <= args.num_invdepth - 1 for lo, hi in index_ranges)
    for hook in hooks:
        hook.remove()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    before = model.encoder.layers[0].conv.weight.detach().clone()
    scaler.step(optimizer)
    scaler.update()
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith('cuda') else None
    assert not torch.equal(before, model.encoder.layers[0].conv.weight)
    checkpoint = Path(args.output).with_suffix('.pth')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'net_state_dict': model.state_dict(), 'net_opts': model.opts}, checkpoint)
    saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
    restored = OmniDS(saved["net_opts"]).to(args.device).eval()
    restored.load_state_dict(saved['net_state_dict'])
    with torch.no_grad():
        result = restored(imgs, grids, iters=args.iters, test_mode=True, valid_mask=candidate)
    assert list(result.shape) == [1, 1, *size] and torch.isfinite(result).all()
    forbidden = [name for name in sys.modules if name.startswith(('timm', 'module.'))
                 or 'erp_cross_attention' in name or 'MultiScaleDeformableAttention' in name]
    assert not forbidden, forbidden
    mask_checks = stereo_mask_checks(model, imgs, grids, candidate, args.iters) if stereo and not full else {}
    report = dict(device=args.device, real_sample=args.real_sample, input=[list(i.shape) for i in imgs],
                  grids=[list(g.shape) for g in grids], shapes=shapes, calls=counts,
                  predictions=[list(p.shape) for p in predictions], loss=float(loss.detach()),
                  gradient_l1=gradients, optimizer_update=True, checkpoint_reload=True,
                  forbidden_imports=forbidden, num_cams=args.num_cams, iterations=args.iters,
                  input_gradient_l1=image_gradients, mask_checks=mask_checks,
                  index_ranges=index_ranges,
                  use_group_variance=model.opts.use_group_variance,
                  use_dynamic_context=model.opts.use_dynamic_context,
                  context_constant=all(torch.equal(contexts[0], c) for c in contexts),
                  hidden_updated=True, lookup_indices_updated=True,
                  training_seconds=training_seconds, peak_allocated_mib=peak_memory)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
