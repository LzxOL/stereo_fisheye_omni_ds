"""独立合成测试：默认尺寸Lookup/运动编码 + 小尺寸双网格完整前后向。

python tools/smoke_dual_grid_refinement.py --device cpu
合成投影仅为测试夹具，不是OmniThings标定，不报告真实深度精度。
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from module_cnn_only.depth_refinement import DepthPyramid, DepthWiseLookup, MotionEncoder, StereoDepthRefinement
from module_cnn_only.update import ConvGRU
from module_cnn_only.network import OmniDS
from module_cnn_only.dual_grid_stereo import DualFisheyeStereoNetwork
from module_cnn_only.loss_functions import sequence_loss
from experiments.local_spherical_grid.adaptive_local_grid_v1 import AdaptiveLocalSphericalGrid
from tools.smoke_cross_grid_warp import synthetic_patch


def make_network(package):
    depth = len(package.metadata['inverse_depth_candidates'])
    return OmniDS(dict(num_cams=2, use_yinyang=True, num_invdepth=depth,
                       num_depth_samples=depth), grid_package=package)


def run_network(model, imgs, qualities, **kwargs):
    return model(imgs, yang_to_yin_quality=qualities[0], yin_to_yang_quality=qualities[1], **kwargs)


def assert_finite_gradients(module):
    gradients = [p.grad for p in module.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)


def check_lookup(device):
    volume = torch.randn(1, 8, 96, 64, 128, device=device)
    valid = torch.ones(1, 1, 96, 64, 128, device=device, dtype=torch.bool)
    pyramid = DepthPyramid(2).to(device)
    volumes, masks = pyramid(volume, valid)
    assert [tuple(v.shape) for v in volumes] == [(1, 8, 96, 64, 128), (1, 8, 48, 64, 128)]
    torch.testing.assert_close(volumes[1], F.avg_pool3d(volume, (2, 1, 1)))
    idx = torch.rand(1, 1, 64, 128, device=device) * 95
    queried = DepthWiseLookup(4)(volumes, masks, idx)
    assert queried.shape == (1, 144, 64, 128) and torch.isfinite(queried).all()

    # 每通道每像素附加不同常数，防止把C、D、H、W错排而蒙混过关。
    d = torch.arange(8, device=device)[None, None, :, None, None].float()
    spatial = torch.tensor([[10., 20., 30.], [40., 50., 60.]], device=device)[None, None, None]
    channel = torch.tensor([0., 100.], device=device)[None, :, None, None, None]
    known = (d + spatial + channel).requires_grad_()
    support = torch.ones(1, 1, 8, 2, 3, device=device, dtype=torch.bool)
    levels, supported = pyramid(known, support)
    query = torch.tensor([[[[-2., 0., 2.25], [3., 6.5, 10.]]]], device=device, requires_grad=True)
    sampled = DepthWiseLookup(1)(levels, supported, query)
    expected = []
    for level in (0, 1):
        for c in (0, 1):
            for offset in (-1, 0, 1):
                q = (query[:, 0] / 2 ** level + offset).clamp(0, 8 // 2 ** level - 1)
                expected.append(q * 2 ** level + (.5 if level else 0.) + spatial[0, 0, 0] + c * 100)
    torch.testing.assert_close(sampled, torch.stack(expected, dim=1))
    sampled.sum().backward()
    assert torch.isfinite(known.grad).all() and known.grad.abs().sum() > 0
    assert torch.isfinite(query.grad).all() and query.grad.abs().sum() > 0

    # 第3候选无效：整数2不依赖ceil，2.5需要2/3两个邻点，必须拒绝。
    support[:, :, 3] = False
    levels, supported = pyramid(known.detach().masked_fill(~support, float('nan')), support)
    integer = DepthWiseLookup(0)(levels[:1], supported[:1], torch.full_like(query, 2.))
    fractional = DepthWiseLookup(0)(levels[:1], supported[:1], torch.full_like(query, 2.5))
    assert integer.abs().sum() > 0 and not fractional.any()
    # Pool(2,3)只保留有效候选2，不被无效NaN污染。
    torch.testing.assert_close(levels[1][:, :, 1], known.detach()[:, :, 2])
    empty_levels, empty_masks = pyramid(torch.full_like(known, float('nan')), torch.zeros_like(support))
    empty = DepthWiseLookup(4)(empty_levels, empty_masks, query.detach())
    assert torch.isfinite(empty).all() and not empty.any()
    return dict(pyramid_shapes=[list(v.shape) for v in volumes], lookup_shape=list(queried.shape),
                integer_fractional_clamp_and_per_pixel_queries='PASS', strict_invalid_neighbors='PASS')


def check_motion_and_gru(device):
    encoder = MotionEncoder().to(device)
    gru = ConvGRU(64, 160).to(device)
    lookup = torch.randn(1, 144, 64, 128, device=device)
    context = torch.randn(1, 32, 64, 128, device=device)
    indices = torch.rand(1, 1, 64, 128, device=device) * 95
    with torch.no_grad():
        motion = encoder(lookup, context, indices, 96)
        inp = torch.cat([context, motion], dim=1)
        updated = gru(torch.zeros(1, 64, 64, 128, device=device), inp)
    assert motion.shape == (1, 128, 64, 128) and inp.shape == (1, 160, 64, 128)
    assert updated.shape == (1, 64, 64, 128) and torch.isfinite(updated).all()
    torch.testing.assert_close(motion[:, -1:], indices / 95)
    return dict(motion_shape=list(motion.shape), gru_input_shape=list(inp.shape), hidden_shape=list(updated.shape))


def make_package():
    """两套真实球面参数化，小尺寸合成等距鱼眼投影，非OmniThings标定。

    参考相机原点不变，目标沿X平移；逐D投影同一r/rho候选点。
    这里的简化投影仅生成测试输入；生产网络仍只读取已有几何包。
    """
    depth = 16
    axis = np.linspace(.01, .6, depth)
    patches = [synthetic_patch('yin', (7, 13), 120, 35, 0),
               synthetic_patch('yang', (13, 9), 100, 40, 25)]
    for p in patches:
        def grid(points):
            xy = np.linalg.norm(points[..., :2], axis=-1)
            theta = np.arctan2(xy, points[..., 2])
            coords = points[..., :2] / np.maximum(xy[..., None], 1e-10) * theta[..., None] / 1.8
            good = np.isfinite(coords).all(-1) & (np.abs(coords) <= 1).all(-1) & p.padding_valid
            return np.where(good[..., None], coords, 2.).astype(np.float32), good
        p.left_grid, p.left_feature_valid = grid(p.rays_ref)
        right, masks = [], []
        for rho in axis:
            target_points = p.rays_ref / rho + np.array([.2, 0., 0.])
            g, v = grid(target_points)
            right.append(g)
            masks.append(v & p.left_feature_valid)
        p.right_grid = np.stack(right)
        p.stereo_valid = np.stack(masks)
        # 部分候选不可见，并保留一个左可见但所有匹配候选均无效的位置。
        p.stereo_valid[3:5, 2, 3] = False
        p.stereo_valid[:, 3, 4] = False
        assert not np.array_equal(p.right_grid[0], p.right_grid[-1])
    package = AdaptiveLocalSphericalGrid(*patches, {}, dict(inverse_depth_candidates=axis.tolist()))
    for target, source in [('yin', 'yang'), ('yang', 'yin')]:
        coords, inside = package.correspondence(target, source)
        package.reference[target + '_to_' + source] = coords.astype(np.float32)
        package.reference[target + '_in_' + source] = inside
    return package


def check_pipeline(device, amp=False):
    package = make_package()
    model = make_network(package).to(device).train()
    imgs = [torch.randn(1, 3, 32, 40, device=device, requires_grad=True) for _ in range(2)]
    qualities = [torch.linspace(.15, .95, 16 * np.prod(p.shape), device=device).reshape(1, 1, 16, *p.shape)
                 for p in (package.yin, package.yang)]
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    counts, shapes, state_inputs, state_outputs = {}, {}, [], []
    scan_shapes = []
    dtypes = {}
    context_inputs = []

    def count(name):
        def hook(module, args, output):
            counts[name] = counts.get(name, 0) + 1
            if torch.is_tensor(output):
                shapes[name] = list(output.shape)
                dtypes[name] = str(output.dtype)
            elif name == 'cnn':
                shapes[name] = [list(f.shape) for f in output]
        return hook

    hooks = [model.encoder.register_forward_hook(count('cnn')),
             model.enhancer.geometry_encoder.register_forward_hook(count('geometry')),
             model.fusion.network.register_forward_hook(count('fusion')),
             model.refinement.aggregation.register_forward_hook(count('aggregation')),
             model.refinement.pyramid.register_forward_hook(count('pyramid')),
             model.refinement.context_proj.register_forward_hook(count('context_init')),
             model.refinement.state_proj.register_forward_hook(count('hidden_init')),
             model.refinement.lookup.register_forward_hook(count('lookup')),
             model.refinement.update_block.encoder.register_forward_hook(count('motion')),
             model.refinement.update_block.gru.register_forward_hook(count('gru'))]

    def record_state(module, args, output):
        state_inputs.append(args[0].detach().clone())
        # UpdateBlock返回的是已经应用像素有效性Mask的实际递归Hidden。
        state_outputs.append(output[0].detach().clone())

    def record_context(module, args):
        context_inputs.append(args[2].detach().clone())
        assert not args[3].requires_grad  # 每轮idx已detach，Hidden不detach。

    def record_scan(module, args):
        scan_shapes.append(dict(left=list(args[0].shape), right=list(args[1].shape), mask=list(args[3].shape)))

    hooks.append(model.refinement.update_block.register_forward_hook(record_state))
    hooks.append(model.refinement.update_block.register_forward_pre_hook(record_context))
    hooks.append(model.enhancer.register_forward_pre_hook(record_scan))
    dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
    scaler = torch.amp.GradScaler('cuda', enabled=amp and device.type == 'cuda')
    losses = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            output = run_network(model, imgs, qualities)
            loss = 0.
            for name in ('yin', 'yang'):
                row = output[name]
                assert row['aggregated_volume'].shape == (1, 8, 16, *getattr(package, name).shape)
                assert row['depth_logits'].shape == row['depth_prob'].shape == (1, 1, 16, *getattr(package, name).shape)
                assert len(row['predictions']) == 3
                assert_finite = [row['init_depth_idx'], *row['predictions'], row['inverse_depth'], row['radial_depth']]
                for value in assert_finite:
                    assert torch.isfinite(value).all() and not value.masked_select(~row['valid']).any()
                assert all(p.min() >= 0 and p.max() <= 15 for p in row['predictions'])
                torch.testing.assert_close(row['depth_prob'].sum(2), row['valid'].float(), rtol=1e-5, atol=1e-6)
                assert not row['depth_prob'].masked_select(~row['candidate_valid']).any()
                gt = torch.full_like(row['init_depth_idx'], 3.75)  # 合成索引监督，不是实际数据GT。
                loss = loss + sequence_loss([row['init_depth_idx'], *row['predictions']], gt, row['valid'])
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for module in (model.encoder, model.fusion, model.refinement.aggregation, model.refinement.classifier,
                       model.refinement.context_proj, model.refinement.state_proj,
                       model.refinement.update_block.encoder, model.refinement.update_block.gru,
                       model.refinement.update_block.depth_head, model.enhancer.local_refiner):
            assert_finite_gradients(module)
        if step == 1:
            assert_finite_gradients(model.enhancer.geometry_encoder)
        for image in imgs:
            assert image.grad is not None and torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        before = model.refinement.classifier.weight.detach().clone()
        scaler.step(optimizer)
        scaler.update()
        assert not torch.equal(before, model.refinement.classifier.weight)
        losses.append(float(loss.detach()))
        if step == 0:
            assert counts == dict(cnn=1, geometry=2, fusion=2, aggregation=2, pyramid=2,
                                  context_init=2, hidden_init=2, lookup=6, motion=6, gru=6)
            for base in (0, 3):
                for i in (base + 1, base + 2):
                    torch.testing.assert_close(state_inputs[i], state_outputs[i - 1])
                    torch.testing.assert_close(context_inputs[i], context_inputs[base])
            assert shapes['lookup'][1] == 144 and shapes['motion'][1] == 128
    for hook in hooks:
        hook.remove()

    # 同一配置、严格checkpoint重载；不静默忽略任何键或尺寸。
    model.eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = make_network(package).to(device).eval()
    restored.load_state_dict(torch.load(buffer, map_location=device, weights_only=True), strict=True)
    with torch.no_grad():
        a, b = run_network(model, imgs, qualities), run_network(restored, imgs, qualities)
    for name in ('yin', 'yang'):
        torch.testing.assert_close(a[name]['final_depth_idx'], b[name]['final_depth_idx'])

    # 旧独立入口不再维护网络实现；它与正式入口可以严格互载，输出相同。
    compatible = DualFisheyeStereoNetwork(package).to(device).eval()
    compatible.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        old_api = compatible(imgs, *qualities)
        inference = run_network(model, imgs, qualities, test_mode=True, iters=3)
        one_iteration = run_network(model, imgs, qualities, iters=1)
        model.opts.mixed_precision = True
        internal_amp = run_network(model, imgs, qualities, iters=1)
        model.opts.mixed_precision = False
    for name in ('yin', 'yang'):
        torch.testing.assert_close(old_api[name]['final_depth_idx'], a[name]['final_depth_idx'])
        torch.testing.assert_close(inference[name], a[name]['final_depth_idx'])
        assert len(one_iteration[name]['predictions']) == 1
        assert torch.isfinite(internal_amp[name]['final_depth_idx']).all()
    assert not hasattr(model, 'gev_module') and not hasattr(model, 'similarity_context')

    # 候选轴端点和小数索引，米制转换不得乘2或误用192轴。
    index = torch.tensor([[[[0., 7.5, 15.]]]], device=device)
    rho, radial = model.index_to_depth(index, torch.ones_like(index, dtype=torch.bool))
    torch.testing.assert_close(rho, torch.tensor([[[[.01, .305, .6]]]], device=device))
    torch.testing.assert_close(radial, 1 / rho)

    # 全无效双目候选：完整网络仍可反传，prob/idx/metric均为0，不伪装有效第0层。
    for patch in (package.yin, package.yang):
        patch.stereo_valid[:] = False
    model.train().zero_grad(set_to_none=True)
    empty = run_network(model, imgs, qualities)
    sum(row['final_depth_idx'].sum() + row['init_depth_idx'].sum() for row in empty.values()).backward()
    for row in empty.values():
        assert not row['valid'].any() and not row['depth_prob'].any() and not row['radial_depth'].any()
        assert not row['final_depth_idx'].any()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

    return dict(precision='AMP ' + str(dtype) if amp else 'FP32', synthetic_losses=losses,
                output_shapes={name: list(a[name]['final_depth_idx'].shape) for name in ('yin', 'yang')},
                recorded_shapes=shapes, scan_shapes=scan_shapes[:2], intermediate_dtypes=dtypes,
                calls_per_forward={k: v // 2 for k, v in counts.items()},
                cnn_both_inputs_and_all_branches_backward='PASS', hidden_recurrence_and_static_context='PASS',
                strict_checkpoint_roundtrip='PASS', canonical_omnids_and_compatibility_api='PASS',
                internal_amp_and_inference_api='PASS', all_invalid='PASS',
                parameters=sum(p.numel() for p in model.parameters()))


def check_disabled(device):
    model = StereoDepthRefinement(num_depth=16, use_refinement=False).to(device)
    cost = torch.randn(1, 8, 16, 8, 16, device=device, requires_grad=True)
    ctx = torch.randn(1, 32, 8, 16, device=device)
    valid = torch.ones(1, 1, 16, 8, 16, device=device, dtype=torch.bool)
    result = model(cost, ctx, valid, valid.any(2))
    assert len(result['predictions']) == 1
    torch.testing.assert_close(result['final_depth_idx'], result['init_depth_idx'])
    result['final_depth_idx'].square().mean().backward()
    assert_finite_gradients(model.classifier)
    assert cost.grad is not None and torch.isfinite(cost.grad).all() and cost.grad.abs().sum() > 0
    return 'PASS'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.manual_seed(20261116)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    start = time.perf_counter()
    result = dict(torch=torch.__version__, device=str(device), lookup=check_lookup(device),
                  motion_and_gru=check_motion_and_gru(device), refinement_disabled=check_disabled(device))
    result['pipeline_fp32'] = check_pipeline(device)
    result['pipeline_amp'] = check_pipeline(device, amp=True)
    result.update(status='PASS', elapsed_seconds=time.perf_counter() - start,
                  cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
                  real_omnithings_gt_test='NOT RUN: synthetic tests only')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
