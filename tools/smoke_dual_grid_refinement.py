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
    # 当前正式双网格路径复用DA Baseline，而非旧的144通道独立实验网络。
    from tools.smoke_dav2_yinyang import synthetic_run
    return synthetic_run(device, amp=amp, use_da=False)


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


def check_static_single_grid(device):
    # 单网格分支也只初始化一次Context，保留每轮Similarity/GEV Lookup和Hidden递归。
    options = dict(num_cams=2, num_invdepth=16, num_depth_samples=16,
                   equi_h=16, equi_w=32, sim_levels=2)
    model = OmniDS(options).to(device).train()
    assert not hasattr(model, '_sample_cnn_erp') and not hasattr(model, 'view_weight_head')
    assert 'use_dynamic_context' not in model.opts
    imgs = [torch.randn(1, 3, 24, 32, device=device, requires_grad=True) for _ in range(2)]
    left_grid = torch.rand(8, 16, 1, 2, device=device) * 1.6 - .8
    grids = [left_grid.expand(-1, -1, 16, -1), torch.rand(8, 16, 16, 2, device=device) * 1.6 - .8]
    valid = torch.ones(1, 1, 16, 8, 16, device=device, dtype=torch.bool)
    contexts, states, counts = [], [], dict(context=0, hidden=0, similarity_lookup=0, gev_lookup=0)
    def count(name):
        def hook(module, args, output):
            counts[name] += 1
        return hook
    hooks = [layer.register_forward_hook(count(name)) for name, layer in (
        ('context', model.context_proj), ('hidden', model.state_proj),
        ('similarity_lookup', model.update_block.encoder.conv_sim1),
        ('gev_lookup', model.update_block.encoder.conv_gev1))]
    def record_inputs(module, args):
        contexts.append(args[2].detach().clone())
        states.append(args[0].detach().clone())
    hooks.append(model.update_block.register_forward_pre_hook(record_inputs))
    predictions = model(imgs, grids, iters=3, valid_mask=valid)
    assert counts == dict(context=1, hidden=1, similarity_lookup=3, gev_lookup=3)
    assert all(torch.equal(contexts[0], item) for item in contexts)
    assert not torch.equal(states[0], states[-1])
    sum(p.square().mean() for p in predictions).backward()
    for image in imgs:
        assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
    for h in hooks:
        h.remove()
    # 复用原有Mask/Lookup/上采样测试，包括无匹配时左Context仍有效。
    from tools.smoke_cnn_only import stereo_mask_checks
    checks = stereo_mask_checks(model, imgs, grids, valid, 3)
    restored = OmniDS(options).to(device).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    return dict(status='PASS', context_fixed=True, hidden_updated=True, calls=counts, mask_checks=checks)


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
    result['static_single_grid'] = check_static_single_grid(device)
    result['pipeline_fp32'] = check_pipeline(device)
    result['pipeline_amp'] = check_pipeline(device, amp=True)
    result.update(status='PASS', elapsed_seconds=time.perf_counter() - start,
                  cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
                  real_omnithings_gt_test='NOT RUN: synthetic tests only')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
