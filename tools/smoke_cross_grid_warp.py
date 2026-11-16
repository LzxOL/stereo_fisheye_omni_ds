"""小尺寸合成球面验证，不读取OmniThings或训练网络。

python tools/smoke_cross_grid_warp.py --device cpu
两套不同旋转/范围/尺寸的SphericalPatch复用现有射线变换和sample_patch。
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.local_spherical_grid.adaptive_local_grid_v1 import (
    SphericalPatch, AdaptiveLocalSphericalGrid, pad_shape, sample_patch,
)
from module_cnn_only.cross_grid_warp import BidirectionalCrossGridWarp


def synthetic_patch(name, native_shape, span, elevation, yaw):
    size = pad_shape(native_shape)
    padding = np.zeros(size, dtype=bool)
    padding[:native_shape[0], :native_shape[1]] = True
    angle = np.deg2rad(yaw)
    basis = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    bounds = dict(longitude_center_deg=0., longitude_half_span_deg=span/2,
                  latitude_min_deg=-elevation, latitude_max_deg=elevation)
    # 只使用几何查询需要的字段，不构造假相机或运行OCam标定。
    p = SphericalPatch(name=name, basis=basis, bounds=bounds, original_shape=native_shape,
        rays_ref=None, left_uv=None, right_uv=None, left_valid=padding,
        target_valid=None, stereo_valid=None, left_grid=None, right_grid=None,
        left_feature_valid=padding, padding_valid=padding)
    x,y = np.meshgrid(np.linspace(-1, 1, size[1]), np.linspace(-1, 1, size[0]))
    p.rays_ref, actual = p.grid_to_direction(np.stack([x,y], axis=-1))
    np.testing.assert_array_equal(actual, padding)
    return p


def known_cost(p, batch, depth, device):
    # G=8：XYZ、可精确检查的D/Batch编号，以及XYZ的线性组合。
    rays = torch.as_tensor(p.rays_ref, dtype=torch.float32, device=device).permute(2,0,1)
    spatial = torch.cat([rays, torch.zeros_like(rays[:1]), rays, rays[:1]*2], dim=0)
    volume = spatial[None,:,None].expand(batch,-1,depth,-1,-1).clone()
    code = 100*torch.arange(batch, device=device)[:,None]+torch.arange(depth, device=device)[None]
    volume[:,3] = code[:,:,None,None]
    padding = torch.as_tensor(p.padding_valid, device=device)[None,None,None]
    volume.masked_fill_(~padding, float('nan'))
    return volume.requires_grad_(), padding.expand(batch,1,depth,-1,-1).clone()


def direct_reference(volume, mask, source, destination):
    b,g,d,h,w = volume.shape
    values = volume.permute(0,2,1,3,4).reshape(b*d,g,h,w)
    support = mask.permute(0,2,1,3,4).reshape(b*d,1,h,w)
    sampled, valid = sample_patch(values, source, destination.rays_ref, support)
    ht,wt = destination.shape
    return sampled.reshape(b,d,g,ht,wt).permute(0,2,1,3,4), valid.reshape(b,d,1,ht,wt).permute(0,2,1,3,4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    yin = synthetic_patch('yin', (9,17), 140, 40, 0)
    yang = synthetic_patch('yang', (17,11), 120, 45, 30)
    batch,depth = 2,10
    package = AdaptiveLocalSphericalGrid(yin,yang,{},dict(inverse_depth_candidates=np.linspace(.01,.6,depth).tolist()))
    for target,source in [('yin','yang'),('yang','yin')]:
        coords,inside = package.correspondence(target,source)
        package.reference[target+'_to_'+source] = coords.astype(np.float32)
        package.reference[target+'_in_'+source] = inside
    # 缺缓存旧包的回退也只能发生在初始化；与预计算路径必须相同。
    uncached = AdaptiveLocalSphericalGrid(yin,yang,{},package.metadata)
    fallback = BidirectionalCrossGridWarp(uncached).to(device)
    with patch.object(package,'correspondence',side_effect=AssertionError('cached geometry recomputed')),\
         patch.object(yin,'direction_to_grid',side_effect=AssertionError('Yin geometry recomputed')),\
         patch.object(yang,'direction_to_grid',side_effect=AssertionError('Yang geometry recomputed')):
        model = BidirectionalCrossGridWarp(package).to(device)
        yl,ym = known_cost(yin,batch,depth,device)
        al,am = known_cost(yang,batch,depth,device)
        result = model(yl,al,ym,am)
    assert sum(p.numel() for p in model.parameters()) == 0
    fallback_result = fallback(yl,al,ym,am)
    for key,value in result.items():
        torch.testing.assert_close(value,fallback_result[key])
    errors,angles = {},{}
    for direction,source,destination,volume,mask in [
        ('yang_to_yin',yang,yin,al,am),('yin_to_yang',yin,yang,yl,ym)]:
        value,valid = result[direction], result[direction+'_valid']
        assert value.shape == (batch,8,depth,*destination.shape)
        assert valid.shape == (batch,1,depth,*destination.shape)
        assert valid.any() and not valid.all()
        assert torch.isfinite(value).all() and not value.masked_select(~valid.expand_as(value)).any()
        assert not valid[...,~torch.as_tensor(destination.padding_valid,device=device)].any()
        expected,expected_valid = direct_reference(volume,mask,source,destination)
        torch.testing.assert_close(value,expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(valid,expected_valid)
        truth = torch.as_tensor(destination.rays_ref,device=device,dtype=torch.float32).permute(2,0,1)
        difference = (value[:,:3]-truth[None,:,None]).abs().masked_select(valid.expand(-1,3,-1,-1,-1))
        error = float(difference.detach().max())
        assert error < .015, (direction,error)  # 球面XYZ非线性，双线性重采样有离散近似误差。
        errors[direction] = error
        observed = value[:,:3].permute(0,2,3,4,1)[valid[:,0]]
        expected_rays = truth.permute(1,2,0)[None,None].expand(batch,depth,-1,-1,-1)[valid[:,0]]
        cross = torch.linalg.cross(observed,expected_rays).norm(dim=-1)
        dot = (observed*expected_rays).sum(-1)
        angle = float(torch.rad2deg(torch.atan2(cross,dot)).detach().max())
        assert angle < .5, (direction,angle)
        angles[direction] = angle
        code = 100*torch.arange(batch,device=device)[:,None]+torch.arange(depth,device=device)[None]
        expected_code = code[:,:,None,None].expand(batch,depth,*destination.shape).float()
        torch.testing.assert_close(value[:,3][valid[:,0]],expected_code[valid[:,0]],rtol=0,atol=3e-5)
    sum(result[k].square().mean() for k in ['yang_to_yin','yin_to_yang']).backward()
    for volume,valid in [(yl,ym),(al,am)]:
        assert torch.isfinite(volume.grad).all() and volume.grad.abs().sum()>0
        assert not volume.grad.masked_select(~valid.expand_as(volume)).any()

    # 目标自身没有匹配支持时，另一张图仍可能提供有效跨网格信息；不能强行合并。
    independent = model(yl,al,torch.zeros_like(ym),am)
    assert independent['yang_to_yin_valid'].any()
    assert not independent['yin_to_yang_valid'].any()
    # 一个正权重源邻点无效，必须剔除对应候选，而不是按剩余邻点重新归一化。
    query = model.yin_to_yang
    h,w = yang.shape
    xy = (query+1)*torch.tensor([w-1,h-1],device=device)/2
    fraction = xy-xy.floor()
    candidates = result['yang_to_yin_valid'][0,0,3] & (fraction[...,0]>.1) & (fraction[...,0]<.9)
    candidates &= (fraction[...,1]>.1) & (fraction[...,1]<.9)
    ty,tx = torch.nonzero(candidates)[0].tolist()
    sx,sy = xy[ty,tx].floor().long().tolist()
    broken_mask = am.clone(); broken_mask[0,0,3,sy,sx] = False
    corrupted = al.detach().clone(); corrupted[0,:,3,sy,sx] = float('nan')
    broken = model(yl,corrupted,ym,broken_mask)
    assert not broken['yang_to_yin_valid'][0,0,3,ty,tx]
    assert not broken['yang_to_yin'][0,:,3,ty,tx].any()
    assert broken['yang_to_yin_valid'][1,0,3,ty,tx] and broken['yang_to_yin_valid'][0,0,4,ty,tx]
    assert torch.isfinite(broken['yang_to_yin']).all()
    # 分块改变不应影响采样；无效源全NaN时输出及反向仍是有限零。
    model.depth_chunk = 3
    smaller_chunks = model(yl,al,ym,am)
    for key,value in result.items():
        torch.testing.assert_close(smaller_chunks[key],value,rtol=1e-5,atol=1e-6)
    empty_y = torch.full_like(yl,float('nan'),requires_grad=True)
    empty_a = torch.full_like(al,float('nan'),requires_grad=True)
    empty = model(empty_y,empty_a,torch.zeros_like(ym),torch.zeros_like(am))
    assert all(torch.isfinite(v).all() and not v.any() for v in empty.values())
    (empty['yang_to_yin'].sum()+empty['yin_to_yang'].sum()).backward()
    assert torch.isfinite(empty_y.grad).all() and not empty_y.grad.any()
    assert torch.isfinite(empty_a.grad).all() and not empty_a.grad.any()
    print(json.dumps(dict(status='PASS',device=str(device),input_shapes=[list(yl.shape),list(al.shape)],
        output_shapes={k:list(v.shape) for k,v in result.items()},max_xyz_interpolation_error=errors,
        max_direction_error_deg=angles,
        original_sample_patch_equivalent=True,no_forward_geometry_recalculation=True,
        depth_and_batch_index_preserved=True,target_stereo_mask_kept_independent=True,
        invalid_neighbor_rejected=True,source_gradients_finite=True,all_invalid_safe=True,
        chunk_equivalence=True,trainable_parameters=0),indent=2))


if __name__ == '__main__':main()
