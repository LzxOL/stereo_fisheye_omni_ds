"""Original shared CNN + native Yin/Yang sweeping only; NOT a Stereo training run."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from adaptive_local_grid_v1 import (AdaptiveLocalSphericalGrid,HERE,sweep_patch,sample_patch,sample_reference)
from camera_geometry import load_pair
from module_cnn_only.featurelayer import FeatureLayers
from module_cnn_only.sweep import spherical_sweep
from utils.image import rgb2gray,normalizeImage


def check_small_interfaces(package,device):
    """Independent feature-grid equivalence, masking and interpolation checks."""
    patch=package.yang
    torch.manual_seed(20261114)
    features=[torch.randn(1,2,384,400,device=device,requires_grad=True) for _ in range(2)]
    left,right,mask=sweep_patch(features,patch,depth_chunk=12)
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    # Compare three depths with the unchanged original sampler, with original masks.
    ids=[0,48,95]
    grids=[torch.as_tensor(patch.left_grid,device=device)[:,:,None].expand(-1,-1,3,-1),
           torch.as_tensor(patch.right_grid[ids],device=device).permute(1,2,0,3)]
    original=spherical_sweep(features,grids)
    for actual,expected in zip([left[:,:,ids],right[:,:,ids]],original):
        expected=expected.permute(0,1,4,2,3).masked_fill(~mask[:,:,ids],0)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
    (left.square().mean()+right.square().mean()).backward()
    assert all(f.grad is not None and torch.isfinite(f.grad).all() and f.grad.abs().sum()>0 for f in features)
    empty=replace(patch,stereo_valid=np.zeros_like(patch.stereo_valid))
    static,static_valid=sample_reference(features[0].detach(),empty)
    assert static_valid.any() and static.abs().sum()>0
    np.testing.assert_array_equal(static_valid[0,0].cpu().numpy(),patch.left_feature_valid)
    f=[x.detach().requires_grad_() for x in features]
    l,r,m=sweep_patch(f,empty,depth_chunk=12)
    assert not m.any() and not l.any() and not r.any()
    (l.sum()+r.sum()).backward()
    assert all(torch.isfinite(x.grad).all() and not x.grad.any() for x in f)
    # Artificial partial candidate invalidity must remove both matching features.
    part=patch.stereo_valid.copy();part[10:20]=False
    l,r,m=sweep_patch([x.detach() for x in features],replace(patch,stereo_valid=part),12)
    assert not m[:,:,10:20].any() and not l[:,:,10:20].any() and not r[:,:,10:20].any()
    # Flatten D as batch for cost queries: only complete active-corner support passes.
    v=torch.ones(96,1,*patch.shape,device=device,requires_grad=True)
    support=torch.as_tensor(patch.stereo_valid,device=device)[:,None]
    rays=package.reference['rays_ref'][::4,::4]
    queried,valid=sample_patch(v,patch,rays,support)
    assert torch.isfinite(queried).all() and not queried[~valid].any()
    torch.testing.assert_close(queried[valid],torch.ones_like(queried[valid]))
    queried.sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    # One invalid nonzero-weight corner cannot leak cost evidence.
    h,w=patch.original_shape
    cx,cy=w//2,h//2
    coordinates=np.array([[[2*(cx+.5)/(patch.shape[1]-1)-1,2*(cy+.5)/(patch.shape[0]-1)-1]]])
    ray,_=patch.grid_to_direction(coordinates)
    bad=torch.ones(1,1,*patch.shape,device=device,dtype=torch.bool);bad[0,0,cy,cx]=False
    q,ok=sample_patch(torch.ones(1,1,*patch.shape,device=device),patch,ray,bad)
    assert not ok.any() and not q.any()
    return dict(original_sampler_equivalent=True,feature_gradients=True,all_invalid_safe=True,
        partial_invalid_safe=True,static_left_context_independent_of_stereo_mask=True,
        strict_query_rejects_invalid_corner=True,output_query_shape=list(queried.shape),
        query_valid_fraction=float(valid.float().mean()))


def run_cnn(package,cameras,images,device,amp,depth_chunk):
    torch.manual_seed(20261114)
    model=FeatureLayers(CH=32,use_rgb=False).to(device).train()
    inputs=[]
    for image,camera in zip(images,cameras):
        normalized=normalizeImage(rgb2gray(image,channel_wise_mean=True),camera.invalid_mask)
        inputs.append(torch.tensor(normalized,dtype=torch.float32,device=device)[None,None])
    optimizer=torch.optim.SGD(model.parameters(),lr=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    gradient_scale=scaler.get_scale()
    if device.type=='cuda':
        torch.cuda.synchronize(device);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
        features=model(inputs)
    for f in features:f.retain_grad()
    rows=[]
    # Backward each patch separately, retaining the shared CNN graph only once.
    # This avoids storing both huge cost volumes at once; no optimizer update between patches.
    for i,patch in enumerate([package.yin,package.yang]):
        left,right,mask=sweep_patch(features,patch,depth_chunk)
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        assert not left.masked_select(~mask.expand_as(left)).any()
        loss=(left.float()-right.float()).square().mean()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward(retain_graph=i==0)
        rows.append(dict(name=patch.name,volume_left=list(left.shape),volume_right=list(right.shape),
            mask=list(mask.shape),original_shape=list(patch.original_shape),padded_shape=list(patch.shape),
            dtype=str(left.dtype),loss=float(loss.detach()),valid_fraction=float(mask.float().mean())))
        del left,right,mask,loss
    grads=[p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(f.grad is not None and torch.isfinite(f.grad).all() and f.grad.abs().sum()>0 for f in features)
    before=next(model.parameters()).detach().clone()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    scaler.step(optimizer)
    scaler.update()
    changed=not torch.equal(before,next(model.parameters()))
    assert changed
    if device.type=='cuda':torch.cuda.synchronize(device)
    return dict(precision='AMP fp16 CNN / fp32 geometric sampling' if amp else 'float32',
        input_shapes=[list(x.shape) for x in inputs],input_feature_shapes=[list(x.shape) for x in features],
        grids=rows,forward='PASS',backward='PASS',cnn_parameter_updated=changed,
        feature_gradients=[float(x.grad.float().abs().sum())/gradient_scale for x in features],
        gradient_scaler_enabled=amp,
        elapsed_seconds=time.perf_counter()-start,
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=HERE/'outputs/input_snapshot')
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,default=HERE/'outputs_v1_yin50/grid_v1_smoke.json')
    p.add_argument('--device',default='cuda:1' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--depth-chunk',type=int,default=8)
    p.add_argument('--frame',type=int,default=4887)
    args=p.parse_args()
    torch.set_num_threads(2)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    package=AdaptiveLocalSphericalGrid.load(args.geometry)
    cameras,images,_,_=load_pair(args.data_root,frame=args.frame)
    result=dict(status='RUNNING',scope='CNN + sweeping only; no GEV/GRU/DA-V2 or Stereo training',
        profile=package.metadata.get('profile','yin60_legacy'),config_hash=package.metadata.get('config_hash'),
        torch=torch.__version__,device=str(device),interfaces=check_small_interfaces(package,device),runs=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for amp in ([False,True] if device.type=='cuda' else [False]):
        row=run_cnn(package,cameras,images,device,amp,args.depth_chunk)
        result['runs'].append(row)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(row,indent=2),flush=True)
    result['status']='PASS'
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
