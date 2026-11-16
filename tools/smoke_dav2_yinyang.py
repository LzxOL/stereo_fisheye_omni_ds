"""DA-V2 Baseline + Yin/Yang联通测试，不启动正式训练。

默认CPU使用明确标记的合成冻结ViT，仅验证真实Adapter/GRU代码。
传入--model_repo/--checkpoint使用官方ViT；--data_root使用真实OCam/RGB。
Loss目标是合成索引，不能作为真实GT误差或精度提升证据。
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from module_cnn_only.network import OmniDS
from module_cnn_only.loss_functions import sequence_loss
from tools.smoke_dual_grid_refinement import make_package


class SyntheticViT(nn.Module):
    """仅测试夹具，不冒充官方模型或预训练特征。"""
    def __init__(self, model_name='vitb'):
        super().__init__()
        self.projection = nn.Linear(3, 768, bias=False)
        self.calls = 0

    def get_intermediate_layers(self, x, layers):
        self.calls += 1
        feature = F.avg_pool2d(x, 14, 14).flatten(2).transpose(1, 2)
        return [self.projection(feature) * (i+1) / 4 for i in range(len(layers))]


def real_geometry(data_root, frame=4887):
    # 只降低测试角采样密度；使用已选窗口/旋转、真实OCam和完整96候选，不改生产缓存。
    import copy
    from experiments.local_spherical_grid.adaptive_local_grid_v1 import AdaptiveLocalSphericalGrid, _make_patch, load_config
    from experiments.local_spherical_grid.camera_geometry import load_pair, pair_geometry
    from experiments.local_spherical_grid.search_yang_grid import depth_axis
    cameras, images, config, _ = load_pair(data_root, frame=frame)
    axis = depth_axis(config)
    cfg = copy.deepcopy(load_config())
    cfg['generation']['native_angular_step_deg'] = 15.
    selection = json.loads((ROOT/'experiments/local_spherical_grid/selected_grid_yin50.json').read_text())
    r, t, _ = pair_geometry(*cameras)
    patches = [_make_patch(name, selection[name+'_bounds'], basis, cameras, r, t, axis, cfg)
               for name, basis in [('yin', np.eye(3)), ('yang', np.asarray(selection['yang_basis_visual']))]]
    package = AdaptiveLocalSphericalGrid(*patches, {}, dict(inverse_depth_candidates=axis.tolist()))
    for a, b in [('yin', 'yang'), ('yang', 'yin')]:
        g, valid = package.correspondence(a, b)
        package.reference[a+'_to_'+b] = g.astype(np.float32)
        package.reference[a+'_in_'+b] = valid
    from utils.image import normalizeImage, rgb2gray
    inputs = [torch.tensor(normalizeImage(rgb2gray(image, channel_wise_mean=True), cam.invalid_mask))[None,None].float()
              for image, cam in zip(images, cameras)]
    return package, inputs, images[0]


def frozen_hash(backbone):
    h = hashlib.sha256()
    for value in backbone.state_dict().values():
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_network(device, amp, use_da, repo='', checkpoint='', data_root=None, baseline_checkpoint=None):
    torch.manual_seed(20261116)
    if data_root:
        package, images, raw_rgb = real_geometry(data_root)
        from module_cnn_only.dav2_geometry import prepare_dav2_rgb
        rgb = prepare_dav2_rgb(raw_rgb[None], repo).to(device)
    else:
        package = make_package()
        images = [torch.randn(1,3,32,40) for _ in range(2)]
        rgb = torch.randn(1,3,28,42,device=device)
    images = [x.to(device).requires_grad_() for x in images]
    depth = len(package.metadata['inverse_depth_candidates'])
    options = dict(num_cams=2, num_invdepth=depth, num_depth_samples=depth,
                   equi_h=16, equi_w=32, use_dav2_geometry=use_da,
                   reference_grid='left_fisheye', dav2_model_repo=str(repo), dav2_checkpoint=str(checkpoint),
                   mixed_precision=amp)
    # 必须继承整个原主干，不能只迁移CNN或另建MotionEncoder。
    baseline = OmniDS(options).to(device)
    loaded_checkpoint = None
    if baseline_checkpoint:
        saved = torch.load(baseline_checkpoint, map_location='cpu', weights_only=False)
        loaded_checkpoint = baseline.load_baseline_weights(saved['net_state_dict'])
        del saved
    model = OmniDS(dict(options, use_yinyang=True), grid_package=package).to(device)
    transfer = model.load_baseline_weights(baseline.state_dict())
    shared_keys = list(baseline.state_dict())
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    assert not hasattr(model, 'refinement')
    assert model.update_block.encoder.conv_sim1.in_channels == 36
    assert model.update_block.encoder.conv_gev1.in_channels == 18
    del baseline
    qualities = [torch.full((1,1,depth,*p.shape), .8, device=device) for p in (package.yin,package.yang)]
    counts, shapes, contexts, states, head_inputs, fusion_outputs, gevs = {}, {}, [], [], [], [], []
    def hook(name):
        def record(module, args, output):
            counts[name] = counts.get(name,0)+1
            if torch.is_tensor(output): shapes[name] = list(output.shape)
            if name == 'fusion': fusion_outputs.append(output.detach().clone())
            if name == 'gev': gevs.append(output.detach().clone())
        return record
    watched = [('cnn',model.encoder),('gev',model.gev_module.regularization),
               ('context',model.context_proj),('hidden',model.state_proj),
               ('motion',model.update_block.encoder),('gru',model.update_block.gru)]
    if use_da:
        watched += [('geometry',model.dav2_geometry),('adapter',model.dav2_geometry.fuse),('fusion',model.geometry_fusion)]
    hooks=[layer.register_forward_hook(hook(name)) for name,layer in watched]
    def record_inputs(module,args):
        contexts.append(args[2].detach().clone())
        states.append(args[0].detach().clone())
    hooks.append(model.update_block.register_forward_pre_hook(record_inputs))
    hooks.append(model.context_proj.register_forward_pre_hook(lambda m,a: head_inputs.append(a[0].detach().clone())))
    hooks.append(model.state_proj.register_forward_pre_hook(lambda m,a: head_inputs.append(a[0].detach().clone())))
    frozen_before = frozen_hash(model.dav2_geometry.backbone) if use_da else None
    model.train()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=amp and device.type=='cuda',init_scale=128.)
    losses=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    import time
    started=time.perf_counter()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output=model(images,iters=3,dav2_rgb=rgb if use_da else None,
                     yang_to_yin_quality=qualities[0],yin_to_yang_quality=qualities[1])
        loss=0.
        for name,row in output.items():
            assert row['gev_volume'].shape==(1,1,depth,*getattr(package,name).shape)
            assert row['final_depth_idx'].shape==(1,1,*(2*n for n in getattr(package,name).shape))
            assert len(row['predictions'])==len(row['native_predictions'])==3
            for prediction in row['predictions']:
                assert torch.isfinite(prediction).all() and prediction.min()>=0 and prediction.max()<=depth-1
                assert not prediction.masked_select(~row['valid']).any()
            gt=torch.full_like(row['final_depth_idx'],depth*.3)
            loss=loss+sequence_loss(row['predictions'],gt,row['valid'])
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        modules=[model.encoder,model.gev_module,model.similarity_context.spatial_aggregator,
                 model.context_proj,model.state_proj,model.update_block.encoder,model.update_block.gru,
                 model.update_block.depth_head,model.update_block.mask]
        if use_da:
            modules += [model.dav2_geometry.projections,model.dav2_geometry.fuse,model.dav2_geometry.refine,model.geometry_fusion]
            assert all(p.grad is None and not p.requires_grad for p in model.dav2_geometry.backbone.parameters())
            assert not model.dav2_geometry.backbone.training
        for module in modules:
            grads=[p.grad for p in module.parameters() if p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), (type(module).__name__, [n for n,p in module.named_parameters() if p.requires_grad and p.grad is None])
            assert sum(float(g.abs().sum()) for g in grads)>0
        for image in images:
            assert image.grad is not None and torch.isfinite(image.grad).all() and image.grad.abs().sum()>0
        if step==1:
            assert sum(float(p.grad.abs().sum()) for p in model.enhancer.geometry_encoder.parameters())>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer);scaler.update()
        losses.append(float(loss.detach()))
        if step==0:
            assert counts['cnn']==1 and counts['gev']==2 and counts['context']==counts['hidden']==2
            assert counts['motion']==counts['gru']==6
            if use_da:
                assert counts['adapter']==1 and counts['geometry']==counts['fusion']==2
                for i in range(2):
                    torch.testing.assert_close(head_inputs[2*i],fusion_outputs[i])
                    torch.testing.assert_close(head_inputs[2*i+1],fusion_outputs[i])
            for base in (0,3):
                assert all(torch.equal(contexts[base],contexts[i]) for i in (base+1,base+2))
                assert not torch.equal(states[base],states[base+2])
    for h in hooks:h.remove()
    assert frozen_before==(frozen_hash(model.dav2_geometry.backbone) if use_da else None)
    model.eval()
    # 只改变辅助RGB，匹配证据必须相同，而原GRU收到的Context必须改变。
    dependence=None
    if use_da:
        matching=[]; fused=[]
        handles=[model.gev_module.regularization.register_forward_hook(lambda m,a,o:matching.append(o.detach().clone())),
                 model.geometry_fusion.register_forward_hook(lambda m,a,o:fused.append(o.detach().clone()))]
        with torch.no_grad():
            for auxiliary in (rgb,rgb+.1):
                model(images,iters=1,dav2_rgb=auxiliary,yang_to_yin_quality=qualities[0],yin_to_yang_quality=qualities[1])
        for i in (0,1):
            torch.testing.assert_close(matching[i],matching[i+2],rtol=0,atol=0)
            assert not torch.equal(fused[i],fused[i+2])
        for h in handles:h.remove()
        dependence=True
    # 保存/严格重载整个模型状态；不忽略尺寸或未知参数。
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'model.pth'
        torch.save(model.state_dict(),path)
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True),strict=True)
    # 全无匹配候选也不能产生NaN或伪有效深度。
    package.yin.stereo_valid[:]=False;package.yang.stereo_valid[:]=False
    with torch.no_grad():
        empty=model(images,iters=1,dav2_rgb=rgb if use_da else None,
                    yang_to_yin_quality=qualities[0],yin_to_yang_quality=qualities[1])
    assert all(not row['valid'].any() and not row['final_depth_idx'].any() for row in empty.values())
    if device.type=='cuda':torch.cuda.synchronize(device)
    return dict(status='PASS',use_dav2=use_da,amp=amp,device=str(device),depth=depth,
                elapsed_seconds=time.perf_counter()-started,synthetic_losses=losses,
                output_shapes={name:list(row['final_depth_idx'].shape) for name,row in output.items()},shapes=shapes,
                original_motion_input_channels=[36,32,18,1],shared_baseline_keys=len(shared_keys),transfer=transfer,
                checkpoint_transfer=loaded_checkpoint,backbone_frozen=True if use_da else None,
                rgb_changes_context_without_changing_gev=dependence,
                peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                supervision='synthetic candidate-index targets; no accuracy claim')


def synthetic_run(device,amp=False,use_da=True):
    # 注入显式的合成冻结ViT，Adapter/Fusion/原Stereo模块均为实际生产实现。
    from module_cnn_only import dav2_geometry as geometry_module
    module=types.ModuleType('depth_anything_v2.dinov2');module.DINOv2=SyntheticViT
    parent=types.ModuleType('depth_anything_v2')
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint=Path(tmp)/'synthetic.pth'
        torch.save({'pretrained.'+k:v for k,v in SyntheticViT().state_dict().items()},checkpoint)
        with patch.dict(sys.modules,{'depth_anything_v2':parent,'depth_anything_v2.dinov2':module}), \
             patch.object(geometry_module,'add_official_source',return_value=None):
            return check_network(device,amp,use_da,repo=tmp,checkpoint=checkpoint)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu')
    p.add_argument('--model_repo',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--baseline_checkpoint',type=Path)
    p.add_argument('--data_root',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args();device=torch.device(args.device)
    torch.set_num_threads(2)
    if device.type=='cuda':torch.cuda.set_device(device)
    if args.model_repo:
        rows=[check_network(device,True,True,args.model_repo,args.checkpoint,args.data_root,args.baseline_checkpoint)]
        kind='official DA-V2 ViT-B; real calibrated RGB' if args.data_root else 'official DA-V2; synthetic inputs'
    else:
        rows=[synthetic_run(device,False,False),synthetic_run(device,False,True),synthetic_run(device,True,True)]
        kind='synthetic frozen ViT; real Adapter/Fusion/GEV/UpdateBlock code'
    result=dict(status='PASS',scope=kind,torch=torch.__version__,runs=rows)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
