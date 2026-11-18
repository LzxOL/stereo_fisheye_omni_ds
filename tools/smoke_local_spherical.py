"""CPU smoke: saved cam4/cam3 OCam coefficients, synthetic RGB/masks/targets."""
from pathlib import Path
import sys
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.local_spherical_grid.adaptive_local_grid_v1 import AdaptiveLocalSphericalGrid, _make_patch, load_config
from experiments.local_spherical_grid.camera_geometry import pair_geometry
from module_cnn_only.network import OmniDS
from utils.ocam import OcamModel


def main():
    torch.manual_seed(19)
    torch.set_num_threads(2)
    cfg = load_config()
    cfg['generation']['native_angular_step_deg'] = 12.
    selection = json.loads((ROOT/'experiments/local_spherical_grid/selected_grid_yin50.json').read_text())
    cameras = []
    for entry in selection['provenance']:
        cam = OcamModel()
        cam.setConfig(entry['config'])
        cam.invalid_mask = np.zeros((cam.height,cam.width), bool)
        cam.invalid_mask[320:430, 300:410] = True
        cameras.append(cam)
    r,t,_ = pair_geometry(*cameras)
    axis = np.linspace(1e-10,.6,16)
    patches = [_make_patch(name,selection[name+'_bounds'],basis,cameras,r,t,axis,cfg)
               for name,basis in [('yin',np.eye(3)),('yang',np.array(selection['yang_basis_visual']))]]
    package = AdaptiveLocalSphericalGrid(*patches,{},dict(inverse_depth_candidates=axis.tolist()))
    quality = {}
    for dest,src in [('yin','yang'),('yang','yin')]:
        xy,v = package.correspondence(dest,src)
        package.reference[dest+'_to_'+src] = xy.astype('float32')
        package.reference[dest+'_in_'+src] = v
        quality[src+'_to_'+dest+'_quality'] = torch.ones(1,1,len(axis),*getattr(package,dest).shape)
    opts = dict(use_yinyang=True,num_invdepth=len(axis),num_depth_samples=len(axis))
    base = OmniDS(opts,package).eval()
    model = OmniDS(dict(opts,use_local_spherical_encoder=True),package,cameras).eval()
    transfer = model.load_baseline_weights(base.state_dict())
    # 全分辨率原 RGB；这里的 Global 输入只用于小规模网络回归。
    rgb = [torch.rand(1,3,c.height,c.width) for c in cameras]
    imgs = [torch.rand(1,1,48,64) for c in cameras]
    def predict(net, **kwargs):
        return net(imgs,iters=2,**quality,**kwargs)
    with torch.no_grad():
        ref = predict(base)
        model.local_spherical.alpha.zero_()
        zero = predict(model,local_rgb=rgb)
        for name in ('yin','yang'):
            assert torch.equal(ref[name]['valid'],zero[name]['valid'])
            torch.testing.assert_close(ref[name]['final_depth_idx'],zero[name]['final_depth_idx'],atol=0,rtol=0)
        model.local_spherical.alpha.fill_(.01)
        active = predict(model,local_rgb=rgb)
        diff = max((active[n]['final_depth_idx']-zero[n]['final_depth_idx']).abs().max().item() for n in zero)
        assert diff > 0

    base_volume = torch.randn(1,32,4,8,8)
    missing = torch.zeros(1,1,4,8,8,dtype=torch.bool)
    fallback = model.local_spherical.combine(base_volume,torch.zeros_like(base_volume),missing)
    assert torch.equal(fallback, base_volume)
    for p in patches:
        chart = model.local_spherical.charts[p.name]
        assert (chart.right_grid[0]!=chart.right_grid[-1]).any()
        assert not chart.right_valid[:,~p.padding_valid].any()
        # 独立检验右图查询：从 atlas 坐标恢复相机射线并直接对比同一 3D 点。
        rays, _ = p.grid_to_direction(chart.right_grid[-1].numpy())
        keep = chart.right_valid[-1].numpy()
        actual = rays[keep] @ r.T
        expected = (p.rays_ref[keep]/axis[-1]) @ r.T + t
        expected /= np.linalg.norm(expected,axis=-1,keepdims=True)
        np.testing.assert_allclose(actual,expected,atol=1e-6)
        uv = cameras[1].rayToPixel(actual.T)
        np.testing.assert_allclose(uv.T,p.right_uv[-1][keep],atol=.001)
        # OCam inverse polynomial is approximate; check angular roundtrip.
        back = cameras[1].pixelToRay(uv).T
        back /= np.linalg.norm(back,axis=-1,keepdims=True)
        assert np.max(np.arccos(np.clip((actual*back).sum(-1),-1,1))) < np.deg2rad(1.)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for step in range(2):
        optim.zero_grad()
        out = predict(model,local_rgb=rgb)
        loss = sum((v['final_depth_idx'][v['valid']]-4).abs().mean() for v in out.values())
        loss.backward()
        for mod in (model.encoder,model.local_spherical.cnn,model.local_spherical.fusion,
                    model.gev_module,model.update_block.gru):
            grads = [p.grad for p in mod.parameters() if p.grad is not None]
            assert grads and all(g.isfinite().all() for g in grads) and any(g.abs().sum()>0 for g in grads)
        optim.step()
        losses.append(loss.item())
    restored = OmniDS(dict(model.opts),package,cameras)
    restored.load_state_dict(model.state_dict(),strict=True)
    print(dict(losses=losses, output_delta=diff, inherited=transfer['inherited_keys'],
               parameters=sum(p.numel() for p in model.local_spherical.parameters()),
               charts=[p.shape for p in patches],status='PASS'))


if __name__ == '__main__':
    main()
