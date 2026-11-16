"""小尺寸训练适配测试：射线投影、索引单位、有效性、梯度和可选重计算。"""
from pathlib import Path
import sys
import json
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.smoke_dual_grid_refinement import make_package
from module_cnn_only.network import OmniDS
from module_cnn_only.yinyang_training import YinYangTrainingGeometry, sampling_quality
from module_cnn_only.geometry_refiner import LocalFeatureRefiner


def main():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    package = make_package()
    # 测试目标沿用Yin射线；生产环境由冻结包提供真实左鱼眼320×480射线。
    package.reference.update(rays_ref=package.yin.rays_ref, left_valid=package.yin.padding_valid)
    for name in ('yin', 'yang'):
        grid, valid = getattr(package, name).direction_to_grid(package.reference['rays_ref'])
        package.reference['to_'+name] = grid
        package.reference['in_'+name] = valid & package.reference['left_valid']
    geometry = YinYangTrainingGeometry(package)
    result = {}
    for name in ('yin', 'yang'):
        h, w = getattr(package, name).shape
        prediction = torch.full((1, 1, 2*h, 2*w), 3.25, requires_grad=True)
        valid = torch.tensor(getattr(package, name).padding_valid)[None, None].repeat_interleave(2, -2).repeat_interleave(2, -1)
        result[name] = dict(predictions=[prediction], valid=valid)
        quality = sampling_quality(getattr(package, name))
        assert np.isfinite(quality).all() and quality.max() <= 1 and quality.min() >= 0
        assert len(np.unique(quality)) > 2  # 质量并非用二值Mask冒充。
    projected, valid = geometry.project(result)
    assert valid.any() and not projected[0].masked_select(~valid).any()
    torch.testing.assert_close(projected[0][valid], torch.full_like(projected[0][valid], 3.25))
    projected[0].sum().backward()
    assert all(row['predictions'][0].grad.abs().sum() > 0 for row in result.values())
    # 在自身网格采样位置2*x处取线性斜坡，检查凸上采样坐标无半像素偏移。
    h, w = package.yin.shape
    ramp = torch.arange(2*w).float()[None, None, None].expand(1, 1, 2*h, -1)
    result['yin']['predictions'] = [ramp]
    result['yang']['valid'] = torch.zeros_like(result['yang']['valid'])
    ramp_out, ramp_valid = geometry.project(result)
    expected = (2*torch.arange(w)).float()[None, None, None].expand(1, 1, h, -1)
    torch.testing.assert_close(ramp_out[0][ramp_valid], expected[ramp_valid], atol=1e-4, rtol=0)
    for row in result.values():
        row['valid'].zero_()
    empty, mask = geometry.project(result)
    assert not mask.any() and not empty[0].any()

    # 完整双网格→同一左参考输出→Sequence Loss，反向至两路CNN特征及新前端。
    model = OmniDS(dict(num_cams=2, use_yinyang=True, training_geometry=True,
                         yinyang_output='left_fisheye', num_invdepth=16, num_depth_samples=16,
                         gradient_checkpointing=True), grid_package=package).train()
    images = [torch.randn(1, 3, 32, 40, requires_grad=True) for _ in range(2)]
    predictions = model(images, iters=3)
    from module_cnn_only.loss_functions import sequence_loss
    gt = torch.full_like(predictions[-1], 5.)
    loss = sequence_loss(predictions, gt, model.output_valid)
    loss.backward()
    assert torch.isfinite(loss)
    for image in images:
        assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
    assert model.enhancer.local_refiner.local.weight.grad.abs().sum() > 0
    assert model.fusion.network[-1].weight.grad.abs().sum() > 0
    # 重计算只改变激活保存策略，不改变数值或梯度。
    refiner = LocalFeatureRefiner(depth_chunk=2).train()
    with torch.no_grad():
        refiner.local.weight.normal_(std=.01)
    volume = torch.randn(1, 32, 4, 5, 7, requires_grad=True)
    encoded = torch.randn(1, 16, 5, 7, requires_grad=True)
    support = torch.rand(1, 1, 4, 5, 7) > .2
    padding = torch.ones(5, 7, dtype=torch.bool)
    first = refiner(volume, encoded, support, padding)
    grad = torch.autograd.grad(first.sum(), (volume, encoded))
    refiner.use_checkpoint = True
    second = refiner(volume, encoded, support, padding)
    grad2 = torch.autograd.grad(second.sum(), (volume, encoded))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for a, b in zip(grad, grad2):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    update = model.update_block
    tensors = [torch.randn(1, c, h, w, requires_grad=True)
               for c in (64, 16, 32, 18, 1)]
    tensors[-1] = (tensors[-1].detach().sigmoid()*15).requires_grad_()
    parameters = tuple(p for layer in (update.encoder, update.gru, update.depth_head, update.mask)
                       for p in layer.parameters())
    update.use_checkpoint = False
    first = update(*tensors)
    grad = torch.autograd.grad(sum(t.square().mean() for t in first), (*tensors, *parameters))
    update.use_checkpoint = True
    second = update(*tensors)
    grad2 = torch.autograd.grad(sum(t.square().mean() for t in second), (*tensors, *parameters))
    for a, b in zip((*first, *grad), (*second, *grad2)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    restored = OmniDS(dict(model.opts), grid_package=package)
    restored.load_state_dict(model.state_dict(), strict=True)
    print(json.dumps(dict(status='PASS', predictions=[list(p.shape) for p in predictions],
                          loss=float(loss.detach()), valid_fraction=model.output_valid.float().mean().item(),
                          constant_index_preserved=True, no_half_pixel_shift=True, empty_mask_zero=True,
                          gradient_to_both_images=True, checkpoint_equivalence=True,
                          gru_checkpoint_output_and_gradient_equivalence=True, strict_reload=True), indent=2))


if __name__ == '__main__':
    main()
