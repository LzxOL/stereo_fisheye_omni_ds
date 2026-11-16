"""独立小尺寸测试：python tools/smoke_geometry_refiner.py [--device cpu]。

不读取数据集、不创建Stereo网络、不写checkpoint；只验证新增强模块。
"""
import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from module_cnn_only.geometry_refiner import GeometryAwareFeatureEnhancer


def fixture(height, width, device):
    batch, channels, depth = 2, 32, 10  # D=10验证8+2分块，B=2验证B/D没有混排。
    rays = np.random.default_rng(height).normal(size=(height, width, 3))
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    padding = np.ones((height, width), dtype=bool)
    padding[-1] = False
    padding[:, -2:] = False
    rays[~padding] = np.nan
    valid = torch.rand(batch, 1, depth, height, width, device=device) > .2
    valid[:, :, :, 1, 1] = False  # 非Padding、所有深度均不可匹配的位置。
    # 故意使Padding的stereo_valid为True，确认模块确实合并padding_valid。
    valid[..., ~torch.as_tensor(padding, device=device)] = True
    effective = valid & torch.as_tensor(padding, device=device)[None, None, None]
    volumes = [torch.randn(batch, channels, depth, height, width, device=device) for _ in range(2)]
    for volume in volumes:
        volume.masked_fill_(~effective, float('nan'))  # 无效NaN不得污染邻居或残差。
        volume.requires_grad_()
    return dict(volumes=volumes, rays=rays, padding=padding, valid=valid, effective=effective)


def forward_pair(model, data):
    return model(*data['volumes'], data['rays'], data['valid'], data['padding'])


def check_outputs(outputs, data):
    for original, enhanced in zip(data['volumes'], outputs):
        assert original.shape == enhanced.shape
        assert torch.isfinite(enhanced).all()
        assert not enhanced.masked_select(~data['effective'].expand_as(enhanced)).any()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(20261116)
    model = GeometryAwareFeatureEnhancer().to(device)
    yin, yang = fixture(7, 11, device), fixture(5, 9, device)
    calls = dict(geometry=0, local=0)
    geometry_shapes = []
    conv_batches = []

    def geo_hook(module, inputs, output):
        calls['geometry'] += 1
        geometry_shapes.append(list(output.shape))

    def local_hook(module, inputs, output):
        calls['local'] += 1

    hooks = [model.geometry_encoder.register_forward_hook(geo_hook),
             model.local_refiner.register_forward_hook(local_hook),
             model.local_refiner.fusion.register_forward_pre_hook(lambda m, args: conv_batches.append(args[0].shape[0]))]
    initial = [forward_pair(model, data) for data in (yin, yang)]
    assert calls == dict(geometry=2, local=4), calls
    assert max(conv_batches) <= 2 * 8 and 2 * 2 in conv_batches
    for data, outputs in zip((yin, yang), initial):
        check_outputs(outputs, data)
        for original, enhanced in zip(data['volumes'], outputs):
            torch.testing.assert_close(enhanced, original.masked_fill(~data['effective'], 0), rtol=0, atol=0)
    for hook in hooks:
        hook.remove()
    assert 'geometry_encoder.B' in dict(model.named_buffers())
    assert 'geometry_encoder.B' not in dict(model.named_parameters())
    original_B = model.geometry_encoder.B.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    first_loss = sum(x.square().mean() for pair in initial for x in pair)
    first_loss.backward()
    assert model.local_refiner.local.weight.grad.abs().sum() > 0
    for parameter in model.geometry_encoder.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert not parameter.grad.any()  # 严格零初始化阻断首轮上游非零梯度，属于预期行为。
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for data in (yin, yang):
        for v in data['volumes']:
            v.grad = None
    updated = [forward_pair(model, data) for data in (yin, yang)]
    for outputs, data in zip(updated, (yin, yang)):
        check_outputs(outputs, data)
    loss = sum(x.square().mean() for pair in updated for x in pair)
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        gradients[name] = float(parameter.grad.abs().sum())
        assert gradients[name] > 0, name
    for data in (yin, yang):
        for v in data['volumes']:
            assert v.grad is not None and torch.isfinite(v.grad).all()
            assert v.grad.abs().sum() > 0
    torch.testing.assert_close(model.geometry_encoder.B, original_B, rtol=0, atol=0)

    with torch.no_grad():
        # 分块大小不改变结果；改变一个D切片也不能改变其它D切片或另一个Batch。
        model.local_refiner.depth_chunk = 3
        chunked = forward_pair(model, yin)
        for a, b in zip(chunked, updated[0]):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        altered = [v.detach().clone() for v in yin['volumes']]
        altered[0][0, :, 4] += 10
        changed = model(*altered, yin['rays'], yin['valid'], yin['padding'])
        other_depths = [i for i in range(10) if i != 4]
        torch.testing.assert_close(changed[0][:, :, other_depths], chunked[0][:, :, other_depths])
        torch.testing.assert_close(changed[0][1], chunked[0][1])
        torch.testing.assert_close(changed[1], chunked[1])

    # 全无效与NaN射线/特征仍输出有限零，反向保持有限零梯度。
    model.zero_grad(set_to_none=True)
    empty = fixture(5, 9, device)
    empty['valid'].zero_()
    empty['padding'][:] = False
    empty['rays'][:] = np.nan
    for v in empty['volumes']:
        with torch.no_grad():
            v.fill_(float('nan'))
    zeros = forward_pair(model, empty)
    assert all(torch.isfinite(v).all() and not v.any() for v in zeros)
    sum(v.sum() for v in zeros).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and not p.grad.any() for p in model.parameters())
    assert all(torch.isfinite(v.grad).all() and not v.grad.any() for v in empty['volumes'])

    # Buffer随state_dict严格恢复；不在磁盘写临时文件。
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = GeometryAwareFeatureEnhancer().to(device)
    restored.load_state_dict(torch.load(buffer, map_location=device, weights_only=True), strict=True)
    torch.testing.assert_close(restored.geometry_encoder.B, model.geometry_encoder.B, rtol=0, atol=0)
    print(json.dumps(dict(status='PASS', device=str(device), torch=torch.__version__,
        volume_shapes=[list(x.shape) for pair in updated for x in pair], geometry_shapes=geometry_shapes,
        calls_for_four_volumes=calls, max_conv_batch=max(conv_batches),
        zero_init_masked_identity=True, first_step_geometry_zero_gradient_expected=True,
        second_step_all_trainable_gradients_nonzero=True, depth_and_batch_independent=True,
        chunk_equivalence=True, invalid_and_padding_zero=True, all_invalid_backward_finite=True,
        strict_checkpoint_reload=True, trainable_parameters=sum(p.numel() for p in model.parameters()),
        fourier_buffer_elements=model.geometry_encoder.B.numel(), second_step_gradient_l1=gradients), indent=2))


if __name__ == '__main__':
    main()
