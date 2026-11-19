"""训练几何适配：固定GPU网格、真实采样质量、双网格预测→左鱼眼输出。

不改变标定/扫描/GRU；GT仍用StereoDataset已有的Rig→左光心点云转换。
"""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from experiments.local_spherical_grid.adaptive_local_grid_v1 import sample_patch
from module_cnn_only.cross_grid_warp import _CachedQuery


def sampling_quality(patch):
    # 真实原生像素的两个球面边长：cos(latitude)*dLongitude 与 dLatitude。
    # Q=1/kappa∈(0,1]，数值越大采样越均匀；不是二值有效性Mask。
    bounds = patch.bounds
    height, width = patch.original_shape
    visual = np.nan_to_num(patch.rays_ref) @ np.diag([-1., -1., 1.]) @ patch.basis
    cos_lat = np.sqrt(np.maximum(0., 1. - visual[..., 1] ** 2))
    dx = np.deg2rad(2 * bounds['longitude_half_span_deg'] / (width - 1)) * cos_lat
    dy = np.deg2rad((bounds['latitude_max_deg'] - bounds['latitude_min_deg']) / (height - 1))
    quality = np.minimum(dx, dy) / np.maximum(np.maximum(dx, dy), 1e-12)
    return np.where(patch.padding_valid, quality, 0.).astype(np.float32)


class YinYangTrainingGeometry(nn.Module):
    def __init__(self, package, factor=2):
        super().__init__()
        self.factor = factor
        self.shapes = {name: getattr(package, name).shape for name in ('yin', 'yang')}
        self.output_size = package.reference['left_valid'].shape

        for name, source in [('yin', 'yang'), ('yang', 'yin')]:
            patch = getattr(package, name)
            # 非持久buffer随model.cuda移动，但不将巨大的几何表重复写入每个checkpoint。
            fields = [
                'left_grid', 'right_grid', 'stereo_valid',
                'left_feature_valid', 'padding_valid', 'rays_ref',
            ]
            for field in fields:
                dtype = torch.bool if field.endswith('valid') else torch.float32
                self.register_buffer(
                    name + '_' + field,
                    torch.as_tensor(getattr(patch, field), dtype=dtype).clone(),
                    persistent=False,
                )

            # 将另一个网格的采样质量投到当前网格，供跨网格 GWC 融合用。
            src = getattr(package, source)
            quality = torch.tensor(sampling_quality(src))[None, None]
            support = torch.tensor(src.padding_valid)[None, None]
            q, _ = sample_patch(quality, src, patch.rays_ref, support)
            self.register_buffer(name + '_cross_quality', q.unsqueeze(2), persistent=False)

            # 将原生连续像素x映射到凸上采样后的factor*x，匹配原网络的stride-2中心定义。
            # 不能直接复用旧归一化坐标，也不能随意加入半像素偏移。
            coordinates = np.asarray(package.reference['to_' + name])
            h, w = patch.shape
            scale = np.array([
                factor * (w - 1) / (factor * w - 1),
                factor * (h - 1) / (factor * h - 1),
            ])
            up_coordinates = (coordinates + 1) * scale - 1
            window = np.asarray(package.reference['in_' + name])
            up_coordinates = np.where(window[..., None], up_coordinates, 2.)
            self.register_buffer(
                name + '_output_grid', torch.tensor(up_coordinates, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(name + '_output_window', torch.tensor(window), persistent=False)

            own_quality = torch.tensor(sampling_quality(patch))[None, None]
            qout, _ = sample_patch(
                own_quality, patch, package.reference['rays_ref'],
                torch.tensor(patch.padding_valid)[None, None],
            )
            self.register_buffer(name + '_output_quality', qout, persistent=False)

    def patch(self, name):
        return SimpleNamespace(**{
            field: getattr(self, name + '_' + field)
            for field in [
                'left_grid', 'right_grid', 'stereo_valid',
                'left_feature_valid', 'padding_valid', 'rays_ref',
            ]
        })

    def cross_quality(self, name, batch, depth):
        return getattr(self, name + '_cross_quality').expand(batch, 1, depth, -1, -1)

    def project(self, result):
        """每轮[B,1,2Hchart,2Wchart]→[B,1,Hleft,Wleft]；严格检查插值邻点。

        重叠处按几何Q归一化混合，只有一路有效时完全取该路；两路都无效为零。
        同一射线的候选索引共享轴，空间插值/融合不改变索引单位。
        """
        predictions = []
        for iteration in range(len(result['yin']['predictions'])):
            values = []
            weights = []
            for name in ('yin', 'yang'):
                h, w = self.shapes[name]
                query = _CachedQuery(
                    (self.factor * h, self.factor * w),
                    getattr(self, name + '_output_grid'),
                    getattr(self, name + '_output_window'),
                )
                sampled, valid = sample_patch(
                    result[name]['predictions'][iteration], query, None, result[name]['valid'],
                )
                quality = getattr(self, name + '_output_quality')
                values.append(sampled.float())
                weights.append(quality.masked_fill(~valid, 0.))

            total = weights[0] + weights[1]
            prediction = (values[0] * weights[0] + values[1] * weights[1]) / total.clamp_min(1e-8)
            output_valid = total > 0
            predictions.append(prediction.masked_fill(~output_valid, 0.))
        return predictions, output_valid
