"""共享球面 CNN。两幅 atlas 使用相同角坐标基底，原点分别在左右光心。"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from dataset_stereo import project_camera
from experiments.local_spherical_grid.adaptive_local_grid_v1 import pixel_to_normalized
from experiments.local_spherical_grid.camera_geometry import pair_geometry


class ChartSampling(nn.Module):
    """缓存 atlas RGB 与右侧逐候选查询坐标。标定表随设备迁移，不写入权重。"""
    def __init__(self, patch, cameras, axis, stride):
        super().__init__()
        rotation, translation, _ = pair_geometry(*cameras)
        h, w = patch.shape
        # 对称 padding 的奇数卷积，输出 j 的中心恰为输入 stride*j。
        # 每两个工作像素中心之间放 stride 个间隔，包含首尾两个中心。
        y, x = np.indices((stride * (h - 1) + 1, stride * (w - 1) + 1))
        atlas_xy = np.stack([x, y], -1) / stride
        rays, inside = patch.grid_to_direction(pixel_to_normalized(atlas_xy, patch.shape))
        self.image_sizes = [(cam.height, cam.width) for cam in cameras]
        supports = []
        for side, (camera, directions) in enumerate(zip(cameras, (rays, rays @ rotation.T))):
            uv, visible = project_camera(camera, directions.reshape(-1, 3).T)
            valid = inside & visible.reshape(inside.shape)
            coords = pixel_to_normalized(uv.T.reshape(*inside.shape, 2), self.image_sizes[side])
            self.buffer(
                'rgb_grid_' + str(side),
                np.where(valid[..., None], coords, 2.).astype('float32'),
            )
            supports.append(valid[::stride, ::stride])
        self.buffer('left_valid', supports[0] & patch.left_feature_valid)

        grids, masks = [], []
        for rho in axis:
            # 正的 rho 包括第0层远场近似。R*r + rho*t 与 X_R 同方向。
            direction_right = patch.rays_ref @ rotation.T + rho * translation
            coords, window = patch.direction_to_grid(direction_right @ rotation)
            coords = coords.astype('float32')
            # 按实际 FP32 grid_sample 坐标检查右 atlas 的所有活动插值邻点。
            xy = (coords + 1) * np.array([w - 1, h - 1], np.float32) / 2
            x, y = xy[..., 0], xy[..., 1]
            valid = window.copy()
            for ix in (np.floor(x).astype(int), np.ceil(x).astype(int)):
                for iy in (np.floor(y).astype(int), np.ceil(y).astype(int)):
                    valid &= (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
                    valid &= supports[1][iy.clip(0, h - 1), ix.clip(0, w - 1)]
            masks.append(valid)
            grids.append(np.where(valid[..., None], coords, 2.))
        self.buffer('right_grid', np.stack(grids))
        right_valid = np.stack(masks)
        self.buffer('right_valid', right_valid)
        # 分母只计原球面扫描有效候选，便于量化局部 atlas 的覆盖及回退。
        joint = patch.stereo_valid
        self.support_counts = (int((joint & right_valid & supports[0]).sum()), int(joint.sum()))

    def buffer(self, name, array):
        self.register_buffer(name, torch.from_numpy(np.ascontiguousarray(array)), persistent=False)


def sample(image, grid):
    """同一标定的 batch 共用 [H,W,2] 坐标，AMP 下几何采样仍用 FP32。"""
    with torch.autocast(device_type=image.device.type, enabled=False):
        value = F.grid_sample(
            image.float(),
            grid[None].expand(len(image), -1, -1, -1),
            padding_mode='zeros',
            align_corners=True,
        )
    return value.to(image.dtype)


class LocalSphericalEncoder(nn.Module):
    def __init__(
        self, package, cameras, channels=32, width=16, layers=2, stride=2,
        train_encoder=True, residual=True, depth_chunk=8, recompute=False,
    ):
        super().__init__()
        if cameras is None or min(width, layers, stride, depth_chunk) < 1:
            raise ValueError('Local spherical encoder requires two calibrated cameras and positive dimensions')
        self.charts = nn.ModuleDict({
            name: ChartSampling(
                getattr(package, name), cameras,
                package.metadata['inverse_depth_candidates'], stride,
            )
            for name in ('yin', 'yang')
        })

        blocks = [nn.Conv2d(3, width, 3, stride, padding=1), nn.ReLU()]
        for _ in range(layers - 1):
            blocks.extend([nn.Conv2d(width, width, 3, padding=1), nn.ReLU()])
        self.cnn = nn.Sequential(*blocks, nn.Conv2d(width, channels, 1)).requires_grad_(train_encoder)
        self.residual = residual
        if residual:
            self.fusion = nn.Conv3d(2 * channels, channels, 1)
            self.alpha = nn.Parameter(torch.tensor(.01))
        self.chunk = depth_chunk
        self.recompute = recompute

    def combine(self, base, local, valid):
        if not self.residual:  # 消融：有局部观测处取 Local，缺失处仍回退 Global。
            return torch.where(valid, local, base)
        delta = self.fusion(torch.cat((base, local), 1))
        return torch.where(valid, base + self.alpha * delta, base)

    def fuse_chunk(self, left, right, left_map, right_map, query, left_valid, right_valid):
        # 右侧按候选深度查图。左侧在同一条射线上沿 D 广播。
        local_right = sample(right_map, query.flatten(0, 1)).reshape_as(right)
        local_left = left_map.unsqueeze(2).expand_as(left)
        return (
            self.combine(left, local_left, left_valid),
            self.combine(right, local_right, right_valid),
        )

    def forward(self, name, rgb, left, right, stereo_valid):
        """每个 chart 编码左右 RGB 一次，逐 D 块对齐，返回两份 [B,C,D,H,W]。"""
        chart = self.charts[name]

        # 1. 原鱼眼 RGB → 左右局部球面图 → 共享 CNN 的两张二维特征图。
        spherical_images = []
        for side, image in enumerate(rgb):
            if image.shape[1:] != (3, *chart.image_sizes[side]):
                raise ValueError('Local input must be native calibrated RGB [B,3,Hfish,Wfish]')
            rgb_grid = getattr(chart, 'rgb_grid_' + str(side))
            spherical_images.append(sample(image, rgb_grid))
        left_map, right_map = self.cnn(torch.cat(spherical_images)).chunk(2)

        # 2. 每次对齐一块深度候选，并与同一位置的 Global 特征融合。
        left_chunks = []
        right_chunks = []
        for start in range(0, right.shape[2], self.chunk):
            depth_slice = slice(start, start + self.chunk)
            right_grid = chart.right_grid[depth_slice]
            stereo_support = stereo_valid[:, :, depth_slice]
            left_valid = stereo_support & chart.left_valid
            right_valid = stereo_support & chart.right_valid[depth_slice]
            args = (
                left[:, :, depth_slice], right[:, :, depth_slice],
                left_map, right_map, right_grid, left_valid, right_valid,
            )
            if self.recompute and self.training:
                fused_left, fused_right = checkpoint(self.fuse_chunk, *args, use_reentrant=False)
            else:
                fused_left, fused_right = self.fuse_chunk(*args)
            left_chunks.append(fused_left)
            right_chunks.append(fused_right)

        # 3. 沿深度维拼回左右完整体 [B,C,D,H,W]。
        return torch.cat(left_chunks, dim=2), torch.cat(right_chunks, dim=2)
