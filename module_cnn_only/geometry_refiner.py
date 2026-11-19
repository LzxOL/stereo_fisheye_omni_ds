"""独立的球面扫描特征增强；不执行投影、扫描、GWC或深度更新。

只创建一个 GeometryAwareFeatureEnhancer，然后分别调用 Yin、Yang：
    yin_l, yin_r = enhancer(yin_l, yin_r, yin.rays_ref, yin_mask, yin.padding_valid)
    yang_l, yang_r = enhancer(yang_l, yang_r, yang.rays_ref, yang_mask, yang.padding_valid)
四个体统一使用 [B,C,D,H,W]；H/W无需相同。每次左右配对调用只编码一次几何。
"""
import math

import torch
import torch.nn as nn


class GeometryEncoder(nn.Module):
    """参考 PDF-Omni GE：XYZ → 固定随机Fourier映射 → 两层MLP。

    rays_ref=[H,W,3]，可直接传入原NumPy float64；padding_valid=[H,W]。
    输出 [B,16,H,W]，同一网格各Batch及左右相机共享编码。
    """
    def __init__(self, out_channels=16, mapping_size=256, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(3, mapping_size) * scale)
        self.mlp = nn.Sequential(
            nn.Linear(2 * mapping_size, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, rays_ref, padding_valid, batch_size=1):
        xyz = torch.as_tensor(rays_ref, device=self.B.device, dtype=torch.float32)
        valid = torch.as_tensor(padding_valid, device=self.B.device, dtype=torch.bool)
        valid = valid & torch.isfinite(xyz).all(dim=-1)
        # Padding射线为NaN：必须在乘Fourier矩阵前清除，不能用 NaN * 0。
        xyz = xyz.masked_fill(~valid[..., None], 0.0)

        # 相位保持FP32；外层AMP不应把高频随机投影先量化到FP16。
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            phase = (2.0 * math.pi) * (xyz @ self.B.float())
            fourier = torch.cat([phase.sin(), phase.cos()], dim=-1)
        encoded = self.mlp(fourier.to(self.mlp[0].weight.dtype))
        encoded = encoded.permute(2, 0, 1).unsqueeze(0)  # [1,16,H,W]
        encoded = encoded.masked_fill(~valid[None, None], 0.0)
        return encoded.expand(batch_size, -1, -1, -1)


class LocalFeatureRefiner(nn.Module):
    """每个D切片独立做2D局部增强，不沿深度卷积。

    [Feature(32), Geometry(16), Valid(1)] → 1x1 Conv → ReLU → 3x3 Conv。
    最后一层零初始化，初始输出等于经过有效性屏蔽的输入。
    """
    def __init__(self, channels=32, geometry_channels=16, depth_chunk=8):
        super().__init__()
        self.depth_chunk = depth_chunk
        self.use_checkpoint = False
        self.fusion = nn.Conv2d(channels + geometry_channels + 1, channels, 1)
        self.relu = nn.ReLU()
        self.local = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.local.weight)
        nn.init.zeros_(self.local.bias)

    def _refine_slice(self, feature, geo, mask):
        base = feature.masked_fill(~mask, 0.0)
        fused = torch.cat(
            [base, geo.masked_fill(~mask, 0.0), mask.to(base.dtype)], dim=1,
        )
        hidden = self.relu(self.fusion(fused))
        # 不让无效位置的1x1偏置经3x3传播到有效邻居；不做补洞或权重重归一化。
        hidden = hidden.masked_fill(~mask, 0.0)
        enhanced = (base + self.local(hidden)).masked_fill(~mask, 0.0)
        return enhanced

    def forward(self, volume, geometry, stereo_valid, padding_valid):
        batch, channels, depth, height, width = volume.shape
        valid = torch.as_tensor(stereo_valid, device=volume.device, dtype=torch.bool)
        padding = torch.as_tensor(padding_valid, device=volume.device, dtype=torch.bool)
        valid = valid & padding[None, None, None]
        geometry = geometry.to(dtype=volume.dtype)
        outputs = []

        for start in range(0, depth, self.depth_chunk):
            stop = min(start + self.depth_chunk, depth)
            count = stop - start
            # B,D相邻后再合并，避免把C/D维直接reshape而打乱对应关系。
            feature = volume[:, :, start:stop].permute(0, 2, 1, 3, 4)
            feature = feature.reshape(batch * count, channels, height, width)
            mask = valid[:, :, start:stop].permute(0, 2, 1, 3, 4)
            mask = mask.reshape(batch * count, 1, height, width)
            geo = geometry[:, None].expand(-1, count, -1, -1, -1)
            geo = geo.reshape(batch * count, geometry.shape[1], height, width)

            if self.use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                enhanced = checkpoint(self._refine_slice, feature, geo, mask, use_reentrant=False)
            else:
                enhanced = self._refine_slice(feature, geo, mask)

            enhanced = enhanced.reshape(batch, count, channels, height, width)
            outputs.append(enhanced.permute(0, 2, 1, 3, 4))
        # 分块限制单次Conv2d工作量；训练时autograd仍保留反向所需的中间量。
        return torch.cat(outputs, dim=2)


class GeometryAwareFeatureEnhancer(nn.Module):
    """共享一套GE与Local Refiner；每次处理同一网格的左、右扫描体。

    Geometry Encoder每次只调用一次，产生 [B,16,H,W]；不对Yin/Yang进行resize。
    最后一层零初始化时，首轮GE/1x1的梯度值为0；末层更新后即可传入上游。
    """
    def __init__(
        self, channels=32, geometry_channels=16, depth_chunk=8,
        mapping_size=256, fourier_scale=10.0,
    ):
        super().__init__()
        self.geometry_encoder = GeometryEncoder(geometry_channels, mapping_size, fourier_scale)
        self.local_refiner = LocalFeatureRefiner(channels, geometry_channels, depth_chunk)

    def forward(self, left_volume, right_volume, rays_ref, stereo_valid, padding_valid):
        geometry = self.geometry_encoder(rays_ref, padding_valid, left_volume.shape[0])
        left = self.local_refiner(left_volume, geometry, stereo_valid, padding_valid)
        right = self.local_refiner(right_volume, geometry, stereo_valid, padding_valid)
        return left, right
