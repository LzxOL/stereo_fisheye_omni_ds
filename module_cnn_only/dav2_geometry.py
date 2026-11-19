"""Frozen official DA-V2 ViT-B + a small trainable, pixel-aligned adapter.

Only intermediate tokens are used. No DPT depth head, relative-depth prediction,
scale fitting, learned sampling offsets or extra attention is introduced.
"""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def add_official_source(model_repo):
    directory = str(Path(model_repo).resolve())
    if not (Path(directory) / 'depth_anything_v2' / 'dinov2.py').is_file():
        raise FileNotFoundError(f'Official DA-V2 source checkout not found: {directory}')
    if directory not in sys.path:
        sys.path.insert(0, directory)


def prepare_dav2_rgb(rgb, model_repo, input_size=518):
    """Original RGB uint8 [B,H,W,3] -> official-normalized [B,3,Hv,Wv].

    Reuse the RGB arrays already returned by StereoDataset: during training
    these have the SAME photometric augmentation as the stereo CNN inputs.
    No grayscale replication or inversion of CNN normalization is involved.
    """
    add_official_source(model_repo)
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
    import cv2
    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError('DA-V2 requires original RGB uint8 [B,H,W,3], not normalized CNN inputs')
    resize = Resize(
        input_size, input_size,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    )
    normalize = NormalizeImage(mean=[.485, .456, .406], std=[.229, .224, .225])
    prepare = PrepareForNet()
    # Official infer_image converts OpenCV BGR to RGB; dataset raw_imgs are
    # already RGB, so applying cvtColor here would incorrectly swap R and B.
    images = [prepare(normalize(resize({'image': image / 255.})))['image'] for image in rgb]
    return torch.from_numpy(np.stack(images))


def vit_sampling_grid(left_cnn_grid, image_size, cnn_size, vit_size):
    """CNN grid -> original pixel -> resized pixel -> ViT patch centre.

    Original CNN centres are pixels 2*j, addressed with align_corners=True.
    OpenCV resize maps u to (u+.5)*Wv/W-.5. A patch token is centred at
    14*j+6.5, not at pixel 14*j. Return align_corners=False token coordinates.
    Keeping these steps explicit avoids borrowing a CNN grid for patch tokens.
    """
    h, w = image_size
    hc, wc = cnn_size
    hv, wv = vit_size
    grid = left_cnn_grid.float()
    pixel = (grid + 1) * grid.new_tensor([wc - 1, hc - 1])  # stride 2 cancels /2
    resized = (pixel + .5) * grid.new_tensor([wv / w, hv / h]) - .5
    token = (resized + .5) / 14 - .5
    return 2 * (token + .5) / grid.new_tensor([wv // 14, hv // 14]) - 1


class ResidualRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
        )

    def forward(self, x):
        return F.relu(x + self.convs(x))


class DAV2GeometryContext(nn.Module):
    def __init__(self, model_repo, checkpoint):
        super().__init__()
        add_official_source(model_repo)
        from depth_anything_v2.dinov2 import DINOv2
        self.backbone = DINOv2(model_name='vitb')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        # Strictly load the official DA-V2 pretrained.* backbone. The unused
        # DPT depth_head is intentionally not constructed or registered.
        weights = {key[len('pretrained.'):]: value for key, value in state.items() if key.startswith('pretrained.')}
        self.backbone.load_state_dict(weights, strict=True)
        self.backbone.requires_grad_(False).eval()
        self.projections = nn.ModuleList([nn.Conv2d(768, 32, 1) for _ in range(4)])
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1),
        )
        self.refine = ResidualRefine()
        self.last_shapes = {}

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_features(self, rgb):  # ViT与Adapter每次前向只计算一次，输出左鱼眼Patch特征。
        if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.shape[-2] % 14 or rgb.shape[-1] % 14:
            raise ValueError('dav2_rgb must be official-preprocessed [B,3,Hv,Wv], with Hv/Wv divisible by 14')
        hv, wv = rgb.shape[-2:]
        # no_grad rather than inference_mode: adapter Conv backward must be
        # allowed to save the frozen activations. Keep the ViT in FP32 under AMP.
        with torch.no_grad(), torch.autocast(device_type=rgb.device.type, enabled=False):
            tokens = self.backbone.get_intermediate_layers(rgb.float(), [2, 5, 8, 11])

        # 把每层 ViT 的 patch 序列还原成二维网格，再投影到 32 通道。
        maps = [
            token.transpose(1, 2).reshape(rgb.shape[0], 768, hv // 14, wv // 14)
            for token in tokens
        ]
        projected = [proj(feat) for proj, feat in zip(self.projections, maps)]
        patch_feature = self.fuse(torch.cat(projected, dim=1))  # [B,32,Hv/14,Wv/14]
        self.last_shapes = dict(
            rgb=list(rgb.shape),
            tokens=[list(t.shape) for t in tokens],
            patch_maps=[list(m.shape) for m in maps],
            projected=[list(p.shape) for p in projected],
            patch_fused=list(patch_feature.shape),
        )
        return patch_feature

    def align_features(self, patch_feature, left_grid, image_size, cnn_size, left_valid, vit_size):
        # 用同一份鱼眼Patch特征分别对齐Yin/Yang；每张网格各自做原ResidualRefine。
        hv, wv = vit_size
        grid = vit_sampling_grid(left_grid, image_size, cnn_size, (hv, wv))
        grid = grid[None].expand(patch_feature.shape[0], -1, -1, -1)
        # Border extension only interpolates the first/last patch centre. It
        # never turns an invalid original-image/FOV position into valid evidence.
        aligned = F.grid_sample(
            patch_feature, grid.to(patch_feature.dtype),
            align_corners=False, mode='bilinear', padding_mode='border',
        )
        aligned = torch.where(left_valid, aligned, 0.)
        geometry = torch.where(left_valid, self.refine(aligned), 0.)
        self.last_shapes['geometry_context'] = list(geometry.shape)
        return geometry

    def forward(self, rgb, left_grid, image_size, cnn_size, left_valid, patch_feature=None):
        # 保留原单网格接口及计算顺序；双网格直接复用extract/align避免重复ViT。
        if patch_feature is None:
            patch_feature = self.extract_features(rgb)
        return self.align_features(patch_feature, left_grid, image_size, cnn_size, left_valid, rgb.shape[-2:])


class GeometryContextFusion(nn.Module):
    """F_CNN + alpha * phi([F_CNN,F_DA]), before both original 2D heads."""
    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1),
        )
        # Small, NONZERO output: start close to the CNN context while allowing
        # adapter gradients on the very first update (a zero head blocks them).
        nn.init.normal_(self.residual[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual[-1].bias)
        # 两者都非零：初始扰动很小，但不会阻断 Adapter 第一轮的梯度。
        self.alpha = nn.Parameter(torch.tensor(.1))

    def forward(self, cnn_reference, geometry, left_valid):
        enhanced = cnn_reference + self.alpha * self.residual(torch.cat([cnn_reference, geometry], dim=1))
        return torch.where(left_valid, enhanced, 0.)
