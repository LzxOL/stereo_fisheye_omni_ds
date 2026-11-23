"""Frozen official DA-V2 ViT-B + pixel-aligned adapter with Scheme I & Scheme II.

Only intermediate ViT tokens are used (blocks [2, 5, 8, 11]); no DPT depth head
or online scale fitting is added.
- Scheme I (yinyang_rot3): Unwarp left fisheye RGB onto low-distortion Yin and
  90-deg CW gravity-aligned Yang charts before running frozen ViT-B, then
  rotate Yang tokens back to native chart coordinates.
- Scheme II (EntropyGatedGeometryContextFusion): Compute GEV matching entropy
  H_gev from the fused stereo cost volume and gate DA-V2 residual injection so
  confident single-peak stereo pixels stay clean while ambiguous / textureless
  regions receive full monocular semantic context.
"""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

DAV2_INPUT_MODES = ('raw_fisheye', 'yinyang_rot3')


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
    resize = Resize(input_size, input_size, resize_target=False, keep_aspect_ratio=True,
                    ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC)
    normalize = NormalizeImage(mean=[.485, .456, .406], std=[.229, .224, .225])
    prepare = PrepareForNet()
    # Official infer_image converts OpenCV BGR to RGB; dataset raw_imgs are
    # already RGB, so applying cvtColor here would incorrectly swap R and B.
    images = [prepare(normalize(resize({'image': image / 255.})))['image'] for image in rgb]
    return torch.from_numpy(np.stack(images))


def prepare_dav2_raw_tensor(rgb):
    """Original RGB uint8 [B,H,W,3] -> GPU-ready float32 [B,3,H,W] in [0, 1].

    Used by Scheme I (`yinyang_rot3`) so chart unwarping, valid-masking,
    ImageNet normalization and 90-deg Yang gravity alignment happen directly on
    GPU without redundant CPU OpenCV fisheye resizing.
    """
    if torch.is_tensor(rgb):
        if rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError('DA-V2 raw tensor requires uint8 [B,H,W,3]')
        return rgb.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8 or arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError('DA-V2 raw tensor requires uint8 [B,H,W,3]')
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2))).float().div_(255.0)


def chart_vit_size(h_c, w_c, short_side):
    """Aspect-preserving (Hv, Wv) divisible by 14 with min side ~= short_side."""
    scale = float(short_side) / float(min(h_c, w_c))
    nh = max(14, int(round(h_c * scale / 14.0)) * 14)
    nw = max(14, int(round(w_c * scale / 14.0)) * 14)
    return nh, nw


def unwarp_chart_rgb(raw_rgb_01, left_grid, left_valid, image_size, cnn_size, target_hw):
    """Unwarp raw left fisheye RGB [B,3,H,W] in [0,1] onto a Yin/Yang chart.

    Uses valid-normalized bilinear upsampling of `left_grid` so out-of-FoV
    sentinel coordinates (2.0) never bleed across the fisheye circle boundary.
    """
    h_img, w_img = image_size
    hc, wc = cnn_size
    grid = left_grid.float()
    px_x = (grid[..., 0] + 1.0) * (wc - 1)
    px_y = (grid[..., 1] + 1.0) * (hc - 1)
    gx = ((px_x + 0.5) / float(w_img)) * 2.0 - 1.0
    gy = ((px_y + 0.5) / float(h_img)) * 2.0 - 1.0

    v = left_valid[0, 0].float()  # [H_c, W_c]
    packed = torch.stack([gx * v, gy * v, v], dim=0)[None]  # [1, 3, H_c, W_c]
    up = F.interpolate(packed, size=target_hw, mode='bilinear', align_corners=False)
    v_up = up[:, 2:3]
    safe_v = v_up.clamp_min(1e-6)
    gx_up = (up[:, 0] / safe_v[:, 0]).clamp(-1.0, 1.0)
    gy_up = (up[:, 1] / safe_v[:, 0]).clamp(-1.0, 1.0)
    grid_up = torch.stack([gx_up, gy_up], dim=-1).expand(raw_rgb_01.shape[0], -1, -1, -1)
    valid_up = v_up > 0.5

    sampled = F.grid_sample(raw_rgb_01.float(), grid_up, mode='bilinear',
                            padding_mode='zeros', align_corners=False)
    sampled = sampled.masked_fill(~valid_up, 0.0)
    return sampled, valid_up


def compute_gev_uncertainty(gev_volume, candidate_valid):
    """Compute detached 3-channel GEV matching uncertainty u_gev in FP32.

    Channels:
      0: H_1  - normalized Shannon entropy of softmax(gev_vol, dim=D) in [0, 1]
      1: H_z  - intra-branch standardized relative entropy clamped to [-3, 3]
                (invariant to logit scale across epochs; Q1 ~ -1, Q4 ~ +1)
      2: H_4  - sharpened-temperature entropy softmax(4 * gev_vol) in [0, 1]
    """
    with torch.no_grad(), torch.autocast(device_type=gev_volume.device.type, enabled=False):
        vol = gev_volume[:, 0].float() if gev_volume.ndim == 5 else gev_volume.float()  # [B, D, H, W]
        vm = candidate_valid[:, 0].bool() if candidate_valid.ndim == 5 else candidate_valid.bool()
        pix_valid = vm.any(dim=1, keepdim=True)  # [B, 1, H, W]
        log_cnt = vm.float().sum(dim=1, keepdim=True).clamp_min(2.0).log()

        def _norm_entropy(temp):
            logits = (vol * temp).masked_fill(~vm, -1e9)
            probs = F.softmax(logits, dim=1).masked_fill(~vm, 0.0)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
            ent = -(probs * probs.clamp_min(1e-9).log()).sum(dim=1, keepdim=True) / log_cnt
            return ent.clamp(0.0, 1.0).masked_fill(~pix_valid, 1.0)

        h1 = _norm_entropy(1.0)
        h4 = _norm_entropy(4.0)

        # Standardize H_1 across valid pixels per sample so gate contrast stays
        # active even when raw H_1 clusters in [0.85, 0.96] at temp=1.
        pv_f = pix_valid.float()
        denom = pv_f.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mean = (h1 * pv_f).sum(dim=(2, 3), keepdim=True) / denom
        var = (((h1 - mean) ** 2) * pv_f).sum(dim=(2, 3), keepdim=True) / denom
        std = torch.sqrt(var + 1e-8)
        hz = ((h1 - mean) / std.clamp_min(1e-3)).clamp(-3.0, 3.0).masked_fill(~pix_valid, 0.0)
        return torch.cat([h1, hz, h4], dim=1)


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
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.GroupNorm(8, 32))

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
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1))
        self.refine = ResidualRefine()
        self.register_buffer('_imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('_imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.last_shapes = {}

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_features(self, rgb):  # ViT与Adapter每次前向只计算一次，输出Patch特征。
        if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.shape[-2] % 14 or rgb.shape[-1] % 14:
            raise ValueError('dav2_rgb must be official-preprocessed [B,3,Hv,Wv], with Hv/Wv divisible by 14')
        hv, wv = rgb.shape[-2:]
        # no_grad rather than inference_mode: adapter Conv backward must be
        # allowed to save the frozen activations. Keep the ViT in FP32 under AMP.
        with torch.no_grad(), torch.autocast(device_type=rgb.device.type, enabled=False):
            tokens = self.backbone.get_intermediate_layers(rgb.float(), [2, 5, 8, 11])
        maps = [token.transpose(1, 2).reshape(rgb.shape[0], 768, hv // 14, wv // 14) for token in tokens]
        projected = [proj(feat) for proj, feat in zip(self.projections, maps)]
        patch_feature = self.fuse(torch.cat(projected, dim=1))  # [B,32,Hv/14,Wv/14]
        self.last_shapes = dict(rgb=list(rgb.shape), tokens=[list(t.shape) for t in tokens],
                                patch_maps=[list(m.shape) for m in maps],
                                projected=[list(p.shape) for p in projected], patch_fused=list(patch_feature.shape))
        return patch_feature

    def align_features(self, patch_feature, left_grid, image_size, cnn_size, left_valid, vit_size):
        # 用同一份鱼眼Patch特征分别对齐Yin/Yang；每张网格各自做原ResidualRefine。
        hv, wv = vit_size
        grid = vit_sampling_grid(left_grid, image_size, cnn_size, (hv, wv))
        grid = grid[None].expand(patch_feature.shape[0], -1, -1, -1)
        # Border extension only interpolates the first/last patch centre. It
        # never turns an invalid original-image/FOV position into valid evidence.
        aligned = F.grid_sample(patch_feature, grid.to(patch_feature.dtype), align_corners=False,
                                mode='bilinear', padding_mode='border')
        aligned = torch.where(left_valid, aligned, 0.)
        geometry = torch.where(left_valid, self.refine(aligned), 0.)
        self.last_shapes['geometry_context'] = list(geometry.shape)
        return geometry

    def extract_chart_geometry(self, raw_rgb_01, left_grid, image_size, cnn_size, left_valid,
                               chart_input_size=336, rot90_k=0):
        """Scheme I: Unwarp raw left RGB to a Yin/Yang chart, optionally rotate
        for gravity alignment (Yang rot90_k=3), run frozen ViT-B + shared adapter,
        rotate back, and align to native chart resolution (H_c, W_c).
        """
        if raw_rgb_01.ndim != 4 or raw_rgb_01.shape[1] != 3:
            raise ValueError('Scheme I expects raw left RGB tensor [B,3,H,W] in [0,1]')
        h_c, w_c = left_grid.shape[:2]
        rot_k = int(rot90_k) % 4
        if rot_k in (1, 3):
            nh_rot, nw_rot = chart_vit_size(w_c, h_c, chart_input_size)
            sample_hw = (nw_rot, nh_rot)
        else:
            sample_hw = chart_vit_size(h_c, w_c, chart_input_size)

        chart_rgb, _ = unwarp_chart_rgb(raw_rgb_01, left_grid, left_valid, image_size, cnn_size, sample_hw)
        chart_norm = (chart_rgb - self._imagenet_mean) / self._imagenet_std
        if rot_k != 0:
            chart_norm = torch.rot90(chart_norm, k=rot_k, dims=(-2, -1)).contiguous()

        patch_feature = self.extract_features(chart_norm)
        if rot_k != 0:
            patch_feature = torch.rot90(patch_feature, k=(4 - rot_k) % 4, dims=(-2, -1)).contiguous()

        aligned = F.interpolate(patch_feature, size=(h_c, w_c), mode='bilinear', align_corners=False)
        aligned = torch.where(left_valid, aligned, 0.)
        geometry = torch.where(left_valid, self.refine(aligned), 0.)
        self.last_shapes['geometry_context'] = list(geometry.shape)
        self.last_shapes['rot90_k'] = rot_k
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
            nn.Conv2d(64, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1))
        # Small, NONZERO output: start close to the CNN context while allowing
        # adapter gradients on the very first update (a zero head blocks them).
        nn.init.normal_(self.residual[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual[-1].bias)
        # 两者都非零：初始扰动很小，但不会阻断 Adapter 第一轮的梯度。
        self.alpha = nn.Parameter(torch.tensor(.1))

    def forward(self, cnn_reference, geometry, left_valid):
        enhanced = cnn_reference + self.alpha * self.residual(torch.cat([cnn_reference, geometry], dim=1))
        return torch.where(left_valid, enhanced, 0.)


class EntropyGatedGeometryContextFusion(nn.Module):
    """Scheme II: GEV matching-entropy gated DA-V2 context fusion.

    F_enh = F_CNN + alpha * gate(u_gev) * phi([F_CNN, F_DA, u_gev])
    Suppresses coarse monocular priors where stereo GEV matching is sharp and
    single-peak (low entropy), and opens the gate on ambiguous / textureless /
    occluded pixels (high entropy).
    """
    def __init__(self):
        super().__init__()
        self.gate_linear = nn.Conv2d(3, 1, 1)
        with torch.no_grad():
            # Monotonic cold-start on standardized entropy H_z:
            # low-entropy Q1 (H_z ~ -1.2) -> gate ~ 0.23; high-entropy Q4 (H_z ~ +1.2) -> gate ~ 0.77.
            self.gate_linear.weight.copy_(torch.tensor([[[[0.0]], [[1.0]], [[0.0]]]]))
            self.gate_linear.bias.zero_()
        self.gate_spatial = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )
        nn.init.normal_(self.gate_spatial[-1].weight, std=1e-3)
        nn.init.zeros_(self.gate_spatial[-1].bias)

        self.residual = nn.Sequential(
            nn.Conv2d(67, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1))
        nn.init.normal_(self.residual[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.last_gate_stats = {}

    def forward(self, cnn_reference, geometry, left_valid, gev_uncertainty):
        u = gev_uncertainty.to(cnn_reference.dtype)
        gate_logits = self.gate_linear(u) + self.gate_spatial(u)
        gate = torch.sigmoid(gate_logits.float()).to(cnn_reference.dtype)
        gate = torch.where(left_valid, gate, 0.)
        fused_in = torch.cat([cnn_reference, geometry, u], dim=1)
        enhanced = cnn_reference + self.alpha * gate * self.residual(fused_in)
        with torch.no_grad():
            vm = left_valid[:, 0]
            if vm.any():
                g_val = gate[:, 0][vm].float()
                hz_val = gev_uncertainty[:, 1][vm].float()
                q25 = torch.quantile(hz_val, 0.25)
                q75 = torch.quantile(hz_val, 0.75)
                self.last_gate_stats = {
                    'gate_mean': round(float(g_val.mean()), 4),
                    'gate_q1_low_ent': round(float(g_val[hz_val <= q25].mean()), 4),
                    'gate_q4_high_ent': round(float(g_val[hz_val >= q75].mean()), 4),
                    'alpha': round(float(self.alpha.detach()), 4),
                }
        return torch.where(left_valid, enhanced, 0.)
