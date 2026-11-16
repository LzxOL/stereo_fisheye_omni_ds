# module_cnn_only/network.py
# 双目CNN/DA-V2主干 + 可选Yin-Yang匹配前端。
# 两种映射共用原Context/State投影、Similarity/GEV Lookup、UpdateBlock和凸上采样。
# Author: Adapted from ROmniStereo and BEVFormer

import torch
import torch.nn as nn
import torch.nn.functional as F
from module_cnn_only.featurelayer import FeatureLayers, Conv2D
# CNN-ONLY: import DINO attention lazily inside the unused DINO class.
from module_cnn_only.update import ConvGRU, DepthHead
from module_cnn_only.similarity_context import SimilarityContext, GEVModule
from module_cnn_only.sweep import spherical_sweep
from utils.common import *

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class DINOFeatureExtractor(nn.Module):
    """
    DINO feature extractor using DINOv3 via timm.
    
    Supported models:
    - DINOv3 (patch 16): 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'
    
    Reference: https://huggingface.co/timm/vit_small_plus_patch16_dinov3.lvd1689m
    """
    def __init__(self, model_name='dinov3_vits16', output_dim=32, freeze=True):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.freeze = freeze
        
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
        
        # Map model names to timm model names
        timm_model_map = {
            'dinov3_vits16': 'vit_small_plus_patch16_dinov3.lvd1689m',
            'dinov3_vitb16': 'vit_base_patch16_dinov3.lvd1689m',
            'dinov3_vitl16': 'vit_large_patch16_dinov3.lvd1689m',
        }
        
        if model_name not in timm_model_map:
            raise ValueError(f"Unknown model: {model_name}. Supported: {list(timm_model_map.keys())}")
        
        timm_name = timm_model_map[model_name]
        print(f"Loading DINOv3 via timm: {timm_name}")
        
        # Get DINO output dimension
        if 'vits' in model_name:
            self.dino_dim = 384
        elif 'vitb' in model_name:
            self.dino_dim = 768
        elif 'vitl' in model_name:
            self.dino_dim = 1024
        else:
            self.dino_dim = 384
        
        # Create model with features_only=True for direct feature map output
        self.dino = timm.create_model(
            timm_name,
            pretrained=True,
            features_only=True,  # Returns feature maps directly [B, C, H, W]
        )
        print(f"Loaded DINOv3 model: {timm_name}")
        
        # Projection layer to match CNN feature dimension
        self.proj = nn.Sequential(
            nn.Conv2d(self.dino_dim, output_dim * 2, 1),
            nn.BatchNorm2d(output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim * 2, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        
        if freeze:
            self._freeze_dino()
    
    def _freeze_dino(self):
        """Freeze DINO backbone weights."""
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()
    
    def train(self, mode=True):
        """Override train to keep DINO frozen."""
        super().train(mode)
        if self.freeze:
            self.dino.eval()
        return self
    
    def forward(self, x, target_size=None):
        """
        Extract DINO features.
        
        Args:
            x: [B, C, H, W] - input image (RGB)
            target_size: (H, W) - target feature map size
        Returns:
            feat: [B, output_dim, H_out, W_out]
        """
        B, C, H, W = x.shape
        # Get feature maps (features_only=True returns list of feature maps)
        if target_size is None:
            H_out, W_out = H // 2, W // 2
        else:
            H_out, W_out = target_size

        with torch.no_grad() if self.freeze else torch.enable_grad():
            features = self.dino(x)  # List of [B, C, H, W] feature maps
            feat = features[-1]  # Use last feature map
        
        # Project to target dimension
        feat = self.proj(feat)
        
        # Resize to target size if specified
        feat = F.interpolate(feat, size=(H_out, W_out), mode='bilinear', align_corners=False)
        
        return feat


class DINOERPExtractor(nn.Module):
    """
    Extract DINO features from fisheye cameras and project to ERP space using cross-attention.
    Same structure as CNN feature extraction for consistency.
    Includes distortion-aware positional embedding for better texture handling at image edges.
    
    Usage:
        1. extract_features(imgs) - Run DINO backbone once
        2. forward(dino_feats, grids, inv_depth_idx) - Cross-attention (can be called multiple times)
    """
    def __init__(self, dino_model_name='dinov3_vits16', output_dim=32,
                 equi_h=80, equi_w=320, num_cams=4, freeze_dino=True,
                 num_heads=4, num_points=4, num_cross_attn_layers=1,
                 fish_h=384, fish_w=400, no_cross_attn=False):
        super().__init__()
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.num_cams = num_cams
        self.output_dim = output_dim
        self.fish_h = fish_h
        self.fish_w = fish_w

        # DINO feature extractor (shared for all cameras)
        self.dino_backbone = DINOFeatureExtractor(
            model_name=dino_model_name,
            output_dim=output_dim,
            freeze=freeze_dino,
        )

        # DINO feature extractor is used directly without LoRA adaptation.

        # Distortion-aware positional embedding
        # Encodes distance from image center and distortion angle
        # self.distortion_embed = nn.Sequential(
        #     nn.Linear(2, output_dim // 4),  # (normalized_distance, normalized_theta)
        #     nn.ReLU(inplace=True),
        #     nn.Linear(output_dim // 4, output_dim // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(output_dim // 2, output_dim)
        # )

        # Cross-attention to project DINO features to ERP space
        # When no_cross_attn=True, use fixed grid sampling instead (ablation baseline)
        if no_cross_attn:
            self.erp_cross_attn = None
        else:
            from module_cnn_only.erp_cross_attention import IterativeERPExtractor
            self.erp_cross_attn = IterativeERPExtractor(
                embed_dims=output_dim,
                equi_h=equi_h,
                equi_w=equi_w,
                num_cams=num_cams,
                num_heads=num_heads,
                num_points=num_points,
                num_layers=num_cross_attn_layers,
                use_geom_bias=True
            )

    def extract_features(self, imgs):
        """
        Extract DINO features from all cameras (run once).
        
        Args:
            imgs: list of 4 fisheye images [B, C, H, W]
        Returns:
            dino_feats_stacked: [B, num_cams, C, H, W]
        """
        dino_feats = []
        for img in imgs:
            # Ensure 3 channels for DINO
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            
            # Extract features (no target_size, keep original resolution)
            feat = self.dino_backbone(img, target_size=None)
            dino_feats.append(feat)
        
        # Stack features: [B, num_cams, C, H, W]
        dino_feats_stacked = torch.stack(dino_feats, dim=1)
        return dino_feats_stacked
    
    def _sample_dino_grid(self, dino_feats_stacked, grids_tensor, inv_depth_idx):
        """Fixed grid sampling fallback for DINO→ERP projection (ablation baseline).

        Uses depth-index interpolation of the per-camera LUT and averages the
        sampled features. This legacy DINO helper is not used by CNN-only.

        Args:
            dino_feats_stacked: [B, num_cams, C, H_dino, W_dino]
            grids_tensor: list of num_cams tensors, each [H, W, D, 2]
            inv_depth_idx: [B, 1, H, W]
        Returns:
            dino_erp: [B, C, H, W]
        """
        B, num_cams, C, _, _ = dino_feats_stacked.shape
        H, W = inv_depth_idx.shape[2], inv_depth_idx.shape[3]
        device = inv_depth_idx.device
        dtype = dino_feats_stacked.dtype
        idx_flat = inv_depth_idx.view(B, -1)  # [B, H*W]

        per_cam = []
        for cam_idx, grid_lut in enumerate(grids_tensor):
            grid_lut = grid_lut.to(device=device, dtype=dtype)  # [H, W, D, 2]
            D = grid_lut.shape[2]

            idx_floor = idx_flat.long().clamp(0, D - 1)
            idx_ceil = (idx_floor + 1).clamp(0, D - 1)
            w = (idx_flat - idx_floor.float()).unsqueeze(-1)  # [B, H*W, 1]

            grid_flat = grid_lut.view(-1, D, 2).unsqueeze(0).expand(B, -1, -1, -1)
            idx_floor_exp = idx_floor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
            idx_ceil_exp = idx_ceil.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)

            g0 = torch.gather(grid_flat, 2, idx_floor_exp).squeeze(2)
            g1 = torch.gather(grid_flat, 2, idx_ceil_exp).squeeze(2)
            grid = ((1 - w) * g0 + w * g1).view(B, H, W, 2)

            feat = F.grid_sample(
                dino_feats_stacked[:, cam_idx], grid,
                mode='bilinear', padding_mode='zeros', align_corners=True,
            )
            per_cam.append(feat)

        return torch.stack(per_cam, dim=0).mean(dim=0)  # [B, C, H, W]

    def forward(self, dino_feats_stacked, reference_points, inv_depth_idx,
                grids_tensor=None):
        """
        Project DINO features to ERP using cross-attention (can be called every iteration).
        When ``self.erp_cross_attn is None`` (ablation), falls back to fixed grid sampling.

        Args:
            dino_feats_stacked: [B, num_cams, C, H, W] - pre-extracted DINO features
            reference_points: [B, H*W, num_cams, 2] - reference points from grids (normalized [0, 1])
            inv_depth_idx: [B, 1, H, W] - current depth estimate
            grids_tensor: (optional) list of num_cams LUT tensors [H, W, D, 2], required when
                          using grid sampling mode (no_cross_attn)
        Returns:
            dino_erp: [B, output_dim, H, W]
        """
        if self.erp_cross_attn is None:
            return self._sample_dino_grid(dino_feats_stacked, grids_tensor, inv_depth_idx)

        # Cross-attention to ERP space with distortion embedding added to value
        dino_erp = self.erp_cross_attn(
            dino_feats_stacked, reference_points, inv_depth_idx
        )
        return dino_erp


# --------------------------------------------------------------------
# Fuse DINO(context) + CNN(detail) in ERP space
# Multiple fusion strategies available for speed vs quality tradeoff
# --------------------------------------------------------------------

class ERPContextFusionPointwise(nn.Module):
    """
    Fast pointwise fusion: O(N) complexity.
    Same spatial location fused via 1x1 conv - semantically correct for ERP alignment.
    ~400x faster than full attention.
    """
    def __init__(self, embed_dims: int, hidden_dims: int = 64):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dims * 2, hidden_dims, 1),
            nn.GroupNorm(8, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, embed_dims, 1),
        )

    def forward(self, dino_erp: torch.Tensor, cnn_erp: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] concat -> [B, 2C, H, W] -> [B, C, H, W]
        combined = torch.cat([dino_erp, cnn_erp], dim=1)
        return dino_erp + self.fuse(combined)

class ERPContextFusionGated(nn.Module):
    """
    Gated fusion: Adaptive per-pixel weighting between DINO and CNN.
    O(N) complexity, ~400x faster than full attention.
    """
    def __init__(self, embed_dims: int, hidden_dims: int = 32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dims * 2, hidden_dims, 3, padding=1),
            nn.GroupNorm(8, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, 1, 1),
            nn.Sigmoid(),
        )
        # Feature transform before gating
        self.dino_proj = nn.Conv2d(embed_dims, embed_dims, 1)
        self.cnn_proj = nn.Conv2d(embed_dims, embed_dims, 1)

    def forward(self, dino_erp: torch.Tensor, cnn_erp: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([dino_erp, cnn_erp], dim=1))  # [B, 1, H, W]
        dino_feat = self.dino_proj(dino_erp)
        cnn_feat = self.cnn_proj(cnn_erp)
        return gate * dino_feat + (1 - gate) * cnn_feat

class ERPContextFusionLocalConv(nn.Module):
    """
    Local conv fusion: Pointwise + local spatial context.
    O(N) complexity with 3x3 receptive field.
    """
    def __init__(self, embed_dims: int, hidden_dims: int = 64):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dims * 2, hidden_dims, 1),
            nn.GroupNorm(8, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, hidden_dims, 3, padding=1, groups=hidden_dims),  # depthwise
            nn.Conv2d(hidden_dims, embed_dims, 1),  # pointwise
        )

    def forward(self, dino_erp: torch.Tensor, cnn_erp: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([dino_erp, cnn_erp], dim=1)
        return dino_erp + self.fuse(combined)

class ERPContextFusionAttention(nn.Module):
    """
    Original full attention fusion: O(N²) complexity.
    Most powerful but SLOW - use only if quality is critical.
    """
    def __init__(self, embed_dims: int, num_heads: int = 4, ffn_dims: int = 128, dropout: float = 0.0):
        super().__init__()
        import inspect
        mha_kwargs = dict(embed_dim=embed_dims, num_heads=num_heads, dropout=dropout)
        if 'batch_first' in inspect.signature(nn.MultiheadAttention).parameters:
            mha_kwargs['batch_first'] = True
            self._batch_first = True
        else:
            self._batch_first = False
        self.attn = nn.MultiheadAttention(**mha_kwargs)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dims, embed_dims),
        )
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, dino_erp: torch.Tensor, cnn_erp: torch.Tensor) -> torch.Tensor:
        B, C, H, W = dino_erp.shape
        if self._batch_first:
            q = dino_erp.flatten(2).permute(0, 2, 1)
            k = cnn_erp.flatten(2).permute(0, 2, 1)
            v = k
            attn_out, _ = self.attn(q, k, v, need_weights=False)
            x = self.norm1(q + attn_out)
            x = self.norm2(x + self.ffn(x))
            return x.permute(0, 2, 1).view(B, C, H, W)

        q = dino_erp.flatten(2).permute(2, 0, 1)
        k = cnn_erp.flatten(2).permute(2, 0, 1)
        v = k
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))
        x = x.permute(1, 2, 0).contiguous().view(B, C, H, W)
        return x


class MotionEncoder(nn.Module):
    """
    Motion encoder with separate similarity, DINO and GEV inputs.
    sim(36) → 64, dino(32) → 64, gev(18) → 64, depth(1) → 32
    concat: 224 → 127 + depth(1) = 128
    """
    def __init__(self, sim_context_dims, dino_dims, gev_dims, hidden_channels=128):
        super().__init__()
        # Similarity context encoder (matching signal from corr branch)
        self.conv_sim1 = nn.Conv2d(sim_context_dims, 64, 1)
        self.conv_sim2 = nn.Conv2d(64, 64, 3, padding=1)

        # DINO feature encoder (scene context)
        self.conv_dino1 = nn.Conv2d(dino_dims, 64, 3, padding=1)
        self.conv_dino2 = nn.Conv2d(64, 64, 3, padding=1)

        # GEV feature encoder (geometry encoding)
        self.conv_gev1 = nn.Conv2d(gev_dims, 64, 1)
        self.conv_gev2 = nn.Conv2d(64, 64, 3, padding=1)

        # Depth encoder
        self.conv_depth1 = nn.Conv2d(1, 32, 7, padding=3)
        self.conv_depth2 = nn.Conv2d(32, 32, 3, padding=1)

        # Combine: sim(64) + dino(64) + gev(64) + depth(32) = 224 → hidden-1
        self.conv_out = nn.Conv2d(64 + 64 + 64 + 32, hidden_channels - 1, 3, padding=1)

    def forward(self, sim_context, dino_feat, gev_feat, depth):  # 共用原主干：把Similarity、增强Context、GEV和当前索引编码为128通道运动特征。
        sim = F.relu(self.conv_sim1(sim_context))
        sim = F.relu(self.conv_sim2(sim))

        dino = F.relu(self.conv_dino1(dino_feat))
        dino = F.relu(self.conv_dino2(dino))

        gev = F.relu(self.conv_gev1(gev_feat))
        gev = F.relu(self.conv_gev2(gev))

        dep = F.relu(self.conv_depth1(depth))
        dep = F.relu(self.conv_depth2(dep))

        out = F.relu(self.conv_out(torch.cat([sim, dino, gev, dep], dim=1)))
        return torch.cat([out, depth], dim=1)


class UpdateBlock(nn.Module):
    """
    Update block with separate similarity context, DINO context and GEV features.
    Similarity profile is looked up internally at each iteration.
    """
    def __init__(self, sim_context_dims, dino_dims, gev_dims, hidden_dim,
                 num_downsample=1, similarity_context=None):
        super().__init__()
        self.num_downsample = num_downsample
        self.similarity_context = similarity_context  # For lookup
        self.use_checkpoint = False

        # Motion encoder with separate sim, dino, gev inputs
        self.encoder = MotionEncoder(sim_context_dims, dino_dims, gev_dims,
                                           hidden_channels=128)
        encoder_output_dim = 128

        # GRU for hidden state update
        # Input: dino_feat (context) + motion_feat
        self.gru = ConvGRU(hidden_dim, encoder_output_dim + dino_dims)

        # Depth prediction head
        self.depth_head = DepthHead(hidden_dim, hidden_dim=128, output_dim=1)

        # Upsampling mask
        factor = 2 ** num_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, (factor ** 2) * 9, 1, padding=0)
        )

    def forward(self, net, similarity_profile, dino_feat, gev_feat, inv_depth,
                no_upsample=False):  # 执行一轮共用GRU更新，返回Hidden、索引增量和上采样权重。
        """
        Args:
            net: [B, hidden_dim, H, W] - hidden state
            similarity_profile: [B, Ds, H, W] - reused similarity profile
            dino_feat: [B, C_dino, H, W] - DINO context feature
            gev_feat: [B, gev_dims, H, W] - GEV lookup feature
            inv_depth: [B, 1, H, W] - current inverse depth estimate
        """
        # Lookup similarity context at current depth (corr branch)
        sim_context = self.similarity_context.lookup_context(similarity_profile, inv_depth)

        # 只重计算无状态的卷积/GRU，不重做Lookup，也不读取另一分支的体缓存。
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._update, net, sim_context, dino_feat, gev_feat, inv_depth,
                              no_upsample, use_reentrant=False)
        return self._update(net, sim_context, dino_feat, gev_feat, inv_depth, no_upsample)

    def _update(self, net, sim_context, dino_feat, gev_feat, inv_depth, no_upsample):
        """原MotionEncoder→ConvGRU→DepthHead/凸上采样权重；无BN及体缓存副作用。"""

        # Encode motion with all three signals
        motion_feat = self.encoder(sim_context, dino_feat, gev_feat, inv_depth)

        # Combine DINO context and motion features
        inp = torch.cat([dino_feat, motion_feat], dim=1)

        # GRU update
        net = self.gru(net, inp)

        # Predict depth update
        delta_depth = self.depth_head(net)

        if no_upsample:
            return net, delta_depth, None

        mask = 0.25 * self.mask(net)
        return net, delta_depth, mask


class OmniDS(nn.Module):
    """双目CNN入口：左参考Context仅初始化一次，所有GRU迭代共用。

    use_yinyang=True只增加双网格匹配前端；两种模式共用原DA-V2 Context和GRU。
    两条路径都只初始化一次静态Context，不按当前深度刷新。
    原始四目OmniDS使用module/network.py，不在此类中执行。
    """

    def __init__(self, varargin=None, grid_package=None):  # 构造原CNN/DA-V2及GRU主干，可追加Yin-Yang匹配前端。
        super().__init__()
        opts = Edict()
        opts.use_rgb = False
        opts.base_channel = 32
        opts.num_downsample = 1
        opts.equi_h = 320
        opts.equi_w = 480
        opts.num_invdepth = 96
        opts.phi_deg = 45.0
        opts.num_heads = 4
        opts.num_points = 4
        opts.num_cross_attn_layers = 1
        opts.mixed_precision = False
        opts.fix_bn = False
        # DINO options
        opts.dino_model = 'dinov3_vits16'  # DINOv3 (patch 16)
        opts.freeze_dino = True
        opts.use_dino_sim = False
        # Similarity context options
        opts.num_depth_samples = 96  # Sparse sampling for similarity profile
        opts.sim_radius = 4
        opts.sim_levels = 4
        opts.similarity_type = 'correlation'
        # Distillation: build the lightweight student encoder (see --distilled)
        opts.use_student = False
        opts.use_group_variance = False
        opts.num_cams = 2
        opts.use_yinyang = False  # 在DA-V2/原GRU主干前增加双网格匹配前端。
        opts.gru_iters = 12
        opts.yinyang_output = 'native'
        opts.gradient_checkpointing = False
        opts.use_dav2_geometry = False
        self.opts = argparse(opts, varargin)
        if self.opts.num_cams != 2:
            raise ValueError('CNN-only now uses stereo static context; use module.network.OmniDS for four cameras')
        if self.opts.num_cams == 2 and self.opts.num_invdepth != self.opts.num_depth_samples:
            raise ValueError('Stereo uses one complete candidate axis: num_invdepth == num_depth_samples')
        # CNN-ONLY: reject incompatible paths, including direct Python callers.
        if self.opts.use_dino_sim or self.opts.use_student:
            raise ValueError("CNN-only does not support use_dino_sim or the distilled student")
        self.opts.cnn_only = True
        # CNN feature extractor (for similarity/matching)
        self.encoder = FeatureLayers(
            self.opts.base_channel, 
            self.opts.use_rgb,
            getattr(self.opts, 'encoder_downsample_twice', False)
        )
        
        # Compute downsampled ERP size
        factor = 2 ** self.opts.num_downsample
        self.erp_h = self.opts.equi_h // factor
        self.erp_w = self.opts.equi_w // factor
        
        cnn_dims = self.opts.base_channel
        dino_dims = self.opts.base_channel  # CNN-ONLY: legacy name for context channels.
        hidden_dim = self.opts.base_channel * 2
        
        # Similarity context module (CNN features → matching signal)
        self.similarity_context = SimilarityContext(
            num_depth_samples=self.opts.num_depth_samples,
            num_invdepth=self.opts.num_invdepth,
            embed_dims=cnn_dims,
            radius=self.opts.sim_radius,
            num_levels=self.opts.sim_levels,
            similarity_type=self.opts.similarity_type,
            # Spatial aggregation options
            use_spatial_aggregation=getattr(self.opts, 'use_spatial_aggregation', True),
            spatial_agg_type=getattr(self.opts, 'spatial_agg_type', 'conv'),
            spatial_hidden_dim=getattr(self.opts, 'spatial_hidden_dim', 32),
            spatial_num_layers=getattr(self.opts, 'spatial_num_layers', 2),
            num_cams=self.opts.num_cams,
        )
        sim_context_dims = self.similarity_context.output_dim
        
        # CNN-ONLY: reuse original heads; context=32, initial hidden=64 (base_channel=32).
        self.context_proj = Conv2D(dino_dims, dino_dims, 1, pad=0, relu=False)
        self.state_proj = Conv2D(dino_dims, hidden_dim, 1, pad=0, relu=False)
        
        # Independent GEV branch: 3D UNet (RegularizationNet3D) with guided
        # excitation over GWC + multi-view variance, exposed as a depth pyramid.
        self._gev_radius = getattr(self.opts, 'gev_radius', 4)
        self.gev_module = GEVModule(
            in_channels=cnn_dims,
            num_groups=getattr(self.opts, 'gev_num_groups', 8),
            reg_channels=getattr(self.opts, 'gev_reg_channels', (16, 32, 48)),
            num_pyramid_levels=getattr(self.opts, 'gev_num_pyramid_levels', 2),
            use_spatial_downsample=getattr(self.opts, 'gev_use_spatial_downsample', True),
            use_group_variance=self.opts.use_group_variance,
        )

        # Update block; similarity_context is passed in for per-iteration lookup
        self.update_block = UpdateBlock(
            sim_context_dims=sim_context_dims,
            dino_dims=dino_dims,
            gev_dims=self.gev_module.lookup_dim(radius=self._gev_radius),
            hidden_dim=hidden_dim,
            num_downsample=self.opts.num_downsample,
            similarity_context=self.similarity_context,
        )
        self.update_block.use_checkpoint = self.opts.gradient_checkpointing

        # Student encoder for DINO distillation. Built only when requested:
        # registering it unconditionally would add feature_student.* to
        # state_dict and break strict loading of checkpoints trained without
        # it, and would force the FoundationStereo import chain on every run.
        if self.opts.use_student:
            from module_cnn_only.network_student import FeatureStudent_mobilenetv2_deconv
            self.feature_student = FeatureStudent_mobilenetv2_deconv()
        else:
            self.feature_student = None

        # Register new parameters ONLY when enabled, after all baseline layers:
        # old state_dict keys and seeded baseline initialization remain intact.
        if self.opts.use_dav2_geometry:
            if (self.opts.num_cams != 2 or self.opts.base_channel != 32 or
                    not (self.opts.use_yinyang or self.opts.get('reference_grid') == 'left_fisheye')):
                raise ValueError('DA-V2 geometry requires stereo, base_channel=32 and reference_grid=left_fisheye')
            if self.opts.get('dav2_fusion_target', 'context') != 'context':
                raise ValueError('This model uses DA-V2 Context fusion, not the previous GEV Guide fusion')
            self.opts.dav2_fusion_target = 'context'
            from module_cnn_only.dav2_geometry import DAV2GeometryContext, GeometryContextFusion
            self.dav2_geometry = DAV2GeometryContext(self.opts.dav2_model_repo, self.opts.dav2_checkpoint)
            self.geometry_fusion = GeometryContextFusion()

        if self.opts.use_yinyang:
            self._init_yinyang(grid_package)
        self.debug_outputs = {}

    def _init_yinyang(self, grid_package):  # 只增加几何匹配前端，Context/GEV/UpdateBlock仍用已训练主干。
        from module_cnn_only.geometry_refiner import GeometryAwareFeatureEnhancer
        from module_cnn_only.cross_grid_warp import BidirectionalCrossGridWarp
        from module_cnn_only.geometry_aware_fusion import GeometryAwareFeatureFusion

        self.grid_package = grid_package
        axis = torch.tensor(grid_package.metadata['inverse_depth_candidates'], dtype=torch.float32)
        if len(axis) != self.opts.num_invdepth or self.opts.use_group_variance:
            raise ValueError('Yin-Yang uses the same D-axis as the baseline and fused GWC without variance')
        self.register_buffer('inverse_depth_candidates', axis)
        self.depth_chunk = self.opts.get('depth_chunk', 8)
        self.enhancer = GeometryAwareFeatureEnhancer(self.opts.base_channel, depth_chunk=self.depth_chunk)
        self.warp = BidirectionalCrossGridWarp(grid_package, self.depth_chunk)
        self.fusion = GeometryAwareFeatureFusion(self.gev_module.num_groups)
        self.fusion.use_checkpoint = self.opts.gradient_checkpointing
        if self.opts.get('training_geometry', False):
            from module_cnn_only.yinyang_training import YinYangTrainingGeometry
            self.training_geometry = YinYangTrainingGeometry(grid_package, 2**self.opts.num_downsample)
        self.enhancer.local_refiner.use_checkpoint = self.opts.gradient_checkpointing

    def load_baseline_weights(self, state_dict):  # 显式继承DA-V2主干权重，新增双网格模块保留自身初始化。
        state = {key.removeprefix('module.'): value for key, value in state_dict.items()}
        current = self.state_dict()
        added = ('enhancer.', 'warp.', 'fusion.', 'inverse_depth_candidates')
        shared = {key for key in current if not key.startswith(added)}
        unused = {'view_weight_head.0.weight', 'view_weight_head.0.bias',
                  'view_weight_head.2.weight', 'view_weight_head.2.bias'}
        missing = shared - state.keys()
        unexpected = state.keys() - shared - unused
        if missing or unexpected:
            raise ValueError(f'Baseline architecture differs: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
        # 只允许明确列出的旧闲置权重；其他键和所有Tensor Shape均严格校验。
        current.update({key: state[key] for key in shared})
        self.load_state_dict(current, strict=True)
        return dict(inherited_keys=len(shared), initialized_keys=sorted(current.keys()-shared),
                    removed_unused_keys=sorted(state.keys() & unused))

    def index_to_depth(self, indices, valid):
        """连续候选索引→(径向逆深度1/m,径向距离m)，无效像素均返回0。

        直接在几何包候选轴上线性插值，兼容实际96层的终点而不误用原192轴。
        第0层使用包中的有限EPS，对应远场近似，不能视作0米深度。
        """
        idx = indices.float().clamp(0, len(self.inverse_depth_candidates) - 1)
        low = idx.floor().long()
        high = (low + 1).clamp(max=len(self.inverse_depth_candidates) - 1)
        axis = self.inverse_depth_candidates.float()
        rho = axis[low] + (idx - low.float()) * (axis[high] - axis[low])
        radial = rho.reciprocal()
        return rho.masked_fill(~valid, 0.), radial.masked_fill(~valid, 0.)

    def _build_yinyang_volumes(self, features, yang_to_yin_quality, yin_to_yang_quality):  # 双网格只准备匹配体和左Context，不在这里做最终深度预测。
        from experiments.local_spherical_grid.adaptive_local_grid_v1 import sweep_patch, sample_reference
        from module_cnn_only.similarity_context import groupwise_correlation_volume

        branches, costs, masks = {}, {}, {}
        for name in ('yin', 'yang'):
            patch = self.training_geometry.patch(name) if hasattr(self, 'training_geometry') else getattr(self.grid_package, name)
            left, right, mask = sweep_patch(features, patch, self.depth_chunk)
            left, right = self.enhancer(left, right, patch.rays_ref, mask, patch.padding_valid)
            # 原Similarity接口使用[B,C,H,W,D]；代价融合接口使用[B,G,D,H,W]。
            ref, target = [v.permute(0, 1, 3, 4, 2) for v in (left, right)]
            lg = torch.as_tensor(patch.left_grid, device=left.device, dtype=torch.float32)
            rg = torch.as_tensor(patch.right_grid, device=left.device, dtype=torch.float32).permute(1, 2, 0, 3)
            grids = [lg[:, :, None].expand(-1, -1, self.opts.num_invdepth, -1), rg]
            profile = self.similarity_context.build_profile(grids=grids, spherical_volumes=[ref, target], valid_mask=mask)
            costs[name] = groupwise_correlation_volume(ref, target, self.gev_module.num_groups).permute(0, 1, 4, 2, 3).contiguous()
            masks[name] = mask
            context, left_valid = sample_reference(features[0], patch)
            branches[name] = dict(profile=profile, profile_valid=mask, context=context, left_valid=left_valid,
                                  left_grid=lg, guide=left.mean(dim=2))

        if yang_to_yin_quality is None:
            batch = features[0].shape[0]
            yang_to_yin_quality = self.training_geometry.cross_quality('yin', batch, self.opts.num_invdepth)
            yin_to_yang_quality = self.training_geometry.cross_quality('yang', batch, self.opts.num_invdepth)
        warped = self.warp(costs['yin'], costs['yang'], masks['yin'], masks['yang'])
        fused = self.fusion.forward_bidirectional(costs['yin'], costs['yang'], masks['yin'], masks['yang'],
                                                 warped, yang_to_yin_quality, yin_to_yang_quality)
        for name, branch in branches.items():
            valid = fused[name+'_valid'] & branch['left_valid'].unsqueeze(2)
            self.gev_module.build_from_cost_volume(fused['fused_'+name], branch.pop('guide'), valid)
            # 两套缓存仅保存Tensor引用，共用同一GEV权重，不能让Yang覆盖Yin的查询体。
            branch.update(candidate_valid=valid, gev_pyramid=self.gev_module._gev_pyramid,
                          gev_valid=self.gev_module._valid_pyramid, fusion_weights=fused[name+'_weights'])
        return branches

    def _build_single_volume(self, features, grids, valid_mask, left_valid_mask):  # 原DA-V2双目前端，计算保持不变。
        # 球面扫描前，整理特征和采样坐标，并确定哪些候选点、哪些像素可以参与后续计算。
        batch = features[0].shape[0]
        grids = [torch.as_tensor(g, device=features[0].device, dtype=torch.float32) for g in grids]
        valid_mask = valid_mask.to(features[0].device)
        for grid in grids:
            visible = (torch.isfinite(grid).all(-1) & (grid.abs() <= 1).all(-1)).permute(2, 0, 1)
            valid_mask = valid_mask & visible[None, None]
        grids = [torch.where(torch.isfinite(g), g, 2.) for g in grids]
        left_grid = grids[0][:, :, 0]
        left_valid = (left_grid.abs() <= 1).all(-1)[None, None].expand(batch, -1, -1, -1)
        if left_valid_mask is not None:
            left_valid = left_valid & left_valid_mask.to(left_valid.device)
            valid_mask = valid_mask & left_valid.unsqueeze(2)
        volumes = spherical_sweep(features, grids)
        profile, ref, target, camera_volumes = self.similarity_context.build_profile(
            grids=grids, spherical_volumes=volumes, return_volumes=True, valid_mask=valid_mask,
            use_triton=self.opts.get('use_triton', False))
        self.gev_module.build(ref, target, camera_volumes, valid_mask=valid_mask,
                              use_triton=self.opts.get('use_triton', False))
        grid = left_grid[None].expand(batch, -1, -1, -1)
        context = F.grid_sample(features[0], grid.to(features[0].dtype), padding_mode='zeros', align_corners=True)
        context = context.masked_fill(~left_valid, 0.)
        return dict(left=dict(profile=profile, profile_valid=valid_mask, context=context, left_valid=left_valid,
                              left_grid=left_grid, candidate_valid=valid_mask,
                              gev_pyramid=self.gev_module._gev_pyramid, gev_valid=self.gev_module._valid_pyramid))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def upsample_depth(self, depth, mask, valid_pixels=None):  # 共用凸上采样：用3×3邻域恢复空间尺寸，不放大候选索引值。
        """Upsample depth using convex combination."""
        B, C, H, W = depth.shape
        factor = 2 ** self.opts.num_downsample
        mask = mask.view(B, 1, 9, factor, factor, H, W)
        
        if self.opts.num_cams == 2:
            # Stereo: indices remain in [0,D-1]; only the spatial grid is upsampled.
            up_depth = F.unfold(depth, [3, 3], padding=1)
            # Exclude missing neighbours and padded borders from the convex combination.
            # This also preserves a constant index field at the image boundary.
            support = torch.ones_like(depth) if valid_pixels is None else valid_pixels.to(depth.dtype)
            support = F.unfold(support, [3, 3], padding=1).view(B, 1, 9, 1, 1, H, W)
            mask = mask.masked_fill(support == 0, torch.finfo(mask.dtype).min)
            mask = torch.softmax(mask, dim=2) * support
        else:
            mask = torch.softmax(mask, dim=2)
            up_depth = F.unfold(factor * depth, [3, 3], padding=1)
        up_depth = up_depth.view(B, C, 9, 1, 1, H, W)
        
        up_depth = torch.sum(mask * up_depth, dim=2)
        up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)
        return up_depth.reshape(B, C, factor * H, factor * W)
    


    def forward(self, imgs, grids=None, iters=None, test_mode=False, ocams=None, distilled=False,
                valid_mask=None, left_valid_mask=None, dav2_rgb=None,
                yang_to_yin_quality=None, yin_to_yang_quality=None):  # 匹配前端之后统一执行DA-V2融合、原Context/State投影和原GRU。
        """双网格只改变前端映射和代价融合，后半部分沿用DA-V2 Baseline。

        dav2_rgb为真实左RGB的官方预处理结果。Yin/Yang返回各自原生网格的
        2倍空间预测；索引单位始终[0,D-1]。单网格训练/推理接口不变。
        """
        iters = self.opts.gru_iters if iters is None else iters
        device_type = imgs[0].device.type
        enabled = self.opts.mixed_precision or torch.is_autocast_enabled(device_type)
        with torch.autocast(device_type=device_type, enabled=enabled):
            features = self.encoder(imgs)
            if not self.opts.mixed_precision and not torch.is_autocast_enabled(device_type):
                features = [feature.float() for feature in features]
            # 这里仅准备前端数据；两种模式都继续执行下面同一套DA-V2/GRU主干。
            if self.opts.use_yinyang:
                branches = self._build_yinyang_volumes(features, yang_to_yin_quality, yin_to_yang_quality)
            else:
                branches = self._build_single_volume(features, grids, valid_mask, left_valid_mask)

            patch_feature = None
            if self.opts.use_dav2_geometry and self.opts.use_yinyang:
                patch_feature = self.dav2_geometry.extract_features(dav2_rgb)
            result = {}
            self.debug_outputs = {}
            for name, branch in branches.items():
                cnn_context = branch['context']
                left_valid = branch['left_valid']
                candidate_valid = branch['candidate_valid']
                pixels = candidate_valid.any(dim=2)
                if self.opts.use_dav2_geometry:
                    geometry = self.dav2_geometry(
                        dav2_rgb, branch['left_grid'], imgs[0].shape[-2:], features[0].shape[-2:],
                        left_valid, patch_feature=patch_feature)
                    cnn_context = self.geometry_fusion(cnn_context, geometry, left_valid)

                # 恢复已训练的两路投影：同一增强Context同时初始化Context和Hidden。
                context = torch.relu(self.context_proj(cnn_context)).masked_fill(~left_valid, 0.)
                net = torch.tanh(self.state_proj(cnn_context)).masked_fill(~left_valid, 0.)
                initial = torch.full((imgs[0].shape[0], 1, *context.shape[-2:]), self.opts.num_invdepth / 2,
                                     device=context.device, dtype=features[0].dtype)
                current = initial
                predictions, native_predictions = [], []
                self.gev_module._gev_pyramid = branch['gev_pyramid']
                self.gev_module._valid_pyramid = branch['gev_valid']
                self.similarity_context._candidate_valid = branch['profile_valid']
                for itr in range(iters):
                    current = current.detach()
                    gev_feature = self.gev_module.lookup(current, radius=self._gev_radius)
                    net, delta, up_mask = self.update_block(net, branch['profile'], context, gev_feature, current,
                                                            no_upsample=test_mode and itr < iters-1)
                    current = (current + delta).clamp(0, self.opts.num_invdepth-1).masked_fill(~pixels, 0.)
                    net = net.masked_fill(~pixels, 0.)
                    native_predictions.append(current)
                    if up_mask is not None:
                        prediction = self.upsample_depth(current, up_mask, pixels).clamp(0, self.opts.num_invdepth-1)
                        output_valid = F.interpolate(pixels.float(), scale_factor=2**self.opts.num_downsample, mode='nearest').bool()
                        predictions.append(prediction.masked_fill(~output_valid, 0.))
                row = dict(predictions=predictions, native_predictions=native_predictions,
                           init_depth_idx=initial.masked_fill(~pixels, 0.), final_depth_idx=predictions[-1],
                           valid=output_valid, native_valid=pixels, candidate_valid=candidate_valid,
                           gev_volume=branch['gev_pyramid'][0])
                if self.opts.use_yinyang:
                    row['inverse_depth'], row['radial_depth'] = self.index_to_depth(row['final_depth_idx'], output_valid)
                    row['fusion_weights'] = branch['fusion_weights']
                result[name] = row
                self.debug_outputs[name] = dict(context=list(context.shape), hidden=list(net.shape),
                                                gev=list(branch['gev_pyramid'][0].shape), prediction=list(predictions[-1].shape))

        self.gev_module.clear_cache()
        self.similarity_context._candidate_valid = None
        if self.opts.use_yinyang and self.opts.yinyang_output == 'left_fisheye':
            predictions, self.output_valid = self.training_geometry.project(result)
            return predictions[-1] if test_mode else predictions
        if self.opts.use_yinyang:
            return {name: row['final_depth_idx'] for name, row in result.items()} if test_mode else result
        return result['left']['final_depth_idx'] if test_mode else result['left']['predictions']
