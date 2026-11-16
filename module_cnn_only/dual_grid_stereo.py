"""独立Yin–Yang双目入口：原CNN/扫描/增强/GWC/Warp/Fusion → 共享聚合与GRU。

不修改正式OmniDS.forward、Dataset、标定或旧训练入口。
采样质量仍由调用者显式提供连续Q；不将二值Mask冒充质量。
"""
import torch
import torch.nn as nn

from experiments.local_spherical_grid.adaptive_local_grid_v1 import sweep_patch, sample_reference
from module_cnn_only.featurelayer import FeatureLayers
from module_cnn_only.geometry_refiner import GeometryAwareFeatureEnhancer
from module_cnn_only.similarity_context import groupwise_correlation_volume
from module_cnn_only.cross_grid_warp import BidirectionalCrossGridWarp
from module_cnn_only.geometry_aware_fusion import GeometryAwareFeatureFusion
from module_cnn_only.depth_refinement import StereoDepthRefinement


class DualFisheyeStereoNetwork(nn.Module):
    """一个固定相机Rig的几何包 + 可配置的共享Stereo网络。

    imgs=[left,right]，每张[B,1或3,Hfish,Wfish]；原图不缩放。
    两quality分别位于目标Yin/Yang网格：[B,1,D,H目标,W目标]。
    输出dict['yin'/'yang']保留各自后半网络结果，没有跨网格Resize或隐式深度融合。
    .to(device)即可，几何坐标保持FP32；使用外层torch.autocast启用AMP。
    """
    def __init__(self, grid_package, use_rgb=False, depth_chunk=8, **refinement_options):
        super().__init__()
        self.grid_package = grid_package
        axis = torch.tensor(grid_package.metadata['inverse_depth_candidates'], dtype=torch.float32)
        if len(axis) < 2 or not torch.isfinite(axis).all() or not (axis > 0).all() or not (axis.diff() > 0).all():
            raise ValueError('Use the actual positive increasing radial inverse-depth candidate axis')
        # 不重建EPS→rho_max轴：原192[::2]的96候选以几何包实际数值为准。
        self.register_buffer('inverse_depth_candidates', axis)
        num_depth = refinement_options.pop('num_depth', len(axis))
        if num_depth != len(axis):
            raise ValueError('Network and grid package must use exactly the same candidate axis')
        self.groups = refinement_options.get('gwc_groups', 8)
        self.channels = refinement_options.get('context_channels', 32)
        self.depth_chunk = depth_chunk
        self.encoder = FeatureLayers(self.channels, use_rgb=use_rgb)
        self.enhancer = GeometryAwareFeatureEnhancer(self.channels, depth_chunk=depth_chunk)
        self.warp = BidirectionalCrossGridWarp(grid_package, depth_chunk)
        self.fusion = GeometryAwareFeatureFusion(self.groups)
        self.refinement = StereoDepthRefinement(num_depth=num_depth, **refinement_options)

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

    def forward(self, imgs, yang_to_yin_quality, yin_to_yang_quality, iters=None):
        if len(imgs) != 2:
            raise ValueError('imgs must be [left_fisheye, right_fisheye]')
        features = self.encoder(imgs)  # 原共享CNN，每Forward一次。
        costs, masks, contexts, context_valid = {}, {}, {}, {}
        for name in ('yin', 'yang'):
            patch = getattr(self.grid_package, name)
            left, right, mask = sweep_patch(features, patch, self.depth_chunk)
            left, right = self.enhancer(left, right, patch.rays_ref, mask, patch.padding_valid)
            # 原GWC函数用[B,C,H,W,D]；仅在接口处明确转换，内部数学计算不变。
            # 原GWC只拆分C为groups/channels_per_group，可直接读取permute视图，
            # 无需为两个巨大的32通道扫描体各做一次contiguous复制。
            cost = groupwise_correlation_volume(left.permute(0, 1, 3, 4, 2),
                                                right.permute(0, 1, 3, 4, 2), self.groups)
            costs[name] = cost.permute(0, 1, 4, 2, 3).contiguous().masked_fill(~mask, 0.)
            masks[name] = mask
            # 必须从未joint-mask的左CNN单独采样，不能取增强后左体的某个D切片。
            contexts[name], context_valid[name] = sample_reference(features[0], patch)

        warped = self.warp(costs['yin'], costs['yang'], masks['yin'], masks['yang'])
        fused = self.fusion.forward_bidirectional(costs['yin'], costs['yang'], masks['yin'], masks['yang'],
                                                 warped, yang_to_yin_quality, yin_to_yang_quality)
        result = {}
        for name in ('yin', 'yang'):
            output = self.refinement(fused['fused_' + name], contexts[name], fused[name + '_valid'],
                                     context_valid[name], iters=iters)
            output['inverse_depth'], output['radial_depth'] = self.index_to_depth(output['final_depth_idx'], output['valid'])
            output['fusion_weights'] = fused[name + '_weights']
            result[name] = output
        return result
