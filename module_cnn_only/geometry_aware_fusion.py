"""独立的Yin/Yang自适应代价融合，不执行Warp、不调用后续Stereo网络。

两路代价均为[B,G,D,H,W]；每个候选位置预测一对权重，在G组之间共享。
Quality是独立的连续可信度Q∈[0,1]，越大越好，不是有效性Mask或原始κ。
当前Warp未生成Q，必须由调用者明确提供；二维几何质量可由调用者沿B/D广播。
"""
import math

import torch
import torch.nn as nn


class GeometryAwareFeatureFusion(nn.Module):
    """共享一个35→16→2的点式Conv3d网络，默认G=8，无空间/深度卷积。

    forward输入：self/cross_cost=[B,G,D,H,W]；两Mask为bool [B,1,D,H,W]；
                 sampling_quality为浮点[B,1,D,H,W]，位于目标网格。
    返回：(fused_cost, union_valid, weights)，分别为[B,G,D,H,W]、
           bool [B,1,D,H,W]、FP32 [B,2,D,H,W]，权重顺序self/cross。
    log(Q+eps)在基础logit不变时单调抑制低质量跨路；由于CNN也接收Q，
    训练后的总权重并不是对任意参数都严格单调的硬约束。
    """
    def __init__(self, groups=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(4 * groups + 3, 16, 1),
            nn.ReLU(),
            nn.Conv3d(16, 2, 1),
        )
        # 初始基础logit为[log(2),0]，Q=1时约2:1偏向自身；保持跨路可学习。
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():
            self.network[-1].bias.copy_(torch.tensor([math.log(2.), 0.]))

    def forward(self, self_cost, cross_cost, self_valid, cross_valid, sampling_quality):
        mask_shape = (self_cost.shape[0], 1, *self_cost.shape[2:])
        if self_cost.shape != cross_cost.shape:
            raise ValueError('Costs must already be aligned to the same target [B,G,D,H,W]')
        if any(m.shape != mask_shape or m.dtype != torch.bool for m in (self_valid, cross_valid)):
            raise ValueError('Validity masks must be bool [B,1,D,H,W]')
        if sampling_quality.shape != mask_shape:
            raise ValueError('Explicit continuous sampling_quality must be [B,1,D,H,W]')
        if not sampling_quality.is_floating_point():
            raise ValueError('Sampling quality is a floating confidence, not a binary mask')
        both = self_valid & cross_valid
        # 先清除无效值，再计算乘积、差异与加权和；NaN*0仍是NaN，不能用乘Mask替代。
        clean_self = self_cost.float().masked_fill(~self_valid, 0.)
        clean_cross = cross_cost.float().masked_fill(~cross_valid, 0.)
        # FP32构建一致性特征，避免AMP下先在半精度中进行乘积。
        with torch.autocast(device_type=self_cost.device.type, enabled=False):
            consistency = (clean_self * clean_cross).masked_fill(~both, 0.)
            difference = (clean_self - clean_cross).abs().masked_fill(~both, 0.)
        quality = sampling_quality.float().masked_fill(~cross_valid, 0.).clamp(0., 1.)
        features = torch.cat([clean_self, clean_cross, consistency, difference,
                              self_valid.float(), cross_valid.float(), quality], dim=1)
        logits = self.network(features)
        with torch.autocast(device_type=self_cost.device.type, enabled=False):
            # Q=0仍是低质量有效样本，不改变几何Mask；只有跨路有效时仍必须取跨路。
            cross_logit = logits[:, 1:2].float() + torch.log(quality + 1e-6)
            logits = torch.cat([logits[:, :1].float(), cross_logit], dim=1)
            available = torch.cat([self_valid, cross_valid], dim=1)
            valid = self_valid | cross_valid
            logits = logits.masked_fill(~available, -torch.inf)
            # 全无效位置先替换为有限logit，再把两权重清零，避免softmax(-inf,-inf)。
            logits = logits.masked_fill(~valid, 0.)
            weights = logits.softmax(dim=1).masked_fill(~available, 0.)
            fused = weights[:, :1] * clean_self + weights[:, 1:2] * clean_cross
        return fused.to(self_cost.dtype), valid, weights

    def forward_bidirectional(self, yin_cost, yang_cost, yin_valid, yang_valid,
                              warped, yang_to_yin_quality, yin_to_yang_quality):
        """直接适配BidirectionalCrossGridWarp返回dict，不重新Warp或Resize。

        两个quality都须在各自目标网格上，Shape分别[B,1,D,Hy,Wy]/[B,1,D,Ha,Wa]。
        两次调用共享self.network。最终dict保留两路代价、有效性及权重。
        """
        yin, ym, yw = self(yin_cost, warped['yang_to_yin'], yin_valid,
                           warped['yang_to_yin_valid'], yang_to_yin_quality)
        yang, am, aw = self(yang_cost, warped['yin_to_yang'], yang_valid,
                            warped['yin_to_yang_valid'], yin_to_yang_quality)
        return dict(fused_yin=yin, fused_yang=yang, yin_valid=ym, yang_valid=am,
                    yin_weights=yw, yang_weights=aw)
