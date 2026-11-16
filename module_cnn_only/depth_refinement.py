"""融合GWC之后的多通道3D聚合、连续候选索引Lookup和二维ConvGRU。

不构建独立GEV/Similarity分支；只从同一个聚合体查询局部匹配特征。
所有体统一[B,C,D,H,W]，二维状态为[B,C,H,W]，索引单位始终为候选层。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_cnn_only.featurelayer import Conv2D
from module_cnn_only.similarity_context import RegularizationNet3D
from module_cnn_only.update import ConvGRU, DepthHead


class CostAggregation3D(RegularizationNet3D):
    """复用原3D U-Net的全部编码/跳连/解码，仅将本实例输出改为多通道。

    [B,G,D,H,W] → [B,Cagg,D,H,W]。不做原GEV的空间池化，不使用Guide；
    两次stride=2要求D/H/W是4的倍数。原RegularizationNet3D类保持不变。
    """
    def __init__(self, in_channels=8, out_channels=8, channels=(16, 32, 48)):
        super().__init__(in_channels, channels)
        self.out_conv = nn.Conv3d(channels[0], out_channels, 1)


class DepthPyramid(nn.Module):
    """[B,C,D,H,W]和bool [B,1,D,H,W] → 两个逐层列表，仅沿D池化。

    全有效时等同avg_pool3d((2,1,1))；部分无效时只平均有效子候选，
    没有支持的粗候选输出0/False。体与掩码每次Forward只构建一次。
    """
    def __init__(self, num_levels=2):
        super().__init__()
        self.num_levels = num_levels

    def forward(self, volume, valid):
        volumes, masks = [volume.masked_fill(~valid, 0.)], [valid]
        for _ in range(1, self.num_levels):
            # 半精度下累加、除法保留FP32，不将缺失候选的零值作为观测参与平均。
            with torch.autocast(device_type=volume.device.type, enabled=False):
                support = F.avg_pool3d(masks[-1].float(), (2, 1, 1))
                pooled = F.avg_pool3d(volumes[-1].float(), (2, 1, 1))
                pooled = pooled / support.clamp_min(1e-6)
            masks.append(support > 0)
            volumes.append(pooled.to(volume.dtype).masked_fill(~masks[-1], 0.))
        return volumes, masks


class DepthWiseLookup(nn.Module):
    """每像素按连续idx查询各层的2r+1个邻居，输出[B,L*C*(2r+1),H,W]。

    顺序为level→channel→offset。坐标idx/2**level沿用OmniDS约定，
    先Clamp再floor/ceil/gather线性插值；不沿H/W采样或插值。
    任一正权重深度邻点无效则该查询返回0，不跨无效候选插值。
    本层不detach；迭代器在每轮入口统一detach状态。
    """
    def __init__(self, radius=4):
        super().__init__()
        self.radius = radius

    def forward(self, volumes, masks, current_depth_idx):
        batch, _, height, width = current_depth_idx.shape
        offsets = torch.arange(-self.radius, self.radius + 1,
                               device=current_depth_idx.device, dtype=torch.float32)
        outputs = []
        for level, (volume, valid) in enumerate(zip(volumes, masks)):
            channels, depth = volume.shape[1:3]
            indices = current_depth_idx.float() / (2 ** level) + offsets[None, :, None, None]
            indices = indices.clamp(0, depth - 1)
            low = indices.floor().long()
            high = (low + 1).clamp(max=depth - 1)
            weight = (indices - low.float()).unsqueeze(1)

            def gather(values, index):
                # gather仅沿D；展平的H*W仍与各像素自己的查询索引一一对应。
                index = index.flatten(2).unsqueeze(1).expand(-1, values.shape[1], -1, -1)
                sampled = values.flatten(3).gather(2, index)
                return sampled.reshape(batch, values.shape[1], -1, height, width)

            low_valid, high_valid = gather(valid, low), gather(valid, high)
            # 只清理查询到的邻点，避免每轮复制整个[B,C,D,H,W]聚合体。
            lo = gather(volume, low).masked_fill(~low_valid, 0.)
            hi = gather(volume, high).masked_fill(~high_valid, 0.)
            support = low_valid & (high_valid | (weight == 0))
            with torch.autocast(device_type=volume.device.type, enabled=False):
                sampled = lo.float() * (1 - weight) + hi.float() * weight
            sampled = sampled.to(volume.dtype).masked_fill(~support, 0.)
            outputs.append(sampled.reshape(batch, channels * len(offsets), height, width))
        return torch.cat(outputs, dim=1)


class MotionEncoder(nn.Module):
    """Lookup→64，CNN Context→64，归一化idx→32；拼接160→127再附加idx。

    输入[B,144,H,W]、[B,32,H,W]、[B,1,H,W]，默认输出[B,128,H,W]。
    归一化只用于网络编码，不改变Lookup、DepthHead残差和预测的候选层单位。
    """
    def __init__(self, lookup_channels=144, context_channels=32, output_channels=128):
        super().__init__()
        self.lookup = nn.Sequential(nn.Conv2d(lookup_channels, 64, 1), nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.context = nn.Sequential(nn.Conv2d(context_channels, 64, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.depth = nn.Sequential(nn.Conv2d(1, 32, 7, padding=3), nn.ReLU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.combine = nn.Sequential(nn.Conv2d(160, output_channels - 1, 3, padding=1), nn.ReLU())

    def forward(self, lookup, context, current_depth_idx, num_depth):
        normalized = current_depth_idx.float() / (num_depth - 1)
        encoded = torch.cat([self.lookup(lookup), self.context(context), self.depth(normalized)], dim=1)
        motion = self.combine(encoded)
        return torch.cat([motion, normalized.to(motion.dtype)], dim=1)


class DepthUpdateBlock(nn.Module):
    """复用原ConvGRU和DepthHead；只将MotionEncoder改成当前三路输入。

    net[B,64,H,W]、Context[B,32,H,W]、Lookup[B,144,H,W]、idx[B,1,H,W]
    → 新net与delta_idx[B,1,H,W]。不做上采样或索引尺度变换。
    """
    def __init__(self, lookup_channels=144, context_channels=32, hidden_dim=64, motion_dim=128):
        super().__init__()
        self.encoder = MotionEncoder(lookup_channels, context_channels, motion_dim)
        self.gru = ConvGRU(hidden_dim, context_channels + motion_dim)
        self.depth_head = DepthHead(hidden_dim)

    def forward(self, net, lookup, context, current_depth_idx, num_depth, valid_pixels):
        motion = self.encoder(lookup, context, current_depth_idx, num_depth)
        motion = motion.masked_fill(~valid_pixels, 0.)
        inp = torch.cat([context, motion], dim=1)
        net = self.gru(net, inp).masked_fill(~valid_pixels, 0.)
        delta = self.depth_head(net).masked_fill(~valid_pixels, 0.)
        return net, delta


class StereoDepthRefinement(nn.Module):
    """单网格后半部分，可用同一实例处理不同H/W的Yin、Yang。

    输入：fused_gwc[B,G,D,H,W]、已投影reference_context[B,32,H,W]、
          candidate_valid bool[B,1,D,H,W]、reference_valid bool[B,1,H,W]。
    返回dict含aggregate/logits/prob、init_depth_idx、逐轮predictions、final_depth_idx
    和有效Mask。输出仍在本网格上，不假装已经生成统一左鱼眼深度图。
    """
    def __init__(self, num_depth=96, gwc_groups=8, aggregated_channels=8,
                 num_pyramid_levels=2, lookup_radius=4, context_channels=32,
                 motion_feature_dim=128, gru_hidden_dim=64, gru_iters=3,
                 reg_channels=(16, 32, 48), use_refinement=True):
        super().__init__()
        if num_depth % 4 or num_depth < max(4, 2 ** (num_pyramid_levels - 1)) or gru_iters < 1:
            raise ValueError('D must fit the 3D U-Net/depth pyramid and gru_iters must be positive')
        self.num_depth = num_depth
        self.gru_iters = gru_iters
        self.use_refinement = use_refinement
        self.aggregation = CostAggregation3D(gwc_groups, aggregated_channels, reg_channels)
        # 相似度logit：越大概率越高。此头独立于保留Cagg通道的聚合体。
        self.classifier = nn.Conv3d(aggregated_channels, 1, 1)
        self.pyramid = DepthPyramid(num_pyramid_levels)
        self.lookup = DepthWiseLookup(lookup_radius)
        self.lookup_channels = aggregated_channels * num_pyramid_levels * (2 * lookup_radius + 1)
        self.context_proj = Conv2D(context_channels, context_channels, 1, pad=0, relu=False)
        self.state_proj = Conv2D(context_channels, gru_hidden_dim, 1, pad=0, relu=False)
        self.update_block = DepthUpdateBlock(self.lookup_channels, context_channels,
                                             gru_hidden_dim, motion_feature_dim)

    def forward(self, fused_gwc, reference_context, candidate_valid, reference_valid, iters=None):
        depth = fused_gwc.shape[2]
        if depth != self.num_depth:
            raise ValueError('Volume D must match the configured candidate axis')
        iterations = self.gru_iters if iters is None else iters
        valid = candidate_valid & reference_valid.unsqueeze(2)
        pixels = valid.any(dim=2)
        aggregated = self.aggregation(fused_gwc.masked_fill(~valid, 0.)).masked_fill(~valid, 0.)
        logits = self.classifier(aggregated)
        with torch.autocast(device_type=fused_gwc.device.type, enabled=False):
            # 全无效像素不能softmax(-inf,...,-inf)，用有限占位后将概率整列清零。
            safe_logits = logits.float().masked_fill(~valid, -torch.inf).masked_fill(~pixels.unsqueeze(2), 0.)
            probability = safe_logits.softmax(dim=2).masked_fill(~valid, 0.)
            axis = torch.arange(depth, device=logits.device, dtype=torch.float32)[None, None, :, None, None]
            initial = (probability * axis).sum(dim=2)

        # 参数初始化路径与原CNN版一致：同一左参考特征投影成Context与Hidden。
        reference = reference_context.masked_fill(~reference_valid, 0.)
        context = F.relu(self.context_proj(reference)).masked_fill(~pixels, 0.)
        net = torch.tanh(self.state_proj(reference)).masked_fill(~pixels, 0.)
        predictions = []
        if self.use_refinement:
            volumes, masks = self.pyramid(aggregated, valid)  # 每个网格每Forward只建一次。
            current = initial
            for _ in range(iterations):
                # 仅断开反复查询的idx路径；Hidden递归和体/Context的梯度仍然保留。
                current = current.detach()
                queried = self.lookup(volumes, masks, current)
                net, delta = self.update_block(net, queried, context, current, depth, pixels)
                current = (current + delta.float()).clamp(0, depth - 1).masked_fill(~pixels, 0.)
                predictions.append(current)
        else:
            predictions.append(initial)
        # init必须单独监督：迭代detach后，只有迭代损失不能训练classifier。
        return dict(aggregated_volume=aggregated, depth_logits=logits.masked_fill(~valid, 0.),
                    depth_prob=probability, init_depth_idx=initial, predictions=predictions,
                    final_depth_idx=predictions[-1], candidate_valid=valid, valid=pixels)
