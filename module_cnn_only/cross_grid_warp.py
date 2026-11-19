"""双向跨网格代价体采样：只变换H/W，保持B/G/D及候选索引不变。

现有GWC输出[B,G,H,W,D]；调用本模块前须显式
    cost = cost.permute(0, 1, 4, 2, 3).contiguous()
本模块不调用GWC，也不融合两路代价。每个固定几何包创建一个实例：
    warp = BidirectionalCrossGridWarp(package).to(device)
    result = warp(yin_cost, yang_cost, yin_stereo_valid, yang_stereo_valid)
"""
import torch
import torch.nn as nn

from experiments.local_spherical_grid.adaptive_local_grid_v1 import sample_patch


class _CachedQuery:
    """只适配sample_patch所需的两个接口，直接返回预计算坐标，不重复算射线。

    shape是源网格的Padding尺寸；coordinates/overlap定义在目标网格上。
    所有正权重插值邻点的有效性判断仍由原sample_patch完成。
    """
    def __init__(self, source_shape, coordinates, overlap):
        self.shape = source_shape
        self.coordinates = coordinates
        self.overlap = overlap

    def direction_to_grid(self, rays_ref):
        return self.coordinates, self.overlap


def _warp_volume(volume, source_valid, query, depth_chunk):
    """[B,G,D,Hs,Ws] + [B,1,D,Hs,Ws] → [B,G,D,Ht,Wt]及bool Mask。

    query已包含几何重叠和目标Padding；不接收目标stereo_valid。
    按D分块，折叠B和当前块的D后，复用原sample_patch做2D双线性采样。
    """
    batch, groups, depth, hs, ws = volume.shape
    ht, wt = query.overlap.shape
    volumes, masks = [], []
    for start in range(0, depth, depth_chunk):
        stop = min(start + depth_chunk, depth)
        count = stop - start
        values = volume[:, :, start:stop].permute(0, 2, 1, 3, 4)
        values = values.reshape(batch * count, groups, hs, ws)
        support = source_valid[:, :, start:stop].permute(0, 2, 1, 3, 4)
        support = support.reshape(batch * count, 1, hs, ws)

        # rays_ref=None：适配器直接提供已缓存的目标→源坐标和window mask。
        sampled, valid = sample_patch(values, query, None, support)
        volumes.append(sampled.reshape(batch, count, groups, ht, wt).permute(0, 2, 1, 3, 4))
        masks.append(valid.reshape(batch, count, 1, ht, wt).permute(0, 2, 1, 3, 4))
    return torch.cat(volumes, dim=2), torch.cat(masks, dim=2)


class BidirectionalCrossGridWarp(nn.Module):
    """同一候选轴下的Yin↔Yang Warping，无可训练参数。

    输入：yin/yang_cost分别[B,G,D,Hy,Wy]、[B,G,D,Ha,Wa]；
         对应stereo_valid分别[B,1,D,Hy,Wy]、[B,1,D,Ha,Wa]，bool。
    输出dict的方向命名为“源→目标”，每个Mask为[B,1,D,H目标,W目标]。
    坐标为align_corners=True、相对于源Padding尺寸的FP32归一化坐标；
    使用AMP时保留本模块buffer为FP32，不对几何坐标做half量化。
    """
    def __init__(self, package, depth_chunk=8):
        super().__init__()
        self.depth_chunk = depth_chunk
        self.num_depth = len(package.metadata['inverse_depth_candidates'])
        self.yin_shape, self.yang_shape = tuple(package.yin.shape), tuple(package.yang.shape)

        for target, source in [('yin', 'yang'), ('yang', 'yin')]:
            patch = getattr(package, target)
            key, mask_key = target + '_to_' + source, target + '_in_' + source
            if key in package.reference and mask_key in package.reference:
                coordinates, overlap = package.reference[key], package.reference[mask_key]
            else:
                # 旧包没有缓存时只在初始化计算一次，Forward不做几何计算。
                coordinates, overlap = package.correspondence(target, source)
            padding = torch.as_tensor(patch.padding_valid, dtype=torch.bool).clone()
            self.register_buffer(key, torch.as_tensor(coordinates, dtype=torch.float32).clone())
            self.register_buffer(mask_key, torch.as_tensor(overlap, dtype=torch.bool).clone() & padding)
            self.register_buffer(target + '_padding', padding)

    def forward(self, yin_cost, yang_cost, yin_stereo_valid, yang_stereo_valid):
        batch, groups, depth = yin_cost.shape[:3]
        if yang_cost.shape[:3] != (batch, groups, depth) or depth != self.num_depth:
            raise ValueError('Both costs must share B/G/D and the geometry package candidate axis')
        # 源Padding也明确剔除，避免调用方误把Padding的Mask设成True。
        yin_support = yin_stereo_valid & self.yin_padding[None, None, None]
        yang_support = yang_stereo_valid & self.yang_padding[None, None, None]

        # Yang→Yin：在Yin位置查询Yang，故使用缓存key yin_to_yang。
        query_yin = _CachedQuery(self.yang_shape, self.yin_to_yang, self.yin_in_yang)
        yang_to_yin, valid_yin = _warp_volume(yang_cost, yang_support, query_yin, self.depth_chunk)

        # Yin→Yang：在Yang位置查询Yin。
        query_yang = _CachedQuery(self.yin_shape, self.yang_to_yin, self.yang_in_yin)
        yin_to_yang, valid_yang = _warp_volume(yin_cost, yin_support, query_yang, self.depth_chunk)

        return dict(
            yang_to_yin=yang_to_yin,
            yang_to_yin_valid=valid_yin,
            yin_to_yang=yin_to_yang,
            yin_to_yang_valid=valid_yang,
        )
