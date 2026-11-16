"""旧独立入口的兼容接口；网络实现已移入network.py的OmniDS。

新代码请直接使用OmniDS(..., grid_package=package)，设置use_yinyang=True。
保留旧类名、调用方式和state_dict键，避免已有实验脚本/权重失效。
"""
from module_cnn_only.network import OmniDS


class DualFisheyeStereoNetwork(OmniDS):
    """将旧构造参数映射到正式OmniDS；不维护第二套网络或Forward算法。"""
    def __init__(self, grid_package, use_rgb=False, depth_chunk=8, **refinement_options):
        depth = refinement_options.get('num_depth', len(grid_package.metadata['inverse_depth_candidates']))
        options = dict(num_cams=2, use_yinyang=True, use_rgb=use_rgb,
                       num_invdepth=depth, num_depth_samples=depth,
                       base_channel=refinement_options.get('context_channels', 32),
                       depth_chunk=depth_chunk, yinyang_refinement=refinement_options)
        super().__init__(options, grid_package=grid_package)

    def forward(self, imgs, yang_to_yin_quality, yin_to_yang_quality, iters=None):
        return super().forward(imgs, iters=iters, yang_to_yin_quality=yang_to_yin_quality,
                               yin_to_yang_quality=yin_to_yang_quality)
