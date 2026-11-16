"""旧调用方式的薄兼容接口，实际网络始终使用network.py的DA-V2主干。"""
from module_cnn_only.network import OmniDS


class DualFisheyeStereoNetwork(OmniDS):
    def __init__(self, grid_package, use_rgb=False, depth_chunk=8, **options):
        depth = options.pop('num_depth', len(grid_package.metadata['inverse_depth_candidates']))
        options.update(num_cams=2, use_yinyang=True, use_rgb=use_rgb, depth_chunk=depth_chunk,
                       num_invdepth=depth, num_depth_samples=depth)
        super().__init__(options, grid_package=grid_package)

    def forward(self, imgs, yang_to_yin_quality, yin_to_yang_quality, iters=None, dav2_rgb=None):
        return super().forward(imgs, iters=iters, dav2_rgb=dav2_rgb,
                               yang_to_yin_quality=yang_to_yin_quality, yin_to_yang_quality=yin_to_yang_quality)
