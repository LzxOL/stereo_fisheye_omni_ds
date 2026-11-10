import torch
import torch.nn.functional as F


def spherical_sweep(fisheye_feats, grids):
    """
    Build spherical (ERP) feature volumes by sampling fisheye features with precomputed grids,

    Args:
        fisheye_feats:
            - list/tuple of N tensors, N=2 (stereo) or 4, each [B,C,H_fish,W_fish], or
            - tensor [B, N, C, H_fish, W_fish]
        grids:
            list of N tensors, each [H_erp,W_erp,D,2] in grid_sample normalized coords.

    Returns:
        sph_feats: N tensors; each camera volume is [B,C,H_erp,W_erp,D].
    """
    if isinstance(fisheye_feats, (list, tuple)):
        assert len(fisheye_feats) in (2, 4), "Expected two or four camera features"
        feats = list(fisheye_feats)
    else:
        assert fisheye_feats.ndim == 5 and fisheye_feats.shape[1] in (2, 4), "Expected [B,N,C,H,W], N=2/4"
        feats = [fisheye_feats[:, i] for i in range(fisheye_feats.shape[1])]
    if len(grids) != len(feats):
        raise ValueError("Each camera feature needs exactly one sampling grid")

    bs = feats[0].shape[0]
    device = feats[0].device
    dtype = feats[0].dtype

    grids = [g.to(device=device, dtype=dtype) for g in grids]
    grids_pad = [torch.cat([torch.zeros_like(g[..., :1]), g], dim=-1) for g in grids]  # [H,W,D,3]

    sph_feats = []
    for feat, grid in zip(feats, grids_pad):
        # Use 5D grid_sample trick (treat last dim as dummy "W"=1) to sample all depths in one call.
        sph_feat = F.grid_sample(feat[..., None], grid.repeat(bs, 1, 1, 1, 1), align_corners=True)
        sph_feats.append(sph_feat)

    return sph_feats

