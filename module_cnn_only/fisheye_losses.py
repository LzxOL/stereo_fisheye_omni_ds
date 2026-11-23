"""Dual-fisheye non-pinhole geometry losses for Yin-Yang stereo matching.

Three data-driven, physically grounded loss components derived from Step 0 diagnosis:
  1. Exp-1 (Anisotropy): Spherical pixel footprint anisotropy weighting kappa(u, v)
     to compensate severe outer-ring fisheye stretching (kappa up to 2.71x).
  2. Exp-2 (Epipolar Arc): Differentiable curved right-fisheye epipolar-arc
     reprojection loss ||u_R(k_hat) - u_R(k*)|| in stride-2 feature pixels,
     targeting high-sensitivity regions (s >= 0.60) causing 39% of >5 errors.
  3. Exp-3 (3D Spherical Normal + Edge): Reciprocal-free, distortion-invariant
     3D surface normal cosine loss on GT-smooth interiors + gentle GT-edge weighting
     on the 6% depth discontinuities causing 32.6% of >5 errors.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dataset_stereo import fisheye_reference_rays, project_camera, applyTransform


def compute_gt_neighbor_jump(gt, valid):
    """Compute local 3x3 max depth jump and strict 3x3 valid mask on [B, 1, H, W]."""
    vm = valid.bool()
    gt_f = gt.float()
    gt_max = F.max_pool2d(gt_f.masked_fill(~vm, -1e9), kernel_size=3, stride=1, padding=1)
    gt_min = -F.max_pool2d((-gt_f).masked_fill(~vm, -1e9), kernel_size=3, stride=1, padding=1)
    jump = torch.maximum(gt_max - gt_f, gt_f - gt_min).masked_fill(~vm, 0.0)
    all_3x3_valid = F.avg_pool2d(vm.float(), kernel_size=3, stride=1, padding=1) > 0.999
    return jump, all_3x3_valid


class FisheyeGeometryLossPack(nn.Module):
    """Precomputed calibrated dual-fisheye buffers and FP32-safe geometry losses."""

    def __init__(self, dataset, grid_package, *,
                 aniso_alpha=0.0,
                 epi_weight=0.0,
                 normal_weight=0.0,
                 edge_alpha=0.0,
                 edge_thresh=3.0,
                 smooth_thresh=1.0,
                 normal_last_iters=4):
        super().__init__()
        self.aniso_alpha = float(aniso_alpha)
        self.epi_weight = float(epi_weight)
        self.normal_weight = float(normal_weight)
        self.edge_alpha = float(edge_alpha)
        self.edge_thresh = float(edge_thresh)
        self.smooth_thresh = float(smooth_thresh)
        self.normal_last_iters = int(normal_last_iters)

        ref = grid_package.reference
        rays_hw3 = np.asarray(ref["rays_ref"], dtype=np.float64)  # [H, W, 3]
        valid_geo = np.asarray(ref["left_valid"], dtype=bool)
        H, W = rays_hw3.shape[:2]

        du = np.zeros_like(rays_hw3)
        dv = np.zeros_like(rays_hw3)
        du[:, 1:-1] = 0.5 * (rays_hw3[:, 2:] - rays_hw3[:, :-2])
        dv[1:-1, :] = 0.5 * (rays_hw3[2:, :] - rays_hw3[:-2, :])
        inner = np.zeros_like(valid_geo)
        inner[1:-1, 1:-1] = (
            valid_geo[1:-1, 1:-1] & valid_geo[1:-1, 2:] & valid_geo[1:-1, :-2]
            & valid_geo[2:, 1:-1] & valid_geo[:-2, 1:-1]
        )

        # 1. Pixel footprint anisotropy kappa(u, v) in [1.0, ~2.71]
        lu = np.linalg.norm(du, axis=-1)
        lv = np.linalg.norm(dv, axis=-1)
        aniso = np.where(inner, np.maximum(lu, lv) / np.maximum(np.minimum(lu, lv), 1e-12), 1.0)
        aniso = np.clip(aniso, 1.0, 3.0).astype(np.float32)
        self.register_buffer("aniso_map", torch.from_numpy(aniso)[None, None], persistent=False)  # [1, 1, H, W]
        self.register_buffer("inner_geo", torch.from_numpy(inner)[None, None], persistent=False)

        # 2. Exact ray & tangent derivative buffers for reciprocal-free 3D normals
        # Shape [1, 3, H, W] in FP32
        self.register_buffer("rays_v", torch.from_numpy(rays_hw3.transpose(2, 0, 1).astype(np.float32))[None], persistent=False)
        self.register_buffer("rays_du", torch.from_numpy(du.transpose(2, 0, 1).astype(np.float32))[None], persistent=False)
        self.register_buffer("rays_dv", torch.from_numpy(dv.transpose(2, 0, 1).astype(np.float32))[None], persistent=False)

        # 3. Curved right-fisheye epipolar lookup table [H, W, D, 2] (in stride-2 feature px) + sensitivity [H, W, D]
        if self.epi_weight > 0.0:
            axis = np.asarray(dataset.axis, dtype=np.float64)
            D = len(axis)
            rays_3n, _, _ = fisheye_reference_rays(dataset.reference, dataset.output_size)
            rays_3n = rays_3n.astype(np.float64)
            curve = np.zeros((D, H * W, 2), dtype=np.float32)
            curve_vis = np.zeros((D, H * W), dtype=bool)
            for d_idx, rho in enumerate(axis):
                pts = applyTransform(dataset.target.rig2cam,
                                     applyTransform(dataset.reference.cam2rig, rays_3n / max(float(rho), 1e-12)))
                pixels, vis = project_camera(dataset.target, pts)
                px_feat = (pixels.T * 0.5).astype(np.float32)  # 384x400 feature px
                px_feat = np.where(vis[:, None], px_feat, 0.0)
                curve[d_idx] = px_feat
                curve_vis[d_idx] = vis
            # Compute local epipolar sensitivity ||du_R / dk|| with safe central differences
            sens = np.zeros((D, H * W), dtype=np.float32)
            for d_idx in range(D):
                lo = max(0, d_idx - 1)
                hi = min(D - 1, d_idx + 1)
                step = np.linalg.norm(curve[hi] - curve[lo], axis=-1) / max(hi - lo, 1)
                ok = curve_vis[lo] & curve_vis[hi]
                sens[d_idx] = np.where(ok, np.clip(step, 0.05, 1.5), 0.5)
            # Store as [H*W, D, 2] (fp16 saves VRAM: 59MB -> 29MB) and [H*W, D]
            curve_hw = np.transpose(curve, (1, 0, 2)).copy()       # [HW, D, 2]
            vis_hw = np.transpose(curve_vis, (1, 0)).copy()        # [HW, D]
            sens_hw = np.transpose(sens, (1, 0)).copy()            # [HW, D]
            self.register_buffer("right_curve", torch.from_numpy(curve_hw).half(), persistent=False)
            self.register_buffer("right_curve_vis", torch.from_numpy(vis_hw), persistent=False)
            self.register_buffer("epi_sens", torch.from_numpy(sens_hw).half(), persistent=False)
            self.num_depth = D
        else:
            self.num_depth = int(len(dataset.axis))

    @property
    def active(self):
        return (self.aniso_alpha > 0.0) or (self.epi_weight > 0.0) or (self.normal_weight > 0.0) or (self.edge_alpha > 0.0)

    def _build_spatial_weight(self, valid, gt_jump=None):
        """Return per-pixel weight [B, 1, H, W] normalized to mean 1.0 over valid pixels."""
        w = torch.ones_like(valid, dtype=torch.float32)
        if self.aniso_alpha > 0.0:
            w = w * (1.0 + self.aniso_alpha * (self.aniso_map - 1.0))
        if self.edge_alpha > 0.0 and gt_jump is not None:
            is_edge = (gt_jump >= self.edge_thresh).float()
            w = w * (1.0 + self.edge_alpha * is_edge)
        vm = valid.bool()
        mean_w = w[vm].mean().clamp_min(1e-6)
        return w / mean_w

    def _sample_right_curve(self, k_hw):
        """Differentiable 1D linear interpolation along [HW, D, 2] epipolar curve.

        Args:
            k_hw: [B, 1, H, W] float32 continuous inverse-depth index in [0, D-1].
        Returns:
            u_r: [B, H*W, 2] float32 right feature coordinates
            vis: [B, H*W] bool validity of both bracketing samples
            sens: [B, H*W] float32 local epipolar sensitivity
        """
        b = k_hw.shape[0]
        D = self.num_depth
        k_flat = k_hw.reshape(b, -1).clamp(0.0, float(D - 1))  # [B, HW]
        k0 = k_flat.floor().long().clamp(0, D - 2)
        k1 = k0 + 1
        t = (k_flat - k0.float()).unsqueeze(-1)  # [B, HW, 1]
        hw_idx = torch.arange(k_flat.shape[1], device=k_hw.device).unsqueeze(0).expand(b, -1)
        u0 = self.right_curve[hw_idx, k0].float()
        u1 = self.right_curve[hw_idx, k1].float()
        u_r = u0 + t * (u1 - u0)
        vis = self.right_curve_vis[hw_idx, k0] & self.right_curve_vis[hw_idx, k1]
        s0 = self.epi_sens[hw_idx, k0].float()
        s1 = self.epi_sens[hw_idx, k1].float()
        sens = s0 + t.squeeze(-1) * (s1 - s0)
        return u_r, vis, sens

    def _epipolar_arc_loss(self, pred_hw, gt_hw, valid_hw, u_gt_cache=None):
        """Charbonnier right-fisheye epipolar arc reprojection distance (stride-2 feat px)."""
        if u_gt_cache is None:
            u_gt, vis_gt, sens_gt = self._sample_right_curve(gt_hw)
        else:
            u_gt, vis_gt, sens_gt = u_gt_cache
        u_pred, vis_pred, _ = self._sample_right_curve(pred_hw)
        arc_dist = torch.sqrt(((u_pred - u_gt) ** 2).sum(dim=-1) + 1e-6) - 1e-3
        arc_dist = arc_dist.clamp_min(0.0)
        # Safe first-order tangent fallback if early-training prediction leaves right FoV
        b = pred_hw.shape[0]
        lin_dist = sens_gt * (pred_hw.reshape(b, -1) - gt_hw.reshape(b, -1)).abs()
        both_vis = vis_pred & vis_gt
        dist = torch.where(both_vis, arc_dist, lin_dist)
        vm = valid_hw.reshape(b, -1).bool()
        return dist[vm].mean()

    def _compute_normals_3d(self, k_hw):
        """Reciprocal-free exact 3D spherical surface normals from continuous inv-depth index k.

        Since P = v / (k * step_rho), P_u x P_v is strictly parallel to:
            t_u x t_v,  where t_u = k * v_u - k_u * v,  t_v = k * v_v - k_v * v.
        Eliminates 1/rho explosion near k -> 0 while remaining 100% fisheye distortion-invariant.
        """
        k_safe = k_hw.float().clamp_min(0.5)  # [B, 1, H, W]
        k_u = torch.zeros_like(k_safe)
        k_v = torch.zeros_like(k_safe)
        k_u[:, :, :, 1:-1] = 0.5 * (k_safe[:, :, :, 2:] - k_safe[:, :, :, :-2])
        k_v[:, :, 1:-1, :] = 0.5 * (k_safe[:, :, 2:, :] - k_safe[:, :, :-2, :])
        t_u = k_safe * self.rays_du - k_u * self.rays_v  # [B, 3, H, W]
        t_v = k_safe * self.rays_dv - k_v * self.rays_v  # [B, 3, H, W]
        n = torch.cross(t_u, t_v, dim=1)
        norm = torch.sqrt((n * n).sum(dim=1, keepdim=True) + 1e-12)
        n_unit = n / norm
        return n_unit, (norm[:, 0] > 1e-7)

    def sequence_loss(self, preds, gt, valid, *, loss_gamma=0.9, include_normal=True, include_epi=True):
        """Drop-in replacement for sequence_loss with calibrated dual-fisheye geometry terms."""
        n_preds = len(preds)
        assert n_preds >= 1
        adjusted_gamma = loss_gamma ** (15.0 / n_preds)
        vm = valid.bool()
        gt_f = gt.float()

        with torch.autocast(device_type=gt.device.type, enabled=False):
            gt_jump = None
            all_3x3_valid = None
            if self.edge_alpha > 0.0 or (self.normal_weight > 0.0 and include_normal):
                gt_jump, all_3x3_valid = compute_gt_neighbor_jump(gt_f, vm)

            use_spatial_w = (self.aniso_alpha > 0.0) or (self.edge_alpha > 0.0)
            w_spatial = self._build_spatial_weight(vm, gt_jump) if use_spatial_w else None

            u_gt_cache = None
            if self.epi_weight > 0.0 and include_epi:
                with torch.no_grad():
                    u_gt_cache = self._sample_right_curve(gt_f)

            n_gt = None
            smooth_mask = None
            if self.normal_weight > 0.0 and include_normal:
                with torch.no_grad():
                    n_gt, ok_gt = self._compute_normals_3d(gt_f)
                    smooth_mask = (
                        vm[:, 0] & all_3x3_valid[:, 0] & self.inner_geo[:, 0]
                        & (gt_jump[:, 0] < self.smooth_thresh) & (gt_f[:, 0] >= 1.0) & ok_gt
                    )

            total = gt_f.new_tensor(0.0)
            normal_start = max(0, n_preds - self.normal_last_iters)
            for i, pred in enumerate(preds):
                pred_f = pred.float()
                i_weight = adjusted_gamma ** (n_preds - i - 1)
                err = (pred_f - gt_f).abs()
                if w_spatial is not None:
                    l1_term = (w_spatial * err)[vm].mean()
                else:
                    l1_term = err[vm].mean()
                step_loss = l1_term

                if self.epi_weight > 0.0 and include_epi:
                    step_loss = step_loss + self.epi_weight * self._epipolar_arc_loss(
                        pred_f, gt_f, vm, u_gt_cache=u_gt_cache
                    )

                if self.normal_weight > 0.0 and include_normal and i >= normal_start and smooth_mask.any():
                    n_pred, ok_pred = self._compute_normals_3d(pred_f)
                    nm = smooth_mask & ok_pred
                    if nm.any():
                        cos_sim = (n_pred * n_gt).sum(dim=1).clamp(-1.0, 1.0)
                        step_loss = step_loss + self.normal_weight * (1.0 - cos_sim[nm]).mean()

                total = total + i_weight * step_loss
            return total
