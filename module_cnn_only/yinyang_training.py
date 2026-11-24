"""训练几何适配：固定GPU网格、真实采样质量、双网格预测→左鱼眼输出。

不改变标定/扫描/GRU；GT仍用StereoDataset已有的Rig→左光心点云转换。
支持原始双线性+静态Q软融合，以及边缘感知Bilateral回投和置信度/分歧感知重叠选路。
"""
import math
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.local_spherical_grid.adaptive_local_grid_v1 import sample_patch

REPROJ_MODES = ('bilinear', 'bilateral', 'nearest')
OVERLAP_MODES = ('q_soft', 'q_hard', 'confidence_disagree')


def sampling_quality(patch):
    # 真实原生像素的两个球面边长：cos(latitude)*dLongitude 与 dLatitude。
    # Q=1/kappa∈(0,1]，数值越大采样越均匀；不是二值有效性Mask。
    b = patch.bounds
    height, width = patch.original_shape
    visual = np.nan_to_num(patch.rays_ref) @ np.diag([-1., -1., 1.]) @ patch.basis
    cos_lat = np.sqrt(np.maximum(0., 1. - visual[..., 1] ** 2))
    dx = np.deg2rad(2 * b['longitude_half_span_deg'] / (width - 1)) * cos_lat
    dy = np.deg2rad((b['latitude_max_deg'] - b['latitude_min_deg']) / (height - 1))
    q = np.minimum(dx, dy) / np.maximum(np.maximum(dx, dy), 1e-12)
    return np.where(patch.padding_valid, q, 0.).astype(np.float32)


class YinYangTrainingGeometry(nn.Module):
    def __init__(self, package, factor=2, *, reproj_mode='bilinear', bilateral_tau=2.0,
                 overlap_mode='q_soft', disagree_delta=3.0, disagree_gamma=0.8, disagree_beta_max=4.0,
                 use_entropy_overlap=False):
        super().__init__()
        if reproj_mode not in REPROJ_MODES:
            raise ValueError(f'Unknown reproj_mode: {reproj_mode!r}; expected one of {REPROJ_MODES}')
        if overlap_mode not in OVERLAP_MODES:
            raise ValueError(f'Unknown overlap_mode: {overlap_mode!r}; expected one of {OVERLAP_MODES}')
        if bilateral_tau <= 0 or disagree_gamma <= 0 or disagree_beta_max < 0:
            raise ValueError('bilateral_tau and disagree_gamma must be positive; disagree_beta_max >= 0')
        self.factor = factor
        self.reproj_mode = reproj_mode
        self.bilateral_tau = float(bilateral_tau)
        self.overlap_mode = overlap_mode
        self.disagree_delta = float(disagree_delta)
        self.disagree_gamma = float(disagree_gamma)
        self.disagree_beta_max = float(disagree_beta_max)
        self.use_entropy_overlap = bool(use_entropy_overlap)
        if overlap_mode == 'confidence_disagree':
            # softplus(raw) 初始为 1.0，对应离线已验证的 Q / (1 + spread) 几何先验。
            init_raw = math.log(math.e - 1.0)
            self.spread_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
            if self.use_entropy_overlap:
                # M3: softplus(-3.0) ~= 0.0486, gentle non-negative cold-start for -w_ent * H_1.
                self.ent_raw = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))
        self.shapes = {name: getattr(package, name).shape for name in ('yin', 'yang')}
        self.output_size = package.reference['left_valid'].shape
        for name, source in [('yin', 'yang'), ('yang', 'yin')]:
            patch = getattr(package, name)
            # 非持久buffer随model.cuda移动，但不将巨大的几何表重复写入每个checkpoint。
            fields = ['left_grid', 'right_grid', 'stereo_valid', 'left_feature_valid', 'padding_valid', 'rays_ref']
            for field in fields:
                dtype = torch.bool if field.endswith('valid') else torch.float32
                self.register_buffer(name + '_' + field, torch.as_tensor(getattr(patch, field), dtype=dtype).clone(),
                                     persistent=False)
            src = getattr(package, source)
            quality = torch.tensor(sampling_quality(src))[None, None]
            support = torch.tensor(src.padding_valid)[None, None]
            q, _ = sample_patch(quality, src, patch.rays_ref, support)
            self.register_buffer(name + '_cross_quality', q.unsqueeze(2), persistent=False)
            # 将原生连续像素x映射到凸上采样后的factor*x，匹配原网络的stride-2中心定义。
            # 不能直接复用旧归一化坐标，也不能随意加入半像素偏移。
            coordinates = np.asarray(package.reference['to_' + name])
            h, w = patch.shape
            scale = np.array([factor * (w - 1) / (factor * w - 1), factor * (h - 1) / (factor * h - 1)])
            up_coordinates = (coordinates + 1) * scale - 1
            window = np.asarray(package.reference['in_' + name])
            up_coordinates = np.where(window[..., None], up_coordinates, 2.)
            up_t = torch.tensor(up_coordinates, dtype=torch.float32)
            self.register_buffer(name + '_output_grid', up_t, persistent=False)
            self.register_buffer(name + '_output_window', torch.tensor(window, dtype=torch.bool), persistent=False)
            own_quality = torch.tensor(sampling_quality(patch))[None, None]
            qout, _ = sample_patch(own_quality, patch, package.reference['rays_ref'],
                                   torch.tensor(patch.padding_valid)[None, None])
            self.register_buffer(name + '_output_quality', qout, persistent=False)

            # 预计算 2x2 球面角点索引与标准双线性权重，与 sample_patch 的严格四邻点支持完全一致。
            h_up, w_up = factor * h, factor * w
            xy = (up_t + 1.0) * torch.tensor([w_up - 1, h_up - 1], dtype=torch.float32) / 2.0
            x, y = xy[..., 0], xy[..., 1]
            x0, y0 = x.floor().long(), y.floor().long()
            fx, fy = x - x0.float(), y - y0.float()
            corners = (
                (x0, y0, (1.0 - fx) * (1.0 - fy)),
                (x0 + 1, y0, fx * (1.0 - fy)),
                (x0, y0 + 1, (1.0 - fx) * fy),
                (x0 + 1, y0 + 1, fx * fy),
            )
            idxs, insides, ws = [], [], []
            for ix, iy, wc in corners:
                inside = (ix >= 0) & (ix < w_up) & (iy >= 0) & (iy < h_up)
                idx = iy.clamp(0, h_up - 1) * w_up + ix.clamp(0, w_up - 1)
                idxs.append(idx)
                insides.append(inside)
                ws.append(wc)
            corner_idx = torch.stack(idxs, dim=0)
            corner_inside = torch.stack(insides, dim=0)
            corner_w = torch.stack(ws, dim=0)
            corner_near = corner_w.masked_fill(~corner_inside, -1.0).argmax(dim=0, keepdim=True)
            self.register_buffer(name + '_corner_idx', corner_idx, persistent=False)
            self.register_buffer(name + '_corner_inside', corner_inside, persistent=False)
            self.register_buffer(name + '_corner_w', corner_w, persistent=False)
            self.register_buffer(name + '_corner_near', corner_near, persistent=False)

    def patch(self, name):
        return SimpleNamespace(**{field: getattr(self, name + '_' + field) for field in
                                  ['left_grid', 'right_grid', 'stereo_valid', 'left_feature_valid',
                                   'padding_valid', 'rays_ref']})

    def cross_quality(self, name, batch, depth):
        return getattr(self, name + '_cross_quality').expand(batch, 1, depth, -1, -1)

    def _project_chart(self, name, pred, support, conf_map=None, *, cached_valid=None):
        """单路 [B,1,2H,2W] → [B,1,Ho,Wo]，FP32下执行严格掩码回投并输出2x2局部深度极差。"""
        with torch.autocast(device_type=pred.device.type, enabled=False):
            b = pred.shape[0]
            corner_idx = getattr(self, name + '_corner_idx')
            corner_inside = getattr(self, name + '_corner_inside')
            corner_w = getattr(self, name + '_corner_w')
            corner_near = getattr(self, name + '_corner_near')
            window = getattr(self, name + '_output_window')
            ho, wo = window.shape
            idx_flat = corner_idx.reshape(1, -1).expand(b, -1)
            sup_bool = support.bool()
            if cached_valid is None:
                with torch.no_grad():
                    obs = sup_bool[:, 0].reshape(b, -1).gather(1, idx_flat).reshape(b, 4, ho, wo)
                    active = corner_w > 0
                    valid = (window[None] & (~active[None] | (corner_inside[None] & obs)).all(dim=1)).unsqueeze(1)
            else:
                valid = cached_valid

            clean = pred.float().masked_fill(~sup_bool, 0.0)
            vals = clean[:, 0].reshape(b, -1).gather(1, idx_flat).reshape(b, 4, ho, wo)
            with torch.no_grad():
                active = corner_w > 0
                vals_det = vals.detach()
                vmax = vals_det.masked_fill(~active[None], -1e9).max(dim=1, keepdim=True).values
                vmin = vals_det.masked_fill(~active[None], 1e9).min(dim=1, keepdim=True).values
                spread = (vmax - vmin).clamp_min(0.0).masked_fill(~valid, 0.0)

            if self.reproj_mode == 'bilinear':
                sampled = (corner_w[None] * vals).sum(dim=1, keepdim=True)
            elif self.reproj_mode == 'nearest':
                sampled = vals.gather(1, corner_near[None].expand(b, -1, -1, -1))
            else:
                # Bilateral 边缘感知回投：平滑区退化为双线性，深度突变边切断跨边缘插值与梯度串扰。
                with torch.no_grad():
                    d_near = vals_det.gather(1, corner_near[None].expand(b, -1, -1, -1))
                    diff = (vals_det - d_near).abs()
                    wb = corner_w[None] * torch.exp(-diff / self.bilateral_tau)
                    wb = wb / wb.sum(dim=1, keepdim=True).clamp_min(1e-8)
                sampled = (wb * vals).sum(dim=1, keepdim=True)
            sampled = sampled.masked_fill(~valid, 0.0)

            conf = None
            if conf_map is not None:
                clean_c = conf_map.float().masked_fill(~sup_bool, 0.0)
                c_vals = clean_c[:, 0].reshape(b, -1).gather(1, idx_flat).reshape(b, 4, ho, wo)
                conf = (corner_w[None] * c_vals).sum(dim=1, keepdim=True).masked_fill(~valid, 0.0)
            return sampled, valid, spread, conf

    def _project_scalar(self, name, scalar_map, support, valid):
        """Bilinearly project a detached [B,1,2H,2W] scalar map (e.g. H_1) to [B,1,Ho,Wo]."""
        with torch.no_grad(), torch.autocast(device_type=scalar_map.device.type, enabled=False):
            b = scalar_map.shape[0]
            corner_idx = getattr(self, name + '_corner_idx')
            corner_w = getattr(self, name + '_corner_w')
            ho, wo = getattr(self, name + '_output_window').shape
            idx_flat = corner_idx.reshape(1, -1).expand(b, -1)
            clean = scalar_map.float().masked_fill(~support.bool(), 0.0)
            vals = clean[:, 0].reshape(b, -1).gather(1, idx_flat).reshape(b, 4, ho, wo)
            return (corner_w[None] * vals).sum(dim=1, keepdim=True).masked_fill(~valid, 0.0)

    def project_init(self, result):
        """M1: Project differentiable GEV initial depth d_0 for Yin & Yang onto left_fisheye."""
        init_preds, init_valid = {}, {}
        for name in ('yin', 'yang'):
            row = result[name]
            d0_2x = row.get('init_pred_2x')
            if d0_2x is None:
                continue
            q = getattr(self, name + '_output_quality')
            d_proj, v_proj, _, _ = self._project_chart(name, d0_2x, row['valid'])
            eff = v_proj & (q > 0)
            init_preds[name] = d_proj.masked_fill(~eff, 0.0)
            init_valid[name] = eff
        return init_preds, init_valid

    def project(self, result, *, return_branches=False):
        """每轮[B,1,2Hchart,2Wchart]→[B,1,Hleft,Wleft]；严格检查插值邻点。

        重叠处按配置执行静态Q软融合、Q硬选路，或置信度+分歧自适应选路。
        只有一路有效时完全取该路；两路都无效为零。
        """
        predictions = []
        branch_preds = {'yin': [], 'yang': []}
        branch_valid = {}
        q_yin = self.yin_output_quality
        q_yang = self.yang_output_quality
        n_iters = len(result['yin']['predictions'])
        v_yin_cached, v_yang_cached = None, None
        h1_yin, h1_yang = None, None
        ent_yin_src = result['yin'].get('entropy_map')
        ent_yang_src = result['yang'].get('entropy_map')
        for iteration in range(n_iters):
            confs_yin = result['yin'].get('conf_logits')
            confs_yang = result['yang'].get('conf_logits')
            d_yin, v_yin, sp_yin, c_yin = self._project_chart(
                'yin', result['yin']['predictions'][iteration], result['yin']['valid'],
                None if confs_yin is None else confs_yin[iteration],
                cached_valid=v_yin_cached,
            )
            d_yang, v_yang, sp_yang, c_yang = self._project_chart(
                'yang', result['yang']['predictions'][iteration], result['yang']['valid'],
                None if confs_yang is None else confs_yang[iteration],
                cached_valid=v_yang_cached,
            )
            v_yin_cached, v_yang_cached = v_yin, v_yang
            if iteration == 0 and self.use_entropy_overlap and ent_yin_src is not None and ent_yang_src is not None:
                h1_yin = self._project_scalar('yin', ent_yin_src, result['yin']['valid'], v_yin)
                h1_yang = self._project_scalar('yang', ent_yang_src, result['yang']['valid'], v_yang)
            eff_yin = v_yin & (q_yin > 0)
            eff_yang = v_yang & (q_yang > 0)
            both = eff_yin & eff_yang
            output_valid = eff_yin | eff_yang

            if return_branches:
                branch_preds['yin'].append(d_yin.masked_fill(~eff_yin, 0.0))
                branch_preds['yang'].append(d_yang.masked_fill(~eff_yang, 0.0))
                branch_valid['yin'] = eff_yin
                branch_valid['yang'] = eff_yang

            with torch.autocast(device_type=d_yin.device.type, enabled=False):
                if self.overlap_mode == 'q_soft':
                    w0 = q_yin.masked_fill(~v_yin, 0.0)
                    w1 = q_yang.masked_fill(~v_yang, 0.0)
                    total = w0 + w1
                    prediction = (d_yin * w0 + d_yang * w1) / total.clamp_min(1e-8)
                elif self.overlap_mode == 'q_hard':
                    single = torch.where(eff_yin, d_yin, d_yang)
                    hard = torch.where(q_yin >= q_yang, d_yin, d_yang)
                    prediction = torch.where(both, hard, single)
                else:
                    sw = F.softplus(self.spread_raw)
                    s_yin = torch.log(q_yin.clamp_min(1e-6)) - sw * torch.log1p(sp_yin)
                    s_yang = torch.log(q_yang.clamp_min(1e-6)) - sw * torch.log1p(sp_yang)
                    if self.use_entropy_overlap and h1_yin is not None and h1_yang is not None:
                        w_ent = F.softplus(self.ent_raw)
                        s_yin = s_yin - w_ent * h1_yin
                        s_yang = s_yang - w_ent * h1_yang
                    if c_yin is not None:
                        s_yin = s_yin + c_yin
                    if c_yang is not None:
                        s_yang = s_yang + c_yang
                    with torch.no_grad():
                        delta = (d_yin - d_yang).abs()
                        beta = 1.0 + self.disagree_beta_max * torch.sigmoid(
                            (delta - self.disagree_delta) / self.disagree_gamma
                        )
                    logits = beta * torch.cat((s_yin, s_yang), dim=1)
                    probs = logits.softmax(dim=1)
                    fused_both = probs[:, :1] * d_yin + probs[:, 1:2] * d_yang
                    single = torch.where(eff_yin, d_yin, d_yang)
                    prediction = torch.where(both, fused_both, single)
                predictions.append(prediction.masked_fill(~output_valid, 0.0))

        if return_branches:
            return predictions, output_valid, branch_preds, branch_valid
        return predictions, output_valid
