"""在真实 OCam 共视集合上搜索局部 Yang；不创建网络，不读 GT。

Yin 固定竖直极轴及低纬度带；其经度窗口由实际候选有效性拟合。
Yang 搜索极轴方向，最短圆弧拟合经度、min/max 拟合纬度。经度中心等价于
绕 Yang 极轴的第三个旋转角，因此不重复搜索与自由经度边界等价的 roll。
"""
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from plotly.offline import get_plotlyjs

from camera_geometry import (load_pair, pair_geometry, spherical_rays, OCAM_FROM_VIS,
                             sample_rgb, weighted_summary, project_camera)
from utils.common import EPS
from visualize_stereo_sphere import save_figure, PLOT_STYLE, QA


# 此处颜色表示“网格归属”，与上一实验表示相机覆盖的配色分开说明。
COLORS = np.array([[232, 235, 240], [0, 114, 178], [230, 159, 0],
                   [155, 81, 185], [205, 60, 65]], dtype=np.uint8)
LABELS = ['No common support', 'Yin only', 'Yang only', 'Yin + Yang', 'Common but uncovered']


def ry(degrees):
    a = np.deg2rad(degrees)
    return np.array([[np.cos(a), 0., np.sin(a)], [0., 1., 0.], [-np.sin(a), 0., np.cos(a)]])


def rx(degrees):
    a = np.deg2rad(degrees)
    return np.array([[1., 0., 0.], [0., np.cos(a), -np.sin(a)], [0., np.sin(a), np.cos(a)]])


def depth_axis(config, num_depth=96, full_num_depth=192):
    # 与 dataset.Dataset.__initSweep 及 StereoDataset 的 [::192//96] 完全同轴。
    # 不能改成 linspace(EPS, 1/min_depth, 96)，否则最近的一层也会改变。
    if num_depth < 2 or full_num_depth % num_depth:
        raise ValueError('num-depth must divide full-num-depth')
    minimum = float(config['config']['omnimvs_sweep_min_depth'])
    maximum = 1. / EPS  # dataset.py 原始默认 max_depth
    lo, hi = 1./maximum, 1./minimum
    step = (hi-lo)/(full_num_depth-1)
    full = np.arange(lo, hi+step, step, dtype=np.float64)
    axis = full[::full_num_depth//num_depth][:num_depth]
    return axis


def depth_visibility(cameras, rays_ocam, rotation, translation, axis):
    """返回左自身 [H,W]、右投影 [D,H,W]、共同有效 [D,H,W]。

    左侧只投影一次。右侧逐深度完整变换；此 Mask 只判断投影与原始 Mask，
    不进行 GT 遮挡检查，也不把合法投影错误解释成无遮挡。
    """
    shape = rays_ocam.shape[:2]
    rays = rays_ocam.reshape(-1, 3).T
    _, left = project_camera(cameras[0], rays)
    target = np.empty((len(axis), *shape), dtype=bool)
    for i, rho in enumerate(axis):
        points = rotation @ (rays / rho) + translation[:, None]
        _, valid = project_camera(cameras[1], points)
        target[i] = valid.reshape(shape)
    left = left.reshape(shape)
    return left, target, target & left[None]


def angular_coordinates(rays_vis, basis):
    local = rays_vis @ basis
    lon = np.rad2deg(np.arctan2(local[..., 0], local[..., 2]))
    lat = np.rad2deg(np.arcsin(np.clip(local[..., 1], -1., 1.)))
    return lon, lat


def shortest_longitude_window(lon, padding, boundary_step):
    """圆周上删去最大的空隙，剩下的圆弧包住所有目标点，正确处理 ±180°。

    返回中心和半宽。先求最小包围弧，再向外添加几何余量和边界量化；
    并没有对未覆盖点作插值或“填洞”。
    """
    angles = np.sort(np.mod(np.asarray(lon).ravel(), 360.))
    if len(angles) == 0:
        raise ValueError('Empty target set: no Yang is needed under the chosen Yin')
    gaps = np.diff(np.r_[angles, angles[0]+360])
    cut = int(np.argmax(gaps))
    start = angles[(cut+1) % len(angles)]
    span = 360.-gaps[cut]
    center = (start+span/2+180) % 360-180
    half = np.ceil((span/2+padding)/boundary_step)*boundary_step
    return float(center), float(half)


def window_mask(lon, lat, bounds):
    delta = (lon-bounds['longitude_center_deg']+180) % 360-180
    return ((np.abs(delta) <= bounds['longitude_half_span_deg']+1e-8) &
            (lat >= bounds['latitude_min_deg']-1e-8) &
            (lat <= bounds['latitude_max_deg']+1e-8))


def fit_yin(lon, lat, valid_any, args):
    low = valid_any & (np.abs(lat) <= args.yin_lat_limit)
    center, half = shortest_longitude_window(lon[low], args.padding_deg, args.boundary_step_deg)
    bounds = dict(longitude_center_deg=center, longitude_half_span_deg=half,
                  latitude_min_deg=-args.yin_lat_limit, latitude_max_deg=args.yin_lat_limit)
    mask = window_mask(lon, lat, bounds)
    return bounds, mask


def fit_candidate(rays, target, yaw, tilt, args):
    basis = ry(yaw) @ rx(tilt)
    lon, lat = angular_coordinates(rays[target], basis)
    # 所有目标点都必须被包住。max-lat 是硬约束，避免以严重畸变换小窗口。
    lower = np.floor((lat.min()-args.padding_deg)/args.boundary_step_deg)*args.boundary_step_deg
    upper = np.ceil((lat.max()+args.padding_deg)/args.boundary_step_deg)*args.boundary_step_deg
    row = dict(yaw_y_deg=float(yaw % 360), tilt_x_deg=float(tilt), feasible=False,
               latitude_min_deg=float(lower), latitude_max_deg=float(upper))
    if max(abs(lower), abs(upper)) > args.yang_max_lat:
        return row
    center, half = shortest_longitude_window(lon, args.padding_deg, args.boundary_step_deg)
    if 2*half >= args.max_yang_lon_span:
        return row
    row.update(feasible=True, longitude_center_deg=center, longitude_half_span_deg=half,
               # 等角采样在固定角分辨率下所需像素数与 Δλ*Δφ 成正比。
               angular_rectangle_deg2=float(2*half*(upper-lower)),
               solid_angle_sr=float(np.deg2rad(2*half)*(np.sin(np.deg2rad(upper))-np.sin(np.deg2rad(lower)))))
    return row


def row_bounds(row):
    return {k: row[k] for k in ['longitude_center_deg', 'longitude_half_span_deg',
                               'latitude_min_deg', 'latitude_max_deg']}


def candidate_order(row):
    return (row['angular_rectangle_deg2'], row['solid_angle_sr'], abs(row['tilt_x_deg']-90), row['yaw_y_deg'])


def search_rotation(rays, target, args):
    rows, seen = [], set()
    def evaluate(yaw, tilt, stage):
        key = (round(yaw % 360, 5), round(tilt, 5))
        if key in seen or not (args.tilt_min <= tilt <= args.tilt_max):
            return
        seen.add(key)
        row = fit_candidate(rays, target, *key, args)
        row['stage'] = stage
        rows.append(row)

    for yaw in np.arange(0., 360., args.coarse_step_deg):
        for tilt in np.arange(args.tilt_min, args.tilt_max+.001, args.coarse_step_deg):
            evaluate(yaw, tilt, 'coarse')
    feasible = sorted([r for r in rows if r['feasible']], key=candidate_order)
    if not feasible:
        raise ValueError('No feasible Yang: relax max latitude / longitude span, or Yin / seam settings')
    for seed in feasible[:args.refine_top]:
        for dyaw in np.arange(-args.coarse_step_deg, args.coarse_step_deg+.001, args.refine_step_deg):
            for dtilt in np.arange(-args.coarse_step_deg, args.coarse_step_deg+.001, args.refine_step_deg):
                evaluate(seed['yaw_y_deg']+dyaw, seed['tilt_x_deg']+dtilt, 'refine')
    seeds = sorted([r for r in rows if r['feasible']], key=candidate_order)[:args.refine_top]
    for seed in seeds:
        for dyaw in np.arange(-args.refine_step_deg, args.refine_step_deg+.001, .5):
            for dtilt in np.arange(-args.refine_step_deg, args.refine_step_deg+.001, .5):
                evaluate(seed['yaw_y_deg']+dyaw, seed['tilt_x_deg']+dtilt, 'fine')
    return rows


def full_resolution_selection(rows, rays_vis, common, area, lon, lat, args):
    yin_bounds, yin = fit_yin(lon, lat, common.any(0), args)
    target = common.any(0) & (~yin | (np.abs(lat) >= args.yin_lat_limit-args.seam_deg))
    # 粗网格可能遗漏 FOV 边界细节；在完整网格重新拟合，而非沿用粗边界。
    short = sorted([r for r in rows if r['feasible']], key=candidate_order)[:args.final_top]
    final = []
    support_weight = area * common.mean(0)
    for row in short:
        candidate = fit_candidate(rays_vis, target, row['yaw_y_deg'], row['tilt_x_deg'], args)
        candidate['stage'] = 'full_resolution_refit'
        if not candidate['feasible']:
            continue
        basis = ry(candidate['yaw_y_deg']) @ rx(candidate['tilt_x_deg'])
        yc_lon, yc_lat = angular_coordinates(rays_vis, basis)
        yang = window_mask(yc_lon, yc_lat, candidate)
        coverage = yin | yang
        candidate['uncovered_depth_direction_count'] = int(np.count_nonzero(common & ~coverage[None]))
        if candidate['uncovered_depth_direction_count']:
            continue
        kp = 1/np.maximum(np.cos(np.deg2rad(lat)), 1e-6)
        ky = 1/np.maximum(np.cos(np.deg2rad(yc_lat)), 1e-6)
        best = np.minimum(np.where(yin, kp, np.inf), np.where(yang, ky, np.inf))
        active = common.any(0)
        candidate['depth_weighted_mean_best_kappa'] = float(np.average(best[active], weights=support_weight[active]))
        final.append(candidate)
    if not final:
        raise ValueError('Full-resolution verification failed; increase final-top or relax constraints')
    winner = min(final, key=lambda r: (r['angular_rectangle_deg2'], r['depth_weighted_mean_best_kappa'],
                                      r['solid_angle_sr'], abs(r['tilt_x_deg']-90), r['yaw_y_deg']))
    basis = ry(winner['yaw_y_deg']) @ rx(winner['tilt_x_deg'])
    # 把最短包围弧的经度中心吸收到第三个旋转，使最终 Yang 经度边界以 0 为中心。
    centered_basis = basis @ ry(winner['longitude_center_deg'])
    final_bounds = row_bounds(winner)
    final_bounds['longitude_center_deg'] = 0.
    yang_lon, yang_lat = angular_coordinates(rays_vis, centered_basis)
    yang = window_mask(yang_lon, yang_lat, final_bounds)
    return winner, final, yin_bounds, final_bounds, centered_basis, yin, yang, yang_lat


def classify(common, yin, yang):
    code = np.zeros(common.shape, np.uint8)
    code[common & yin & ~yang] = 1
    code[common & yang & ~yin] = 2
    code[common & yin & yang] = 3
    code[common & ~(yin | yang)] = 4
    return code


def native_rays(bounds, basis, size):
    h, w = size
    longitude = np.linspace(bounds['longitude_center_deg']-bounds['longitude_half_span_deg'],
                            bounds['longitude_center_deg']+bounds['longitude_half_span_deg'], w)
    latitude = np.linspace(bounds['latitude_max_deg'], bounds['latitude_min_deg'], h)
    lon, lat = np.meshgrid(np.deg2rad(longitude), np.deg2rad(latitude))
    local = np.stack((np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)), -1)
    visual = local @ basis.T
    return visual @ OCAM_FROM_VIS.T, visual


def native_patch(out, name, bounds, basis, cameras, images, rotation, translation, axis, size):
    rays, visual = native_rays(bounds, basis, size)
    left, right, common = depth_visibility(cameras, rays, rotation, translation, axis)
    rgb, sampled_valid, left_uv = sample_rgb(cameras[0], images[0], rays.reshape(-1, 3).T, size)
    assert np.array_equal(left, sampled_valid)
    lon_step = 2*bounds['longitude_half_span_deg']/(size[1]-1)
    lat_step = (bounds['latitude_max_deg']-bounds['latitude_min_deg'])/(size[0]-1)
    native_lat = np.linspace(bounds['latitude_max_deg'], bounds['latitude_min_deg'], size[0])
    dx = np.broadcast_to(np.cos(np.deg2rad(native_lat))[:, None]*lon_step, size)
    dy = np.full(size, lat_step)
    pixel_anisotropy = np.maximum(dx, dy)/np.minimum(dx, dy)
    sampling = dict(shape_dhw=list(common.shape), longitude_step_deg=float(lon_step),
                    latitude_step_deg=float(lat_step), pixel_anisotropy_max=float(pixel_anisotropy.max()))
    Image.fromarray(rgb).save(out/f'{name}_reference_rgb.png')
    np.savez_compressed(out/f'{name}_depth_validity.npz', rays_reference=rays.astype(np.float32),
        left_valid=left, target_valid=right, valid_mask=common, inverse_depth_candidates=axis,
        latitude_min_max_deg=[bounds['latitude_min_deg'], bounds['latitude_max_deg']],
        longitude_center_half_span_deg=[bounds['longitude_center_deg'], bounds['longitude_half_span_deg']],
        basis_visual=basis, left_pixels_uv=left_uv.astype(np.float32),
        native_pixel_anisotropy=pixel_anisotropy.astype(np.float32))
    return dict(rays=rays, visual=visual, left=left, right=right, common=common, rgb=rgb,
                bounds=bounds, basis=basis, sampling=sampling)


def coverage_rows(common, yin, yang, area, primary_lat, yang_lat, axis):
    result = []
    kp = 1/np.maximum(np.cos(np.deg2rad(primary_lat)), 1e-6)
    ky = 1/np.maximum(np.cos(np.deg2rad(yang_lat)), 1e-6)
    best = np.minimum(np.where(yin, kp, np.inf), np.where(yang, ky, np.inf))
    for i, mask in enumerate(common):
        total = area[mask].sum()
        fraction = lambda x: float(area[mask & x].sum()/total) if total else 0.
        result.append(dict(index=i, inverse_depth_per_m=float(axis[i]), radius_m=float(1/axis[i]),
            common_sphere_fraction=float(total/area.sum()), yin_of_common=fraction(yin),
            yang_of_common=fraction(yang), overlap_of_common=fraction(yin & yang),
            uncovered_of_common=fraction(~(yin | yang)),
            high_primary_of_common=fraction(kp > 2), high_best_of_common=fraction(best > 2),
            primary_mean_kappa=weighted_summary(kp, mask, area)['mean'],
            best_mean_kappa=weighted_summary(best, mask, area)['mean']))
    return result


def sphere_map(ax, values, title, cmap=None, norm=None, vmin=None, vmax=None):
    image = ax.imshow(values, extent=[-180, 180, -90, 90], origin='upper', aspect='auto',
                      cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set(title=title, xlabel='Primary longitude (deg)', ylabel='Primary latitude (deg)',
           xticks=np.arange(-180, 181, 60), yticks=np.arange(-90, 91, 30))
    ax.grid(alpha=.18)
    return image


def plot_results(out, common, yin, yang, area, lat, yang_lat, candidates, winner, native, axis, rows):
    cmap = ListedColormap(COLORS/255)
    norm = BoundaryNorm(np.arange(-.5, 5), 5)
    union = common.any(0)
    fig, ax = plt.subplots(figsize=(11, 5.8), layout='constrained')
    sphere_map(ax, yin.astype(np.uint8)+2*yang.astype(np.uint8),
               'Geometric chart windows | independent of camera validity', cmap, norm)
    ax.legend(handles=[Patch(facecolor=COLORS[i]/255, label=name) for i, name in enumerate(
              ['Outside both charts', 'Yin only', 'Yang only', 'Yin + Yang'])],
              loc='upper center', bbox_to_anchor=(.5, -.18), ncol=4, frameon=False)
    save_figure(fig, out/'chart_windows.png')
    fig, ax = plt.subplots(figsize=(11, 5.8), layout='constrained')
    sphere_map(ax, classify(union, yin, yang), f'Yin / searched Yang | valid at ANY of {len(axis)} candidates', cmap, norm)
    ax.legend(handles=[Patch(facecolor=COLORS[i]/255, label=LABELS[i]) for i in range(5)],
              loc='upper center', bbox_to_anchor=(.5, -.18), ncol=3, frameon=False)
    save_figure(fig, out/'yin_yang_regions.png')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout='constrained')
    for ax, mask, title in zip(axes, [yin & union, yang & union, yin & yang & union],
                               ['Yin: valid union', 'Yang: valid union', 'Yin / Yang overlap']):
        sphere_map(ax, mask, title, 'Greys', vmin=0, vmax=1)
    save_figure(fig, out/'yin_yang_overlap_masks.png')

    chosen = sorted(set([0, len(axis)//4, len(axis)//2, 3*len(axis)//4, len(axis)-1]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for ax, idx in zip(axes.ravel(), chosen):
        radius = 'far limit' if idx == 0 else f'{1/axis[idx]:.2f} m'
        sphere_map(ax, classify(common[idx], yin, yang), f'idx {idx}: {radius}', cmap, norm)
    image = sphere_map(axes.ravel()[-1], common.mean(0), 'Fraction of candidates with valid projection',
                       'viridis', vmin=0, vmax=1)
    fig.colorbar(image, ax=axes.ravel()[-1], fraction=.035, label='Valid candidate fraction')
    save_figure(fig, out/'depth_validity_overview.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
    for ax, patch, name in zip(axes, native, ['Yin', 'Yang']):
        b = patch['bounds']
        extent = [b['longitude_center_deg']-b['longitude_half_span_deg'],
                  b['longitude_center_deg']+b['longitude_half_span_deg'], b['latitude_min_deg'], b['latitude_max_deg']]
        image = ax.imshow(patch['common'].mean(0), extent=extent, origin='upper', aspect='auto',
                          cmap='viridis', vmin=0, vmax=1)
        ax.set(title=f'{name} native grid | {patch["common"].shape}', xlabel=f'{name} longitude (deg)', ylabel=f'{name} latitude (deg)')
        ax.set_xticks(np.linspace(extent[0], extent[1], 5))
        ax.set_yticks(np.linspace(extent[2], extent[3], 5))
    fig.colorbar(image, ax=axes.tolist(), fraction=.025, label=f'Fraction of {len(axis)} valid candidates')
    save_figure(fig, out/'native_depth_validity.png')

    valid = [r for r in candidates if r['feasible']]
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    sc = ax.scatter([r['yaw_y_deg'] for r in valid], [r['tilt_x_deg'] for r in valid],
                    c=[r['angular_rectangle_deg2'] for r in valid], cmap='viridis', s=12, alpha=.75)
    ax.plot(winner['yaw_y_deg'], winner['tilt_x_deg'], marker='*', color='#D55E00', markersize=14,
            markeredgecolor='black', label='Selected after full-resolution refit', clip_on=False)
    ax.set(xlabel='Yaw about visual Y (deg)', ylabel='Tilt about X (deg)', title='Feasible searched orientations; each point has fitted boundaries')
    ax.set_xlim(0, 360); ax.set_xticks(np.arange(0, 361, 60))
    ax.set_ylim(min(r['tilt_x_deg'] for r in candidates), max(r['tilt_x_deg'] for r in candidates))
    ax.set_yticks(np.linspace(*ax.get_ylim(), 7))
    ax.legend(loc='lower left', frameon=False)
    fig.colorbar(sc, ax=ax, label='Yang angular rectangle (deg squared)')
    save_figure(fig, out/'rotation_search.png', vector=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    idx = np.arange(len(axis))
    for key, name, color, style in [('yin_of_common', 'Yin', '#0072B2', '-'), ('yang_of_common', 'Yang', '#E69F00', '--'),
                                   ('overlap_of_common', 'Chart overlap', '#9B51B9', ':')]:
        axes[0].plot(idx, [100*r[key] for r in rows], label=name, color=color, linestyle=style)
    axes[0].set(xlabel='Inverse-depth candidate index', ylabel='Area / common valid area (%)', ylim=(0, 100), title='Grid coverage across the unchanged depth axis')
    axes[0].set_xlim(0, len(axis)-1)
    axes[0].set_xticks(sorted(set([0, len(axis)//4, len(axis)//2, 3*len(axis)//4, len(axis)-1])))
    axes[0].legend(frameon=False)
    axes[1].plot(idx, [r['primary_mean_kappa'] for r in rows], label='Primary full ERP', color='#0072B2')
    axes[1].plot(idx, [r['best_mean_kappa'] for r in rows], label='Lower kappa in Yin / Yang', color='#D55E00', linestyle='--')
    axes[1].set(xlabel='Inverse-depth candidate index', ylabel='Area-weighted mean kappa', title='Only common valid directions are evaluated', ylim=(0, 3))
    axes[1].set_xlim(0, len(axis)-1)
    axes[1].set_xticks(axes[0].get_xticks())
    axes[1].legend(frameon=False)
    save_figure(fig, out/'coverage_and_distortion_vs_depth.png', vector=True)


def chart_lines(bounds, basis):
    lo = bounds['longitude_center_deg']-bounds['longitude_half_span_deg']
    hi = bounds['longitude_center_deg']+bounds['longitude_half_span_deg']
    bottom, top = bounds['latitude_min_deg'], bounds['latitude_max_deg']
    result = []
    for lat in np.unique(np.r_[bottom, np.arange(np.ceil(bottom/15)*15, top, 15), top]):
        lam = np.deg2rad(np.linspace(lo, hi, 180)); phi = np.deg2rad(lat)
        pts = np.stack((np.cos(phi)*np.sin(lam), np.full_like(lam, np.sin(phi)), np.cos(phi)*np.cos(lam)), 1)
        result.append(np.round(pts @ basis.T, 5).tolist())
    for lon in np.unique(np.r_[lo, np.arange(np.ceil(lo/30)*30, hi, 30), hi]):
        phi = np.deg2rad(np.linspace(bottom, top, 150)); lam = np.deg2rad(lon)
        pts = np.stack((np.cos(phi)*np.sin(lam), np.sin(phi), np.cos(phi)*np.cos(lam)), 1)
        result.append(np.round(pts @ basis.T, 5).tolist())
    return result


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Automatic local Yin / Yang search</title>
<style>body{font-family:system-ui;margin:20px;background:#f7f9fc;color:#162535}.box{background:white;padding:16px;border-radius:12px;margin-bottom:14px}
.controls{display:flex;gap:15px;flex-wrap:wrap}select{padding:6px}#sphere{height:650px}#erp{height:450px}.native{display:grid;grid-template-columns:1fr 1fr}.native>div{height:380px}
p{line-height:1.6}h1{font-size:24px}label{white-space:nowrap}</style><script>__PLOTLY__</script>
<h1>OmniThings cam4 + cam3 · 自动搜索局部 Yin / Yang</h1><div class="box">
<div class="controls"><label>深度 <select id="depth"><option value="any">任一候选有效（Union）</option><option value="all">全部候选有效（Intersection）</option></select></label>
<label>候选索引 <input id="slider" type="range" min="0" max="__MAXIDX__" value="48"></label>
<label>区域 <select id="view"><option value="both">Yin + Yang</option><option value="yin">Yin</option><option value="yang">Yang</option>
<option value="overlap">Yin/Yang 重叠</option><option value="uncovered">有效但未覆盖</option><option value="valid">左右共同有效</option></select></label>
<label>着色 <select id="style"><option value="region">网格归属</option><option value="rgb">左鱼眼真实 RGB</option>
<option value="fraction">深度有效比例</option><option value="kappa">有效网格择优 kappa</option></select></label>
<label><input id="grid" type="checkbox" checked>网格与边界</label></div>
<p id="info"></p><p>蓝色=仅 Yin，橙色=仅 Yang，紫色=两网格重叠，红色=共同有效但未被覆盖，灰色=无共同有效支持。</p>
<p>__PARAMETERS__</p><p>球面显示为单位射线；候选有效性使用真实米制距离及完整 R,t。Mask 只表示 FOV/投影/原始 invalid_mask 有效，不代表物体无遮挡。
Union 表示每个方向至少存在一层有效候选，Intersection 表示 __D__ 层全部有效；都不是实际深度 GT。</p></div>
<div id="sphere" class="box"></div><div id="erp" class="box"></div><div class="native box"><div id="yin"></div><div id="yang"></div></div>
<div class="box"><p>两张原生网格图与球面覆盖共享同一候选索引。此处 Yin、Yang 是两套参数化网格，不是两台相机。
左参考射线原点均为 cam4，右投影均进入 cam3。选中的 Yang 是当前有限粗到细搜索及硬约束下的最好结果，不是全局最优证明。</p></div>
<script>
const D=__DATA__,G=__GRIDS__,palette=['rgb(232,235,240)','rgb(0,114,178)','rgb(230,159,0)','rgb(155,81,185)','rgb(205,60,65)'];
const depth=document.getElementById('depth'),slider=document.getElementById('slider');
D.rho.forEach((rho,i)=>depth.add(new Option(`idx ${i} · ${i===0?'远场极限':(1/rho).toFixed(3)+' m'}`,''+i)));
function support(masks,i,key){if(key==='any')return masks.some(a=>a[i]);if(key==='all')return masks.every(a=>a[i]);return !!masks[+key][i];}
function select(code,v){return v==='yin'?code===1||code===3:v==='yang'?code===2||code===3:v==='overlap'?code===3:v==='uncovered'?code===4:code>0;}
function lines(segs,color,name){let x=[],y=[],z=[];segs.forEach(a=>{a.forEach(p=>{x.push(p[0]*1.01);y.push(p[1]*1.01);z.push(p[2]*1.01)});x.push(null);y.push(null);z.push(null)});
return {type:'scatter3d',mode:'lines',x,y,z,line:{color,width:2},name,hoverinfo:'skip'};}
function update(){const key=depth.value,v=document.getElementById('view').value,style=document.getElementById('style').value;
if(key!=='any'&&key!=='all')slider.value=key;
let x=[],y=[],z=[],colors=[],hover=[],raster=[],count=0,miss=0,ov=0;
for(let r=0;r<D.h;r++){let row=[];for(let c=0;c<D.w;c++){const i=r*D.w+c,valid=support(D.masks,i,key),code=valid?(D.yin[i]?(D.yang[i]?3:1):(D.yang[i]?2:4)):0;
if(valid)count++;if(code===4)miss++;if(code===3)ov++;const use=select(code,v);let val;
if(style==='fraction')val=D.fraction[i];else if(style==='kappa')val=D.best[i];else if(style==='rgb')val=D.rgb[i];else val=palette[code].match(/\d+/g).map(Number);
if(use){const p=D.rays[i];x.push(p[0]);y.push(p[1]);z.push(p[2]);colors.push(Array.isArray(val)?`rgb(${val})`:val);
hover.push(`primary lon ${D.lon[i].toFixed(1)}°, lat ${D.lat[i].toFixed(1)}°<br>grid code ${code}, valid fraction ${D.fraction[i].toFixed(2)}`);}
row.push(use?val:((style==='fraction'||style==='kappa')?null:[232,235,240]));}raster.push(row);}
const heat=style==='fraction'||style==='kappa',marker={size:3,color:colors};if(heat)Object.assign(marker,{colorscale:'Viridis',cmin:style==='fraction'?0:1,cmax:style==='fraction'?1:2,showscale:true,colorbar:{title:{text:style==='fraction'?'Valid fraction':'Best kappa'}}});
const traces=[{type:'scatter3d',mode:'markers',x,y,z,marker,text:hover,hoverinfo:'text',name:'Selected valid directions'}];
if(document.getElementById('grid').checked)traces.push(lines(G.yin,'#0072B2','Yin grid'),lines(G.yang,'#D55E00','Searched Yang grid'));
Plotly.react('sphere',traces,{margin:{t:20,b:15,l:5,r:5},uirevision:'sphere',legend:{orientation:'h'},scene:{aspectmode:'cube',
 xaxis:{title:{text:'Xvis: image-left'},range:[-1.08,1.08]},yaxis:{title:{text:'Yvis: up'},range:[-1.08,1.08]},zaxis:{title:{text:'Zvis: cam4 forward'},range:[-1.08,1.08]},
 camera:{up:{x:0,y:1,z:0},eye:{x:1.6,y:.7,z:1.6}}}},{responsive:true,scrollZoom:true});
const xy={x0:D.lon[0],dx:D.lon[1]-D.lon[0],y0:D.lat[0],dy:D.lat[D.w]-D.lat[0]};
const erp=heat?{type:'heatmap',z:raster,...xy,colorscale:'Viridis',zmin:style==='fraction'?0:1,zmax:style==='fraction'?1:2,
 colorbar:{title:{text:style==='fraction'?'Valid fraction':'Best kappa'}}}:{type:'image',z:raster,...xy};
Plotly.react('erp',[erp],{title:{text:'Same rays shown in primary coordinates'},margin:{t:40,b:50,l:65,r:65},
 xaxis:{title:{text:'Primary longitude (deg)'},range:[-180,180],dtick:60},yaxis:{title:{text:'Primary latitude (deg)'},range:[-90,90],dtick:30}},{responsive:true});
for(const name of ['yin','yang']){const p=D.native[name];let a=[];for(let r=0;r<p.h;r++){let row=[];for(let c=0;c<p.w;c++)row.push(support(p.masks,r*p.w+c,key)?1:0);a.push(row);}
Plotly.react(name,[{type:'heatmap',z:a,x0:p.x0,dx:p.dx,y0:p.y0,dy:p.dy,zmin:0,zmax:1,colorscale:[[0,'#e8ebf0'],[1,name==='yin'?'#0072B2':'#e69f00']],showscale:false}],
{title:{text:name.toUpperCase()+' native depth validity'},margin:{t:45,b:55,l:65,r:10},xaxis:{title:{text:name+' longitude (deg)'}},
 yaxis:{title:{text:name+' latitude (deg)'}}},{responsive:true});}
const note=key==='any'?'任一候选有效':key==='all'?'全部候选有效':`候选 ${key} · rho=${D.rho[+key].toPrecision(5)} 1/m · ${+key===0?'远场极限':(1/D.rho[+key]).toFixed(3)+' m'}`;
const exact=key==='any'?D.union_summary:key==='all'?D.intersection_summary:D.rows[+key];
document.getElementById('info').textContent=`${note}。按球面面积：共视/全球 ${(100*exact.common_sphere_fraction).toFixed(2)}%；重叠/共视 ${(100*exact.overlap_of_common).toFixed(2)}%；未覆盖/共视 ${(100*exact.uncovered_of_common).toFixed(4)}%。显示仅降采样，数值统计使用完整网格。`;
}
slider.max=D.rho.length-1;slider.value=Math.min(48,D.rho.length-1);slider.addEventListener('input',()=>{depth.value=slider.value;update()});
['depth','view','style','grid'].forEach(id=>document.getElementById(id).addEventListener('change',update));update();
</script></html>'''


def write_html(out, args, common, yin, yang, rays_vis, lon, lat, rgb, yang_lat, native, axis, rows, summary, bounds_yin, bounds_yang, basis_yang, winner):
    step = args.display_stride
    sel = np.s_[::step, ::step]
    small = lambda a: a[sel]
    flat = lambda a: np.round(small(a).ravel(), 5).tolist()
    kp = 1/np.maximum(np.cos(np.deg2rad(lat)), 1e-6)
    ky = 1/np.maximum(np.cos(np.deg2rad(yang_lat)), 1e-6)
    best = np.minimum(np.where(yin, kp, 1e3), np.where(yang, ky, 1e3))
    data = dict(h=small(lat).shape[0], w=small(lat).shape[1], rho=axis.tolist(),
        rays=np.round(small(rays_vis).reshape(-1, 3), 5).tolist(), lon=flat(lon), lat=flat(lat),
        rgb=small(rgb).reshape(-1, 3).tolist(), yin=small(yin).astype(int).ravel().tolist(),
        yang=small(yang).astype(int).ravel().tolist(), masks=common[:, ::step, ::step].reshape(len(axis), -1).astype(int).tolist(),
        fraction=flat(common.mean(0)), best=flat(best), rows=rows, union_summary=summary['union'],
        intersection_summary=summary['intersection'], native={})
    for name, p in zip(['yin', 'yang'], native):
        h, w = p['left'][::2, ::2].shape
        b = p['bounds']; full_h, full_w = p['left'].shape
        data['native'][name] = dict(h=h, w=w, masks=p['common'][:, ::2, ::2].reshape(len(axis), -1).astype(int).tolist(),
            x0=b['longitude_center_deg']-b['longitude_half_span_deg'], dx=2*b['longitude_half_span_deg']/(full_w-1)*2,
            y0=b['latitude_max_deg'], dy=-(b['latitude_max_deg']-b['latitude_min_deg'])/(full_h-1)*2)
    grids = dict(yin=chart_lines(bounds_yin, np.eye(3)), yang=chart_lines(bounds_yang, basis_yang))
    params = (f"Yang 旋转 R=Ry({winner['yaw_y_deg']:.2f}°) Rx({winner['tilt_x_deg']:.2f}°) "
              f"Ry({winner['longitude_center_deg']:.2f}°)；Yang 经度 ±{bounds_yang['longitude_half_span_deg']:.2f}°，"
              f"纬度 [{bounds_yang['latitude_min_deg']:.2f}°, {bounds_yang['latitude_max_deg']:.2f}°]。"
              f"Yin 纬度 ±{args.yin_lat_limit:g}°；接缝目标带 {args.seam_deg:g}°；边界余量 {args.padding_deg:g}°。")
    substitutions = {'__PLOTLY__': get_plotlyjs(), '__DATA__': json.dumps(data, separators=(',', ':')), '__D__': str(len(axis)),
        '__GRIDS__': json.dumps(grids, separators=(',', ':')), '__PARAMETERS__': params, '__MAXIDX__': str(len(axis)-1)}
    import re
    (out/'yin_yang_search.html').write_text(re.sub(r'__[A-Z]+__', lambda m: substitutions[m.group()], HTML))


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parent/'outputs/input_snapshot')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'outputs_yang_search')
    parser.add_argument('--frame-id', type=int, default=4887)
    parser.add_argument('--ref-cam', type=int, default=4)
    parser.add_argument('--target-cam', type=int, default=3)
    parser.add_argument('--num-depth', type=int, default=96)
    parser.add_argument('--full-num-depth', type=int, default=192)
    parser.add_argument('--search-height', type=int, default=90)
    parser.add_argument('--search-width', type=int, default=180)
    parser.add_argument('--sphere-height', type=int, default=360)
    parser.add_argument('--sphere-width', type=int, default=720)
    parser.add_argument('--native-size', type=int, nargs=2, help='Optional fixed H W; default uses nearly equal angular steps')
    parser.add_argument('--native-angular-step-deg', type=float, default=.75)
    parser.add_argument('--yin-lat-limit', type=float, default=60.)
    parser.add_argument('--seam-deg', type=float, default=5.)
    parser.add_argument('--padding-deg', type=float, default=2.)
    parser.add_argument('--boundary-step-deg', type=float, default=.5)
    parser.add_argument('--yang-max-lat', type=float, default=45.)
    parser.add_argument('--max-yang-lon-span', type=float, default=355.)
    parser.add_argument('--tilt-min', type=float, default=45.)
    parser.add_argument('--tilt-max', type=float, default=135.)
    parser.add_argument('--coarse-step-deg', type=float, default=15.)
    parser.add_argument('--refine-step-deg', type=float, default=2.)
    parser.add_argument('--refine-top', type=int, default=6)
    parser.add_argument('--final-top', type=int, default=48)
    parser.add_argument('--display-stride', type=int, default=4)
    args = parser.parse_args()
    if not (0 < args.seam_deg < args.yin_lat_limit < 90 and 0 < args.yang_max_lat < 90):
        parser.error('Require 0 < seam < Yin latitude <90 and 0<Yang max latitude<90')
    if min(args.coarse_step_deg, args.refine_step_deg, args.boundary_step_deg, args.padding_deg) <= 0:
        parser.error('Search steps and padding must be positive')
    if args.display_stride < 1 or min(args.sphere_height, args.sphere_width)//args.display_stride < 2:
        parser.error('Display must retain at least 2 samples on each axis')
    if ((args.native_size is not None and min(args.native_size) < 2) or args.native_angular_step_deg <= 0
            or args.refine_top < 1 or args.final_top < 1):
        parser.error('Invalid native size / number of candidates')
    out = args.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_STYLE); logging.getLogger('fontTools').setLevel(logging.WARNING)
    cameras, images, config, provenance = load_pair(args.data_root, args.ref_cam, args.target_cam, args.frame_id)
    rotation, translation, geometry = pair_geometry(*cameras)
    axis = depth_axis(config, args.num_depth, args.full_num_depth)
    print(f'Actual axis: D={len(axis)}, far={1/axis[0]:.9g}m, nearest={1/axis[-1]:.9f}m', flush=True)

    rays_s, visual_s, lon_s, lat_s, _ = spherical_rays(args.search_height, args.search_width)
    _, _, common_s = depth_visibility(cameras, rays_s, rotation, translation, axis)
    _, yin_s = fit_yin(np.rad2deg(lon_s), np.rad2deg(lat_s), common_s.any(0), args)
    target_s = common_s.any(0) & (~yin_s | (np.abs(np.rad2deg(lat_s)) >= args.yin_lat_limit-args.seam_deg))
    print('Searching orientation and fitted rectangular boundaries...', flush=True)
    candidates = search_rotation(visual_s, target_s, args)
    print(f'Searched {len(candidates)} orientations, {sum(r["feasible"] for r in candidates)} feasible', flush=True)
    rays, visual, lon_rad, lat_rad, area = spherical_rays(args.sphere_height, args.sphere_width)
    lon, lat = np.rad2deg(lon_rad), np.rad2deg(lat_rad)
    left, right, common = depth_visibility(cameras, rays, rotation, translation, axis)
    winner, final, byin, byang, basis, yin, yang, yang_lat = full_resolution_selection(
        candidates, visual, common, area, lon, lat, args)
    print('Selected:', json.dumps(winner, indent=2), flush=True)
    rows = coverage_rows(common, yin, yang, area, lat, yang_lat, axis)
    summary = {}
    for name, mask in [('union', common.any(0)), ('intersection', common.all(0))]:
        total = area[mask].sum()
        summary[name] = dict(common_sphere_fraction=float(total/area.sum()),
            overlap_of_common=float(area[mask & yin & yang].sum()/total) if total else 0.,
            uncovered_of_common=float(area[mask & ~(yin | yang)].sum()/total) if total else 0.)
    native = []
    for name, bounds, transform in [('yin', byin, np.eye(3)), ('yang', byang, basis)]:
        # 窗口长宽比不同，默认分别导出近等角步长网格。强行给相同 H/W 会改变
        # 经/纬角步长比，1/cos(latitude) 便不再代表实际单像素的全部各向异性。
        size = tuple(args.native_size) if args.native_size else (
            int(np.ceil((bounds['latitude_max_deg']-bounds['latitude_min_deg'])/args.native_angular_step_deg))+1,
            int(np.ceil(2*bounds['longitude_half_span_deg']/args.native_angular_step_deg))+1)
        native.append(native_patch(out, name, bounds, transform, cameras, images, rotation, translation, axis, size))
    rgb, _, _ = sample_rgb(cameras[0], images[0], rays.reshape(-1, 3).T, rays.shape[:2])
    # 验证两个坐标网格只是重参数化；同方向仍对应同一三维点与相机投影。
    reconstructed = (visual @ basis) @ basis.T
    roundtrip = float(np.max(np.abs(reconstructed-visual)))
    assert roundtrip < 1e-12 and np.isclose(np.linalg.det(basis), 1.)
    # 用两套经纬角重新生成同一射线，再比较真实左右投影，避免只检查矩阵乘法。
    test = (common.any(0) & yin & yang).ravel()
    same_ray = visual.reshape(-1, 3)[test][::17]
    restored = []
    for transform in [np.eye(3), basis]:
        az, el = angular_coordinates(same_ray, transform)
        az, el = np.deg2rad(az), np.deg2rad(el)
        local = np.stack((np.cos(el)*np.sin(az), np.sin(el), np.cos(el)*np.cos(az)), -1)
        restored.append(local @ transform.T @ OCAM_FROM_VIS.T)
    ray_agreement = float(np.max(np.abs(restored[0]-restored[1])))
    reprojection_error = 0.
    for i in [0, len(axis)//2, len(axis)-1]:
        for cam_index in [0, 1]:
            projected = []
            for ray in restored:
                points = ray.T / axis[i]
                if cam_index:
                    points = rotation @ points + translation[:, None]
                uv, valid = project_camera(cameras[cam_index], points)
                projected.append((uv, valid))
            both = projected[0][1] & projected[1][1]
            assert np.array_equal(projected[0][1], projected[1][1])
            if both.any():
                reprojection_error = max(reprojection_error, float(np.max(np.abs(
                    projected[0][0][:, both]-projected[1][0][:, both]))))
    assert ray_agreement < 1e-12 and reprojection_error < 1e-8
    assert not np.any(common & ~(yin | yang)[None])
    seam = common.any(0) & (np.abs(lat) >= args.yin_lat_limit-args.seam_deg) & (np.abs(lat) <= args.yin_lat_limit)
    assert np.all((yin & yang)[seam]), 'The valid seam band must be shared by both charts'
    for patch in native:
        assert not np.any(patch['common'] & ~patch['left'][None])
        assert np.array_equal(patch['common'], patch['right'] & patch['left'][None])
        assert np.all(patch['rgb'][~patch['left']] == 0)
    np.savez_compressed(out/'reference_depth_masks.npz', inverse_depth_candidates=axis,
        rays_reference=rays.astype(np.float32), area_sr=area, left_valid=left,
        target_valid=right, common_valid=common, yin=yin, yang=yang, chart_overlap=yin & yang,
        basis_yang_visual=basis, yin_bounds=np.array(list(byin.values())), yang_bounds=np.array(list(byang.values())),
        primary_latitude_deg=lat.astype(np.float32), yang_latitude_deg=yang_lat.astype(np.float32))
    result = dict(geometry=geometry, provenance=provenance,
        config_sha256=hashlib.sha256((args.data_root.expanduser()/'config.yaml').read_bytes()).hexdigest(),
        frame_id=args.frame_id, inverse_depth_candidates=axis.tolist(),
        depth_axis_note='Original 192 candidates using EPS and min_depth; [::192//96], identical candidate units',
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        objective='Cover every sampled valid depth-direction pair plus a seam band; enforce |Yang latitude|<=limit; '
                  'minimize angular rectangle area, break ties with depth/area weighted kappa',
        search_count=len(candidates), feasible_count=sum(r['feasible'] for r in candidates),
        full_resolution_candidates=len(final), selected=winner, yin_bounds=byin, yang_bounds=byang,
        yang_basis_visual=basis.tolist(), yang_basis_reference=(OCAM_FROM_VIS @ basis).tolist(),
        native_sampling={name: patch['sampling'] for name, patch in zip(['yin', 'yang'], native)},
        yang_rotation_convention='R_visual_from_yang=Ry(yaw_y_deg) @ Rx(tilt_x_deg) @ Ry(longitude_center_deg from selected)',
        summary=summary, depth_rows=rows,
        validation=dict(status='PASS', direction_roundtrip_max=roundtrip, rotation_determinant=float(np.linalg.det(basis)),
            overlap_ray_agreement_max=ray_agreement, overlap_reprojection_max_px=reprojection_error,
            uncovered_valid_depth_direction_count=int(np.count_nonzero(common & ~(yin | yang)[None])),
            valid_seam_directions=int(np.count_nonzero(seam)), all_valid_seam_in_overlap=True,
            native_mask_shapes=[list(p['common'].shape) for p in native]),
        limitations=['Finite discrete orientation search is not a global optimum proof',
            'Coverage guarantee is on sampled directions and the configured candidate axis, not a continuous proof',
            'Masks are FOV/projection validity, not GT occlusion or true surface depth',
            'Uniform candidate averaging is a diagnostic convention, not a learned depth prior',
            'Coordinate kappa assumes equal angular spacing; NPZ also records actual native pixel anisotropy',
            'No neural model, training, GT conversion or depth fusion is changed'])
    (out/'search_result.json').write_text(json.dumps(result, indent=2)+'\n')
    save_csv(out/'search_candidates.csv', candidates)
    save_csv(out/'full_resolution_candidates.csv', final)
    save_csv(out/'depth_coverage.csv', rows)
    print('Rendering masks and offline browser...', flush=True)
    plot_results(out, common, yin, yang, area, lat, yang_lat, candidates, winner, native, axis, rows)
    write_html(out, args, common, yin, yang, visual, lon, lat, rgb, yang_lat, native, axis, rows, summary,
               byin, byang, basis, winner)
    (out/'figure_qa.json').write_text(json.dumps(QA, indent=2))
    lines = ['# 自动搜索 Yin / Yang 几何结果', '',
        f'真实 cam{args.ref_cam}/cam{args.target_cam}，第 {args.frame_id} 帧；不读取 GT、不运行网络。', '',
        '## 搜索定义', '', result['objective'], '',
        f'粗到细搜索 {len(candidates)} 个旋转，{result["feasible_count"]} 个满足约束；'
        f'完整 {args.sphere_height}×{args.sphere_width} 网格重新拟合 {len(final)} 个候选。', '',
        '## 最终参数', '', '```json', json.dumps(dict(selected=winner, yin=byin, yang=byang,
            yang_basis_visual=basis.tolist()), indent=2), '```', '',
        f'候选轴：D={len(axis)}；最近距离 {1/axis[-1]:.9f} m；'
        '第 0 层为原工程 EPS 定义的远场极限，不是新增一层不同单位的索引。', '',
        f'所有 {len(axis)} 层、完整分析网格上的有效候选均被 Yin/Yang 覆盖，未覆盖计数为 0。',
        f'有效接缝带方向数 {int(seam.sum())}，全部位于两网格重叠。', '',
        '默认原生 Mask 使用近似相等的经/纬角步长；两个窗口分别决定 H/W。'
        '可用 --native-size H W 导出固定尺寸，但其像素各向异性需额外考虑角步长比。',
        '```json', json.dumps(result['native_sampling'], indent=2), '```', '',
        '## 各深度统计（球面面积加权）', '',
        '| idx | 径向距离 m | 共视/全球 | 两网格重叠/共视 | 未覆盖/共视 | 原 ERP 平均 κ | 两网格择优平均 κ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for i in sorted(set([0, len(axis)//4, len(axis)//2, 3*len(axis)//4, len(axis)-1])):
        r=rows[i]; distance='远场极限' if i == 0 else f'{r["radius_m"]:.4f}'
        lines.append(f'|{i}|{distance}|{100*r["common_sphere_fraction"]:.3f}%|{100*r["overlap_of_common"]:.3f}%|'
                     f'{100*r["uncovered_of_common"]:.4f}%|{r["primary_mean_kappa"]:.4f}|{r["best_mean_kappa"]:.4f}|')
    lines += ['', '## 查看文件', '', '- `yin_yang_search.html`：离线 3D、ERP、两张原生网格；逐候选切换 Mask。',
        '- `yin_yang_regions.png`、`yin_yang_overlap_masks.png`：网格归属与重叠。',
        '- `depth_validity_overview.png`、`native_depth_validity.png`：不同深度及原生网格 Mask。',
        '- `reference_depth_masks.npz`：完整参考球面上的逐候选左右有效性、共同有效性与两网格 Mask。',
        '- `yin_depth_validity.npz` / `yang_depth_validity.npz`：原生网格 [D,H,W] Mask、射线、深度轴、基矩阵与左采样像素。', '',
        '## 限制', '', *['- '+x for x in result['limitations']], '',
        'Yin/Yang 是两套网格；cam4/cam3 是两台相机。网格重叠与双目共视是两种不同的交集，不能混淆。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('DONE:', out, flush=True)


if __name__ == '__main__':
    main()
