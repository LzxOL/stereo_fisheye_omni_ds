"""冻结 Yin/Yang 的 GT 离线诊断；绝不将 GT 用于网格或推理。

OmniThings 当前副本只有 Rig-centered ERP inverse-depth TIFF。这里恢复其
已观测点云，检查 cam4 径向候选范围，以及同一稀疏点云的双相机 Z-buffer
可见性估计。没有独立 cam3 深度 GT 时，不能声称获得完整 visibility truth。
"""
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
import tifffile

from camera_geometry import (load_pair, pair_geometry, spherical_rays, OCAM_FROM_VIS,
                             project_camera)
from search_yang_grid import depth_axis, depth_visibility, angular_coordinates, window_mask, native_rays
from visualize_stereo_sphere import PLOT_STYLE, QA, save_figure
from dataset import Dataset, makeSphericalRays
from dataset_stereo import cloud_visibility
from utils.geometry import applyTransform
from utils.image import readImageFloat
from utils.common import Edict


def ratio(numerator, denominator, weights=None):
    if weights is None:
        total = int(np.count_nonzero(denominator))
        return float(np.count_nonzero(numerator & denominator)/total) if total else None
    total = float(weights[denominator].sum())
    return float(weights[numerator & denominator].sum()/total) if total else None


def rig_cell_area(shape):
    """与 makeSphericalRays(...,90) 的含端点纬度一致的源 Rig 单元面积。

    这是原始 Rig 方向的面积权重，不是 cam4 光心下重参数化后的球面面积。
    后者需要独立重投影栅格，不能通过平移点云后直接沿用此权重冒充。
    """
    h, w = shape
    phi = np.linspace(-np.pi/2, np.pi/2, h)
    edges = np.r_[-np.pi/2, (phi[:-1]+phi[1:])/2, np.pi/2]
    area = np.repeat(((np.sin(edges[1:])-np.sin(edges[:-1]))*2*np.pi/w)[:, None], w, 1)
    assert np.isclose(area.sum(), 4*np.pi)
    return area


def range_statistics(frame, region, selected, ref_distance, raw_rig_distance, area, axis):
    nearest, farthest = 1/axis[-1], 1/axis[0]
    too_near = ref_distance < nearest
    too_far = ref_distance > farthest
    outside = too_near | too_far
    values = ref_distance[selected]
    return dict(frame=frame, region=region, source_point_count=int(selected.sum()),
        reference_radius_min_m=float(values.min()) if len(values) else None,
        reference_radius_median_m=float(np.median(values)) if len(values) else None,
        reference_radius_max_m=float(values.max()) if len(values) else None,
        below_nearest_count=int(np.count_nonzero(selected & too_near)),
        above_farthest_count=int(np.count_nonzero(selected & too_far)),
        outside_candidate_count=int(np.count_nonzero(selected & outside)),
        below_nearest_point_fraction=ratio(too_near, selected),
        outside_candidate_point_fraction=ratio(outside, selected),
        below_nearest_source_rig_area_fraction=ratio(too_near, selected, area),
        outside_candidate_source_rig_area_fraction=ratio(outside, selected, area),
        raw_rig_depth_below_nearest_point_fraction=ratio(raw_rig_distance < nearest, selected),
        denominator='valid raw Rig GT samples satisfying this region; counts are source samples, not dense camera pixels',
        radius_quantity='norm(P_cam4); raw_rig_depth_below_nearest uses original norm(P_rig) separately')


def depth_buffer(camera, points, valid):
    pixels, _ = project_camera(camera, points)
    ids = np.flatnonzero(valid)
    uv = pixels[:, ids]
    pixel_id = np.rint(uv[1]).astype(int)*camera.width+np.rint(uv[0]).astype(int)
    result = np.full(camera.height*camera.width, np.inf)
    np.minimum.at(result, pixel_id, np.linalg.norm(points[:, ids], axis=0))
    return result.reshape(camera.height, camera.width)


def plot_frame(out, frame, shape, raw_rho, raw_valid, radius, geometric, visible, yin, yang, axis,
               images, cam4_depth, cam3_depth):
    def field(values, fill=np.nan):
        arr = np.full(np.prod(shape), fill, dtype=np.float64)
        arr[raw_valid.ravel()] = values
        return arr.reshape(shape)
    extent = None
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), layout='constrained')
    entries = [(np.ma.masked_where(~raw_valid, raw_rho), 'Stored Rig inverse depth\n(float TIFF page)', 'viridis', 0, None),
        (field(radius), 'cam4 radius at source Rig samples\n(display clipped at 20 m)', 'viridis', 0, 20),
        (field(radius < 1/axis[-1], fill=0), 'cam4 radius below\nnearest candidate', 'Greys', 0, 1),
        (field(geometric, fill=0), 'Both cameras:\ngeometric projection valid', 'Greys', 0, 1),
        (field(visible, fill=0), 'Both cameras: Rig-cloud\nestimated visibility', 'Greys', 0, 1)]
    for ax, (value, title, cmap, lo, hi) in zip(axes.ravel(), entries):
        image = ax.imshow(value, origin='upper', cmap=cmap, vmin=lo, vmax=hi, aspect='auto')
        ax.set(title=title, xlabel='Source Rig GT column', ylabel='Source Rig GT row')
        if cmap != 'Greys':
            fig.colorbar(image, ax=ax, fraction=.035, label='1/m' if 'inverse depth' in title else 'm', extend='max' if hi else 'neither')
    code = np.zeros(len(radius), dtype=np.uint8)
    code[geometric & yin & ~yang] = 1
    code[geometric & yang & ~yin] = 2
    code[geometric & yin & yang] = 3
    image = axes.ravel()[-1].imshow(field(code, fill=0), origin='upper', aspect='auto',
        cmap=ListedColormap(['#e8ebf0', '#0072B2', '#E69F00', '#9B51B9']), norm=BoundaryNorm(np.arange(-.5,4),4))
    axes.ravel()[-1].set(title='Frozen chart membership\nat geometric GT samples', xlabel='Source Rig GT column', ylabel='Source Rig GT row')
    for ax in axes.ravel():
        ax.set_xlim(0, shape[1]-1); ax.set_ylim(shape[0]-1, 0)
        ax.set_xticks(np.linspace(0, shape[1]-1, 5)); ax.set_yticks(np.linspace(0, shape[0]-1, 5))
    fig.suptitle(f'Frame {frame}: offline diagnostics on original Rig GT raster (not cam4 ERP)')
    save_figure(fig, out/f'gt_samples_{frame:05d}.png')
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5), layout='constrained')
    for col, (img, dep, name) in enumerate(zip(images, [cam4_depth, cam3_depth], ['cam4', 'cam3'])):
        axes[0, col].imshow(img); axes[0, col].set(title=f'{name} original RGB')
        d = np.ma.masked_where(~np.isfinite(dep), dep)
        image = axes[1, col].imshow(d, vmin=0, vmax=20, cmap='viridis')
        axes[1, col].set_facecolor('#e8ebf0')
        axes[1, col].set(title=f'{name}: sparse Rig-cloud\nradial depth estimate')
        fig.colorbar(image, ax=axes[1, col], fraction=.035, label='m', extend='max')
    for ax in axes.ravel():
        ax.set(xlabel='Original fisheye column', ylabel='Original fisheye row')
        ax.set_xlim(0, images[0].shape[1]-1); ax.set_ylim(images[0].shape[0]-1, 0)
        ax.set_xticks(np.linspace(0, images[0].shape[1]-1, 5)); ax.set_yticks(np.linspace(0, images[0].shape[0]-1, 5))
    save_figure(fig, out/f'camera_cloud_depth_{frame:05d}.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[4740, 4886, 4887, 4888, 5000])
    parser.add_argument('--gt-width', type=int, default=640)
    args = parser.parse_args()
    root, out = args.data_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_STYLE); logging.getLogger('fontTools').setLevel(logging.WARNING)
    selection_bytes = args.selection.read_bytes()
    selected = json.loads(selection_bytes)
    axis = np.array(selected['inverse_depth_candidates'])
    byin, byang = selected['yin_bounds'], selected['yang_bounds']
    basis = np.array(selected['yang_basis_visual'])
    ref_id = selected['geometry']['reference_id']; target_id = selected['geometry']['target_id']
    # 只读地核查目录与真实 TIFF 页面；不寻找/下载额外数据，也不自动写 Mask。
    all_tiffs = list(root.rglob('*.tiff'))+list(root.rglob('*.tif'))
    parent_counts = {}
    for p in all_tiffs:
        name = str(p.parent.relative_to(root))
        parent_counts[name] = parent_counts.get(name, 0)+1
    independent_candidates = [str(p.relative_to(root)) for p in all_tiffs
        if any(token in str(p.relative_to(root)).lower() for token in ['cam3', 'cam4'])]
    if independent_candidates:
        raise RuntimeError('Potential camera GT found; inspect its convention before claiming visibility: '+str(independent_candidates[:10]))

    cameras, images, config, provenance = load_pair(root, ref_id, target_id, args.frames[0])
    assert np.array_equal(depth_axis(config, len(axis)), axis)
    # 与真正训练 Dataset 的原始 192 轴比对，不实例化新的双目 GT 转换路径。
    db = Dataset('omnithings', Edict(use_rgb=True), load_lut=False, db_root=str(root.parent))
    original_axis = db.invdepths[::len(db.invdepths)//len(axis)][:len(axis)]
    assert np.array_equal(original_axis, axis)
    if db.gt_phi != 90:
        raise ValueError(f'Unexpected actual gt_phi: {db.gt_phi}')
    geometry_rays, _, _, _, _ = spherical_rays(45, 90)
    native_hashes = {}
    for name, bounds, transform in [('yin', byin, np.eye(3)), ('yang', byang, basis)]:
        size = tuple(selected['native_sampling'][name]['shape_dhw'][1:])
        rays, _ = native_rays(bounds, transform, size)
        native_hashes[name] = hashlib.sha256(rays.tobytes()).hexdigest()

    rows, frame_results, mask_hashes = [], [], []
    for frame in args.frames:
        cameras, images, config, prov = load_pair(root, ref_id, target_id, frame)
        rotation, translation, report = pair_geometry(*cameras)
        gt_path = root/f'omnidepth_gt_{args.gt_width}/{frame:05d}.tiff'
        with tifffile.TiffFile(gt_path) as file:
            pages = [dict(shape=list(p.shape), dtype=str(p.dtype)) for p in file.pages]
        rho = readImageFloat(str(gt_path))
        if not isinstance(rho, np.ndarray) or rho.ndim != 2 or not np.issubdtype(rho.dtype, np.floating):
            raise ValueError('Expected float inverse-depth page, not RGB thumbnail')
        good = np.isfinite(rho) & (rho > 0)
        rays = makeSphericalRays(rho.shape, db.gt_phi)
        rig_distance = 1/rho[good].astype(np.float64)
        points_rig = rays[:, good.ravel()]*rig_distance[None]
        points_ref = applyTransform(cameras[0].rig2cam, points_rig)
        points_target = applyTransform(cameras[1].rig2cam, points_rig)
        radius = np.linalg.norm(points_ref, axis=0)
        uv_ref, geom_ref = project_camera(cameras[0], points_ref)
        uv_target, geom_target = project_camera(cameras[1], points_target)
        geom_both = geom_ref & geom_target
        # 与独立两级 Rig 变换比较：只检验代数/投影一致性，不把它当相机 GT 精度。
        direct_target = rotation @ points_ref+translation[:, None]
        point_error = float(np.max(np.abs(points_target-direct_target)))
        relative_point_error = float(np.max(np.linalg.norm(points_target-direct_target, axis=0)/
                                            np.maximum(np.linalg.norm(points_target, axis=0), 1.)))
        direct_uv, direct_valid = project_camera(cameras[1], direct_target)
        uv_error = float(np.max(np.abs(uv_target[:, geom_target]-direct_uv[:, geom_target])))
        # 存储数据含约 3e10 m 的远场值，绝对坐标舍入可达微米；按点距离归一化
        # 检查变换，同时独立报告实际像素重投影误差，不能误把远场舍入当标定错误。
        assert np.array_equal(geom_target, direct_valid) and relative_point_error < 1e-12 and uv_error < 1e-7
        ref_visible_est = cloud_visibility(cameras[0], points_ref)
        target_visible_est = cloud_visibility(cameras[1], points_target)
        visible_est = geom_both & ref_visible_est & target_visible_est
        # 没有独立 per-camera GT，全部几何有效点的“真实 visibility 状态”仍未知。
        independent_visibility_known = np.zeros(len(radius), dtype=bool)
        direction_vis = (points_ref/radius[None]).T @ OCAM_FROM_VIS
        lon, lat = angular_coordinates(direction_vis, np.eye(3))
        ylon, ylat = angular_coordinates(direction_vis, basis)
        yin = window_mask(lon, lat, byin); yang = window_mask(ylon, ylat, byang)
        weights = rig_cell_area(rho.shape)[good]
        sets = dict(raw_rig_valid=np.ones(len(radius), bool), cam4_geometric=geom_ref,
            binocular_geometric=geom_both, binocular_rigcloud_estimated_visible=visible_est,
            binocular_geometric_finite_under_1000m=geom_both & (rig_distance <= 1000),
            yin_geometric=geom_both & yin, yang_geometric=geom_both & yang,
            overlap_geometric=geom_both & yin & yang,
            yin_estimated_visible=visible_est & yin, yang_estimated_visible=visible_est & yang,
            overlap_estimated_visible=visible_est & yin & yang)
        rows.extend(range_statistics(frame, name, mask, radius, rig_distance, weights, axis)
                    for name, mask in sets.items())
        visibility_regions = {}
        for name, mask in [('all', np.ones(len(radius), bool)), ('yin', yin), ('yang', yang), ('overlap', yin & yang)]:
            denominator = geom_both & mask
            visibility_regions[name] = dict(geometric_point_count=int(denominator.sum()),
                estimated_visible_point_count=int(np.count_nonzero(visible_est & mask)),
                estimated_visible_fraction_of_geometric=ratio(visible_est, denominator),
                independent_true_visibility_known_count=0,
                independent_true_visibility_unknown_count=int(denominator.sum()))
        _, _, geometry_mask = depth_visibility(cameras, geometry_rays, rotation, translation, axis)
        mask_hash = hashlib.sha256(geometry_mask.tobytes()).hexdigest(); mask_hashes.append(mask_hash)
        ref_buffer = depth_buffer(cameras[0], points_ref, geom_ref)
        target_buffer = depth_buffer(cameras[1], points_target, geom_target)
        observed_pixels = np.isfinite(ref_buffer) & (ref_buffer > 0)
        below_pixels = observed_pixels & (ref_buffer < 1/axis[-1])
        outside_pixels = observed_pixels & ((ref_buffer < 1/axis[-1]) | (ref_buffer > 1/axis[0]))
        cam4_pixel_statistics = dict(observed_sparse_rig_gt_pixel_count=int(observed_pixels.sum()),
            below_nearest_pixel_count=int(below_pixels.sum()), outside_candidates_pixel_count=int(outside_pixels.sum()),
            below_nearest_pixel_fraction=float(below_pixels.sum()/observed_pixels.sum()) if observed_pixels.any() else None,
            outside_candidates_pixel_fraction=float(outside_pixels.sum()/observed_pixels.sum()) if observed_pixels.any() else None,
            unknown_pixel_count=int((~observed_pixels).sum()),
            denominator='cam4 original image pixels with nearest Rig-cloud GT support; sparse, no interpolation or hole filling')
        np.savez_compressed(out/f'gt_diagnostics_{frame:05d}.npz', raw_inverse_depth=rho,
            raw_valid=good, source_point_ids=np.flatnonzero(good.ravel()), points_rig=points_rig,
            points_cam4=points_ref, points_cam3=points_target, cam4_radial_depth_m=radius,
            geometric_cam4=geom_ref, geometric_cam3=geom_target, geometric_valid=geom_both,
            estimated_visible_cam4=ref_visible_est, estimated_visible_cam3=target_visible_est,
            estimated_visible_both=visible_est, independent_true_visibility_known=independent_visibility_known,
            yin=yin, yang=yang, overlap=yin & yang, inverse_depth_candidates=axis,
            cam4_cloud_depth=ref_buffer, cam3_cloud_depth=target_buffer, source_rig_area_sr=weights,
            candidate_geometry_test_mask=geometry_mask)
        plot_frame(out, frame, rho.shape, rho, good, radius, geom_both, visible_est, yin, yang, axis,
                   images, ref_buffer, target_buffer)
        frame_results.append(dict(frame_id=frame, raw_gt_path=str(gt_path), tiff_pages=pages,
            raw_gt_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(), float_page_shape=list(rho.shape),
            float_page_dtype=str(rho.dtype), raw_valid_count=int(good.sum()), invalid_count=int((~good).sum()),
            inverse_depth_min_per_m=float(rho[good].min()), inverse_depth_max_per_m=float(rho[good].max()),
            raw_rig_depth_min_m=float(rig_distance.min()), raw_rig_depth_max_m=float(rig_distance.max()),
            rig_depth_over_1000m_source_sample_count=int(np.count_nonzero(rig_distance > 1000)),
            cam4_sparse_pixel_range_statistics=cam4_pixel_statistics,
            transform_max_error_m=point_error, transform_max_relative_error=relative_point_error,
            reprojection_max_error_px=uv_error,
            geometry_mask_shape=list(geometry_mask.shape), geometry_mask_sha256=mask_hash,
            native_ray_sha256=native_hashes, visibility_regions=visibility_regions))
        print('Frame',frame,'geometric=',int(geom_both.sum()),'estimated visible=',int(visible_est.sum()),flush=True)

    assert len(set(mask_hashes)) == 1, 'Calibrated geometry changed across same-rig RGB frames'
    summary = dict(status='PASS_WITH_LIMITATIONS', selected_parameters_unchanged=True,
        selected_json_sha256=hashlib.sha256(selection_bytes).hexdigest(), frames=frame_results,
        gt_storage='Two-page TIFF: uint8 RGB thumbnail + float32 Rig-centered radial inverse depth (1/m)',
        original_gt_phi_deg=db.gt_phi, all_tiff_parent_counts=parent_counts,
        independent_camera_gt_files_found=independent_candidates,
        independent_camera_gt_available=False,
        legacy_readme_conflict=dict(path=str(root/'README.txt'),
            observed='Legacy README describes min_depth=55, latitude ±45 deg and a different Rig centre; '
                     'current config uses min depth 1.65m and current dataset code uses gt_phi=90 for float 320x640 TIFF',
            diagnostic_convention='Follow current unmodified dataset.py / dbhelper.py; external GT angular metadata not independently proven'),
        visibility_truth_status='UNAVAILABLE: only same Rig-observed point-cloud Z-buffer estimate is possible',
        zbuffer_estimator='Existing dataset_stereo.cloud_visibility; nearest rounded camera pixel; tolerance 0.01m +1%; no filling',
        zbuffer_limitation='Queries are part of the same sparse Rig-observed cloud. Unseen occluders and surfaces are absent; '
                           'self-support makes estimated visibility optimistic. Pixel collisions can also create discretization errors.',
        candidate_axis=dict(D=len(axis), inverse_depth_min=float(axis[0]), inverse_depth_max=float(axis[-1]),
            inverse_depth_step=float(axis[1]-axis[0]), farthest_radius_m=float(1/axis[0]), nearest_radius_m=float(1/axis[-1]),
            zero_index_definition='rho0 = sys.float_info.epsilon, not inverse-depth zero; finite numeric far limit',
            matches_original_dataset192_subsample=True, original_axis_max_difference=float(np.max(np.abs(original_axis-axis)))),
        geometry_content_independence=dict(frame_count=len(args.frames), common_mask_sha256=mask_hashes[0],
            masks_equal_for_all_frames=True, native_rays_equal_for_all_frames=True),
        range_rows=rows, weighting_note='Point fractions count valid original Rig TIFF samples. Optional weighted fractions use '
            'original Rig sphere solid angle only; they are not cam4-centred sphere area fractions.',
        gt_used_for_inference=False, gt_used_for_grid_generation=False)
    (out/'gt_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'figure_qa.json').write_text(json.dumps(QA,indent=2))
    with (out/'gt_range_by_region.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# GT 深度范围与跨相机可见性离线诊断','','## 数据与定义','',
        '已核查当前数据副本所有 TIFF 的父目录，仅发现 Rig-centered `omnidepth_gt_640`，没有独立 cam4/cam3 深度 GT。',
        '每个 TIFF 的首图是 uint8 RGB 缩略图，浮点页才是逆深度；本程序复用 `readImageFloat`，未把彩色缩略图当深度。',
        '数据自带 README.txt 是旧格式描述（min_depth=55、phi±45°、另一个 Rig 中心），与当前 YAML 及浮点 TIFF 形式冲突。本诊断遵守当前工程 gt_phi=90，不擅自改 GT 定义；该旧文档冲突不能被当作已经独立解决。',
        '使用原 `makeSphericalRays(raw_shape, gt_phi=90)` 恢复 Rig 点云，再用原外参转换到 cam4。距离量为 `||P_cam4||`，不是原 Rig 距离，也不是针孔 z 深度。',
        '没有修改或调用新的训练 GT 转换；只在独立验证脚本里对原始完整 GT 做诊断。','',
        '## 候选轴','',f'D={len(axis)}；rho0={axis[0]:.17g} 1/m，数值远限={1/axis[0]:.17g} m。',
        f'最近候选={1/axis[-1]:.12f} m；原 192 候选 `[::2]` 与冻结 96 轴逐项完全相同。','',
        '## 逐帧点比例','',
        '| frame | 有效 Rig GT 点数 | 双目几何有效点 | 同 Rig 点云估计双目可见比例 | 双目几何有效点中超近范围比例 |',
        '|---|---:|---:|---:|---:|']
    for frame in frame_results:
        r=next(r for r in rows if r['frame']==frame['frame_id'] and r['region']=='binocular_geometric')
        visibility=frame['visibility_regions']['all']
        lines.append(f"|{frame['frame_id']}|{frame['raw_valid_count']}|{visibility['geometric_point_count']}|"
                     f"{100*visibility['estimated_visible_fraction_of_geometric']:.3f}%|{100*r['below_nearest_point_fraction']:.3f}%|")
    lines += ['','更多 Yin / Yang / 交集及 cam4 全视场分母结果见 `gt_range_by_region.csv`。所有比例是原始 Rig GT 样本的比例；另外提供 Rig 球面面积权重结果，不冒充 cam4 球面面积。','',
        '## Geometric Mask 与 Visibility','',
        '- `geometric_valid`：同一 GT 三维点在 cam4/cam3 都通过 OCam FOV、图像边界和原始 invalid_mask。',
        '- `estimated_visible_both`：现有 Rig 点云分别投影至两台相机，经最近点 Z-buffer 后都未被该点云判为遮挡。',
        '- `independent_true_visibility_known` 全为 False：没有独立相机 GT，因此真实 visibility 仍未知。',
        '- 同一 Rig 点云中的每个查询点也贡献 Z-buffer，存在 self-support；Rig 不可见的遮挡物和表面没有被恢复，估计可能偏乐观。最近像素离散也可能误判。',
        '- 因此不能把本实验的可见性估计称为完整场景无遮挡真值，更不能声称已经满足独立 cam3 深度 GT 验证。','',
        '## 多帧不变性','',
        f'实际读取 {len(args.frames)} 帧原图和 GT。冻结 native rays 的 SHA256 不变；45×90 分析格、96 层候选的几何 Mask 哈希逐帧完全相同。GT 可见性和深度范围随内容变化。','',
        'GT 只用于本离线诊断，不参与 Yin/Yang 参数、网格生成或实际推理。']
    lines += ['', '## cam4 原始像素栅格的补充比例', '',
        '以下分母是 cam4 的 768×800 原始像素中，实际有 Rig 点云最近表面支持的像素。未观测像素保持未知，不插值补洞；这仍不是独立 cam4 深度 GT。', '',
        '| frame | 已观测 cam4 像素 | 未知 cam4 像素 | 超近像素 | 超近比例 |',
        '|---|---:|---:|---:|---:|']
    for frame in frame_results:
        p = frame['cam4_sparse_pixel_range_statistics']
        lines.append(f"|{frame['frame_id']}|{p['observed_sparse_rig_gt_pixel_count']}|{p['unknown_pixel_count']}|"
                     f"{p['below_nearest_pixel_count']}|{100*p['below_nearest_pixel_fraction']:.3f}%|")
    lines += ['', '上述5帧只是实际诊断样本，不能将其超近比例当作整个 OmniThings 数据集的分布。',
        '这类范围外 GT 与网格方向覆盖空洞是不同问题；本程序只报告，不改变冻结的深度轴或网格。']
    (out/'REPORT_GT.md').write_text('\n'.join(lines)+'\n')
    print('DONE',out,flush=True)


if __name__=='__main__':
    main()
