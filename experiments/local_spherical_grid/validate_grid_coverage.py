"""冻结 Yin/Yang 的分辨率、离散插值支持和接缝审查；不搜索、不修改参数。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from camera_geometry import load_pair, pair_geometry, spherical_rays
from search_yang_grid import (angular_coordinates, window_mask, native_rays,
    depth_visibility, classify, COLORS, LABELS, sphere_map)
from visualize_stereo_sphere import save_figure, PLOT_STYLE, QA


def native_lookup(visual, basis, bounds, shape):
    """同一球面方向→原生网格连续像素；保持 align_corners=True 的端点定义。

    只为安全查表而限幅，inside 独立保存，因此不会把窗口外点错误变成有效点。
    floor/ceil 可能相同；仅有正插值权重的角点才应要求有效。
    """
    lon, lat = angular_coordinates(visual, basis)
    h, w = shape
    delta = (lon-bounds['longitude_center_deg']+180) % 360-180
    u = (delta/(2*bounds['longitude_half_span_deg'])+.5)*(w-1)
    v = (bounds['latitude_max_deg']-lat)/(bounds['latitude_max_deg']-bounds['latitude_min_deg'])*(h-1)
    inside = window_mask(lon, lat, bounds)
    u = np.clip(u, 0, w-1); v = np.clip(v, 0, h-1)
    x0, x1 = np.floor(u).astype(np.int32), np.ceil(u).astype(np.int32)
    y0, y1 = np.floor(v).astype(np.int32), np.ceil(v).astype(np.int32)
    wx, wy = u-x0, v-y0
    ids = [(y0*w+x0).ravel(), (y0*w+x1).ravel(), (y1*w+x0).ravel(), (y1*w+x1).ravel()]
    weights = [((1-wx)*(1-wy)).ravel(), (wx*(1-wy)).ravel(), ((1-wx)*wy).ravel(), (wx*wy).ravel()]
    return dict(inside=inside.ravel(), ids=ids, active=[a>1e-10 for a in weights],
        nearest=(np.rint(v).astype(np.int32)*w+np.rint(u).astype(np.int32)).ravel(),
        lon=lon, lat=lat)


def supports(mask, lookup):
    values = mask.ravel()
    strict = lookup['inside'].copy()
    any_corner = np.zeros_like(strict)
    for ids, active in zip(lookup['ids'], lookup['active']):
        corner = values[ids]
        strict &= ~active | corner
        any_corner |= active & corner
    nearest = lookup['inside'] & values[lookup['nearest']]
    return nearest, strict, any_corner & lookup['inside']


def handoff_boundary(owner, valid):
    """主 ERP 上相邻有效点使用不同网格的交界，左右经度采用周期连接。"""
    seam = np.zeros_like(valid)
    right = valid & np.roll(valid, -1, 1) & (owner != np.roll(owner, -1, 1))
    seam |= right | np.roll(right, 1, 1)
    down = valid[:-1] & valid[1:] & (owner[:-1] != owner[1:])
    seam[:-1] |= down; seam[1:] |= down
    return seam


def export_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def sample_counts(charts, axis, tolerance_deg=1e-7):
    """空间覆盖的冗余不等于两张非一致网格上坐标完全相同的采样点。

    同时统计：落入另一个窗口的冗余 native samples、以及给定角容差下真正
    近重合的方向对。后者不使用高分辨率 ERP 像素数量冒充实际采样预算。
    """
    x, y = charts
    directions = [p['visual'].reshape(-1, 3) for p in charts]
    distances, match = cKDTree(directions[1]).query(directions[0], workers=1)
    angle = np.rad2deg(2*np.arcsin(np.minimum(distances/2, 1)))
    pair_ids = np.flatnonzero(angle <= tolerance_deg)
    other = []
    for own, target in [(x, y), (y, x)]:
        lon, lat = angular_coordinates(own['visual'], target['basis'])
        other.append(window_mask(lon, lat, target['bounds']))
    result = []
    for d in range(len(axis)):
        vm = [p['common'][d] for p in charts]
        coincident = int(np.count_nonzero(vm[0].ravel()[pair_ids] & vm[1].ravel()[match[pair_ids]]))
        result.append(dict(index=d, yin_total_samples=int(vm[0].size), yang_total_samples=int(vm[1].size),
            total_native_lattice_samples=int(vm[0].size+vm[1].size),
            yin_joint_valid_samples=int(vm[0].sum()), yang_joint_valid_samples=int(vm[1].sum()),
            yin_samples_inside_other_window=int(np.count_nonzero(vm[0] & other[0])),
            yang_samples_inside_other_window=int(np.count_nonzero(vm[1] & other[1])),
            coincident_valid_direction_pairs=coincident, coincidence_tolerance_deg=tolerance_deg))
    return result, dict(all_geometric_coincident_direction_pairs=int(len(pair_ids)),
        tolerance_deg=tolerance_deg, minimum_cross_grid_direction_distance_deg=float(angle.min()),
        total_native_samples_per_depth=int(len(directions[0])+len(directions[1])),
        total_3d_candidate_lattice_positions=int((len(directions[0])+len(directions[1]))*len(axis)))


def native_checks(charts, axis):
    rows = []
    for chart in charts:
        yin_lon, yin_lat = angular_coordinates(chart['visual'], charts[0]['basis'])
        yang_lon, yang_lat = angular_coordinates(chart['visual'], charts[1]['basis'])
        yin = window_mask(yin_lon, yin_lat, charts[0]['bounds'])
        yang = window_mask(yang_lon, yang_lat, charts[1]['bounds'])
        lookups = [native_lookup(chart['visual'], p['basis'], p['bounds'], p['common'].shape[1:]) for p in charts]
        for d in range(len(axis)):
            gt = chart['common'][d].ravel()
            available = [supports(p['common'][d], lookups[i]) for i, p in enumerate(charts)]
            strict = available[0][1] | available[1][1]
            nearest = available[0][0] | available[1][0]
            rows.append(dict(chart=chart['name'], index=d, valid_sample_count=int(gt.sum()),
                geometric_uncovered=int(np.count_nonzero(gt & ~(yin|yang).ravel())),
                strict_bilinear_unsupported=int(np.count_nonzero(gt & ~strict)),
                nearest_unsupported=int(np.count_nonzero(gt & ~nearest))))
    return rows


def native_seam_budget(charts, axis, cells):
    rows=[]
    for n in cells:
        bands=[]
        for i,p in enumerate(charts):
            h,w=p['common'].shape[1:]
            y,x=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
            edge=(x<n)|(x>=w-n)|(y<n)|(y>=h-n)
            other=charts[1-i]
            lon,lat=angular_coordinates(p['visual'],other['basis'])
            bands.append(edge & window_mask(lon,lat,other['bounds']))
        for d in range(len(axis)):
            counts=[int(np.count_nonzero(p['common'][d]&bands[i])) for i,p in enumerate(charts)]
            rows.append(dict(index=d,halo_cells_per_chart=n,
                yin_native_overlap_boundary_band_samples=counts[0],yang_native_overlap_boundary_band_samples=counts[1],
                total_native_overlap_boundary_band_samples=sum(counts)))
    return rows


def analyze_resolution(cameras, rotation, translation, selection, charts, size, out, seam_cells, reuse=False):
    h, w = size; dcount = len(selection['inverse_depth_candidates'])
    axis = np.asarray(selection['inverse_depth_candidates'])
    rays, visual, lon_rad, lat_rad, area = spherical_rays(h, w)
    lon, lat = np.rad2deg(lon_rad), np.rad2deg(lat_rad)
    yin = window_mask(lon, lat, charts[0]['bounds'])
    yang_lon, yang_lat = angular_coordinates(visual, charts[1]['basis'])
    yang = window_mask(yang_lon, yang_lat, charts[1]['bounds'])
    kp = 1/np.maximum(np.cos(lat_rad), 1e-6)
    ky = 1/np.maximum(np.cos(np.deg2rad(yang_lat)), 1e-6)
    owner = np.where(np.where(yin, kp, np.inf) <= np.where(yang, ky, np.inf), 1, 2)
    cache=out/f'{h}x{w}'/'coverage_masks.npz'
    if reuse and cache.exists():
        print(f'Reusing frozen {h}x{w} projection cache; checking axis and masks...',flush=True)
        with np.load(cache) as previous:
            assert np.array_equal(previous['axis'],axis)
            left,target,common=previous['left_valid'],previous['target_valid'],previous['common_valid']
        assert common.shape==(dcount,h,w) and np.array_equal(common,target & left[None])
    else:
        print(f'Projecting frozen {h}x{w} x {dcount} candidates...', flush=True)
        left, target, common = depth_visibility(cameras, rays, rotation, translation, axis)
    lookups = [native_lookup(visual, p['basis'], p['bounds'], p['common'].shape[1:]) for p in charts]
    holes = {name: np.zeros_like(common) for name in ['nearest', 'strict_bilinear', 'any_valid_corner']}
    geometric_holes = common & ~(yin|yang)[None]
    rows, seam_rows = [], []
    handoff_union = np.zeros((h, w), bool)
    strict_failure_union = np.zeros((h, w), bool)
    halo_fail_union = {n: np.zeros((h, w), bool) for n in seam_cells}
    same = lambda a: a.reshape(h, w)
    # native 中生成上下文完整性 Mask；只是诊断，不修改原始相机/网格 Mask。
    halos = {n: [minimum_filter(p['common'], size=(1, 2*n+1, 2*n+1), mode='constant', cval=0)
                 for p in charts] for n in seam_cells}
    farea = area.ravel()
    for d in range(dcount):
        valid = common[d]
        total = area[valid].sum()
        fraction = lambda mask: float(area[valid & mask].sum()/total) if total else 0.
        available = [supports(p['common'][d], lookups[i]) for i, p in enumerate(charts)]
        near, strict, any_corner = [same(available[0][j] | available[1][j]) for j in range(3)]
        for name, supported in [('nearest', near), ('strict_bilinear', strict), ('any_valid_corner', any_corner)]:
            holes[name][d] = valid & ~supported
        # 单像素参考格边界剔除仅用于定位问题，不将其改变为“最终无效”。
        interior = minimum_filter(valid, size=3, mode='constant', cval=0)
        seam = handoff_boundary(owner, valid)
        handoff_union |= seam
        strict_failure_union |= valid & ~strict
        seam_total = int(seam.sum())
        dual = same(available[0][1] & available[1][1])
        owner_supported=same(np.where(owner.ravel()==1,available[0][1],available[1][1]))
        rows.append(dict(height=h, width=w, index=d, valid_depth_direction_count=int(valid.sum()),
            geometric_uncovered_count=int(geometric_holes[d].sum()),
            geometric_coverage_ratio=1-fraction(~(yin|yang)),
            nearest_unsupported_count=int(holes['nearest'][d].sum()), nearest_supported_area_ratio=fraction(near),
            strict_bilinear_unsupported_count=int(holes['strict_bilinear'][d].sum()), strict_bilinear_supported_area_ratio=fraction(strict),
            any_corner_unsupported_count=int(holes['any_valid_corner'][d].sum()), any_corner_supported_area_ratio=fraction(any_corner),
            strict_unsupported_in_reference_3x3_interior=int(np.count_nonzero(valid & ~strict & interior)),
            high_latitude_uncovered_count=int(np.count_nonzero(geometric_holes[d] & (np.abs(lat)>=60))),
            geometric_overlap_area_sr=float(area[valid & yin & yang].sum()), overlap_of_common=fraction(yin & yang),
            handoff_sample_count=seam_total, strict_dual_supported_handoff_count=int(np.count_nonzero(seam & dual)),
            strict_union_supported_handoff_count=int(np.count_nonzero(seam & strict)),
            strict_owner_supported_handoff_count=int(np.count_nonzero(seam & owner_supported))))
        for n in seam_cells:
            supported = [supports(halos[n][i][d], lookups[i])[1] for i in range(2)]
            both = same(supported[0] & supported[1])
            either = same(supported[0] | supported[1])
            owner_halo=same(np.where(owner.ravel()==1,supported[0],supported[1]))
            halo_fail_union[n] |= seam & ~both
            step = [p['steps'] for p in charts]
            seam_rows.append(dict(height=h, width=w, index=d, halo_cells_per_chart=n,
                geometric_coverage_ratio=rows[-1]['geometric_coverage_ratio'], overlap_of_common=rows[-1]['overlap_of_common'],
                geometric_overlap_area_sr=rows[-1]['geometric_overlap_area_sr'],
                n_context_supported_area_ratio=fraction(either),
                handoff_sample_count=seam_total, dual_context_supported_handoff_count=int(np.count_nonzero(seam & both)),
                union_context_supported_handoff_count=int(np.count_nonzero(seam & either)),
                owner_context_supported_handoff_count=int(np.count_nonzero(seam & owner_halo)),
                handoff_dual_context_support_ratio=float(np.count_nonzero(seam & both)/seam_total) if seam_total else 1.,
                yin_nominal_halo_lon_deg=n*step[0][0], yin_nominal_halo_lat_deg=n*step[0][1],
                yang_nominal_halo_lon_deg=n*step[1][0], yang_nominal_halo_lat_deg=n*step[1][1]))
    # 计数与球面面积都报告，避免高纬度 ERP 的像素计数夸大问题。
    weighted_valid = sum(float(area[x].sum()) for x in common)
    summary = dict(height=h, width=w, num_depth=dcount,
        valid_depth_direction_count=sum(r['valid_depth_direction_count'] for r in rows),
        geometric_uncovered_count=int(geometric_holes.sum()),
        geometric_coverage_ratio=1.-sum(float(area[x].sum()) for x in geometric_holes)/weighted_valid,
        nearest_unsupported_count=int(holes['nearest'].sum()),
        strict_bilinear_unsupported_count=int(holes['strict_bilinear'].sum()),
        any_corner_unsupported_count=int(holes['any_valid_corner'].sum()),
        nearest_unsupported_weighted_fraction=sum(float(area[x].sum()) for x in holes['nearest'])/weighted_valid,
        strict_unsupported_weighted_fraction=sum(float(area[x].sum()) for x in holes['strict_bilinear'])/weighted_valid,
        any_corner_unsupported_weighted_fraction=sum(float(area[x].sum()) for x in holes['any_valid_corner'])/weighted_valid,
        native_nearest_support_is_not_continuous_coverage_proof=True)
    folder = out/f'{h}x{w}'; folder.mkdir(parents=True, exist_ok=True)
    export_csv(folder/'coverage_per_depth.csv', rows); export_csv(folder/'seam_per_depth.csv', seam_rows)
    np.savez_compressed(folder/'coverage_masks.npz', axis=axis, area_sr=area, left_valid=left, target_valid=target,
        common_valid=common, yin=yin, yang=yang, geometric_uncovered=geometric_holes,
        nearest_unsupported=holes['nearest'], strict_bilinear_unsupported=holes['strict_bilinear'],
        any_corner_unsupported=holes['any_valid_corner'], handoff_union=handoff_union,
        strict_unsupported_union=strict_failure_union,
        **{f'halo_{n}_unsupported_handoff_union':a for n,a in halo_fail_union.items()})
    cmap = ListedColormap(COLORS/255); norm = BoundaryNorm(np.arange(-.5, 5), 5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    sphere_map(axes[0], classify(common.any(0), yin, yang), f'Frozen windows, {h} x {w}', cmap, norm)
    sphere_map(axes[1], holes['strict_bilinear'].mean(0), 'Strict bilinear unsupported / 96 candidates', 'magma', vmin=0, vmax=1)
    fig.colorbar(axes[1].images[0], ax=axes[1], fraction=.025, label='Unsupported candidate fraction')
    save_figure(fig, folder/'coverage_vs_sampling_support.png')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for ax, (name, value) in zip(axes.ravel(), [('Nearest', holes['nearest'].mean(0)),
            ('All active bilinear corners', holes['strict_bilinear'].mean(0)),
            ('Any positive-weight valid corner', holes['any_valid_corner'].mean(0)),
            ('Geometric window uncovered', geometric_holes.mean(0))]):
        image=sphere_map(ax, value, name, 'magma', vmin=0, vmax=.1)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=.02, label='Fraction of candidates (display 0-0.1)')
    save_figure(fig, folder/'native_sampling_holes.png')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for ax, n in zip(axes.ravel(), seam_cells[:4]):
        sphere_map(ax, halo_fail_union[n], f'{n}-cell halo: any-depth handoff support failure', 'Greys', vmin=0, vmax=1)
        ax.set_xlim(-30, 120); ax.set_ylim(15, 70)
        ax.set_xticks([-30, 0, 30, 60, 90, 120]); ax.set_yticks([15, 30, 45, 60, 70])
    save_figure(fig, folder/'seam_zoom.png')
    print('Result:', json.dumps(summary), flush=True)
    # 只将小尺寸概览传给最终HTML，完整Mask保存NPZ，不把高分辨率数据塞入浏览器。
    preview = dict(valid_any=common.any(0)[::max(1,h//90), ::max(1,w//180)],
        nearest_holes=holes['nearest'].any(0)[::max(1,h//90), ::max(1,w//180)],
        strict_holes=holes['strict_bilinear'].any(0)[::max(1,h//90), ::max(1,w//180)])
    return summary, rows, seam_rows, preview


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--native-dir',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resolutions',type=int,nargs='+',default=[360,720,720,1440])
    p.add_argument('--reuse-projections',action='store_true',help='Only reuse masks with the exact same frozen selection SHA and axis')
    args=p.parse_args()
    if len(args.resolutions)%2:p.error('resolutions must be H W pairs')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    source_bytes=args.selection.read_bytes(); s=json.loads(source_bytes)
    if args.reuse_projections:
        prior=json.loads((out/'coverage_validation.json').read_text())
        assert prior['frozen_selection_sha256']==hashlib.sha256(source_bytes).hexdigest()
    cameras,images,cfg,prov=load_pair(args.data_root,4,3,s['frame_id'])
    rot,tr,_=pair_geometry(*cameras); axis=np.asarray(s['inverse_depth_candidates'])
    plt.rcParams.update(PLOT_STYLE)
    charts=[]
    for name,bounds,basis in [('yin',s['yin_bounds'],np.eye(3)),('yang',s['yang_bounds'],np.asarray(s['yang_basis_visual']))]:
        original=np.load(args.native_dir/f'{name}_depth_validity.npz')
        shape=original['valid_mask'].shape[1:]
        rays,visual=native_rays(bounds,basis,shape)
        left,right,common=depth_visibility(cameras,rays,rot,tr,axis)
        assert np.array_equal(common,original['valid_mask']), 'Frozen native masks changed'
        assert np.array_equal(axis,original['inverse_depth_candidates'])
        charts.append(dict(name=name,bounds=bounds,basis=basis,rays=rays,visual=visual,left=left,
            common=common,steps=(2*bounds['longitude_half_span_deg']/(shape[1]-1),
             (bounds['latitude_max_deg']-bounds['latitude_min_deg'])/(shape[0]-1))))
    count_rows,budget=sample_counts(charts,axis)
    native_rows=native_checks(charts,axis)
    export_csv(out/'actual_native_sample_counts.csv',count_rows)
    export_csv(out/'native_grid_coverage.csv',native_rows)
    summaries=[]; seam_summary=[]
    # 同时报告每侧 halo=2/4/6，以及总宽2/4/6对应的每侧1/2/3，消除“宽度”歧义。
    halo_cells=[2,4,6,1,3]
    export_csv(out/'native_seam_sample_budgets.csv',native_seam_budget(charts,axis,halo_cells))
    for h,w in zip(args.resolutions[::2],args.resolutions[1::2]):
        summary,rows,seams,preview=analyze_resolution(cameras,rot,tr,s,charts,(h,w),out,halo_cells,args.reuse_projections)
        summaries.append(summary)
        for n in halo_cells:
            a=[r for r in seams if r['halo_cells_per_chart']==n]
            samples=sum(r['handoff_sample_count'] for r in a)
            supported=sum(r['dual_context_supported_handoff_count'] for r in a)
            seam_summary.append(dict(height=h,width=w,halo_cells_per_chart=n,
                total_handoff_samples=samples,dual_context_supported_handoff_samples=supported,
                dual_context_support_ratio=supported/samples if samples else 1.,
                union_context_support_ratio=sum(r['union_context_supported_handoff_count'] for r in a)/samples if samples else 1.,
                owner_context_support_ratio=sum(r['owner_context_supported_handoff_count'] for r in a)/samples if samples else 1.,
                geometric_coverage_ratio=summary['geometric_coverage_ratio'],
                actual_geometric_overlap_changes=False,total_native_samples_per_depth=budget['total_native_samples_per_depth'],
                mean_overlap_of_common=float(np.mean([r['overlap_of_common'] for r in a]))))
    export_csv(out/'seam_summary.csv',seam_summary)
    result=dict(frozen_selection_sha256=hashlib.sha256(source_bytes).hexdigest(),
        selected_parameters_unchanged=True,resolutions=summaries,native_budget=budget,
        native_checks=dict(rows=len(native_rows),geometric_uncovered=sum(r['geometric_uncovered'] for r in native_rows),
            strict_unsupported=sum(r['strict_bilinear_unsupported'] for r in native_rows),
            nearest_unsupported=sum(r['nearest_unsupported'] for r in native_rows)),seam_summary=seam_summary,
        definitions=dict(geometric='Analytic point lies in either fixed chart and both cameras project validly',
            strict_bilinear='All positive-weight native bilinear corners must be joint-valid at that depth',
            any_corner='At least one positive-weight native corner is joint-valid; not a claim of an implemented renormalizer',
            halo='n cells inward on each native chart; context square is (2n+1)x(2n+1)',
            total_width='For total width 2/4/6 cells, use the additional per-side n=1/2/3 rows',
            handoff='Adjacent jointly valid reference rays whose lower-coordinate-kappa chart differs',
            dual_support_caution='Both charts supported at every interface-side point is stronger than union continuity; chart truncation can make dual false while union remains supported',
            duplicate='Native directions within 1e-7 degree; distinct from redundant samples inside the other chart'),
        limitations=['Strict corner support failure is a discretization/mask issue, not geometric window failure',
            'Context erosion is diagnostic only and is not applied to final coverage or model masks',
            'No parameter refitting, search, GT use or network modification'])
    assert args.selection.read_bytes()==source_bytes
    (out/'coverage_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'figure_qa.json').write_text(json.dumps(QA,indent=2))
    print('DONE',out,flush=True)


if __name__=='__main__':main()
