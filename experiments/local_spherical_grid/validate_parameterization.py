"""冻结 Yin / Yang 的参数化质量审查；只读选定参数与几何 Mask，不做搜索。"""
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from camera_geometry import spherical_rays
from search_yang_grid import angular_coordinates, window_mask
from visualize_stereo_sphere import PLOT_STYLE, QA, save_figure


def weighted_stats(values, weights, support=None):
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if support is not None:
        mask &= support
    value, weight = values[mask].astype(np.float64), weights[mask].astype(np.float64)
    if not len(value):
        return dict(count=0, weighted_mean=None, p95=None, p99=None, maximum=None, weight_sr=0.)
    order = np.argsort(value, kind='stable')
    cdf = (np.cumsum(weight[order])-.5*weight[order])/weight.sum()
    return dict(count=int(len(value)), weighted_mean=float(np.average(value, weights=weight)),
                p95=float(np.interp(.95, cdf, value[order])), p99=float(np.interp(.99, cdf, value[order])),
                maximum=float(value.max()), weight_sr=float(weight.sum()))


def coordinate_kappa(latitude_deg):
    return 1/np.maximum(np.abs(np.cos(np.deg2rad(latitude_deg))), 1e-12)


def actual_pixel_anisotropy(latitude_deg, lon_step_deg, lat_step_deg, finite=False):
    cosine = np.abs(np.cos(np.deg2rad(latitude_deg)))
    if finite:
        # 同一纬线相邻经度点间的球面最短距离；纬向相邻点距离恰为 Δφ。
        horizontal = 2*np.arcsin(np.clip(cosine*np.sin(np.deg2rad(lon_step_deg)/2), 0, 1))
    else:
        horizontal = cosine*np.deg2rad(lon_step_deg)
    vertical = np.full_like(cosine, np.deg2rad(lat_step_deg))
    return np.maximum(horizontal, vertical)/np.maximum(np.minimum(horizontal, vertical), 1e-12)


def native_area(bounds, shape):
    """Endpoint 样本的 Voronoi 半格截到固定矩形边界，端点不是整格权重。"""
    h, w = shape
    lon = np.linspace(bounds['longitude_center_deg']-bounds['longitude_half_span_deg'],
                      bounds['longitude_center_deg']+bounds['longitude_half_span_deg'], w)
    lat = np.linspace(bounds['latitude_max_deg'], bounds['latitude_min_deg'], h)
    lon_edges = np.r_[lon[0], .5*(lon[:-1]+lon[1:]), lon[-1]]
    lat_edges = np.r_[lat[0], .5*(lat[:-1]+lat[1:]), lat[-1]]
    area = ((np.sin(np.deg2rad(lat_edges[:-1]))-np.sin(np.deg2rad(lat_edges[1:])))[:, None] *
            np.diff(np.deg2rad(lon_edges))[None])
    exact = (np.deg2rad(2*bounds['longitude_half_span_deg']) *
             (np.sin(np.deg2rad(bounds['latitude_max_deg']))-np.sin(np.deg2rad(bounds['latitude_min_deg']))))
    assert np.isclose(area.sum(), exact, atol=1e-12)
    return lon, lat, area


def map_erp(ax, value, mask, title, upper=2.):
    im = ax.imshow(np.ma.array(value, mask=~mask), extent=(-180, 180, -90, 90),
                   origin='upper', aspect='auto', cmap='viridis', norm=LogNorm(vmin=1, vmax=upper))
    ax.set_facecolor('#e8ebf0')
    ax.set(title=title, xlabel='Primary longitude (deg)', ylabel='Primary latitude (deg)',
           xticks=np.arange(-180, 181, 60), yticks=np.arange(-90, 91, 30))
    ax.grid(alpha=.15)
    return im


def write_heatmaps(out, maps, supports, lat, yang_lat, native, area, depth_weights):
    # 三张单独热力图采用同一色标，以便比较，不从小范围自动拉伸夸大差异。
    for key, title in [('yin_coordinate', 'Yin coordinate kappa | common valid union'),
                        ('yang_coordinate', 'Yang coordinate kappa | common valid union'),
                        ('best_coordinate', 'Lower coordinate kappa | available Yin / Yang')]:
        fig, ax = plt.subplots(figsize=(11, 5.4), layout='constrained')
        im = map_erp(ax, maps[key], supports[key], title)
        bar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=.15, fraction=.06,
                           label='Coordinate anisotropy (dimensionless)', ticks=[1, 1.1, 1.25, 1.5, 2])
        bar.set_ticklabels(['1', '1.1', '1.25', '1.5', '2']); bar.minorticks_off()
        save_figure(fig, out/f'{key}_heatmap.png')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for ax, key, title in zip(axes, ['yin_pixel', 'yang_pixel', 'best_pixel'],
                              ['Yin actual angular steps', 'Yang actual angular steps', 'Lower pixel anisotropy']):
        im = map_erp(ax, maps[key], supports[key], title, upper=2.01)
    bar = fig.colorbar(im, ax=axes.tolist(), fraction=.025, label='Local pixel anisotropy', ticks=[1,1.25,1.5,2])
    bar.set_ticklabels(['1','1.25','1.5','2']); bar.minorticks_off()
    save_figure(fig, out/'actual_pixel_anisotropy_erp.png')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for col, name in enumerate(['yin','yang']):
        p = native[name]; b = p['bounds']
        extent = [p['lon'][0], p['lon'][-1], p['lat'][-1], p['lat'][0]]
        for row, key in enumerate(['coordinate', 'pixel']):
            im = axes[row,col].imshow(np.ma.array(p[key], mask=~p['valid_any']), extent=extent,
                origin='upper', aspect='auto', cmap='viridis', norm=LogNorm(1,2.01))
            axes[row,col].set(title=f'{name.capitalize()} {key} | {p["valid"].shape[1:]}',
                xlabel=f'{name.capitalize()} longitude (deg)', ylabel=f'{name.capitalize()} latitude (deg)')
            axes[row,col].set_xticks(np.linspace(extent[0],extent[1],5))
            axes[row,col].set_yticks(np.linspace(extent[2],extent[3],5))
            axes[row,col].set_facecolor('#e8ebf0')
    bar=fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.025, label='Anisotropy (shared scale)', ticks=[1,1.25,1.5,2])
    bar.set_ticklabels(['1','1.25','1.5','2']); bar.minorticks_off()
    save_figure(fig, out/'native_coordinate_vs_pixel.png')

    fig, ax = plt.subplots(figsize=(8,4.5), layout='constrained')
    for key, label, color, style in [('primary_coordinate','Primary ERP','#777777',':'),
                                    ('best_coordinate','Lower coordinate kappa','#0072B2','-'),
                                    ('best_pixel','Lower actual pixel anisotropy','#D55E00','--')]:
        mask=supports[key]; v=maps[key][mask]; w=depth_weights[mask]
        order=np.argsort(v); cdf=np.cumsum(w[order])/w.sum()
        # 阶梯CDF来自确定性几何样本，不是独立场景样本的置信区间。
        ax.plot(v[order],100*cdf,color=color,linestyle=style,label=label)
    ax.set(xscale='log',xlim=(1,240),ylim=(0,100),xlabel='Anisotropy (dimensionless, log scale)',
           ylabel='Depth / spherical-area weighted CDF (%)',title='Common valid directions across all 96 candidates')
    ax.set_xticks([1,1.5,2,4,16,64,229], ['1','1.5','2','4','16','64','229'])
    ax.legend(frameon=False); ax.grid(alpha=.2)
    save_figure(fig, out/'anisotropy_weighted_cdf.png', vector=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-dir',type=Path,default=Path(__file__).resolve().parent/'outputs_yang_search')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'outputs_validation/quality')
    args=parser.parse_args()
    source=args.selected_dir.expanduser().resolve(); out=args.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    frozen_path=source/'search_result.json'; frozen_bytes=frozen_path.read_bytes(); selection=json.loads(frozen_bytes)
    plt.rcParams.update(PLOT_STYLE); logging.getLogger('fontTools').setLevel(logging.WARNING)
    QA.clear()
    with np.load(source/'reference_depth_masks.npz') as m:
        common=m['common_valid'].copy(); area=m['area_sr'].copy(); axis=m['inverse_depth_candidates'].copy()
        yin=m['yin'].copy(); yang=m['yang'].copy()
    h,w=area.shape
    _, visual, lon_rad, lat_rad, regenerated_area=spherical_rays(h,w)
    assert np.array_equal(area,regenerated_area)
    basis=np.array(selection['yang_basis_visual'])
    _, yang_lat=angular_coordinates(visual,basis); lat=np.rad2deg(lat_rad)
    assert np.array_equal(yin,window_mask(np.rad2deg(lon_rad),lat,selection['yin_bounds']))
    yang_lon,_=angular_coordinates(visual,basis)
    assert np.array_equal(yang,window_mask(yang_lon,yang_lat,selection['yang_bounds']))
    union=common.any(0); depth_weights=area*common.sum(0)
    masks={'yin':yin,'yang':yang}
    native={}; native_stats={}
    for name in ['yin','yang']:
        bounds=selection[f'{name}_bounds']
        with np.load(source/f'{name}_depth_validity.npz') as m:
            valid=m['valid_mask'].copy(); saved_pixel=m['native_pixel_anisotropy'].copy()
            assert np.array_equal(axis,m['inverse_depth_candidates'])
        lon, native_lat, native_weights=native_area(bounds,valid.shape[1:])
        latitude=np.broadcast_to(native_lat[:,None], valid.shape[1:])
        dx=2*bounds['longitude_half_span_deg']/(valid.shape[2]-1)
        dy=(bounds['latitude_max_deg']-bounds['latitude_min_deg'])/(valid.shape[1]-1)
        coord=coordinate_kappa(latitude)
        pixel=actual_pixel_anisotropy(latitude,dx,dy)
        finite=actual_pixel_anisotropy(latitude,dx,dy,finite=True)
        assert np.allclose(pixel,saved_pixel,atol=2e-7)
        native[name]=dict(bounds=bounds,lon=lon,lat=native_lat,valid=valid,valid_any=valid.any(0),
            area=native_weights,coordinate=coord,pixel=pixel,finite=finite,lon_step=dx,lat_step=dy)
        full_summary={k:weighted_stats(v,native_weights) for k,v in [('coordinate',coord),('pixel',pixel),('finite_geodesic_pixel',finite)]}
        valid_summary={k:weighted_stats(v,native_weights*valid.sum(0)) for k,v in [('coordinate',coord),('pixel',pixel),('finite_geodesic_pixel',finite)]}
        native_stats[name]=dict(shape_dhw=list(valid.shape),longitude_step_deg=dx,latitude_step_deg=dy,
            step_ratio=dx/dy,patch_solid_angle_sr=float(native_weights.sum()),
            full_patch=full_summary,depth_area_weighted_valid=valid_summary,
            differential_vs_finite_geodesic_max_abs=float(np.max(np.abs(pixel-finite))))
        np.savez_compressed(out/f'{name}_native_quality.npz',area_sr=native_weights,valid_any=valid.any(0),
            depth_valid_count=valid.sum(0),coordinate_kappa=coord,local_pixel_anisotropy=pixel,
            finite_geodesic_pixel_anisotropy=finite,latitude_deg=latitude,
            longitude_deg=np.broadcast_to(lon[None],valid.shape[1:]),inverse_depth_candidates=axis)

    primary=coordinate_kappa(lat); cy=coordinate_kappa(yang_lat)
    py=actual_pixel_anisotropy(lat,native['yin']['lon_step'],native['yin']['lat_step'])
    pg=actual_pixel_anisotropy(yang_lat,native['yang']['lon_step'],native['yang']['lat_step'])
    best_coord=np.minimum(np.where(yin,primary,np.inf),np.where(yang,cy,np.inf))
    best_pixel=np.minimum(np.where(yin,py,np.inf),np.where(yang,pg,np.inf))
    maps=dict(primary_coordinate=primary,yin_coordinate=primary,yang_coordinate=cy,best_coordinate=best_coord,
              yin_pixel=py,yang_pixel=pg,best_pixel=best_pixel)
    supports={key:union & (yin if key.startswith('yin') else yang if key.startswith('yang') else (yin|yang)) for key in maps}
    supports['primary_coordinate']=union
    rows=[];aggregate={}; union_stats={}
    for key,value in maps.items():
        aggregate[key]=weighted_stats(value,depth_weights,supports[key])
        union_stats[key]=weighted_stats(value,area,supports[key])
        for i,mask in enumerate(common):
            st=weighted_stats(value,area,mask & supports[key])
            rows.append(dict(index=i,radius_m=float(1/axis[i]),metric=key,**st))
    assert np.isclose(aggregate['best_coordinate']['weighted_mean'],selection['selected']['depth_weighted_mean_best_kappa'],atol=1e-12)
    result=dict(status='PASS',frozen_selection_sha256=hashlib.sha256(frozen_bytes).hexdigest(),
        source_mask_shape=list(common.shape),weighting='sum over 96 equally weighted candidates, exact spherical solid angle per reference cell; chart-only summaries restricted to chart and common camera validity',
        coordinate_formula='kappa=1/abs(cos(latitude)), assumes equal infinitesimal angular steps',
        local_pixel_formula='max(delta_lon*cos(latitude),delta_lat)/min(delta_lon*cos(latitude),delta_lat)',
        finite_pixel_formula='horizontal=2*asin(cos(latitude)*sin(delta_lon/2)); vertical=delta_lat; ratio=max/min',
        percentile_definition='Spherical-area weighted, linearly interpolated at centered cumulative weights; includes only finite supported samples',
        native_area_definition='Voronoi half-cell edges at midpoints, truncated exactly to frozen chart rectangle; endpoint and corner samples have half/quarter angular-cell support',
        depth_area_weighted=aggregate,union_area_weighted=union_stats,native=native_stats,
        selection_preserved=True,limits=['No GT used; parameterization metrics do not establish depth accuracy',
            'Common FOV mask is not occlusion visibility','Pixel anisotropy is local spherical spacing, not CNN receptive field or interpolation support',
            'Finite sampled maximum is not a bound at unobserved directions; primary ERP diverges at the exact poles'])
    np.savez_compressed(out/'reference_quality_maps.npz',**{k:np.where(supports[k],v,np.nan) for k,v in maps.items()},
        common_valid_count=common.sum(0),area_sr=area,primary_latitude_deg=lat,yang_latitude_deg=yang_lat,
        yin=yin,yang=yang,inverse_depth_candidates=axis)
    (out/'quality_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'quality_per_depth.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'quality_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['metric','domain','count','weighted_mean','p95','p99','maximum','weight_sr']);writer.writeheader()
        for domain,data in [('depth_area',aggregate),('union_area',union_stats)]:
            for key,st in data.items():writer.writerow(dict(metric=key,domain=domain,**st))
    write_heatmaps(out,maps,supports,lat,yang_lat,native,area,depth_weights)
    (out/'figure_qa.json').write_text(json.dumps(QA,indent=2)+'\n')
    lines=['# 冻结网格参数化质量审查','',f'读取冻结文件 SHA256：`{result["frozen_selection_sha256"]}`。不搜索、不改参数、不读取 GT。',
        '', '## 统计口径','', '使用完整 360×720×96 几何候选 Mask。每层按球面格子的精确立体角加权，96 个候选等权；'
        '这里的深度权重不是场景真实深度概率。Yin/Yang 各自统计只使用该窗口内双目投影有效的方向；择优必须受对应窗口约束。',
        '', '| 指标 | 面积/深度加权均值 | P95 | P99 | 最大值 |','|---|---:|---:|---:|---:|']
    for key,st in aggregate.items():lines.append(f'|{key}|{st["weighted_mean"]:.8f}|{st["p95"]:.8f}|{st["p99"]:.8f}|{st["maximum"]:.8f}|')
    lines+=['','## 理论坐标指标和真实像素间距指标','',
        '`κ=1/|cosφ|` 只描述等角步长的经纬坐标各向异性。原生网格的两个角步长并非完全相等，必须使用 `max(Δλ cosφ, Δφ)/min(...)`。',
        '此外还计算了相邻射线的有限球面距离比，区分局部微分指标与有限格距指标。当前角步长较小，两者接近但并非完全相同。',
        '', '| 原生网格 | Shape | Δλ (deg) | Δφ (deg) | 理论 κ 最大值 | 像素指标最大值 | 有限射线距离比最大值 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,st in native_stats.items():
        full=st['full_patch'];lines.append(f'|{name}|{st["shape_dhw"]}|{st["longitude_step_deg"]:.9f}|{st["latitude_step_deg"]:.9f}|'
            f'{full["coordinate"]["maximum"]:.9f}|{full["pixel"]["maximum"]:.9f}|{full["finite_geodesic_pixel"]["maximum"]:.9f}|')
    lines+=['','上表最大值按整个冻结矩形记录，包括窗口内相机不可见区域；JSON 另外给出真正双目有效部分的面积/深度加权分布。',
        '', '## 原生面积权重','', result['native_area_definition'],
        '经纬端点是采样点，不代表边界外还有一整格面积。以相邻采样点的中点作为面积边界，并截到冻结矩形；'
        '每个原生网格面积之和与解析矩形球面面积一致。不能把所有原生像素等权后称为球面面积加权。',
        '', '## 发现','',
        f'- 既有 depth-weighted mean best kappa 得到复算：{aggregate["best_coordinate"]["weighted_mean"]:.8f}，与冻结搜索记录一致。',
        f'- Yin 的理论最大 κ=2，而当前原生格距下像素各向异性最大为 {native_stats["yin"]["full_patch"]["pixel"]["maximum"]:.8f}；'
        '因此不能把“理论 κ≤2”写成“所有实际像素各向异性都严格≤2”。这不是外参或旋转错误。',
        '- 原始主 ERP 极点处理论发散；有限分析网格未直接采极点，因此其最大值依赖网格分辨率。',
        '- 以上度量不检查 CNN 插值支持、接缝跨网格映射或真实遮挡；这些应由其他验证模块单独报告。',
        '', '## 文件','', '- `yin_coordinate_heatmap.png`、`yang_coordinate_heatmap.png`、`best_coordinate_heatmap.png`：理论 κ 热力图。',
        '- `actual_pixel_anisotropy_erp.png`、`native_coordinate_vs_pixel.png`：实际角步长下的像素各向异性。',
        '- `anisotropy_weighted_cdf.png`：面积/深度加权累计分布。',
        '- `quality_summary.json/csv`、`quality_per_depth.csv` 和 NPZ：全部真实数值及有效域。']
    (out/'REPORT_QUALITY.md').write_text('\n'.join(lines)+'\n')
    assert frozen_path.read_bytes()==frozen_bytes
    print(json.dumps(dict(status=result['status'],depth_area_weighted=aggregate,native=native_stats,
                         figures_with_text_outside_canvas=[x for x in QA if x['text_outside_canvas']]),indent=2),flush=True)
    print('DONE:',out,flush=True)


if __name__=='__main__':main()
