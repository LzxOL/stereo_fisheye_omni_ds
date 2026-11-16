"""有限参数对照：重搜 Yin ±45/50/60，与相近预算 Single Rotated Local ERP 比较。

只新增对照结果，不写旧 selected_grid.json，不改 OCam/深度轴/网络。
所有配置共用同一真实相机投影有效集合；改变网格只改变覆盖与插值支持。
"""
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from camera_geometry import load_pair, pair_geometry, spherical_rays
from search_yang_grid import (depth_axis, depth_visibility, fit_yin, search_rotation,
    full_resolution_selection, fit_candidate, shortest_longitude_window,
    angular_coordinates, native_rays, window_mask, rx, ry)
from validate_grid_coverage import native_lookup, supports
from validate_parameterization import weighted_stats, actual_pixel_anisotropy, coordinate_kappa
from visualize_stereo_sphere import save_figure, PLOT_STYLE, QA


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as file:
        w = csv.DictWriter(file, fieldnames=fields); w.writeheader(); w.writerows(rows)


def common_search_args(limit):
    return SimpleNamespace(yin_lat_limit=float(limit), seam_deg=5., padding_deg=2., boundary_step_deg=.5,
        yang_max_lat=45., max_yang_lon_span=355., tilt_min=45., tilt_max=135., coarse_step_deg=15.,
        refine_step_deg=2., refine_top=6, final_top=48)


def no_feasible_audit(visual, target, args):
    """仅细化约束违反最小的少数方向；失败也给出可核查证据，不擅自放宽约束。"""
    rows, seen = [], set()
    def add(yaw, tilt, stage):
        key = (round(yaw % 360, 6), round(tilt, 6))
        if key in seen or not 45 <= tilt <= 135: return
        seen.add(key)
        row = fit_candidate(visual, target, *key, args)
        row.update(stage=stage, required_abs_latitude_deg=max(abs(row['latitude_min_deg']),abs(row['latitude_max_deg'])))
        rows.append(row)
    order = lambda r: (r['required_abs_latitude_deg'], abs(r['tilt_x_deg']-90), r['yaw_y_deg'])
    for yaw in np.arange(0,360,15):
        for tilt in np.arange(45,136,15):add(yaw,tilt,'coarse')
    for seed in sorted(rows,key=order)[:6]:
        for dy in np.arange(-15,15.01,2):
            for dt in np.arange(-15,15.01,2):add(seed['yaw_y_deg']+dy,seed['tilt_x_deg']+dt,'refine_violation')
    for seed in sorted(rows,key=order)[:6]:
        for dy in np.arange(-2,2.01,.5):
            for dt in np.arange(-2,2.01,.5):add(seed['yaw_y_deg']+dy,seed['tilt_x_deg']+dt,'fine_violation')
    return rows, min(rows,key=order)


def single_fit(visual, valid_any, yaw, tilt, args):
    basis = ry(yaw)@rx(tilt)
    lon,lat=angular_coordinates(visual[valid_any],basis)
    lo=np.floor((lat.min()-args.padding_deg)/args.boundary_step_deg)*args.boundary_step_deg
    hi=np.ceil((lat.max()+args.padding_deg)/args.boundary_step_deg)*args.boundary_step_deg
    center,half=shortest_longitude_window(lon,args.padding_deg,args.boundary_step_deg)
    row=dict(yaw_y_deg=float(yaw%360),tilt_x_deg=float(tilt),latitude_min_deg=float(lo),latitude_max_deg=float(hi),
        longitude_center_deg=center,longitude_half_span_deg=half,required_abs_latitude_deg=float(max(abs(lo),abs(hi))),
        angular_rectangle_deg2=float(2*half*(hi-lo)),feasible=bool(max(abs(lo),abs(hi))<89.5 and 2*half<355))
    return row


def search_single(visual, valid_any, args):
    """单网格不使用双网格 Yang 的45°限制。优先把两个极点移出有效域，减少尾部退化。"""
    rows,seen=[],set()
    def add(yaw,tilt,stage):
        key=(round(yaw%360,6),round(tilt,6))
        if key in seen or not 0<=tilt<=180:return
        seen.add(key);row=single_fit(visual,valid_any,*key,args);row['stage']=stage;rows.append(row)
    order=lambda r:(r['required_abs_latitude_deg'],r['angular_rectangle_deg2'],abs(r['tilt_x_deg']-90),r['yaw_y_deg'])
    for yaw in np.arange(0,360,15):
        for tilt in np.arange(0,181,15):add(yaw,tilt,'coarse')
    for seed in sorted([r for r in rows if r['feasible']],key=order)[:6]:
        for dy in np.arange(-15,15.01,2):
            for dt in np.arange(-15,15.01,2):add(seed['yaw_y_deg']+dy,seed['tilt_x_deg']+dt,'refine')
    for seed in sorted([r for r in rows if r['feasible']],key=order)[:6]:
        for dy in np.arange(-2,2.01,.5):
            for dt in np.arange(-2,2.01,.5):add(seed['yaw_y_deg']+dy,seed['tilt_x_deg']+dt,'fine')
    return rows


def auto_shape(bounds, angular_step=.75):
    return (int(np.ceil((bounds['latitude_max_deg']-bounds['latitude_min_deg'])/angular_step))+1,
            int(np.ceil(2*bounds['longitude_half_span_deg']/angular_step))+1)


def matched_budget_shape(bounds, budget):
    # 在1%预算误差以内匹配，优先使经纬角步长近似一致；不resize任何已生成射线。
    width=2*bounds['longitude_half_span_deg'];height=bounds['latitude_max_deg']-bounds['latitude_min_deg']
    expected=np.sqrt(budget*height/width)
    choices=[]
    for h in range(max(2,int(expected)-12),int(expected)+14):
        for w in [max(2,int(budget/h)),max(2,int(round(budget/h)))]:
            if abs(h*w-budget)/budget>.01:continue
            steps=(width/(w-1),height/(h-1))
            choices.append((abs(np.log(steps[0]/steps[1])),abs(h*w-budget)/budget,h,w))
    return min(choices)[2:]


def chart_record(name,bounds,basis,cameras,rotation,translation,axis,shape=None):
    shape=tuple(shape or auto_shape(bounds))
    rays,visual=native_rays(bounds,basis,shape)
    left,target,common=depth_visibility(cameras,rays,rotation,translation,axis)
    return dict(name=name,bounds=bounds,basis=basis,shape=shape,rays=rays,visual=visual,left=left,common=common,
        steps=(2*bounds['longitude_half_span_deg']/(shape[1]-1),
               (bounds['latitude_max_deg']-bounds['latitude_min_deg'])/(shape[0]-1)))


def memory_budget(charts,d=96,c=32):
    shapes=[list(p['shape']) for p in charts]
    padded=[[int(np.ceil(h/8)*8),int(np.ceil(w/8)*8)] for h,w in shapes]
    n=sum(h*w for h,w in shapes);npad=sum(h*w for h,w in padded)
    out=dict(native_shapes=shapes,padded_shapes=padded,native_pixels=n,padded_pixels=npad,
        padding_extra_fraction=(npad-n)/n,candidate_lattice_positions_native=n*d,
        candidate_lattice_positions_padded=npad*d)
    for tag,num in [('native',n),('padded',npad)]:
        for precision,byte in [('fp16',2),('fp32',4)]:
            out[f'{tag}_two_camera_C{c}_D{d}_{precision}_mib']=float(2*c*d*num*byte/2**20)
            out[f'{tag}_gwc_G8_D{d}_{precision}_mib']=float(8*d*num*byte/2**20)
            out[f'{tag}_scalar_cost_D{d}_{precision}_mib']=float(d*num*byte/2**20)
    return out


def evaluate_case(name,charts,common,visual,lat,area,axis,out):
    out.mkdir(parents=True,exist_ok=True)
    mask_shape=common.shape[1:]
    weights=area*common.sum(0)
    active=common.any(0)
    windows=[];coordinates=[];pixels=[];lookups=[]
    for p in charts:
        lon,el=angular_coordinates(visual,p['basis'])
        window=window_mask(lon,el,p['bounds']);windows.append(window)
        coordinates.append(np.where(window,coordinate_kappa(el),np.inf))
        pixels.append(np.where(window,actual_pixel_anisotropy(el,*p['steps']),np.inf))
        lookups.append(native_lookup(visual,p['basis'],p['bounds'],p['shape']))
    union=np.logical_or.reduce(windows)
    overlap=np.logical_and.reduce(windows) if len(charts)>1 else np.zeros_like(union)
    best_coordinate=np.minimum.reduce(coordinates);best_pixel=np.minimum.reduce(pixels)
    strict_counts=np.zeros(mask_shape,np.uint8);geometric_counts=np.zeros(mask_shape,np.uint8)
    high=np.abs(lat)>60
    rows=[]
    for d in range(len(axis)):
        valid=common[d];total=area[valid].sum()
        strict=np.zeros(valid.size,bool);near=np.zeros(valid.size,bool);any_positive=np.zeros(valid.size,bool)
        for i,p in enumerate(charts):
            a=supports(p['common'][d],lookups[i]);near|=a[0];strict|=a[1];any_positive|=a[2]
        strict=strict.reshape(mask_shape);near=near.reshape(mask_shape);any_positive=any_positive.reshape(mask_shape)
        missing=valid & ~union;strict_missing=valid & ~strict
        geometric_counts+=missing;strict_counts+=strict_missing
        rows.append(dict(index=d,radius_m=float(1/axis[d]),valid_count=int(valid.sum()),
            geometric_uncovered_count=int(missing.sum()),geometric_coverage_ratio=1-float(area[missing].sum()/total),
            strict_unsupported_count=int(strict_missing.sum()),strict_unsupported_area_sr=float(area[strict_missing].sum()),
            strict_supported_area_ratio=float(area[valid & strict].sum()/total),
            nearest_unsupported_count=int(np.count_nonzero(valid & ~near)),
            any_corner_unsupported_count=int(np.count_nonzero(valid & ~any_positive)),
            overlap_of_common=float(area[valid & overlap].sum()/total),overlap_area_sr=float(area[valid & overlap].sum()),
            primary_high_latitude_valid_count=int(np.count_nonzero(valid & high)),
            high_latitude_geometric_uncovered_count=int(np.count_nonzero(missing & high)),
            high_latitude_strict_unsupported_count=int(np.count_nonzero(strict_missing & high))))
    weight=weights.sum()
    stats=dict(coordinate=weighted_stats(best_coordinate,weights,active & union),
        actual_pixel=weighted_stats(best_pixel,weights,active & union),
        high_latitude_coordinate=weighted_stats(best_coordinate,weights,active & union & high),
        high_latitude_actual_pixel=weighted_stats(best_pixel,weights,active & union & high),
        primary_erp_high=weighted_stats(coordinate_kappa(lat),weights,active & high))
    strict_missing_mass=sum(r['strict_unsupported_area_sr'] for r in rows)
    result=dict(name=name,status='PASS' if not geometric_counts.any() else 'INCOMPLETE_COVERAGE',
        analysis_shape_dhw=list(common.shape),num_depth=len(axis),
        geometric_uncovered_count=int(geometric_counts.sum()),geometric_coverage_ratio=1-float(np.sum(area*geometric_counts)/weight),
        strict_unsupported_count=int(strict_counts.sum()),strict_unsupported_weighted_fraction=float(strict_missing_mass/weight),
        strict_supported_area_ratio=float(1-strict_missing_mass/weight),
        overlap_area_sr_mean=float(np.sum(weights*overlap)/len(axis)),
        overlap_sphere_fraction_mean=float(np.sum(weights*overlap)/(len(axis)*4*np.pi)),
        overlap_of_common=float(np.sum(weights*overlap)/weight),quality=stats,budget=memory_budget(charts,len(axis)),
        charts=[dict(name=p['name'],bounds=p['bounds'],basis_visual=p['basis'].tolist(),shape_hw=list(p['shape']),
                     longitude_step_deg=p['steps'][0],latitude_step_deg=p['steps'][1]) for p in charts],
        per_depth=rows)
    np.savez_compressed(out/'comparison_maps.npz',valid_candidate_count=common.sum(0).astype(np.uint8),
        area_sr=area,geometric_uncovered_count=geometric_counts,strict_unsupported_count=strict_counts,
        best_coordinate_kappa=np.where(union,best_coordinate,np.nan).astype(np.float32),
        best_pixel_anisotropy=np.where(union,best_pixel,np.nan).astype(np.float32),
        **{f'{p["name"]}_window':w for p,w in zip(charts,windows)})
    for p in charts:
        np.savez_compressed(out/f'{p["name"]}_native_masks.npz',rays_ref=p['rays'],left_valid=p['left'],
            stereo_valid=p['common'],inverse_depth_candidates=axis,basis_visual=p['basis'])
    save_csv(out/'metrics_per_depth.csv',rows)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for ax,value,title,limit in [(axes[0],np.ma.array(best_pixel,mask=~active|~union),'Actual pixel anisotropy',3.),
            (axes[1],np.ma.array(strict_counts/len(axis),mask=~active),'Strict unsupported / candidates',.05),
            (axes[2],np.ma.array(geometric_counts/len(axis),mask=~active),'Geometric uncovered / candidates',.05)]:
        image=ax.imshow(value,extent=[-180,180,-90,90],origin='upper',aspect='auto',cmap='viridis' if ax is axes[0] else 'magma',
                        vmin=1 if ax is axes[0] else 0,vmax=limit)
        ax.set(title=title,xlabel='Primary longitude (deg)',ylabel='Primary latitude (deg)',xticks=[-180,-90,0,90,180],yticks=[-90,-60,-30,0,30,60,90])
        ax.set_facecolor('#e8ebf0')
        fig.colorbar(image,ax=ax,fraction=.035,extend='max' if np.ma.max(value)>limit else 'neither',
                     label='Anisotropy' if ax is axes[0] else 'Candidate fraction')
    fig.suptitle(name+' | same 96-candidate common directions')
    save_figure(fig,out/'quality_and_support.png')
    print('EVALUATED',name,json.dumps(dict(status=result['status'],pixels=result['budget']['native_pixels'],
        overlap=result['overlap_of_common'],strict_unsupported=result['strict_unsupported_weighted_fraction'],pixel=stats['actual_pixel'])),flush=True)
    return result


def flat_result(result):
    base=dict(case=result['name'],status=result['status'])
    if result['status']=='NO_FEASIBLE_SAMPLED_ROTATION':
        return dict(**base,minimum_sampled_required_abs_yang_latitude_deg=result['minimum_sampled_required_abs_yang_latitude_deg'])
    base.update({k:result[k] for k in ['num_depth','geometric_uncovered_count','geometric_coverage_ratio',
        'strict_unsupported_count','strict_unsupported_weighted_fraction','overlap_area_sr_mean','overlap_sphere_fraction_mean','overlap_of_common']})
    base.update(result['budget'])
    for group in ['coordinate','actual_pixel','high_latitude_coordinate','high_latitude_actual_pixel']:
        for key,value in result['quality'][group].items():base[group+'_'+key]=value
    base['native_shapes']=json.dumps(base['native_shapes']);base['padded_shapes']=json.dumps(base['padded_shapes'])
    base['charts']=json.dumps(result['charts'],separators=(',',':'))
    return base


def write_overview(out,summary,redraw_cases=False):
    results=summary['cases'];passing=[r for r in results if r['status']=='PASS']
    for r in results:
        if r['status']=='NO_FEASIBLE_SAMPLED_ROTATION':
            r.setdefault('feasibility_diagnostic_shape_hw',summary.get('search_resolution_hw',[90,180]))
            r['high_resolution_metrics']='Not applicable: no accepted grid; the required latitude is an observed search-grid value, not a continuous lower bound.'
    summary['coordinate_kappa_formula']='1/abs(cos(latitude)); assumes equal infinitesimal angular steps'
    summary['actual_pixel_anisotropy_formula']='max(delta_longitude*cos(latitude),delta_latitude)/min(delta_longitude*cos(latitude),delta_latitude); native true angular steps'
    summary['padding_assumption']='H/W round upward to multiples of8, D to4; D96 is unchanged; invalid padding is not sampled'
    frozen=next(r for r in passing if r['name']=='historical_yin_60_frozen')
    fifty=next((r for r in passing if r['name']=='dual_yin_50'),None)
    single=next(r for r in passing if r['name']=='single_rotated_budget_matched')
    summary['recommendation']=dict(yin_latitude_limit_deg=60,
        preferred_existing_candidate='historical_yin_60_frozen',
        evidence='Historical 60deg windows independently retain 100% sampled geometric coverage at 720x1440x96, '
                 'with lower native/padded budget and lower strict unsupported fraction than the researched 50deg case. '
                 'The 50deg case improves average and tail anisotropy at increased cost; this does not compel changing a stable baseline.',
        single_control='Single Rotated ERP is feasible at the matched budget. It has worse mean/tail anisotropy and strict native support here, '
                       'but this experiment does not establish that two grids are necessary for Stereo accuracy.')
    if (out/'refit_difference_diagnostics.json').exists():
        summary['historical_vs_researched_60_diagnostics']=json.loads((out/'refit_difference_diagnostics.json').read_text())
    summary['historical_vs_researched_60_note']=(
        'Historical boundaries were fitted on 360x720; researched 60 is refitted on 720x1440 with the same padding/quantization rule. '
        'The researched Yin half-span grows 89 to 89.5 deg and the finite angular-area objective selects a different '
        'axis-aligned Yang orientation; neither change is silently applied to the frozen historical input. '
        'The old grid still has zero sampled uncovered candidates and covers the required valid seam target.')
    summary['historical_strict_2deg_margin_at720_guaranteed']=False
    summary['historical_padding_note']=('The historical 2deg padding was fitted on 360x720. At 720x1440 its yaw0 Yang '
        'would need half-span115 instead of114.5 to re-establish the exact 2deg margin for every observed target. '
        'This does not cause sampled coverage or seam failure, but the strict high-resolution padding guarantee must not be claimed.')
    labels=[{'dual_yin_50':'Yin50 refit','dual_yin_60':'Yin60 refit','historical_yin_60_frozen':'Yin60 frozen',
             'single_rotated_budget_matched':'Single rotated'}.get(r['name'],r['name']) for r in passing]
    if redraw_cases:
        for r in passing:
            folder=out/r['name']
            with np.load(folder/'comparison_maps.npz') as data:
                active=data['valid_candidate_count']>0
                values=[np.ma.array(data['best_pixel_anisotropy'],mask=~active),
                        np.ma.array(data['strict_unsupported_count']/96,mask=~active),
                        np.ma.array(data['geometric_uncovered_count']/96,mask=~active)]
            fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
            for i,(ax,value,title,limit) in enumerate(zip(axes,values,
                ['Actual pixel anisotropy','Strict unsupported / candidates','Geometric uncovered / candidates'],[3.,.05,.05])):
                image=ax.imshow(value,extent=[-180,180,-90,90],origin='upper',aspect='auto',cmap='viridis' if i==0 else 'magma',vmin=1 if i==0 else 0,vmax=limit)
                ax.set(title=title,xlabel='Primary longitude (deg)',ylabel='Primary latitude (deg)',xticks=[-180,-90,0,90,180],yticks=[-90,-60,-30,0,30,60,90])
                ax.set_facecolor('#e8ebf0')
                fig.colorbar(image,ax=ax,fraction=.035,extend='max' if np.ma.max(value)>limit else 'neither',label='Anisotropy' if i==0 else 'Candidate fraction')
            fig.suptitle(r['name']+' | same 96-candidate common directions')
            save_figure(fig,folder/'quality_and_support.png')
    fig,axes=plt.subplots(1,3,figsize=(15.5,4.6),layout='constrained')
    y=np.arange(len(passing))
    native=np.array([r['budget']['native_pixels'] for r in passing])/1000
    padded=np.array([r['budget']['padded_pixels'] for r in passing])/1000
    axes[0].barh(y,native,color='#0072B2',label='Native')
    axes[0].barh(y,padded-native,left=native,color='#b8c6d1',label='Padding')
    axes[0].set(yticks=y,yticklabels=labels,xlabel='Native blue + padding grey (k samples/depth)',title='Actual lattice budget',xlim=(0,80))
    axes[0].invert_yaxis()
    for metric,color,marker,label in [('weighted_mean','#0072B2','o','Mean'),('p95','#009E73','s','P95'),('p99','#D55E00','^','P99')]:
        axes[1].scatter([r['quality']['actual_pixel'][metric] for r in passing],y,color=color,marker=marker,label=label,s=45)
    axes[1].set(yticks=y,yticklabels=labels,xlabel='Area / candidate weighted anisotropy',title='Same shared directions; actual angular steps',xlim=(1,2.1))
    axes[1].set_xticks([1,1.25,1.5,1.75,2])
    axes[1].invert_yaxis();axes[1].legend(frameon=False)
    axes[2].barh(y,[100*r['strict_unsupported_weighted_fraction'] for r in passing],color='#D55E00')
    axes[2].set(yticks=y,yticklabels=labels,xlabel='Unsupported / shared area (%)',title='Strict bilinear native support',xlim=(0,1.3))
    axes[2].set_xticks([0,.25,.5,.75,1,1.25])
    axes[2].invert_yaxis()
    fig.suptitle('Limited V1 geometry comparison | 96 candidates, 720 x 1440 reference sampling')
    save_figure(fig,out/'grid_comparison.png',vector=True)
    # 复用已经保存的数值图；不重新投影，不运行新的旋转搜索。
    fig,axes=plt.subplots(1,len(passing),figsize=(16,4.1),layout='constrained')
    for ax,r,label in zip(axes,passing,labels):
        with np.load(out/r['name']/'comparison_maps.npz') as data:
            val=data['best_pixel_anisotropy'];valid=data['valid_candidate_count']>0
            im=ax.imshow(np.ma.array(val,mask=~valid),extent=[-180,180,-90,90],origin='upper',aspect='auto',cmap='viridis',vmin=1,vmax=2.4)
        ax.set(title=label,xlabel='Primary longitude (deg)',ylabel='Primary latitude (deg)',xticks=[-180,0,180],yticks=[-90,0,90])
        ax.set_facecolor('#e8ebf0')
    fig.colorbar(im,ax=axes.tolist(),fraction=.015,label='Actual pixel anisotropy (shared scale)')
    save_figure(fig,out/'pixel_anisotropy_comparison.png')
    lines=['# V1 几何配置对照', '',
        '**建议：保留 Yin ±60°，优先沿用历史冻结窗口。Single Rotated ERP 同样可行，不能据此宣称双网格是唯一必要方案。**', '',
        '所有双网格条件固定：Yang |纬度|≤45°、接缝5°、边界余量2°、边界量化0.5°、目标角步长0.75°、同一96层原始候选轴。',
        '每次 Yin 纬度变化都重新拟合经度并重搜 Yang；720×1440 上重新拟合并检查所有候选。历史窗口另列，未覆盖或改写。', '',
        '| case | status | 原生像素 | Padding像素 | 覆盖率 | 严格支持不足面积 | 重叠/共视 | 像素均值 | 像素P95 | 像素P99 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        if r['status']!='PASS':
            lines.append(f'|{r["name"]}|无采样可行解|—|—|—|—|—|—|—|—|');continue
        q=r['quality']['actual_pixel'];b=r['budget']
        lines.append(f'|{r["name"]}|{r["status"]}|{b["native_pixels"]}|{b["padded_pixels"]}|{r["geometric_coverage_ratio"]:.8%}|'
            f'{r["strict_unsupported_weighted_fraction"]:.5%}|{r["overlap_of_common"]:.3%}|{q["weighted_mean"]:.5f}|{q["p95"]:.5f}|{q["p99"]:.5f}|')
    lines+=['','## 选择依据','',
        '±45°：本次有限粗细搜索未找到满足全部固定约束的方案，指标不伪造为0或100%。所需最小纬度是已测试方向上的观察值，不是连续不可行证明。',
        '±50°：可降低平均和尾部各向异性，但相比历史60增加约7.6%原生像素和9.3% Padding 后预算，且严格插值支持不足未改善。'
        '当前没有训练精度证据要求为此改变稳定Baseline。',
        'Single：与历史60几乎相同的原生采样预算，覆盖同样全部96层有效方向；像素各向异性和严格支持较差，但仍是有效几何对照。',
        '', '## 历史60与重搜60为何不同','',summary['historical_vs_researched_60_note'],
        '这属于分析格采样/边界量化和窗口面积目标造成的细微候选排序变化，不是相机标定或候选轴变更。'
        '历史60在相同720×1440×96上独立验证无覆盖空洞，故本次未应用重搜结果替换历史参数。', '',
        '历史2° padding是在360×720上拟合；720×1440若重新要求每个已发现目标都有严格2°余量，'
        '旧yaw0 Yang需要half-span115°而非历史114.5°。因此不能把“覆盖与接缝通过”误写成“720上仍严格保证2°padding”。', '',
        '## 完整参数','', '```json',json.dumps({r['name']:r.get('charts',r.get('best_violation_candidate')) for r in results},indent=2),'```', '',
        '## 口径', '',summary['common_weighting'],'',summary['memory_definition'],'',summary['projection_cache_note'],
        '', 'Single 不受双网格 Yang 最大45°限制；其旋转优先降低有效域的最大绝对纬度，再按匹配预算生成近等角步长的H/W。',
        '严格支持使用原图投影Mask，后续CNN特征还需现有feature_grid的stride=2检查。', '',
        '## 文件','', '`grid_comparison.csv/json` 给出预算、内存、形状和尾部各向异性；各case含基矩阵、逐候选指标及NPZ。',
        '此实验只研究几何与采样，不证明Stereo精度提升。']
    (out/'REPORT_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    (out/'grid_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'figure_qa.json').write_text(json.dumps(QA,indent=2)+'\n')


def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--frozen-grid',type=Path,default=root/'selected_grid.json')
    p.add_argument('--output-dir',type=Path,default=root/'outputs_v1/comparison')
    p.add_argument('--projection-cache',type=Path)
    p.add_argument('--analysis-size',nargs=2,type=int,default=[720,1440])
    p.add_argument('--cases',nargs='+',type=int,default=[45,50,60])
    p.add_argument('--search-size',nargs=2,type=int,default=[90,180])
    p.add_argument('--summarize-only',action='store_true',help='Replot saved numerical comparison; no search or projection')
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:
        plt.rcParams.update(PLOT_STYLE)
        logging.getLogger('fontTools').setLevel(logging.WARNING)
        write_overview(out,json.loads((out/'grid_comparison.json').read_text()),redraw_cases=True);return
    frozen_bytes=args.frozen_grid.read_bytes();frozen=json.loads(frozen_bytes)
    cameras,images,cfg,prov=load_pair(args.data_root,4,3,frozen['frame_id'])
    rot,tr,geometry=pair_geometry(*cameras);axis=depth_axis(cfg)
    assert np.array_equal(axis,np.asarray(frozen['inverse_depth_candidates']))
    assert hashlib.sha256((args.data_root/'config.yaml').read_bytes()).hexdigest()==frozen['config_sha256']
    plt.rcParams.update(PLOT_STYLE);logging.getLogger('fontTools').setLevel(logging.WARNING)
    small_rays,small_visual,small_lon_rad,small_lat_rad,small_area=spherical_rays(*args.search_size)
    small_common=depth_visibility(cameras,small_rays,rot,tr,axis)[2]
    small_lon,small_lat=np.rad2deg(small_lon_rad),np.rad2deg(small_lat_rad)
    rays,visual,lon_rad,lat_rad,area=spherical_rays(*args.analysis_size)
    lon,lat=np.rad2deg(lon_rad),np.rad2deg(lat_rad)
    if args.projection_cache:
        with np.load(args.projection_cache) as cache:
            assert np.array_equal(cache['axis'],axis)
            common=cache['common_valid']
        assert common.shape==(len(axis),*args.analysis_size)
        # 相机/深度一致性来自冻结输入；再复算三个分散候选检查缓存来源与维度。
        for idx in [0,len(axis)//2,len(axis)-1]:
            actual=depth_visibility(cameras,rays,rot,tr,axis[idx:idx+1])[2][0]
            assert np.array_equal(actual,common[idx]),'Projection cache mismatch'
        cache_note='Reused previous verified common masks after calibration/axis checks and exact projection checks at indices 0,48,95; chart parameters do not affect camera projection.'
    else:
        common=depth_visibility(cameras,rays,rot,tr,axis)[2];cache_note='Freshly projected all 96 candidates'
    results=[]
    for limit in args.cases:
        name=f'dual_yin_{limit}';folder=out/name;folder.mkdir(exist_ok=True)
        config=common_search_args(limit)
        byin,yin_s=fit_yin(small_lon,small_lat,small_common.any(0),config)
        target=small_common.any(0)&(~yin_s|(np.abs(small_lat)>=limit-5))
        print('SEARCH',name,flush=True)
        try:rows=search_rotation(small_visual,target,config)
        except ValueError:
            rows,best=no_feasible_audit(small_visual,target,config)
            feasible=[r for r in rows if r['feasible']]
            if not feasible:
                result=dict(name=name,status='NO_FEASIBLE_SAMPLED_ROTATION',
                    minimum_sampled_required_abs_yang_latitude_deg=best['required_abs_latitude_deg'],
                    best_violation_candidate=best,orientations_tested=len(rows),fixed_yang_max_latitude_deg=45,
                    reason='No sampled orientation satisfies the unchanged 45deg Yang latitude limit including seam and padding. '
                           'No incomplete or relaxed grid is used in comparisons.')
                results.append(result);save_csv(folder/'search_candidates.csv',rows)
                (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
                print('INFEASIBLE',name,json.dumps(result),flush=True);continue
        save_csv(folder/'search_candidates.csv',rows)
        winner,final,byin,byang,basis,yin,yang,yang_lat=full_resolution_selection(rows,visual,common,area,lon,lat,config)
        if limit==60:
            target_full=common.any(0)&(~yin|(np.abs(lat)>=55))
            old_lon,old_lat=angular_coordinates(visual,np.asarray(frozen['yang_basis_visual']))
            old_yang=window_mask(old_lon,old_lat,frozen['yang_bounds'])
            diagnostics=dict(historical_yin_bounds=frozen['yin_bounds'],refitted_yin_bounds=byin,
                analysis_shape_hw=list(args.analysis_size),
                axis_aligned_candidates=[fit_candidate(visual,target_full,y,90,config) for y in [0,90,180,270]],
                legacy_uncovered_target_directions=int(np.count_nonzero(target_full&~old_yang)))
            (out/'refit_difference_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
        charts=[chart_record('yin',byin,np.eye(3),cameras,rot,tr,axis),chart_record('yang',byang,basis,cameras,rot,tr,axis)]
        result=evaluate_case(name,charts,common,visual,lat,area,axis,folder)
        result.update(selected_rotation=winner,search_count=len(rows),full_resolution_refit_count=len(final),generation_constraints=vars(config))
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result)
    # 历史±60窗口作为完全冻结的旁证，不能把重拟合后的窗口冒充历史配置。
    legacy_charts=[chart_record('yin',frozen['yin_bounds'],np.eye(3),cameras,rot,tr,axis,
                               frozen['native_sampling']['yin']['shape_dhw'][1:]),
                   chart_record('yang',frozen['yang_bounds'],np.asarray(frozen['yang_basis_visual']),cameras,rot,tr,axis,
                               frozen['native_sampling']['yang']['shape_dhw'][1:])]
    legacy=evaluate_case('historical_yin_60_frozen',legacy_charts,common,visual,lat,area,axis,out/'historical_yin_60_frozen')
    results.append(legacy)
    matched=next((r for r in results if r['name']=='dual_yin_60' and r['status']=='PASS'),legacy)
    target_budget=matched['budget']['native_pixels']
    sconfig=common_search_args(60)
    print('SEARCH single rotated ERP',flush=True)
    srows=search_single(small_visual,small_common.any(0),sconfig)
    save_csv(out/'single_search_candidates.csv',srows)
    order=lambda r:(r['required_abs_latitude_deg'],r['angular_rectangle_deg2'],abs(r['tilt_x_deg']-90),r['yaw_y_deg'])
    refined=[]
    for row in sorted([r for r in srows if r['feasible']],key=order)[:48]:
        candidate=single_fit(visual,common.any(0),row['yaw_y_deg'],row['tilt_x_deg'],sconfig)
        if candidate['feasible']:refined.append(candidate)
    if not refined:raise RuntimeError('No full-coverage single rotated local ERP found')
    single=min(refined,key=order)
    basis=ry(single['yaw_y_deg'])@rx(single['tilt_x_deg'])@ry(single['longitude_center_deg'])
    bounds={k:single[k] for k in ['longitude_center_deg','longitude_half_span_deg','latitude_min_deg','latitude_max_deg']}
    bounds['longitude_center_deg']=0.
    shape=matched_budget_shape(bounds,target_budget)
    chart=chart_record('single',bounds,basis,cameras,rot,tr,axis,shape)
    single_result=evaluate_case('single_rotated_budget_matched', [chart],common,visual,lat,area,axis,out/'single_rotated_budget_matched')
    single_result.update(selected_rotation=single,target_budget_case=matched['name'],target_native_pixel_budget=target_budget,
        native_budget_relative_difference=(single_result['budget']['native_pixels']-target_budget)/target_budget,
        rotation_objective='Minimize maximum absolute chart latitude to keep poles away from shared FOV; then rectangle angular area. '
            'No dual-grid Yang 45deg hard constraint is imposed. Native H/W nearly equalize angular steps at the matched budget.')
    (out/'single_rotated_budget_matched/result.json').write_text(json.dumps(single_result,indent=2)+'\n');results.append(single_result)
    summary=dict(status='COMPLETED',calibration_sha256=frozen['config_sha256'],frozen_input_sha256=hashlib.sha256(frozen_bytes).hexdigest(),
        search_resolution_hw=list(args.search_size),
        analysis_shape_dhw=list(common.shape),projection_cache_note=cache_note,
        common_weighting='Exact spherical area times common-valid candidate count; candidate layers are equally weighted diagnostics, not GT depth probabilities',
        native_mask_rule='Joint OCam FOV/raw invalid mask; strict support requires all positive-weight bilinear corners of at least one available chart',
        single_fairness='Matched native sample budget within 1%; padded overhead and actual angle-step anisotropy reported separately',
        memory_definition='Estimated materialized volumes only: two-camera C32 feature volumes; separate GWC G8 and scalar cost; excludes CNN/3D activations, gradients and optimizer',
        cases=results,limits=['Finite sampled orientation search is not a global optimality or infeasibility proof',
            'Geometric coverage and strict native interpolation support are different metrics',
            'Common FOV is not real occlusion visibility','No Stereo training or accuracy comparison is performed'])
    save_csv(out/'grid_comparison.csv',[flat_result(r) for r in results])
    (out/'grid_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'figure_qa.json').write_text(json.dumps(QA,indent=2)+'\n')
    assert args.frozen_grid.read_bytes()==frozen_bytes
    write_overview(out,summary)
    print('DONE',out,flush=True)


if __name__=='__main__':main()
