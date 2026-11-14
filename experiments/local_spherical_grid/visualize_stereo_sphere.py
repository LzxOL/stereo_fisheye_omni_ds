"""OmniThings 双鱼眼球面覆盖与有限局部补偿网格；仅 CPU 几何及 RGB 可视化。"""
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from plotly.offline import get_plotlyjs
import plotly.graph_objects as go

from camera_geometry import (load_pair, pair_geometry, spherical_rays, sphere_projection,
    complementary_coordinates, coverage_statistics, validate_geometry, OCAM_FROM_VIS)

COLORS = np.array([[232, 235, 240], [0, 114, 178], [0, 158, 115], [155, 81, 185]], np.uint8)
PLOT_STYLE = {'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
              'axes.spines.right': False, 'savefig.dpi': 180, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
QA = []


def save_figure(fig, path, vector=False):
    """确定性检查文字是否出画布，并保存浏览图；最终仍需人工读图。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width > 0 and box.height > 0 and (box.x0 < -2 or box.y0 < -2 or
                box.x1 > fig.bbox.width+2 or box.y1 > fig.bbox.height+2):
            outside.append(text.get_text())
    QA.append({'figure': str(path.name), 'text_outside_canvas': outside})
    fig.savefig(path, facecolor='white')
    if vector:
        fig.savefig(path.with_suffix('.pdf'), facecolor='white')
        fig.savefig(path.with_suffix('.svg'), facecolor='white')
        Image.open(path).convert('L').save(path.with_name(path.stem+'_grayscale.png'))
    plt.close(fig)


def map_axis(ax, title):
    ax.set(title=title, xlabel='Longitude (deg; positive = image-left)', ylabel='Latitude (deg)')
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(alpha=.25, linewidth=.5)


def plot_maps(folder, item, kp, kc, patch, phi_c, threshold, label, ref, target):
    folder.mkdir(parents=True, exist_ok=True)
    extent = (-180, 180, -90, 90)
    overlap = item['overlap']
    cmap = ListedColormap(COLORS/255)
    code = item['left'].astype(int) + 2*item['right'].astype(int)
    for side, cam in [('left', ref), ('right', target)]:
        fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
        rgb = item[f'{side}_rgb'].copy()
        rgb[~item[side]] = COLORS[0]
        ax.imshow(rgb, extent=extent, origin='upper', aspect='auto')
        map_axis(ax, f'cam{cam} RGB on reference sphere | {label}')
        save_figure(fig, folder/f'spherical_rgb_cam{cam}.png')
        Image.fromarray(item[f'{side}_rgb']).save(folder/f'spherical_rgb_cam{cam}_raw.png')
        Image.fromarray((item[side]*255).astype('uint8')).save(folder/f'valid_cam{cam}_raw.png')

    fig, ax = plt.subplots(figsize=(11, 5.8), layout='constrained')
    ax.imshow(code, cmap=cmap, norm=BoundaryNorm(np.arange(-.5, 4), 4),
              extent=extent, origin='upper', aspect='auto', interpolation='nearest')
    map_axis(ax, f'Solid-angle coverage | {label}')
    ax.legend(handles=[Patch(facecolor=COLORS[i]/255, label=name) for i, name in enumerate(
        ['Uncovered', f'cam{ref} only', f'cam{target} only', 'Both cameras'])],
        loc='upper center', bbox_to_anchor=(.5, -.18), ncol=4, frameon=False)
    save_figure(fig, folder/'erp_coverage.png')

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), layout='constrained')
    for ax, mask, name in zip(axes, [item['left'], item['right'], overlap],
                              [f'cam{ref} valid', f'cam{target} valid', 'Common valid / FOV only']):
        ax.imshow(mask, cmap='Greys', vmin=0, vmax=1, extent=extent, origin='upper', aspect='auto')
        map_axis(ax, name)
    fig.suptitle(label)
    save_figure(fig, folder/'spherical_overlap.png')
    Image.fromarray((overlap*255).astype('uint8')).save(folder/'overlap_raw.png')

    best = np.where(patch, np.minimum(kp, kc), kp)
    for values, mask, name, title in [(kp, overlap, 'distortion_primary', 'Primary ERP'),
            (kc, overlap & patch, 'distortion_complementary', 'Complementary LOCAL patch'),
            (best, overlap, 'distortion_best', 'Lower kappa; outside patch = primary')]:
        fig, ax = plt.subplots(figsize=(11, 5.8), layout='constrained')
        image = ax.imshow(np.ma.array(values, mask=~mask), extent=extent, origin='upper', aspect='auto',
                          cmap='viridis', norm=LogNorm(vmin=1, vmax=8))
        ax.set_facecolor(COLORS[0]/255)
        map_axis(ax, f'{title} | common valid directions | {label}')
        bar = fig.colorbar(image, ax=ax, orientation='horizontal', pad=.14, fraction=.06, extend='max',
                     ticks=[1, 1.4, 2, 4, 8], label='Anisotropy kappa = 1 / |cos(latitude)| (dimensionless)')
        bar.set_ticks([1, 1.4, 2, 4, 8], labels=['1', '1.4', '2', '4', '8'])
        bar.minorticks_off()
        # 等值线只画在双目交集，不把未覆盖区域算入改进。
        ax.contour(np.ma.array(kp, mask=~overlap), levels=[threshold], extent=extent,
                   origin='upper', colors='white', linewidths=.8)
        save_figure(fig, folder/f'{name}.png')
    fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
    layer = np.ma.array(np.abs(np.rad2deg(phi_c)), mask=~(overlap & (kp > threshold)))
    image = ax.imshow(layer, extent=extent, origin='upper', aspect='auto', cmap='cividis', vmin=0, vmax=90)
    ax.set_facecolor(COLORS[0]/255)
    map_axis(ax, f'Primary high-distortion directions: |latitude in C| | {label}')
    fig.colorbar(image, ax=ax, label='Absolute complementary latitude (deg)')
    save_figure(fig, folder/'primary_high_to_complement_latitude.png')


def plot_statistics(out, stats, angular, area, phi, phi_c, overlap, kp, threshold, ref, target):
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    radii = [x['radius_m'] for x in stats]
    for field, name, color, marker in [('left_fraction', f'cam{ref} coverage', '#0072B2', 'o'),
            ('right_fraction', f'cam{target} coverage', '#009E73', 's'),
            ('overlap_fraction', 'Common coverage', '#9B51B9', '^')]:
        line, = ax.plot(radii, [100*x[field] for x in stats], color=color, marker=marker, label=name)
        ax.axhline(100*angular[field], color=color, linestyle='--', alpha=.7, linewidth=1)
    ax.set(xscale='log', xlabel=f'Radius about cam{ref} centre (m)', ylabel='Area / full sphere (%)',
           title='Finite-radius coverage; dashed = angular limit', ylim=(0, 100))
    ax.set_xticks(radii, [str(r) for r in radii])
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    save_figure(fig, out/'coverage_vs_radius.png', vector=True)

    high = overlap & (kp > threshold)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    if high.any():
        weights = area[high] / area[high].sum()*100
        axes[0].hist(np.abs(np.rad2deg(phi[high])), bins=np.arange(0, 91, 5), weights=weights,
                     histtype='step', linewidth=2, label='Primary latitude', color='#0072B2')
        axes[0].hist(np.abs(np.rad2deg(phi_c[high])), bins=np.arange(0, 91, 5), weights=weights,
                     histtype='step', linewidth=2, linestyle='--', label='Complementary latitude', color='#D55E00')
    axes[0].set(xlabel='Absolute latitude (deg)', ylabel='Primary-high common area per bin (%)',
                title='Same directions in two coordinate frames', xlim=(0, 90))
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0, max(5., np.ceil(axes[0].get_ylim()[1]/5)*5))
    primary = np.array([angular['kappa_primary']['mean'], angular['kappa_primary']['p95']])
    best = np.array([angular['kappa_best_with_patch']['mean'], angular['kappa_best_with_patch']['p95']])
    for i, name in enumerate(['Area-weighted mean', 'Area-weighted p95']):
        axes[1].plot([0, 1], [primary[i], best[i]], marker=['o', 's'][i], label=name)
    axes[1].set(xticks=[0, 1], xticklabels=['Primary only', 'Lower kappa in patch'], ylabel='Kappa (dimensionless)',
                title='Angular common area; actual finite patch', ylim=(0, np.ceil(max(primary)*1.15)))
    axes[1].legend(frameon=False)
    save_figure(fig, out/'complementary_statistics.png', vector=True)


def grid_lines(basis=None, lon_limit=180., lat_limit=90.):
    segments = []
    for lat in np.arange(-lat_limit, lat_limit+.1, 15.):
        lon = np.linspace(-lon_limit, lon_limit, 241)
        phi, lam = np.deg2rad(np.full_like(lon, lat)), np.deg2rad(lon)
        segments.append(np.stack((np.cos(phi)*np.sin(lam), np.sin(phi), np.cos(phi)*np.cos(lam)), 1))
    for lon in np.arange(-lon_limit, lon_limit+.1, 30.):
        lat = np.linspace(-lat_limit, lat_limit, 181)
        phi, lam = np.deg2rad(lat), np.deg2rad(np.full_like(lat, lon))
        segments.append(np.stack((np.cos(phi)*np.sin(lam), np.sin(phi), np.cos(phi)*np.cos(lam)), 1))
    if basis is not None:
        segments = [p @ basis.T for p in segments]
    return [np.round(p, 5).tolist() for p in segments]


def camera_traces(report):
    def visual(p):
        return (OCAM_FROM_VIS.T @ np.array(p)).tolist()
    centers = [[0., 0., 0.], visual(report['target_center_reference_m'])]
    axes = [[0., 0., 1.], visual(report['target_optical_axis_reference'])]
    names = [f"cam{report['reference_id']} reference", f"cam{report['target_id']} target"]
    traces = []
    for p, axis, name, color in zip(centers, axes, names, ['#0072B2', '#009E73']):
        q = np.array(p) + .45*np.array(axis)
        traces.extend([dict(type='scatter3d', mode='markers+text', x=[p[0]], y=[p[1]], z=[p[2]],
                            text=[name], textposition='top center', marker=dict(size=7, color=color), name=name),
                       dict(type='scatter3d', mode='lines', x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]],
                            line=dict(color=color, width=7), name=name+' +Z optical axis')])
    traces.append(dict(type='scatter3d', mode='lines', x=[p[0] for p in centers], y=[p[1] for p in centers],
                       z=[p[2] for p in centers], line=dict(color='#333333', width=5, dash='dash'), name='Physical baseline'))
    rig = visual(report['rig_center_reference_m'])
    traces.append(dict(type='scatter3d', mode='markers+text', x=[rig[0]], y=[rig[1]], z=[rig[2]],
                       text=['Rig centre (not sphere centre)'], textposition='bottom center',
                       marker=dict(size=5, color='#777777', symbol='diamond'), name='Rig centre'))
    return traces


APP = r'''<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>cam4/cam3 sphere geometry</title>
<style>body{font-family:system-ui,sans-serif;margin:20px;color:#162535;background:#f7f9fc}h1{font-size:24px}
.panel{background:white;border-radius:12px;padding:15px;margin-bottom:12px}.controls{display:flex;flex-wrap:wrap;gap:16px}select{padding:6px}
#sphere{height:720px}#erp{height:420px}p{line-height:1.6}.legend{font-size:14px;color:#435269}label{white-space:nowrap}</style>
<script>__PLOTLY__</script>
<h1>OmniThings · cam__REF__ + cam__TGT__ · frame __FRAME__</h1>
<div class="panel"><div class="controls">
<label>模式 / Radius <select id="mode"></select></label>
<label>区域 <select id="view"><option value="both">Both Cameras</option><option value="left">cam__REF__ Only</option>
<option value="right">cam__TGT__ Only</option><option value="overlap">Overlap Mask</option>
<option value="left_only">cam__REF__ Only Region</option><option value="right_only">cam__TGT__ Only Region</option></select></label>
<label>着色 <select id="style"><option value="rgb">RGB 纹理</option><option value="coverage">Coverage Mask</option>
<option value="kp">Primary kappa</option><option value="kc">Complementary kappa (patch)</option>
<option value="best">Lower kappa (within patch)</option></select></label>
<label><input id="primary" type="checkbox" checked>主网格</label>
<label><input id="complement" type="checkbox" checked>局部补偿网格</label>
<label><input id="cameras" type="checkbox" checked>真实相机 / 基线</label></div>
<p id="note"></p><p id="stats" class="legend"></p>
<p class="legend">蓝：cam__REF__；绿：cam__TGT__；紫：共同有效；灰：未覆盖。主网格灰色，补偿网格橙色。
RGB 的 Both 模式仅作颜色均值展示；不是多视图融合或真实表面重建。拖动旋转，滚轮缩放。</p></div>
<div id="sphere" class="panel"></div><div id="erp" class="panel"></div>
<div class="panel"><p>坐标：所有投影在真实左 OCam 坐标完成。显示坐标通过绕光轴 180° 的旋转得到：+X 向图像左，+Y 向上，+Z 沿左光轴。
正经度指向图像左。Angular 是远场方向覆盖，忽略平移；Finite 使用完整外参的同一个 cam__REF__ 中心虚拟球面。
有效性是投影/FOV/原始 Mask 有效，不代表没有物体遮挡。</p>
<p>补偿 Patch：经度 ±__LON__°、纬度 ±__LAT__°。kappa 图仅显示共同有效方向；补偿图进一步受 Patch Mask 限制。
颜色范围 1–8，仅显示截断，JSON 统计未截断。未覆盖点不参与改善统计。0.5/1 m 低于数据扫描下限 __MINDEPTH__ m，仅作几何探针。</p></div>
<script>
const D=__DATA__, primaryGrid=__PRIMARY__, compGrid=__COMP__, cameraTraces=__CAMERAS__;
const palette=['rgb(232,235,240)','rgb(0,114,178)','rgb(0,158,115)','rgb(155,81,185)'];
const mode=document.getElementById('mode');D.modes.forEach((m,i)=>mode.add(new Option(m.name,i))); mode.value=__INITIAL__;
function selected(code,v){const l=!!(code&1),r=!!(code&2); return v==='left'?l:v==='right'?r:v==='overlap'?l&&r:v==='left_only'?l&&!r:v==='right_only'?r&&!l:l||r;}
function colorAt(i,m,v,s){const code=m.code[i],l=!!(code&1),r=!!(code&2);
 if(s==='kp')return D.kp[i];if(s==='kc')return D.kc[i];if(s==='best')return D.patch[i]?Math.min(D.kp[i],D.kc[i]):D.kp[i];
 if(s==='coverage'||v==='overlap')return palette[code];
 let a=m.lrgb[i],b=m.rrgb[i];if(v==='left'||v==='left_only'||!r)return `rgb(${a})`;
 if(v==='right'||v==='right_only'||!l)return `rgb(${b})`;
 return `rgb(${a.map((x,j)=>Math.round((x+b[j])/2))})`;}
function lineTrace(segments,r,color,name){let x=[],y=[],z=[];segments.forEach(a=>{a.forEach(p=>{x.push(r*p[0]);y.push(r*p[1]);z.push(r*p[2]);});x.push(null);y.push(null);z.push(null);});
 return {type:'scatter3d',mode:'lines',x,y,z,line:{color,width:1.5},name,hoverinfo:'skip'};}
function update(){const m=D.modes[+mode.value],v=document.getElementById('view').value,s=document.getElementById('style').value;
 const radius=m.radius||1, heat=s==='kp'||s==='kc'||s==='best';let x=[],y=[],z=[],color=[],hover=[];
 for(let i=0;i<D.rays.length;i++){const c=m.code[i];let use=selected(c,v);
 if(heat)use=use&&c===3&&(s!=='kc'||D.patch[i]); if(!use)continue;
 const p=D.rays[i];x.push(radius*p[0]);y.push(radius*p[1]);z.push(radius*p[2]);color.push(colorAt(i,m,v,s));
 hover.push(`lon ${D.lon[i].toFixed(1)}°, lat ${D.lat[i].toFixed(1)}°<br>kP ${D.kp[i].toFixed(2)}, kC ${D.kc[i].toFixed(2)}<br>common ${c===3}, patch ${!!D.patch[i]}`);}
 const marker={size:3,color,opacity:1};if(heat)Object.assign(marker,{colorscale:'Viridis',cmin:1,cmax:8,showscale:true,colorbar:{title:'kappa (1–8 display)'}});
 const traces=[{type:'scatter3d',mode:'markers',x,y,z,marker,text:hover,hoverinfo:'text',name:'Sphere samples'}];
 // 未覆盖方向只画稀疏浅灰底，绝不填入相机 RGB。
 if(!heat){let gx=[],gy=[],gz=[];for(let i=0;i<D.rays.length;i+=4){if(m.code[i]===0){const p=D.rays[i];gx.push(radius*p[0]);gy.push(radius*p[1]);gz.push(radius*p[2]);}}
 traces.unshift({type:'scatter3d',mode:'markers',x:gx,y:gy,z:gz,marker:{size:1.4,color:'#dce1e8',opacity:.2},hoverinfo:'skip',showlegend:false});}
 if(document.getElementById('primary').checked)traces.push(lineTrace(primaryGrid,radius*1.005,'#728093','Primary grid'));
 if(document.getElementById('complement').checked)traces.push(lineTrace(compGrid,radius*1.012,'#D55E00','Local complementary grid'));
 if(document.getElementById('cameras').checked)traces.push(...cameraTraces);
 Plotly.react('sphere',traces,{margin:{t:20,b:5,l:5,r:5},uirevision:'sphere',legend:{orientation:'h',y:1.04},
 scene:{aspectmode:'data',xaxis:{title:{text:'Xvis: image-left (m)'}},yaxis:{title:{text:'Yvis: up (m)'}},zaxis:{title:{text:'Zvis: ref-forward (m)'}},
 camera:{up:{x:0,y:1,z:0},eye:{x:1.65,y:.65,z:1.65}},bgcolor:'white'}},{responsive:true,scrollZoom:true});
 let raster=[];for(let row=0;row<D.height;row++){let a=[];for(let col=0;col<D.width;col++){const i=row*D.width+col,c=m.code[i];let use=selected(c,v);
 if(heat)use=use&&c===3&&(s!=='kc'||D.patch[i]);if(heat)a.push(use?colorAt(i,m,v,s):null);
 else {let rgb=use?colorAt(i,m,v,s):palette[0];a.push(rgb.match(/\d+/g).map(Number));}}raster.push(a);}
 const common={x0:D.lon[0],dx:D.lon[1]-D.lon[0],y0:D.lat[0],dy:D.lat[D.width]-D.lat[0]};
 const trace=heat?{type:'heatmap',z:raster,...common,colorscale:'Viridis',zmin:1,zmax:8,colorbar:{title:'kappa'},connectgaps:false}:
 {type:'image',z:raster,...common};
 Plotly.react('erp',[trace],{title:{text:'Same spherical directions in primary ERP coordinates'},margin:{t:45,b:55,l:65,r:60},
 xaxis:{title:{text:'Longitude (deg)'},range:[-180,180],dtick:60},yaxis:{title:{text:'Latitude (deg)'},range:[-90,90],dtick:30,autorange:false},paper_bgcolor:'white'},{responsive:true});
 document.getElementById('note').textContent=m.radius?`Finite radius ${m.radius} m：真实 R,t；球面中心为 cam__REF__，不是 Rig。`:
 'Angular coverage：只使用旋转，分析远场方向。显示物理相机位置用于理解布局；纹理投影忽略相机平移。';
 const a=m.stats;const latText=a.overlap_latitude_min_deg===null?'无共视方向':`${a.overlap_latitude_min_deg.toFixed(2)}° ~ ${a.overlap_latitude_max_deg.toFixed(2)}°`;
 document.getElementById('stats').textContent=`球面面积：左 ${(100*a.left_fraction).toFixed(2)}%，右 ${(100*a.right_fraction).toFixed(2)}%，交集 ${(100*a.overlap_fraction).toFixed(2)}%；交集/左 ${(100*a.overlap_of_left).toFixed(2)}%。共视纬度 ${latText}。`;
 document.getElementById('stats').textContent+=` 主高畸变 (kappa>${D.threshold}) 占共视面积 ${(100*a.high_primary_of_overlap).toFixed(2)}%；Patch 内择优后 ${(100*a.high_best_of_overlap).toFixed(2)}%。`;
}
['mode','view','style','primary','complement','cameras'].forEach(id=>document.getElementById(id).addEventListener('change',update));update();
</script></html>'''


def write_interactive(out, args, items, stats, rays_vis, lon, phi, kp, kc, patch, basis, report, min_depth):
    # 完整分辨率用于统计；HTML 只降低绘制密度，不重新计算/改变 Mask。
    step = max(1, args.display_stride)
    selection = np.s_[::step, ::step]
    small = lambda a: a[selection]
    flat = lambda a: np.round(small(a).ravel(), 5).tolist()
    data = dict(height=small(phi).shape[0], width=small(phi).shape[1], threshold=args.kappa_threshold,
                rays=np.round(small(rays_vis).reshape(-1, 3), 5).tolist(),
                lon=flat(np.rad2deg(lon)), lat=flat(np.rad2deg(phi)), kp=flat(kp), kc=flat(kc),
                patch=small(patch).astype(int).ravel().tolist(), modes=[])
    for item, stat in zip(items, stats):
        radius = item['radius_m']
        data['modes'].append(dict(name='Angular (rotation only)' if radius is None else f'Finite {radius:g} m',
            radius=radius, stats=stat, code=(small(item['left']).astype(int)+2*small(item['right'])).ravel().tolist(),
            lrgb=small(item['left_rgb']).reshape(-1, 3).tolist(), rrgb=small(item['right_rgb']).reshape(-1, 3).tolist()))
    substitutions = {'__PLOTLY__': get_plotlyjs(), '__DATA__': json.dumps(data, separators=(',', ':')),
        '__PRIMARY__': json.dumps(grid_lines(), separators=(',', ':')),
        '__COMP__': json.dumps(grid_lines(basis, args.complement_lon_span/2, args.complement_lat_limit), separators=(',', ':')),
        '__CAMERAS__': json.dumps(camera_traces(report)), '__REF__': str(args.ref_cam), '__TGT__': str(args.target_cam),
        '__FRAME__': str(args.frame_id), '__LON__': f'{args.complement_lon_span/2:g}',
        '__LAT__': f'{args.complement_lat_limit:g}', '__MINDEPTH__': str(min_depth)}
    # 替换一次，不扫描已插入的 Plotly 库，避免意外替换第三方脚本。
    import re
    finite_initial = next((i+1 for i, r in enumerate(args.radii) if min_depth is not None and r >= min_depth), 1)
    for filename, initial in [('angular_coverage.html', 0), ('finite_radius_coverage.html', finite_initial)]:
        substitutions['__INITIAL__'] = str(initial)
        html = re.sub(r'__[A-Z]+__', lambda m: substitutions.get(m.group(), m.group()), APP)
        (out/filename).write_text(html)
    rig = go.Figure(camera_traces(report))
    rig.update_layout(title=f"Actual cam{args.ref_cam}/cam{args.target_cam} rig | baseline {report['baseline_m']:.6f} m | axes {report['optical_axis_angle_deg']:.1f} deg",
        height=800, scene=dict(aspectmode='data', xaxis_title='Xvis: image-left (m)', yaxis_title='Yvis: up (m)',
        zaxis_title='Zvis: reference-forward (m)', camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.5, y=1, z=1.5))))
    rig.write_html(out/'camera_rig_3d.html', include_plotlyjs=True, full_html=True)


def write_report(out, result, args):
    g, angular = result['geometry'], result['coverage'][0]
    lines = [f'# cam{args.ref_cam} + cam{args.target_cam} actual geometry report', '',
        f'Frame {args.frame_id}; ERP analysis {args.sphere_height} x {args.sphere_width}. No GT, CNN or training.', '',
        f"Baseline: {g['baseline_m']:.9f} m; optical axes: {g['optical_axis_angle_deg']:.6f} deg.",
        f"Reference centre (Rig): {g['reference_center_rig_m']}; target centre (Rig): {g['target_center_rig_m']}.", '',
        'All percentages below use exact spherical solid angles. Overlap means FOV/mask validity, not absence of occlusion.', '',
        '| Mode | L / sphere | R / sphere | Common / sphere | Common / L | Common latitude |',
        '|---|---:|---:|---:|---:|---|']
    for row in result['coverage']:
        label = 'Angular' if row['radius_m'] is None else f"{row['radius_m']:g} m"
        latitude_text = ('No common valid directions' if row['overlap_latitude_min_deg'] is None else
            f"{row['overlap_latitude_min_deg']:.2f} to {row['overlap_latitude_max_deg']:.2f} deg")
        lines.append(f"|{label}|{100*row['left_fraction']:.3f}%|{100*row['right_fraction']:.3f}%|"
                     f"{100*row['overlap_fraction']:.3f}%|{100*row['overlap_of_left']:.3f}%|"
                     f"{latitude_text}|")
    lines += ['', '## 实际结论', '',
        '1. 本次 cam4/cam3 是 Rig 的左向与后向相机，正交朝向；绝不是平行双目。',
        '2. 共视主体位于两路光轴之间，并向上、下延伸；正经度朝左图像的左侧。高纬度还包含更宽经度范围。',
        f"3. 远场共视到达 ±{angular['overlap_latitude_max_deg']:.2f}° 的采样纬度；"
        f"其中 {100*angular['high_primary_of_overlap']:.2f}% 的共视面积满足 kappa>{args.kappa_threshold:g}。",
        f"4. 水平极轴 Patch 覆盖全部主高畸变共视方向的 {100*angular['high_primary_covered_by_patch']:.2f}%；"
        f"这些方向在补偿坐标中的绝对纬度最大为 {angular['primary_high_abs_latitude_in_complement_deg']['maximum']:.2f}°。",
        '5. 后续优先研究实际共视中的上下高纬度区域，保留逐深度 FOV/Mask 与遮挡约束；'
        '不必把已经低畸变的赤道区域或双目不可见区域纳入补偿。', '',
        '这些结果验证的是参数化畸变降低，不是深度精度改善。有限半径和真实场景遮挡仍需后续独立研究。', '',
        '## Complementary local grid (angular common directions)', '',
        f"Primary kappa > {args.kappa_threshold:g}: {100*angular['high_primary_of_overlap']:.3f}% of common area.",
        f"Actual local patch covers {100*angular['high_primary_covered_by_patch']:.3f}% of those high-distortion directions.",
        f"High-region |latitude_C| summary: {angular['primary_high_abs_latitude_in_complement_deg']}.",
        f"Primary kappa: {angular['kappa_primary']}; lower kappa within patch: {angular['kappa_best_with_patch']}.",
        f"Residual high-distortion common area: {100*angular['high_best_of_overlap']:.3f}%.", '',
        'The finite rectangular patch is a diagnostic choice, not the final Yin–Yang architecture. '
        'Outside the patch, primary kappa is retained; no out-of-patch improvement is counted.', '',
        '## Validation', '', '```json', json.dumps(result['validation'], indent=2), '```', '',
        '## Interpretation limits', '',
        '- Finite spheres are virtual surfaces, not scene GT. RGB coincidence is not a depth accuracy metric.',
        '- Direction/FOV overlap does not detect real object occlusion.',
        '- 0.5 and 1 m are below the config sweep minimum (1.65 m) and are geometric probes only.',
        '- OCam forward/inverse fitted polynomials need not be exact inverses; actual errors are reported above.',
        '- Kappa describes coordinate anisotropy, not stereo triangulation strength or learned depth accuracy.',
        '- Statistics depend on the configured sphere resolution; boundary cells have discretization error.', '',
        '## Files', '',
        '- `angular_coverage.html`, `finite_radius_coverage.html`: offline interactive sphere and ERP, all modes available.',
        '- `camera_rig_3d.html`: actual optical centres, optical axes, rig centre and baseline.',
        '- `angular/`, `finite_*m/`: RGB maps, coverage and distortion PNGs, raw RGB/mask PNGs, numeric projection NPZ.',
        '- `geometry_stats.json`, `coverage_stats.csv`, `distortion_stats.csv`: machine-readable real measurements.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--frame-id', type=int, default=4887)
    parser.add_argument('--ref-cam', type=int, default=4)
    parser.add_argument('--target-cam', type=int, default=3)
    parser.add_argument('--sphere-height', type=int, default=360)
    parser.add_argument('--sphere-width', type=int, default=720)
    parser.add_argument('--radii', type=float, nargs='+', default=[.5, 1., 2., 5., 10.])
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'outputs')
    parser.add_argument('--display-stride', type=int, default=3)
    parser.add_argument('--complement-lon-span', type=float, default=240.)
    parser.add_argument('--complement-lat-limit', type=float, default=45.)
    parser.add_argument('--kappa-threshold', type=float, default=2.)
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    if min(args.radii) <= 0 or args.sphere_height < 16 or args.sphere_width < 32 or args.display_stride < 1:
        parser.error('Positive radii/stride and sphere resolution at least 16 x 32 required')
    if not (0 < args.complement_lon_span < 360 and 0 < args.complement_lat_limit < 90):
        parser.error('Complement must be a finite patch: longitude span <360, latitude limit <90')
    if args.kappa_threshold <= 1:
        parser.error('kappa-threshold must exceed 1')
    if min(args.sphere_height, args.sphere_width) // args.display_stride < 2:
        parser.error('display-stride must preserve at least two display samples per axis')
    args.radii = sorted(set(args.radii))
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_STYLE)
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    cameras, images, config, provenance = load_pair(args.data_root, args.ref_cam, args.target_cam, args.frame_id)
    rotation, translation, report = pair_geometry(*cameras)
    print(json.dumps(report, indent=2), flush=True)
    rays, rays_vis, lon, phi, area = spherical_rays(args.sphere_height, args.sphere_width)
    basis, phi_c, lon_c, patch, kc = complementary_coordinates(rays_vis, report,
        args.complement_lon_span, args.complement_lat_limit)
    kp = 1 / np.maximum(np.abs(np.cos(phi)), 1e-6)
    angular = sphere_projection(cameras, images, rays, rotation, translation)
    finite = [sphere_projection(cameras, images, rays, rotation, translation, r) for r in args.radii]
    items = [angular, *finite]
    stats = [coverage_statistics(x, area, phi, phi_c, patch, kp, kc, args.kappa_threshold) for x in items]
    validation = validate_geometry(cameras, images, rotation, translation, rays, basis, rays_vis, angular, finite)
    print('Validation:', json.dumps(validation, indent=2), flush=True)
    min_depth = config.get('config', {}).get('omnimvs_sweep_min_depth')
    result = dict(frame_id=args.frame_id, geometry=report, provenance=provenance,
        config_sha256=hashlib.sha256((args.data_root/'config.yaml').read_bytes()).hexdigest(),
        image_sha256=[hashlib.sha256(Path(x['image']).read_bytes()).hexdigest() for x in provenance],
        resolution_hw=[args.sphere_height, args.sphere_width], training_sweep_min_depth_m=min_depth,
        coordinates=dict(complement_basis_visual=basis.tolist(), complement_pole_reference=(OCAM_FROM_VIS @ basis[:, 1]).tolist(),
            complement_lon_span_deg=args.complement_lon_span, complement_latitude_limit_deg=args.complement_lat_limit,
            primary_high_kappa_threshold=args.kappa_threshold, exact_sphere_area_sr=float(area.sum())),
        coverage=stats, validation=validation,
        limitations=['FOV overlap is not occlusion visibility', 'Virtual sphere RGB is not actual scene GT',
                     'Kappa is coordinate anisotropy only', 'Finite patch only; no full Yin-Yang or depth fusion'])
    (out/'geometry_stats.json').write_text(json.dumps(result, indent=2)+'\n')
    with (out/'coverage_stats.csv').open('w', newline='') as file:
        keys = ['radius_m', 'left_fraction', 'right_fraction', 'overlap_fraction', 'overlap_of_left',
                'left_only_fraction', 'right_only_fraction', 'uncovered_fraction',
                'overlap_latitude_min_deg', 'overlap_latitude_max_deg']
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for stat in stats:
            writer.writerow({k: 'angular' if k == 'radius_m' and stat[k] is None else stat[k] for k in keys})
    with (out/'distortion_stats.csv').open('w', newline='') as file:
        keys = ['mode', 'high_primary_of_overlap', 'patch_of_overlap', 'high_primary_covered_by_patch',
                'high_best_of_overlap', 'primary_mean_kappa', 'best_mean_kappa', 'primary_p95_kappa', 'best_p95_kappa']
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for stat in stats:
            row = {k: stat[k] for k in keys[1:5]}
            row.update(mode='angular' if stat['radius_m'] is None else str(stat['radius_m']),
                primary_mean_kappa=stat['kappa_primary']['mean'], best_mean_kappa=stat['kappa_best_with_patch']['mean'],
                primary_p95_kappa=stat['kappa_primary']['p95'], best_p95_kappa=stat['kappa_best_with_patch']['p95'])
            writer.writerow(row)
    for item in items:
        radius = item['radius_m']
        folder = out/('angular' if radius is None else f'finite_{radius:g}m')
        print(f'Rendering {folder.name}', flush=True)
        plot_maps(folder, item, kp, kc, patch, phi_c, args.kappa_threshold,
            'Angular / rotation only' if radius is None else f'Finite {radius:g} m / full R,t', args.ref_cam, args.target_cam)
        np.savez_compressed(folder/'projection_data.npz', rays_reference=rays.astype('float32'),
            area_sr=area, left_uv=item['left_uv'].astype('float32'), right_uv=item['right_uv'].astype('float32'),
            left_valid=item['left'], right_valid=item['right'], overlap=item['overlap'], complement_patch=patch,
            kappa_primary=kp.astype('float32'), kappa_complementary=kc.astype('float32'),
            latitude_primary_deg=np.rad2deg(phi).astype('float32'), latitude_complementary_deg=np.rad2deg(phi_c).astype('float32'))
    for image, cam in zip(images, (args.ref_cam, args.target_cam)):
        Image.fromarray(image).save(out/f'input_cam{cam}_{args.frame_id:05d}.png')
    plot_statistics(out, stats[1:], stats[0], area, phi, phi_c, angular['overlap'], kp,
                    args.kappa_threshold, args.ref_cam, args.target_cam)
    write_interactive(out, args, items, stats, rays_vis, lon, phi, kp, kc, patch, basis, report, min_depth)
    write_report(out, result, args)
    (out/'figure_qa.json').write_text(json.dumps(QA, indent=2))
    print('DONE:', out, flush=True)


if __name__ == '__main__':
    main()
