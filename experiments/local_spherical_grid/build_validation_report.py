"""汇总冻结网格验证结果，生成离线报告与支持缺失可视化；不生成/调整参数。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
from plotly.offline import get_plotlyjs
from camera_geometry import spherical_rays
from search_yang_grid import chart_lines,sphere_map
from visualize_stereo_sphere import save_figure,PLOT_STYLE,QA


HTML=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Frozen Adaptive Local Spherical Grid validation</title>
<style>body{font-family:system-ui;margin:20px;background:#f7f9fc;color:#162535}h1{font-size:24px}.box{background:white;padding:16px;border-radius:12px;margin-bottom:14px}p{line-height:1.6}
.controls{display:flex;gap:18px;flex-wrap:wrap}select{padding:6px}#sphere{height:620px}#erp{height:420px}.warn{border-left:5px solid #d55e00}</style>
<script>__PLOTLY__</script><h1>Adaptive Local Spherical Grid · 冻结参数审查</h1>
<div class="box warn"><p><b>结论：坐标与窗口覆盖通过；离散共同支持存在缺失。</b>几何未覆盖=0，不等于原生插值没有问题。
以下支持缺失使用两相机共同有效的原生 Mask，针对匹配/体查询；不是左 RGB 本身的投影错误。</p>
<p>红色：没有所选规则要求的原生共同支持；蓝绿：有支持；灰色：该深度没有几何共视。
严格规则缺少部分有效角点；“任一角点”规则失败才表示所有正权重角点均无有效证据。几何和GT数据未用于重新搜索或调参。</p></div>
<div class="box"><div class="controls"><label>分析网格 <select id="res"></select></label>
<label>候选索引 <select id="idx"><option>0</option><option>24</option><option selected>48</option><option>72</option><option>95</option></select></label>
<label>规则 <select id="rule"><option value="geometry">仅几何窗口</option><option value="nearest">最近邻共同支持</option>
<option value="strict" selected>所有正权重双线性角点</option><option value="any">任一正权重角点</option></select></label>
<label><input id="grid" type="checkbox" checked>冻结网格边界</label></div><p id="info"></p>
<p>两个网格的球心都是 cam4。显示为单位射线，深度有效性使用真实 R,t。坐标旋转和各窗口边界保持原结果不变。
粗显示可能漏掉稀有缺失点，因此另用一个红色菱形标出完整网格中的真实缺失示例；标记大小不代表面积比例。</p></div>
<div id="sphere" class="box"></div><div id="erp" class="box"></div>
<div class="box"><p><a href="frozen_geometry.html">原完整96候选几何交互页</a> · <a href="REPORT_VALIDATION.md">完整验证报告</a> ·
<a href="coverage/720x1440/native_sampling_holes.png">高分辨率支持缺失图</a> ·
<a href="coverage/720x1440/seam_zoom.png">接缝局部放大</a> ·
<a href="coordinates_rgb/cam4_idx48_overlap_rgb.png">同源RGB重叠差异</a> ·
<a href="quality/best_coordinate_heatmap.png">择优参数化质量</a> ·
<a href="gt/REPORT_GT.md">GT诊断及可见性限制</a></p>
<p>原生网格尺寸：Yin 161×239、Yang 80×307。高分辨率参考球面只用于审查，没有修改原生尺寸或网络配置。
GT仅用于离线诊断；没有独立cam3深度真值，Rig点云Z-buffer估计不能当作真实无遮挡Mask。</p></div>
<script>const D=__DATA__,G=__GRIDS__;let res=document.getElementById('res');D.forEach((a,i)=>res.add(new Option(a.name,i)));res.value=D.length-1;
function lines(segs,color,name){let x=[],y=[],z=[];segs.forEach(a=>{a.forEach(p=>{x.push(p[0]*1.01);y.push(p[1]*1.01);z.push(p[2]*1.01)});x.push(null);y.push(null);z.push(null)});return {type:'scatter3d',mode:'lines',x,y,z,line:{color,width:1.5},name,hoverinfo:'skip'};}
function update(){const a=D[+res.value],idx=+document.getElementById('idx').value,k=a.indices.indexOf(idx),rule=document.getElementById('rule').value;
const valid=a.valid[k],holes=a[rule][k];let x=[],y=[],z=[],color=[],hover=[],raster=[];
for(let r=0;r<a.h;r++){let row=[];for(let c=0;c<a.w;c++){const i=r*a.w+c;
const rgb=!valid[i]?[232,235,240]:holes[i]?[205,60,65]:[0,158,115];row.push(rgb);
if(valid[i]){const p=a.rays[i];x.push(p[0]);y.push(p[1]);z.push(p[2]);color.push(`rgb(${rgb})`);hover.push(`primary lon ${a.lon[i].toFixed(2)}°, lat ${a.lat[i].toFixed(2)}°<br>support missing: ${!!holes[i]}`);}}raster.push(row);}
const traces=[{type:'scatter3d',mode:'markers',x,y,z,marker:{size:3,color},text:hover,hoverinfo:'text',name:'Jointly valid directions'}];
const ex=a.examples[rule][k];if(ex)traces.push({type:'scatter3d',mode:'markers+text',x:[ex.ray[0]],y:[ex.ray[1]],z:[ex.ray[2]],
 marker:{size:8,color:'#c52935',symbol:'diamond'},text:['Actual unsupported example'],textposition:'top center',name:'Full-resolution failure example'});
if(document.getElementById('grid').checked)traces.push(lines(G.yin,'#0072B2','Yin'),lines(G.yang,'#D55E00','Yang'));
Plotly.react('sphere',traces,{uirevision:'sphere',margin:{t:25,b:10,l:5,r:5},legend:{orientation:'h'},scene:{aspectmode:'cube',
 xaxis:{title:{text:'Xvis: image-left'},range:[-1.08,1.08]},yaxis:{title:{text:'Yvis: up'},range:[-1.08,1.08]},
 zaxis:{title:{text:'Zvis: cam4 forward'},range:[-1.08,1.08]},camera:{up:{x:0,y:1,z:0},eye:{x:1.65,y:.6,z:1.65}}}},{responsive:true,scrollZoom:true});
const erp=[{type:'image',z:raster,x0:a.lon[0],dx:a.lon[1]-a.lon[0],y0:a.lat[0],dy:a.lat[a.w]-a.lat[0]}];
if(ex)erp.push({type:'scatter',mode:'markers',x:[ex.lon],y:[ex.lat],marker:{size:10,color:'#c52935',symbol:'diamond-open'},name:'Actual unsupported example'});
Plotly.react('erp',erp,
{title:{text:'Same directions in primary ERP; red = missing selected native support'},margin:{t:45,b:55,l:65,r:30},
xaxis:{title:{text:'Primary longitude (deg)'},range:[-180,180],dtick:60},yaxis:{title:{text:'Primary latitude (deg)'},range:[-90,90],dtick:30}},{responsive:true});
let count=rule==='geometry'?a.rows[k].geometric_uncovered_count:rule==='nearest'?a.rows[k].nearest_unsupported_count:rule==='strict'?a.rows[k].strict_bilinear_unsupported_count:a.rows[k].any_corner_unsupported_count;
let ratio=rule==='geometry'?1-a.rows[k].geometric_coverage_ratio:rule==='nearest'?1-a.rows[k].nearest_supported_area_ratio:rule==='strict'?1-a.rows[k].strict_bilinear_supported_area_ratio:1-a.rows[k].any_corner_supported_area_ratio;
document.getElementById('info').textContent=`${a.name}，idx ${idx}，${idx===0?'远场极限':(1/a.axis[idx]).toFixed(4)+' m'}：完整分析网格缺少支持样本 ${count.toLocaleString()}，该层球面面积加权 ${ (100*ratio).toFixed(6)}%。显示点云仅降采样；完整布尔Mask保存在NPZ。`;
}['res','idx','rule','grid'].forEach(id=>document.getElementById(id).addEventListener('change',update));update();</script></html>'''


def dashboard(out, selection):
    data=[]; indices=[0,24,48,72,95]
    for folder in sorted((out/'coverage').glob('*x*')):
        if not folder.is_dir():continue
        h,w=map(int,folder.name.split('x'))
        sh,sw=max(1,h//120),max(1,w//240)
        with np.load(folder/'coverage_masks.npz') as a:
            _,visual,lon,lat,_=spherical_rays(h,w)
            sub=np.s_[::sh,::sw]
            masks={};examples={}
            for name,key in [('valid','common_valid'),('geometry','geometric_uncovered'),('nearest','nearest_unsupported'),
                             ('strict','strict_bilinear_unsupported'),('any','any_corner_unsupported')]:
                full=a[key]
                masks[name]=full[indices,::sh,::sw].reshape(len(indices),-1).astype(int).tolist()
                if name!='valid':
                    examples[name]=[]
                    for idx in indices:
                        fail=np.flatnonzero(full[idx])
                        if not len(fail):examples[name].append(None);continue
                        y,x=np.unravel_index(fail[len(fail)//2],(h,w))
                        examples[name].append(dict(ray=visual[y,x].tolist(),lon=float(np.rad2deg(lon[y,x])),lat=float(np.rad2deg(lat[y,x]))))
            with (folder/'coverage_per_depth.csv').open() as f: rows=list(csv.DictReader(f))
            rows=[{k:float(v) for k,v in rows[i].items()} for i in indices]
            data.append(dict(name=folder.name,h=visual[sub].shape[0],w=visual[sub].shape[1],indices=indices,
                rays=np.round(visual[sub].reshape(-1,3),5).tolist(),lon=np.rad2deg(lon[sub]).ravel().tolist(),
                lat=np.rad2deg(lat[sub]).ravel().tolist(),axis=a['axis'].tolist(),rows=rows,examples=examples,**masks))
    grids=dict(yin=chart_lines(selection['yin_bounds'],np.eye(3)),
               yang=chart_lines(selection['yang_bounds'],np.asarray(selection['yang_basis_visual'])))
    replace={'__PLOTLY__':get_plotlyjs(),'__DATA__':json.dumps(data,separators=(',',':')),'__GRIDS__':json.dumps(grids,separators=(',',':'))}
    import re
    (out/'validation_dashboard.html').write_text(re.sub(r'__[A-Z]+__',lambda m:replace[m.group()],HTML))


def evidence_plots(out):
    plt.rcParams.update(PLOT_STYLE)
    with np.load(out/'coverage/720x1440/coverage_masks.npz') as a:
        valid=a['common_valid'].any(0)
        strict=a['strict_bilinear_unsupported'].any(0)
        empty=a['any_corner_unsupported'].any(0)
        geom=a['geometric_uncovered'].any(0)
        handoff=a['handoff_union']
    code=np.where(valid,1,0).astype(np.uint8)
    code[strict]=2;code[empty]=3;code[geom]=4
    colors=['#e8ebf0','#79b7d8','#e69f00','#cd3c41','#cc79a7']
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    regions=[('North high latitude',(-180,180),(60,90),[-180,-120,-60,0,60,120,180],[60,70,80,90]),
             ('Handoff neighbourhood',(-20,90),(40,65),[-20,0,20,40,60,80],[40,45,50,55,60,65]),
             ('South high latitude',(-180,180),(-90,-60),[-180,-120,-60,0,60,120,180],[-90,-80,-70,-60])]
    for ax,(title,xlim,ylim,xticks,yticks) in zip(axes,regions):
        sphere_map(ax,code,title,ListedColormap(colors),BoundaryNorm(np.arange(-.5,5),5))
        ax.contour(handoff.astype(float),levels=[.5],extent=[-180,180,-90,90],origin='upper',colors='black',linewidths=.5)
        ax.set(xlim=xlim,ylim=ylim,xticks=xticks,yticks=yticks)
    fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(colors,
        ['No geometric common support','Strict support complete','Some corners invalid','No valid weighted corner','Window uncovered'])],
        loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Frozen 720 x 1440 grid; failures at any of 96 candidates; black = handoff neighbourhood')
    save_figure(fig,out/'evidence_high_latitude_and_seam.png')

    review=json.loads((out/'quality/coverage_lookup_independent_check.json').read_text())
    ex=review['examples'][0]
    corners=ex['native_support']['yang']['four_corners']
    fig,axes=plt.subplots(1,3,figsize=(11,4.4),layout='constrained')
    for ax,key,title in zip(axes,['left_valid','target_valid','joint_valid'],['Left native support','Target native support','Joint native support']):
        data=np.array([int(c[key]) for c in corners]).reshape(2,2)
        ax.imshow(data,cmap=ListedColormap(['#e8ebf0','#0072b2']),vmin=0,vmax=1)
        for (r,c),weight in np.ndenumerate(np.array([c['weight'] for c in corners]).reshape(2,2)):
            ax.text(c,r,f'{int(data[r,c])}\nw={weight:.6f}',ha='center',va='center',color='white' if data[r,c] else '#222')
        ax.set(title=title,xlabel='Yang column',ylabel='Yang row',xticks=[0,1],xticklabels=[287,288],
               yticks=[0,1],yticklabels=[51,52])
    fig.suptitle('Counterexample: exact query valid in BOTH cameras, but all target native corners invalid\n'
                 'idx48, primary lon=-97.125 deg, lat=79.625 deg; no parameter changes')
    save_figure(fig,out/'evidence_zero_joint_support.png')
    np.savez_compressed(out/'evidence_counterexample.npz',
        weights=np.array([c['weight'] for c in corners]),left_valid=np.array([c['left_valid'] for c in corners]),
        target_valid=np.array([c['target_valid'] for c in corners]),joint_valid=np.array([c['joint_valid'] for c in corners]),
        exact_left_uv=ex['left_pixel'],exact_target_uv=ex['target_pixel'],
        primary_lon_lat_deg=[ex['primary_lon_deg'],ex['primary_lat_deg']],depth_index=ex['candidate'],
        radial_distance_m=ex['radial_distance_m'])
    (out/'validation_figure_qa.json').write_text(json.dumps(QA,indent=2))


def report(out, selection, cov, coords, gt, quality):
    high=max(cov['resolutions'],key=lambda x:x['height'])
    ang=max(r['maximum'] for r in coords['random_direction_roundtrips'].values())
    px=max(r['analytic_pixel_ray_chart_pixel_error_px']['maximum'] for r in coords['pixel_roundtrips'])
    added=max(r['added_by_chart_max_px'] for r in coords['pixel_roundtrips'])
    best=quality['depth_area_weighted']['best_coordinate']
    budget=cov['native_budget']
    lines=['# REPORT_VALIDATION — Adaptive Local Spherical Grid 冻结参数审查','',
        '## 结论先行','',
        '**当前网格可以作为后续双目特征映射和 Spherical Sweeping 的几何原型基础，但尚不能称为“无空洞、严格无缝、真实可见性已验证”的完整训练输入。**','',
        f'坐标和解析投影链路通过，360×720 与 720×1440 的 96 层窗口覆盖均为 100%。'
        f'但高分辨率检查发现 {high["strict_bilinear_unsupported_count"]:,} 个深度×方向样本缺少完整原生共同有效角点支持，'
        f'面积/深度加权为 {100*high["strict_unsupported_weighted_fraction"]:.6f}%；'
        f'甚至任一正权重角点规则仍有 {high["any_corner_unsupported_count"]:,} 个样本完全无原生共同支持。',
        '这些是两相机共同有效匹配体的局部离散支持问题，不能直接等同于左 RGB/CNN 投影错误。'
        '原生样本中心上的自检通过，也不能推出任意连续查询点上都没有空洞。','',
        '此外，同源 RGB 经两套原生网格重采样后存在可测的边缘色差；部分其他帧的 GT 小于最近深度候选。'
        '数据副本没有独立 cam3/cam4 深度 GT，不能完成独立目标深度真值下的遮挡验证。','',
        '本次未重做搜索、未修改选定边界、原生尺寸、候选轴、网络、GWC/Correlation、DA-V2 Adapter 或原 GT 转换代码。','',
        '## 1. 冻结参数与坐标系','',
        f'选定参数 SHA256：`{cov["frozen_selection_sha256"]}`。','',
        '| 项目 | 冻结定义 |','|---|---|',
        '| 原点 | cam4 光心；Yin/Yang 只是同原点的坐标旋转 |',
        '| cam4 / OCam | +X 图像右、+Y 向下、+Z 沿左光轴 |',
        '| Yin / visual | +X 图像左、+Y 向上、+Z 沿左光轴 |',
        '| Yin 窗口 | 经度中心45°、半宽89°，即[-44°,134°]；纬度[-60°,60°] |',
        '| Yang 窗口 | 最终经度中心0°、半宽114.5°；纬度[-22°,37°] |',
        '| Yang 极轴 | 沿 cam4 +Z；零经度方向沿 cam3 光轴，原点仍是cam4 |','',
        '定义 `A=R_cam4_from_Yin=diag(-1,-1,1)`，`Q=R_Yin_from_Yang`。'
        'Q 的**列**是 Yang 的三个坐标轴在 Yin 系中的坐标，不是相机投影矩阵，也没有基线平移。','',
        '```text','A = [-1  0  0; 0 -1  0; 0  0  1]',
        'Q ≈ [ 0  0  1; 1  0  0; 0  1  0]','',
        '列向量正向：r_cam4 = A r_Yin = A Q r_Yang',
        '列向量逆向：r_Yin = A.T r_cam4；r_Yang = Q.T A.T r_cam4','',
        'N×3 行向量批次：R_Yin = R_cam4 @ A；R_Yang = R_cam4 @ A @ Q',
        '行向量逆向：R_cam4 = R_Yang @ Q.T @ A.T','',
        'Q = Ry(yaw_y=0°) @ Rx(tilt_x=90°) @ Ry(原经度中心=90°)',
        '```','',
        '矩阵右端先作用于列向量。Yang 的原经度中心90°已经吸收到Q中，所以最终窗口中心为0°。'
        '**Yin中心45°仅表示窗口裁剪，不能再额外乘Ry(45°)**。真实外参仍单独使用 `X3=R34 X4+t34`。','',
        f'固定种子、100000 条随机球面方向的最大往返角误差 **{ang:.8g}°**；'
        f'真实鱼眼 Pixel→Ray→Yin/Yang→Pixel 最大误差 **{px:.9g} px**，'
        f'其中网格坐标变换新增误差仅 **{added:.8g} px**。'
        'cam3 测试使用真实光心与 cam4 球面的射线交点，未忽略平移。','',
        '## 2. 分辨率、窗口覆盖与原生插值支持','',
        '下面的计数是深度×方向样本；比例按每格精确立体角、96候选等权计算。不是实际场景深度概率。','',
        '| 分析网格 | 有效候选样本 | 窗口未覆盖 | 几何覆盖率 | 最近邻无支持 | 严格双线性无完整支持 | 任一正权角点也无支持 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in cov['resolutions']:
        lines.append(f'|{r["height"]}×{r["width"]}×96|{r["valid_depth_direction_count"]:,}|{r["geometric_uncovered_count"]}|'
          f'{100*r["geometric_coverage_ratio"]:.3f}%|{r["nearest_unsupported_count"]:,} ({100*r["nearest_unsupported_weighted_fraction"]:.6f}%)|'
          f'{r["strict_bilinear_unsupported_count"]:,} ({100*r["strict_unsupported_weighted_fraction"]:.6f}%)|'
          f'{r["any_corner_unsupported_count"]:,} ({100*r["any_corner_unsupported_weighted_fraction"]:.6f}%)|')
    lines+=['','原生网格 Yin=[96,161,239]、Yang=[96,80,307]：分别检查全部96层，几何未覆盖、'
        '本格中心最近邻与严格插值无支持计数均为0。这个测试存在天然self-support，必须与更密参考球面的查询测试分开解释。','',
        '- 窗口覆盖：查询方向落在至少一个冻结窗口内，且该候选在两相机几何有效。',
        '- 最近邻共同支持：至少一个窗口的最近原生样本在两相机中均有效。',
        '- 严格双线性：所有正权重的原生角点必须共同有效；缺少部分角点不等于完全没有证据。',
        '- 任一正权角点：至少一个有效角点；这是诊断下界，**没有擅自实现重归一化、补洞或改变推理行为**。','',
        '高纬度窗口未覆盖数为0。插值缺失图、接缝局部图和按层CSV保留所有问题位置，未用简单插值填满。',
        '独立复核还使用另一套列向量/atan2实现和 Torch float64 grid_sample，在11,043个方向、两网格、idx0/48/95上逐一比较，'
        '最近邻、严格及任一角点结果均无不一致。示例：idx48，主纬度79.625°、经度−97.125°，两相机精确投影合法，'
        '但Yin窗外、Yang的2×2共同有效角点权重和为0。这是离散共同支持的真实反例，不是旋转算错。'
        '独立复核细节见 `quality/COVERAGE_REVIEW.md` 与 `coverage_lookup_independent_check.json`。',
        '![覆盖与严格支持](coverage/720x1440/coverage_vs_sampling_support.png)','',
        '![原生支持缺失](coverage/720x1440/native_sampling_holes.png)','',
        '![高纬度与接缝放大](evidence_high_latitude_and_seam.png)','',
        '放大图显示96层问题位置的并集，且主ERP在高纬度会拉伸显示面积；不能用屏幕橙色面积估计问题比例。'
        '0.884%等数值仍来自逐层球面立体角加权，未把不同深度的问题位置重复并集后再当面积比例。','',
        '![无共同支持反例](evidence_zero_joint_support.png)','',
        '## 3. 深度候选与真实GT范围','',
        '候选轴与原 Dataset 的192层 `[::2]` 逐项完全相同；不能重新用96个端点linspace替代。',
        f'- rho[0]={gt["candidate_axis"]["inverse_depth_min"]:.17g} 1/m，即 double EPS，数值远限为 '
        f'{gt["candidate_axis"]["farthest_radius_m"]:.0f} m。它不是逆深度恰好0，也不是真正无穷远。',
        f'- 最近候选径向距离={gt["candidate_axis"]["nearest_radius_m"]:.12f} m，定义在cam4光心。',
        '- TIFF 首图是彩色缩略图；读取原 `readImageFloat()` 得到320×640 float32逆深度页。'
        '先按原 `makeSphericalRays(...,gt_phi=90)` 恢复Rig点云，再用原外参得到cam4径向距离；未复用裁剪的Rig GT index。','',
        '| 帧 | 有Rig点云GT支持的cam4原图像素 | 其中超近比例 | 双目几何有效源GT点 | 其中超近/范围外比例 | Rig点云估计双目可见比例 |',
        '|---|---:|---:|---:|---:|---:|']
    for frame in gt['frames']:
        fid=frame['frame_id']; row=next(r for r in gt['range_rows'] if r['frame']==fid and r['region']=='binocular_geometric')
        pix=frame['cam4_sparse_pixel_range_statistics']; vis=frame['visibility_regions']['all']
        lines.append(f'|{fid}|{pix["observed_sparse_rig_gt_pixel_count"]:,}|{100*pix["below_nearest_pixel_fraction"]:.5f}%|'
          f'{row["source_point_count"]:,}|{100*row["outside_candidate_point_fraction"]:.5f}%|'
          f'{100*vis["estimated_visible_fraction_of_geometric"]:.3f}%|')
    lines+=['','像素列只针对投到cam4后经Z-buffer保留的稀疏GT支持，不是完整cam4深度真值；'
        '源GT点列的分母是原Rig栅格样本，不能与cam4像素或cam4球面面积混淆。'
        'CSV另含Rig球面面积权重及Yin、Yang、交集分组。超过候选远限的样本比例为0；超近样本不应被解释成有效[0,95]监督。','',
        '**未完成的真值条件：没有独立cam3/cam4深度GT。** '
        '输出 `geometric_valid`、`estimated_visible_both`、`independent_true_visibility_known` 分开保存。'
        '同一Rig点云的Z-buffer存在self-support，无法恢复Rig不可见遮挡物；最近像素取整也会引入离散误判。'
        '它只能作为可见性估计，不能声称真实无遮挡Mask已验证。','',
        '数据副本旧README.txt中的min_depth/角度/中心描述与当前YAML和浮点TIFF形式存在冲突；'
        '本诊断按当前工程gt_phi=90和标定约定执行，没有擅改原GT转换。角度元数据没有新增的独立验证，报告保留此限制。','',
        'GT只用于以上离线诊断，不参与任何网格生成、参数拟合或实际推理。','',
        '## 4. 固定参数下2/4/6单元接缝分析','',
        '**保持两个矩形边界不动，物理重叠面积不会因为“请求的接缝宽度”而变化。** '
        '本实验将n定义为每张原生网格内侧n个单元的上下文halo，要求局部(2n+1)²共同有效支持。'
        '若n指双侧总宽2/4/6，则对应每侧1/2/3；附表CSV同时给出，避免宽度定义歧义。'
        '这些腐蚀Mask仅用于诊断，没有用于减少最终coverage或删掉原本有效方向。','',
        '这是匹配体共同有效证据的保守上下文检查，不是运行了CNN，也不是证明单相机CNN特征会按同样比例断裂。'
        '平均重叠比例是先逐层计算球面面积比例、再对96层等权平均。','',
        'handoff由相邻、几何有效方向的较低κ所属网格变化定义，经度周期连接。'
        '在Yang裁剪边界，一侧自然不属于Yang，因此“双网格在每个界面点都同时有完整上下文”强于union连续性。'
        '下面同时给出二者，不能把较低的dual比例误读为同样比例的视场断开。','',
        '| 每侧halo单元n | 几何覆盖 | 平均重叠/共视 | 总原生采样预算/层 | 界面union上下文支持 | 界面owner上下文支持 | 界面双网格上下文支持 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in cov['seam_summary']:
        if r['height']==high['height'] and r['halo_cells_per_chart'] in (2,4,6):
            lines.append(f'|{r["halo_cells_per_chart"]}|{100*r["geometric_coverage_ratio"]:.2f}%|'
              f'{100*r["mean_overlap_of_common"]:.3f}%|{r["total_native_samples_per_depth"]:,}|'
              f'{100*r["union_context_support_ratio"]:.3f}%|{100*r["owner_context_support_ratio"]:.3f}%|'
              f'{100*r["dual_context_support_ratio"]:.3f}%|')
    counts=list(csv.DictReader((out/'coverage/actual_native_sample_counts.csv').open()))
    lines+=['',f'总采样预算：Yin 38,479 + Yang 24,560 = **{budget["total_native_samples_per_depth"]:,} 点/层**；'
        f'96层为 **{budget["total_3d_candidate_lattice_positions"]:,} 个候选格点**。','',
        f'在 {budget["tolerance_deg"]:g}° 容差下，两个原生格点集合的近重合方向对为 '
        f'**{budget["all_geometric_coincident_direction_pairs"]}**；最接近的跨网格方向相差 '
        f'{budget["minimum_cross_grid_direction_distance_deg"]:.8f}°。'
        '因此42%～50%重叠面积不代表同样比例的格点坐标完全重复。','',
        '| 候选 | Yin有效点 | Yang有效点 | Yin落入另一窗口的冗余样本 | Yang落入另一窗口的冗余样本 | 近重合有效方向对 |',
        '|---|---:|---:|---:|---:|---:|']
    for i in [0,48,95]:
        r=counts[i];lines.append(f'|{i}|{r["yin_joint_valid_samples"]}|{r["yang_joint_valid_samples"]}|'
            f'{r["yin_samples_inside_other_window"]}|{r["yang_samples_inside_other_window"]}|{r["coincident_valid_direction_pairs"]}|')
    lines+=['','重叠边界带实际native样本数随n的统计见 `coverage/native_seam_sample_budgets.csv`；'
        '总体冗余与预算保持固定。严格插值支持、上下文支持、几何连续覆盖和RGB连续性是不同的验收条件。',
        '![接缝放大](coverage/720x1440/seam_zoom.png)','',
        '## 5. 参数化质量与实际像素各向异性','',
        '理论 `κ=1/|cosφ|`；实际局部像素指标 '
        '`max(Δλ cosφ,Δφ)/min(Δλ cosφ,Δφ)`，另有有限相邻射线角距离比。'
        '统计采用双目共同有效域、精确球面面积和候选等权，原生endpoint采用截到窗口边界的半格面积。','',
        '| 指标 | 加权均值 | P95 | P99 | Max |','|---|---:|---:|---:|---:|']
    for key,label in [('yin_coordinate','Yin理论κ'),('yang_coordinate','Yang理论κ'),('best_coordinate','择优理论κ'),
                      ('yin_pixel','Yin实际像素'),('yang_pixel','Yang实际像素'),('best_pixel','择优实际像素')]:
        r=quality['depth_area_weighted'][key];lines.append(f'|{label}|{r["weighted_mean"]:.8f}|{r["p95"]:.8f}|{r["p99"]:.8f}|{r["maximum"]:.8f}|')
    lines+=['',f'原搜索记录的择优均值 **{best["weighted_mean"]:.11f}** 得到复算。'
        '整个原生矩形的Yin理论最大2、实际像素最大2.00561798；Yang理论最大1.25213566、实际像素最大1.24957477。'
        '不能把较低均值写成所有像素都接近各向同性。','',
        '择优κ局部峰值约位于主纬度±52.75°、经度±5.75°附近：相应Yang纬度刚超过冻结上界37°，只能使用Yin。'
        '这是紧窗口下的局部代价，未调整边界。',
        '![择优κ](quality/best_coordinate_heatmap.png)','',
        '## 6. RGB、特征索引和跨帧验证','',
        '解析链路正确不意味着有限网格RGB严格无缝。原图→原生网格→原图的RGB MAE约5.376～7.037/255；'
        '同源两网格重叠区RGB差均值3.057～4.116/255、P99约28.66～44.49，误差集中于物体边缘与高频纹理。'
        '同流程UV场最大误差约0.01077原图像素，说明不能把RGB色差直接归因于旋转或外参错误。','',
        'Yin经度正方向朝图像左，所以原生图的水平朝向与原图像素u递增方向不同，这是明确坐标约定；'
        '投回原鱼眼坐标后未见非预期轴翻转。','',
        'CNN stride=2检查只读调用原feature_grid。12组原图有效但特征坐标无效数量均为0。'
        '正确映射是raw_uv/2再align_corners归一化；若错误按整图resize比例，最大偏差约0.588特征像素。'
        '没有运行或修改Encoder，未宣称已验证特征语义。','',
        'RGB抽查4887/4888/4889，GT抽查4740/4886/4887/4888/5000。相同Rig下几何Mask及射线哈希不变，'
        '内容相关GT可见性和范围随帧变化。',
        '![同源重叠RGB](coordinates_rgb/cam4_idx48_overlap_rgb.png)','',
        '## 7. 文件、复现与下一步边界','',
        '- `validation_dashboard.html`：离线3D/ERP查看几何、最近邻、严格及任一角点支持缺失。',
        '- `frozen_geometry.html`：完整96候选的原冻结网格交互页，参数未重搜索。',
        '- `coverage/`：两分辨率完整D×H×W布尔NPZ、逐层CSV、接缝与采样预算。',
        '- `coordinates_rgb/`：解析链路、原生RGB、鱼眼往返、同源重叠差异及CNN索引诊断。',
        '- `gt/`：真实浮点GT诊断、几何Mask、Rig点云估计可见性和未知真值Mask。',
        '- `quality/`：理论κ、实际像素与有限格距指标，max/P95/P99/面积加权均值。','',
        '三个子Agent分别验证坐标/RGB、GT、参数化质量；主程序验证覆盖/插值/接缝并合并报告。'
        '完整数据路径与复现命令在实验README的验证章节。','',
        '**下一步可以基于当前旋转和窗口进行带Mask的特征映射及Spherical Sweeping小规模实现。** '
        '在宣称可稳定正式训练前，需要明确原生共同支持不足的处理策略、跨网格特征交换/融合及边界上下文规则，'
        '保留实际超近GT范围限制，并承认独立目标深度真值缺失。'
        '这些属于后续实现和数据条件，本次没有擅自改动任何选定参数。','']
    (out/'REPORT_VALIDATION.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--frozen-html',type=Path,required=True)
    a=p.parse_args();out=a.output_dir
    frozen=a.selection.read_bytes();s=json.loads(frozen)
    cov=json.loads((out/'coverage/coverage_validation.json').read_text())
    co=json.loads((out/'coordinates_rgb/coordinates_rgb_result.json').read_text())
    gt=json.loads((out/'gt/gt_validation.json').read_text())
    q=json.loads((out/'quality/quality_summary.json').read_text())
    sha=hashlib.sha256(frozen).hexdigest()
    assert sha==cov['frozen_selection_sha256']==co['frozen_parameters_sha256']==gt['selected_json_sha256']==q['frozen_selection_sha256']
    report(out,s,cov,co,gt,q);dashboard(out,s);evidence_plots(out)
    shutil.copy2(a.frozen_html,out/'frozen_geometry.html')
    shutil.copy2(a.selection,out/'FROZEN_SELECTION.json')
    summary=dict(status='CONDITIONAL_GEOMETRY_BASIS',frozen_selection_sha256=sha,selected_parameters_unchanged=True,
        geometric_coverage_all_checked_samples=True,continuous_discrete_support_complete=False,
        independent_target_depth_visibility_verified=False,gt_used_for_grid_generation=False,
        model_or_gt_conversion_changed=False,coordinate_rgb_frames=co['frames'],
        gt_frames=[x['frame_id'] for x in gt['frames']],coverage_results=cov['resolutions'],
        best_coordinate_kappa=q['depth_area_weighted']['best_coordinate'])
    (out/'VALIDATION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert a.selection.read_bytes()==frozen
    print('DONE',out/'REPORT_VALIDATION.md')


if __name__=='__main__':main()
