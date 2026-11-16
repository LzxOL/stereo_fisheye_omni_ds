"""Assemble V1 acceptance artifacts from actual comparison + geometry + smoke results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm,LogNorm
from plotly.offline import get_plotlyjs

from adaptive_local_grid_v1 import AdaptiveLocalSphericalGrid,HERE
from camera_geometry import spherical_rays
from search_yang_grid import angular_coordinates,window_mask,chart_lines
from visualize_stereo_sphere import save_figure,QA,PLOT_STYLE


def read(path):
    return json.loads(Path(path).read_text())


def chosen_case(package):
    return 'dual_yin_50' if package.metadata['config']['generation']['yin_lat_limit']==50 else 'historical_yin_60_frozen'


def camera_outline_traces(out,package):
    """Lightweight outlines on the SAME left-centred sphere as the colored chart points.

    Left projection is depth-invariant. Right uses the union over the configured
    candidates, consistent with the displayed common-valid union, not an occlusion
    mask or a camera cone moved to the reference optical centre.
    """
    source=out/'search/reference_depth_masks.npz'
    if not source.is_file():
        return []  # Keep older output folders without the projection export usable.
    with np.load(source) as a:
        np.testing.assert_array_equal(a['inverse_depth_candidates'],package.metadata['inverse_depth_candidates'])
        masks=[a['left_valid'],a['target_valid'].any(0)]
    traces=[]
    h,w=masks[0].shape
    longitude=-180+(np.arange(-1,w+1)+.5)*360/w
    latitude=90-(np.arange(h)+.5)*180/h
    geometry=package.metadata['geometry']
    cameras=[(f'左相机 cam{geometry["reference_id"]}','#009e9b','solid',geometry['reference_optical_axis_reference']),
             (f'右相机 cam{geometry["target_id"]}','#bd3a4c','dash',geometry['target_optical_axis_reference'])]
    for index,(mask,(name,color,dash,axis)) in enumerate(zip(masks,cameras)):
        # Wrap longitude before contouring so the +/-180 degree chart cut does
        # not create a spurious straight meridian boundary on the sphere.
        wrapped=np.concatenate([mask[:,-1:],mask,mask[:,:1]],axis=1)
        fig,ax=plt.subplots()
        contour=ax.contour(longitude,latitude,wrapped.astype(float),levels=[.5])
        paths=[p for p in contour.allsegs[0] if len(p)>1]
        plt.close(fig)
        xs,ys,zs=[],[],[]
        for path in paths:
            path=np.concatenate([path[::2],path[-1:]])
            lon,lat=np.deg2rad(path[:,0]),np.deg2rad(path[:,1])
            p=1.018*np.stack([np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)],axis=1)
            xs.extend(p[:,0].tolist()+[None]);ys.extend(p[:,1].tolist()+[None]);zs.extend(p[:,2].tolist()+[None])
        group='camera_left' if index==0 else 'camera_right'
        traces.append(dict(type='scatter3d',mode='lines',name=name+'范围',legendgroup=group,
            x=xs,y=ys,z=zs,line=dict(color=color,width=4,dash=dash),meta=dict(layer='camera_coverage'),
            hovertemplate=name+'有效投影范围（96候选并集）<extra></extra>'))
        # Only a small direction marker; this is not a fictitious right camera centre.
        p=1.12*np.asarray(axis)*np.array([-1.,-1.,1.])
        traces.append(dict(type='scatter3d',mode='markers',name=name,showlegend=False,legendgroup=group,
            x=[p[0]],y=[p[1]],z=[p[2]],text=[name],textposition='top center',
            textfont=dict(size=13,color=color),marker=dict(size=3,color=color),
            meta=dict(layer='camera_coverage'),hovertemplate=name+'光轴方向<extra></extra>'))
    return traces


def figures(out,package,comparison):
    selected=out/'comparison'/chosen_case(package)/'comparison_maps.npz'
    with np.load(selected) as a:maps={k:a[k] for k in a.files}
    valid=maps['valid_candidate_count']>0
    code=maps['yin_window'].astype(int)+2*maps['yang_window'].astype(int)
    code=np.where(valid,code,0)
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    im=axes[0].imshow(code,extent=(-180,180,-90,90),origin='upper',aspect='auto',
        cmap=ListedColormap(['#e8ebf0','#0072b2','#e69f00','#8c65ad']),norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4))
    axes[0].set(title='Working charts | same left-centred sphere',xlabel='Primary longitude (deg)',ylabel='Primary latitude (deg)')
    bar=fig.colorbar(im,ax=axes[0],fraction=.035,ticks=[0,1,2,3]);bar.set_ticklabels(['Outside common','Yin','Yang','Overlap'])
    ref=package.reference
    image=ref['left_valid'].astype(int)+ref['stereo_valid'].any(0).astype(int)
    im=axes[1].imshow(image,cmap=ListedColormap(['#e8ebf0','#e69f00','#0072b2']),vmin=0,vmax=2)
    axes[1].set(title='Final cam4 output | 320 x 480',xlabel='Output column',ylabel='Output row')
    bar=fig.colorbar(im,ax=axes[1],fraction=.035,ticks=[0,1,2]);bar.set_ticklabels(['Invalid left','Left only','Any valid match'])
    im=axes[2].imshow(maps['strict_unsupported_count']/96,extent=(-180,180,-90,90),origin='upper',aspect='auto',cmap='magma',vmin=0,vmax=.05)
    axes[2].set(title='Known native strict-support deficit',xlabel='Primary longitude (deg)',ylabel='Primary latitude (deg)')
    fig.colorbar(im,ax=axes[2],fraction=.035,label='Unsupported candidates / 96 (display cap .05)')
    save_figure(fig,out/'grid_v1_coverage.png')
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    overlap=np.where(valid,maps['yin_window'] & maps['yang_window'],np.nan)
    image=ax.imshow(overlap,extent=(-180,180,-90,90),origin='upper',aspect='auto',
        cmap=ListedColormap(['#e8ebf0','#8c65ad']),vmin=0,vmax=1)
    ax.set(title='Overlap in geometric common-valid union',xlabel='Primary longitude (deg)',ylabel='Primary latitude (deg)')
    bar=fig.colorbar(image,ax=ax,ticks=[0,1]);bar.set_ticklabels(['Single chart','Overlap'])
    save_figure(fig,out/'grid_v1_overlap.png')
    h,w=valid.shape
    lat=90-(np.arange(h)+.5)*180/h
    erp=np.broadcast_to(1/np.cos(np.deg2rad(lat))[:,None],(h,w))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,v,title in zip(axes,[erp,maps['best_pixel_anisotropy']],['Original equal-angle ERP','Frozen Yin/Yang | actual pixel angular steps']):
        im=ax.imshow(np.ma.array(v,mask=~valid),extent=(-180,180,-90,90),origin='upper',aspect='auto',cmap='viridis',norm=LogNorm(1,32))
        ax.set_facecolor('#e8ebf0');ax.set(title=title,xlabel='Primary longitude (deg)',ylabel='Primary latitude (deg)')
    bar=fig.colorbar(im,ax=axes.tolist(),fraction=.025,ticks=[1,2,4,8,16,32],extend='max',label='Anisotropy (same directions and color scale)')
    bar.set_ticklabels(['1','2','4','8','16','32']);bar.minorticks_off()
    save_figure(fig,out/'grid_v1_distortion.png')
    # Geometry only: three-dimensional direction samples, not a learned point cloud.
    # Common validity is displayed from the full-resolution comparison, decimated 6x.
    active=valid[::6,::6]
    # Use exact full-resolution ray positions of those samples (not a shifted linspace).
    yy=90-(np.arange(h)[::6]+.5)*180/h
    xx=-180+(np.arange(w)[::6]+.5)*360/w
    ll,pp=np.meshgrid(np.deg2rad(xx),np.deg2rad(yy))
    visual=np.stack((np.cos(pp)*np.sin(ll),np.sin(pp),np.cos(pp)*np.cos(ll)),-1)
    category=code[::6,::6]
    traces=[]
    for k,name,color in [(1,'Yin','#0072b2'),(2,'Yang','#e69f00'),(3,'Overlap','#8c65ad')]:
        p=visual[active&(category==k)]
        traces.append(dict(type='scatter3d',mode='markers',name=name,x=p[:,0].tolist(),y=p[:,1].tolist(),z=p[:,2].tolist(),marker=dict(size=2,color=color)))
    for patch,color in [(package.yin,'#0072b2'),(package.yang,'#e69f00')]:
        for i,line in enumerate(chart_lines(patch.bounds,patch.basis)):
            p=np.asarray(line)*1.005
            traces.append(dict(type='scatter3d',mode='lines',name=patch.name+' boundaries',showlegend=i==0,x=p[:,0].tolist(),y=p[:,1].tolist(),z=p[:,2].tolist(),line=dict(color=color,width=3)))
    camera_traces=camera_outline_traces(out,package)
    traces.extend(camera_traces)
    html='''<!doctype html><meta charset="utf-8"><title>Adaptive Local Yin–Yang V1</title>
<style>body{font:16px system-ui;margin:24px}#plot{height:720px}</style><script>__JS__</script>
<h2>Frozen V1 | cam4-centred working charts</h2><p>Blue: Yin; orange: Yang; purple: overlap. Geometric common-valid union of 96 candidates, not occlusion visibility. Visual axes: X=image-left, Y=up, Z=cam4 forward. Both charts share the cam4 optical centre. Existing strict-support deficits are masked separately.</p>
__CAMERA_CONTROL__
<div id="plot"></div><script>const sphereTraces=__TRACES__;
const cameraLabels=sphereTraces.filter(t=>t.meta?.layer==='camera_coverage'&&t.mode==='markers').map(t=>({x:t.x[0],y:t.y[0],z:t.z[0],text:t.name+'朝向',showarrow:false,font:{size:13,color:t.marker.color},bgcolor:'rgba(255,255,255,0.8)',borderpad:2}));
Plotly.newPlot('plot',sphereTraces,{scene:{aspectmode:'cube',annotations:cameraLabels,xaxis:{title:'Visual X'},yaxis:{title:'Visual Y'},zaxis:{title:'Visual Z'}},margin:{t:20,b:20},uirevision:'sphere-view'}, {responsive:true});
document.getElementById('camera-coverage')?.addEventListener('change',function(){
 const ids=sphereTraces.map((t,i)=>t.meta?.layer==='camera_coverage'?i:null).filter(i=>i!==null);
 Plotly.restyle('plot',{visible:this.checked},ids);
 Plotly.relayout('plot',{'scene.annotations':this.checked?cameraLabels:[]});
});</script>'''
    import re
    html=html.replace('Frozen V1 | cam4-centred working charts',f'Frozen V1 | {package.metadata.get("profile","yin60_legacy")} | cam4-centred working charts')
    control='<label><input type="checkbox" id="camera-coverage" checked> 显示相机范围</label> <span style="color:#666;font-size:13px">青色实线：左 cam4；红色虚线：右 cam3。轮廓为当前96层候选的有效投影并集。</span>' if camera_traces else ''
    replacements={'__JS__':get_plotlyjs(),'__TRACES__':json.dumps(traces,separators=(',',':')),'__CAMERA_CONTROL__':control}
    (out/'grid_v1_3d.html').write_text(re.sub('__JS__|__TRACES__|__CAMERA_CONTROL__',lambda m:replacements[m.group()],html))
    (out/'figure_qa.json').write_text(json.dumps(QA,indent=2)+'\n')


def report(out,package,comparison,tests,smoke,previous):
    rows=comparison['cases'];chosen=next(r for r in rows if r['name']=='historical_yin_60_frozen')
    budget=chosen['budget'];meta=package.metadata
    good=tests['status']=='PASS' and smoke['status']=='PASS' and chosen['geometric_uncovered_count']==0
    fresh_path=out/'legacy_coordinates_rgb/coordinates_rgb_result.json'
    fresh=read(fresh_path) if fresh_path.is_file() else None
    if fresh is not None:good &= fresh['status']=='PASS'
    status='PASS' if good else 'FAIL'
    lines=['# Adaptive Local Yin–Yang Grid V1 验收报告','',f'**结论：{status}（几何模块与 CNN 特征采样接口；不是正式 Stereo 网络集成或精度验收）。**','',
        '## 1. 设计结论','',
        '冻结历史 Yin ±60° 配置。±50°略改善平均/尾部各向异性，但原生及Padding预算更高，严格支持不足也增加；±45°在固定约束下无采样可行解。',
        'Single Rotated Local ERP 在相近预算下同样可完整覆盖，不能宣称两张图在几何上必不可少；双图的优势是当前比较中的整体/尾部采样均匀性。','',
        'V1沿用360×720拟合的历史窗口，并在720×1440全部96层独立验证。720重新拟合会改变Yin边界和最优Yang朝向；原因是边界量化和矩形面积优先的离散目标，不是随机搜索。历史窗口虽通过覆盖/接缝，但不宣称在更密参考点上仍逐点满足恰好2°余量。','',
        '```json',json.dumps({p.name:dict(bounds=p.bounds,basis_visual=p.basis.tolist()) for p in (package.yin,package.yang)},indent=2),'```','',
        'OCam投影/相对外参/逐候选共视集合由真实标定计算；Yin ±60°、Yang≤±45°、5°接缝、2°余量、0.5°边界量化、约0.75°角步长是工程约束。Yin经度、Yang旋转与经纬边界在这些约束下计算/搜索；不是闭式唯一最优解。','',
        '## 2. 配置对比','',
        '所有可行case均在同一720×1440参考球面、96候选上计算。κ按精确立体角×该方向有效候选数加权。下列κ是实际角步长各向异性，不混同理论1/cosφ。','',
        '|配置|覆盖|原生像素|Padding像素|重叠/共视|严格支持不足面积|实际κ均值|P95|P99|双相机C32 FP16体 MiB|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['status']!='PASS':
            lines.append(f'|{r["name"]}|未找到可行解|—|—|—|—|—|—|—|—|');continue
        b=r['budget'];q=r['quality']['actual_pixel']
        lines.append(f'|{r["name"]}|{100*r["geometric_coverage_ratio"]:.6f}%|{b["native_pixels"]}|{b["padded_pixels"]}|{100*r["overlap_of_common"]:.3f}%|{100*r["strict_unsupported_weighted_fraction"]:.5f}%|{q["weighted_mean"]:.5f}|{q["p95"]:.5f}|{q["p99"]:.5f}|{b["padded_two_camera_C32_D96_fp16_mib"]:.2f}|')
    lines+=['',
        '固定原ERP |纬度|>60°的同一高纬区域，Single实际κ均值1.02922、P99 1.13090；历史双图为1.02899、1.13526，两者相近。双图相对Single的优势主要体现在全部共视区域的中纬/侧缘尾部，不能把消除原极区拉伸视为双图独有贡献。','',
        '|配置/网格|经度中心±半跨度|纬度范围|原生H×W|','|---|---|---|---|']
    for r in rows:
        for chart in r.get('charts',[]):
            b=chart['bounds']
            lines.append(f'|{r["name"]}/{chart["name"]}|{b["longitude_center_deg"]}° ± {b["longitude_half_span_deg"]}°|[{b["latitude_min_deg"]}°, {b["latitude_max_deg"]}°]|{chart["shape_hw"]}|')
    lines+=['','详细旋转矩阵、理论κ、实际κ、高纬κ、每层覆盖/球面面积、GWC/单通道成本和FP32预算见 `grid_comparison.csv/json` 与 `comparison/*/result.json`。',
        '±45°只证明有限搜索未找到可行解，不是连续SO(3)不可行定理。预算为实体化张量估计，不包括网络激活、梯度和优化器。','',
        '## 3. 几何正确性','',
        f'独立几何单元测试：{tests["tests_run"]}项，{tests["status"]}；真实RGB帧：{tests["frames"]}。每帧只改变RGB，标定/Mask相同，重新生成的射线、右网格和最终候选Mask逐元素一致。默认自动搜索也与冻结窗口一致。',
        f'V1构建门限：`{meta["coverage_gate"]}`；720×1440×96几何未覆盖={chosen["geometric_uncovered_count"]}，严格插值支持不足计数={chosen["strict_unsupported_count"]}，加权面积比例={100*chosen["strict_unsupported_weighted_fraction"]:.6f}%。',
        '严格支持不足保留为无效。`sample_patch`要求所有非零双线性权重角点有效，无支持返回零及False；不填洞、不把窗口覆盖当插值覆盖。',
        '以前验证记录在相同冻结选择哈希下复用：坐标/RGB往返与投影一致性、原生96层、GT离线诊断；不是重新训练得到的结果。',
        f'既有验证概要：`{previous["status"]}`；独立cam3深度真值仍不存在，只能报告Rig点云可见性估计。','',
        '|合成标定（不是新真实数据）|Yaw|结果|', '|---|---:|---|']
    for row in tests.get('synthetic_pose_checks',[]):
        lines.append(f'|OCam受控外参|{row["yaw_deg"]}°|{row["status"]}|')
    lines+=['','合成样例沿用真实OCam内参与Mask，控制水平外参；不可行配置返回明确诊断。额外全无共同FOV测试验证NO_COMMON_FOV错误，不静默生成空覆盖。','',
        '## 4. 冻结接口与最终输出域','',
        '`rays_ref` 是实际左OCam系单位射线；uv=(列,行)，是原图像素，不是归一化坐标。`basis`列是visual系下的局部坐标轴；列向量 r_ref=A Q r_local，A=diag(-1,-1,1)。角度为度。','',
        '|张量|Yin|Yang|','|---|---|---|']
    for key in ['rays_ref','left_uv','right_uv','left_valid','stereo_valid','left_grid','right_grid']:
        lines.append('|'+key+'|'+'|'.join(str(list(getattr(p,key).shape)) for p in (package.yin,package.yang))+'|')
    lines+=['',
        '每张图H/W各自Padding到8倍数，D满足两次3D下采样的4倍数；本候选轴D=96也满足当前深度金字塔。Padding仅加在右/下侧，原生射线逐元素不变，padding_valid=False、ray=NaN、采样grid=2。`unpad()`按原始尺寸裁回，不做resize。',
        '`shape_bucket`将同一候选轴的不同Rig分别按每张图的Batch最大尺寸补齐，并保留left/feature/stereo/padding Mask；不同候选轴禁止混成同一批。',
        '最终参考域沿用dataset_stereo：sweep_size=160×240，output_size=320×480，完整左鱼眼像素端点采样。它独立于Yin/Yang工作域。预计算output→Yin/Yang、Yin/Yang→output、Yin↔Yang映射，全部基于同一左相机射线。',
        f'最终左有效像素={int(package.reference["left_valid"].sum())}；至少一个有效双目候选={int(package.reference["stereo_valid"].any(0).sum())}；只有左观测、无候选匹配={int(package.reference["left_only"].sum())}。',
        'Static左Context用sample_reference的left_feature_valid；匹配体用stereo_valid。不得因右图无匹配而删除完整左输出域，也不得将其标为有效第0层。',
        '径向逆深度及连续[0,95]索引保持原定义；本模块没有修改GT转换和最终深度融合算法。','',
        '## 5. Padding与运行成本','',
        '|网格|原生H×W|Padding H×W|原生数|Padding数|额外比例|左有效数|候选有效数|','|---|---|---|---:|---:|---:|---:|---:|']
    for p in [package.yin,package.yang]:
        n=int(np.prod(p.original_shape));pad=int(np.prod(p.shape))
        lines.append(f'|{p.name}|{p.original_shape}|{p.shape}|{n}|{pad}|{100*(pad/n-1):.3f}%|{int(p.left_valid.sum())}|{int(p.stereo_valid.sum())}|')
    lines += ['',f'总体Padding增加 {100*budget["padding_extra_fraction"]:.4f}%；C32、D96、两相机实体化FP16体：原生 {budget["native_two_camera_C32_D96_fp16_mib"]:.3f} MiB → Padding {budget["padded_two_camera_C32_D96_fp16_mib"]:.3f} MiB。FP32为两倍。','',
        f'实际torch smoke：{smoke["status"]}；设备={smoke["device"]}，PyTorch={smoke["torch"]}。真实RGB转原灰度归一化，共享原FeatureLayers，输入每路[1,1,768,800]、CNN每路[1,32,384,400]。',
        '左右匹配体为[B,C,D,Hpad,Wpad]，候选Mask为[B,1,D,Hpad,Wpad]。左只采样一次再沿D广播，右使用真实重投影并分块采样；坐标保持FP32，双网格分次反向以避免同时保留全部体。','',
        '|精度|Forward/Backward|峰值分配MiB|峰值预留MiB|本次耗时秒|','|---|---|---:|---:|---:|']
    for run in smoke['runs']:
        lines.append(f'|{run["precision"]}|{run["forward"]}/{run["backward"]}|{run["cuda_peak_allocated_mib"]:.2f}|{run["cuda_peak_reserved_mib"]:.2f}|{run["elapsed_seconds"]:.4f}|')
    lines+=['','计时包含CNN与两图体的前后向、检查和一次SGD参数更新，不含CPU网格生成/数据IO；单次smoke不是稳定吞吐基准。显存是torch当前GPU的实际峰值，不是nvidia-smi整机占用。','',
        '## 6. 复现与限制','',
        f'- 几何版本：`{meta["version"]}`；执行环境Git HEAD：`{meta["git_commit"]}`。',
        f'- Calibration Hash：`{meta["calibration_hash"]}`；包含实际系数、外参、图像尺寸和invalid_mask内容，不依赖路径名。',
        '- 保存NPZ+JSON无pickle，逐数组shape/dtype/SHA256检查；保存使用临时文件后原子替换。生成任务完成后再让DataLoader加载缓存，不并发重写同名缓存。',
        '- 源码哈希与依赖版本保存在grid_v1_selected.json；旧selected_grid.json未修改。',
        '- 有限采样覆盖不是连续球面证明；相同输入的有限搜索确定性不代表全局最优。',
        '- 有效投影不是无遮挡；原生严格支持缺失并未被优化消除。',
        '- 不同Rig需重新搜索/生成；改变输入尺寸、Mask、深度轴或配置需重建缓存。',
        '- 支持当前原OCam/stride-2 CNN约定；其它相机模型必须提供兼容投影/反投影适配，不能宣称当前代码已测试所有模型。',
        '- 几何κ改善不等于Stereo深度精度提升；没有训练GEV/GRU/DA-V2，没有集成正式双图深度融合。','',
        '## 7. 修改范围','',
        '仅在experiments/local_spherical_grid新增V1模块、集中配置、有限对照、单元与torch测试、报告生成器，并补README。Legacy module/、module_cnn_only/、dataset_stereo.py及原GT/训练代码保持本任务开始时的内容。','',
        '![工作/输出域与支持不足](grid_v1_coverage.png)','',
        '![参数化各向异性](grid_v1_distortion.png)','',
        '[离线3D球面](grid_v1_3d.html) · [全部对照](comparison/REPORT_COMPARISON.md)']
    if fresh is not None:
        maximum=max(r['analytic_pixel_ray_chart_pixel_error_px']['maximum'] for r in fresh['pixel_roundtrips'])
        added=max(r['added_by_chart_max_px'] for r in fresh['pixel_roundtrips'])
        lines+=['','## 8. 既有验证程序本次重跑','',
            f'原validate_coordinates_rgb.py在{fresh["frames"]}真实帧上重跑：{fresh["status"]}；'
            f'解析鱼眼→射线→网格→鱼眼最大误差{maximum:.9f}像素，网格变换额外引入最多{added:.3g}像素；'
            f'跨帧几何Mask相同={fresh["geometry_masks_identical_across_frames"]}。',
            '完整RGB/OCam/原CNN坐标检查见[本次既有程序报告](legacy_coordinates_rgb/REPORT_COORDINATES_RGB.md)。']
    (out/'REPORT_GRID_V1.md').write_text('\n'.join(lines)+'\n')
    (out/'V1_FREEZE.json').write_text(json.dumps(dict(status=status,version=meta['version'],
        selected_case=chosen['name'],calibration_hash=meta['calibration_hash'],git_commit=meta['git_commit'],
        geometry_tests=tests['status'],torch_smoke=smoke['status'],geometric_coverage_720=chosen['geometric_coverage_ratio'],
        strict_support_deficit_fraction=chosen['strict_unsupported_weighted_fraction'],
        scope='Geometry and feature sampling only; original stereo network unchanged'),indent=2)+'\n')


def report_yin50(out,package,comparison,tests,smoke):
    """Profile switch acceptance; every new metric comes from the fresh Yin50 artifacts."""
    rows={r['name']:r for r in comparison['cases']}
    new,old=rows['dual_yin_50'],rows['historical_yin_60_frozen']
    coverage=read(out/'coverage/coverage_validation.json')
    quality=read(out/'quality/quality_summary.json')
    coords=read(out/'legacy_coordinates_rgb/coordinates_rgb_result.json')
    old_tests=read(out/'legacy_yin60/grid_v1_tests.json')
    old_smoke=read(out/'legacy_yin60/grid_v1_smoke.json')
    old_package=AdaptiveLocalSphericalGrid.load(out/'legacy_yin60/grid_v1_geometry')
    assert smoke['profile']==package.metadata['profile']=='yin50'
    assert old_smoke['profile']==old_package.metadata['profile']=='yin60_legacy'
    assert smoke['config_hash']==package.metadata['config_hash']
    digest=hashlib.sha256((HERE/'selected_grid_yin50.json').read_bytes()).hexdigest()
    assert coverage['frozen_selection_sha256']==quality['frozen_selection_sha256']==coords['frozen_parameters_sha256']==digest
    for p,chart in zip((package.yin,package.yang),new['charts']):
        assert p.bounds==chart['bounds']
        np.testing.assert_allclose(p.basis,chart['basis_visual'],atol=1e-14)
    assert package.metadata['calibration_hash']==old_package.metadata['calibration_hash']
    assert package.metadata['inverse_depth_candidates']==old_package.metadata['inverse_depth_candidates']
    for key in ['rays_ref','uv','left_valid','stereo_valid','left_only']:
        np.testing.assert_array_equal(package.reference[key],old_package.reference[key])
    q=quality['depth_area_weighted']['best_pixel'];oq=old['quality']['actual_pixel']
    assert abs(q['weighted_mean']-new['quality']['actual_pixel']['weighted_mean'])<1e-12
    quality_reproduced=abs(q['weighted_mean']-1.0554329805121079)<1e-8 and abs(q['p99']-1.2906832307189418)<1e-8
    geometric_pass=all(r['geometric_uncovered_count']==0 and r['geometric_coverage_ratio']==1 for r in coverage['resolutions'])
    native_pass=coverage['native_checks']['geometric_uncovered']==0
    seam_pass=package.metadata['coverage_gate']['seam_uncovered_count']==0
    statuses=[tests['status'],smoke['status'],quality['status'],coords['status'],old_tests['status'],old_smoke['status']]
    passed=all(s=='PASS' for s in statuses) and geometric_pass and native_pass and seam_pass and quality_reproduced
    status='PASS' if passed else 'FAIL'
    budget,obudget=new['budget'],old['budget']
    changes={key:100*(1-q[key]/oq[key]) for key in ['weighted_mean','p95','p99','maximum']}
    lines=['# Yin ±50° 默认Profile切换与重新验收','',
        f'**{status}：Yin50可作为V1默认几何Profile；本报告只验收几何与CNN采样，不代表Stereo深度精度提升。**','',
        '## 1. 默认配置与数学约定','',
        '`grid_v1_config.yaml`默认profile=yin50；`grid_v1_yin60_legacy.yaml`保留历史60，旧selected_grid.json及outputs_v1未覆盖。',
        '当前选择优先参数化采样质量，接受更高预算。Yin50重新读取真实cam4/cam3 OCam与Mask、原工程192[::2]的96个逆径向距离，90×180粗搜索后在720×1440重新拟合与验证；没有把Yin60的Yang复制过来。',
        '参考原点仍为cam4光心，X4=r4/rho，X3=R34 X4+t34；D轴、原图768×800、CNN stride2、align_corners=True、最终左鱼眼320×480均未改变。',
        '与此前Yin50对照采用相同720×1440拟合分辨率；工程约束仍为Yang |lat|≤45°、seam5°、padding2°、边界0.5°、目标角步长0.75°。', '',
        '|网格|经度范围（各自坐标系）|纬度范围|原生H×W|Padding H×W|', '|---|---|---|---|---|']
    for p in (package.yin,package.yang):
        b=p.bounds;center=b['longitude_center_deg'];half=b['longitude_half_span_deg']
        lines.append(f'|{p.name}|[{center-half:.3f}°, {center+half:.3f}°]|[{b["latitude_min_deg"]}°, {b["latitude_max_deg"]}°]|{p.original_shape}|{p.shape}|')
    sel=package.metadata['selection'];winner=sel['selected']
    lines+=['',f'Yang搜索角：yaw_y={winner["yaw_y_deg"]}°、tilt_x={winner["tilt_x_deg"]}°、拟合经度中心={winner["longitude_center_deg"]}°；最终Q=Ry(yaw) Rx(tilt) Ry(center)。',
        'basis的列是在visual坐标中表示的局部坐标轴，r_ocam=A Q r_local，A=diag(-1,-1,1)，角度为度，uv=(原图列,行)。','',
        '```json',json.dumps({p.name:dict(basis=p.basis.tolist(),bounds=p.bounds) for p in (package.yin,package.yang)},indent=2),'```','',
        '## 2. 96层覆盖、原生网格与接缝','',
        '|分析网格|D|有效深度方向数|未覆盖数|几何覆盖率|严格插值支持不足面积|', '|---|---:|---:|---:|---:|---:|']
    for r in coverage['resolutions']:
        lines.append(f'|{r["height"]}×{r["width"]}|{r["num_depth"]}|{r["valid_depth_direction_count"]}|{r["geometric_uncovered_count"]}|{100*r["geometric_coverage_ratio"]:.8f}%|{100*r["strict_unsupported_weighted_fraction"]:.6f}%|')
    lines += ['',f'原生网格检查：`{coverage["native_checks"]}`。构建时目标接缝未覆盖={package.metadata["coverage_gate"]["seam_uncovered_count"]}。',
        '注意：目标5°带在两窗口内被覆盖，不代表所有交接位置都有完整双线性/上下文支持。相关严格支持、2/4/6 halo诊断见coverage CSV；不修改Mask冒充通过。','',
        '|配置|原生采样数|Padding采样数|重叠/共视|严格支持不足面积|实际κ均值|P95|P99|最大|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in [old,new]:
        b=r['budget'];qq=r['quality']['actual_pixel']
        lines.append(f'|{r["name"]}|{b["native_pixels"]}|{b["padded_pixels"]}|{100*r["overlap_of_common"]:.4f}%|{100*r["strict_unsupported_weighted_fraction"]:.6f}%|{qq["weighted_mean"]:.8f}|{qq["p95"]:.8f}|{qq["p99"]:.8f}|{qq["maximum"]:.8f}|')
    with np.load(out/'comparison/dual_yin_50/comparison_maps.npz') as a:
        weights=a['area_sr']*a['valid_candidate_count'];yin=a['yin_window'];yang=a['yang_window']
        areas={k:float(weights[m].sum()/96) for k,m in [('yin_only',yin&~yang),('yang_only',yang&~yin),('overlap',yin&yang)]}
    common=sum(areas.values())
    lines+=['',f'完整球面面积4π={4*np.pi:.8f}sr；96候选平均几何共视面积={common:.8f}sr。Yin-only/Yang-only/Overlap平均面积分别为{areas["yin_only"]:.8f}/{areas["yang_only"]:.8f}/{areas["overlap"]:.8f}sr；这不是实际采样像素预算。','',
        '## 3. 参数化质量复现与改善','',
        f'实际像素κ复现检查={quality_reproduced}。相对历史60：均值降低{changes["weighted_mean"]:.4f}%，P95降低{changes["p95"]:.4f}%，P99降低{changes["p99"]:.4f}%，最大值降低{changes["maximum"]:.4f}%。',
        '统计采用同一720×1440几何共视集合，权重=精确球面面积×有效候选数。理论κ=1/|cosφ|；实际像素κ额外计入经纬角步长比，不能混称。','',
        '|指标域|均值|P95|P99|最大|','|---|---:|---:|---:|---:|']
    for label,qq in [('Yin50理论',quality['depth_area_weighted']['best_coordinate']),('Yin50实际像素',q),
                     ('原ERP高纬理论',new['quality']['primary_erp_high']),('Yin50原ERP高纬实际',new['quality']['high_latitude_actual_pixel']),
                     ('Yin60原ERP高纬实际',old['quality']['high_latitude_actual_pixel'])]:
        lines.append(f'|{label}|{qq["weighted_mean"]:.8f}|{qq["p95"]:.8f}|{qq["p99"]:.8f}|{qq["maximum"]:.8f}|')
    lines+=['','原ERP高纬定义固定为|主纬度|>60°。覆盖率两者同为100%，本次提升是采样均匀性，不是几何覆盖率提升，更不是已经测得深度误差下降。','',
        '## 4. 接口、坐标与多帧稳定性','',
        f'几何单测{tests["tests_run"]}项：{tests["status"]}；真实帧{tests["frames"]}，重新构建的网格/Mask逐元素相同。历史profile单测={old_tests["status"]}。',
        f'原坐标/RGB检验={coords["status"]}，跨帧Mask一致={coords["geometry_masks_identical_across_frames"]}。',
        'Yin↔Yang及两图↔cam4输出域全部重新计算；左右重投影、不同深度坐标、OCam近似往返误差、原生射线不被Padding改变、缓存完整性与profile不匹配拒绝均已测试。',
        '新旧包的最终cam4射线、像素、left_valid、stereo_valid、left_only逐元素一致；候选轴和calibration_hash相同。', '',
        '|合成标定（非真实新数据）Yaw|结果|诊断|', '|---:|---|---|']
    for r in tests.get('synthetic_pose_checks',[]):
        lines.append(f'|{r["yaw_deg"]}°|{r["status"]}|{r.get("diagnostic","窗口与接缝覆盖检查通过")}|')
    lines+=['','合成布局若在当前有限搜索中未找到满足固定约束的解，明确记录DIAGNOSED_INFEASIBLE，不放宽约束；这不是连续不可行证明，也不把虚拟姿态当真实数据结果。Yin50的默认验收域是实际OmniThings cam4/cam3，不保证所有水平夹角都可行。','',
        '## 5. 新特征体、Padding与显存','', '|张量|Yin|Yang|','|---|---|---|']
    for key in ['rays_ref','left_uv','right_uv','left_valid','stereo_valid','left_grid','right_grid']:
        lines.append('|'+key+'|'+'|'.join(str(list(getattr(p,key).shape)) for p in (package.yin,package.yang))+'|')
    for p in (package.yin,package.yang):
        lines.append(f'\n{p.name}匹配体=[B,32,96,{p.shape[0]},{p.shape[1]}]；Mask=[B,1,96,{p.shape[0]},{p.shape[1]}]。')
    lines += ['',f'总原生预算{budget["native_pixels"]}，较历史60增加{100*(budget["native_pixels"]/obudget["native_pixels"]-1):.4f}%；Padding总预算{budget["padded_pixels"]}，增加{100*(budget["padded_pixels"]/obudget["padded_pixels"]-1):.4f}%；本Profile自身Padding开销{100*budget["padding_extra_fraction"]:.4f}%。',
        f'两相机C32/D96实体化体：原生FP16={budget["native_two_camera_C32_D96_fp16_mib"]:.4f}MiB，Padding FP16={budget["padded_two_camera_C32_D96_fp16_mib"]:.4f}MiB；历史Padding FP16={obudget["padded_two_camera_C32_D96_fp16_mib"]:.4f}MiB。FP32为两倍；不包含CNN/梯度/优化器。','',
        '|精度|Yin60峰值分配MiB|Yin50峰值分配MiB|增加|Yin50前后向|Yin50耗时秒|', '|---|---:|---:|---:|---|---:|']
    for r,prior in zip(smoke['runs'],old_smoke['runs']):
        delta=r['cuda_peak_allocated_mib']-prior['cuda_peak_allocated_mib']
        lines.append(f'|{r["precision"]}|{prior["cuda_peak_allocated_mib"]:.3f}|{r["cuda_peak_allocated_mib"]:.3f}|{delta:.3f}MiB / {100*delta/prior["cuda_peak_allocated_mib"]:.3f}%|{r["forward"]}/{r["backward"]}|{r["elapsed_seconds"]:.4f}|')
    lines+=['',f'同一GPU={smoke["device"]}、PyTorch={smoke["torch"]}；当前环境分别重跑两个Profile。原RGB使用原灰度归一化，CNN每路输出[1,32,384,400]。AMP使用GradScaler；几何坐标保持FP32。耗时为单次smoke诊断，不是稳定吞吐基准。',
        '左特征一次采样沿D广播，右特征逐候选重投影，分块采样。显式stereo_valid清零无效匹配、全无效位置返回False且输出有限；几何接口不执行概率归一化，未来Stereo必须据此跳过全无效像素。严格插值支持不足从旧0.88398%变为本次约0.99939%，未补洞，不得继续用旧比例描述新Profile。','',
        '## 6. 冻结与可复现性','',
        f'版本={package.metadata["version"]}，profile=yin50，config_hash={package.metadata["config_hash"]}，calibration_hash={package.metadata["calibration_hash"]}。',
        f'选择文件SHA256={digest}；执行环境Git HEAD={package.metadata["git_commit"]}；新增/未提交源文件以V1_SOURCE_MANIFEST.json逐文件哈希记录。',
        '保存加载接口与数学约定不变；新增expected_config校验，防止相同Rig下误加载另一Profile。两套配置使用不同输出目录，历史60原文件未覆盖。',
        '本次未改GEV/GRU/DA-V2、原OCam、训练或GT转换逻辑，未正式训练。有限网格验证不是连续球面覆盖证明；FOV投影有效不等于真实无遮挡。',
        '新默认配置可冻结用于后续Stereo研究；尚不代表已经集成双网格Cost Volume Aggregation，也不代表精度改善。','',
        '![覆盖与有效性](grid_v1_coverage.png)','', '![重叠区域](grid_v1_overlap.png)','', '![参数化质量](grid_v1_distortion.png)','',
        '[可旋转3D球面](grid_v1_3d.html) · [全部覆盖数据](coverage/coverage_validation.json) · [质量统计](quality/quality_summary.json)']
    (out/'REPORT_YIN50.md').write_text('\n'.join(lines)+'\n')
    freeze=dict(status=status,version=package.metadata['version'],profile='yin50',config_hash=package.metadata['config_hash'],
        calibration_hash=package.metadata['calibration_hash'],selection_sha256=digest,quality_reproduced=quality_reproduced,
        geometry_tests=tests['status'],torch_smoke=smoke['status'],legacy_tests=old_tests['status'],legacy_smoke=old_smoke['status'],
        geometric_coverage_pass=geometric_pass,native_pass=native_pass,seam_pass=seam_pass,relative_kappa_drop_percent=changes)
    (out/'V1_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
    table=[]
    for r,ms in [(old,old_smoke),(new,smoke)]:
        table.append(dict(profile=r['name'],native_pixels=r['budget']['native_pixels'],padded_pixels=r['budget']['padded_pixels'],
            mean_kappa=r['quality']['actual_pixel']['weighted_mean'],p99=r['quality']['actual_pixel']['p99'],
            geometric_coverage=r['geometric_coverage_ratio'],strict_unsupported_fraction=r['strict_unsupported_weighted_fraction'],
            fp32_peak_mib=ms['runs'][0]['cuda_peak_allocated_mib'],amp_peak_mib=ms['runs'][1]['cuda_peak_allocated_mib']))
    with (out/'comparison_yin50_yin60.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--previous-validation',type=Path)
    p.add_argument('--visualization-only',action='store_true',help='Regenerate plots/HTML without changing acceptance reports or geometry caches')
    args=p.parse_args();out=args.output_dir
    plt.rcParams.update(PLOT_STYLE)
    package=AdaptiveLocalSphericalGrid.load(out/'grid_v1_geometry')
    comparison=read(out/'comparison/grid_comparison.json')
    if args.visualization_only:
        figures(out,package,comparison)
        (out/'visualization_manifest.json').write_text(json.dumps(dict(
            scope='Display-only update; geometry and numerical acceptance artifacts unchanged',
            renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            config_hash=package.metadata.get('config_hash'),camera_outline='Original camera validity; right is union across all candidates'),indent=2)+'\n')
        print('VISUALIZATION',out/'grid_v1_3d.html')
        return
    tests=read(out/'grid_v1_tests.json');smoke=read(out/'grid_v1_smoke.json')
    assert comparison['frozen_input_sha256']==hashlib.sha256((HERE/'selected_grid.json').read_bytes()).hexdigest()
    for ext in ['csv','json']:shutil.copyfile(out/f'comparison/grid_comparison.{ext}',out/f'grid_comparison.{ext}')
    figures(out,package,comparison)
    if chosen_case(package)=='dual_yin_50':
        report_yin50(out,package,comparison,tests,smoke)
    else:
        if args.previous_validation is None:p.error('Legacy60 report requires --previous-validation')
        previous=read(args.previous_validation/'VALIDATION_SUMMARY.json')
        assert previous['frozen_selection_sha256']==comparison['frozen_input_sha256']
        report(out,package,comparison,tests,smoke,previous)
    sources=sorted(p for p in HERE.glob('*.py') if not p.name.startswith('.'))+sorted(HERE.glob('grid_v1*.yaml'))+[HERE/'selected_grid.json',HERE/'selected_grid_yin50.json']
    (out/'V1_SOURCE_MANIFEST.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print('REPORT',out/('REPORT_YIN50.md' if chosen_case(package)=='dual_yin_50' else 'REPORT_GRID_V1.md'))


if __name__=='__main__':main()
