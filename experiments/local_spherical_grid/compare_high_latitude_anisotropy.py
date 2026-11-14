"""只读已有验证结果，比较同一高纬度几何共视方向在 ERP / Yin-Yang 的各向异性。

本脚本仅依赖 NumPy / Matplotlib / Plotly，不加载网络、不搜索、不修改网格。
统计权重=球面格点的精确立体角×该方向有效候选数量，与原质量报告一致。
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm


def plotly_bundle(root):
    # 纯绘图环境若没有Python Plotly，复用前次已生成离线页面中的同一JS库。
    try:
        from plotly.offline import get_plotlyjs
        return get_plotlyjs()
    except ModuleNotFoundError:
        import re
        previous=root/'outputs_yang_search/yin_yang_search.html'
        match=re.search(r'<script>([\s\S]*?)</script>',previous.read_text())
        if match is None or 'plotly.js' not in match.group(1):
            raise RuntimeError('Plotly library or the prior offline page is required')
        return match.group(1)


def stats(values, weights, mask):
    good=mask & np.isfinite(values) & (weights>0)
    x,w=values[good],weights[good]
    order=np.argsort(x,kind='stable')
    cdf=(np.cumsum(w[order])-.5*w[order])/w.sum()
    return dict(count=int(good.sum()),mean=float(np.average(x,weights=w)),
        p95=float(np.interp(.95,cdf,x[order])),p99=float(np.interp(.99,cdf,x[order])),
        maximum=float(x.max()))


def configure_font():
    names={f.name for f in font_manager.fontManager.ttflist}
    for name in ['Noto Sans CJK SC','Source Han Sans SC','PingFang SC','Heiti SC','SimHei']:
        if name in names:
            plt.rcParams['font.family']=[name,'DejaVu Sans']
            break
    else:
        raise RuntimeError('A Chinese font is needed for the Chinese figures')
    plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'savefig.dpi':180,
                         'pdf.fonttype':42,'svg.fonttype':'none'})


def save(fig,path,qa,vector=False):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():continue
        b=text.get_window_extent(renderer)
        if b.width and b.height and (b.x0 < -2 or b.y0 < -2 or b.x1 > fig.bbox.width+2 or b.y1 > fig.bbox.height+2):
            outside.append(text.get_text())
    qa.append(dict(figure=path.name,text_outside_canvas=outside))
    fig.savefig(path,facecolor='white')
    if vector:
        for ext in ['pdf','svg']:fig.savefig(path.with_suffix('.'+ext),facecolor='white')
    plt.close(fig)


HTML=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ERP 与阴阳网格各向异性对比</title>
<style>body{font-family:system-ui;background:#f7f9fc;color:#162535;margin:20px}.box{padding:18px;background:white;border-radius:12px;margin-bottom:16px}h1{font-size:24px}p{line-height:1.65}#plot{height:510px}select{padding:7px}b{color:#005f9e}</style>
<script>__PLOTLY__</script><h1>同一批球面方向：ERP → 当前局部 Yin/Yang</h1>
<div class="box"><label>统计区域 <select id="region"><option value="60" selected>主 ERP 高畸变区域 |纬度| >60°</option>
<option value="70">更高纬区域 |纬度| >70°</option><option value="80">接近极区 |纬度| >80°</option><option value="0">全部几何共视区域</option></select></label>
<p id="summary"></p><p>κ=1/|cos纬度|，越接近1越均匀。两张图使用<b>相同色标、相同三维方向、相同有效域</b>。
右图仍按主 ERP 位置排列，颜色表示这些方向在阴阳网格中的κ；它不是 Yang 图的实际形状。
悬停可查看同一方向的前后值及转换后的 Yang 纬度。</p></div>
<div id="plot" class="box"></div><div class="box"><p>统计使用完整360×720参考球面、全部96候选，按精确球面面积与有效候选数量加权；显示图仅降低绘制分辨率。
色标最高显示32，超过32用顶色，JSON/CSV统计未截断。ERP精确极点理论发散，有限格点最大值依赖分析分辨率。</p>
<p>这里衡量的是参数化和采样各向异性，尚不代表深度误差降低，也不消除既有的原生共同支持缺失。</p>
<p><a href="anisotropy_comparison.png">完整对比图</a> · <a href="high_region_in_yang.png">高纬方向在Yang原生坐标中的位置</a> · <a href="REPORT_ANISOTROPY.md">统计说明</a></p></div>
<script>const D=__DATA__,region=document.getElementById('region');
function update(){const threshold=+region.value,k=String(threshold),a=D.summary[k];let z0=[],z1=[],cd=[];
for(let r=0;r<D.h;r++){let p=[],q=[],c=[];for(let col=0;col<D.w;col++){let i=r*D.w+col;
const use=D.valid[i]&&(threshold===0||Math.abs(D.lat[i])>threshold);p.push(use?Math.log2(D.before[i]):null);q.push(use?Math.log2(D.after[i]):null);
c.push([D.before[i],D.after[i],D.yanglat[i]]);}z0.push(p);z1.push(q);cd.push(c);}
const xy={x0:D.lon[0],dx:D.lon[1]-D.lon[0],y0:D.lat[0],dy:D.lat[D.w]-D.lat[0]};
const common={type:'heatmap',...xy,coloraxis:'coloraxis',customdata:cd,connectgaps:false,
hovertemplate:'主经度 %{x:.2f}°，主纬度 %{y:.2f}°<br>ERP κ=%{customdata[0]:.4f}<br>阴阳 κ=%{customdata[1]:.4f}<br>Yang纬度=%{customdata[2]:.2f}°<extra></extra>'};
Plotly.react('plot',[{...common,z:z0,xaxis:'x',yaxis:'y'},{...common,z:z1,xaxis:'x2',yaxis:'y2'}],
{margin:{t:55,b:65,l:60,r:90},annotations:[{text:'原 ERP：同一批方向',x:.22,y:1.12,xref:'paper',yref:'paper',showarrow:false},
{text:'阴阳参数化后：同一批方向',x:.79,y:1.12,xref:'paper',yref:'paper',showarrow:false}],
xaxis:{domain:[0,.46],title:{text:'主经度 (°)'},range:[-180,180],dtick:90},xaxis2:{domain:[.54,1],title:{text:'主经度 (°)'},range:[-180,180],dtick:90},
yaxis:{domain:[0,1],title:{text:'主纬度 (°)'},range:[-90,90],dtick:30},yaxis2:{domain:[0,1],title:{text:'主纬度 (°)'},range:[-90,90],dtick:30,anchor:'x2'},
coloraxis:{cmin:0,cmax:5,colorscale:'Viridis',colorbar:{title:{text:'各向异性 κ'},tickvals:[0,1,2,3,4,5],ticktext:['1','2','4','8','16','≥32']}}},{responsive:true});
document.getElementById('summary').innerHTML=`面积/96候选加权平均 κ：<b>${a.erp.mean.toFixed(3)} → ${a.yinyang.mean.toFixed(3)}</b>；P99：<b>${a.erp.p99.toFixed(3)} → ${a.yinyang.p99.toFixed(3)}</b>。`;
}region.addEventListener('change',update);update();</script></html>'''


def figures(out,data,summary,threshold,selection,native,qa):
    mask=data['high'];before=data['before'];after=data['after'];weights=data['weights']
    fig,axes=plt.subplots(2,2,figsize=(14.5,8.7),layout='constrained',gridspec_kw={'height_ratios':[1.1,1]})
    for ax,value,title in zip(axes[0],[before,after],['原 ERP：高纬区域的各向异性','阴阳网格后：同一批方向的各向异性']):
        im=ax.imshow(np.ma.array(value,mask=~mask),extent=[-180,180,-90,90],origin='upper',aspect='auto',
                     cmap='viridis',norm=LogNorm(1,32))
        ax.set_facecolor('#e8ebf0');ax.set(title=title,xlabel='主 ERP 经度 (°)',ylabel='主 ERP 纬度 (°)',
                    xticks=[-180,-90,0,90,180],yticks=[-90,-60,-30,0,30,60,90])
        ax.grid(alpha=.15)
    bar=fig.colorbar(im,ax=axes[0].tolist(),fraction=.025,extend='max',ticks=[1,2,4,8,16,32],label='κ（共享对数色标；>32显示封顶）')
    bar.set_ticklabels(['1','2','4','8','16','32']);bar.minorticks_off()
    ax=axes[1,0]
    for array,label,color,style in [(before,'原 ERP','#777777','-'),(after,'阴阳网格（理论）','#0072b2','-'),
                (data['pixel'],'阴阳网格（实际角步长）','#d55e00','--')]:
        x=array[mask];w=weights[mask];order=np.argsort(x);cdf=np.cumsum(w[order])/w.sum()
        ax.plot(x[order],100*cdf,label=label,color=color,linestyle=style,linewidth=2)
    ax.set(xscale='log',xlim=(1,256),ylim=(0,100),xlabel='各向异性（对数轴，1为各向同性）',
           ylabel='球面面积 / 深度加权累计比例 (%)',title='只比较同一高纬区域的分布')
    ax.set_xticks([1,2,4,8,16,32,64,128,256],['1','2','4','8','16','32','64','128','256'])
    ax.set_yticks([0,20,40,60,80,100]);ax.grid(alpha=.2);ax.legend(loc='lower right',frameon=False)
    ax=axes[1,1];ax.axis('off')
    fields=[('mean','面积加权均值'),('p95','P95'),('p99','P99'),('maximum','采样最大值')]
    rows=[[label]+[f'{summary[key][field]:.3f}' for key in ['erp','yinyang','pixel']] for field,label in fields]
    table=ax.table(cellText=rows,colLabels=['指标','原 ERP','阴阳理论κ','阴阳实际像素'],loc='center',bbox=[0,.28,1,.53],cellLoc='center')
    table.auto_set_font_size(False);table.set_fontsize(11)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#dce2e8')
        if r==0:cell.set_facecolor('#e9f0f6');cell.set_text_props(weight='bold')
    reduction=100*(1-summary['yinyang']['mean']/summary['erp']['mean'])
    ax.set_title(f'主 ERP |纬度|>{threshold:g}° 的同一几何共视区域')
    ax.text(.5,.15,f'平均 κ 数值下降 {reduction:.2f}%；新值接近1',ha='center',va='center',fontsize=13,color='#005f9e')
    ax.text(.5,.02,'这是参数化质量改善，不是深度误差下降比例。',ha='center',va='center',fontsize=10,color='#555')
    fig.suptitle('高畸变区域：ERP 与当前局部阴阳网格的公平对比',fontsize=17)
    save(fig,out/'anisotropy_comparison.png',qa,vector=True)

    # 辅助图在 Yang 自己的 native 网格显示对应区域；统计仍用上面的同一参考方向。
    lon=native['longitude_deg'];lat=native['latitude_deg'];q=np.asarray(selection['yang_basis_visual'])
    lam,phi=np.deg2rad(lon),np.deg2rad(lat)
    rays=np.stack((np.cos(phi)*np.sin(lam),np.sin(phi),np.cos(phi)*np.cos(lam)),-1)@q.T
    primary_lat=np.rad2deg(np.arcsin(np.clip(rays[...,1],-1,1)))
    active=(np.abs(primary_lat)>threshold)&(native['depth_valid_count']>0)
    b=selection['yang_bounds'];extent=[lon.min(),lon.max(),lat.min(),lat.max()]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,value,title,cmap,vmin,vmax in [(axes[0],np.abs(primary_lat),'这些方向原来在主 ERP 的纬度','cividis',threshold,90),
            (axes[1],native['coordinate_kappa'],'这些方向在 Yang 中的各向异性','viridis',1,1.16)]:
        im=ax.imshow(np.ma.array(value,mask=~active),extent=extent,origin='upper',aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set_facecolor('#e8ebf0');ax.set(title=title,xlabel='Yang 经度 (°)',ylabel='Yang 纬度 (°)',
                  xlim=(extent[0],extent[1]),ylim=(b['latitude_min_deg'],b['latitude_max_deg']))
        ax.set_xticks([-110,-55,0,55,110]);ax.set_yticks([-22,-10,0,10,20,30,37]);ax.grid(alpha=.2)
        fig.colorbar(im,ax=ax,fraction=.035,label='原主纬度绝对值 (°)' if ax is axes[0] else 'Yang 理论 κ')
    fig.suptitle('高纬方向在 Yang 自身坐标中的位置（辅助示意；native 格点与参考格点不同）',fontsize=13)
    save(fig,out/'high_region_in_yang.png',qa,vector=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent
    p.add_argument('--quality-dir',type=Path,default=root/'outputs_validation/quality')
    p.add_argument('--selection',type=Path,default=root/'outputs_yang_search/search_result.json')
    p.add_argument('--output-dir',type=Path,default=root/'outputs_anisotropy_comparison')
    p.add_argument('--latitude-threshold',type=float,default=60.)
    p.add_argument('--stats-only',action='store_true')
    args=p.parse_args()
    if not 0<args.latitude_threshold<90:p.error('latitude-threshold must lie in (0,90)')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    frozen=args.selection.read_bytes();selection=json.loads(frozen)
    with np.load(args.quality_dir/'reference_quality_maps.npz') as a:
        data=dict(before=a['primary_coordinate'],after=a['best_coordinate'],pixel=a['best_pixel'],
            lat=a['primary_latitude_deg'],yanglat=a['yang_latitude_deg'],weights=a['area_sr']*a['common_valid_count'],
            valid=a['common_valid_count']>0,yin=a['yin'],yang=a['yang'])
    data['high']=data['valid']&(np.abs(data['lat'])>args.latitude_threshold)
    assert np.isfinite(data['after'][data['high']]).all()
    metrics={'erp':data['before'],'yinyang':data['after'],'pixel':data['pixel']}
    groups={}
    for t in [0.,60.,70.,80.,args.latitude_threshold]:
        key=f'{t:g}';mask=data['valid']&((np.abs(data['lat'])>t) if t else True)
        groups[key]={k:stats(v,data['weights'],mask) for k,v in metrics.items()}
    summary=groups[f'{args.latitude_threshold:g}']
    weight=data['weights'];high=data['high']
    details=dict(frozen_selection_sha256=hashlib.sha256(frozen).hexdigest(),threshold_deg=args.latitude_threshold,
        mask_definition='Same reference directions: primary |latitude|>threshold and at least one binocular geometric-valid candidate',
        weighting='Exact spherical area times valid candidate count; 96 equal candidates, no GT',
        high_fraction_of_depth_area_common=float(weight[high].sum()/weight.sum()),
        high_covered_by_yang=bool(data['yang'][high].all()),high_yin_covered_count=int(data['yin'][high].sum()),
        all_sampled_high_directions_improved=bool(np.all(data['after'][high]<data['before'][high])),
        high_yang_latitude_min_deg=float(data['yanglat'][high].min()),high_yang_latitude_max_deg=float(data['yanglat'][high].max()),
        high_summary=summary,groups=groups,mean_kappa_relative_drop=1-summary['yinyang']['mean']/summary['erp']['mean'],
        limits=['Metric is coordinate/sampling anisotropy, not depth error',
                'ERP maximum depends on sampled resolution; exact poles diverge',
                'Native joint-support holes remain a separate unresolved validation finding'])
    (out/'comparison_stats.json').write_text(json.dumps(details,indent=2,ensure_ascii=False)+'\n')
    rows=[]
    for threshold,group in groups.items():
        for name,s in group.items():rows.append(dict(primary_abs_latitude_threshold_deg=threshold,metric=name,**s))
    with (out/'comparison_stats.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if args.stats_only:print(json.dumps(details,indent=2,ensure_ascii=False));return
    configure_font();qa=[]
    with np.load(args.quality_dir/'yang_native_quality.npz') as a:native={k:a[k] for k in a.files}
    figures(out,data,summary,args.latitude_threshold,selection,native,qa)
    h,w=data['lat'].shape;step=3
    longitude=-180+(np.arange(w)+.5)*360/w
    lon=np.broadcast_to(longitude[None],(h,w))
    small=lambda a:np.round(a[::step,::step].ravel(),6).tolist()
    # 非有效位置仅为安全序列化填1，绘制仍受独立valid mask控制。
    safe=lambda a:np.where(np.isfinite(a),a,1.)
    web=dict(h=data['lat'][::step,::step].shape[0],w=data['lat'][::step,::step].shape[1],
        before=small(safe(data['before'])),after=small(safe(data['after'])),lat=small(data['lat']),yanglat=small(data['yanglat']),
        lon=small(lon),valid=data['valid'][::step,::step].astype(int).ravel().tolist(),summary=groups)
    import re
    replace={'__PLOTLY__':plotly_bundle(root),'__DATA__':json.dumps(web,separators=(',',':'))}
    (out/'anisotropy_comparison.html').write_text(re.sub(r'__[A-Z]+__',lambda m:replace[m.group()],HTML))
    np.savez_compressed(out/'comparison_data.npz',**data)
    (out/'figure_qa.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False)+'\n')
    lines=['# 主ERP高畸变区域的各向异性对比','',
        f'固定网格、同一组主ERP |纬度|>{args.latitude_threshold:g}° 的双目几何有效方向，按球面面积×有效候选数加权；没有改变参数或使用GT。','',
        '| 指标 | 原ERP | 阴阳理论κ | 阴阳实际像素 |','|---|---:|---:|---:|']
    for key,title in [('mean','均值'),('p95','P95'),('p99','P99'),('maximum','采样最大值')]:
        lines.append('|'+title+'|'+'|'.join(f'{summary[k][key]:.8f}' for k in metrics)+'|')
    lines+=['',f'平均κ数值下降 {100*details["mean_kappa_relative_drop"]:.3f}%。本次采样高纬方向的理论κ全部降低。',
        f'该区域占全96候选几何共视域面积/深度权重的 {100*details["high_fraction_of_depth_area_common"]:.3f}%；'
        f'当前均由Yang覆盖，其Yang纬度范围为 [{details["high_yang_latitude_min_deg"]:.4f}°, {details["high_yang_latitude_max_deg"]:.4f}°]。','',
        '此前1.073567是全共视域的择优均值；本页约1.030636只统计原ERP高纬部分。区域不同，不能混用。','',
        'κ=1/|cosφ| 描述等角经纬参数化的各向异性，1为各向同性。实际原生像素还需考虑经纬角步长比，'
        '因此单独报告max(Δλcosφ,Δφ)/min(...)。',
        '热力图前后保持同一批三维方向、同一有效域及色标；后图按主ERP位置排列，只替换这些方向在新网格中的κ。'
        '辅助Yang图在Yang原生坐标中展示对应区域，native采样点不同，不用于前后统计。',
        'ERP精确极点理论发散，当前229.1838是360×720中心采样格点的最大值。色标32以上显示封顶，统计未截断。','',
        '这是参数化质量改善，不是深度误差降低比例；原生共同有效角点的支持缺失仍按此前验证结论保留。','',
        '![同一区域对比](anisotropy_comparison.png)','',
        '![Yang原生坐标中的对应位置](high_region_in_yang.png)']
    (out/'REPORT_ANISOTROPY.md').write_text('\n'.join(lines)+'\n')
    assert args.selection.read_bytes()==frozen
    print('DONE',out)


if __name__=='__main__':main()
