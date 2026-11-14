# cam4 + cam3：左光心球面覆盖与局部补偿网格

这是独立的 **OCam 几何 + RGB 可视化实验**。只新增本目录，没有改动 Stereo 网络、数据集或训练入口；不运行模型、不读取 GT、不评估深度精度。

Git 中保存实验源码和 `selected_grid.json`（当前 cam4/cam3 搜索结果的原样冻结副本）。下述 `outputs*` 中的数据快照、NPZ、HTML、图片和验证报告为本地生成物，不随代码提交。新克隆仓库需要将 `--data-root` 指向实际 OmniThings 目录，再按下文命令生成；`selected_grid.json` 可作为验证脚本的 `--selection` / `--selected-json` 输入，但不能代替相机 Mask、RGB 或 GT 数据。后续 GT 诊断脚本只离线读取 GT，不参与网格搜索。

## 直接查看本次实际结果

输出位于 `outputs/`：

- `camera_rig_3d.html`：真实相机位置、光轴、基线和 Rig 中心。
- `angular_coverage.html`：默认显示只含旋转的远场方向覆盖。
- `finite_radius_coverage.html`：默认显示 2 m 有限半径球面，可切换 0.5 / 1 / 2 / 5 / 10 m 和 Angular。
- `angular/`、`finite_2m/` 等：带坐标的 RGB、Mask、畸变 PNG；`*_raw.png` 为无坐标原始展开图。
- `coverage_vs_radius.png`、`complementary_statistics.png`：球面面积覆盖率和参数化畸变对照，另有 PDF/SVG。
- `geometry_stats.json`、两个 CSV、`REPORT.md`：标定、验证误差、逐半径统计和结果解释。
- 每种模式的 `projection_data.npz`：真实 cam4 射线、左右像素投影、有效 Mask、精确球面面积、两套纬度和 kappa，可直接复查。
- `input_snapshot/`：本次同帧两张原始图、原始 Mask、YAML 和 OCam 文本的来源副本；不修改原始数据。

HTML 自带 Plotly，不依赖 CDN、网络或额外图片路径，下载后直接用 Chrome/Edge 打开。支持鼠标旋转、缩放，六种区域选择，RGB / Coverage / 主网格畸变 / 补偿网格畸变 / 局部择优，以及网格和相机显示开关。HTML 只降低显示采样密度，统计仍使用完整分析网格。

## 运行命令

在项目已有 Python 环境运行，不需要额外模型或权重。复用的 OCam 工具间接依赖项目的 PyTorch，但实验不构建网络、只使用 CPU；另外使用 NumPy、SciPy、OpenCV、Pillow、PyYAML、Matplotlib、Plotly。

服务器容器内：

```bash
cd /workspace/stereo_fisheye_omni_ds
/opt/pdf-omni-venv/bin/python experiments/local_spherical_grid/visualize_stereo_sphere.py \
  --data-root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset/omnithings \
  --frame-id 4887 --ref-cam 4 --target-cam 3 \
  --sphere-height 360 --sphere-width 720 \
  --radii 0.5 1 2 5 10 \
  --display-stride 3 \
  --complement-lon-span 240 --complement-lat-limit 45 \
  --output-dir /cache/local_spherical_grid_20261114
```

本地使用已有数据目录，或本次附带的真实单帧输入副本：

```bash
cd /Users/linzhaoxian/lzx-ws/project/OmniDS
python experiments/local_spherical_grid/visualize_stereo_sphere.py \
  --data-root experiments/local_spherical_grid/outputs/input_snapshot --frame-id 4887 \
  --sphere-height 360 --sphere-width 720 --radii 0.5 1 2 5 10 \
  --output-dir /tmp/omnithings_sphere_reproduce
```

本次先实际运行了 180×360 小规模验证，再运行 360×720 完整分析。输入原图始终为 768×800 RGB，没有 CNN 尺度、归一化或增广。0.5/1 m 是低于 config 扫描下限 1.65 m 的几何探针，不是训练候选。

## 代码阅读顺序

1. `camera_geometry.py::load_pair`：从现存 `config.yaml` 构建 `OcamModel`，只读取现成鱼眼 Mask 和同帧 RGB，不创建/修改数据文件。`ocam*.txt` 仅记录来源；实际系数以当前工程使用的 YAML 为准。
2. `pair_geometry`：复用 `utils.geometry` 的 Rodrigues 外参，计算 `T_target_from_reference = T_target_from_rig @ T_rig_from_reference`。
3. `spherical_rays`：生成完整参考球面用于覆盖分析；这不是要替换当前网络的左鱼眼参考网格。每个球面格点一条真实 cam4 射线。
4. `sphere_projection` / `sample_rgb`：使用工程 `OcamModel.rayToPixel` 与 `dataset_stereo.project_camera`，从原始图像双线性采样，显式清零无效颜色。
5. `complementary_coordinates`：建立水平极轴的旋转坐标系及有限矩形 Patch Mask。这里只变换同一条三维射线的坐标表示。
6. `coverage_statistics` / `validate_geometry`：球面面积加权、实际往返验证及 Angular / Finite 区别。
7. `visualize_stereo_sphere.py`：生成 PNG、离线 HTML、CSV、NPZ 和报告。

## 坐标、投影与有效性

`pose` 的前 3 个量是 Rodrigues 旋转向量，后 3 个是平移，含义为 Camera-to-Rig，不能当作 Euler 角或 Rig-to-Camera。

本次实际标定：cam4 的 Rig 位置为 `(-0.6,0,0)` m，cam3 为 `(0,0,-0.6)` m，光轴角 90°，基线 `sqrt(0.6²+0.6²)=0.8485281374` m。cam3 光心在 cam4 系为 `(-0.6,0,-0.6)`，光轴为 `(-1,0,0)`；cam4 光轴为 `(0,0,1)`。这对应 Rig 的左向与后向相机。

OCam 采用 `+X` 图像右、`+Y` 图像下、`+Z` 沿光轴。为了使纬度正方向向上，显示坐标通过 `R_ocam_from_vis=diag(-1,-1,+1)` 显式旋转 180°：显示 `+X` 图像左、`+Y` 向上、`+Z` 沿左光轴，因此正经度朝图像左。所有实际相机投影仍在真实 OCam 系完成，没有对投影系数或三维点做反射。

像素 `p=(u,v)` 从 0 开始，u 为列、v 为行；YAML `center=[383,399]` 按工程约定是行、列，因此光轴像素是 `(u=399,v=383)`。本实验直接采样原图，不使用 CNN 特征图的 stride=2 坐标换算。

- **Angular**：`X4=r4`、`X3=R34 r4`。只表示远场方向覆盖，两个相机的实际光心没有被当作同一个物理光心。3D 图仍显示真实布局。
- **Finite**：`X4=d r4`、`X3=R34 X4+t34`。球面以 cam4 为中心，半径为真实米制距离。
- 有效性同时检查 OCam FOV、原始图像边界及双线性四邻点的 invalid mask。越界点不会被裁到边界后冒充有效点。
- 左右同时有效得到 overlap。**这是几何投影有效，不是物体遮挡可见**。虚拟球面不是场景真实表面，因此 RGB 不必在两视图一致，不能据此判断深度正确性。

## 面积与补偿统计

主 ERP 的像素取经纬度中心，避免极点重复。每格立体角为

`dΩ = dλ * [sin(φ_top) - sin(φ_bottom)]`，总和严格为 `4π`。

这是 `cos(latitude)` 面积加权的精确格点形式，不使用 ERP 像素数量代表球面面积。

- `left_fraction/right_fraction/overlap_fraction/left_only_fraction/right_only_fraction` 的分母都是整个球面。
- `overlap_of_left` 的分母是左相机有效球面面积。
- `high_primary_of_overlap/high_best_of_overlap` 的分母是双目共视球面面积。
- `high_primary_covered_by_patch` 的分母是主网格高畸变共视面积。
- 平均值、p50、p95 均按球面面积加权；最大值依赖采样分辨率，不能把未采到极点当作极点畸变有限。

主极轴为显示竖直轴。补偿极轴取水平光轴角平分方向的正交轴；两套坐标系均为正交旋转，行列式为 +1。

`κP=1/max(|cos φP|,ε)`，`φC=asin(nCᵀr)`，`κC=1/max(|cos φC|,ε)`。

默认补偿 Patch 仅覆盖 **补偿坐标中的经度 ±120°、纬度 ±45°**，不是完整 360° Yin–Yang；矩形尺寸只是初步诊断选择，可由 CLI 调整。`best` 在 Patch 内取 `min(κP,κC)`，Patch 外仍为 `κP`。统计只使用双目 overlap，绝不将未覆盖方向算入改善。

当前 kappa 是单位球面经纬采样的理论各向异性，不包含某个未来矩形网格的具体像素间距，也不衡量三角测量强度、遮挡或网络精度。旋转能改善坐标畸变，不等于已经验证深度结果变好。

## 验证与参考

脚本实际检查 OCam Pixel→Ray→Pixel、多级外参变换一致性、左网格深度不变性、原图 RGB 双线性对照、无效位置清零、远半径趋近 Angular，以及补偿坐标重建同一条射线。OCam 拟合多项式存在很小的近似误差，报告实际误差而非声称精确互逆。

优先依据当前数据 `config.yaml` 和项目源码。用户提供的参考链接：

- [Omnidirectional Stereo Dataset](https://rvlab.snu.ac.kr/research/omnistereo)
- [官方 OmniMVS PyTorch](https://github.com/hyu-cvlab/omnimvs-pytorch)
- [OmniMVS 非官方 PyTorch 实现](https://github.com/matsuren/omnimvs_pytorch)

后续值得进一步研究的是**实际共视区域中主 ERP 的上下高纬度部分**，还应受每个候选深度的双目 Mask 和遮挡影响。本次没有构建深度融合、完整 Yin–Yang 或任何网络分支。

## 自动搜索 Yang（新增）

`search_yang_grid.py` 复用以上 OCam、外参、原图采样和精确球面面积统计。仍是纯几何实验，没有修改前面的固定补偿窗口程序，也没有接入网络。

本次的 **Yin / Yang 表示两套参数化网格**，不是 cam4 / cam3 两台相机：

- Yin 固定竖直极轴、纬度 ±60°；经度窗口自动包住所有候选下有效的低纬度共视方向。
- Yang 的极轴方向与矩形边界自动搜索，纬度绝对值限制在 45°以内。
- Yang 必须覆盖 Yin 未覆盖的全部有效方向，并覆盖 Yin 边缘内侧 5° 的有效接缝带。
- 每条边保留 2° 余量，边界向外量化到 0.5°。这些是本次实验约束，可通过 CLI 更改，不是已经确定的网络结构。

搜索首先在 90×180 球面、96 层候选上运行，再对候选旋转作细化；最终在 360×720、相同 96 层候选上重新拟合边界。完整网格上若存在任意未覆盖的有效深度候选，该配置会被拒绝。

优化目标是 **在覆盖、接缝和最大纬度约束下最小化 Yang 的经纬矩形角度面积 `Δλ×Δφ`**。它对应相同角步长下的像素预算；同成本时再比较球面面积和按深度/球面面积加权的 κ。重叠面积会单独记录，当前没有宣称其同时达到最小。

旋转定义为：

```text
R_visual_from_Yang = Ry(yaw) @ Rx(tilt) @ Ry(longitude_center)
```

前两个角决定极轴，第三个角由最短包围经度圆弧的中心求得，并吸收到最终基矩阵。绕极轴旋转和自由平移经度窗口是等价自由度，因此无需重复枚举；最终 Yang 经度边界以 0° 为中心。JSON 同时保存旋转角、3×3 基矩阵和真实 OCam 系中的基矩阵，以避免 Euler 角含义不明确。

默认候选轴严格沿用原 `Dataset.__initSweep` 的 EPS、1.65 m 下限及 192 层定义，再 `[::2]` 得到 96 层；**不是重新 linspace 96 个端点**。最近一层实际为 1.6586842105 m，第 0 层保留工程的远场极限定义。已经在真实完整数据目录上，与原 Dataset 的结果逐项比较通过。

运行命令：

```bash
cd /Users/linzhaoxian/lzx-ws/project/OmniDS
python experiments/local_spherical_grid/search_yang_grid.py \
  --data-root experiments/local_spherical_grid/outputs/input_snapshot \
  --frame-id 4887 --ref-cam 4 --target-cam 3 \
  --num-depth 96 --full-num-depth 192 \
  --search-height 90 --search-width 180 \
  --sphere-height 360 --sphere-width 720 \
  --yin-lat-limit 60 --seam-deg 5 --yang-max-lat 45 \
  --padding-deg 2 --boundary-step-deg 0.5 \
  --native-angular-step-deg 0.75 \
  --output-dir experiments/local_spherical_grid/outputs_yang_search
```

服务器使用相同参数，Python 为 `/opt/pdf-omni-venv/bin/python`，数据目录可使用完整真实数据目录，或 `/cache/local_spherical_grid_20261114/input_snapshot`；输出可放在 `/cache/yang_search_full_20261114`。

默认两套原生网格分别按约 0.75° 的相等经纬角步长决定 H/W。若直接给两个不同长宽比的窗口相同 H/W，则角步长比也会影响实际像素各向异性。可用 `--native-size H W` 导出固定尺寸；NPZ 中另外保存 `native_pixel_anisotropy`，不将它与单纯 `1/cos(latitude)` 的坐标指标混为一谈。这些尺寸只用于几何 Mask 导出，不修改现有网络的张量形状。

输出在 `outputs_yang_search/`：

- `yin_yang_search.html`：离线交互球面、主 ERP 和两套原生网格；可选择任意候选索引、Union 或 Intersection，显示各网格、重叠、未覆盖、RGB 和深度有效比例。
- `chart_windows.png`：不考虑相机有效性的纯网格窗口；`yin_yang_regions.png`：与真实共同有效区域求交后的网格归属。
- `yin_yang_overlap_masks.png`、`depth_validity_overview.png`、`native_depth_validity.png`：重叠及逐深度有效性对照。
- `reference_depth_masks.npz`：完整参考球面上 `[D,H,W]` 的右相机有效性、共同有效性；左相机 Mask 为 `[H,W]`，网格归属与网格重叠也是 `[H,W]`。
- `yin_depth_validity.npz`、`yang_depth_validity.npz`：各自原生网格上的 `left_valid=[H,W]`、`target_valid=[D,H,W]`、`valid_mask=[D,H,W]`。其中 `valid_mask=left_valid & target_valid`；同时保存真实射线、候选轴及基矩阵。
- `search_result.json`、`search_candidates.csv`、`full_resolution_candidates.csv`、`depth_coverage.csv`、`REPORT.md`：搜索参数、可行性、边界、逐候选球面面积统计和实际验证结果。

Union 表示至少有一个候选深度有效，Intersection 表示所有候选都有效；均不是场景深度。每层 Mask 仍然只是几何投影 / FOV / 原始 invalid_mask 有效性，没有进行 GT 遮挡推断。覆盖保证限于已检查的离散射线和候选轴；搜索结果也不是连续 SO(3) 上的全局最优证明。

## 冻结参数可靠性验证（新增）

结果在 `outputs_validation/REPORT_VALIDATION.md`、`validation_dashboard.html`；没有修改选定旋转、窗口、候选轴、原生尺寸或原训练/GT转换代码。

验证程序分别负责：

- `validate_grid_coverage.py`：360×720、720×1440、全部96候选的解析覆盖；原生最近邻/严格双线性/任一有效角点共同支持；2/4/6单元接缝上下文和实际采样预算。
- `validate_coordinates_rgb.py`：随机球面/真实鱼眼像素完整解析往返，原生RGB往返、同源重叠差异、多帧和原CNN坐标约定只读核查。
- `validate_gt_diagnostics.py`：读取真实浮点GT页，Rig→cam4径向距离范围、跨相机几何有效与Rig点云Z-buffer可见性估计。没有独立cam3/cam4深度真值，因此没有伪称完整可见性GT验证。
- `validate_parameterization.py`：理论κ、真实角步长的像素各向异性、有限相邻射线距离比及面积加权max/P95/P99/mean。
- `build_validation_report.py`：只汇总结果，生成新的报告和离线HTML，不运行搜索。

在当前服务器容器内从项目根目录运行：

```bash
/opt/pdf-omni-venv/bin/python experiments/local_spherical_grid/validate_grid_coverage.py \
  --selection /cache/yang_search_full_20261114/search_result.json \
  --native-dir /cache/yang_search_full_20261114 \
  --data-root /cache/local_spherical_grid_20261114/input_snapshot \
  --output-dir /cache/adaptive_grid_validation_20261114/coverage \
  --resolutions 360 720 720 1440

/opt/pdf-omni-venv/bin/python experiments/local_spherical_grid/validate_coordinates_rgb.py \
  --data-root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset/omnithings \
  --selected-json /cache/yang_search_full_20261114/search_result.json \
  --output-dir /cache/adaptive_grid_validation_20261114/coordinates_rgb \
  --frames 4887 4888 4889 --random-directions 100000 --random-pixels 12000

/opt/pdf-omni-venv/bin/python experiments/local_spherical_grid/validate_gt_diagnostics.py \
  --data-root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset/omnithings \
  --selection /cache/yang_search_full_20261114/search_result.json \
  --output-dir /cache/adaptive_grid_validation_20261114/gt \
  --frames 4740 4886 4887 4888 5000

/opt/pdf-omni-venv/bin/python experiments/local_spherical_grid/validate_parameterization.py \
  --selected-dir /cache/yang_search_full_20261114 \
  --output-dir /cache/adaptive_grid_validation_20261114/quality

/opt/pdf-omni-venv/bin/python experiments/local_spherical_grid/build_validation_report.py \
  --output-dir /cache/adaptive_grid_validation_20261114 \
  --selection /cache/yang_search_full_20261114/search_result.json \
  --frozen-html /cache/yang_search_full_20261114/yin_yang_search.html
```

本地运行使用项目原Python环境并替换上述数据/输出路径。覆盖、坐标、质量检查可使用已有快照；多帧RGB和GT诊断需要完整数据目录，不能将单帧快照伪当完整数据。

重要口径：几何窗口覆盖100%不等于原生共同有效角点支持100%，共同匹配支持不足也不等于左RGB本身投影有洞。2/4/6接缝分析保持物理窗口不变；文中n为每侧原生halo厚度，若n定义为双侧总宽，另有每侧1/2/3的对照。GT严格只用于离线诊断。

## 同一高纬区域的各向异性对比

`compare_high_latitude_anisotropy.py` 只读前面质量验证生成的 NPZ，在同一批主 ERP 高纬方向上比较原 ERP、阴阳理论参数化和实际像素角步长的各向异性，不重新搜索或改变冻结参数。

```bash
python experiments/local_spherical_grid/compare_high_latitude_anisotropy.py \
  --quality-dir experiments/local_spherical_grid/outputs_validation/quality \
  --selection experiments/local_spherical_grid/selected_grid.json \
  --latitude-threshold 60 \
  --output-dir experiments/local_spherical_grid/outputs_anisotropy_comparison
```

需要 NumPy、Matplotlib、Plotly 和可用中文字体。首先运行 `validate_parameterization.py` 生成 `reference_quality_maps.npz`、`yang_native_quality.npz`。输出 PNG/PDF/SVG、离线 HTML、CSV/JSON、NPZ 和 `REPORT_ANISOTROPY.md`；`--stats-only` 可仅计算统计。

当前 cam4/cam3 的主 ERP `|纬度|>60°` 几何共视区域，按球面面积乘有效候选数量加权：理论 κ 均值从 4.55413647 降到 1.03063577，P99 从 25.47133706 降到 1.13749755。该子区域均由 Yang 覆盖。此处只反映参数化质量，不代表深度误差改善；原生共同支持缺失仍按验证结论保留。
