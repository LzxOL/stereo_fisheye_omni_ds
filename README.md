# OmniDS Dual-Fisheye Stereo — Yin–Yang + DA-V2 (`yin-yang-DA-V2-11-23`)

本分支保存当前最新一代 **双鱼眼 Yin–Yang + DA-V2 立体匹配网络方案（`11-23` 版本）**，在上一版本 [`yin-yang-DA-V2-11-19`](https://github.com/LzxOL/stereo_fisheye_omni_ds/tree/yin-yang-DA-V2-11-19) 的基础上，完成了 **Yin–Yang V2 架构重构、面向鱼眼球面的 3D 几何感知损失（Loss 3）**，以及 **DA-V2 单目先验融合改进（方案一 + 方案二）**。

---

## 与上一版本（`yin-yang-DA-V2-11-19`）的核心差异清单

### 1. 架构与算法对比总表

| 模块 / 维度 | 上一版本 (`yin-yang-DA-V2-11-19`) | 本版本 (`yin-yang-DA-V2-11-23`) | 核心代码位置与开关 |
| :--- | :--- | :--- | :--- |
| **输入图像 & 真值** | 灰度单通道；原生鱼眼点云投影 GT（存在稀疏空洞） | **3 通道 RGB 输入** + **三角化 Mesh GT (`MeshGT_v1`)** 完整致密监督 | `dataset_stereo.py`, `utils/image.py`<br>`--use_rgb --mesh_gt_root ...` |
| **匹配前端特征** | 包含 `GeometryAwareFeatureEnhancer` (`GeometryRefiner`)；纯深层空洞卷积特征 | **移除过平滑且高显存的 `GeometryRefiner`**；新增 **浅层高频门控跳跃融合 (`Shallow Skip`, RF $13\times13$)** | `module_cnn_only/featurelayer.py`, `network.py`<br>`--no_geometry_refiner --use_shallow_skip` |
| **双网格回投 (`1A`)** | 标准双线性回投 (`bilinear`)，物体深度断崖处存在跨边缘混合与飞点 | **边缘感知 Bilateral 回投 (`bilateral`)**：平滑区保持双线性，深度突变处按 $|\Delta d|/\tau$ 切断跨边缘插值 | `module_cnn_only/yinyang_training.py`<br>`--yinyang_reproj_mode bilateral --yinyang_bilateral_tau 2.0` |
| **重叠区融合 (`1B`+`1C`)** | 仅靠静态采样质量 $Q$ 软加权 (`q_soft`)；仅监督融合输出 | **置信度 + 分歧感知重叠选路 (`confidence_disagree`)**：结合局部极差 `spread`、GRU `confidence_head` 与分歧温度锐化；增加 **Yin/Yang 单路辅助监督** | `module_cnn_only/yinyang_training.py`, `network.py`, `train.py`<br>`--yinyang_overlap_mode confidence_disagree --yinyang_aux_weight 0.25` |
| **几何感知损失 (`Loss 3`)** | 仅使用逐像素 `Smooth L1` 序列损失（忽略鱼眼发散射线与 3D 曲面几何） | 新增 **`FisheyeGeometryLossPack`**：无倒数爆炸的 **球面 3D 表面法向一致性损失（末 4 轮 GRU）+ GT 深度边缘加权**（同时保留 Loss 1/2 消融开关与梯度累积） | `module_cnn_only/fisheye_losses.py`, `train.py`<br>`--fisheye_normal_weight 0.5 --fisheye_edge_alpha 1.0` |
| **DA-V2 输入形态（方案一）** | 将高畸变原始鱼眼图像 (`518×546`) 直接送入冻结 `DINOv2 ViT-B`，再回采到 Yin/Yang（边缘畸变严重且 Yang 视图侧躺 $90^\circ$） | **Yin–Yang 解畸变 + Yang 顺时针 $90^\circ$ 重力对齐输入 (`yinyang_rot3`)**：在 GPU 上采用有效掩码归一化插值（杜绝 FoV 外 `2.0` 哨兵污染）生成低畸变 `Yin (336×532)` 与重力对齐 `Yang_rot3 (868×336)` 分别提取 Token，再精确逆旋转（误差 `0.0`）回投对齐原生网格 | `module_cnn_only/dav2_geometry.py`, `network.py`, `train.py`<br>`--dav2_input_mode yinyang_rot3 --dav2_chart_input_size 336` |
| **DA-V2 先验融合（方案二）** | 空间无差别的全局标量残差融合：$F_{\text{cnn}} + \alpha \cdot \phi([F_{\text{cnn}}, F_{\text{da}}])$（双目高置信区易受粗糙单目先验干扰） | **GEV 匹配熵 ($H_{\text{gev}}$) 门控融合 (`EntropyGatedGeometryContextFusion`)**：从融合 GEV 代价体提取 `detach` 的 3 通道不确定度 $u_{\text{gev}}=[H_1, H_z, H_4]$，通过单调冷启动门控 $g(u_{\text{gev}}) \in (0,1)$ 在双目笃定区（低熵）抑制单目噪声、在弱纹理/遮挡区（高熵）放开单目语义补全 | `module_cnn_only/dav2_geometry.py`, `network.py`, `train.py`<br>`--dav2_entropy_gate` |

---

### 2. 文件级变更明细（相对 `yin-yang-DA-V2-11-19`）

* **新增文件**：
  * `module_cnn_only/fisheye_losses.py`：双鱼眼非针孔几何感知损失包（主力优选 `Loss 3`：无倒数爆炸球面 3D 法向一致性 + 深度边缘加权；含 `Loss 1/2` 消融接口）。
  * `train_yinyang_v3_dav2.sh`：`11-23` 旗舰版启动脚本（支持 `STAGE=S1 / S12 / FULL` 一键切换与梯度累积）。
  * `train_yinyang_v2_rgb.sh`：`11-21` Yin–Yang V2 渐进消融启动脚本（`STAGE=A/B/C/D`）。
* **核心修改文件**：
  * `module_cnn_only/dav2_geometry.py`：新增 `prepare_dav2_raw_tensor`、`chart_vit_size`、`unwarp_chart_rgb`、`DAV2GeometryContext.extract_chart_geometry`（方案一）以及 `compute_gev_uncertainty`、`EntropyGatedGeometryContextFusion`（方案二）。
  * `module_cnn_only/network.py`：接入 `use_shallow_skip`、`no_geometry_refiner`、`confidence_head`、方案一 (`dav2_input_mode='yinyang_rot3'`) 与方案二 (`dav2_entropy_gate=True`)；默认开关保持对历史 `11-19` / `11-21` 权重的 `strict=True` 向后兼容。
  * `module_cnn_only/yinyang_training.py`：实现 Bilateral 边缘感知回投、局部深度极差 `spread` 与置信度/分歧自适应重叠融合。
  * `module_cnn_only/featurelayer.py`：新增浅层高频特征门控残差跳跃融合 (`shallow_proj`, `shallow_gate`)。
  * `dataset_stereo.py` & `utils/image.py`：新增 `MeshGT_v1` 加载接口与 RGB 光度增强支持。
  * `train.py`：新增方案一/方案二 CLI 参数、几何损失 CLI 参数、`--gradient_accumulation`、`--stop_after_epoch` 与完整的 checkpoint restore 架构一致性校验。

---

### 3. 已完成的定量实验与单元测试验证

1. **Yin–Yang V2（30 Epochs 全量从零训练，RGB + `MeshGT_v1`, `bs=4`）**：
   * 单网格 DA-V2 Baseline (`e29.pth`)：`>1: 14.761% | >3: 4.761% | >5: 2.751% | MAE: 0.806 | RMS: 2.549`
   * **Yin–Yang V2 (`e29.pth`)**：**`>1: 13.377% (-1.384 pp) | >3: 4.593% (-0.168 pp) | >5: 2.714% (-0.037 pp) | MAE: 0.760 (-0.046) | RMS: 2.510 (-0.039)`**（5/5 项全胜）。
2. **优选几何损失 `Loss 3`（10 Epochs 筛选，基于 Yin–Yang V2, `bs=2 accum=2`）**：
   * Yin–Yang V2 Baseline (Ep 10)：`>1: 17.86% | >3: 6.13% | >5: 3.42% | MAE: 0.928 | RMS: 2.708`
   * **`+ Loss 3 (3D Normal + Edge, Ep 10)`**：**`>1: 17.593% | >3: 5.906% | >5: 3.317% | MAE: 0.912 | RMS: 2.672`**（Ep 9/10 连续 5/5 项全胜）。
3. **方案一 & 方案二（零训练 CPU 探针 + 代码级单元/GPU 端到端冒烟验证）**：
   * **方案一探针（24 帧, `1.14M` 像素）**：相对原始 518 鱼眼输入，解畸变 + `Yang_rot3` 使 DA-V2 相关系数提升（`Yin: 0.632→0.703`, `Yang: 0.622→0.671`），GT-WLS `MAE` 下降 **`17.3%`**（`7.288→6.030`），外圈 (`θ≥75°`) `>3` 下降 **`8.12 pp`**（`72.27%→64.15%`）。
   * **方案二探针**：GEV 匹配熵 $H_{\text{gev}}$ 将双目误差拉开 **`10.3×`（`MAE: 0.151 vs 1.556`）与 `25×`（`>5: 0.24% vs 6.00%`）**；低熵 `Q1` 区 DA-V2 胜率仅 `0.40%`，而在双目 `>5` 崩溃像素上零样本 DA-V2 胜率高达 **`44.59%`**。
   * **实现验证**：`unwarp_chart_rgb` 零 FoV 哨兵越界、`rot90` 往返误差 `0.0`、旧版 `e29.pth` `strict=True` 兼容加载；GPU 端到端 AMP 冒烟测试（`STAGE=FULL`, `bs=2, accum=2`）可训练参数 `1,795,568`，峰值显存 **`15,956 MiB`**，零 NaN，梯度与验证链路全通。

---

## 快速启动（`11-23` 旗舰版训练）

```bash
# 完整旗舰版：Yin-Yang V2 + 方案一(yinyang_rot3) + 方案二(GEV熵门控) + Loss 3(3D法向+边缘)
STAGE=FULL GPU_ID=0 BATCH_SIZE=2 GRAD_ACCUM=2 ./train_yinyang_v3_dav2.sh

# 仅消融方案一 + 方案二（不含 Loss 3）
STAGE=S12 GPU_ID=0 BATCH_SIZE=2 GRAD_ACCUM=2 ./train_yinyang_v3_dav2.sh

# 10-Epoch 快速筛选（保持 30-Epoch OneCycleLR 曲线）
STAGE=FULL GPU_ID=0 BATCH_SIZE=2 GRAD_ACCUM=2 STOP_AFTER_EPOCH=10 ./train_yinyang_v3_dav2.sh
```

---

## Original OmniDS Preparation & Acknowledgements

> Original project page: https://parkchaesong.github.io/omnids/

### Installation
```bash
conda create -n omnids python=3.9
conda activate omnids
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Datasets
Download OmniThings / OmniHouse / Sunny from [RVLab OmniStereo](https://rvlab.snu.ac.kr/research/omnistereo).

### Acknowledgements
This project builds upon [OmniDS](https://github.com/parkchaesong/OmniDS), [OmniMVS](https://github.com/hyu-cvlab/omnimvs-pytorch), [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), and [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2).
