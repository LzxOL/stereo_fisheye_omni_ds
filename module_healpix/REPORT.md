# HEALPix Direct GWC 修改与验证

HEALPix 球面扫描后，原始左右特征直接进入 GWC 和现有 Similarity Profile 计算。匹配路径已移除 Geometry-aware Feature Refiner 及其 XYZ Fourier Encoding、Geometry MLP、局部残差卷积。左光心射线、OCam 标定、PAD-XY 索引、深度候选和几何 Mask 均沿用原实现。

## 改动文件

| 文件 | 变更 |
|---|---|
| [`module_healpix/network.py`](network.py) | 移除匹配 Refiner，原始分块扫描特征直接构建 GWC 和 Similarity Profile。旧检查点通过 `load_pretrained` 转移共享权重并记录被丢弃的 `enhancer.*`。 |
| [`module_healpix/ops.py`](ops.py) | 移除仅匹配 Refiner 使用的 `refine_chunk`。GEV、跨面卷积和更新器算子原样保留。 |
| [`module_healpix/validate.py`](validate.py) | 新增直接匹配的形状、归一化、NaN/Inf 输入与梯度检查，以及 Face-aware GEV 跨面脉冲对照。 |

沿用 32 通道、8 组 GWC 和 96 层径向逆深度候选。每组 4 通道，分数为 `sum(left × right) / sqrt(4)`。特征乘法前按 `stereo_valid` 清零，GWC 和 Similarity Profile 输出也按原候选 Mask 清零。Similarity Profile 保持 32 通道点积除以 `sqrt(32)`。GEV Guide 改用原始左特征在 D 上的零掩码均值，分母仍为 96。Face-aware GEV、Similarity Lookup、ConvGRU、上采样和左鱼眼回投沿用原实现。Yin–Yang、CNN Encoder 与可选 DA-V2 Context 分支均未修改。

## 关键张量

实际运行 `B=1, Nside=64, D=96`，扫描按 8 个深度候选分块。共享阶段的张量形状与原 Nside 64 验证记录逐项一致。

| 阶段 | 形状 |
|---|---|
| 左／右 CNN 特征 | `[1,32,384,400]` 各一份 |
| 左／右扫描特征，从每个深度块计算 | `[1,12,32,8,64,64]` 各一份 |
| Boolean 候选 Mask | `[1,12,1,96,64,64]` |
| GWC → Face-aware GEV | `[1,12,8,96,64,64]` |
| Similarity Profile / GEV Guide | `[1,12,96,64,64]` / `[1,12,32,64,64]` |
| GEV Stem → Down1 → Down2 | `[1,12,16,96,32,32]` → `[1,12,32,48,16,16]` → `[1,12,48,24,8,8]` |
| GEV 两级 Lookup 体 | `[1,12,1,96,64,64]` 和 `[1,12,1,48,64,64]` |
| ConvGRU Hidden / 左鱼眼输出 | `[1,12,64,64,64]` / `[1,1,320,480]` |

## 实测结果

一批真实 OmniThings cam4/cam3 帧 `04887`，预训练权重 `e28`，RTX 5090、FP32。使用 CNN-only Context、默认 **12 轮**更新和激活重计算完成单批前向与反向。输入原图为鱼眼 RGB，按 `e28` 的单通道预处理送入 CNN，实际输入为 `[1,1,768,800]`。本批覆盖该检查点的输入模式。

- **数值对照通过。** HEALPix 原 96 层轴、左／右采样网格和 Mask 与 Stage 1 一致。Nside 8 的扫描特征、GWC、原始 Similarity 对照最大绝对差均为 **0**。全 1 特征的单组 GWC 为 **2.0**，符合 `4 / sqrt(4)`。
- **跨面聚合通过。** Face-aware GEV 的 3D 卷积从相邻面读取同深度脉冲，目标响应 **1.0**。逐面零填充对照为 **0**，其他深度无响应。将来源标为无效并注入 NaN 后响应为 **0**。原 HEALPix 父子采样和跨面 GRU 对照也通过。
- **掩码和梯度通过。** 左右无效特征分别注入 NaN 和 Inf，前向与清零对照一致，反向梯度有限且无效位置梯度为零。真实批次有 **1,877,152 / 4,718,592** 个有效扫描候选。GWC、Similarity Profile、Guide、GEV 中间层和最终预测的有限值及无效位置清零检查通过。CNN、Similarity 聚合、GEV、Context、GRU 和预测头的 **14 个抽查权重**均有有限且非零的梯度。
- **真实输出通过。** 左鱼眼输出 **61,222** 个有效像素，有效输出满足 cam4/cam3 上下候选几何约束，径向深度与逆深度互为倒数。该批次前向约 **518 ms**、反向约 **1,736 ms**，CUDA 已分配峰值约 **4,568 MiB**。严格迁移 **282** 个共享状态条目，跳过旧分支条目，其中 **9** 个属于匹配 Refiner。

此次使用验证损失检查梯度，未执行优化器更新或覆盖检查点。有效像素数是运行时覆盖率，深度精度仍需独立评估。


## 复现验证

从仓库根目录运行。使用 OmniThings 的数据目录、已有 Yin–Yang e28 权重，以及与 Torch 版本匹配的 Earth2Grid 和 healpy。以下路径由运行环境提供。

```bash
selection=experiments/local_spherical_grid/selected_grid_yin50.json
python experiments/healpix_probe/run_probe.py \
  --data "$OMNITHINGS" --checkpoint "$CHECKPOINT" --selection "$selection" \
  --out "$STAGE1_GRIDS" --nsides 8 64 --save-grids --repeats 1 --device cuda:0
python -m module_healpix.validate \
  --data "$OMNITHINGS" --checkpoint "$CHECKPOINT" --selection "$selection" \
  --grids "$STAGE1_GRIDS" --out "$VALIDATION_OUT" --device cuda:0 --backward
```
