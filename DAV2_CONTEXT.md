# DA-V2 Geometry Context：第一版 Context / ConvGRU 融合

本版在当前未提交的 DA-V2 实现上移动信息入口，保留已实现的骨干、Adapter、RGB 预处理和左鱼眼对齐。
**DA-V2 不再参与 GEV Guide。** `module_cnn_only/similarity_context.py` 已恢复至 CNN-only 基线版本。
开关仍为 `--use_dav2_geometry`，默认关闭；冻结基线分支 `CNN_Only-11-11` 未修改。

## 当前数据流

```text
真实左 RGB → 官方 Resize/Normalize → Frozen DA-V2 ViT-B
               → blocks [2,5,8,11] → 四个 768→32 投影
               → Concat128 → Conv3x3(128→64)+GN+ReLU → Conv1x1(64→32)
               → 左参考 Grid 对齐 → Residual Conv → Geometry Context32
                                                          ↓
左 CNN → 左 Grid Sampling → CNN Context32 ────────────→ 残差融合
                                                          ↓
                                                  Enhanced Context32
                                                ↙                    ↘
                                   context_proj+ReLU32          state_proj+Tanh64
                                         ↓                         Hidden初始化
       Similarity Lookup36、GEV Lookup18、当前索引1 → 原 MotionEncoder128   ↓
                                         ↓                            ↓
                              Context32 + Motion128 → 原 ConvGRU → delta_idx
```

当前 MotionEncoder 原本就接收 Context（代码中的旧变量名 `dino_feat`），同时 Context
也直接与 Motion Feature 拼接输入 ConvGRU。本次沿用两条连接，不改编码器或 GRU 公式。
Hidden 只初始化一次；默认静态 Context 在各轮复用，Similarity/GEV 仍按当前深度动态查询。
动态Context代码已删除；当前CNN与DA-V2融合结果在所有GRU轮次中固定，Hidden持续递归。

## 融合与初始化

`GeometryContextFusion` 沿用上一版的小卷积残差模块：

```text
phi = Conv1x1(64→32) + GroupNorm + ReLU + Conv1x1(32→32)
F_enhanced = F_CNN + alpha * phi(Concat(F_CNN, F_DA))
```

`alpha` 是可学习标量，初始化为 0.1；最后一层卷积权重 std=0.001、bias=0。
二者均非零，初始扰动小，同时不会阻断第一步 Adapter 梯度。
同一份融合特征同时输入 `context_proj` 和 `state_proj`，保持原初始化路径一致。
输出再按左侧有效性 Mask 清零。没有新增 Attention、可学习偏移或 DA-V2 深度预测。

## 原有几何与匹配保持不变

- 原始左右输入仍为 768×800；CNN 可继续用原灰度方式，不强制改成 RGB。
- DA-V2 使用 `StereoDataset` 已返回的真实 RGB；训练时与 CNN 使用同一份光度增强结果。
- 使用官方 DA-V2 Base `pretrained.*` 权重严格加载 ViT-B，骨干固定 FP32、eval、no_grad。
  不实例化 DPT 深度头。Adapter、Fusion 和原 Stereo 参数正常参与最终 Sequence Loss。
- 原 CNN Context 由左 Grid 采样；Geometry Context 使用同一射线位置。
- GWC 继续仅由左右 CNN 球面体构建，GEV Guide 继续为原 `ref.mean(dim=-1)`。
  3D U-Net、GEV 金字塔、Lookup、96 层索引及凸上采样不变。
- 输出是左鱼眼参考网格上的连续径向逆深度候选索引 `[0,95]`，不是 ERP，也不是 DA-V2 相对深度。

ViT 对齐不能直接套用 CNN Grid。CNN stride=2 特征的原图像素为：

```text
u = (grid_x + 1) * (W_CNN - 1)
u_resize = (u + 0.5) * W_ViT / W_image - 0.5
u_token  = (u_resize + 0.5) / 14 - 0.5
```

按 token 中心生成 `align_corners=False` 坐标后采样。Patch 边缘仅做边界特征延拓，
原始 FOV/invalid-mask 仍决定有效性。不会把视场外坐标裁到边缘并当成有效。

## 正式尺寸的关键 Tensor

| 节点 | B=1 的 Shape |
|---|---|
| DA-V2 预处理 RGB | `[1,3,518,546]` |
| 四层 Patch Token | 每层 `[1,1443,768]` |
| 四层 Patch Map | 每层 `[1,768,37,39]` |
| 各层独立投影 | 每层 `[1,32,37,39]` |
| 融合 Patch Feature | `[1,32,37,39]` |
| Geometry / Enhanced Context | `[1,32,160,240]` |
| Context / Hidden | `[1,32,160,240]` / `[1,64,160,240]` |
| GWC / GEV | `[1,8,96,160,240]` / `[1,1,96,160,240]` |
| Motion Feature / GRU Input | `[1,128,160,240]` / `[1,160,160,240]` |
| 每轮输出 | `[1,1,320,480]` |

## 实际验证（2026-11-12）

小尺寸 D/H/W=16/8/16 与正式尺寸 96/160/240 均使用真实 OmniThings 标定、RGB 和左光心 GT，
完成3次 AMP 参数更新、12轮 GRU、2帧 held-out 验证及严格 checkpoint 重载。

- ViT 全部权重哈希前后相同，无骨干梯度；四个投影头、融合卷积、局部细化、alpha、CNN、GEV、GRU 均收到有限非零梯度。
- 正式尺寸 alpha 三次更新约为 `0.100000 → 0.100499 → 0.100930 → 0.101364`。
- Fusion 输出与两个投影头的输入逐位一致。每次 Forward 的 ViT/Adapter、Fusion、Context Head、Hidden Head 各执行一次，MotionEncoder/GRU 执行12次。
- 相同初始 CNN 参数与输入下，启用前后的 Similarity、GWC、CNN Guide、GEV 指纹完全一致，证明没有改变匹配证据的初始计算。
- 固定 CNN 输入，只扰动辅助 RGB 的依赖检查通过：Context 会变化，Similarity/GWC/CNN Guide/GEV 完全不变；增强 Context 确实进入原 MotionEncoder 和 ConvGRU 的 Context 通道。该扰动不是训练增强或精度样本。
- 左侧及匹配候选全部无效时输出有限且为零，未由融合偏置产生有效深度。
- 官方 RGB 预处理逐位一致；坐标 ramp 检验最大误差0.000183像素，错误复用 CNN Grid 则约1.446像素。
- 开关关闭：原权重严格加载、初始参数、CPU前向与梯度、GPU前向逐位一致；不导入 DA-V2。
- 原 `train.py` 入口真实增强RGB训练4批、实际更新3次、验证2帧并保存成功；恢复后 global_step=5、optimizer_updates=4，权重/optimizer/scheduler/scaler均恢复。该入口 checkpoint 的全部 ViT 权重仍与官方权重逐位相同。
- 原有实际 CNN-only checkpoint 严格加载通过；上一版 Guide checkpoint 被明确拒绝。关闭开关的四目路径也通过原 Smoke Test。

正式尺寸、Batch=1 的短测（RTX5090，共享GPU，包含梯度检查）：

| 指标 | CNN-only | DA-V2 Context |
|---|---:|---:|
| 可训练参数 | 1,577,432 | 1,773,529 |
| 冻结参数 | 0 | 86,580,480 |
| 峰值 allocated | 4714.50 MiB | 5125.22 MiB |
| 去掉首步后的2步均值 | 0.687 s/step | 1.116 s/step |
| 3次更新后，两帧验证 MAE（以D归一化的百分数） | 14.88645 | 14.88653 |

新增峰值约410.72 MiB；RGB预处理另约0.014 s/step。共享GPU、样本少且存在warmup差异，
这些时间仅为Smoke记录，不是稳定吞吐基准。验证 MAE 转候选索引需乘 `96/100`，
不能解释为米。这里只证明可训练，**没有证明优于 CNN-only，也不能与已训练12轮的权重直接比较。**

完整数值和后续训练入口验证见 `DAV2_CONTEXT_VALIDATION.json`。

## 运行方式

在已有服务器环境的项目根目录运行；官方模型代码和权重仍位于 `/cache/models/`。
缺少时通过 `--dav2_model_repo`、`--dav2_checkpoint` 指定，关闭开关无需这些依赖。

```bash
# 正式尺寸 Smoke Test，同时测试关闭开关的旧代码回归。
CUDA_VISIBLE_DEVICES=3 python tools/smoke_dav2_geometry.py \
  --db_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset \
  --baseline_repo /workspace/stereo_fisheye_omni_ds --full_size \
  --output /tmp/dav2_context_smoke.json

# 直接通过原训练入口验证真实增强RGB、GT、保存和验证。
CUDA_VISIBLE_DEVICES=3 python train.py --stereo --reference_grid left_fisheye \
  --use_dav2_geometry --ref_cam cam4 --target_cam cam3 \
  --db_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset \
  --mixed_precision --batch_size 1 --total_epochs 100 --train_iters 12 \
  --max_train_steps 4 --max_eval_samples 2 --name DAV2_Context_Smoke

# 后续正式训练示例：不含Smoke限制，独立实验名。
CUDA_VISIBLE_DEVICES=2 python train.py --stereo --reference_grid left_fisheye \
  --use_dav2_geometry --ref_cam cam4 --target_cam cam3 \
  --db_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset \
  --mixed_precision --batch_size 1 --total_epochs 100 --train_iters 12 \
  --lr 0.0005 --wdecay 0.00001 --name OmniDS_DAV2_Context
```

Batch=1 是当前入口默认的保守起点；未实测此新模式的 Batch=6。正式精度对照需要统一有效
Batch、优化器更新预算、数据划分和评估区域。原 CNN-only 及之前 DA-V2 Guide 短测的结果均保留。
Guide 版 checkpoint 没有 Context 版的 alpha/入口标记，恢复会明确拒绝，不静默混用。

## 基于原 `test.py` 的双目评估与组会可视化

评估直接在原 `test.py` 增加 `--stereo` 分支，保留四目评估路径，没有另一套评估入口。
原脚本使用四目 `Dataset`、不传双目 Mask / DA-V2 RGB，且 `StereoDataset.makeVisImage`
明确不支持原全景可视化，因此不能直接拿原四目命令评估此 checkpoint。

新增分支沿用 `train.py` 中的 `StereoDataset.loadSample → 原 forward → data.evalError`，
严格恢复 checkpoint 的网络参数、相机对、标定、候选轴、GT 几何、原验证划分和迭代次数，
关闭数据增强。DA-V2 使用真实 RGB 和原预处理。权重严格加载，不忽略不匹配项。

```bash
CUDA_VISIBLE_DEVICES=2 python -W ignore::FutureWarning test.py --stereo \
  --ckpt checkpoints/omnids_dav2_context_default_20261112/omnids_dav2_context_default_20261112_e10.pth \
  --db_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset \
  --output /cache/evaluation/dav2_context_epoch11_validation512 \
  --num_visuals 12 \
  --training_record /cache/training/omnids_dav2_context_default_20261112
```

首次检查可增加 `--max_samples 2 --num_visuals 2`，并使用不同的输出目录。
输出非空时明确拒绝覆盖；正式评估默认运行 checkpoint 固定的 512 张验证图，官方 1024 张测试集不动。
`e10` 是第 11 轮，不能把这次验证称为独立测试集结果。

- `summary.json`、`per_frame_metrics.csv`：逐图指标、原指标重算结果和几何/权重来源。
- `arrays/*.npz`：选定帧的原始浮点预测索引、逆深度、GT、有效性和输入图。
- `overview_*.png/pdf`、`figures/frame_*.png/pdf`：RGB / GT / 预测 / 索引误差。
- `validation_history.csv`、`training_loss_blocks.csv`、`training_summary.png/pdf`：保存的各轮验证指标与训练趋势。
- `*.layout.json`：图中文字越界检查结果。

12 张可视化在推理前按验证列表位置均匀选取，不依据效果挑选。预测是左鱼眼参考网格上的双目结果，
不是 ERP，也不是 DA-V2 单目深度。GT 是原 Rig GT 按已有标定/Z-buffer 重投影得到，未缩放替代、未填洞。
预测仅展示左右候选的有效支持区域；GT/误差仅显示有效监督区域。灰色代表无效或没有 GT，不能解读为误差零。
GT 与预测使用所有帧共享的固定逆深度色标；误差图默认将显示上限设为 10 个索引，数值指标不做截断。
候选 0 近似无穷远，因此评估保留原逆深度索引指标，不将其盲目取倒数后汇报为米制深度误差。

按本次指令，新的正式训练应等待当前左鱼眼 CNN-only 训练完成12轮并保存、退出后再启动；
验证期间没有停止或重启当前训练。新模型不会覆盖既有 CNN-only checkpoint。

## Yin-Yang增量接入（2026-11-16）

`network.py::OmniDS`已把Yin/Yang匹配前端接到本文原主干之前。
ViT和Adapter的Patch融合每次Forward只做一次，两网格分别使用真实左Grid对齐；
同一GeometryContextFusion、context_proj、state_proj、原MotionEncoder与UpdateBlock共用。
匹配分支仍为Similarity36 + GEV18，没有把144通道独立实验更新器作为当前训练主干。
原单网格前向和梯度已逐项回归一致；详见YINYANG_REFINEMENT.md及DAV2_YINYANG_VALIDATION.json。
