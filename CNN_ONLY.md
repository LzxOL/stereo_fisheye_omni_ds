# OmniDS CNN-only Baseline

当前双目默认已进一步简化为 **GWC-only + Static Left Context**，开关和最新验证见文末。四目路径保持原有行为；下文此前阶段的动态 Context/Variance 描述是可恢复的消融路径。

本次是在当前工作树上完整复制 `module/` 得到 `module_cnn_only/`，再做最小改动。
原 `module/` 及其中原有的未提交阅读注释均保留，已逐文件核对 SHA256 未变化。
这仍然是**四目、Rig-centered ERP 深度预测**，没有引入之前 PDF-Omni 双目 Baseline 的结构。

## 代码改动

- `module_cnn_only/network.py`：保留原 CNN、Similarity/GEV、MotionEncoder、ConvGRU、DepthHead 和凸上采样；注释 DINO 特征提取、DINO ERP Attention、Context Fusion 的初始化与调用。DINO 类定义保留，Attention 的导入放到该未使用类内部。
- `module_cnn_only/similarity_context.py`、`network_student.py`：仅把内部 `from module.*` 改到副本包；算法不变。
- 其余复制文件：保持原实现。学生网络、DINO Attention 和 Triton 文件仍然存在，但 CNN-only 默认不导入它们。
- `train.py`：增加 `--cnn_only`，解析参数后选择包；默认实验名改为 `OmniDS_CNN_only`；拒绝 DINO Similarity/蒸馏；保存模式标记。CNN-only 下使原有 `--train_iters` 真正传入训练和验证 Forward；原作者训练路径原先未传入该参数，未修改其行为。
- `test.py`：增加模式选择；CNN-only 从 Checkpoint 的 `net_opts` 恢复网络和扫描配置，仅 AMP、Triton 开关保留为运行时参数。这样修复原训练/测试默认 GEV 金字塔分别为 2/4 层导致的通道不兼容，不改变原模式的默认值。
- 两个入口：未请求 `--wandb` 时不导入 wandb。CNN-only 加载自己保存的受信 Checkpoint 时显式使用 `weights_only=False`，兼容新 PyTorch 对 EasyDict 序列化的加载行为；仍然严格加载权重，不接受原 DINO Checkpoint。
- `tools/smoke_cnn_only.py`：唯一新增的测试工具，使用原 `Dataset.buildLookupTable()` 和真实 OCam 标定构造网格，不随意生成随机 Grid，不写入数据集 LUT 缓存。

没有修改 Dataset、Loss、OCam 投影、四目 Reference/Target 融合、GWC、方差、空间聚合或 3D U-Net。

## 数据流和实际尺寸

下面是原训练默认配置下的实际 Forward 尺寸：`B=1, C=32`，完整候选 192，`num_downsample=1`，ERP 输出 160×640。

| 节点 | Tensor Shape |
|---|---|
| 四路灰度输入 | 每路 `[1,1,768,800]` |
| 原 CNN Encoder | 每路 `[1,32,384,400]` |
| 原 LUT | 每路 `[80,320,96,2]` |
| 球面扫描 | 每路 `[1,32,80,320,96]` |
| Similarity Profile | `[1,96,80,320]` |
| GEV，空间上采样之前 | `[1,1,96,40,160]` |
| GEV，空间上采样之后 | `[1,1,96,80,320]` |
| 当前深度采样的 CNN ERP | `[1,32,80,320]` |
| Context：原 `context_proj` + ReLU | `[1,32,80,320]` |
| 初始 Hidden：原 `state_proj` + Tanh | `[1,64,80,320]` |
| Similarity Lookup，4 层、半径 4 | `[1,36,80,320]` |
| GEV Lookup，默认 2 层、半径 4 | `[1,18,80,320]` |
| MotionEncoder 输出 | `[1,128,80,320]` |
| ConvGRU 输入：Context + Motion | `[1,160,80,320]` |
| 递归 Hidden | `[1,64,80,320]` |
| 每轮深度索引增量 | `[1,1,80,320]` |
| 上采样 Mask | `[1,36,80,320]` |
| 每轮训练预测 / 最终推理输出 | `[1,1,160,640]` |

GEV 金字塔也可显式配置为 4 层，此时 Lookup 是 36 通道；并非新增默认设置。

初始深度保持原代码的 `D_grid / 2`，即工作索引 48，不是 OT 初始化。
输出也保留原凸上采样中的 `factor * depth`，回到完整 192 候选的索引单位，与原 GT、Sequence Loss 配合。
没有引入 PDF-Omni 的 Split-Volume 或其他索引定义。

CNN 只运行一次，Similarity Profile 和 GEV 只构建一次。第 0 轮前采样一次 CNN ERP，初始化 Context 和 Hidden；之后每轮先 detach 当前索引，再用该索引重新采样 CNN ERP、更新 Context。Hidden 始终传到下一轮，不重新初始化。
UpdateBlock 中保留的 `dino_feat`/`conv_dino*` 名称现在接收 CNN Context，本次没有为重命名重构它们。

## 训练和测试

在 CUDA 环境、项目根目录运行；`--db_root` 是 `omnithings/` 的父目录。

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --cnn_only --name OmniDS_CNN_only \
  --db_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset \
  --dbname omnithings --mixed_precision --batch_size 1 --train_iters 12
```

上面沿用原作者默认 100 Epoch，可用 `--total_epochs` 指定；本次没有启动完整训练。
随机初始化不需要 `--pretrain_ckpt`。恢复时传自己的 `--restore_ckpt`，并沿用训练时的结构参数。
`--cnn_only` 不兼容 `--distilled`、`--use_dino_sim`；原 DINO 权重不能直接作为 CNN-only 权重加载。

```bash
CUDA_VISIBLE_DEVICES=1 python test.py \
  --cnn_only --ckpt checkpoints/OmniDS_CNN_only/OmniDS_CNN_only_e0.pth \
  --db_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset \
  --dbname omnithings --mixed_precision --train_iters 12
```

CNN-only 测试时以保存的 `net_opts` 为准，不需手动重复训练时的网络尺寸、通道和金字塔参数。

## 已执行验证（2026-11-10）

本地使用 `pdf_omni` Conda 环境验证语法、CNN-only 包导入、两个入口的模式选择及不兼容选项拒绝。
原 `module/` 所有文件 SHA256 均未改变；AST 比较确认 MotionEncoder、UpdateBlock、CNN ERP Sampling、凸上采样未变。其他九个副本模块与原文件仅允许内部包导入路径不同。

服务器独立验证目录：容器 `my-baseline-dev:/tmp/OmniDS-cnn-only`。测试使用物理 GPU 1，没有重启或修改原 Baseline 的训练任务。

```bash
# CPU 小尺寸：随机图像 + 真实标定网格，非真实样本性能测试。
python tools/smoke_cnn_only.py \
  --db_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset \
  --device cpu --output /tmp/omnids_cnn_cpu.json

# CUDA + AMP：原四路完整鱼眼、原 192 候选、原 ERP GT，一次更新后重载推理。
CUDA_VISIBLE_DEVICES=1 python tools/smoke_cnn_only.py \
  --db_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset \
  --device cuda --real_sample --mixed_precision --num_invdepth 192 \
  --output /tmp/omnids_cnn_real_cuda.json
```

- CPU：工作体 `D=16,H=16,W=32`，3 轮前向、Backward、参数更新和 Checkpoint 重载推理通过。
- CUDA + AMP 真实样本：工作体 `D=96,H=80,W=320`，3 轮前向、Backward、参数更新和重载推理通过。各关键训练分支均有有限、非零梯度。
- Hook 计数确认每次 Forward 中 CNN、Profile、GEV、Hidden 初始化各 1 次；Context 共 3 次，未变为固定 Context。
- 原入口联调：用临时驱动限制 Dataset 的索引列表，实际执行 `train.py` 的 4 个训练样本和 1 个评估样本，生成其原格式 DataParallel Checkpoint；再由 `test.py --cnn_only` 加载，对 1 个真实样本推理通过。没有为测试复制训练循环或修改 Dataset 文件。
- 该入口联调实际执行 2 次 optimizer update；另 2 次由原 GradScaler 跳过并降低 scale。保留作者 AMP/调度器顺序，本次未重写训练策略。进程峰值 allocated 6215.92 MiB，此数值仅对应这次 3 轮受限验证。
- 测试进程没有导入 `timm`、原 `module.*`、ERP Cross-Attention 或 MultiScaleDeformableAttention，没有下载 DINO 权重。

没有运行完整 Epoch/收敛实验，没有测试 Triton 路径，也未执行原 DINO 分支的权重推理。当前结果证明 CNN-only 的完整数据流及训练/测试接口可运行，不代表其精度已经达到原论文模型。

## 双目网络接口（网络阶段；真实 GT 训练适配见文末）

上文的训练命令仍然用于四目 CNN-only。现在同一包通过 `num_cams=2/4` 保留两条路径，默认仍为四目；没有复制另一套网络，也没有修改原始 `module/`、Dataset、训练和测试入口。

### 使用方式与单位

```python
import torch
from module_cnn_only.network import OmniDS
from utils.common import Edict

model = OmniDS(Edict(num_cams=2, mixed_precision=True)).cuda()
# left/right: [B,1或3,768,800]；默认 Encoder 将 RGB 求和为灰度。
# 若需要原 RGB Encoder，构造时同时设置 use_rgb=True，并输入三通道。
# grid_left/grid_right: [160,240,96,2]，align_corners=True 的归一化坐标。
# valid: bool [B,1,96,160,240]，由调用方计算左右共同候选有效性。
predictions = model([left, right], [grid_left, grid_right],
                    valid_mask=valid, iters=12)
# 12 个 [B,1,320,480]，数值为连续候选索引 [0,95]，不是米或水平视差。

model.eval()
with torch.no_grad():
    final_index = model([left, right], [grid_left, grid_right],
                        valid_mask=valid, iters=12, test_mode=True)
```

两张图共享原 `FeatureLayers`，不是两个独立 Encoder。网格以左相机光心为原点；左网格必须沿 D 保持不变，支持 `expand`/非连续内存；右网格随 D 改变。同一批次共享这对 Grid，候选 Mask 可以按 B 不同。

双目默认设置集中在原网络构造参数中：`num_invdepth=num_depth_samples=96`、`equi_h=320`、`equi_w=480`、`num_downsample=1`。这里 `equi_h/w` 是最终预测尺寸，工作体和 GRU 的尺寸为其一半，即 160×240。小尺寸测试覆盖这些参数，不在其他模块硬编码 D/H/W。

工作轴与输出使用同一套 96 个候选，索引不再乘 2。若调用方从原 192 个逆深度候选按 `[0,2,...,190]` 取出 96 个值，则输出索引 q 对应这条工作轴的 `rho_min + q * delta_rho`；空间上采样不改变这个公式。完整候选值和 GT 换算由下一阶段的数据适配负责。

### 双目数据流

- `sweep.py`：沿用原 5D `grid_sample`，接受两路或四路特征和对应 Grid。
- `SimilarityProfileEncoder`：双目直接取左体为 Reference、右体为 Target，跳过前后/左右混合 MLP；使用完整 D 轴，保持原相关性公式和空间聚合。
- `GEVModule`：GWC 与两相机方差各 8 通道，输入 3D U-Net 仍为 16 通道。保留默认 `torch.var(correction=1)`，双目逐通道方差为 `(left-right)^2/2`。
- `_sample_cnn_erp`：当前索引采样左右特征，权重头从 `4C→C→4` 变为 `2C→C→2`。右候选不可见时用左参考；没有任何有效候选的像素输出零。整数索引不错误地要求下一层有效，分数索引则要求相邻两层均有效。
- 第零轮初始化 Hidden 和 Context，其余每轮只刷新 Context。保留 MotionEncoder、ConvGRU、DepthHead、Hidden 的递归传递和索引 detach；每轮更新后限制索引到 `[0,D-1]`。
- 保留凸上采样头和结构，双目只做空间上采样。无效邻居/图像边界在 Softmax 前排除，防止无效零值把有效深度拉向第零层，也保证常数索引场在边界保持常数；四目仍按原方式计算。

调用方的 `valid_mask` 应包含两相机 FOV、图像范围和相机 invalid_mask。网络额外检查坐标有限性和范围；无效相关性/方差/聚合值清零，Lookup 的支持 Mask 随深度金字塔同步池化。全无效像素的最终输出为零，不能将其当作有效的第零层深度。

双目模式明确拒绝 `use_triton=True`，当前融合内核写死四相机；四目 Triton 路径保留。双目与四目的权重不严格兼容，尤其是视图权重头和四目混合 MLP；没有实现自动迁移或静默部分加载。

### 实测尺寸与验证

| 双目张量 | 实际 Shape，B=1 |
|---|---|
| 两路 CNN 特征 | 每路 `[1,32,384,400]` |
| 左/右球面体 | 每路 `[1,32,160,240,96]` |
| Similarity Profile | `[1,96,160,240]` |
| GEV，恢复空间分辨率前 | `[1,1,96,80,120]` |
| GEV，恢复空间分辨率后 | `[1,1,96,160,240]` |
| Similarity / GEV Lookup | `[1,36,160,240]` / `[1,18,160,240]` |
| Context / Hidden | `[1,32,160,240]` / `[1,64,160,240]` |
| Motion / GRU Input | `[1,128,160,240]` / `[1,160,160,240]` |
| 增量 / 上采样 Mask | `[1,1,160,240]` / `[1,36,160,240]` |
| 每轮预测 / 最终输出 | `[1,1,320,480]` |

GEV 的内部空间池化和两次 U-Net 下采样为 `96×160×240 → 96×80×120 → 48×40×60 → 24×20×30`，可完整恢复。

```bash
# 小尺寸 CPU：真实 OCam 标定网格 + 随机图像 + 人工索引损失，含 Mask 边界测试。
python tools/smoke_cnn_only.py --num_cams 2 \
  --db_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset \
  --device cpu --output /tmp/omnids_stereo_cpu.json

# 正式尺寸：真实左右图像、真实标定、12轮 AMP 前后向与参数更新。
# 这个独立网络工具使用人工索引目标；真实左光心 GT 的训练命令见文末。
CUDA_VISIBLE_DEVICES=1 python tools/smoke_cnn_only.py --num_cams 2 \
  --full_size --real_sample --iters 12 \
  --db_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset \
  --device cuda --mixed_precision --output /tmp/omnids_stereo_cuda_12.json
```

实际测试在服务器容器独立目录 `/tmp/OmniDS-stereo-validation` 执行，没有启动正式训练：

- 小尺寸 `D/H/W=16/16/24`，3 轮 Forward/Backward、参数更新和严格重载通过。
- 左右输入、共享 CNN、Context/State、Similarity、GEV、GRU、DepthHead 和上采样头均获得有限非零梯度。
- 部分候选无效、全部无效、NaN 网格占位符、左侧 Context 回退、无效 Lookup、常数索引上采样及孤立有效邻居的极端权重测试通过。
- 正式 `D/H/W=96/160/240`、768×800 双输入、Batch Size=1、AMP、12轮 Forward/Backward、更新和重载推理通过。训练步峰值 allocated 为 6076.90 MiB（约 5.93 GiB）；这不是 CUDA 总占用，也不是未来正式训练的显存保证。
- 正式测试中 CNN、Profile、GEV、Hidden 初始化各调用 1 次；Context 和每种 Lookup 各调用 12 次。
- `B=2` 的 RGB 输入前后向通过；错误相机数、Triton、候选轴不一致、缺失/错误 Mask、错误 Grid 尺寸及左 Grid 随深度变化均被拒绝。
- 使用修改前的独立代码快照、相同权重和真实标定小尺寸四目输入，比较 3 轮训练输出、所有参数梯度及推理输出，均逐元素完全一致；旧四目权重严格加载通过。
- 原 `module/`、`utils/`、`dataset.py`、`train.py`、`test.py` 与本次修改前的 SHA256 均一致。

机器可读结果见 `CNN_ONLY_STEREO_VALIDATION.json`。

**上述结果属于独立网络验证，不是真实 GT 训练。** 当前 `train.py --stereo` 已按文末章节接入左光心 GT；`--cnn_only` 单独使用仍是四目，原 `test.py` 仍未适配双目数据，不能直接用它评估双目权重。

## 基础双目 Baseline：GWC-only + Static Left Context

双目默认 `use_group_variance=False`、`use_dynamic_context=False`。原方差和动态采样代码均保留在开关分支中；四目默认仍使用二者，不改变原四目权重结构与计算。

```python
model = OmniDS(Edict(num_cams=2, mixed_precision=True)).cuda()
predictions = model([left, right], [grid_left, grid_right],
                    valid_mask=joint_valid, left_valid_mask=left_valid, iters=12)
```

- `GEVModule` 不再计算 `stack/var` 或拼接方差，直接将 GWC 输入原 3D U-Net，入口由 16 改为 8 通道；Guided Excitation、内部通道、输出及深度金字塔不变。
- Context 在迭代前只对左 CNN 特征做一次 `grid_sample`，使用深度不变的左 Grid；两个原投影头产生静态 Context 和初始 Hidden。GRU 始终复用同一 Context，Hidden 继续递归更新。
- Similarity 和 GEV 每轮仍根据当前索引 Lookup。没有移除匹配分支或修改 MotionEncoder/GRU/DepthHead/凸上采样。
- `left_valid_mask` 为 bool `[B,1,160,240]`，只描述左相机自身的有效投影；不能用左右共同候选的 `any(D)` 替代。未提供时由左 Grid 的有限性和范围推导，因此调用方必须在左 Grid 中将相机 invalid_mask 编码为越界值，或显式提供左 Mask。
- `valid_mask` 仍为 bool `[B,1,96,160,240]`，描述共同候选有效性。左 Mask 若显式提供，也会排除对应位置的匹配证据；相反，右侧不可见不会清空有效左 Context。最终无双目支持的位置仍输出零。

默认关键尺寸没有变化：Context `[B,32,160,240]`、Hidden `[B,64,160,240]`、Similarity/GEV Lookup 为 36/18 通道、GRU 输入 `[B,160,160,240]`、输出 `[B,1,320,480]` 且索引保持 `[0,95]`。U-Net 入口现在是 `[B,8,96,80,120]`（已执行原空间池化）。

恢复原双目创新路径时明确设置：

```python
model = OmniDS(Edict(num_cams=2, use_group_variance=True, use_dynamic_context=True))
```

开关相互独立，支持 GWC-only/Variance 与 Static/Dynamic 的四种组合。旧 16 通道 GEV 权重不能严格加载到基础 8 通道结构，程序会正常报错；未实现 `strict=False` 或静默跳过。恢复两个开关后旧双目权重可严格加载。

最新验证沿用同一个 Smoke Test：

```bash
# 基础双目，小尺寸前后向、Mask 和权重重载。
python tools/smoke_cnn_only.py --num_cams 2 --db_root /data

# 恢复两个创新模块，检查原双目路径。
python tools/smoke_cnn_only.py --num_cams 2 --use_group_variance --use_dynamic_context --db_root /data

# 正式尺寸、真实图像/标定、12轮 AMP；网络测试仍使用人工索引目标。
CUDA_VISIBLE_DEVICES=1 python tools/smoke_cnn_only.py --num_cams 2 \
  --full_size --real_sample --iters 12 --mixed_precision --device cuda \
  --db_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset
```

实际验证通过：

- 基础模式禁用 `Tensor.var` 和原动态采样方法的执行，Forward/Backward 仍通过；实测 U-Net 入口为 8 通道。
- 12 轮中 Context/Hidden 初始化各 1 次，Similarity/GEV Lookup 各 12 次；逐轮 Context 数值完全相同，Hidden 与查询索引持续变化。
- 左侧独立有效性、全部匹配候选无效但左 Context 仍有效、显式左 Mask、部分/全部无效候选、NaN 网格、上采样与 AMP 检查通过。
- 两个开关分别开启和同时开启的路径、严格保存重载通过；四目与之前独立快照的训练输出、梯度及推理结果逐元素一致。
- 默认正式尺寸 12 轮训练步峰值 allocated 4802.32 MiB（约 4.69 GiB），该数字仅对应这次网络 Smoke Test，不代表完整训练显存。

上述结果是人工索引目标的网络验证；下面单独列出新增的真实左光心 GT 训练验证，二者不混用。


## 基于原 train.py 的双目训练适配

只在现有 `train.py` 中增加 `--stereo` 分支，没有复制训练入口、优化器或训练循环。原 `fetch_optimizer` 与参数计数函数 AST 不变，仍用原 AdamW、OneCycleLR、Sequence Loss、梯度裁剪和 Checkpoint 字段；没有 LEC 或 DINO 蒸馏。`--stereo` 自动选择 CNN-only、两目、96 候选和基础开关 False/False；不加它时保持原四目行为和默认尺寸。

唯一新增的数据文件 `dataset_stereo.py` 继承原 `Dataset`，复用原 OCam、Rig 变换、读图、灰度转换、invalid-mask 归一化及光度增强。它仍返回原 `(imgs, gt, valid, raw_imgs)`，使训练循环无需换一套数据格式。这里不能直接调用原四路 `loadImages` 或已裁剪的 Rig GT 加载器：

- 只读取指定左右鱼眼，保持 768×800，不做几何增强。使用两张图共享的原 `photo_aug`，不调用四目 eraser。
- 使用真实配置中的完整 192 逆深度轴，按 `[0,2,...,190]` 选出 96 值，训练/输出/GT 都解释为工作轴 `[0,95]`。边界之外的 GT 无效，不把第 191 层误算进来。
- 最终 320×480 的角度网格是输出定义；工作 160×240 射线取最终网格 `[::2,::2]`，对应凸上采样的 2×空间关系。左 Grid 深度广播，右 Grid 按深度变换/投影，joint 与 left Mask 独立计算。
- 从完整原始逆深度 TIFF 恢复 Rig 点云，变换到左相机，重新计算径向距离和方向，直接 Z-buffer 到最终 320×480。没有缩放原 GT、工作 GT 插值或索引乘 2。
- 监督要求 GT 点在左右相机的有效视场内，并进行已观测点云的左右相机 Z-buffer 检查；这不能恢复 Rig 中心未观测到的表面。GT Mask 再与工作 joint-any 的最近邻 2×输出支持及相邻候选有效性相交，空洞保持无效。
- 官方 9216 个训练样本默认划出 512 验证，实际训练 8704；官方 1024 测试样本保留。`--val_count 0` 可禁用验证，不对空数组计算均值。

在 CUDA 环境的 OmniDS 根目录运行：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --stereo \
  --ref_cam cam4 --target_cam cam3 \
  --db_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset \
  --dbname omnithings --mixed_precision --batch_size 1 --train_iters 12 \
  --name OmniDS_Stereo_Basic
```

`--db_root` 是 `omnithings/` 的父目录。默认工作体 96×160×240、输出 320×480。原训练默认 100 Epoch 不变，可自行传 `--total_epochs`；本次没有启动完整训练。相机对可配置，cam4→cam3 是这次验证用的实际标定相机对，不是硬编码模型假设或精度最优结论。默认角度是根据实际共同视场推导的中心、水平130°和垂直±45°，逐候选 Mask 仍决定有效性。

恢复创新模块时增加 `--use_group_variance --use_dynamic_context`。恢复训练用 `--restore_ckpt`，只加载自己的兼容双目 Checkpoint；权重严格加载，恢复时校验相机标定、候选轴、角度、网格 metadata 和两个消融开关，禁止静默混用 Rig GT/旧192轴或不兼容16通道权重。动态 Context 开关即使不改变权重 Shape，也必须与恢复 Checkpoint 一致，否则明确报错。Checkpoint 额外保存 scheduler、GradScaler、global_step、optimizer_updates、真实几何和训练参数。双目 AMP 跳过 optimizer 更新时也跳过 scheduler；原四目更新顺序不改。

有限规模真实 GT 验证命令：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --stereo \
  --ref_cam cam4 --target_cam cam3 \
  --db_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset \
  --mixed_precision --total_epochs 1 --max_train_steps 4 --max_eval_samples 1 \
  --no_augment --name OmniDS_stereo_basic_train_smoke
```

`--max_train_steps` 是受限 Smoke Test：只保存一个部分 Epoch Checkpoint，并标记 `bounded_smoke=True`，不应把它当作完成完整 Epoch 的模型。正式训练去掉这两个 max 参数。

实际结果（2026-11-10，容器 `/tmp/OmniDS-basic`）：

- 真实数据读取：两路 `[1,768,800]`、最终 GT `[320,480]`；首个训练样本有效监督约21.15%，有效索引约0～69.06；未插值填补空洞。
- 正式96×160×240工作体、12轮AMP，原 `train.py` 执行4个真实有监督batch，完成3次 optimizer 更新，1次由GradScaler跳过；随后1帧held-out验证通过并保存原格式DataParallel权重。
- 由该 Checkpoint 恢复后执行1个batch，global_step从4变5，optimizer_updates从3变4；scheduler恢复并与更新次数一致，完成验证及保存。
- 另外1个FP32有监督batch与 `--val_count 0` 检查通过，验证明确跳过、没有空均值NaN；刻意改变保存的 Dynamic Context 开关后，恢复会被明确拒绝，不会因权重 Shape 相同而静默改变模型行为。
- 独立稀疏Rig GT单点检查通过：左径向距离5.03587125083058、96轴索引31.29051399230957，与独立坐标变换一致；最近表面Z-buffer检查通过。

原 `Dataset.evalError` 也继续复用：它报告以 D 归一化的百分比误差，日志中的 MAE/RMS 不是米；Sequence Loss 则在 `[0,95]` 工作索引单位中计算。这些受限验证证明数据/梯度/更新/恢复接口可执行，不证明网络已收敛。

独立 `test.py`、通用双目可视化和其他数据集尚未适配。训练内部的双目验证已接好，不要使用原四目全景可视化或 Rig GT 测试路径解释双目输出。
