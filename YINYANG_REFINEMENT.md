# Yin–Yang 多通道代价聚合与 GRU V1

实现位置是当前 `OmniDS` 项目的 `module_cnn_only/`，不是另建 PDF-Omni 四目分支。
原 `module/`、`module_cnn_only/network.py`、扫描、几何、融合、Dataset 和训练入口均未改动。
以前独立实现的前半部分在新入口中串联；旧网络继续使用原有接口与权重。

## 文件与数据流

- `module_cnn_only/depth_refinement.py`：多通道 3D 聚合、分类初始化、深度金字塔、Lookup、三路 Motion Encoder 和迭代器。
- `module_cnn_only/dual_grid_stereo.py`：完整双网格双目入口 `DualFisheyeStereoNetwork`。
- `tools/smoke_dual_grid_refinement.py`：默认尺寸组件测试、小尺寸完整网络前后向及 AMP/权重重载验证。

```text
Left/Right → 原共享 FeatureLayers（一次）
             ├─ Yin sweep → 共享 Geometry Enhancer → 原 GWC ─┐
             └─ Yang sweep → 同一个 Enhancer → 原 GWC ──────┤
                                                        ↓
                                       原 BidirectionalCrossGridWarp
                                                        ↓
                              原 GeometryAwareFeatureFusion（共享）
                                        ┌───────────────┴──────────────┐
                                     Yin fused                    Yang fused
                                        ↓                             ↓
                              同一个 StereoDepthRefinement，分别调用
                                        ↓
                        原 3D U-Net 主体，输出从 1 改为 8 通道
                             ├─ 独立 Conv3d(8,1,1) → Masked Softmax → init_idx
                             └─ DepthPyramid（仅 D 池化，每网格一次）
                                        ↓
                                 DepthWiseLookup
                                        ↓
                 Lookup + 投影后的静态左 CNN Context + 当前归一化 idx
                                        ↓
                          MotionEncoder → 原 ConvGRU → 原 DepthHead
                                        ↓
                             idx = clamp(detach(idx)+delta, 0, D-1)
                                        ↓
                              默认 3 轮，保留各轮和初始预测
```

两网格共享 CNN、Geometry Encoder/Refiner、Fusion、3D 聚合、分类器、Context/State 投影、MotionEncoder、GRU 和 DepthHead 的参数。
但两者的 Hidden State、金字塔、预测张量彼此独立，不在不同球面方向间直接混用隐藏状态。

## 配置与输入约定

`StereoDepthRefinement` 构造参数集中定义默认值，H/W 从输入读取：

```python
options = dict(
    num_depth=96, gwc_groups=8, aggregated_channels=8,
    num_pyramid_levels=2, lookup_radius=4,
    context_channels=32, motion_feature_dim=128,
    gru_hidden_dim=64, gru_iters=3,
    reg_channels=(16, 32, 48), use_refinement=True,
)
```

完整入口的 D 必须等于几何包候选数；`num_depth` 不传时直接使用包中的 D。
`use_refinement=False` 只输出 Soft-argmax 初始化，用于消融。
外部 `torch.autocast` 控制 AMP，使用 AMP 时不要对整个模型调用 `.half()`：几何缓存保留 FP32。

`forward(imgs, yang_to_yin_quality, yin_to_yang_quality, iters=None)`：

- `imgs=[left,right]`；原共享 CNN 默认灰度模式，行为与 `FeatureLayers` 一致；RGB 模式通过 `use_rgb=True` 配置。
- 两个 Quality 分别在 Yin/Yang 目标网格上，浮点 `[B,1,D,H,W]`，Q∈[0,1]。
- 沿用原 Fusion 的显式连续 Quality 接口。本阶段没有自动发明质量估计器，正式使用前需要调用方提供 Q；不能把 Mask 当成 Q。
- 一个模型实例使用一个相机 Rig 的几何包，batch 内共享几何；不宣称已支持不同 Rig 混批。

没有将双网格 Resize 为 `64×128`。该尺寸仅为附件要求的独立组件测试尺寸。
实际 Yin50 缓存读取结果：Yin native `135×212`，padding 后 `136×216`；Yang native `121×324`，padding 后 `128×328`。

| 张量 | 单网格通用 Shape | D=96、组件测试 H=64/W=128 |
|---|---|---|
| Fused GWC / 聚合特征 | `[B,8,D,H,W]` | `[1,8,96,64,128]` |
| Depth logits / probability | `[B,1,D,H,W]` | `[1,1,96,64,128]` |
| 深度金字塔第 1 层 | `[B,8,D/2,H,W]` | `[1,8,48,64,128]` |
| 初始化 / 每轮预测 / delta | `[B,1,H,W]` | `[1,1,64,128]` |
| Lookup | `[B,8×2×9,H,W]` | `[1,144,64,128]` |
| 静态左 Context | `[B,32,H,W]` | `[1,32,64,128]` |
| Motion | `[B,128,H,W]` | `[1,128,64,128]` |
| GRU input | `[B,160,H,W]` | `[1,160,64,128]` |
| Hidden | `[B,64,H,W]` | `[1,64,64,128]` |

新后半部分不使用原 GEV 的预先 2× 空间池化，直接经过两次 U-Net 下采样/上采样，因此 D/H/W 要求是 4 的倍数。两套原生 padding 尺寸均满足。
原 GWC 函数仍接收 `[B,C,H,W,D]`；新入口在其前后显式 permute，其他接口保持 `[B,C,D,H,W]`。

## Mask、初始化与 Lookup

1. 输入聚合前清除无效代价，聚合后再次清除无效候选。
2. 分类头输出的是可学习匹配 logit，使用正向 `softmax(logits)`；不把它当作固定距离代价加负号。
3. 初始化只对有效候选归一化；全无效列的概率、初始值和有效 Mask 均为零/False，避免 `softmax(-inf,...,-inf)`。
4. 左 Context 使用现成 `sample_reference`，按左自身 `left_feature_valid` 采样，不复用 joint-mask 之后的扫描切片。迭代输出有效性再取融合候选有效性与左参考有效性的交集。
5. 金字塔只沿 D 平均。部分无效的池化窗口按有效子候选归一化，全无效输出零；全有效时与普通 `avg_pool3d` 完全一致。
6. 第 l 层以 `idx / 2**l + [-r,...,r]` 查询，沿用 OmniDS 索引约定。先 Clamp，再 floor/ceil/gather 和线性插值；无需 3D `grid_sample`。
7. 只有正权重邻点都有效时才能插值；整数查询不依赖零权重的 ceil 邻点。缺支持时输出零。
8. 各层查询按 level→channel→offset 合并成 144 通道；金字塔只构建一次，且 Lookup 不会每轮复制完整 5D 体。

GRU 处理的是局部二维状态。把完整 Volume 直接输入二维 GRU，必须把 D×C 压成巨大通道数；它也无法显式表达“当前估计附近”的搜索中心。
Lookup 用连续索引选取多尺度局部候选，将匹配证据从 `C×D` 缩到 `C×L×(2r+1)`，让同一个二维 GRU 每轮关注当前解附近，并以残差修正解。
它仍能通过取到的体特征向 3D 聚合和 CNN 反传，既没有将整个体 detach，也没有重新构建候选轴。

## 与原 OmniDS 的区别

- 复用原 `RegularizationNet3D` 主体、`Conv2D` 投影、`ConvGRU`、`DepthHead`；原类代码不变，仅新聚合实例替换输出层为 8 通道。
- 原 GEV 是独立的单通道匹配分支；这里直接聚合已经融合的 GWC，不再建独立 GEV、Similarity 分支或方差分支。
- 原 MotionEncoder 有 Similarity/DINO/GEV 三路加 depth；现在只有多通道 Lookup/CNN Context/depth，按任务的 64+64+32→127+1 结构实现，原四路编码器不适合原样调用。
- 原 CNN-only 用候选轴中点初始化；这里使用分类概率 Soft-argmax 初始化，不引入 PDF-Omni 的 OT。
- 采用静态左 Context；不引入 DINO/DA-V2 或动态跨相机 Context。
- 不执行原凸上采样：当前交付保留两套原生网格的预测。没有擅自设计统一输出网格融合策略。

实际几何缓存的候选为原 192 轴的 `[::2]` 共 96 层：首层 rho≈`2.220446049250313e-16`，是远场有限 EPS 近似；末层 rho≈`0.6028875138822783 /m`，最近距离 `1.6586842105263155 m`。
`index_to_depth` 在包中的实际候选轴上线性插值，不重新假设原 192 层的终点，不将输出索引乘 2。转换结果分别为径向逆深度和径向距离，不能当光轴 Z-depth。

## 训练接口和当前边界

旧 `train.py --stereo` 返回的是单个左鱼眼/局部 ERP 的 GT；新网络输出两个不同参数化上的预测。
因此本阶段没有加一个表面可运行、实际却对错 GT 的命令行开关。**现有训练命令还不能直接训练这个新双网格入口。**
下一阶段需在数据适配层把已有左中心三维 GT 按射线映射到各 chart，保留 Z-buffer/监督 Mask；不能直接 Resize，也不能复用 Rig-centered GT。
同样需要确定真实连续采样质量 Q 的来源。这里实现的是完整可反传网络与明确接口，不宣称正式数据训练闭环已经完成。

在训练循环中，模型调用与现有优化器、AMP、Sequence Loss 可按以下方式连接；前提是调用方已经正确准备各网格 GT：

```python
from module_cnn_only.dual_grid_stereo import DualFisheyeStereoNetwork
from module_cnn_only.loss_functions import sequence_loss

net = DualFisheyeStereoNetwork(package, **options).to(device)
# optimizer/scheduler/GradScaler 沿用现有训练配置；不覆盖原 checkpoint。
outputs = net([left_img, right_img], quality_yang_to_yin, quality_yin_to_yang)
losses = []
for name in ('yin', 'yang'):
    row = outputs[name]
    valid = gt_valid[name] & row['valid']
    if valid.any():  # 现有 sequence_loss 对空监督会求mean得到NaN，调用方须跳过。
        # 每轮idx detach后，迭代损失无法训练初始化分类头，必须同时监督init。
        predictions = [row['init_depth_idx'], *row['predictions']]
        losses.append(sequence_loss(predictions, gt_index[name], valid))
if losses:
    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()
```

上述示例需 `import torch` 并创建现有 optimizer；不是可直接执行的数据训练脚本。
GT 索引也必须对应包中同一 96 候选轴，所有预测与 Mask 在对应 chart 上监督。
checkpoint 除 `net.state_dict()` 外应保存 options、optimizer/scheduler、几何包配置/哈希和候选轴；严格重载时重新提供同一几何包。
原 CNN-only/PDF-Omni 权重不是新结构的严格兼容权重，不能使用 `strict=False` 隐藏结构差异。

## 实际验证

本地 macOS、PyTorch 2.11.0、CPU：

```bash
/Users/linzhaoxian/anaconda3/envs/pdf_omni/bin/python -B tools/smoke_dual_grid_refinement.py --device cpu
```

- 默认 `D/H/W=96/64/128` 的 Pyramid、144 通道 Lookup、128 通道 Motion、160 通道 GRU input 与 64 通道 Hidden：通过。
- 合成已知值：整数、浮点插值、边界 Clamp、每像素不同索引、跨通道排列、无效邻点及反向传播：通过。
- 完整小网络：原始图像 `[1,3,32,40]`，共享 CNN 每路输出 `[1,32,16,20]`。
- 四扫描体：Yin 左/右各 `[1,32,16,8,16]`；Yang 左/右各 `[1,32,16,16,16]`。
- 3 轮输出：Yin 各 `[1,1,8,16]`；Yang 各 `[1,1,16,16]`。
- FP32 与 CPU BF16 autocast 各两步优化：loss 有限，左右输入、CNN、增强、融合、聚合、分类、Context/State、Motion、GRU、DepthHead 梯度有限且有效。
- 增强末层零初始化使其上游首步梯度为零；第二步已验证 Geometry Encoder 获得非零梯度。
- Hook 检查：CNN 1 次；每个网格聚合/Pyramid/Context/Hidden 初始化各 1 次；每个网格 Lookup/GRU 各 3 次。Hidden 持续递归，Context 固定。
- 部分无效及全无效候选、严格 checkpoint 保存/重载、关闭 refinement 消融、索引到实际候选轴转换：通过。
- 完整新模型参数量：`1,345,523`。

没有运行 CUDA FP16、真实原生大网格前后向或 OmniThings 真 GT 训练；不能将 CPU BF16 结果当作 GPU 显存验证或精度提升证据。合成质量及合成监督只用于验证实现。
没有启动正式训练、下载数据或改写已有权重。
