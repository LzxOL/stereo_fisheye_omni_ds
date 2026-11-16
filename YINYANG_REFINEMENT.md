# 在已训练DA-V2 Baseline上增加Yin-Yang前端

本版修正此前“独立双网格网络”的组织方式：正式入口始终是
`module_cnn_only/network.py::OmniDS`，两种网格共用已训练DA-V2与GRU主干。
`dual_grid_stereo.py`仅适配旧调用方式，不维护第二套训练网络。

## 当前完整流程

```text
Left / Right → 原共享CNN
       ↓
单网格扫描，或 Yin/Yang扫描 → Geometry Enhancer → GWC → 双向Warp / Fusion
       ↓
原Similarity + Spatial Aggregation      原Guided 3D U-Net → GEV
       │                                      │
       └──────────────────┬───────────────────┘
                          ↓ 每轮Lookup
Left RGB → Frozen DA-V2 ViT-B → 原四层Adapter（每Forward一次）
                                ↓
                分别按Yin/Yang左Grid对齐 + 原ResidualRefine
                                ↓
左CNN静态Context → 原GeometryContextFusion ← DA-V2几何特征
                                ↓
                   原context_proj / 原state_proj
                         Context32 / Hidden64
                                ↓
          原MotionEncoder → 原UpdateBlock / ConvGRU / DepthHead
                                ↓
                       原凸上采样 → 每轮索引预测
```

关键点：`forward()`不会在双网格前端后提前返回。它先得到每个网格的匹配体与左Context，
再在同一个后半循环中执行DA-V2、两路投影、GRU和上采样。这里没有换掉原GRU权重。

## 文件和调用关系

- `network.py::_build_yinyang_volumes`：扫描、增强、两路Similarity/GWC、跨网格融合；只准备前端数据。
- `network.py::forward`：统一执行DA-V2 Context与原深度迭代；Yin/Yang共用权重，各自保留状态。
- `dav2_geometry.py::extract_features`：原ViT与Adapter，计算一次左鱼眼Patch特征。
- `dav2_geometry.py::forward/align_features`：复用Patch特征，按各自真实左Grid对齐；没有Resize两套原生网格。
- `similarity_context.py::GEVModule.build_from_cost_volume`：融合GWC直接进入原Guide/U-Net/金字塔。原`build()`也调用此方法，数学计算不变。
- `network.py::load_baseline_weights`：显式继承已训练主干，新增几何前端保留初始化。所有共享键和Shape严格校验。
- `tools/smoke_dav2_yinyang.py`：当前正式结构的合成与官方模型Smoke Test。

## 与此前独立实验的差别

当前版本以训练过的DA-V2主干为准：

| 节点 | 当前共用主干 |
|---|---|
| 左CNN / DA-V2对齐特征 | 各`[B,32,H,W]` |
| 增强Context / 初始Hidden | `[B,32,H,W]` / `[B,64,H,W]` |
| GEV | `[B,1,96,H,W]` |
| Similarity Lookup | `[B,36,H,W]`，4层×9 |
| GEV Lookup | `[B,18,H,W]`，2层×9 |
| Motion / GRU Input | `[B,128,H,W]` / `[B,160,H,W]` |
| 初始索引 | 原中点初始化：D/2，D=96时为48 |
| 原DepthHead增量 | `[B,1,H,W]`，候选索引单位 |
| 凸上采样预测 | `[B,1,2H,2W]`，索引不乘2 |

早先的“8通道聚合体→144通道Lookup→独立MotionEncoder/Soft-argmax初始化”
仍保留在`depth_refinement.py`供单独实验，但不再注册或调用于当前`OmniDS`。
这是为了真正沿用已训练主干，而不是把DA-V2接到另一套不兼容的更新器上。

生产Yin50原生Padding尺寸为Yin136×216、Yang128×328，对应当前原凸上采样结果
分别为272×432和256×656；没有把它们伪装成同一320×480左鱼眼图。
后续正式双网格训练仍需对应GT映射或统一输出映射；不能直接Resize GT。

## 使用与继承权重

```python
from module_cnn_only.network import OmniDS

options = dict(
    num_cams=2, num_invdepth=96, num_depth_samples=96,
    use_dav2_geometry=True, use_yinyang=True,
    dav2_model_repo='/cache/models/Depth-Anything-V2',
    dav2_checkpoint='/cache/models/depth_anything_v2_vitb.pth',
    mixed_precision=True,
)
model = OmniDS(options, grid_package=package).cuda()
report = model.load_baseline_weights(snapshot['net_state_dict'])
outputs = model(
    [left, right], dav2_rgb=prepared_left_rgb, iters=12,
    yang_to_yin_quality=quality_on_yin,
    yin_to_yang_quality=quality_on_yang,
)
```

RGB仍使用原`prepare_dav2_rgb`官方预处理，来自真实左RGB，而非复制灰度。
Q仍由调用方明确提供连续质量，不把Mask冒充质量。
DA-V2仅增强Context和Hidden初始化，不进入GWC或GEV Guide。

`load_baseline_weights()`支持DataParallel前缀；只明确移除旧闲置视图权重头的4个键，
并严格继承CNN、Similarity、GEV、Context/State、原UpdateBlock、ViT、Adapter、Fusion。
不会用`strict=False`隐藏缺失或尺寸问题。新前端参数正常初始化，使用新的optimizer；
本接口是预训练权重迁移，不宣称已迁移旧optimizer或可以直接续跑原单网格训练。
旧独立Yin-Yang实验权重结构不同，不能作为本版DA主干权重直接加载。

## 实际验证

CPU：

```bash
python -B tools/smoke_dav2_yinyang.py --device cpu
```

CPU的ViT是明确标记的合成测试替身；Adapter/Fusion/GEV/原UpdateBlock均为实际代码。
已验证DA关闭、DA开启FP32、CPU BF16，双网格前后向、融合到两个投影头的连接、静态
Context、Hidden递归、全部无效Mask、严格状态重载。扰动RGB改变Context而不改变GEV。
原单网格DA前向和所有参数梯度与修改前逐项完全一致。

服务器还实际使用：官方ViT-B、现成官方预训练权重、真实OmniThings cam4/cam3
标定和04887帧RGB，以及`omnids_dav2_context_resume30_20261113_e27.pth`。
在独立临时目录运行，没有覆盖服务器原代码/模型，也没有停止任何已有训练。

```bash
python -B tools/smoke_dav2_yinyang.py --device cuda:2 \
  --model_repo /cache/models/Depth-Anything-V2 \
  --checkpoint /cache/models/depth_anything_v2_vitb.pth \
  --baseline_checkpoint /workspace/stereo_fisheye_omni_ds/checkpoints/omnids_dav2_context_resume30_20261113/omnids_dav2_context_resume30_20261113_e27.pth \
  --data_root /root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset/omnithings
```

为控制Smoke开销，只降低测试网格角采样密度到15°；窗口、旋转、OCam和完整96候选
仍为真实几何。测试网格Yin8×16、Yang8×24，原图仍768×800，DA输入518×546。
完成2次AMP优化、每网格3轮GRU，继承481个共享state项、新增20个前端state项，
Backbone哈希不变，Adapter/Fusion/原GRU梯度有效。小网格测试峰值allocated约1834.84MiB。
这些显存与耗时不能代替生产原生大网格的profiling。

损失使用合成索引目标，Q使用合成常数0.8；这次验证证明接线及可训练性，不是
真实GT精度评估，不能据此声称超过e27的结果。真实GT训练和最终统一输出仍是后续工作。
详见`DAV2_YINYANG_VALIDATION.json`。
