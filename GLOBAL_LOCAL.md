# Global–Local 双域匹配特征

基于 003e34a。训练、验证共用 train.py 的 stereo_forward。StereoDataset 返回左右各一份真实 RGB，两路共享已有光度增强。Global 输入沿用数据集标准化，Local 输入取原分辨率 RGB / 255，DA-V2 单独预处理左 RGB。

## 如何读取前向

1. module_cnn_only/network.py 中，OmniDS.forward 执行原 FeatureLayers。_build_yinyang_volumes 对 Global 特征执行 sweep_patch 和 GeometryAwareFeatureEnhancer。
2. module_cnn_only/local_spherical_encoder.py 中，ChartSampling 用冻结几何包的角坐标定义生成每个相机自身光心的 RGB atlas。右 atlas 的坐标轴平行于左参考轴，投影时先乘左到右的旋转。使用每个相机的 OCam 和 invalid_mask 验证原图采样邻点。表注册为非持久 buffer。
3. 工作像素 j 对应 atlas RGB 像素 stride*j。每个方向有 stride*(N-1)+1 个输入像素。3×3 对称 padding 卷积的采样中心与这个定义一致。每个 chart 将左右 atlas 按 batch 拼合并调用一次共享 CNN。默认两次 3×3 卷积、16 隐通道、1×1 输出 32 通道。原鱼眼 FeatureLayers 有 18 个卷积层，这里选浅层 CNN 用于补充局部纹理。
4. 左 Local 特征沿 D 广播。右侧逐候选计算 R*r + rho*t，再把方向旋回 atlas 轴，调用 direction_to_grid。它与对三维点 R*(r/rho)+t 归一化有相同方向。直接使用包内正逆深度轴，包含首层的远场近似。FP32 坐标及严格双线性邻点检查用于每个查询，按深度分块采样。
5. Geometry Enhancer 后添加 alpha * Conv1x1([Global,Local])。alpha 初值为 0.01，四路共用参数。各相机局部采样缺失时保留该相机原描述。GEV guide 取融合前左特征。融合后的匹配描述同时进入原 Similarity 和每个 chart 的 8 组 GWC。跨网格融合、GEV、DA-V2 Context、ConvGRU 及左鱼眼输出继续使用原链路。

## 运行与消融

在训练容器仓库目录运行。以下为独立新实验的示例命令。

```bash
RUN_NAME=omnids_dav2_yinyang_global_local BATCH_SIZE=1 \
bash train_yinyang_omnithings.sh --use_local_spherical_encoder
```

通过 PRETRAIN_CKPT 指定单网格 DA-V2 或 Yin–Yang 的权重可以初始化新实验。RESTORE_CKPT 恢复完整训练状态。--local_width、--local_layers、--local_rgb_stride 的默认值分别为 16、2、2。--freeze_local_cnn 固定局部编码器。--local_only 在局部有效处取 Local 特征，缺失处回退到 Global。

单网格对照使用 train.py 的 --stereo --use_dav2_geometry --reference_grid left_fisheye。双网格基线另加 --use_yinyang 和 --yinyang_cache。新实验再加 --use_local_spherical_encoder。

## 验证记录

- 本地运行 `python tools/smoke_local_spherical.py` 和 `python tools/smoke_dav2_yinyang.py`。前者使用合成 RGB、Mask 和监督，采用存档 OCam 系数。涵盖右侧三维点重投影、零残差等价、无效回退、两步反传和严格重载。后者回归原路径。
- 容器使用 train.py，--train_iters 2 --total_epochs 1 --batch_size 1 --max_train_steps 2 --max_eval_samples 2 --num_workers 0 --gradient_checkpointing --mixed_precision --amp_init_scale 128。采用 cam4/cam3、Yin50、D=96，继承固定 batch 4 实验 e20 的 501 个状态键。实测完整新 checkpoint 严格恢复。单独测试单网格 DA-V2 e27 继承 481 个状态键。
- 两次训练 Loss 为 2.35326 和 1.87370，有效监督占 29.06% 和 30.47%。每步耗时 70.22 和 1.06 秒（含首次启动成本，共享 GPU）。峰值分配 9911 MiB，保留 10166 MiB。两帧验证归一化索引 MAE 1.43463%，RMS 5.06425%。本次样本量只适合验证链路。
- 在原双目有效候选中，两侧 Local 同时可用的比例为 Yin 2091398/2223856，Yang 2487224/2742354。右 atlas 沿用同名 chart 角窗，超出窗口时保留 Global 的证据。
- 真实帧 4887 从同一个新 checkpoint 将残差系数置零，输出有效域相同；启用 Local 的平均/最大绝对输出变化为 0.00815/0.83855 个候选索引。
- 默认新增 5393 个可训练参数。B=1 时，两 chart、两相机的 Local CNN 合计约 0.464 G MAC，96 层逐点残差卷积约 28.06 G MAC，此计数只包括新增卷积前向。
