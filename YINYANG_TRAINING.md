# DA-V2 Context + Yin/Yang：训练入口

沿用 `train.py --stereo`，不是另写训练框架。默认四目 `module/` 和关闭
`--use_yinyang` 时的单网格双目接口均保留。AdamW、OneCycleLR、梯度裁剪1.0、
原Sequence Loss和12轮更新均沿用现有代码；不增加LEC Loss。

## 训练中的数据流

1. `StereoDataset` 读取 cam4/cam3：CNN使用原灰度归一化输入，DA-V2使用同次光度增强的真实左RGB。
2. 加载已经验证的 Yin50 几何包，严格检查相机标定、配置哈希、96层候选轴和左参考输出尺寸。
3. 原共享CNN → 两网格扫描 → 共享几何局部增强 → 各自Similarity/GWC → 跨网格Warp和可学习代价融合。
4. 两套代价体共用原GEV U-Net权重，分别保存Lookup缓存。
5. 冻结ViT和可训练Adapter只提取一次；分别对齐两网格，与各自静态左CNN Context融合。
6. 原Context/State投影 → 原MotionEncoder和ConvGRU → 每轮凸上采样，候选索引始终为 `[0,95]`。
7. `YinYangTrainingGeometry.project()` 用真实射线对应关系，将每轮两网格预测投回同一左鱼眼 `320×480` 输出。
   双线性采样严格检查有效邻点；重叠处按几何质量归一化混合，单路有效完全取该路，两路无效为零。
8. 原完整Rig GT点云 → 左光心径向GT → 左鱼眼Z-buffer，沿用 `dataset_stereo.py` 的实现。
   不缩放原GT、不用Rig索引直接监督；监督还与当前候选有效性及预测投影有效性求交。

Yin/Yang使用各自分辨率，不Resize特征体。训练适配只增加最终预测的确定性几何投影。
源网格粗像素 `x` 对应凸上采样网格的 `2*x`，保留原stride-2采样中心。

## 实际张量

完整真实数据检查中，batch size 2、D=96：

| 张量 | Yin | Yang |
|---|---|---|
| 扫描/增强特征，左右各一份 | `[2,32,96,136,216]` | `[2,32,96,128,328]` |
| GWC / 融合代价体 | `[2,8,96,136,216]` | `[2,8,96,128,328]` |
| GEV | `[2,1,96,136,216]` | `[2,1,96,128,328]` |
| 静态Context | `[2,32,136,216]` | `[2,32,128,328]` |
| GRU Hidden | `[2,64,136,216]` | `[2,64,128,328]` |
| 每轮原生预测 | `[2,1,272,432]` | `[2,1,256,656]` |

每轮监督预测 `[2,1,320,480]`。Similarity / GEV Lookup 通道分别36 / 18，
Motion为128，GRU输入为160；沿用已训练DA-V2 Baseline配置。

## 几何质量与显存

使用原生像素各向异性：`dx=cos(latitude)*dLongitude`、`dy=dLatitude`，
`Q=min(dx,dy)/max(dx,dy)=1/kappa`。跨网格Q按源网格真实射线采样，
不是常数或二值Mask；GT不参与网格/Q生成。

固定网格以非持久buffer随模型移动到GPU，不在每批重新传输，也不写入每个大checkpoint；
checkpoint另存完整几何元数据和标定/配置哈希，恢复时重新校验外部几何缓存。

`--gradient_checkpointing` 对局部增强和无状态的MotionEncoder/ConvGRU/DepthHead卷积重计算，
不重做扫描、Lookup或修改U-Net BatchNorm统计。小尺寸测试验证输出及全部输入/卷积参数梯度逐值一致。
关闭开关不增加任何参数、不改变checkpoint键。

从头训练时曾在第42批定位到融合输入的有限一致性特征达到80437.92，
被权重卷积的autocast转成FP16后溢出。因此融合的两层1×1×1权重卷积保持FP32，
其余网络继续AMP；不改变融合公式、不裁剪代价，也不以NaN替换掩盖问题。
`tools/smoke_geometry_aware_fusion.py` 额外验证大代价输入在AMP下的前后向均有限。

## 启动与恢复

在服务器容器 `stereo-fisheye-omni-ds-dev` 的代码目录执行：

```bash
GPU_ID=0 BATCH_SIZE=2 bash train_yinyang_omnithings.sh --gpu_memory_fraction 0.90
```

默认实验名 `omnids_dav2_yinyang_scratch30_bs2_20261117`。不加载旧双目实验的
`e27.pth`，也不传 `--pretrain_ckpt` 或 `--restore_ckpt`；CNN、Adapter、Context Fusion、
几何增强、代价融合、GEV和GRU等全部可训练模块使用自身初始化，从epoch 1开始训练30轮。
Frozen DA-V2 Backbone仍使用官方预训练权重并保持冻结。
Optimizer、scheduler、AMP scaler及更新计数均从头建立，不覆盖已停止的旧初始化实验。

```bash
GPU_ID=0 BATCH_SIZE=2 RESTORE_CKPT=/绝对路径/当前实验_eN.pth \
  bash train_yinyang_omnithings.sh --gpu_memory_fraction 0.90
```

恢复会严格加载模型、optimizer、scheduler、scaler和更新计数，并检查几何一致性。
`--restore_ckpt` 与 `--pretrain_ckpt` 不能同时使用。
仅当显式设置 `PRETRAIN_CKPT` 时，脚本才会迁移旧双目权重；默认不设置。

默认训练8704帧、验证512帧，官方1024张测试集保留；30轮和LR等参数均可通过末尾CLI覆盖。
`--gpu_memory_fraction` 是共享GPU上的分配预算，不是占用虚拟空闲显存。
正式实验按用户指定保持batch size 2。GPU 0其他任务结束后，分配预算调整为90%；
原55%预算下的批大小测试属于历史配置。

## 检查与监控

```bash
python tools/smoke_yinyang_training.py
python tools/smoke_dav2_yinyang.py --device cpu
tail -f checkpoints/omnids_dav2_yinyang_scratch30_bs2_20261117/steps.csv
```

每步CSV包含Loss、索引MAE、有效监督比例、优化器更新次数、AMP scale、耗时及峰值显存。
每轮 `epochs.jsonl` 保存验证指标/耗时；checkpoint保存原训练状态、几何元数据和训练参数。
`YINYANG_TRAINING_VALIDATION.json` 记录本次实际测试和批大小选择结果。
少量训练/验证只证明流程可训练，不构成精度提升结论。
