# CKPD 私有 Dataset501 首轮服务器操作

本轮只支持普通 2D PlainConvUNet、Dataset501 原固定 fold 0、rank=4、4 个选中编码器卷积层；保留 TopK10、早停、SGD/PolyLR、plans、数据划分和完整体积评估。编码器仅低秩 A/B 可训练，解码器正常训练。

代码的本地验证是小型 CPU 合成验证；没有运行真实校准、训练或评估。完整生产模型显存、CUDA 数值、实际 prepared 数据和来源独立性仍需服务器预检。不要把本地测试当作医学效果。

## 1. 部署并找到源权重

把交付压缩包解压到新的独立目录，例如 `C:\lijialin\ckpd_trial_20261012`。压缩包只含源码/测试/说明，不含数据或权重；不要覆盖现有实验目录。它复用服务器已有环境，不需要安装新依赖。

在服务器 PowerShell 中执行下列只读查询；它会搜索各个独立 results 根目录，而不局限于当前 nnUNet_results 环境变量：

```powershell
Get-ChildItem 'C:\lijialin\models3d\nnUNet' -Directory -Filter 'nnUNet_results*' |
    Get-ChildItem -Directory -Filter 'Dataset508*' |
    Get-ChildItem -Recurse -File -Filter 'checkpoint_best.pth' |
    Where-Object { $_.FullName -match '\\nnUNetTrainerTopK10EarlyStopping__nnUNetPlansFrom501__2d\\fold_0\\' } |
    Select-Object LastWriteTime, FullName |
    Format-List
```

明确选一份普通 nnU-Net 的 Dataset508 fold-0 checkpoint；不使用 UPerNet/其他深度的权重，不通过通配符默认取第一份。如果没有匹配，去掉 Where-Object 那一行查看 Dataset508 的其他候选路径，再核对模型和 plans。checkpoint_best 与 checkpoint_final 是不同权重；本轮明确选择 best，并在新对照中使用同一文件。

旧六组私有微调的源 checkpoint 尚未核实。旧分数只是历史参考；要判断 CKPD 的净效果，应重新运行下文的同权重全参数对照。源数据与私有数据须经数据所有者确认患者独立；文件名不同不能证明患者不同。

## 2. 校准（以下命令使用 Anaconda Prompt / CMD）

激活以往训练用的环境（历史记录为 nnunet5090），核对 nnunetv2==2.8.1，并保持原 nnUNet_raw、nnUNet_preprocessed 指向现有数据。先检查 CPU/RAM/GPU/VRAM；不要为了让检查通过而停止其他任务。输入的 checkpoint 由你在步骤 1 明确选定。

```bat
conda activate nnunet5090
cd /d C:\lijialin\ckpd_trial_20261012
set "nnUNet_extTrainer=%CD%\nnunet_ext_trainers"
set "nnUNet_compile=false"
python -c "from importlib.metadata import version; print(version('nnunetv2')); assert version('nnunetv2') == '2.8.1'"
set "NNUNET_CKPD_SOURCE_CHECKPOINT=把这里替换为步骤1选定的checkpoint_best.pth完整路径"
set "NNUNET_CKPD_CALIBRATION=C:\lijialin\models3d\nnUNet\ckpd_calibration\dataset508-fold0-r4-k4-20261012.pth"
python tools\calibrate_nnunet_ckpd.py ^
  --preprocessed-dir "%nnUNet_preprocessed%\Dataset508_ISLES2022DWI" ^
  --dataset-json "%nnUNet_raw%\Dataset508_ISLES2022DWI\dataset.json" ^
  --plans-json "%nnUNet_preprocessed%\Dataset508_ISLES2022DWI\nnUNetPlansFrom501.json" ^
  --source-checkpoint "%NNUNET_CKPD_SOURCE_CHECKPOINT%" ^
  --output "%NNUNET_CKPD_CALIBRATION%" ^
  --configuration 2d --fold 0 --cases 8 --rank 4 --layers 4 ^
  --max-positions 256 --max-covariance-mib 512 --seed 20261012 --device cuda
```

只有校准成功且打印 artifact SHA256 后才继续。它从 source plans 的 data_identifier 找实际数组目录，读取源 fold 的固定训练样本；不使用 GT 选择病例、切片或 crop。输出已存在时拒绝覆盖。CUDA 来源权重不能用 CPU 校准入口规避 GPU 检查；GPU 编号必须与官方加载器兼容。

## 3. CKPD 首次训练

先确认新的输出路径没有同名 CKPD checkpoint。保留原 Dataset501 splits_final.json，禁止重新划分。训练显存峰值需按原 batch/patch 做服务器预检；校准通过不等于训练显存通过。

```bat
set "nnUNet_results=C:\lijialin\models3d\nnUNet\nnUNet_results_ISLES508_CKPD_ES"
nnUNetv2_train Dataset501_StrokeLesion 2d 0 ^
  -tr nnUNetTrainerCKPDTopK10EarlyStopping -p nnUNetPlans ^
  -pretrained_weights "%NNUNET_CKPD_SOURCE_CHECKPOINT%" --val_best
```

预期结果位于：
`%nnUNet_results%\Dataset501_StrokeLesion\nnUNetTrainerCKPDTopK10EarlyStopping__nnUNetPlans__2d\fold_0`

恢复同一 CKPD 运行时只使用其自己的 checkpoint，不重新加载 ISLES 权重：

```bat
set "nnUNet_extTrainer=%CD%\nnunet_ext_trainers"
set "nnUNet_compile=false"
set "nnUNet_results=C:\lijialin\models3d\nnUNet\nnUNet_results_ISLES508_CKPD_ES"
set NNUNET_CKPD_SOURCE_CHECKPOINT=
set NNUNET_CKPD_CALIBRATION=
nnUNetv2_train Dataset501_StrokeLesion 2d 0 ^
  -tr nnUNetTrainerCKPDTopK10EarlyStopping -p nnUNetPlans --c --val_best
```

checkpoint 保存合并后的常规推理权重和独立的训练恢复状态。恢复随机/优化器等状态不意味着官方多进程数据增强队列和未来 batch 顺序逐位重放。

## 4. 同权重全参数对照与评估

如果无法确认旧微调的初始化权重，就在新的独立目录运行原 Trainer；再次设置步骤 1 那个完全相同的源文件路径。其余配置和 GPU 设置与 CKPD 一致：

```bat
set "NNUNET_CKPD_SOURCE_CHECKPOINT=把这里替换为与CKPD完全相同的源checkpoint路径"
set "nnUNet_results=C:\lijialin\models3d\nnUNet\nnUNet_results_ISLES508_CKPD_control_ES"
set "nnUNet_compile=false"
nnUNetv2_train Dataset501_StrokeLesion 2d 0 ^
  -tr nnUNetTrainerTopK10EarlyStopping -p nnUNetPlans ^
  -pretrained_weights "%NNUNET_CKPD_SOURCE_CHECKPOINT%" --val_best
```

两组都在原 fold-0 的完整患者体积空间评估，比较 Dice、IoU、F2、Recall、AVD、LCD、HD95 及 TP/FP/FN；保留逐病例结果和初始化权重/校准文件 SHA256。源数据 50 例与私有数据 19 例的跨队列分数不能直接证明迁移收益。fold-0 是开发证据，不是五折 OOF 或临床结论。

详细数学、来源假设、限制和推理命令见 `docs/NNUNET_CKPD_FINETUNE.md`。独立审查与本地验证记录见 `docs/NNUNET_CKPD_REVIEW.md`。

## 5. 只验证最佳 checkpoint

如需单独验证已完成运行，沿用步骤 3 的环境和结果根目录，再执行：

```bat
nnUNetv2_train Dataset501_StrokeLesion 2d 0 ^
  -tr nnUNetTrainerCKPDTopK10EarlyStopping -p nnUNetPlans --val --val_best
```

必须同时带 `--val --val_best`；只有 `--val_best` 会进入训练。推理使用相同扩展目录和普通合并权重，详细命令见英文说明。

## 6. 训练结束后的 NumPy EMA 加载兼容性

若日志已显示 Training done，随后报 CKPD checkpoint _best_ema is invalid，旧校验可能拒绝官方 logger 保存的 NumPy 实数标量。更新 CKPD 工作树后，沿用原结果根目录和扩展目录，使用第 5 节的 --val --val_best 命令加载现有最佳权重进行验证。该恢复步骤不需要重新校准或重新训练，不修改 checkpoint 或原始数据。EMA pseudo Dice 仍是训练监控值，正式比较使用完整体积评估结果。
