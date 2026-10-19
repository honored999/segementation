# 去脑外DWI：nnU-Net TopK10＋早停，固定fold0

## 本轮协议

- 数据：Dataset501原有95例，原固定五折不变；本次只跑fold0（76训练＋19开发验证），不使用GenSeg内部61/15划分。
- 图像：真实DWI，HD-BET2.0.1、固定既有权重、默认TTA、仅原始切片平面内扩3mm，原HD-BET掩膜为空的层保留整层，非空体积要求不变。76例复用targets的原空间脑掩膜；19例应用同一固定流程。不得用GT补掩膜。
- 原空间保留像素不变、掩膜外置零；不使用GenSeg预处理缓存中的强度填充值。标签文件完整复制，不删除病灶标签。3mm规则仍是工程候选，不能保证所有脑组织/病灶完整保留。
- 继承原Dataset501的nnUNetPlans.json和split文件。原dataset_fingerprint.json也作为固定基线兼容文件复制（官方Trainer启动时要求该文件），并在provenance另存及明确标注为继承元数据；不声称它是去脑外图像重新测得的fingerprint。只重新预处理，不重新规划patch/batch/spacing/network。去脑外可能改变nnU-Net自动非零裁剪和逐例归一化，这是输入变化的后果。
- 新nnUNetTrainerTopK10EarlyStopping：普通PlainConvUNet与原有深监督保留，Dice+最难10%CE，MAX_EPOCHS1000、MIN_TRAINING_EPOCHS300、PATIENCE100、MIN_DELTA1e-4，监控ema_fg_dice。checkpoint_best仍沿用官方选择规则。
- 外部19例已是既有开发验证集，可用于本次普通nnU-Net验证/早停，但不能称为全新独立测试集。正式比较使用原空间完整体积指标；不要用训练EMA代替最终Dice。
- 若旧基线没有相同Trainer、深监督和早停规则，需要补匹配的原始DWI对照，才能把差异归于去脑外处理。无合成数据、无GenSeg权重、无预训练权重接入。

## 1. 在数据电脑CMD同步代码并设置原始来源

这份流程在C:\lijialin电脑的Anaconda Prompt/CMD执行。不要将CMD的set命令直接粘贴到PowerShell。

```bat
conda activate nnunet5090
set "REPO=C:\lijialin\segementation\.worktrees\genseg-reference"
cd /d "%REPO%"
git status --short
git pull --ff-only origin codex/genseg-reference
set "PYTHONPATH=%REPO%;%PYTHONPATH%"
set "ORIG_RAW=C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion"
set "ORIG_PP=C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion"
set "TARGETS=C:\lijialin\HD-BET_outputs\genseg3mm_trial01\targets"
set "TARGET_SHA=23d25e2b2a6d7c7ca2a270690a9d4d35116ff4f834458b48a9049ff889894a56"
set "TRIAL=C:\lijialin\HD-BET_outputs\nnunet_hdbet3mm_fold0_trial01"
```

如果git显示有服务器本地修改，先检查并保留；不reset/强制覆盖。TRIAL若已存在任何内容，改用trial02等新目录，保留失败输出，不覆盖或伪装恢复。

## 2. 只读检查与准备完整95例派生数据

```bat
python -m experiments.hdbet_nnunet.cli check --raw-dataset "%ORIG_RAW%" --preprocessed-dataset "%ORIG_PP%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%TRIAL%" --gpu 1
```

check仅元数据/文件哈希检查，无患者图像或标签数组、模型、CUDA上下文加载，不创建输出目录；原空间几何与NIfTI缩放限制在prepare逐例验证；不启动HD-BET。通过后运行prepare：

```bat
python -m experiments.hdbet_nnunet.cli prepare --raw-dataset "%ORIG_RAW%" --preprocessed-dataset "%ORIG_PP%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%TRIAL%" --gpu 1
```

prepare会复用76例原空间掩膜，为19例运行相同固定HD-BET。新输出下的nnUNet_raw/nnUNet_preprocessed/nnUNet_results完全独立；Dataset501名称相同是为了复用原plans，不会覆盖原目录。

完成后先看终端的完成回执，以及76例训练病例的QA原图/去脑外图/保留边界（cases/<病例>/qa/zXXXX.png）。验证19例不生成供调参选择的QA图，统一固定处理规则。重点看脑顶、脑底、小脑与贴边病灶；默认不读GT数组做覆盖统计；需要时给prepare追加--audit-label-coverage，仅报告训练76例。GT覆盖统计只作审计，不能用来逐例修补或为19例调参。未确认质量前不启动长训练。失败时保留日志，勿改动原始标签/掩膜。

## 3. 切换全部三个nnU-Net根目录，仅重新预处理

```bat
set "nnUNet_raw=%TRIAL%\nnUNet_raw"
set "nnUNet_preprocessed=%TRIAL%\nnUNet_preprocessed"
set "nnUNet_results=%TRIAL%\nnUNet_results"
set "nnUNet_extTrainer=%REPO%\nnunet_ext_trainers"
```

preprocess-check确认独立根目录、完整回执、原计划/划分以及本轮预处理目录尚不存在，；随后资源守卫通过才启动一次预处理：

```bat
python -m experiments.hdbet_nnunet.cli preprocess-check --output "%TRIAL%" --gpu 1 && python -m experiments.hdbet_nnunet.cli guard --gpu 1 && nnUNetv2_preprocess -d 501 -c 2d -plans_name nnUNetPlans -np 2
```

不要重复运行预处理命令：nnU-Net官方预处理器会删除并重建已有配置输出目录。本工具的preprocess-check要求目标配置目录不存在；发生中断先保留现场检查，不删除目录重跑。不要运行nnUNetv2_plan_and_preprocess，它会重新规划。本流程沿用既有nnUNetPlans，只对新图像重算预处理数据；原split已在prepare时复制到新的preprocessed数据集目录。

## 4. 训练前核对与fold0启动

```bat
python -m experiments.hdbet_nnunet.cli ready --output "%TRIAL%" --gpu 1
```

ready仅核对文件/元数据和95例预处理文件是否齐全且非空；trainer-check在服务器实际导入并验证Trainer发现，不构建模型。ready通过后，在同一个CMD选物理GPU1（先用nvidia-smi核对UUID）：

```bat
nvidia-smi -i 1 --query-gpu=uuid,name,memory.used,memory.total --format=csv
set "CUDA_VISIBLE_DEVICES=GPU-6c48a375-dd11-ff27-ed3d-0508532a9c6d"
set "nnUNet_n_proc_DA=2"
python -m experiments.hdbet_nnunet.cli ready --output "%TRIAL%" --gpu 1 && python -m experiments.hdbet_nnunet.cli guard --gpu 1 && python -m experiments.hdbet_nnunet.cli trainer-check --gpu 1 && python -m experiments.hdbet_nnunet.cli guard --gpu 1 && nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerTopK10EarlyStopping -p nnUNetPlans --val_best
```

UUID必须与这台电脑nvidia-smi -i1的输出一致；不一致则使用实际UUID。只设置CUDA_VISIBLE_DEVICES选择GPU；-device参数不能写GPU编号。数据增强worker数固定为2只是运行资源设置；与旧对照比较需记录。GPU/CPU/RAM/VRAM启动守卫不能保证整个多小时训练始终低于阈值，运行中查看nvidia-smi与内存，不干扰他人任务。

训练首次启动不用--c或-pretrained_weights。--val_best让训练后的完整体积验证直接使用checkpoint_best，避免把默认checkpoint_final验证与best结果混在同一validation目录。

结果预计在：

```text
C:\lijialin\HD-BET_outputs\nnunet_hdbet3mm_fold0_trial01\nnUNet_results\Dataset501_StrokeLesion\nnUNetTrainerTopK10EarlyStopping__nnUNetPlans__2d\fold_0\
```

关键文件：training_log、progress.png、checkpoint_best.pth、checkpoint_final.pth、validation/summary.json及19例预测。最终还应使用仓库原有完整体积评估脚本对原始19例标签做一致指标比较；不以EMA或新裁剪空间直接代替原空间指标。19例数量与成员必须一致。

## 5. 完整体积指标

训练正常结束、validation预测完整后，在同一CMD核对19例成员，再计算原空间指标：

```bat
set "FOLDOUT=%nnUNet_results%\Dataset501_StrokeLesion\nnUNetTrainerTopK10EarlyStopping__nnUNetPlans__2d\fold_0"
python -c "import json,os; from pathlib import Path; s=json.loads((Path(os.environ['ORIG_PP'])/'splits_final.json').read_text(encoding='utf-8-sig'))[0]; p=Path(os.environ['FOLDOUT'])/'validation'; ids={x.name[:-7] for x in p.glob('*.nii.gz')}; assert ids==set(s['val']) and len(ids)==19, 'Validation case mismatch'; print('FIXED19 PASS')" && python -m experiments.hdbet_nnunet.cli guard --gpu 1 && python evaluate_segmentation_metrics.py --pred-dir "%FOLDOUT%\validation" --gt-dir "%ORIG_RAW%\labelsTr" --output-dir "%TRIAL%\evaluation_best_fold0" --f2-mode paper
```

评估脚本逐例检查尺寸、spacing、origin、direction；95例原GT中额外76例会被忽略，预测必须预先核对正好是固定19例。沿用原评估f2-mode（此处paper）；若旧基线使用standard，两边统一后比较。结果目录应尚不存在，避免同名报表被覆盖。该评估与mask质量审计不同，不按去脑外掩膜裁掉GT或预测后再评估。

## 验证边界与官方接口

本地仅AST与文本/差异静态审查和独立Level3静态审阅；未运行pytest、患者数组处理、HD-BET推理、nnU-Net预处理或训练。服务器首次check/prepare/ready成功才分别建立对应运行证据。

官方固定2.8.1接口：
- https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py
- https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/nnunetv2/experiment_planning/plan_and_preprocess_api.py
- https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/nnunetv2/run/run_training.py