# 固定一张训练切片检查生成器能否学习

用途：从已完成GD500权重出发，保持原BN/SGD/学习率/L1/架构不变，重复同一张内部61例中的有病灶切片，观察前后结构与梯度。它是工程诊断，不是正式训练、泛化评估或BN因果试验。固定图的误差降低不能证明生成新样本合格；短预算不拟合也不能证明模型没有能力。

本地只做静态检查/独立审阅，不执行模型/pytest/CUDA。以下在数据电脑Anaconda Prompt / CMD执行。诊断不保存gan.pt/joint.pt，不能把其结果作为正式预训练权重交接。

## 1. 同步代码和路径

```bat
cd /d C:\lijialin\segementation\.worktrees\genseg-reference
git status --short --branch
git pull --ff-only
conda activate nnunet5090
set "REPO=C:\lijialin\segementation\.worktrees\genseg-reference"
set "nnUNet_raw=C:\lijialin\models3d\nnUNet\nnUNet_raw"
set "nnUNet_preprocessed=C:\lijialin\models3d\nnUNet\nnUNet_preprocessed"
set "SOURCE=%nnUNet_preprocessed%\Dataset501_StrokeLesion\nnUNetPlans_2d"
set "BASELINE=C:\lijialin\genseg30_derived_trial01\generator_5000steps_seed20261014_attempt02"
set "TARGETS=C:\lijialin\HD-BET_outputs\genseg3mm_trial01\targets"
set "TARGET_SHA=23d25e2b2a6d7c7ca2a270690a9d4d35116ff4f834458b48a9049ff889894a56"
set "PRETRAIN=C:\lijialin\HD-BET_outputs\genseg_gd_pretrain_500steps_trial01"
set "PRETRAIN_SHA=33dc65d16c4d03fafa1d2d9884d61e11f298a9621751c126b2b72c1bc08957a6"
set "PYTHONPATH=%REPO%;%PYTHONPATH%"
```

若存在未提交更改阻止pull，保留并核对，不reset。PRETRAIN必须含原gan.pt和complete.json；本地只复制回执的500_trial01不能用于运行。所有诊断输出必须全新，目录已存在则换trial02，不覆盖删除旧结果。

## 2. 先3步检查新诊断入口

```bat
set "DIAG=C:\lijialin\HD-BET_outputs\genseg_single_slice_3steps_trial01"
python -m experiments.genseg_joint.cli diagnose-single-slice --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --pretrained-run "%PRETRAIN%" --pretrained-sha256 "%PRETRAIN_SHA%" --output "%DIAG%" --gpu 1 --steps 3
powershell -NoProfile -Command "Get-Content -Raw -LiteralPath (Join-Path $env:DIAG 'diagnostic.json') | ConvertFrom-Json | Select-Object complete,completed_steps,weights_saved | Format-List"
explorer "%DIAG%"
```

只在命令正常结束、diagnostic.json顶层complete=true、请求更新数完成，且无非有限数/资源拒绝/审计错误后继续100步。资源保护保持原阈值和所选GPU范围；遇到异常保留failed.json和日志，不删除保护或更换分辨率冒充同次诊断。

选择规则固定为内部61例按标识排序的第1例及其第1张有效有病灶切片，不读取内部15或外部19的图像/标签，不重新拟合缩放。没有合适平面会报错，不暗中换病例。输入始终是同一个病灶掩膜，目标始终是同一个现有3mm DWI平面。

## 3. 新目录做100步固定切片诊断

```bat
set "DIAG=C:\lijialin\HD-BET_outputs\genseg_single_slice_100steps_trial01"
python -m experiments.genseg_joint.cli diagnose-single-slice --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --pretrained-run "%PRETRAIN%" --pretrained-sha256 "%PRETRAIN_SHA%" --output "%DIAG%" --gpu 1 --steps 100
powershell -NoProfile -Command "Get-Content -Raw -LiteralPath (Join-Path $env:DIAG 'diagnostic.json') | ConvertFrom-Json | Select-Object complete,completed_steps,weights_saved | Format-List"
explorer "%DIAG%"
```

100步从同一个原GD500重新初始化，不接续3步诊断、没有中断恢复。G/D更新使用已有PretrainEngine，架构不更新，BN统计冻结；不创建S，不改变学习率/损失。

## 4. 看哪些文件、怎样解释

- before.png和after.png：相同目标/标签/生成/叠加四联图，优先看第3栏是否出现脑轮廓和内部纹理，不能把红色标签当成生成病灶。
- diagnostic.json：完成记录、初始化身份、固定样本、前后目标与输出统计、L1误差、梯度和激活诊断等。顶层completed_steps记录完成的诊断更新数；protocol.selection是固定切片，observations.before/after.statistics.valid中normalized_l1和generated_spatial_std用于对比。无自动“学会/没学会”结论。
- steps.jsonl：每次实际更新的G/D损失与梯度及资源记录。guard_events.jsonl记录有限步间利用率等待。

保持相同保存缩放和原生H/W，不逐图增强对比度。若误差下降但空间灰度变化仍很小、图仍是平灰，优先检查是否只学到整体亮度。若前层梯度/激活很弱而后层主要变化，可支持进一步做受控BN/优化器诊断，但本次不能据此宣布其为根因。

检查单张训练图是为了定位学习链路；即使出现结构，也不代表能生成不同病例或处理空掩膜。暂不扩大联合训练预算，不将诊断输出投入nnU-Net增强训练。

把整个3步或100步诊断目录复制回本地HD-BET_outputs，提供路径即可继续只读检查。保留全部原始数据、splits_final.json、目标缓存、旧权重和旧失败目录。
