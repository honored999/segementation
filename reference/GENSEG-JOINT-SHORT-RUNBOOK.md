# GenSeg 增加训练：先3步验证，再500+100步观察

状态：本地仅做源码/语法检查及独立静态审阅，不执行pytest、模型或数值运行；真实连续多步验证按用户要求放到数据电脑。本次独立Level3静态审阅PASS；运行结果尚未验证，见GENSEG-JOINT-MULTISTEP-REVIEW.json。此预算是本项目的短程探索设定，不是论文默认值，不承诺图像质量或分割收益。保持原95例固定五折和fold0的76/19，内部61权重训练/15架构优化不变。

## 同步与通用路径（Anaconda Prompt / CMD）

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
set "PYTHONPATH=%REPO%;%PYTHONPATH%"
cd /d "%REPO%"
```

本地仅静态检查，使用newconda做AST语法解析而不导入模型；数据电脑沿用已验证的CUDA环境nnunet5090；不要仅为环境名升级依赖。若git pull或输出目录冲突，保留已有修改/结果，核对后再处理，不reset/覆盖。以下所有输出目录均必须全新，已存在则换attempt02等后缀。旧5000步生成器只作来源核验，不能加载为这轮61例训练初始化。

步间修正仅在同步后释放本进程未使用的PyTorch缓存，再沿用原资源检查；固定显存分配上限不超过启动时允许值。不会清除模型/优化器活跃张量、不改80%阈值、不自动重试忙碌GPU。若GPU利用率采样仍记为繁忙，保留报错与资源记录，先核对服务器实况再决定；不把本地静态通过当作服务器已可稳定多步。

## 1. 真实GPU连续3步预检

```bat
set "PRETRAIN=C:\lijialin\HD-BET_outputs\genseg_gd_pretrain_3steps_trial01"
set "JOINT=C:\lijialin\HD-BET_outputs\genseg_joint_3steps_trial01"
python -m experiments.genseg_joint.cli check --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%PRETRAIN%"
python -m experiments.genseg_joint.cli pretrain --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%PRETRAIN%" --gpu 1 --steps 3 --l1-weight 100
```

只有上一步成功且complete.json顶层complete=true、steps=3，才继续：

```bat
type "%PRETRAIN%\complete.json"
for /f %H in ('powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $env:PRETRAIN gan.pt)).Hash.ToLowerInvariant()"') do set "PRETRAIN_SHA=%H"
python -m experiments.genseg_joint.cli check --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%JOINT%" --pretrained-run "%PRETRAIN%" --pretrained-sha256 "%PRETRAIN_SHA%"
python -m experiments.genseg_joint.cli train --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%JOINT%" --gpu 1 --steps 3 --l1-weight 100 --pretrained-run "%PRETRAIN%" --pretrained-sha256 "%PRETRAIN_SHA%"
type "%JOINT%\complete.json"
```

确认joint的steps.jsonl有连续1/2/3三条成功记录，顶层complete=true且steps=3；每步G/D/S损失和梯度有限，架构变化非零。资源拒绝、CUDA OOM、数值错误均保留原输出/错误，不能删除保护/降分辨率冒充同次实验。不把预检3+3结果当作训练效果；后面的500+100是新目录的全新训练，不是续接3步权重。

## 2. G/D预训练500步

确认3+3在真实GPU通过再执行：

```bat
set "PRETRAIN=C:\lijialin\HD-BET_outputs\genseg_gd_pretrain_500steps_trial01"
set "JOINT=C:\lijialin\HD-BET_outputs\genseg_joint_100steps_from500_trial01"
python -m experiments.genseg_joint.cli check --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%PRETRAIN%"
python -m experiments.genseg_joint.cli pretrain --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%PRETRAIN%" --gpu 1 --steps 500 --l1-weight 100
```

每步是一次单切片采样更新，不是1个epoch；500步不保证完整遍历所有患者/切片。采样、SGD率、L1、冻结BN、目标来源和61例缩放协议保持原值，未因灰图调整参数。

## 3. 从该500步G/D初始化，联合100步

只有500步完成回执通过再执行：

```bat
type "%PRETRAIN%\complete.json"
for /f %H in ('powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $env:PRETRAIN gan.pt)).Hash.ToLowerInvariant()"') do set "PRETRAIN_SHA=%H"
python -m experiments.genseg_joint.cli check --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%JOINT%" --pretrained-run "%PRETRAIN%" --pretrained-sha256 "%PRETRAIN_SHA%"
python -m experiments.genseg_joint.cli train --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%JOINT%" --gpu 1 --steps 100 --l1-weight 100 --pretrained-run "%PRETRAIN%" --pretrained-sha256 "%PRETRAIN_SHA%"
type "%JOINT%\complete.json"
```

新建U-Net和架构Adam状态；预训练500步和联合100步分别记数。仍只在整个run正常完成后保存gan.pt/joint.pt；没有中断续训/周期恢复能力。中途失败不会自动变成完成结果，下次保留失败目录并新建尝试。交接完成G/D到联合是初始化，不是恢复joint.pt。

## 4. 同批切片预览

```bat
set "PREVIEW=C:\lijialin\HD-BET_outputs\genseg_joint_preview_500plus100_trial01"
for /f %H in ('powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $env:JOINT joint.pt)).Hash.ToLowerInvariant()"') do set "JOINT_SHA=%H"
python -m experiments.genseg_joint.cli preview --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --run "%JOINT%" --checkpoint-sha256 "%JOINT_SHA%" --output "%PREVIEW%" --gpu 1 --patient-limit 4
explorer "%PREVIEW%"
```

interactive CMD使用单%H，写入.bat文件则用%%H。不得用重算哈希绕过回执不匹配。

保持相同前4个内部61病例、切片选择和显示规则，对照之前1+1灰图。观察是否出现脑结构、病灶与标签关系、空掩膜一致输出和伪影；红色是叠加标签，不是生成病灶。无论图像是否更清楚，这次预算比较不能证明去脑外带来提升，亦不能作为nnU-Net增强数据合格证据。未使用外部19例进行参数/预算选择，也未运行原始全体积分割评估。
