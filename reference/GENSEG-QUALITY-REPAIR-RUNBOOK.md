# GenSeg 质量修复：第一阶段运行与验收

本轮范围：新版本 G/D 预训练、固定切片诊断、epoch 覆盖和预览兼容。联合策略须等待真实数据学习门槛；新预训练不得交给旧冻结 BN 的联合路径。工程测试通过不代表灰图已解决，也不代表论文复现或合成数据可用于 nnU-Net。

## 已有证据（2026-10-19 只读复核）

- GD500 预览保存协议：500 actual updates，SGD + eval BN；之后联合运行100 actual updates。均非500轮。
- 原 GD500 SHA256：`33dc65d16c4d03fafa1d2d9884d61e11f298a9621751c126b2b72c1bc08957a6`。
- 原 joint100 SHA256：`51ae3e080b1580516709f1853ac175af354a92bd8ce845e435e880ca3a9d0b82`。
- 已有固定片诊断 `genseg_single_slice_100steps_trial01/diagnostic.json`：complete=true、completed_steps=100、weights_saved=false。有效区 L1 0.547455→0.496053，生成 std 0.002434→0.002484；病灶 L1 0.157034→0.179835。after.png 仍近乎平灰。损失下降不等于结构学习恢复。
- 上述为历史失败证据，不重复运行相同旧配置，不改旧回执或输出。PNG 与回执足以核实这些现象，不能替代新策略真实运行。

## 保护范围与方法身份

95例固定五折及 fold0=76/19 不变；GenSeg 权重训练/缩放仅 inner61，内部15保留架构验证角色，外部19不参与本阶段。保持原生512网格、ignore=-1及独立输出。真实图像、GT、3mm targets、旧权重和原微调均不改。

Adam lr=2e-5、betas=(0.5,0.999)、L1=100是旧工程基线；作者本地源码也使用 Adam，但本轮不声称这些数值均由论文规定。Adam + train BN + epoch 是组合候选修复，不作单因素因果归因。mask-only 输入仍缺少脑解剖层面条件。

`runtime.py` 是当前 HD-BET 微调回执的共享哈希依赖。不要在正在微调的 `genseg-reference` checkout 执行 pull；不要回填旧哈希。服务器单独创建修复 checkout：

```bat
cd /d C:\lijialin\segementation\.worktrees\genseg-reference
git fetch origin && git worktree add -b codex/genseg-quality-repair C:\lijialin\segementation\.worktrees\genseg-quality-repair origin/codex/genseg-quality-repair
cd /d C:\lijialin\segementation\.worktrees\genseg-quality-repair
git status --short --branch
git rev-parse HEAD
conda activate nnunet5090
```

分支或目录已存在时先核对，勿覆盖；后续只在修复 checkout 用 `git pull --ff-only`。保留数据电脑已工作的 CUDA 环境，本地 pytest 使用 newconda。

```bat
set "REPO=C:\lijialin\segementation\.worktrees\genseg-quality-repair"
set "nnUNet_raw=C:\lijialin\models3d\nnUNet\nnUNet_raw"
set "nnUNet_preprocessed=C:\lijialin\models3d\nnUNet\nnUNet_preprocessed"
set "SOURCE=%nnUNet_preprocessed%\Dataset501_StrokeLesion\nnUNetPlans_2d"
set "BASELINE=C:\lijialin\genseg30_derived_trial01\generator_5000steps_seed20261014_attempt02"
set "TARGETS=C:\lijialin\HD-BET_outputs\genseg3mm_trial01\targets"
set "TARGET_SHA=23d25e2b2a6d7c7ca2a270690a9d4d35116ff4f834458b48a9049ff889894a56"
set "GD500=C:\lijialin\HD-BET_outputs\genseg_gd_pretrain_500steps_trial01"
set "GD500_SHA=33dc65d16c4d03fafa1d2d9884d61e11f298a9621751c126b2b72c1bc08957a6"
set "PYTHONPATH=%REPO%;%PYTHONPATH%"
cd /d "%REPO%"
```

## 验收顺序

1. check 只读检查源/目标；新固定切片诊断先检查状态与保存后 eval 输出。冷启动和 GD500 恢复是不同对照；Adam 状态从零开始，不能称为 resume。
2. 比较同片、同缩放、同预算的有效/病灶/非病灶 L1、动态范围/std与目标中位数常数基线；对子区明确使用全有效区常数还是子区中位数。结构和语义需人工检查；std增大也可能是噪声。
3. 先状态正确和单片学习，再检查固定 inner61 有病灶/空mask多片。不能用内部15或外部19观感制定门槛。
4. 状态或质量失败立即停止扩预算，保留 failed/诊断文件。complete只表示流程完成，不代表图像质量通过。5000更新、500轮及联合长训练均未自动授权。
5. 新联合 actual/virtual BN 与优化器交接是下一阶段；旧高阶链保留，不机械全局 train 或 SGD→Adam，不把单步SGD代理称为实际Adam精确导数。

## 数据电脑 CMD 命令（运行前先固定预算）

下列诊断固定100更新，仅提供候选检查命令，本轮没有启动GPU训练。两个诊断使用同一固定inner61有病灶层、同一缩放；cold检查从零学习，gd500检查旧失败权重恢复，二者不能互相替代。若目录已存在换trial02，不覆盖。

先只读检查：

```bat
set "DIAG_COLD=C:\lijialin\HD-BET_outputs\genseg_repair_cold100_trial01"
python -m experiments.genseg_joint.cli check --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%DIAG_COLD%"
```

新策略冷启动100更新：

```bat
python -m experiments.genseg_joint.cli diagnose-single-slice --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%DIAG_COLD%" --gpu 1 --steps 100 --strategy adam_trainbn_epoch_v1 --initialization cold
```

从原GD500恢复100更新（fresh Adam，不继承SGD状态）：

```bat
set "DIAG_RECOVERY=C:\lijialin\HD-BET_outputs\genseg_repair_gd500_recovery100_trial01"
python -m experiments.genseg_joint.cli diagnose-single-slice --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --pretrained-run "%GD500%" --pretrained-sha256 "%GD500_SHA%" --output "%DIAG_RECOVERY%" --gpu 1 --steps 100 --strategy adam_trainbn_epoch_v1 --initialization gd500
```

只在固定片状态/学习检查及人工语义检查通过后，才考虑以下单轮、最多100更新的覆盖工程试跑；100不足一轮时完整轮数必须仍为0，不能当成已训练一轮。此处不是5000更新或长训练授权。

```bat
set "PRETRAIN=C:\lijialin\HD-BET_outputs\genseg_repair_epoch1_cap100_trial01"
python -m experiments.genseg_joint.cli pretrain --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%PRETRAIN%" --gpu 1 --steps 100 --epochs 1 --strategy adam_trainbn_epoch_v1
```

检查complete.json与protocol.json中的actual_steps、total_valid_planes、completed_epochs、当前轮进度、唯一覆盖率及BN/Adam策略。只有命令正常结束且complete=true后读取精确checkpoint SHA：

```bat
for /f %H in ('powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $env:PRETRAIN gan.pt)).Hash.ToLowerInvariant()"') do set "PRETRAIN_SHA=%H"
set "PREVIEW=C:\lijialin\HD-BET_outputs\genseg_repair_epoch1_cap100_preview_trial01"
python -m experiments.genseg_joint.cli preview --phase gan_pretrain --run "%PRETRAIN%" --checkpoint-sha256 "%PRETRAIN_SHA%" --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%PREVIEW%" --gpu 1 --patient-limit 4
```

交互CMD用 `%H`，保存为.bat时改为 `%%H`。哈希必须与原完成回执匹配，不能手动回填回执。预览固定inner61排序前4例及既有有病灶/空mask规则，同尺度显示。新预训练进入旧`check --pretrained-run`/`train --pretrained-run`应明确报不兼容；联合短试跑暂不执行，待新联合策略独立实现审阅。

## 2026-10-19：重复 eval 检查报错后的重试

服务器cf62a91在before probe报`Repeated eval output changed`，尚未执行训练更新。代码中原先第一次G前向启用梯度、第二次`no_grad`，不是相同执行条件；后续补丁统一两次eval的梯度模式并保留逐位相等及全部状态纯度检查。若仍失败，错误会报告最大绝对差、dtype/device及梯度模式，用于继续定位；不先放宽容差或改变训练确定性设置。CPU回归不能证明服务器CUDA已通过。

仅在修复checkout更新，保留原失败输出目录：

```bat
cd /d C:\lijialin\segementation\.worktrees\genseg-quality-repair
git status --short --branch
git pull --ff-only
set "DIAG_COLD=C:\lijialin\HD-BET_outputs\genseg_repair_cold100_trial02"
```

然后重跑上文冷启动诊断命令；若trial02已存在，选择尚不存在的trial03。不要删除/覆盖旧目录，不在微调checkout执行pull。参考split文件若曾按核实的原字节换行为CRLF，则不属于随机重划分；不要用reset/重建split来消除它的状态提示。

## 重复前向数值契约修订（服务器已提供幅度证据）

服务器在两次相同eval/grad条件下报告float32输出最大绝对差`7.450580596923828e-09`。前一次统一上下文并未消除该差异；不能因此认定模型状态改变。PyTorch说明数学相同的浮点运算不保证逐位相等：[官方数值精度说明](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)。这支持使用明确数值容差，不证明某个CUDA kernel是唯一原因。

下一次运行前固定诊断输出重复性容差`atol=1e-7, rtol=0`，对应单位归一化float32精度量级（epsilon约1.192e-7）；仍要求shape/device/dtype/context匹配且两次输出有限。日志如实保存实际exact、within-tolerance与max_abs_diff，质量状态检查消费within-tolerance。参数、BN、优化器、随机状态和运行模式的不可变检查仍保持完全严格。没有改变训练、学习质量阈值或确定性设置。

这是面向后续独立运行的诊断数值契约修订，不追改旧失败结果。保留trial01/trial02；更新修复checkout后，冷启动输出改为尚不存在的`genseg_repair_cold100_trial03`（若已存在继续新编号），再执行原100步命令。服务器CUDA和生成质量仍需新回执确认。
