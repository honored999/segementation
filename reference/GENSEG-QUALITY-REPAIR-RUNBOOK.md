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

## after-probe再次超差后的定位（2026-10-20）

服务器在437a150的训练更新循环后，after probe报告最大绝对差1.4901161193847656e-7（finite、matched context），超过固定atol1e-7。保留该失败目录、steps.jsonl、before.png及failed.json；它不是完整质量验收。用户回传failed.json已确认completed_steps=100、error_type=ValueError；100次更新已执行，失败在after观测。本诊断不保存权重，进程结束后不能从这些日志恢复模型或补造after结果。

本次不继续增大容差。作者上采样分支使用ConvTranspose2d，官方文档说明cuDNN可能选择非确定性算法。修订仅让probe使用局部cuDNN deterministic=True/benchmark=False上下文，保留原enabled/TF32设置并在正常/异常退出时恢复；实际训练engine.step使用原调用者设置。诊断协议记录这一观测计算策略；训练/优化器/BN/随机状态/atol和学习质量门槛均不变。

服务器更新后先做1步工程执行检查，新目录如`genseg_repair_cold1_deterministicprobe_trial01`，其余冷启动命令相同，仅steps=1。检查before/after、completed_steps=1和探针状态；1步不是学习质量验收，不自动扩预算。将该次结果和旧100步失败回执一起交回，再决定下一步，避免重复100次更新而丢失收尾证据。

## 下一项单因素观测：同权重的 BN running-stat / batch-stat 对照

已只读核实281d3ae的冷启动100步完整回执：complete=true、completed_steps=100、state=true；before/after重复输出差均0，cuDNN配置退出恢复。有效区L1 0.861486→0.604038，病灶L1 0.258631→0.163445，常数基线0.139193；后图有粗略结构但圆形亮边/颗粒明显，质量未通过。末步实际G损失24.3659与after eval G损失61.9690存在差距，但测量时点、BN模式和判别器前向不同，不能单独归因为BN。

下一项测量使用显式`--compare-bn-modes`，保持现有eval图和quality gate为准；对同一切片/权重额外测G的BN batch statistics，使用私有functional buffers、隔离随机状态，临时BN模式恢复，不更新真实统计、不改D或optimizer、不改变训练预算。额外batchstats仅记录同尺度统计量，不另存图片；原before/after图与质量门槛仍只使用eval。

旧诊断没有保存权重，无法事后补算同权重输出。新诊断保持原seed、目标和100次更新预算，单独目录如`genseg_repair_cold100_bn_modes_trial01`；新增项是观测，不换seed/加步数/调参，不追改旧失败结果。若batchstats明显优于running-stat eval，才支持进一步查BN运行统计/校准；若两者均差，需继续查单片拟合和条件/目标表示。两种结果都不能单独证明泛化或合成数据合格。

在数据电脑原CMD窗口保留SOURCE/BASELINE/TARGETS/TARGET_SHA/nnUNet_raw变量，修复checkout工作区干净后更新，再执行：

```bat
cd /d C:\lijialin\segementation\.worktrees\genseg-quality-repair
git status --short --branch
git pull --ff-only
set "DIAG_COLD=C:\lijialin\HD-BET_outputs\genseg_repair_cold100_bn_modes_trial01"
python -m experiments.genseg_joint.cli diagnose-single-slice --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%DIAG_COLD%" --gpu 1 --steps 100 --strategy adam_trainbn_epoch_v1 --initialization cold --compare-bn-modes
```

若新目录已存在，改trial02。完成后交回diagnostic.json，重点比较observations.after.statistics.valid.normalized_l1与observations.after.bn_mode_comparison.batchstats_statistics.valid.normalized_l1，同时保留常数基线和原质量判定。这仍是固定单片诊断；此阶段100次更新便于控制变量，完整数据覆盖训练才使用epochs。
## 冻结权重的单片BN校准诊断（2026-10-20）

上一项BN模式对照已完成100次更新：after eval有效区L1=0.581747，batchstats=0.231518；病灶L1=0.172503/0.136353。两者有效区均未优于常数0.139193；BN模式影响明显，不能据此认定是唯一质量问题。

最小设计与执行顺序：只在after观测额外做一次校准，参数/optimizer/训练预算保持原样；复制BN缓冲区，将G的BN临时设为train、momentum=1，在同一训练片条件输入上no_grad前向一次，随后全G切回eval，使用校准私有缓冲区测量。保留PyTorch running variance的unbiased语义，因此不承诺校准eval逐位等于batchstats。所有临时模式、momentum、RNG和后端均须在成功/异常时恢复，真实BN不回写；原图/统计/gate仍只取原eval。该测量不是inner61全队列校准，也不是新正式推理协议。

实现仅接入原diagnose-single-slice入口，默认关闭，开关为--compare-bn-calibration，并自动包含既有BN模式对照。扩展现有pytest用例核对手算/functional输出、私有缓冲区和异常恢复，随后运行受影响单文件；独立只读审查最终SHA后提交。无需新增训练器、checkpoint/resume或批量测试函数。
数据电脑仍需新建输出目录：旧诊断未保存权重，无法在已结束进程上补做校准。沿用原CMD环境变量，修复checkout干净后更新：

```bat
cd /d C:\lijialin\segementation\.worktrees\genseg-quality-repair
git status --short --branch
git pull --ff-only
set "DIAG_COLD=C:\lijialin\HD-BET_outputs\genseg_repair_cold100_bn_calibration_trial01"
python -m experiments.genseg_joint.cli diagnose-single-slice --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%DIAG_COLD%" --gpu 1 --steps 100 --strategy adam_trainbn_epoch_v1 --initialization cold --compare-bn-calibration
```

保持同seed/同切片/同100更新，不覆盖旧目录，不加学习率或训练预算。完成后交回diagnostic.json：原eval为observations.after.statistics；batchstats为observations.after.bn_mode_comparison.batchstats_statistics；校准后eval为observations.after.bn_calibration.calibrated_eval_statistics。比较三者有效区/病灶区L1及常数基线。校准结果只定位误差来源，不替换原FAILED工程门槛，也不能证明多患者泛化或正式合成数据合格。

## 下一项单因素观测：G目标仅保留L1重建项（2026-10-20，本轮未执行）

上一项BN校准诊断（`genseg_repair_cold100_bn_calibration_trial01`）已完成100次更新：同权重after原eval有效区L1=0.5696046、病灶=0.1540500；私有BN校准后eval=0.2556465/0.1405047；batchstats=0.2282112/0.1406347；有效区仍差于常数基线0.1391926。BN影响明显但不是唯一问题。现有诊断不保存权重、不支持resume，故不再追加BN变体。

本轮唯一实验因素是G目标的对抗项：默认`gan_l1`为`adversarial(D(fake-for-G)) + 100*valid_L1`，本轮`--g-objective l1_only`改为仅`100*valid_L1`。保留100系数，不改成裸L1，否则同时改变梯度尺度。D仍按原目标、原Adam和前向顺序训练；D(fake-for-G)前向仍然执行，使每步BN计数仍为G1/D3，只是它的对抗项不再进入G梯度。因此隔离的是“G对抗损失项”，不是删除整个GAN：不删D、不冻结D、不改现有state gate和原质量门槛。

记录解释：protocol、每步`steps.jsonl`和before/after probe都记录`g_objective`、`g_adversarial_coefficient`（1或0）、`g_l1_coefficient`=100、`g_optimized_objective`、`g_adversarial_loss`、`g_l1_loss`。l1_only下`g_loss`就是100*valid_L1（与`g_l1_loss`相等），不能与默认`gan_l1`的总loss直接比较；比较对象仍是原eval/batchstats/校准后eval的valid/lesion/background L1、最优常数基线及before/after图。`--compare-bn-calibration`仍为私有after-only观察，不回写真实BN，不替换质量门槛；`learning_quality_passed`仍不会自动置true。

固定同seed=20261018、同G/D/alpha/buffers初始化、同固定inner61训练片、同条件/目标/归一化/padding/有效区、100更新、lr2e-5、Adam betas(.5,.999)、eps1e-8、无decay、原BN策略/更新时序；不换seed、不加步数、不调参、不自动进入epochs/joint。仅支持`adam_trainbn_epoch_v1`+`cold`，`legacy`或`gd500`组合会被CLI与诊断入口拒绝。

候选命令（本轮只做实现与CPU验证，未在数据电脑执行）：

```bat
cd /d C:\lijialin\segementation\.worktrees\genseg-quality-repair
git status --short --branch
git pull --ff-only
set "DIAG_L1ONLY=C:\lijialin\HD-BET_outputs\genseg_repair_cold100_l1only_trial01"
python -m experiments.genseg_joint.cli diagnose-single-slice --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%DIAG_L1ONLY%" --gpu 1 --steps 100 --strategy adam_trainbn_epoch_v1 --initialization cold --g-objective l1_only --compare-bn-calibration
```

若新目录已存在改trial02/trial03，不覆盖。完成后交回diagnostic.json：比较observations.after.statistics（原eval）、observations.after.bn_mode_comparison.batchstats_statistics、observations.after.bn_calibration.calibrated_eval_statistics的valid/lesion L1与常数基线；检查protocol与steps.jsonl的`g_objective`/系数标记，避免把l1_only的`g_loss`与gan_l1总loss直接对比。

边界：这是固定训练片的单因素工程诊断；同片拟合改善不等于合成数据有效、不等于论文复现或泛化提升。内部15/外部19不参与本次诊断或调参。

## 诊断证据补齐：ready.json与按模式预览图（2026-10-20，本轮只实现+CPU验证）

前两项诊断共有的证据缺口是“训练前身份不可自证”和“看不到batchstats/校准eval的实际生成图”。本轮只补证据，不改网络、坐标、损失、归一化、BN策略、seed、预算或质量门槛。

**F1：不可变ready.json。** `diagnose-single-slice`在固定片已选定、实际scaling已由inner61算出、身份复审计通过之后，且在构造G/D或执行第一次更新之前，独占地新建`ready.json`；`started.json`仍然是最早日志，不延迟、不覆盖、不回填。字段：`format`、`classification`、`diagnostic_only`、`complete=false`、`stage=ready_after_fixed_plane_before_models_or_updates`、内嵌`policy`与`protocol`（复用既有seed/steps/g_objective/rates/l1_weight/scaling含fit_ids/selection/target identity与manifest/source/implementation sha256）、`started_sha256`、`same_training_slice=true`、`formal_data_identity=false`，以及`arrays`中`condition`、`normalized_target`、`label`三项的dtype=float32、shape=[1,1,H,W]（以实际数组为准；当前真实固定片为512×512，256仅为小图测试示例）、C顺序与SHA256。三个哈希直接由既有CPU数组计算，不新增病人数组读取、不做GPU传输、不消耗RNG、不改变输入。表示约定写在ready内：`condition = (label==1)`，`normalized_target`为inner61有效区线性minmax到[-1,1]且ignore置0，`label∈{-1,0,1}`，指标有效区条件为`label>=0`。`ready.json`的SHA256既纳入`diagnostic.json`的`output_logs_sha256`，也进入身份清单并纳入运行结束前的最终审计，因此写入后篡改会被检出。早期拒绝（初始资源守卫、输出目录非法等）合法地不产生ready，`failed.json`也不引用它；ready写成功之后才失败时保留ready并在`failed.json`记录其file与sha256；哈希或序列化失败必须直接报错，不得产生complete标记。

**F2：可选按模式预览图。** 新开关`--save-mode-previews`默认关闭，只作用于`diagnose-single-slice`，且必须与`--compare-bn-modes`或`--compare-bn-calibration`一起使用，否则CLI与诊断入口在读取数据、创建输出目录之前即报错。开启后只在after观测多存图：`after.png`仍是原eval；`after_batchstats.png`来自既有的batchstats前向；同时开启校准时另存`after_calibrated_eval.png`，来自既有的私有缓冲区校准eval前向。不额外做生成器前向、不重复切换模式、不额外做D前向，直接复用`plane_statistics`原先丢弃的native生成数组；窗口、原网格、有效区黑背景与label叠加与原`render`一致，标题和JSON manifest注明batchstats或private calibrated eval。这些PNG只作定性定位，不参与gate：原statistics、bn_mode_comparison、bn_calibration与quality_assessment仍全部取自原eval。关闭开关时图片集合与旧行为一致（除新增ready.json），不增加任何前向。

候选命令（本轮未在数据电脑执行，仅在cold目录上追加开关即可）：

```bat
cd /d C:\lijialin\segementation\.worktrees\genseg-quality-repair
git status --short --branch
git pull --ff-only
set "DIAG_PREVIEW=C:\lijialin\HD-BET_outputs\genseg_repair_cold100_mode_previews_trial01"
python -m experiments.genseg_joint.cli diagnose-single-slice --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%DIAG_PREVIEW%" --gpu 1 --steps 100 --strategy adam_trainbn_epoch_v1 --initialization cold --compare-bn-calibration --save-mode-previews
```

交回时独立复算三个CPU数组的SHA256，分别对照`ready.json`的`arrays`对应字段；另将`ready.json`整文件SHA256与`diagnostic.json.output_logs_sha256['ready.json']`对照，两类哈希不能互相替代。写入时序由`stage`标记及ready写入先于模型构造/step的代码和测试证据核对，不能仅凭文件存在证明。再看`diagnostic.json`的`observations.after.mode_previews`记录的文件名、模式、SHA256与显示参数是否与PNG一致。边界不变：这是固定训练片的证据补齐，不构成正式合成数据合格或泛化结论。

## 单次有界收敛检查：500更新轨迹与诊断快照（2026-10-20，本轮只实现+CPU验证）

100更新冷启动已重复多次（原eval/batchstats/校准eval有效区L1约0.606/0.232/0.257，均差于常数0.1392；外层圆边、模糊、网格仍未消除），继续重复100更新不再是新证据。本轮只改观察方式：保留现有GAN+L1设置，同权重从零训练到固定上界500，在before/100/250/500四个点记录同一片轨迹，回答“100之后固定训练片结构是否继续恢复还是平台/退化”。500只是候选上界，不是论文规定、也不是已授权预算；100→500的预算与额外probe成本属于探索性支出。旧运行未保存权重，无法续训，因此这仍是全新冷启动；旧输出与旧权重一律不动。

**开关与拒绝。** 新开关`--convergence-audit`默认关闭，只作用于`diagnose-single-slice`。仅允许`--strategy adam_trainbn_epoch_v1 --initialization cold --g-objective gan_l1`且`--steps 500`；CLI与诊断入口在读取数据、创建输出目录之前即拒绝其它组合（含legacy/gd500/l1_only/非500步）。不开启时行为、命令、文件集合、权重禁令与旧诊断完全一致。

**观测点。** before与原诊断相同；100/250/500复用既有私有probe，记录原eval、batchstats、私有校准eval的valid/lesion/background L1、常数基线、range/std与同尺度模式PNG（`step000100_eval.png`、`step000100_batchstats.png`、`step000100_calibrated_eval.png`等，按step+模式命名，后续点不覆盖）。500点即最终after：写`after.png`与既有after字段，不重复前向、不在其后追加观测。中间probe完全私有：不改参数、BN、Adam状态、RNG或后端；必须在同一资源守卫内；任一点probe失败立即停止，不再执行更新。

**轨迹与判定。** 计划在运行前固定（点数/点位置/模式），`trajectory.json`独立记录计划、各点统计、PNG与状态审计（eval重复一致性、BN计数、Adam步数、后端恢复），`steps.jsonl`连续记录1..500且只在计划点带`convergence_observation`。质量门槛`quality_assessment`仍只由before与最终原始eval决定；三个模式与三个中间点只用于定位，不作为挑选最佳模式/最佳步或替换gate的依据。训练loss不与eval混合，不把三模式合并成一条“改善曲线”。到500即停：不看图像质量决定是否继续，不自动重跑、不换seed、不进joint；状态/数值/资源失败立即终止并记录已完成步数与失败信息，不用部分状态继续优化。

**诊断快照边界。** 只有该模式在完整500步与最终状态审计通过后写一个`diagnostic_state.pt`：G/D state_dict（含真实BN缓冲区）、alpha、模型配置、source与代码身份、seed、实际步数、输入哈希/scaling/选择/协议；全部CPU detached，安全`weights_only`只读回读并与内存结构核对，文件SHA256进入`diagnostic.json`的`output_logs_sha256`与convergence记录。文件内显式`diagnostic_only`、`not_pretraining`、`not_joint_handoff`、`no_resume`、`training_receipt=false`，不命名为`gan.pt`/`joint.pt`，不写`complete.json`，不保存Adam矩或逐点权重，不扩展为checkpoint/resume系统；默认诊断仍禁止任何`*.pt`。失败不产生complete标记，可留下`failed.json`引用部分快照（标记未验证），不声称可续训。

**判定规则（预先固定）。** 逐点比较100→250→500同模式的valid/lesion/background误差与同一窗口图像；若500仍不过原工程质量门槛或结构仍不合格，则保留负结果并停止本预算，不自动扩到1000/5000。只有出现明确恢复且eval稳定，才另议固定多片检查，永不直接全量训练。单片通过不等于多患者或合成数据有效。

候选命令（500更新预算尚未授权，本轮未在数据电脑执行；需用户单独批准、独立审查、主分支提交推送后再运行；旧回执无完整更新墙钟，不编造ETA）：

```bat
cd /d C:\lijialin\segementation\.worktrees\genseg-quality-repair
git status --short --branch
git pull --ff-only
set "DIAG_CONV=C:\lijialin\HD-BET_outputs\genseg_repair_cold500_conv_trial01"
python -m experiments.genseg_joint.cli diagnose-single-slice --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%DIAG_CONV%" --gpu 1 --steps 500 --strategy adam_trainbn_epoch_v1 --initialization cold --g-objective gan_l1 --convergence-audit
```

若新目录已存在改trial02，不覆盖。交回`diagnostic.json`（before/after原eval、100/250/500三点statistics、`convergence`）、`trajectory.json`、`diagnostic_state.pt`及其SHA256；使用快照前须独立核对ready/输入哈希与快照内身份，不能仅凭文件存在证明。


快照异常语义补充：仅在本次成功独占创建文件后登记partial；写入、关闭、读回或最终元数据失败均保留文件和原异常。size/hash尽力获取，失败字段为null并附原因，不能遮蔽原异常，也不能认领已存在文件。
资源说明：保存时CPU payload与CPU readback同时存活，峰值约两份完整G/D状态，此外还有训练GPU模型与Adam；未测量真实模型内存，不提供MB或ETA估算。tiny CPU恢复测试不证明生产网络配置身份，且fixture前向不使用alpha；alpha另通过真实restore_arch接口核对全部别名。
收敛模式覆盖说明：继承policy中的after_only字段描述普通模式；收敛模式在每个计划点（含中间点）执行BN校准与mode-preview，以顶层convergence policy为准。observation.resources_before复用同一update unit更新前采样的state，不表示更新后重新测量资源；不增加资源采样或改变guard语义。
