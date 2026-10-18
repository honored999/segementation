# GenSeg 联合训练：作者 U-Net 首轮运行步骤

这是 DWI 的工程试跑：验证生成器虚拟更新、分割器虚拟更新、内部验证损失反馈到架构参数的链路。不是论文数值复现，不是分割效果结论。当前仅支持全新初始化、零预训练的短试跑；旧的5000步生成器只用于核对已有3mm目标来源，不会加载为初始权重。

## 代码与输出

先把 genseg_joint_unet_20261018.zip 复制到数据电脑 C:\lijialin\HD-BET_outputs。新模块 experiments/genseg_joint 及本次 reference/GENSEG-JOINT-* 文档需先放入数据电脑的仓库根目录，同名文件夹合并，不整体替换已有 experiments 或 reference。服务器先更新代码并提交推送、本地 git pull --ff-only 的同步方式仍适用。用户数据、权重、targets 和本次结果不加入 Git。

所有试验输出继续放在 C:\lijialin\HD-BET_outputs。若 genseg_joint_trial01 已有任何输出，换 trial02 等新目录；保留失败或中断结果，不覆盖、不伪装续训。既有 genseg3mm_trial01\targets 只读复用。

## 1. 在 Anaconda Prompt / CMD 中设置路径

```bat
conda activate nnunet5090
set "REPO=C:\lijialin\segementation\.worktrees\genseg-reference"
set "nnUNet_raw=C:\lijialin\models3d\nnUNet\nnUNet_raw"
set "nnUNet_preprocessed=C:\lijialin\models3d\nnUNet\nnUNet_preprocessed"
set "SOURCE=%nnUNet_preprocessed%\Dataset501_StrokeLesion\nnUNetPlans_2d"
set "BASELINE=C:\lijialin\genseg30_derived_trial01\generator_5000steps_seed20261014_attempt02"
set "TARGETS=C:\lijialin\HD-BET_outputs\genseg3mm_trial01\targets"
set "TARGET_SHA=23d25e2b2a6d7c7ca2a270690a9d4d35116ff4f834458b48a9049ff889894a56"
set "JOINT=C:\lijialin\HD-BET_outputs\genseg_joint_trial01"
set "PYTHONPATH=%REPO%;%PYTHONPATH%"
cd /d "%REPO%"
```

本步骤不修改旧 TRIAL 变量，不影响此前的离线生成器训练结果。

## 2. 只读检查

```bat
python -m experiments.genseg_joint.cli check --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%JOINT%"
```

应输出 CHECK PASS 和 inner61/internal15/external19。检查目标掩膜缓存、文件哈希及来源，不构建神经网络，不创建输出目录，不读取患者图像/病灶数组。原始五折划分不改动；内部61/15是另外记录的固定患者分组。

## 3. 只启动1步 GPU 工程试跑

确认GPU1空闲，并等上一步通过后执行：

```bat
python -m experiments.genseg_joint.cli train --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%JOINT%" --gpu 1 --steps 1 --pretrain-steps 0 --l1-weight 100
```

使用作者完整 U-Net、生成器宽度64/8次下采样及原预处理平面，只做右侧/下侧padding，不缩放图像。没有自动5000步预算。默认种子20261018，源自本次工程协议，不冒充原论文参数。GAN L1系数在命令中明确设为100。生成器和分割器的虚拟更新、G/D/S实际权重更新采用无动量SGD；外层架构参数采用Adam（lr=1e-4、betas=0.5/0.999、eps=1e-8、无权重衰减），避免冷启动小梯度在float32的SGD中舍入为零。BatchNorm使用冻结统计值，这些是本次工程适配，详见协议。

程序会固定物理GPU的UUID并检查资源，首次整链运行的显存峰值尚需在数据电脑实测。若资源保护或显存分配拒绝，保留报错，不删保护、不改分辨率冒充同一试验；该目录留作未完成记录，下次使用新目录。

## 4. 查看证据

运行完成后检查：

```bat
dir "%JOINT%"
type "%JOINT%\steps.jsonl"
type "%JOINT%\complete.json"
```

主要文件：
- steps.jsonl：实际更新步、训练/内部验证来源、损失、架构梯度和参数变化、资源信息。
- protocol.json：内部61/15划分、缩放范围、超参数及与论文的差异。
- joint.pt：G/D/U-Net、架构参数、优化器和随机状态。
- complete.json：最后写出的完成回执及checkpoint哈希。
- failed.json：仅失败时写出；出现它不代表训练完成。

把终端输出和 steps.jsonl / complete.json 或 failed.json 发回核对。首轮检查的是梯度链和实际资源，不评价图像好坏。此冷启动版本不提供恢复训练、长训练质量保证、外部19例评估或nnU-Net训练。

## 数据与效果边界

GAN真实目标、U-Net真实训练图像、内部15例架构验证图像均使用同一套3mm处理结果；输入缩放只在内部61例拟合。外部19例图像/标签不读取。既有nnUNet预处理及3mm背景填充值沿用之前来源，不声称重新在61例内拟合了所有预处理。

旧离线生成器用了76例，新联合试验只用61例进行权重训练，因此二者不能直接作为联合训练增益的公平比较。未来需要相同内部61例、相同初始化/预算/预处理的离线对照，再在未参与优化的外部19例原始完整体积上评估。当前没有这类效果结果。

注意：外部19例仍是既有fold0开发验证集，不是全新的最终测试集。此前脑掩膜的肉眼检查已涉及其中个别病例，因此不把后续19例结果称作完全独立、无偏的最终测试。本次保证是它们不参与这次联合训练的图像读取、缩放拟合、架构优化和模型选择。



## 交付包解压命令（仅首次新增模块）

在CMD执行，不加Force；已有同名文件时先核对版本：

```bat
powershell -NoProfile -Command "Expand-Archive -LiteralPath 'C:\lijialin\HD-BET_outputs\genseg_joint_unet_20261018.zip' -DestinationPath 'C:\lijialin\segementation\.worktrees\genseg-reference'"
```

若按服务器先提交推送的方式，解压到服务器上的实际仓库根目录，然后只暂存新增模块与说明：

```bat
git status --short
git add experiments/genseg_joint reference/GENSEG-JOINT-*
git diff --cached --stat
git diff --cached --check
```

确认暂存区无数据/权重/既有splits_final.json及其它无关修改，再在服务器提交推送；数据电脑在同一分支git pull --ff-only。交付文档中的哈希标识打包时文件字节；Git正常换行转换可能改变字节哈希，应区分换行变化与实际代码修改，不重算目标数据哈希绕过检查。
