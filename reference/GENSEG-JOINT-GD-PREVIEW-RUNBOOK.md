# 查看500步G/D预训练与100步联合训练的生成结果

本入口用于定位灰图出现在哪个阶段。它不会继续训练，不改变BN、优化器、种子、数据划分或检查点。当前任务的本地验证仅静态检查/独立静态审阅；首次真实运行在数据电脑。PNG只是定性预览，不是生成训练数据集。

## 1. 数据电脑同步及环境（Anaconda Prompt / CMD）

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
set "GD_PREVIEW=C:\lijialin\HD-BET_outputs\genseg_gd_preview_500steps_trial01"
set "PYTHONPATH=%REPO%;%PYTHONPATH%"
```

PRETRAIN是数据电脑的完整已完成训练目录，内含gan.pt和complete.json。本地E盘500_trial01目前只见复制的complete.json，不能用它进行推理。若数据电脑训练目录名称不同，只改PRETRAIN，保持权重哈希与该回执一致。输出目录必须全新，已存在换trial02等，不能覆盖/删除旧结果。如果pull被本地修改阻止，保留修改并核对，不reset。

## 2. 导出500步G/D预训练的四联图

```bat
python -m experiments.genseg_joint.cli preview --phase gan_pretrain --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --run "%PRETRAIN%" --checkpoint-sha256 "%PRETRAIN_SHA%" --output "%GD_PREVIEW%" --gpu 1 --patient-limit 4
type "%GD_PREVIEW%\preview.json"
explorer "%GD_PREVIEW%"
```

程序严格核验完成回执、显式权重哈希、源/目标/固定61-15-19角色和已保存的61例缩放，只构造G并恢复G和两组架构参数，eval/no_grad推理；不构造D/S或优化器。gan.pt完整CPU反序列化会读到D张量，但不等于运行D。资源保护不变，GPU1繁忙或资源不足时停止，保留结果与错误；不能为查看而绕过保护。

启动时可能打印initialize network with normal：这是创建G后恢复保存权重的正常过程，不是重新训练。预览成功以preview.json顶层complete=true为准。默认按内部61例排序前4例，每例第一张病灶片和最多2张有效空病灶片，与旧joint预览选择一致。四栏是：真实3mm目标 / 病灶标签 / 生成DWI / 生成DWI与标签叠加。红色只是叠加标签；圆形边界可来自忽略区域屏蔽，不代表学到了脑轮廓。

## 3. 对照已有联合100步预览

数据电脑已有：

```bat
explorer "C:\lijialin\HD-BET_outputs\genseg_joint_preview_500plus100_trial01"
```

打开两个目录中相同名字的PNG，对照第三栏。例如同一病例、同一z、同一lesion/empty类别。生成前后采用同一保存缩放，不逐图增强灰度。若已有joint预览目录实际名称不同，以实际目录为准。

- GD500已是灰图、joint100也灰：优先排查G/D阶段，不能认定joint破坏了原本好的生成器。
- GD500有结构而joint100变灰：优先检查联合阶段的参数更新/BN/权重交接；这只是定位，不等于已经证实原因。
- 两者都有相似结构：继续检查PNG统一显示范围及生成病灶与标签一致性，不能凭红色叠加判断成功。

空掩膜预览的输入条件相同且当前G没有额外噪声/切片位置输入，忽略区域显示处理之前可产生相同输出；这本身不是权重加载错误的证据。当前500+100结果不合格，不把它投入nnU-Net增强训练。

把新GD_PREVIEW整个目录复制到本地HD-BET_outputs后，提供本地路径，继续做只读图像检查。暂不为诊断增加训练步数或更换固定划分。
