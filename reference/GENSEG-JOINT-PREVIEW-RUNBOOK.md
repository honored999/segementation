# 联合权重生成图像预览

用途：从已完成 joint.pt 生成少量四联图，查看真实3mm目标、病灶标签、生成DWI、生成DWI与标签叠加。仅是定性的工程预览，不是分割指标评估，也不是可直接用于训练的生成数据集。命令已与实现接口核对；测试与独立审阅证据见本次PREVIEW审阅回执。

## 数据电脑同步与路径

在Anaconda Prompt / CMD中执行。保留已有未提交修改；若git pull被阻止，先核对，不reset、不强制覆盖。

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
set "JOINT=C:\lijialin\HD-BET_outputs\genseg_joint_trial02"
set "JOINT_SHA=c45f95c73cf7c403307d349e092dd45b59e9ebfeda3201554e39a910d32c5b88"
set "PREVIEW=C:\lijialin\HD-BET_outputs\genseg_joint_preview_trial01"
set "PYTHONPATH=%REPO%;%PYTHONPATH%"
cd /d "%REPO%"
```

JOINT请指向你刚完成的联合试跑目录。如果当时使用了其它目录名，只修改JOINT，保留真实checkpoint哈希。上述SHA是用户发回完成回执中的值，本地没有读取数据电脑权重。BASELINE旧76例权重仍只用于目标来源核验，不作为本次生成器初始化。

本地开发测试优先newconda；数据电脑继续使用已通过真实CUDA试跑的环境，示例沿用nnunet5090，不安装/升级依赖。

## 运行预览

```bat
python -m experiments.genseg_joint.cli preview --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --run "%JOINT%" --checkpoint-sha256 "%JOINT_SHA%" --output "%PREVIEW%" --gpu 1 --patient-limit 4
```

输出目录必须全新。如果genseg_joint_preview_trial01已存在（含失败/未完成输出），换trial02等新名字，不删除覆盖原结果。保护原始数据、固定splits_final.json、已有targets及训练目录。

程序核对顶层complete=true的joint回执、显式checkpoint哈希、模型配置、数据来源、内部61/15/19角色及既有缩放范围，只恢复生成器和两组架构张量。不会构建D/U-Net或优化器，不进行训练。现有joint.pt整体反序列化可能把其中D/S张量读到CPU，这不等于构建或使用这些模型。

默认使用内部61例按标识排序后的前4例；每例选第一张有病灶切片及最多两张有效无病灶切片。实际张数和缺失类别记录在preview.json。内部15例和外部19例图像/标签不读取。固定选择用于复查图像问题，不是随机代表性样本或效果证明。

生成过程使用原生预处理平面，右/下padding后再裁回原H/W；输入病灶掩膜不做随机翻转。保存的训练缩放范围用于反变换和统一灰度显示，不为预览重新拟合，也不独立拉伸每张图。-1忽略区域显示黑色并从统计中排除。PNG仅显示8位图像，显示窗口/裁剪计数和来源哈希见preview.json，不保留医学NIfTI几何，也不替代原图。

## 在哪里看

成功结束后：

```bat
dir "%PREVIEW%"
type "%PREVIEW%\preview.json"
explorer "%PREVIEW%"
```

打开目录中的PNG即可看四联图，顺序为：真实3mm DWI / 病灶标签 / 生成DWI / 生成DWI+病灶叠加。以preview.json顶层complete=true为预览完成证据；失败时保留现有输出，不能手工补完成标记。所选病例全部没有有效平面时应报告失败，不以0张PNG冒充预览完成。

检查脑结构是否出现、病灶亮区是否与标签对应、空病灶切片是否出现未标注亮灶，以及条纹/棋盘格/规则块状伪影。当前只做了1步G/D预训练+1步联合更新，图片可能接近未训练输出。预览完成只证明推理/渲染链路，不证明图像质量合格或医学分割收益。先发回几张PNG和preview.json核对，再决定是否扩大训练预算。
