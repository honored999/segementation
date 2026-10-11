# nnU-Net 探索实验：排除15例 A/B 与筛选70例五折

归档已准备的脚本、服务器 Trainer 快照和期望元数据；运行脚本已修复外部 Trainer 来源检查，训练协议、比较脚本和启动器保持原包行为。私有病例划分和真实配置不发布。本包尚未在真实服务器运行验证。

## 更新现有服务器检出

在 Windows 服务器 Anaconda Prompt，更新现有实验仓库检出：

```bat
cd /d "你的现有服务器仓库根目录"
git status --short
git pull --ff-only origin codex/nnunet-exclude15-ab
conda activate nnunet5090
cd tools\nnunet_exclude15_ab
```

更新前 git status --short 应为空；有修改先人工保留/处理，不 reset、覆盖或自动 stash。私有忽略文件不会随 pull 分发。服务器沿用 nnunet5090、nnunetv2==2.8.1，不随意升级依赖。

## 私有配置

首选从已有私有交付包复制 original_splits.json 和 experiment_config.json 到本目录，两者均被 gitignore。不要公开真实病例 ID 或私人服务器路径。

否则，只读复制已有 Dataset501 的五折 splits_final.json 为 original_splits.json；复制 experiment_config.example.json 为 experiment_config.json，并在本地填写服务器路径和既定名单。excluded_cases 示例为空，必须填写恰好15个不同 ID：原 fold 0 训练集13个、验证集2个。不要重新选例、随机重建原划分。示例路径是通用占位符，不能直接运行。预训练预期字节数370907859继承原包，须核对私有交付身份；大小不替代 checkpoint 身份和兼容性验证。source_hashes 是三个所附 Trainer 的准确 SHA-256。

附带 expected_plans.json、expected_dataset.json；服务器元数据必须精确匹配，原划分成员必须匹配私有快照。不得为了通过检查修改源元数据。raw 影像、标签和 canonical splits 保持只读；官方训练可能生成/解包既有预处理缓存。

## 固定协议

原95例，既定排除15例。A训练原76例；B从其中排除13例，训练63例。A/B使用同一剩余17例在线验证、早停及最佳 checkpoint 选择。最终在原19例完整体积空间评估，包括排除名单中的两例验证病例，真实 ID 只留在私有文件。19例仍是开发验证集，不是独立测试集。

继承原95例 cached plans 和预处理统计，不针对剩余80例重新规划或预处理。两组使用同一外部 ISLES checkpoint_best.pth，从外部预训练重新微调，不续训 Dataset501 checkpoint；manifest 记录预训练 SHA-256。旧日志存在 best/final checkpoint 歧义，不能称为旧 checkpoint 精确复现；新 A 的17例选模型规则也不同于旧19例选择。

沿用 TopK10 + early stopping 快照：batch12、patch512×512、每轮250训练/50在线验证迭代、lr0.01、foreground0.33、deep supervision、最大1000轮、300轮warmup/100轮patience。损失、网络、优化器和增强保持原包设置；仅本进程 Trainer 实例重定向 do_split。

## 运行与输出

```bat
call check_only.cmd
call run_AB_seed0.cmd
```

先确认 A/B 均显示 CHECK_OK。check-only 只读元数据，不加载影像、模型或 checkpoint 内容，也不证明真实运行兼容。启动器依次运行 A、B、比较；正式启动资源占用达到80%会退出，不干预其他进程。私有配置选择 GPU，A/B保持一致。

输出位于 results_parent\seed_0\A 和 B：
- experiment_manifest.json：配置、成员、预训练/受保护元数据哈希、状态和资源。
- evaluation_summary.json：原19例主结果、剩余17例辅助结果。
- evaluation19_case_metrics.csv：Dice/IoU（0–1）、TP/FP/FN（体素数）、selection标记。
- seed_0\paired19_A_vs_B.csv：逐例 Dice 和 B−A 百分点。
- 各 run 的 nnUNetTrainerTopK10EarlyStopping__nnUNetPlans__2d\fold_0：checkpoint；其 evaluation_original19\validation：预测和完整体积 summary。

私有衍生产物放在独立 results_parent，不放入源码或原始数据，不提交 Git。恢复模型时保留 trainers，并令 nnUNet_extTrainer 指向该目录。

已有同种子/arm目录拒绝覆盖或续训；配对CSV排他创建。中断保留 FAILED_OR_ABORTED manifest，不纳入结果。先解决原因，再换新的独立 results_parent 重跑整对；不删除旧输出伪装同一run。训练完成而验证失败时也不能盲目继续训练，应另行准备只评估方案。

重复时同步修改启动器 SEED 与 PYTHONHASHSEED（例如均为1或2），每个种子配对跑相同 A/B。主进程种子固定，但官方增强 worker/CUDA 保持非确定行为，不保证位级复现。

## 解释边界

15例由更早 OOF 指标选择，且跨训练配置选例；这是探索性开发证据，不能证明标注质量差、最终泛化或临床改善。先看相同原19例 B−A，再看17例辅助结果；多个配对种子只能提供稳定性线索。

未来另设计随机排除13个训练病例的 C 组，保留相同选择/评估队列和配对种子，以区分定向排除与样本数减少影响。当前脚本未实现 C。

本地验证覆盖合成成员、指标、输出路径守卫，以及 nnU-Net 实际外部模块查找和清理逻辑；不导入模型或真实医学数据。真实 checkpoint 加载、CUDA、增强、早停、服务器运行及完整体积数值尚未验证；原包归档与来源检查修复分别完成独立审查；Git 换行属性固定 Python 文件为 LF，保证 Windows 检出后的 Trainer 哈希一致。测试通过不是正式实验认可。

## Trainer 来源检查报错后的重跑

nnU-Net 2.8.1 在查找外部 Trainer 后会清理新导入模块。此时对 Trainer 类调用 inspect.getfile 会抛出 TypeError，尽管该类来自 Python 文件。本脚本提前保留预期 Trainer 模块，并严格核对模块来源路径与返回实例的类对象身份；三个 Trainer 源码快照及 source_hashes 保持原样。

对于已复制到服务器工具目录的旧包，优先在现有仓库中 git pull --ff-only 更新完整工具目录；手动更新时必须同时替换 run_experiment.py 并放入 cv70.py（旧 A/B 指标导出也依赖它）。保留私有 experiment_config.json、original_splits.json 和 trainers。先保留 FAILED_OR_ABORTED 的旧输出，在私有配置中将 results_parent 改为一个新的独立输出目录，再依次运行 check_only.cmd 和 run_AB_seed0.cmd。同一配对仍沿用既定 seed、划分和训练配置。不要删除旧结果或修改 canonical splits 来通过检查。

来源检查的本地回归通过，只能证明该加载与校验问题得到覆盖；真实服务器的 checkpoint、CUDA 和完整训练仍须另行验证。

## 本地轻量验证命令

在仓库根目录，先检查 CPU、RAM、全部 GPU/VRAM，再使用本地 newconda 执行轻量测试：

```powershell
D:\Anaconda\envs\newconda\python.exe -B -m unittest discover -s tools/nnunet_exclude15_ab/tests -p "test_*.py" -v
```

测试只使用合成 subject 名称和临时小文件，不读取私有配置或划分。来源回归仅导入 nnU-Net 的类查找工具，不导入 Trainer 模型或 checkpoint。本公开仓库中的示例只有通用路径和空排除名单；实际配置与 original_splits.json 由操作者在本地复制为忽略文件。

## 按历史五折结果筛选70例并重新五折

此方案独立于上面的15例 A/B，旧 A/B 保持原行为。在原95例完整五折 OOF 汇总中按前景 Dice 从低到高排序，同分按病例 ID 排序，排除最低25例，剩余70例。原先最低15例包含在25例中；这次允许排除病例来自任意旧折，不再固定旧17例验证集。

对保留70例排序后，以 split_seed=0 做一次可复现随机打乱，固定五个互斥14例验证集，各自训练集为其余56例。每例恰好在一折验证、其余四折训练。新划分保存在独立的私有 cv70_protocol.json；原95例 splits_final.json 和原始影像不修改，也不物理删除病例。

五折均从同一个外部 ISLES checkpoint_best.pth 重新微调，训练 seed=0，使用现有2D TopK10+EarlyStopping、batch12、patch512、lr0.01、250/50迭代、深监督、最大1000轮、300轮warmup及100轮patience。继承原95例 plans 和预处理统计，不针对70例重新规划。每折在本折14例上早停和选 checkpoint_best，然后导出14例原始完整体积预测和指标。

### 服务器运行

在现有干净仓库和 codex/nnunet-exclude15-ab 分支上 git pull --ff-only 更新。将私有 cv70_protocol.json、experiment_config.cv70.json 放在本工具目录，保留旧 experiment_config.json/original_splits.json。新配置的 results_parent 指向独立的新五折输出目录；不要复用 seed_0/A/B 的路径。

确认私有协议后，在本工具目录运行：

```bat
conda activate nnunet5090
python cv70.py check --protocol cv70_protocol.json --splits original_splits.json
call run_cv70_seed0.cmd
```

需要只检查某一折而不加载模型时：

```bat
python run_experiment.py --config experiment_config.cv70.json --cv70-fold 0 --seed 0 --check-only
```

输出位于 results_parent\seed_0\fold_0 到 fold_4；每折有 experiment_manifest.json、evaluation_summary.json、evaluation14_case_metrics.csv。五折汇总为 seed_0\oof70_case_metrics.csv 和 oof70_summary.json。只需复制这两个汇总及五个 manifest 即可核对完整性和多指标；HD95/LCD须另用预测影像与GT评估，现有CSV不能计算。

运行前核对新配置的服务器 raw/preprocessed/checkpoint/plans/GPU 路径。启动器先检查全部五折的元数据和成员，再按0到4折串行训练；每折必须显示56train/14val。运行中出现错误就停止，保留已完成/失败折的 manifest，不自动覆盖或续训。

五折完整结束后仅从这五个最佳模型各自的14例验证结果汇总70例，每例一次，得到探索性 OOF70结果。所有折 COMPLETE、元数据保持原样、配置/预训练权重/划分相同才允许汇总。检查未完成折不能当作完整五折结果。

### 解释范围

该70例是依据同一批病例的历史模型分数筛选的开发队列，且本折验证集用于早停/最佳checkpoint选择。五折 OOF 可以描述该筛选队列上的表现，但不是独立测试或无偏泛化估计。不能将它直接与原95例平均分、此前17/19例单折或fold0模型预测全部95例的分数作改善结论。至少应将旧95例 OOF 指标筛到同一70例作描述性参照，并明确模型/训练/划分差异。低Dice不能直接诊断标签质量。
