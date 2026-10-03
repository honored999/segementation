# H2Former 外部监督预训练与 Dataset501 固定五折微调

本文件描述工程接口；真实实验、完整模型/GPU 和数据协议全部 **SERVER VALIDATION PENDING**。本地合成测试不证明预训练有效、患者独立或服务器兼容。跨数据集患者/扫描重叠未核实是正式实验阻塞项，不能用文件名或图像哈希代替患者映射。

## 数据与接口合同

- `--stage pretrain` 必须同时指定外部 `--dataset-id` 与用户提供的 `--splits-file`，拒绝 Dataset501。默认源为转换工作树约定的 Dataset508_ISLES2022DWI；不得自行改成其他源。
- 外部 split 格式为 `[{"train": ["synthetic-a"], "val": ["synthetic-b"]}]`，这里仅是格式示意，不是实际病例或划分。可用单 fold；拒绝空集合、重复、交叉、非法/路径穿越 ID 和越界 fold/病例。
- `--patient-map` 是 case ID 到患者 ID 的 JSON 对象；所有会话应映射到同一患者。若 dataset.json 已有 `patient_map`，会自动检查；两份映射冲突时拒绝。没有映射不代表已证明患者独立。真实 split 和映射由用户/服务器提供，不生成随机比例或真实划分。
- `--stage finetune` 使用原 Dataset501 五折；可省略 `--dataset-id`。显式 split 必须与原五折规范内容相同，不覆盖原文件。
- 不给 stage 时保留旧入口、optimizer 默认 SGD 和旧 plan_hash；来源参数必须配合明确 stage。历史未带来源字段的 checkpoint 仅允许旧默认入口恢复，不能当外部预训练断点。新阶段 checkpoint 不允许经旧入口绕过来源检查。
- 新阶段仅支持基础版 `h2former / single_output`、单通道、背景/病灶 0/1、plans patch_size 精确 512×512；dataset.json 必须位于数据根或其父目录。b2nd 使用源自身获准 plans 和既有归一化结果，不重复归一化。
- raw 在线算法原有固定面内 spacing 为 0.4892368018627167；新阶段 raw plans 必须匹配此值，否则明确拒绝。不同 spacing 的外部源应使用源自身的 b2nd，不照搬 Dataset501 spacing。本实现不改预处理、采样、增强或几何算法。
- `--source-version` 记录用户核实的数据版本/转换 provenance 引用；未提供时读取 dataset.json 的 source_version，否则记录 UNVERIFIED。此状态允许合成工程检查，但真实运行前必须补齐协议，不能凭此开展正式实验。

resolved_config 和 checkpoint 的 `source_identity` 记录 stage、dataset_id、版本引用、规范 split、所选 train/val 病例、plans、dataset.json、患者映射的 SHA256、源类型、规范根路径、通道/标签/patch 合同。规范 split 摘要忽略对象键序和病例顺序，保留 fold 顺序与集合归属。根路径可迁移，但以上内容身份必须相同；路径变化不证明文件内容没变，服务器仍须核实数据版本。旧 plan_hash 不无条件增加来源字段；新来源合同单独校验。

## 权重初始化与完整恢复

`--pretrained-checkpoint` 只用于 finetune，与 `--resume` 互斥。只接受带完整外部 pretrain 来源合同的基础 H2Former 同构 checkpoint。先验证 schema、身份、通道、标签、完整 keys、shape、dtype、layout 和所有张量有限性，再 `strict=True` 加载全部参数和持久 buffers；拒绝 Lite/光电/部分权重/丢 head/encoder-only。初始化失败不会部分修改模型，优化器不参与迁移。

初始化不恢复源 optimizer、scheduler、epoch/global_step、best Dice、早停计数或 Python/NumPy/Torch RNG。目标训练状态重新开始。`initialization_provenance` 记录来源绝对路径、文件 SHA256、模型合同及源 dataset/split/plans 身份。这里的业务 weights-only 与 `torch.load(weights_only=True)` 的受限反序列化选项是两回事；读取复用现有 schema 校验和 NumPy 白名单，并补入正式 checkpoint RNG 所需 NumPy 数组类型。

同阶段 resume 校验来源、旧 plan_hash、policies、模型和初始化 provenance，再预检恢复字段、优化器组/张量及 scheduler/RNG。正式恢复只反序列化一次；应用失败时回滚模型参数/buffers、优化器状态/组/LR、scheduler 属性和 Python/NumPy/Torch RNG，并重新抛出原异常。回滚自身失败时保留原异常并附注/链接回滚错误。备份成本为一份模型 state_dict 的 CPU 张量和一份既有优化器状态（原设备），另有少量组/scheduler/RNG；不复制完整模型实例。优化器兼容性预检可能短暂分配源优化器的设备张量，完整 H2Former/GPU 峰值仍需服务器核实。目标 finetune resume 从目标 checkpoint 与同目录 resolved_config 恢复 provenance，不重新加载外部源文件；源文件移走后仍能恢复。终止早停 checkpoint 不再进入训练/selection。不要在 resume 时重新传 pretrained-checkpoint。所有同一运行参数，包括 patient-map/source-version/数据源类型，要保持一致。

新阶段输出必须不存在，resolve 后不能等于、位于或包围原始/预处理数据、源 checkpoint 目录、记录的源数据目录、受保护源码或已有实验目录。标准 nnUNet_raw / nnUNet_preprocessed 父根也保护。已有 checkpoint_*.pth 或 resolved_config.json 的父目录不能包含新运行。resume 仅使用 checkpoint 所在的对应阶段目录，核验其中 resolved_config，绝不清理或覆盖旧目录给新实验腾空间。自定义服务器布局中的其他未声明数据/实验路径必须在执行前核实，代码不能发现未提供的任意路径。

## 配置检查命令

以下是服务器 Bash 模板，在仓库根执行。所有路径与版本变量由用户提供；输出变量必须是不存在的独立目录。这里不含 `--confirm-run`，不会启动训练。Windows 本地优先 `D:\Anaconda\Scripts\conda.exe run -n newconda`；服务器 conda 路径由用户核实。不下载安装依赖。

外部监督预训练（外部验证只用于源早停，不能利用 Dataset501 标签挑源权重）：

```bash
conda run -n newconda python -m standalone_nnunet2d.formal_train \
  --stage pretrain --dataset-id Dataset508_ISLES2022DWI \
  --splits-file "$EXTERNAL_SPLITS" --patient-map "$EXTERNAL_PATIENT_MAP" \
  --source-version "$EXTERNAL_VERSION" \
  --preprocessed-root "$EXTERNAL_B2ND" --plans "$EXTERNAL_PLANS" \
  --output-root "$PRETRAIN_RUN" --fold 0 \
  --model h2former --supervision-mode single_output --optimizer adamw \
  --epochs 1000 --batch-size 12 --device cuda:0
```

Dataset501 权重初始化微调（fold 0 示例）：

```bash
conda run -n newconda python -m standalone_nnunet2d.formal_train \
  --stage finetune --dataset-id Dataset501 --source-version "$TARGET_VERSION" \
  --preprocessed-root "$TARGET_B2ND" --plans "$TARGET_PLANS" \
  --pretrained-checkpoint "$PRETRAIN_RUN/checkpoint_best.pth" \
  --output-root "$FINETUNE_FOLD0" --fold 0 \
  --model h2former --supervision-mode single_output --optimizer adamw \
  --epochs 1000 --batch-size 12 --device cuda:0
```

随机初始化匹配对照（除初始化和独立输出位置，所有条件与微调一致）：

```bash
conda run -n newconda python -m standalone_nnunet2d.formal_train \
  --stage finetune --dataset-id Dataset501 --source-version "$TARGET_VERSION" \
  --preprocessed-root "$TARGET_B2ND" --plans "$TARGET_PLANS" \
  --output-root "$CONTROL_FOLD0" --fold 0 \
  --model h2former --supervision-mode single_output --optimizer adamw \
  --epochs 1000 --batch-size 12 --device cuda:0
```

同阶段恢复目标微调；预训练恢复同理，须保留其 dataset-id、splits-file、patient-map、source-version，替换为源阶段 output 与 latest checkpoint：

```bash
conda run -n newconda python -m standalone_nnunet2d.formal_train \
  --stage finetune --dataset-id Dataset501 --source-version "$TARGET_VERSION" \
  --preprocessed-root "$TARGET_B2ND" --plans "$TARGET_PLANS" \
  --output-root "$FINETUNE_FOLD0" --fold 0 \
  --resume "$FINETUNE_FOLD0/checkpoint_latest.pth" \
  --model h2former --supervision-mode single_output --optimizer adamw \
  --epochs 1000 --batch-size 12 --device cuda:0
```

dry-run 检查 CLI、plans 元数据、split/患者映射、dataset.json 合同、摘要和输出边界。带源 checkpoint 时受限读取 payload，并共享实际初始化的完整来源 schema、基础 H2Former 身份、pretrain 阶段及来源合同校验；阶段 resume 校验保存的 source_identity 和 initialization_provenance schema（含 root）。这些错误在数据计算、模型构建、设备迁移和输出创建之前拒绝。读取可能分配 CPU 张量；不构建模型、不恢复状态、不读取病例影像、不测 GPU、不训练、不写输出。完整模型 keys/shape/dtype/layout/有限性与目标匹配检查在确认执行后、严格加载前进行；实际加载重新读取并校验当前文件，预检不授权跳过后续检查。dry-run 不验证 optimizer/scheduler/RNG 的完整恢复兼容性。打印配置不能证明病例存在、真实几何正确、b2nd/plans 实际配对或 GPU 可运行；dry-run 也不能证明预训练效果。

真实执行只能在另获授权并完成下面清单后，向同一命令追加 `--confirm-run`。本文件的命令未执行过真实训练。

## 固定五折与早停

分别为 fold 0–4 提供不存在的五个 FINETUNE 与五个 CONTROL 输出目录。可用布局 `/experiments/h2former-transfer/<run-id>/finetune/fold_0` … `fold_4` 和 `/experiments/h2former-transfer/<run-id>/control/fold_0` … `fold_4`，共享父目录不能已有 checkpoint/resolved_config。各 fold 使用原固定 split，完全相同的 source 权重；只有跨数据集患者独立已核实时才可共享这份源权重。逐折执行上述命令并设置 fold 和独立输出，不生成/替换五折。

保持 1000 epoch 最大日程、250 iterations/epoch、既有 batch size/损失/增强/采样和优化器策略。每 10 个完成 epoch 使用全部 prepared-case 验证体积，所有切片一次，病例 Dice 宏平均，不重复归一化。epoch 100 开始计耐心；patience=10 次选择检查，min_delta=0.001 只控制耐心重置。严格 Dice 改善即保存 best，包括小于 min_delta 的改善；latest 持久化完整状态。

prepared-case 选模 Dice 不是原始患者空间正式指标。fold 0 仅开发筛查；预训练效果必须通过匹配的固定五折、原始全体积 OOF 正式评估。现有 validate_fold 的源空间评估/几何语义未改变。不要把预训练选择或微调在线验证分数当最终性能。

## 服务器待验证清单

全部 **SERVER VALIDATION PENDING**，执行前另获授权：

1. ISLES2022 转换 provenance、许可/数据版本/排除项与 dataset.json 一致。
2. DWI 单通道、标签 0/1、病例配对、方向/spacing/origin/形状和真实几何。
3. 用户提供的患者级 train/val 与患者/会话映射，外部验证不借用目标标签。
4. 外部源与 Dataset501 患者及扫描重叠：未核实即正式实验阻塞，不用文件名/哈希替代。
5. 源自身 plans、预处理/b2nd、512×512、归一化/spacing 配对；不套用目标 spacing。
6. prepared-case 选择空间、覆盖所有验证病例与切片，正式源空间 OOF 另行验证。
7. 一批真实前向/反向、loss/梯度有限性；完整 H2Former state keys/buffers 转移。
8. GPU 显存、吞吐、CPU/RAM 资源；任一相关资源预计达到 80% 则降级或延后。
9. 新输出的双向边界、所有自定义原始/预处理/旧实验位置和 symlink，旧数据只读。
10. 外部 pretrain → strict target 初始化 → 新目标状态 → 完整 target resume 链，记录来源 provenance；终止早停不再训练。

独立 Level 3 审查由主代理手动安排，针对未提交 diff、untracked 文件哈希与精确测试快照。实现者自查不是独立审查；没有审查 PASS 和服务器证据，不宣布正式实验就绪。
