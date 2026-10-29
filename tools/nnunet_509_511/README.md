# 509 / 511 fold0 顺序实验

只新增本目录三个文件；部署时将整个目录放在服务器仓库的 tools/nnunet_509_511/。workflow.py 用 parents[2] 定位仓库，复用根目录评估器和 nnunet_ext_trainers，不新增 Trainer 或指标实现。

## CMD 启动

已核实的环境激活命令：

    call "C:\ProgramData\anaconda3\Scripts\activate.bat" nnunet5090

BAT 检查 CONDA_DEFAULT_ENV；未激活 nnunet5090 时调用以上命令，路径缺失或激活失败即退出。已激活时沿用环境，helper 还核对 sys.executable 是否来自 CONDA_PREFIX，拒绝其他 Python。

在仓库根目录运行，默认仅检查文件名、JSON、split、源码/元数据哈希和 checkpoint 文件状态，生成计划与收据：

    tools\nnunet_509_511\run.bat
    tools\nnunet_509_511\run.bat plan

默认模式不导入 torch/nnunetv2，不读取图像头或体素，不加载/哈希大 checkpoint，不抽指纹、不预处理、不训练、不评估。importlib.metadata 仅读取已安装包的版本元数据。需要真实服务器路径及已安装包才能生成完整计划；失败会保留已创建的输出根和 receipt.json。

执行必须显式指定 execute，并由用户选择物理 GPU。例如，确认物理 GPU 1 可用、任务管理器当前 CPU 为 6% 后：

    tools\nnunet_509_511\run.bat execute 1 6

也可不提供 CPU 数字；此时记录 CPU unavailable，不使用 WMI/psutil 伪造门控：

    tools\nnunet_509_511\run.bat execute 1

GPU 也可传完整 GPU UUID，或通过 NNUNET_PHYSICAL_GPU 指定物理索引/完整 UUID：

    set NNUNET_PHYSICAL_GPU=1
    tools\nnunet_509_511\run.bat execute

脚本不会自动挑选 GPU。启动时将用户指定的物理 GPU 解析为 UUID，通过 CUDA_VISIBLE_DEVICES 固定映射，子进程中该卡是逻辑 cuda:0。每个计算命令前记录 nvidia-smi 的 UUID、利用率、已用/总显存，以及 Windows GlobalMemoryStatusEx 的 RAM 使用率；任一相关读数达到 80% 即停止。GPU 读数不可用会停止；CPU 只有手工提供时才作为门控。这是瞬时检查，不能保证 batch12 的训练峰值内存足够，用户需在启动前选择空闲且容量足够的卡。长任务期间的手工 CPU 数字不会自动刷新；脚本不排队、不监控抢卡、不结束其他进程、不自动降 batch 或改 plans。

## 四组固定顺序

| 顺序 / 子目录 | 数据集 | 现有 Trainer | 初始化 |
|---|---|---|---|
| 01_509_CEDice | 509 | nnUNetTrainerEarlyStopping | 从头 |
| 02_509_TopK10 | 509 | nnUNetTrainerTopK10EarlyStopping | 从头 |
| 03_511_TopK10_scratch | 511 | nnUNetTrainerTopK10EarlyStopping | 从头 |
| 04_511_TopK10_ISLES508 | 511 | nnUNetTrainerTopK10EarlyStopping | 官方 -pretrained_weights |

四组均为完整网络、保留 DS；不使用 depth / noDS 变体，不冻结层。标准组沿用官方 CE+Dice；TopK10 组沿用现有 Dice+最难 10% CE。早停复用公共 Trainer：max1000、patience100、delta1e-4，监控 ema_fg_dice，不设最低训练轮数。首个完成轮次建立基线，之后未达到 delta 的轮次立即累计；达到 delta 时清零，因此新训练始终无足够改善时最早在完成第 101 轮后停止。checkpoint 的状态键和已保存计数保持兼容；旧 checkpoint 续训保留原计数，不追溯重算过去 warmup 期间的轮次；已触发早停的 checkpoint 续训仍只执行启动/结束清理。种子、优化器及官方 checkpoint_best 严格改善选择规则保持不变。

微调源：

    C:\lijialin\models3d\nnUNet\nnUNet_results\Dataset508_ISLES2022DWI\nnUNetTrainerTopK10EarlyStopping__nnUNetPlansFrom501__2d\fold_0\checkpoint_best.pth

要求源文件大小为已核实的 370907859 字节。默认仅记录 stat；execute 才流式计算 SHA256，第四组启动前再次校验，若源文件已变化则拒绝微调。官方 load_pretrained_weights 跳过名称含 .seg_layers. 的权重，因此迁移编码器/解码器，所有分割头重新初始化，随后全网络训练。优化器、epoch、调度器和早停状态不从源模型恢复；不传 --c。

## plans、split 与只读输入

原始输入根固定：

    C:\lijialin\models3d\nnUNet\nnUNet_raw
      Dataset509_CI1DWIHigh
      Dataset511_CI1DWIHighB0BET3mm

从上述 508 实验目录的 plans.json / dataset.json 读取来源。沿用源 2d512、batch12、8 stage、约 0.489 mm spacing、ZScoreNormalization、use_mask=false 及其余所有实际配置。只按官方 plans 迁移规则修改 dataset_name、plans_name、各配置 data_identifier；读写器固定为与来源一致的 SimpleITKIO。新 plans 名称为 nnUNetPlansFrom508。没有调用自动 planner，不能把这些配置称作目标数据集自动规划的最佳配置。

两套原始 splits_final.json 必须逐字节一致，511 的 SHA256 必须为：

    cfcad46bf1f47d397cf7ea8b5c3a40c2a49db77767eb343c6e3a92e3e88831d4

检查 5 折、相同 128 个安全 ID、各折 train/val 不相交且无重复、每例只验证一次、fold0 精确 100/28；数据目录必须精确对应全部 128 个单通道 *_0000.nii.gz 和标签 *.nii.gz 文件。精确 ID 由已固定哈希的 split 定义，不重建划分。execute 在任何训练前把同一份 split 原始字节复制进两套独立缓存，每次训练前再次核对。 官方预处理器还需读取缓存中的 dataset.json；execute 在任何指纹提取或预处理前，以独占创建模式将各目标 raw 数据集各自的 dataset.json 原始字节复制进对应新缓存，不使用 508 的 dataset.json。

execute 还对 fold0 的 28 例 509 / 511 标签逐个计算流式 SHA256；任一字节不同立即失败，不解码或修复标签。该检查要求字节一致，重压缩但体素相同也会拒绝。父任务仍需核实输入来源/头信息；该检查不能替代科学来源审查或患者分组审查。

每个目标数据集先运行官方 fingerprint，再调用官方 preprocess 使用迁移后的 plans；不向 raw 或共享 nnUNet_preprocessed 写入。接口以父任务已核对的服务器安装版为准：

    from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import extract_fingerprint_entry
    # -d ID -np 2 --verify_dataset_integrity --no_pbar
    from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import preprocess_entry
    # -d ID -plans_name nnUNetPlansFrom508 -c 2d -np 2 --no_pbar
    from nnunetv2.run.run_training import run_training_entry
    # ID 2d 0 -tr TRAINER -p nnUNetPlansFrom508 --val_best
    # 仅第四组附加 -pretrained_weights SOURCE_BEST

每个入口由当前环境 sys.executable -c 调用。nnUNet_extTrainer 指向仓库现有 nnunet_ext_trainers；同一目录显式加入 PYTHONPATH，支持现有 Trainer 兄弟模块的导入。fingerprint/preprocess 进程 2、DA 2、官方导出进程 2，ITK/OMP/MKL/OpenBLAS/NumExpr 线程 2。

## 输出、评估与失败

每次调用新建以下根目录，带微秒时间戳且拒绝已有根目录：

    C:\lijialin\HD-BET_outputs\results\nnunet509_511_fold0_YYYYMMDD_HHMMSS_ffffff\

plans、split 缓存、预处理、临时文件、运行时缓存（含 torch 编译/模型缓存、CUDA、Matplotlib 等）、各组结果、预测、指标、日志和收据均位于该根目录下：

    plan.json                         全部命令、关键环境、源/目标 plans
    receipt.json                      状态、版本、文件哈希、设备/资源及步骤退出码
    nnUNet_preprocessed\<Dataset>\    隔离缓存，共享给对应数据集的两组
    01_509_CEDice\nnUNet_results\...   其余三组各自独立的 nnUNet_results
    <组>\nnUNet_results\<Dataset>\<Trainer>__nnUNetPlansFrom508__2d\fold_0\validation\
    <组>\metrics\                     case_metrics / summary_metrics
    <步骤>.log                        官方输出实时同步到终端和日志，保留进度与七指标表
    comparison.csv                    四组全部成功后才生成

官方训练的 --val_best 在训练后用 checkpoint_best 导出恢复到原始完整体积空间的 validation 预测。每组结束后先检查预测恰为固定 fold0 的 28 例文件，再调用现有 evaluate_segmentation_metrics.py，四组统一使用原始 509 labelsTr，传 --f2-mode paper。现有评估器检查 size/spacing/origin/direction；若 511 未恢复到 509 原始空间会直接失败，不自动重采样或修复。

comparison.csv 原样摘录各组现有 summary_metrics.csv 的七列：Dice (%)、IOU (%)、F2 (%)、AVD (%)、LCD、Rec (%)、HD95 (mm)，保持已有单位、宏平均和无效值含义，不另算指标。已有 TP/FP/FN 仍保留在各组 summary 中。paper F2 沿用 5TP/(5TP+4FP+FN)。只有四组都完成且每组 n_cases=28 / f2_mode=paper 时才写对比表。

任何命令非零、数据/几何/资源检查失败即停止，保存输出和 FAILED 收据，并保留子命令的退出码；不自动重跑、续训、删除或覆盖。再次调用会创建另一个根目录，重复实验须由用户重新明确执行。plan 输出不是实验结果；四组结果属于固定 fold0 开发比较，不能冒充五折 OOF、独立患者验证或正式临床证据。

交付仅做静态核查；未运行本脚本、BAT、依赖导入探针、预处理、训练、评估或测试。服务器执行与独立审查由父任务负责。
