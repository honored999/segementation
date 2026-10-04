你是用户另行手动派发的独立 Level 3 只读 reviewer。建议 LunaMax；如实际型号/推理强度不可验证，明确写 unknown。你未实施本任务，须独立检查，不把实施者自查算作独立审查。

禁止自动创建子代理、独立任务、reviewer/fixer/validator；禁止嵌套代理。不得修改任何源代码、测试、原 plans、记忆、site-packages、数据；不得 commit/push/merge/reset/stash/clean。只在任务笔记中新建一个审查报告文件，或直接返回报告。用户要求命令不要沙箱试跑，直接申请批准；优先 D:\Anaconda\envs\newconda\python.exe，不安装/升级依赖。每次测试重新 CPU/RAM/GPU/VRAM 预检，>=80% 降为最轻可信验证；不得干预其他进程。真实训练/评估/数据/真实 checkpoint/CUDA 性能均未授权。

完整读取规格：
E:\study\研一\work14-图像分割\segementation\docs\nnunet_upernet_selected_stages_manual_task.md
以及审查目标根 AGENTS.md；按需读取 project-memory、level3-review、medical-experiment-integrity、scientific-experiment-integrity、test-validation、resource-aware-testing 技能。

审查目标（新隔离工作树，不切换/reset）：
E:\study\研一\work14-图像分割\segementation-upernet-configurable-stages
branch=codex/upernet-configurable-stages
base/HEAD=8a41ce3b0d4e2c980c9e9063a60b055b0a841d89
未提交/未合并。启动重查 git worktree list --porcelain、branch、HEAD、git status --short --untracked-files=all。先用 SHA256 核对 .task-notes/upernet-selected-stages/snapshot_manifest.json 的所有 files（该文件自身哈希由实施 HANDOFF 提供，避免自引用）；包括新增/untracked 文件，不能仅看 git diff。ignored_files 仅记录生成 pycache/pytest cache，也应确认无隐藏生产/测试逻辑。若源快照漂移，明确记录并只对实际检查的快照给结论，不能沿用旧测试声称通过。

只读基线：
E:\study\研一\work14-图像分割\segementation-upernet-es-topk10
codex/nnunet-upernet-es-topk10 @ 8a41ce3b0d4e2c980c9e9063a60b055b0a841d89
保护工作树：E:\study\研一\work14-图像分割\segementation
feat/h2former-stroke @ d717a7a82d30a1d23fbb2c3aab4425c1708806fa
只读参考：E:\study\研一\work14-图像分割\segementation\.worktrees\upernet-no-stage7
codex/upernet-no-stage7 @ f1b9ca87e28bda0c8941b639d87349b7d9012338
不得修改/合并这些工作树；NoStage7 不是本次方案。

精确实施范围：
- nnunet_ext_trainers/nnUNetTrainerMixins.py（tracked diff：严格选择 helper + 原 decoder 最小 N 层泛化）
- nnunet_ext_trainers/nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping.py（新增）
- nnunet_ext_trainers/create_upernet_stage_plans.py（新增）
- nnunet_ext_trainers/tests/test_upernet_selected_stages.py（新增）
- nnunet_ext_trainers/README_upernet_selected_stages.md（新增）
- .task-notes/upernet-selected-stages/（设计/验证/快照/交接）
- 分支 .project-memory/{STATUS,GOALS,NEXT,LOG}.md（未集成、审查 PENDING 的 branch-local 记录）
原 test_official_upernet_trainers.py 未修改。其他 Trainer、模型、训练策略/循环未修改。

先阅读 design.md、validation.md 和 validation/{focused-final,baseline-final,affected-final}.json，再独立核对下面风险点：
1. resolved ConfigurationManager.configuration 参数读取、继承、缺键自动四层、显式非法不 fallback。参数不可传入 arch_kwargs；支持 stage0、2..n_stages、[1,3,5,7]/[1,3,5,6]/[2,3,4,5]/全八层。拒绝 bool/float/string/null、重复/乱序/越界和任一轴相邻尺度不严格下降；实际 forward BCHW/通道/batch/空间验证未削弱。
2. 完整官方 encoder 仍保留且执行，未选 stage/最末 stage 不删除。N-1 top-down、最深 PPM(1,2,4)、FPN128、N*128 fusion、exact-size bilinear align_corners=False、raw single output；原四层模块名/state tensor keys/shape/初始化/融合数值路径维持。新网络 extra_state 是独立 Trainer 的有意协议变化，旧 Trainer 无该字段。
3. 最敏感：新网络 get_extra_state/root load pre-hook 是否在任何参数复制前拒绝同形状不同选层、缺失/损坏/类型伪装身份；新 Trainer.load_checkpoint 是否在继承 loader 的 logger/my_init_kwargs/optimizer 写入前校验；是否真正覆盖官方 predictor.initialize_from_trained_model_folder 直接 network_weights load 和多折切换。核对本机 nnunetv2==2.8.1 实际源码，而非只看测试。检查 DDP key prefix、OptimizedModule unwrap/compile 与 strict 加载边界。官方初始化只加载第一 fold，后续 fold 在 predictor fold loop 加载时拒绝；不得要求修改第三方 predictor 以提前扫描全部 fold。CUDA/DDP 运行没有已验证证据。
4. accounting 是否包括完整 encoder、实际所选 decoder，并用真实 Conv2d 几何覆盖奇数和 anisotropic。实施有意只给新 wrapper 修复 encoder 库的 floor-based 奇数计数，legacy wrapper 保留原行为；确认这符合旧行为兼容及本次 new-model accounting 规格。独立判断 frozen baseline 和 forward-hook 证据是否有效。
5. plans CLI 仅 UTF-8 JSON，独立文件名/plans_name，每组合独立结果目录；原 data_identifier/配置/继承/architecture/training 字段保留、不预处理。resolved path 同路径/已有输出/命名不匹配/歧义 JSON/非法配置均拒绝。exclusive x creation 防止竞争覆盖，写前验证，源 bytes 保持。无真实数据访问/写入，无 --force。
6. loss/early stop/optimizer/scheduler/split/augmentation/batch/patch/foreground/inference policy 与基线一致。网络身份改变不能继续旧 checkpoint。文档 stage0 八层 final concat 8倍只是结构算术，不能误称显存/性能实测。

实施者工程证据：focused 55 passed, baseline 25 passed, affected external suite 105 passed，均 exit 0。首次 RED 缺新 API；JSON 歧义 RED；两次 fixture 契约错误及修复全部留存。真实 official CPU Trainer 初始化/未初始化 resume、strict checkpoint round-trip、真实 official predictor 合成目录和 multi-fold 滑窗；tiny complete eight-stage 129x129/two encoder channels backward+accounting；frozen git-show 四层 tensor/numerical 回归。不是服务器/真实训练/正式评估证据。

如独立复测必要，启动自己的资源预检，优先单项；例如在目标根执行：
& 'D:\Anaconda\envs\newconda\python.exe' '.task-notes\upernet-selected-stages\run_validation.py' 'review-unique-label' 'nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_identity_rejected_before_parameters_optimizer_and_logger'
资源安全允许且确有必要再跑 focused/affected；不得运行根/standalone 大套件仅为仪式。复测会创建 task-local evidence，不能悄悄覆盖已有证据。请记录新报告和快照之间的关系。

最终返回 PASS 或 BLOCKING（必要时 NON-BLOCKING）。BLOCKING 必须有具体文件/函数/行号、可复现证据、违反规格和最小修复范围；风格偏好/推测/已明确 deferred 的服务器性能不独立算 blocker。不修改代码，不代派 fixer。终结 HANDOFF：status=COMPLETED/BLOCKED/FAILED、role=manual independent read-only Level3 reviewer、requested/observed model/profile、reviewed absolute worktree/branch/base/manifest SHA256、PASS/BLOCKING/NON-BLOCKING、独立检查证据/测试命令退出码/资源预检、未验证 server/CUDA/DDP/性能、保护边界及文件变化、下一步。不要声称真实数据/整体实验验收通过。
