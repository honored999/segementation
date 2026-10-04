你是用户手动派发的独立 Level 3 复审 reviewer。建议 LunaMax；实际型号/强度不可核验则写 unknown。不得创建任何子代理/任务/reviewer/fixer/validator；不得修改生产代码、测试、plans、数据、依赖、记忆或原报告。无 commit/push/merge/reset/stash/clean。只读审查，可在任务笔记内新增唯一复审报告/测试证据，不能覆盖既有记录。命令按用户要求直接申请批准，不沙箱试跑；newconda 优先 D:\Anaconda\envs\newconda\python.exe。每次测试重新 CPU/RAM/GPU/VRAM 预检，>=80% 降为最轻可信验证。不得真实训练、正式评估、真实 checkpoint 或 CUDA/多 GPU 实验。

任务书（完整功能规格、范围和科学边界）：
E:\study\研一\work14-图像分割\segementation\docs\nnunet_upernet_selected_stages_manual_task.md
目标：
E:\study\研一\work14-图像分割\segementation-upernet-configurable-stages
branch=codex/upernet-configurable-stages
base/HEAD=8a41ce3b0d4e2c980c9e9063a60b055b0a841d89，未提交/未合并。
读取 root AGENTS 和必要 level3-review/test-validation/resource-aware-testing/scientific/medical-integrity 技能。启动核对 worktree list、branch/HEAD、status；保护 root feat/h2former-stroke@d717a7a 和 baseline codex/nnunet-upernet-es-topk10@8a41ce3 与 NoStage7@f1b9ca8 均只读，禁止切换/reset/合并它们。

原独立审查：.task-notes/upernet-selected-stages/review_level3_20261003.md，BLOCKING P1，未选最后 stage 导致 default DDP 第二次 forward reduction 错误。原 manifest snapshot_manifest.json 的外部锚点仍为：
8655a1725ce8ae8da79d3b7d581525a07d70034ff6262c57ca6971e779eb1878
原 manifest/审查报告/证据已保留，不能拿旧锚点当当前源码快照。

新复审对象：.task-notes/upernet-selected-stages/snapshot_manifest_ddp_fix.json。
先核对实施者 HANDOFF 单独提供的此新 manifest 自身 SHA256，再逐项核对 files、bytes、SHA256（包含新增/untracked）；自身排除以避免自引用。ignored_files 是生成 caches，核对没有隐藏生产/测试代码。changes_since_original 列出相对原快照变化；source_sha256 应匹配新 fix-ddp-focused-final / fix-ddp-affected-final，旧 55/105 测试仅证明原快照。

阅读 ddp_fix_validation.md、ddp_fix_delta.patch（旧 Trainer/tests 文本按旧 SHA256 重建确认）、README 的 DDP 章节和两份新 final JSON。修复生产范围只有新 nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping.py：
1. 在 super.initialize() 完成官方 compile/SyncBatchNorm/optimizer/DDP 后、首次 forward 前，仅 is_ddp 且最深选择早于末 stage 时，释放旧 wrapper/reducer，再用相同 module/parameters/device_ids/output_device/process_group 和官方默认选项重建 DDP(find_unused_parameters=True)。包含最后 stage 的组合、单设备和原 Trainers 保持原 wrapper/path。完整 encoder 仍执行，未选末层参数保留 requires_grad=True，grad=None，不冻结/删层/造零梯度，不复制训练循环。
2. load_checkpoint 在规范化 module. prefix 的 network_weights 校验后，把此同一 mapping 放进 checkpoint shallow copy 再委托继承 loader。修复实际 plain DDP 时外部 wrapper keys 使原 loader 保留 prefix、传入 inner model 的问题。未改原 Trainer 的 loader；caller checkpoint 不覆写；identity mismatch 必须仍在参数/optimizer/logger 写入前拒绝。
新增测试仅 test_upernet_selected_stages.py：CPU Gloo rank0/world_size1，FileStore ASCII 路径，真实新 initialize 集成 + real DDP reducer；为可在 CPU 执行，只有 CUDA-dependent parent initialize 被替换为 compile-then-DDP fixture，修复/guard 不 mock。plain/eager compile；[0,3] control/[0,2] 两次 backward；末层两次执行、unused grad None、optimizer/parameter IDs preserved、weakref 检查旧 wrapper 已释放/control unchanged。两个 plain/eager real DDP checkpoint roundtrip/prefix/mismatch cases。

重点独立核对：
- P1 实际 reduction 缺陷关闭，DDP graph/reducer/hooks 生命周期正确；重建初始化前后参数、optimizer、device/process group、sync/compile 顺序无改变；包含最后 stage 不增加 unused detection。
- unused detection 与真正未用参数 None 梯度保持一致，未向原训练/权重衰减语义加入dummy梯度、冻结或删层。
- DDP 与 eager compile checkpoint 保存/加载/前缀及 identity guards，官方 predictor 的 direct load 路径仍完整。输入 checkpoint shallow copy/normalized mapping 不能引入绕过检查或误写。
- shared Mixins/其他 Trainer/旧 regression source 与原已审快照一致；全功能 focused 和 affected 证据匹配精确当前 source；文档明确 CPU shim 和实际 CUDA/multi-rank 证据边界。
- 原独立 PASS 范围不会自动证明新加载路径；独立选择必要重点复测，不必重复无变化大套件。

实施者 fresh evidence：fix-ddp-final-green 6 passed；fix-ddp-focused-final 61 passed；fix-ddp-affected-final 111 passed，均 exit0。每条命令新 preflight，均<80%。RED 与中间失败完整保留：2 control pass/2 reduction fail；id复用错误修正为weakref；真实 plain DDP prefix failure 单项RED后修复。不是独立复审 PASS。

若复测，在目标根申请执行例如（每次 runner 会预检；新唯一 label，PYTHONDONTWRITEBYTECODE=1 和 no cacheprovider）：
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\newconda\python.exe' -B '.task-notes\upernet-selected-stages\run_validation.py' 're-review-ddp-unique' 'nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_selected_trainer_ddp_two_iterations' 'nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_selected_trainer_real_ddp_checkpoint_identity' '-p' 'no:cacheprovider'
另择原 identity/predictor/checkpoint 节点复测以验证加载回归（先新预检）。不得关闭其他进程。不要把 CPU initializer shim 说成真实 official CUDA Trainer 测试。CUDA/SyncBatchNorm/NCCL/multi-rank、多 GPU、真实医疗数据、全尺寸八层性能仍 NOT RUN；它们不单独构成推测性 blocker。

返回 PASS / BLOCKING / 可选 NON-BLOCKING，并对原 P1 明确 CLOSED 或 STILL_BLOCKING。任何 blocker 给具体文件/函数/行、证据、违反规格与最小修复范围。不可实现/代派fixer。终结 HANDOFF：status=COMPLETED/BLOCKED/FAILED，role=manual independent Level3 re-review，requested/observed profile、branch/base/新manifest自身hash、结论/原P1状态、独立测试精确命令退出码/资源、source snapshot漂移情况、新报告/证据路径、未验证范围、保护/Git边界。没有真实训练/正式性能的 PASS 声称。
