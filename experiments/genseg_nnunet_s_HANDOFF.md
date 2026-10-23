# HANDOFF — GenSeg nnU-Net2D S trial

status: IMPLEMENTED_LOCAL_CPU_PASS / MANUAL_LEVEL3_PENDING / PRODUCTION_NOT_RUN

角色：任务书指定的independent implementation owner，本聊天直接实现；没有自动/嵌套代理。
请求：执行 `docs/GenSeg_S替换nnUNet2D_手动实现任务书_20261023.md`。
实际可核实profile：当前Codex聊天，不能验证LunaMax/DSH型号，未冒称该profile。

新工作树：`C:\Users\Windows\.codex\worktrees\genseg-nnunet2d-s-trial\segementation`。
分支：`codex/genseg-nnunet2d-s-trial`。
起止HEAD：`890ee77d34c08c528d82ebdf2b0fead76386782c`；初始干净，最终8个tracked修改、5个untracked新文件，index为空。
应用原生工作树工具选择了上述托管路径，没有在建议的E:路径覆盖/重置任何既有目录或分支。
受保护主工作区仍 `codex/genseg-reference@5472832bcd077c7e98244a8f7016b7c6ad997428`、干净；
基准quality-repair HEAD不变，仅原两份历史HANDOFF untracked。没有改它们。

## 决定及实现

显式新策略 `joint_adam_positive_nnunet2d_sgdproxy_v1`；新kind `nnunet2d_plainconv_1logit_v1`。
官方实际安装get_network_from_plans构建PlainConvUNet；绑定实际SOURCE完整plans/dataset/identity、
architecture kwargs/import项、norm、唯一参数/共享别名/闲置heads、runtime版本与源码SHA。
check仅meta layout，不分配完整生产模型；训练时实际factory随机初始化，不加载旧S。
保留plans norm/activation/depth/width/He init，只适配1logit、关闭deep supervision。
保留BCE+foreground Dice、sigmoid>=.5、single-card/full-slice原joint unit；不是默认nnU-Net Trainer。
只支持明确可审计的IN(track_running_stats=False)、无dropout的2D PlainConvUNet；其他实际plans失败，不回退reference。

G/D原GD20权重+BN+alpha，fresh全部Adam/RNG/sampler；原G/D1/3计数和Adam校验保留。
S IN无BN/running buffers、日志BN0，实际仍两次S forward。未使用deep-supervision heads保持在官方网络和
optimizer参数组内，仅精确名单允许None梯度/无moments；任何其他断链/漏moment拒绝。
共享权重校验别名一致，实际参数去重所有权；functional_call finally恢复原Parameter对象，避免共享模块污染live状态。
私有eval virtual G/S单步SGD和staged alpha Adam未改算法，非论文精确复现。

schema3保存完整描述及S state、Adam ownership/moments/steps、alpha/BN/source；旧schema2默认及旧新混读拒错保留。
S-only安全CPU reader重建实际plans网络，fixed-input eval bit-exact测试通过。
G preview及完整19推理接通，继续原support-v2、有效域/原始空间导出、CACHE/PRED/METRICS隔离、异常保留和GT后置门禁。
推理 `--s-kind nnunet2d` 显式选择；安装版S源码SHA也纳入预测/评分审计。
末轮保存，无best/早停/resume；不允许outer19挑权重/阈值/超参。

## 验证

详见 `experiments/genseg_nnunet_s_VALIDATION.md` 的精确命令、资源和完整失败修复记录。
最终guard：CPU11%、RAM67.9535%、GPU0利用9%、VRAM967/6144MiB；单线程newconda CPU、CUDA禁用。
最终 `27 passed, 4 warnings in 39.40s` / exit0：新9例（8函数类别）+旧路径18例。
真实nnU-Net tiny540参数；production proxy三步actual Adam、无S BN、私有状态、
direct/indirect非零+冻结锚点FD+断链控制、schema3保存安全读回/坏状态拒绝、
新runtime/check/save/preview和合成exact19/support-v2几何/评分准入均有证据。
第三方SciPy弃用warnings4条；未安装/升级依赖。
`git diff --check`通过；新文件也需按下表SHA一起审查（git diff默认不会显示untracked内容）。

真实服务器plans/GD20权重/完整production512网络/高阶CUDA/真实preflight/峰值/80epoch/19预测指标/G质量全部NOT_RUN。
本机tiny结果不证明5090容纳、完整网络高阶CUDA支持、合成质量或分割提升。
工程实现未被这些生产NOT_RUN项阻塞。

## 完整服务器模板与人工审查待办

完整模板在 `experiments/genseg_nnunet_s_RUNBOOK.md`，阶段check -> 新目录3-unit -> 用户检查后GD20 fresh80epoch -> final80 audit -> 可选G preview -> 新CACHE/PRED exact19 -> accept -> 原evaluate脚本 --f2-mode paper。
模板使用已记录的Windows服务器SOURCE/RAW/BASELINE/TARGETS/GD20路径，新增checkout/输出是未来路径，未声称已存在。
所有预算显式人工输入；实际峰值UNKNOWN，保留既有80%资源guard；不执行任何服务器命令。

手动Level3重点：
1. meta layout与实际SOURCE绑定、实际IN/stride/空间可用性，错误plans不得替代；
2. functional_call共享模块finally恢复原Parameter对象，virtual direct/indirect梯度与actual Adam所有权；
3. 官方闲置heads名单是否精确，moments缺席政策仅该名单，新旧schema隔离；
4. 保存/重建/固定输出、安装源码hash审计、S-only恢复和support-v2全19/geometry/GT后置；
5. 实际单卡preflight收集double-backward、显存/RAM峰值、日志与G伪影；任何协议变化交回用户决策。
主代理拥有后续Git/记忆/集成；本分支尚未合入master，不写canonical或global memory。
无stage/commit/push，无真实训练，无raw/GT/split/旧weights/results/作者源码改动，无代理。

## 精确文件SHA256（工作树快照，非commit）

| 文件 | SHA256 |
|---|---|
| experiments/genseg_joint/core.py | c1b03509f6386cd325a22b86fdc09bea1c810b9b4d8084f226bd66510c0b449e |
| experiments/genseg_joint/joint_checkpoint.py | 600e6dcb9ca4f9d8cc5aa8eca7e841c928cc7f834112d32f2d153024b19c1dfa |
| experiments/genseg_joint/joint_cli.py | b5ddace68d18993bbcced291e01509d7b6261fd6fa0fc49dc8a600e6da09567a |
| experiments/genseg_joint/joint_repair.py | 726a8fb7311b7f4a18f481f8b6ef0b5ba13ced49e8dee1cbb23902296e6d7343 |
| experiments/genseg_joint/joint_runtime.py | e885e1d95eebbc999b1b24fd845388e21086ef91e18e48568cbc4388b7a5cdae |
| experiments/genseg_joint/models.py | dd69938ea02098b43e8bcc43ae324ce1fbcbb34a3a284399d43ba02619e9a30a |
| experiments/genseg_joint/preview.py | ae159e0ddf263244e62d1489f7335a6593e33dd45af3e76b39e08a3a07f2a705 |
| experiments/genseg_segment.py | c5647f21d4f33fab481b0edae081957e31f7ec3de3cad012adcffa6775590ac1 |
| experiments/genseg_joint/nnunet_s.py | d4c2f13a40eda34857daba36914a8012df501fc9aa4bae6e4435a9b0d8f99dca |
| experiments/genseg_joint/tests/test_nnunet_s.py | ae9857bfaa8d78ecbc1d862203f33b874d888f4b1eaecd9d542ae122410024ec |
| experiments/genseg_nnunet_s_RUNBOOK.md | 636d56f3c5676b391b832fe1aaf5b7172b73c1d6958f7c48a8a0fe52f8b06de2 |
| experiments/genseg_nnunet_s_VALIDATION.md | 4a68437a6c36c0bef7a159084eacf0ca1ae067358a51f018b2a04bce8f1f5c6a |

本HANDOFF自身不能包含自身SHA；其最终SHA由交付时外部校验输出记录。
