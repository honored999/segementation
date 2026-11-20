# HANDOFF · 光学首层 3×3 / 5×5 / 7×7

实现位于独立工作树 `.worktrees/optical-first-conv-kernel-size`、分支 `codex/optical-first-conv-kernel-size`，基于 `cf0299210a2128f5cc273fdc76eba843144d2def`。原光学工作树的未提交修改、服务器历史运行和消融队列均保留。

- 入口为 `tools/psf_conversion/first_layer_experiment/run.py`，接受服务器 `runs/optical_first_conv_*` 下的新目录和 `preflight` 或 `formal`。先验证原队列已进入终态、已审计缓存和标签的哈希、基线 SHA-256 及 fold 0 的 100/28 划分。每个阶段均执行资源检查或等待。正式入口要求同一源码的预检已完成。
- 电学目标核采用 250 更新、LR 0.001、种子 12345，复用原训练器的损失、增强和训练采样。采用 eval 模式保持后端缓冲区、首层 bias 和后端权重固定。5×5、7×7 从居中复制的 3×3 起步。三个组均重新学习首层核。
- 532 nm 拟合沿用 n=800、FFT=3600、dx=400 nm 和距离 2400 µm，窗口分别为 51、85、119，保持像素响应叠加和全有效 PSF 支持。改动前验证历史拟合器 SHA-256 `d3034d83ef8bf216be431c162c2aaf654dc3e158734c83fa91e04dc2c8f4b53f`、传播依赖 `b9d7cf9c302814eb232d13d74c26220c5d371ccd628b4db353225e87193fce1e`、P400 相位依赖 `a71e0b7dccb4f0e9fccd094a059edc0b0ed7195ea88549efd9c324a1f3d3be20`。保留历史传播参数供三组配对。以轴向估算，Nyquist 为 1.25 cycles/µm，ASM 带限约 0.540 cycles/µm，当前传播器未应用该带限。工件记录裁剪后边缘能量和质心，FFT 收敛与窗口外能量仍待单独验证。
- 光学微调沿用 fresh SGD、LR 0.001、PolyLR、最多 1000 轮、耐心 30、min_delta 1e-4，冻结拟合 PSF、增益、尺度与首层 bias，以最佳 checkpoint 在原空间评分。每组保留电学、拟合直推、冻结后微调三阶段的病例级配对表。Precision 为空预测记零，任一掩膜为空的 HD95 留空并计有效例数。F2 用 paper 公式。
- 本地 newconda 的 16 项聚焦测试通过。合成 PlainConvUNet 三种核尺寸通过前向、单步首层更新、其余状态不变及严格重载。合成 CUDA 短测在每组第 0 通道的正负支路各跑 2 步，损失有限，产物在 `reports/first_layer_engineering_20261120/k{3,5,7}`。临时合成 NIfTI 还通过原空间评分、空预测 Precision、HD95 有效例数和配对表检查。这些工程验证使用合成输入。
- 对 `172.16.0.138:22` 两次 SSH 探测均在认证前连接超时。服务器预检和正式三组均未启动，尚无真实训练、PSF、推理或配对指标。预检设计为每组 1 次电学更新、32 通道正负支路各 1 拟合步、1 次后端更新并重载、1 个完整验证体积三阶段预测和评分。恢复连接后，先用已验证的 `C:/Users/user4/.conda/envs/nnunet5090/python.exe` 执行预检；通过后用同一入口执行正式模式。

```powershell
& C:/Users/user4/.conda/envs/nnunet5090/python.exe <checkout>/tools/psf_conversion/first_layer_experiment/run.py C:/lijialin/results/segementation/runs/optical_first_conv_20261120_01 preflight
# 同一份源码预检通过后，把最后一个参数改为 formal。
```

服务器工件规划在 `<run>/{preflight,formal}/k{3,5,7}`。每组内的 `electrical_target/model/fold_0/checkpoint_target.pth`、`psf532/`、`backend/model/fold_0/` 和 `*_metrics/` 提供阶段证据。跨组病例表和汇总分别为 `paired_metrics.private.csv`、`summary_metrics.json`。正式结果仅作 fold 0 开发验证，光学阶段属于 PSF 仿真。
