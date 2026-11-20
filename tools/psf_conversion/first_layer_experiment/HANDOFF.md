# HANDOFF · 光学首层 5×5 / 7×7 全网从头训练

当前协议使用 Dataset509_CI1DWIHigh，固定 fold 0 的 100 个训练和 28 个验证体积。只修改转成光学的首层 Conv2d。5×5 和 7×7 各自随机初始化，先用普通 Dice + CE 与深监督训练全网，再从各自最佳 checkpoint 导出首层目标核。电学阶段沿用 nnU-Net He 初始化、SGD、PolyLR、初始 LR 0.01、DA2 和种子 12345。

后续每组使用 532 nm 拟合 PSF，评估电学和拟合 PSF 直接推理。固定 PSF、增益、尺度和首层 bias，只微调电学后端，初始 LR 0.001。两次训练均用验证 EMA 前景 Dice 早停，最少 0 轮、耐心 30、min_delta 1e-4、最多 1000 轮。结果以最佳 checkpoint 在原空间评分，Precision 对空预测计零，空掩膜导致 HD95 无定义时记录有效例数，F2 沿用 paper 公式。

既有 3×3 保留作历史参照。它的电学训练耐心为 100，历史结果路径在新运行目录的 historical_k3_reference.json。先前从 3×3 checkpoint 出发只更新首层 250 步的试验已取消，不纳入结果。

入口为 tools/psf_conversion/first_layer_experiment/run.py，传入新的服务器 runs/optical_first_conv_* 目录和 preflight 或 formal。预检每个尺寸分别跑单步全网训练与验证、各 PSF 支路单步拟合、固定 PSF 后端单步更新与重载，以及单体积三阶段推理和评分。正式模式要求同一快照的预检成功。正式工件分布在 formal/k5 和 formal/k7，保留原生最佳 checkpoint、首层权重导出、psf532 和三阶段多指标。跨组逐病例及汇总文件为 paired_metrics.private.csv、summary_metrics.json。

532 nm 拟合沿用 n=800、FFT=3600、dx=400 nm、距离 2400 µm、每核采样单元 17×17 及 pixel_psf_superposition_v1。此前的合成拟合短测已生成 51、85、119 三种正确尺寸的工件；FFT 收敛和窗口外能量尚待验证。结果范围是单折开发验证和 PSF 仿真。
