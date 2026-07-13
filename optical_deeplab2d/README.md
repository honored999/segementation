# OpticalConv + 2D DeepLabV3+

独立的 CI-1 DWI 二值病灶实验：比较 `HybridOpticalDeepLabV3Plus`（理想的可训练签名 5×5 光学卷积）和 `ElectronicDeepLabV3Plus`（灰度复制为三通道）的公平基线。模型输出均为 logits，不在模型中使用 sigmoid。

## 数据

默认使用 `data/ci1_dwi_2d_dedup/manifest.csv`。`patient` 列是患者 ID 的唯一来源，文件名中的哈希值不可用于提取患者；所以一个患者的 D1/D2/D3/D7/D14 均被分入同一个 fold。完整数据有 2,445 对 512×512 PNG，其中 1,110 个空 mask 被保留。不会写入或修改原始数据。

## 服务器安装与运行

```powershell
conda activate newconda
pip install -r optical_deeplab2d/requirements.txt
python optical_deeplab2d/datasets/inspect_dataset.py --data-root data/ci1_dwi_2d_dedup --output-dir optical_deeplab2d/outputs/dataset_check
python optical_deeplab2d/train.py --config optical_deeplab2d/configs/electronic_baseline.yaml --data-root data/ci1_dwi_2d_dedup --fold 0 --output-dir optical_deeplab2d/outputs/electronic_fold0
python optical_deeplab2d/train.py --config optical_deeplab2d/configs/hybrid_ideal.yaml --data-root data/ci1_dwi_2d_dedup --fold 0 --output-dir optical_deeplab2d/outputs/hybrid_ideal_fold0
```

训练前在服务器运行：完整 512×512 的双模型 shape/梯度测试，以及 `--overfit-small-batch` 至少 100 次迭代。若出现 CUDA OOM，请降低 batch size、使用梯度累积或降低 workers；不会自动缩小 512×512 图像。SMP 不支持 MobileNetV2 时会发出警告并显式回退 ResNet18，实际 encoder 会记录在 checkpoint。

本机只运行 `pytest`、`--help` 和编译检查；不运行完整 shape、过拟合或训练。
