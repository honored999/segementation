"""Train a UNet teacher model on CI-1 DWI slices.

中文说明：
这个脚本在同一套 CI-1 DWI 单通道数据上训练常规 UNet 教师网络，
用于和光学/无跳跃连接学生网络对比，判断问题来自数据难度还是学生网络瓶颈。

The teacher is a conventional segmentation baseline for the same single-channel
DWI data used by train_ci1_dwi_student_noskip_32ch.py. Its main purpose is to
separate data difficulty from the optical/no-skip student bottleneck.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_ci1_dwi_student_noskip_32ch import (
    BCEDiceLoss,
    BinaryMetrics,
    BinaryMetricAccumulator,
    CI1DwiSliceDataset,
    CI1DwiTensorCacheDataset,
    configure_torch_threads,
    count_parameters,
    manifest_uses_tensor_cache,
    read_manifest_rows,
    split_manifest_rows_by_patient,
    visualize_predictions,
)


@dataclass
class CI1DwiUNetTeacherConfig:
    manifest_path: Path = Path("data") / "ci1_dwi_tensor_cache_256" / "cache_manifest.csv"
    output_dir: Path = Path("results") / "ci1_dwi_unet_teacher"
    batch_size: int = 8
    num_epochs: int = 40
    learning_rate: float = 1e-3
    train_split: float = 0.8
    split_seed: int = 42
    num_workers: int = 0
    image_height: int = 256
    image_width: int = 256
    base_channels: int = 32
    save_predictions_every: int = 5
    torch_threads: int | None = None
    torch_interop_threads: int | None = None
    bce_weight: float = 1.0
    dice_weight: float = 1.0


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class CI1DwiUNetTeacher(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        x = self.bottleneck(self.pool(skip3))
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return self.head(x)


def get_loaders(config: CI1DwiUNetTeacherConfig) -> tuple[DataLoader, DataLoader]:
    rows = read_manifest_rows(config.manifest_path)
    train_rows, val_rows = split_manifest_rows_by_patient(
        rows,
        train_split=config.train_split,
        seed=config.split_seed,
    )

    if manifest_uses_tensor_cache(rows):
        train_dataset = CI1DwiTensorCacheDataset(config.manifest_path, train_rows)
        val_dataset = CI1DwiTensorCacheDataset(config.manifest_path, val_rows)
    else:
        train_dataset = CI1DwiSliceDataset(
            config.manifest_path,
            train_rows,
            image_height=config.image_height,
            image_width=config.image_width,
        )
        val_dataset = CI1DwiSliceDataset(
            config.manifest_path,
            val_rows,
            image_height=config.image_height,
            image_width=config.image_width,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, BinaryMetrics]:
    model.train()
    total_loss, num_batches = 0.0, 0
    metric_accumulator = BinaryMetricAccumulator()

    for images, masks in tqdm(loader, desc=f"Teacher epoch {epoch}"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        metric_accumulator.update(logits.detach(), masks)
        num_batches += 1

    return total_loss / num_batches, metric_accumulator.compute()


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, BinaryMetrics]:
    model.eval()
    total_loss, num_batches = 0.0, 0
    metric_accumulator = BinaryMetricAccumulator()

    for images, masks in tqdm(loader, desc="Teacher validating"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += float(loss.item())
        metric_accumulator.update(logits, masks)
        num_batches += 1

    return total_loss / num_batches, metric_accumulator.compute()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CI-1 DWI UNet teacher.")
    parser.add_argument("--manifest-path", type=Path, default=CI1DwiUNetTeacherConfig.manifest_path)
    parser.add_argument("--output-dir", type=Path, default=CI1DwiUNetTeacherConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=CI1DwiUNetTeacherConfig.num_epochs)
    parser.add_argument("--batch-size", type=int, default=CI1DwiUNetTeacherConfig.batch_size)
    parser.add_argument("--lr", type=float, default=CI1DwiUNetTeacherConfig.learning_rate)
    parser.add_argument("--height", type=int, default=CI1DwiUNetTeacherConfig.image_height)
    parser.add_argument("--width", type=int, default=CI1DwiUNetTeacherConfig.image_width)
    parser.add_argument("--base-channels", type=int, default=CI1DwiUNetTeacherConfig.base_channels)
    parser.add_argument("--num-workers", type=int, default=CI1DwiUNetTeacherConfig.num_workers)
    parser.add_argument("--bce-weight", type=float, default=CI1DwiUNetTeacherConfig.bce_weight)
    parser.add_argument("--dice-weight", type=float, default=CI1DwiUNetTeacherConfig.dice_weight)
    parser.add_argument("--torch-threads", type=int, default=CI1DwiUNetTeacherConfig.torch_threads)
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=CI1DwiUNetTeacherConfig.torch_interop_threads,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_torch_threads(args.torch_threads, args.torch_interop_threads)
    config = CI1DwiUNetTeacherConfig(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        image_height=args.height,
        image_width=args.width,
        base_channels=args.base_channels,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        torch_threads=args.torch_threads,
        torch_interop_threads=args.torch_interop_threads,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Torch CPU threads: {torch.get_num_threads()}")
    print(f"Torch inter-op threads: {torch.get_num_interop_threads()}")
    print(f"Manifest: {config.manifest_path}")
    print(f"Output: {config.output_dir}")

    train_loader, val_loader = get_loaders(config)
    print(f"Train slices: {len(train_loader.dataset)}")
    print(f"Val slices:   {len(val_loader.dataset)}")

    model = CI1DwiUNetTeacher(base_channels=config.base_channels).to(device)
    print(f"Trainable params: {count_parameters(model):,}")
    print(f"Base channels: {config.base_channels}")

    criterion = BCEDiceLoss(config.bce_weight, config.dice_weight)
    print(f"Loss: BCE*{config.bce_weight:g} + Dice*{config.dice_weight:g}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_dice = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics.positive_dice)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} "
            f"iou {train_metrics.iou:.4f} dice {train_metrics.dice:.4f} "
            f"pos_iou {train_metrics.positive_iou:.4f} pos_dice {train_metrics.positive_dice:.4f} | "
            f"val loss {val_loss:.4f} "
            f"iou {val_metrics.iou:.4f} dice {val_metrics.dice:.4f} "
            f"pos_iou {val_metrics.positive_iou:.4f} pos_dice {val_metrics.positive_dice:.4f} "
            f"pred_pos {val_metrics.predicted_positive_slices}/{val_metrics.total_slices}"
        )

        if val_metrics.positive_dice > best_dice:
            best_dice = val_metrics.positive_dice
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "val_dice": val_metrics.dice,
                    "val_iou": val_metrics.iou,
                    "val_positive_dice": val_metrics.positive_dice,
                    "val_positive_iou": val_metrics.positive_iou,
                    "val_positive_mask_slices": val_metrics.positive_mask_slices,
                    "val_predicted_positive_slices": val_metrics.predicted_positive_slices,
                    "config": {
                        "image_height": config.image_height,
                        "image_width": config.image_width,
                        "base_channels": config.base_channels,
                        "manifest_path": str(config.manifest_path),
                        "bce_weight": config.bce_weight,
                        "dice_weight": config.dice_weight,
                    },
                },
                config.output_dir / "ci1_dwi_unet_teacher_best.pth",
            )
            print(
                f"[OK] Saved best teacher checkpoint from epoch {epoch} "
                f"positive Dice={val_metrics.positive_dice:.4f}"
            )
            visualize_predictions(
                model,
                val_loader,
                device,
                config.output_dir / "predictions_best.png",
            )

        if epoch == 1 or epoch % config.save_predictions_every == 0:
            visualize_predictions(
                model,
                val_loader,
                device,
                config.output_dir / f"predictions_epoch_{epoch:03d}.png",
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best teacher epoch: {best_epoch}, positive Dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
