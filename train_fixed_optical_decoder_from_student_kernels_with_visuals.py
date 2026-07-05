"""
Train decoder only with fixed optical kernels from student2_noskip_32ch_best.pth.

中文说明：
这个脚本固定已经训练好的学生网络光学前端卷积核，只重新训练后端电子解码器，
用于验证固定光学特征提取后，解码器是否还能完成分割任务。

Goal:
- Load optical_kernels from a trained OpticalElectronicStudent2NoSkip32 checkpoint.
- Freeze the optical front-end.
- Recompute bottleneck features from RGB images by channel-wise optical convolution:
    R image * R kernel + G image * G kernel + B image * B kernel
- Train only the electronic decoder from scratch.

This is meant to align with unet_student2_noskip_32ch.py while making the
optical feature extraction fixed.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Robust project path setup
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SEG_DIR = SRC_DIR / "segmentation"
if str(SEG_DIR) not in sys.path:
    sys.path.insert(0, str(SEG_DIR))

try:
    from src.segmentation.unet_student2_noskip_32ch import (
        OpticalElectronicStudent2NoSkip32,
        Student2NoSkip32Config,
    )
    from src.segmentation.unet_student2 import (
        CarvanaStudentDataset,
        calculate_iou,
        calculate_dice,
        count_parameters,
    )
except ImportError:
    # Fallback when running inside src/segmentation
    from unet_student2_noskip_32ch import (  # type: ignore
        OpticalElectronicStudent2NoSkip32,
        Student2NoSkip32Config,
    )
    from unet_student2 import (  # type: ignore
        CarvanaStudentDataset,
        calculate_iou,
        calculate_dice,
        count_parameters,
    )


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
class TrainConfig:
    # Data
    data_root = PROJECT_ROOT / "data/carvana"
    image_height = 216
    image_width = 384
    train_split = 0.8
    split_seed = 42
    num_workers = 4
    preload_data_to_gpu = True
    preload_sensor_inputs = True

    # Model / checkpoint
    # Your original student script saves the best state_dict directly here.
    student_ckpt_path = PROJECT_ROOT / "results/student2_noskip_32ch_best.pth"
    num_kernels = 32
    sensor_size = (47, 83)
    decoder_size1 = (94, 166)
    decoder_size2 = (188, 332)
    decoder_size3 = (216, 384)

    # Training
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-3
    early_stopping_patience = 15

    # Output
    output_dir = PROJECT_ROOT / "results/fixed_optical_decoder_from_student_kernels"

    # Optional: normally False for your requested experiment.
    # False = use only optical_kernels from checkpoint; decoder starts random.
    # True  = initialize decoder weights from checkpoint, but still train decoder only.
    init_decoder_from_student = False


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class FixedStudentKernelDecoder(nn.Module):
    """
    Fixed optical encoder + trainable electronic decoder.

    Optical part:
        x RGB image -> adaptive_avg_pool2d -> per-channel conv -> channel sum

    Decoder part:
        Same up1/up2/up3 structure as OpticalElectronicStudent2NoSkip32.
    """

    def __init__(self, student_ckpt_path: str, init_decoder_from_student: bool = False):
        super().__init__()
        config = Student2NoSkip32Config()

        # A reference model is convenient because it has the same parameter names.
        reference = OpticalElectronicStudent2NoSkip32(
            in_channels=3,
            num_kernels=config.num_kernels,
            out_channels=1,
        )

        state = torch.load(student_ckpt_path, map_location="cpu")
        # Support both direct state_dict and wrapped checkpoint dict.
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state

        missing, unexpected = reference.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARNING] Missing keys when loading student checkpoint: {missing}")
        if unexpected:
            print(f"[WARNING] Unexpected keys when loading student checkpoint: {unexpected}")

        # Fixed optical kernels from trained student.
        # Shape: (32, 3, 3, 3)
        self.register_buffer("optical_kernels_3x3", reference.optical_kernels.detach().clone())
        self.num_kernels = self.optical_kernels_3x3.shape[0]
        self.sensor_size = TrainConfig.sensor_size

        # New decoder. By default random init, because the requested experiment is
        # fixed optical front-end + train decoder only.
        self.decoder = OpticalElectronicStudent2NoSkip32(
            in_channels=3,
            num_kernels=config.num_kernels,
            out_channels=1,
        )

        if init_decoder_from_student:
            print("[INFO] Initializing decoder weights from student checkpoint.")
            self.decoder.up1_conv.load_state_dict(reference.up1_conv.state_dict())
            self.decoder.up1_bn.load_state_dict(reference.up1_bn.state_dict())
            self.decoder.up2_conv.load_state_dict(reference.up2_conv.state_dict())
            self.decoder.up2_bn.load_state_dict(reference.up2_bn.state_dict())
            self.decoder.up3_conv.load_state_dict(reference.up3_conv.state_dict())
        else:
            print("[INFO] Decoder is randomly initialized.")

        # Make sure decoder's own optical kernels are not used/trained.
        self.decoder.optical_kernels.requires_grad_(False)

        print("[OK] Loaded and froze optical kernels from:", student_ckpt_path)
        print("     optical_kernels_3x3:", tuple(self.optical_kernels_3x3.shape))

    @staticmethod
    def upsample_kernel(kernel_3x3: torch.Tensor) -> torch.Tensor:
        return F.interpolate(kernel_3x3, size=(11, 11), mode="nearest")

    def fixed_optical_forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 216, 384)
        Returns:
            bottleneck: (B, 32, Hs, Ws), following the original conv settings.

        The computation is explicitly:
            bottleneck_k = conv(R, kernel_k_R) + conv(G, kernel_k_G) + conv(B, kernel_k_B)
        """
        if tuple(images.shape[-2:]) == tuple(self.sensor_size):
            x_downsampled = images
        else:
            x_downsampled = F.adaptive_avg_pool2d(images, self.sensor_size)  # (B, 3, 47, 83)
        kernels = self.upsample_kernel(self.optical_kernels_3x3)             # (32, 3, 11, 11)

        # Explicit channel-wise optical convolution + sum.
        bottleneck = None
        for c in range(3):
            img_c = x_downsampled[:, c:c + 1, :, :]      # (B, 1, 47, 83)
            ker_c = kernels[:, c:c + 1, :, :]            # (32, 1, 11, 11)
            feat_c = F.conv2d(img_c, ker_c, padding=5, bias=None)
            bottleneck = feat_c if bottleneck is None else bottleneck + feat_c

        return bottleneck

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            bottleneck = self.fixed_optical_forward(images)

        # Electronic decoder only; same upsampling structure as the student model.
        x = F.interpolate(bottleneck, size=TrainConfig.decoder_size1, mode="bilinear", align_corners=False)
        x = F.relu(self.decoder.up1_bn(self.decoder.up1_conv(x)))

        x = F.interpolate(x, size=TrainConfig.decoder_size2, mode="bilinear", align_corners=False)
        x = F.relu(self.decoder.up2_bn(self.decoder.up2_conv(x)))

        x = F.interpolate(x, size=TrainConfig.decoder_size3, mode="bilinear", align_corners=False)
        logits = self.decoder.up3_conv(x)
        return logits


# -----------------------------------------------------------------------------
# Train / validate helpers
# -----------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        iou = calculate_iou(logits, masks)
        dice = calculate_dice(logits, masks)

        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "iou": f"{iou:.4f}",
            "dice": f"{dice:.4f}",
        })

    return total_loss / num_batches, total_iou / num_batches, total_dice / num_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0

    for images, masks in tqdm(loader, desc="Validating"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        total_iou += calculate_iou(logits, masks)
        total_dice += calculate_dice(logits, masks)
        num_batches += 1

    return total_loss / num_batches, total_iou / num_batches, total_dice / num_batches


def preload_loader_to_gpu(loader, device, sensor_size=None, desc="Preloading"):
    images_list, masks_list = [], []
    for images, masks in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if sensor_size is not None:
            images = F.adaptive_avg_pool2d(images, sensor_size)
        images_list.append(images.detach())
        masks_list.append(masks.detach())

    images_tensor = torch.cat(images_list, dim=0).contiguous()
    masks_tensor = torch.cat(masks_list, dim=0).contiguous()
    print(
        f"[GPU preload] {desc}: images={tuple(images_tensor.shape)} "
        f"masks={tuple(masks_tensor.shape)} device={images_tensor.device}"
    )
    return images_tensor, masks_tensor


def iter_gpu_tensor_batches(images, masks, batch_size, shuffle):
    num_samples = images.shape[0]
    if shuffle:
        indices = torch.randperm(num_samples, device=images.device)
    else:
        indices = torch.arange(num_samples, device=images.device)

    for start in range(0, num_samples, batch_size):
        batch_indices = indices[start:start + batch_size]
        yield images[batch_indices], masks[batch_indices]


@torch.no_grad()
def visualize_predictions(model, loader, device, output_path, num_samples=4):
    model.eval()
    images_list, masks_list, preds_list = [], [], []

    for images, masks in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.sigmoid(logits) > 0.5

        images_list.append(images.cpu())
        masks_list.append(masks.cpu())
        preds_list.append(preds.cpu())

        if len(images_list) * images.size(0) >= num_samples:
            break

    images = torch.cat(images_list, dim=0)[:num_samples]
    masks = torch.cat(masks_list, dim=0)[:num_samples]
    preds = torch.cat(preds_list, dim=0)[:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        img = np.clip(images[i].permute(1, 2, 0).numpy(), 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i, 0].numpy(), cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds[i, 0].numpy(), cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()



@torch.no_grad()
def visualize_fixed_optical_kernels(model, output_path):
    """Visualize fixed optical kernels loaded from student checkpoint.

    Saves a 4x8 grid. Each tile shows the RGB-mean 11x11 kernel for one optical channel.
    Also saves a separate RGB-channel version for R/G/B components.
    """
    kernels_3x3 = model.optical_kernels_3x3.detach().cpu()  # (32, 3, 3, 3)
    kernels_11x11 = F.interpolate(kernels_3x3, size=(11, 11), mode="nearest").numpy()

    # Mean over RGB channels, comparable to the original student visualization.
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for idx in range(kernels_11x11.shape[0]):
        row, col = idx // 8, idx % 8
        ax = axes[row, col]
        k = kernels_11x11[idx].mean(axis=0)
        vmax = max(abs(k.min()), abs(k.max()), 1e-8)
        ax.imshow(k, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"K{idx:02d}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Fixed optical kernels from student checkpoint, RGB mean", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Per RGB channel visualization: rows are R/G/B, columns are kernels.
    rgb_path = Path(output_path).with_name(Path(output_path).stem + "_rgb_channels" + Path(output_path).suffix)
    fig, axes = plt.subplots(3, 32, figsize=(32, 4.5))
    channel_names = ["R", "G", "B"]
    for c in range(3):
        for idx in range(kernels_11x11.shape[0]):
            ax = axes[c, idx]
            k = kernels_11x11[idx, c]
            vmax = max(abs(k.min()), abs(k.max()), 1e-8)
            ax.imshow(k, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            if c == 0:
                ax.set_title(f"K{idx:02d}", fontsize=6)
            if idx == 0:
                ax.set_ylabel(channel_names[c], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Fixed optical kernels by RGB channel", fontsize=12)
    plt.tight_layout()
    plt.savefig(rgb_path, dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def visualize_bottleneck_features(model, loader, device, output_path, sample_index=0):
    """Visualize one sample's fixed optical bottleneck features.

    Saves:
    - input image
    - GT mask
    - 32 bottleneck feature maps after fixed optical convolution
    - RGB-channel contribution maps averaged over 32 kernels
    """
    model.eval()
    images, masks = next(iter(loader))
    images = images.to(device)
    masks = masks.to(device)

    image = images[sample_index:sample_index + 1]
    mask = masks[sample_index:sample_index + 1]

    x_downsampled = F.adaptive_avg_pool2d(image, model.sensor_size)
    kernels = model.upsample_kernel(model.optical_kernels_3x3)

    channel_feats = []
    for c in range(3):
        img_c = x_downsampled[:, c:c + 1, :, :]
        ker_c = kernels[:, c:c + 1, :, :]
        feat_c = F.conv2d(img_c, ker_c, padding=5, bias=None)  # (1,32,47,83)
        channel_feats.append(feat_c)

    bottleneck = channel_feats[0] + channel_feats[1] + channel_feats[2]
    bottleneck_np = bottleneck.squeeze(0).detach().cpu().numpy()  # (32,47,83)

    # 32 feature maps.
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for idx in range(32):
        row, col = idx // 8, idx % 8
        ax = axes[row, col]
        feat = bottleneck_np[idx]
        vmax = max(abs(feat.min()), abs(feat.max()), 1e-8)
        ax.imshow(feat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"F{idx:02d}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Fixed optical bottleneck feature maps, 32 channels", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Input / GT / RGB contributions.
    summary_path = Path(output_path).with_name(Path(output_path).stem + "_summary" + Path(output_path).suffix)
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    img_np = np.clip(image.squeeze(0).detach().cpu().permute(1, 2, 0).numpy(), 0, 1)
    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(mask.squeeze().detach().cpu().numpy(), cmap="gray")
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    for c, name in enumerate(["R contribution", "G contribution", "B contribution"]):
        contrib = channel_feats[c].squeeze(0).mean(dim=0).detach().cpu().numpy()
        vmax = max(abs(contrib.min()), abs(contrib.max()), 1e-8)
        axes[c + 2].imshow(contrib, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[c + 2].set_title(name)
        axes[c + 2].axis("off")

    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_training_history(history, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_iou"], label="Train")
    axes[1].plot(epochs, history["val_iou"], label="Val")
    axes[1].set_title("IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_dice"], label="Train")
    axes[2].plot(epochs, history["val_dice"], label="Val")
    axes[2].set_title("Dice")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Dice")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    cfg = TrainConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Student checkpoint: {cfg.student_ckpt_path}")
    print(f"Output directory: {output_dir}")

    # Dataset: same source as the original student script.
    full_dataset = CarvanaStudentDataset(
        data_root=cfg.data_root,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        train=True,
    )

    train_size = int(cfg.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.split_seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    model = FixedStudentKernelDecoder(
        student_ckpt_path=cfg.student_ckpt_path,
        init_decoder_from_student=cfg.init_decoder_from_student,
    ).to(device)

    # Only decoder parameters should be trainable.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Total parameters:     {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Visualize fixed optical front-end before training.
    visualize_fixed_optical_kernels(
        model,
        output_dir / "fixed_optical_kernels.png",
    )
    visualize_bottleneck_features(
        model,
        val_loader,
        device,
        output_dir / "fixed_optical_bottleneck_features_sample0.png",
        sample_index=0,
    )
    print("[OK] Saved fixed optical kernel and bottleneck feature visualizations.")

    train_gpu_data = None
    val_gpu_data = None
    if cfg.preload_data_to_gpu:
        preload_sensor_size = cfg.sensor_size if cfg.preload_sensor_inputs else None
        train_gpu_data = preload_loader_to_gpu(
            train_loader,
            device,
            sensor_size=preload_sensor_size,
            desc="Preloading train data to GPU",
        )
        val_gpu_data = preload_loader_to_gpu(
            val_loader,
            device,
            sensor_size=preload_sensor_size,
            desc="Preloading val data to GPU",
        )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(trainable_params, lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    history = {
        "train_loss": [], "train_iou": [], "train_dice": [],
        "val_loss": [], "val_iou": [], "val_dice": [],
    }

    best_val_dice = 0.0
    best_val_iou = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, cfg.num_epochs + 1):
        print("\n" + "=" * 70)
        print(f"Epoch {epoch}/{cfg.num_epochs}")
        print("=" * 70)

        if train_gpu_data is not None:
            train_images, train_masks = train_gpu_data
            train_source = iter_gpu_tensor_batches(train_images, train_masks, cfg.batch_size, shuffle=True)
        else:
            train_source = train_loader

        if val_gpu_data is not None:
            val_images, val_masks = val_gpu_data
            val_source = iter_gpu_tensor_batches(val_images, val_masks, cfg.batch_size, shuffle=False)
        else:
            val_source = val_loader

        train_loss, train_iou, train_dice = train_epoch(
            model, train_source, criterion, optimizer, device, epoch
        )
        val_loss, val_iou, val_dice = validate(model, val_source, criterion, device)
        scheduler.step(val_dice)

        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_iou)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)

        print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_iou = val_iou
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "decoder_state_dict": {
                    "up1_conv": model.decoder.up1_conv.state_dict(),
                    "up1_bn": model.decoder.up1_bn.state_dict(),
                    "up2_conv": model.decoder.up2_conv.state_dict(),
                    "up2_bn": model.decoder.up2_bn.state_dict(),
                    "up3_conv": model.decoder.up3_conv.state_dict(),
                },
                "val_dice": val_dice,
                "val_iou": val_iou,
                "student_ckpt_path": cfg.student_ckpt_path,
                "init_decoder_from_student": cfg.init_decoder_from_student,
                "sensor_size": cfg.sensor_size,
                "kernel_upsample_size": (11, 11),
                "decoder_size1": cfg.decoder_size1,
                "decoder_size2": cfg.decoder_size2,
                "decoder_size3": cfg.decoder_size3,
            }, output_dir / "fixed_optical_decoder_best.pth")
            print(f"[OK] Saved best model: Dice={val_dice:.4f}, IoU={val_iou:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement: {epochs_without_improvement}/{cfg.early_stopping_patience}")

        visualize_predictions(
            model,
            val_loader,
            device,
            output_dir / f"predictions_epoch_{epoch:03d}.png",
            num_samples=4,
        )

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print("Early stopping triggered.")
            break

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, output_dir / f"checkpoint_epoch_{epoch:03d}.pth")

    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "sensor_size": cfg.sensor_size,
        "kernel_upsample_size": (11, 11),
        "decoder_size1": cfg.decoder_size1,
        "decoder_size2": cfg.decoder_size2,
        "decoder_size3": cfg.decoder_size3,
    }, output_dir / "fixed_optical_decoder_final.pth")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_training_history(history, output_dir / "training_history.png")

    print("\n" + "=" * 70)
    print("Training complete")
    print(f"Best Val Dice: {best_val_dice:.4f}")
    print(f"Best Val IoU:  {best_val_iou:.4f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
