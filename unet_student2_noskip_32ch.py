"""Optical-Electronic Hybrid Student Network 2 WITHOUT skip connections (32 channels).

Architecture:
- Input: 384×216 RGB images
- Optical encoder: 32 kernels
- Electronic decoder WITHOUT skip connections
- Output: 384×216 binary segmentation mask
"""

from __future__ import annotations
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加路径以支持从其他目录导入
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet_student2 import (
    CarvanaStudentDataset, calculate_iou, calculate_dice, count_parameters
)


@dataclass
class Student2NoSkip32Config:
    """Configuration for Student Network 2 (no skip, 32ch) training."""
    batch_size: int = 8
    num_epochs: int = 40
    learning_rate: float = 1e-3
    data_root: str = "./data/carvana"
    num_workers: int = 4
    image_height: int = 216
    image_width: int = 384
    num_kernels: int = 32
    use_knowledge_distillation: bool = False
    teacher_checkpoint: str = "./results/unet_carvana_best.pth"
    kd_temperature: float = 1.5
    kd_alpha: float = 0.3


class OpticalElectronicStudent2NoSkip32(nn.Module):
    """Optical-Electronic Hybrid Student Network 2 WITHOUT skip connections (32ch)."""

    def __init__(self, in_channels=3, num_kernels=32, out_channels=1):
        super().__init__()
        self.num_kernels = num_kernels
        
        # Optical encoder - 32 通道卷积核
        self.optical_kernels = nn.Parameter(torch.randn(num_kernels, in_channels, 3, 3) * 0.01)
        
        # Electronic decoder (NO skip connections)
        self.up1_conv = nn.Conv2d(num_kernels, 24, kernel_size=3, padding=1)
        self.up1_bn = nn.BatchNorm2d(24)
        self.up2_conv = nn.Conv2d(24, 16, kernel_size=3, padding=1)
        self.up2_bn = nn.BatchNorm2d(16)
        self.up3_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def upsample_kernel(self, kernel_3x3):
        return F.interpolate(kernel_3x3, size=(6, 6), mode='nearest')

    def forward(self, x):
        sensor_size = (27, 48)
        
        # Optical Encoder
        kernels = self.upsample_kernel(self.optical_kernels)
        # 先缩小到sensor尺寸
        x_downsampled = F.adaptive_avg_pool2d(x, sensor_size)  # (B, 3, 27, 48)
        # 再进行卷积（padding=2保证尺寸不变）
        bottleneck = F.conv2d(x_downsampled, kernels, padding=2, bias=None)  # (B, 32, 27, 48)
        
        # Electronic Decoder
        up1 = F.interpolate(bottleneck, size=(54, 96), mode='bilinear', align_corners=False)
        up1 = F.relu(self.up1_bn(self.up1_conv(up1)))
        
        up2 = F.interpolate(up1, size=(108, 192), mode='bilinear', align_corners=False)
        up2 = F.relu(self.up2_bn(self.up2_conv(up2)))
        
        up3 = F.interpolate(up2, size=(216, 384), mode='bilinear', align_corners=False)
        logits = self.up3_conv(up3)
        return logits


def get_loaders(config):
    full_dataset = CarvanaStudentDataset(
        data_root=config.data_root, image_height=config.image_height,
        image_width=config.image_width, train=True,
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    if len(full_dataset) > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    else:
        train_dataset, val_dataset = full_dataset, full_dataset

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    return train_loader, val_loader


def visualize_predictions(model, loader, device, save_dir="results", num_samples=8):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    images_list, masks_list, preds_list = [], [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
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
    plt.savefig(os.path.join(save_dir, "student2_noskip_32ch_predictions.png"), dpi=150)
    plt.close()
    print("Saved: student2_noskip_32ch_predictions.png")


def visualize_optical_kernels(model, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        kernels = model.optical_kernels.cpu()

    kernel_6x6 = F.interpolate(kernels, size=(6, 6), mode='nearest').numpy()

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for idx in range(32):
        row, col = idx // 8, idx % 8
        ax = axes[row, col]
        k = kernel_6x6[idx].mean(axis=0)
        vmax = max(abs(k.min()), abs(k.max()))
        ax.imshow(k, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.axis('off')

    fig.suptitle("Student2 NoSkip 32ch Optical Kernels (6×6, pos/neg)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "student2_noskip_32ch_optical_kernels.png"), dpi=150)
    plt.close()
    print("Saved: student2_noskip_32ch_optical_kernels.png")


def main():
    config = Student2NoSkip32Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_loaders(config)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    student = OpticalElectronicStudent2NoSkip32(
        in_channels=3, num_kernels=config.num_kernels,
    ).to(device)
    
    print(f"Student2 NoSkip 32ch params: {count_parameters(student):,}")
    print(f"Optical kernels: {config.num_kernels}")

    teacher = None
    if config.use_knowledge_distillation and os.path.exists(config.teacher_checkpoint):
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from src.segmentation.unet_teacher import UNet
        teacher = UNet(in_channels=3, out_channels=1).to(device)
        teacher.load_state_dict(torch.load(config.teacher_checkpoint, map_location=device))
        teacher.eval()
        print(f"Loaded teacher from: {config.teacher_checkpoint}")

    optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_dice, best_state, best_epoch = 0.0, None, -1
    pbar = tqdm(range(config.num_epochs), desc="Training Student2 NoSkip 32ch")
    
    for epoch in pbar:
        student.train()
        train_loss, train_iou, teacher_iou, num_batches = 0.0, 0.0, 0.0, 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = student(images)
            
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                    teacher_iou += calculate_iou(teacher_logits, masks)
                student_soft = torch.sigmoid(logits / config.kd_temperature)
                teacher_soft = torch.sigmoid(teacher_logits / config.kd_temperature)
                kd_loss = F.mse_loss(student_soft, teacher_soft) * (config.kd_temperature ** 2)
                gt_loss = criterion(logits, masks)
                loss = config.kd_alpha * kd_loss + (1 - config.kd_alpha) * gt_loss
            else:
                loss = criterion(logits, masks)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iou += calculate_iou(logits, masks)
            num_batches += 1

        train_loss /= num_batches
        train_iou /= num_batches
        teacher_iou /= num_batches if teacher is not None else 1.0

        student.eval()
        val_loss, val_dice, val_iou, num_val = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = student(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                val_dice += calculate_dice(logits, masks)
                val_iou += calculate_iou(logits, masks)
                num_val += 1

        val_loss /= num_val
        val_dice /= num_val
        val_iou /= num_val
        scheduler.step(val_dice)

        postfix_dict = {
            "loss": f"{train_loss:.4f}",
            "train_iou": f"{train_iou:.4f}",
            "val_iou": f"{val_iou:.4f}",
            "best": f"{best_dice:.4f}"
        }
        if teacher is not None:
            postfix_dict["teacher_iou"] = f"{teacher_iou:.4f}"
        
        pbar.set_postfix(postfix_dict)

        if val_dice > best_dice:
            best_dice = val_dice
            best_state = student.state_dict()
            best_epoch = epoch + 1

    if best_state is not None:
        os.makedirs("results", exist_ok=True)
        ckpt_path = os.path.join("results", "student2_noskip_32ch_best.pth")
        torch.save(best_state, ckpt_path)
        print(f"\nBest model from epoch {best_epoch} with Dice={best_dice:.4f}")
        print(f"Saved to: {ckpt_path}")

        student.load_state_dict(best_state)
        visualize_predictions(student, val_loader, device)
        visualize_optical_kernels(student)
        print("Done!")


if __name__ == "__main__":
    main()
