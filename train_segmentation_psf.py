"""为分割任务训练多色 PSF。

中文说明：
这个脚本用于训练多色 PSF，使其尽量模拟学生网络中的光学卷积核响应，
服务于后续把电学卷积核转换为超表面 PSF 的流程。

从 noskip_32ch 模型中读取 32 个光学卷积核，分离正负后批量训练 PSF。
每个卷积核形状为 (3, 3, 3)，上采样后嵌入到输出网格中心。
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# 解决 OpenMP 库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

# 路径设置
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.psf.fitted_phase_parameters_p400 import compute_phase_fitted
from psf.propagation import create_propagation_layer


@dataclass
class SegmentationPSFConfig:
    """分割任务 PSF 配置。"""
    dx: float           # 超表面单元尺寸（米）
    n: int              # 超表面采样点数
    f: float            # 焦距（米）
    wavelengths: List[float]  # RGB 波长（米）
    camera_pixel_size: Optional[float] = None  # 相机像素尺寸（米）
    device: str = "cuda"


class TrainableSegmentationMetasurface(nn.Module):
    """用于分割任务的可训练多色元表面。
    
    所有计算都在超表面分辨率上进行，不进行相机插值。
    """

    def __init__(self, cfg: SegmentationPSFConfig) -> None:
        super().__init__()
        self.cfg = cfg
        torch_device = torch.device(cfg.device)

        # 可训练柱宽参数（超表面分辨率）
        self.width_nm = nn.Parameter(
            torch.zeros((cfg.n, cfg.n), dtype=torch.float32, device=torch_device)
        )

        # 入射场（超表面分辨率）
        incident = torch.ones(
            (3, cfg.n, cfg.n), dtype=torch.complex64, device=torch_device
        )
        self.register_buffer("incident", incident)

        # 多波长传播算子（在超表面分辨率上）
        wavelengths_dict = {
            "red": float(cfg.wavelengths[2]),
            "green": float(cfg.wavelengths[1]),
            "blue": float(cfg.wavelengths[0]),
        }

        padding_px = cfg.n

        self.propagator = create_propagation_layer(
            distance=float(cfg.f),
            wavelength=wavelengths_dict,
            pixel_size=float(cfg.dx),
            device=cfg.device,
            padding=padding_px,
        )
        
        # 计算相机像素到超表面单元的缩放比例
        if cfg.camera_pixel_size is not None:
            self.camera_to_meta_scale = cfg.camera_pixel_size / cfg.dx
            print(f"[Metasurface] 超表面网格: {cfg.n}x{cfg.n} @ {cfg.dx*1e9:.1f}nm")
            print(f"[Metasurface] 相机像素尺寸: {cfg.camera_pixel_size*1e6:.2f}um")
            print(f"[Metasurface] 每个相机像素 = {self.camera_to_meta_scale:.2f} 个超表面单元")
        else:
            self.camera_to_meta_scale = 1.0
            print(f"[Metasurface] 网格: {cfg.n}x{cfg.n} @ {cfg.dx*1e9:.1f}nm (无相机参数)")

    def forward(self) -> torch.Tensor:
        """前向传播，返回 RGB PSF（超表面分辨率）。"""
        # 计算柱宽度（超表面分辨率）
        width_nm = 80.0 + 240.0 * torch.sigmoid(self.width_nm)

        wavelengths_nm = [int(round(w * 1e9)) for w in self.cfg.wavelengths]
        blue_nm, green_nm, red_nm = wavelengths_nm

        # 计算相位（超表面分辨率）
        phase_red = compute_phase_fitted(width_nm, red_nm)
        phase_green = compute_phase_fitted(width_nm, green_nm)
        phase_blue = compute_phase_fitted(width_nm, blue_nm)

        phase_stack = torch.stack((phase_red, phase_green, phase_blue), dim=0)
        t_rgb = torch.exp(1j * phase_stack.to(torch.complex64))

        field_in = self.incident * t_rgb
        field_in = field_in.unsqueeze(0)

        field_out = self.propagator(field_in)
        intensity = torch.abs(field_out) ** 2
        return intensity.squeeze(0)


def load_segmentation_config(config_path: str, device: str = "cuda") -> SegmentationPSFConfig:
    """从 YAML 加载分割任务 PSF 配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    f_m = float(cfg_dict.get("f_um", 2400.0)) * 1e-6
    dx_m = float(cfg_dict.get("dx_nm", 400.0)) * 1e-9
    n = int(cfg_dict.get("n", 800))

    wavelengths_cfg = cfg_dict.get("wavelengths_rgb_nm", [450.0, 520.0, 635.0])
    wavelengths_m = [float(w) * 1e-9 for w in wavelengths_cfg]

    # 相机像素尺寸（可选）
    camera_pixel_nm = cfg_dict.get("camera_pixel_nm", 3450)
    camera_pixel_m = float(camera_pixel_nm) * 1e-9 if camera_pixel_nm is not None else None

    return SegmentationPSFConfig(
        dx=dx_m,
        n=n,
        f=f_m,
        wavelengths=wavelengths_m,
        camera_pixel_size=camera_pixel_m,
        device=device,
    )


def load_optical_kernels(checkpoint_path: str, device: str = "cpu", model_type: str = "segmentation") -> np.ndarray:
    """从模型checkpoint加载光学卷积核。
    
    支持两种模型类型：
    - segmentation: 分割模型 (OpticalElectronicStudent2NoSkip32)
    - multitask: 多任务模型 (MultitaskModel)
    
    Args:
        checkpoint_path: checkpoint文件路径
        device: 设备
        model_type: 模型类型 ("segmentation" 或 "multitask")
    
    Returns:
        光学卷积核，形状 (32, 3, 3, 3)
    """
    print(f"Loading optical kernels from: {checkpoint_path}")
    print(f"Model type: {model_type}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 检查checkpoint格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 完整的训练checkpoint（包含optimizer等）
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded training checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
    else:
        # 只有模型权重
        state_dict = checkpoint
        print(f"Loaded model weights")
    
    # 提取optical_kernels
    if 'optical_kernels' in state_dict:
        kernels = state_dict['optical_kernels'].cpu().numpy()
    else:
        raise KeyError("'optical_kernels' not found in checkpoint. "
                      f"Available keys: {list(state_dict.keys())}")
    
    print(f"Loaded optical kernels: {kernels.shape}")
    print(f"Kernel value range: [{kernels.min():.4f}, {kernels.max():.4f}]")
    
    return kernels


def split_positive_negative(kernels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将卷积核分离为正核和负核。"""
    kernels_pos = np.maximum(kernels, 0.0)
    kernels_neg = np.maximum(-kernels, 0.0)
    return kernels_pos, kernels_neg


def normalize_kernels(kernels_pos: np.ndarray, kernels_neg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对每个卷积核进行归一化。"""
    num_kernels = kernels_pos.shape[0]
    for i in range(num_kernels):
        max_pos = float(np.max(kernels_pos[i]))
        max_neg = float(np.max(kernels_neg[i]))
        max_val = max(max_pos, max_neg)
        if max_val > 0.0:
            kernels_pos[i] = kernels_pos[i] / max_val
            kernels_neg[i] = kernels_neg[i] / max_val
    return kernels_pos, kernels_neg


def fill_zero_channels(kernels_pos: np.ndarray, kernels_neg: np.ndarray, threshold: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """检查并填充全零通道。
    
    如果某个核的某个通道全为0，则用对应核的值填充（不取相反值）。
    例如：负核红通道为全0，则直接复制正核红通道的值。
    
    注意：split_positive_negative后，正负核的值都是非负的。
    
    Args:
        kernels_pos: 正核，形状 (N, 3, H, W)，值都 >= 0
        kernels_neg: 负核，形状 (N, 3, H, W)，值都 >= 0
        threshold: 判断为零的阈值
    
    Returns:
        填充后的正核和负核
    """
    num_kernels, num_channels = kernels_pos.shape[0], kernels_pos.shape[1]
    kernels_pos = kernels_pos.copy()
    kernels_neg = kernels_neg.copy()
    
    fill_count = 0
    
    for i in range(num_kernels):
        for c in range(num_channels):
            # 检查正核通道是否全零
            if np.abs(kernels_pos[i, c]).max() < threshold:
                # 直接复制负核的值（不取相反值）
                kernels_pos[i, c] = kernels_neg[i, c].copy()
                fill_count += 1
                print(f"  Kernel #{i}, channel {c}: Filled pos with neg (was zero)")
            
            # 检查负核通道是否全零
            if np.abs(kernels_neg[i, c]).max() < threshold:
                # 直接复制正核的值（不取相反值）
                kernels_neg[i, c] = kernels_pos[i, c].copy()
                fill_count += 1
                print(f"  Kernel #{i}, channel {c}: Filled neg with pos (was zero)")
    
    if fill_count > 0:
        print(f"✓ Filled {fill_count} zero channels with opposite kernel values")
    else:
        print(f"✓ No zero channels found, all channels have non-zero values")
    
    return kernels_pos, kernels_neg


def build_target_field(
    kernel: np.ndarray,
    meta_n: int,
    patch_size_meta: int,
) -> np.ndarray:
    """将 3×3×3 卷积核上采样并嵌入到超表面网格中心。
    
    Args:
        kernel: 单个卷积核，形状 (3, 3, 3)
        meta_n: 超表面网格尺寸
        patch_size_meta: 目标 patch 大小（超表面单元数）
    
    Returns:
        目标场，形状 (3, meta_n, meta_n)
    """
    c_in, k_h, k_w = kernel.shape
    
    # 上采样到 patch_size_meta × patch_size_meta
    kernel_t = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0)
    kernel_upsampled = F.interpolate(
        kernel_t,
        size=(patch_size_meta, patch_size_meta),
        mode="nearest",
    ).squeeze(0).numpy()

    # 嵌入到超表面网格中心
    target_full = np.zeros((c_in, meta_n, meta_n), dtype=np.float32)
    center = meta_n // 2
    half = patch_size_meta // 2
    start = center - half
    end = start + patch_size_meta  # 确保尺寸匹配

    target_full[:, start:end, start:end] = kernel_upsampled
    return target_full


def train_single_psf(
    cfg: SegmentationPSFConfig,
    target_field: np.ndarray,
    patch_size_meta: int,
    num_steps: int = 500,
    lr: float = 5e-3,
    patience: int = 10,
) -> Tuple[TrainableSegmentationMetasurface, float]:
    """训练单个 PSF。"""
    device = torch.device(cfg.device)
    target_full = torch.from_numpy(target_field).to(device)

    meta_n = cfg.n
    center = meta_n // 2
    half = patch_size_meta // 2
    start = center - half
    end = start + patch_size_meta  # 确保尺寸匹配

    model = TrainableSegmentationMetasurface(cfg).to(device)
    with torch.no_grad():
        model.width_nm.uniform_(-1.0, 1.0)
    
    optimizer = torch.optim.Adam([model.width_nm], lr=lr)

    best_loss = float("inf")
    best_width = None
    steps_no_improve = 0

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        intensity_rgb = model()

        # 形状匹配损失
        eps = 1e-12
        c = intensity_rgb.shape[0]
        psf_flat = intensity_rgb.reshape(c, -1)
        target_flat = target_full.reshape(c, -1)
        psf_norm = psf_flat / (psf_flat.norm(dim=1, keepdim=True) + eps)
        target_norm = target_flat / (target_flat.norm(dim=1, keepdim=True) + eps)

        diff = psf_norm - target_norm
        s_c = diff.pow(2).sum(dim=1)
        loss = torch.sqrt((s_c * s_c).mean())
        loss_value = float(loss.item())

        if loss_value < best_loss:
            best_loss = loss_value
            steps_no_improve = 0
            best_width = model.width_nm.detach().cpu().clone()
        else:
            steps_no_improve += 1

        loss.backward()
        optimizer.step()

        if steps_no_improve >= patience:
            break

    if best_width is not None:
        model.width_nm.data.copy_(best_width.to(model.width_nm.device))

    return model, best_loss


def visualize_kernel_and_psf(
    kernel: np.ndarray,
    psf: np.ndarray,
    save_path: str,
    kernel_idx: int,
    is_positive: bool,
    patch_size: int = 6,
    metasurface_n: int = 800,
    camera_n: int = 92,
) -> None:
    """可视化单个卷积核和对应的 PSF。
    
    Args:
        kernel: 卷积核，形状 (3, 3, 3)
        psf: PSF，形状 (3, metasurface_n, metasurface_n) - 超表面分辨率
        save_path: 保存路径
        kernel_idx: 卷积核索引
        is_positive: 是否为正核
        patch_size: 相机像素数（用于标注）
        metasurface_n: 超表面网格尺寸
        camera_n: 相机网格尺寸
    """
    sign = "pos" if is_positive else "neg"
    
    # 计算对应的超表面单元数
    scale_factor = metasurface_n / camera_n
    patch_size_meta = int(patch_size * scale_factor)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    channel_cmaps = ["Reds", "Greens", "Blues"]
    
    for c in range(3):
        cmap = channel_cmaps[c]
        
        # 卷积核
        ax_k = axes[0, c]
        k_img = kernel[c]
        vmax_k = max(abs(k_img.min()), abs(k_img.max()), 1e-8)
        ax_k.imshow(k_img, cmap=cmap, vmin=0, vmax=vmax_k)
        ax_k.set_title(f"Kernel ch{c}")
        ax_k.axis("off")
        
        # PSF 中心区域（超表面单元尺度）
        ax_p = axes[1, c]
        n = psf.shape[1]
        center = n // 2
        margin = max(5, patch_size_meta // 3)
        crop = min((patch_size_meta + 2 * margin) // 2, center)
        psf_center = psf[c, center-crop:center+crop, center-crop:center+crop]
        vmax_p = psf_center.max() + 1e-12
        ax_p.imshow(psf_center, cmap=cmap, vmin=0, vmax=vmax_p)
        ax_p.set_title(f"PSF ch{c} ({crop*2}x{crop*2} meta units)")
        ax_p.axis("off")

    fig.suptitle(f"Kernel #{kernel_idx} ({sign})\n"
                 f"{patch_size}x{patch_size} camera pixels = "
                 f"~{patch_size_meta}x{patch_size_meta} metasurface units", 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """批量训练分割任务的 PSF。"""
    # ========== 配置选项 ==========
    DEBUG_SINGLE_KERNEL = False  # 设为 True 只训练第一个卷积核查看效果
    
    # 模型选择：
    # - "segmentation": 使用分割模型 (student2_noskip_32ch_best.pth)
    # - "multitask": 使用多任务模型 (multitask_best.pth)
    MODEL_TYPE = "segmentation"  # 可改为 "multitask"
    
    # ========== 训练超参数（可调整） ==========
    TRAINING_CONFIG = {
        'num_steps': 500,       # 训练步数
        'lr': 5e-3,             # 学习率
        'patience': 10,         # 早停patience
        'patch_size_camera': 6, # 相机像素数
    }
    
    print(f"\n{'='*70}")
    print("Training Configuration")
    print(f"{'='*70}")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")
    # ===============================
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # 加载配置
    config_path = os.path.join(base_dir, "configs", "segmentation.yaml")
    
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    cfg = load_segmentation_config(config_path, device=device)
    print(f"Config: n={cfg.n}, dx={cfg.dx*1e9:.1f}nm, f={cfg.f*1e6:.1f}um")
    
    # 创建模型以获取相机到超表面的缩放比例
    temp_model = TrainableSegmentationMetasurface(cfg)
    camera_to_meta_scale = temp_model.camera_to_meta_scale
    del temp_model

    # 根据模型类型选择checkpoint路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if MODEL_TYPE == "multitask":
        checkpoint_path = os.path.join(base_dir, "results", "multitask", "multitask_best.pth")
        output_dir = os.path.join(base_dir, "results", f"multitask_psf_{timestamp}")
    else:  # segmentation
        checkpoint_path = os.path.join(base_dir, "results", "student2_noskip_32ch_best.pth")
        output_dir = os.path.join(base_dir, "results", f"segmentation_psf_{timestamp}")
    
    print(f"\nOutput directory: {output_dir}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        if MODEL_TYPE == "multitask":
            print("Please train the multitask model first: python src/multitask/train_multitask.py")
        else:
            print("Please train the segmentation model first: python src/segmentation/unet_student2_noskip_32ch.py")
        return

    # 加载光学卷积核
    kernels = load_optical_kernels(checkpoint_path, model_type=MODEL_TYPE)  # (32, 3, 3, 3)
    
    # 分离正负卷积核
    kernels_pos, kernels_neg = split_positive_negative(kernels)
    
    # 填充全零通道（在归一化之前）
    print("\nChecking for zero channels...")
    kernels_pos, kernels_neg = fill_zero_channels(kernels_pos, kernels_neg)
    
    # 归一化
    kernels_pos, kernels_neg = normalize_kernels(kernels_pos, kernels_neg)
    
    num_kernels = kernels.shape[0]
    
    # 如果是调试模式，只训练第一个卷积核
    if DEBUG_SINGLE_KERNEL:
        num_kernels = 1
        print(f"[DEBUG MODE] Training only kernel #0 (pos + neg)")
    else:
        print(f"Training PSF for {num_kernels} kernels (pos + neg = {num_kernels * 2} total)")

    # 输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存分离后的卷积核
    np.save(os.path.join(output_dir, "kernels_pos.npy"), kernels_pos)
    np.save(os.path.join(output_dir, "kernels_neg.npy"), kernels_neg)
    print(f"Saved kernels to: {output_dir}")

    # PSF 训练参数
    # patch_size_camera: 相机像素数（用户输入）
    # patch_size_meta: 超表面单元数（内部计算）
    patch_size_camera = TRAINING_CONFIG['patch_size_camera']
    patch_size_meta = int(patch_size_camera * camera_to_meta_scale)  # 转换为超表面单元数
    
    print(f"\nPatch size: {patch_size_camera}×{patch_size_camera} camera pixels")
    print(f"           = {patch_size_meta}×{patch_size_meta} metasurface units")
    
    num_steps = TRAINING_CONFIG['num_steps']
    lr = TRAINING_CONFIG['lr']
    patience = TRAINING_CONFIG['patience']

    # 存储所有训练结果
    all_widths_pos = []
    all_widths_neg = []
    all_psfs_pos = []
    all_psfs_neg = []
    all_losses = []

    # 批量训练
    pbar = tqdm(range(num_kernels), desc="Training PSFs")
    for i in pbar:
        # 训练正核 PSF
        target_pos = build_target_field(kernels_pos[i], cfg.n, patch_size_meta)
        model_pos, loss_pos = train_single_psf(
            cfg, target_pos, patch_size_meta, num_steps, lr, patience
        )
        
        with torch.no_grad():
            width_pos = 80.0 + 240.0 * torch.sigmoid(model_pos.width_nm.cpu())
            psf_pos = model_pos().cpu().numpy()
        
        all_widths_pos.append(width_pos.numpy())
        all_psfs_pos.append(psf_pos)

        # 训练负核 PSF
        target_neg = build_target_field(kernels_neg[i], cfg.n, patch_size_meta)
        model_neg, loss_neg = train_single_psf(
            cfg, target_neg, patch_size_meta, num_steps, lr, patience
        )
        
        with torch.no_grad():
            width_neg = 80.0 + 240.0 * torch.sigmoid(model_neg.width_nm.cpu())
            psf_neg = model_neg().cpu().numpy()
        
        all_widths_neg.append(width_neg.numpy())
        all_psfs_neg.append(psf_neg)

        all_losses.append((loss_pos, loss_neg))
        pbar.set_postfix(loss_pos=f"{loss_pos:.3e}", loss_neg=f"{loss_neg:.3e}")

        # 可视化（调试模式下全部可视化，批量模式下只可视化前几个）
        if DEBUG_SINGLE_KERNEL or i < 4:
            visualize_kernel_and_psf(
                kernels_pos[i], psf_pos,
                os.path.join(output_dir, f"kernel_{i:02d}_pos.png"),
                i, is_positive=True, patch_size=patch_size_camera,
                metasurface_n=cfg.n, camera_n=int(cfg.n * cfg.dx / cfg.camera_pixel_size) if cfg.camera_pixel_size else cfg.n
            )
            visualize_kernel_and_psf(
                kernels_neg[i], psf_neg,
                os.path.join(output_dir, f"kernel_{i:02d}_neg.png"),
                i, is_positive=False, patch_size=patch_size_camera,
                metasurface_n=cfg.n, camera_n=int(cfg.n * cfg.dx / cfg.camera_pixel_size) if cfg.camera_pixel_size else cfg.n
            )

    # 保存所有结果
    all_widths_pos = np.stack(all_widths_pos, axis=0)  # (32, n, n)
    all_widths_neg = np.stack(all_widths_neg, axis=0)
    all_psfs_pos = np.stack(all_psfs_pos, axis=0)      # (32, 3, output_n, output_n)
    all_psfs_neg = np.stack(all_psfs_neg, axis=0)

    np.save(os.path.join(output_dir, "widths_pos.npy"), all_widths_pos)
    np.save(os.path.join(output_dir, "widths_neg.npy"), all_widths_neg)
    np.save(os.path.join(output_dir, "psfs_pos.npy"), all_psfs_pos)
    np.save(os.path.join(output_dir, "psfs_neg.npy"), all_psfs_neg)

    # 保存损失记录
    losses_arr = np.array(all_losses)
    np.save(os.path.join(output_dir, "losses.npy"), losses_arr)
    
    # 保存训练配置
    config_save = {
        'timestamp': timestamp,
        'model_type': MODEL_TYPE,
        'phase_method': 'fitted_polynomial',
        'training_config': TRAINING_CONFIG,
        'metasurface_config': {
            'n': cfg.n,
            'dx_nm': cfg.dx * 1e9,
            'f_um': cfg.f * 1e6,
            'wavelengths_nm': [w * 1e9 for w in cfg.wavelengths],
            'camera_pixel_nm': cfg.camera_pixel_size * 1e9 if cfg.camera_pixel_size else None,
        },
        'patch_size_camera': patch_size_camera,
        'patch_size_meta': patch_size_meta,
        'num_kernels': num_kernels,
    }
    
    import json
    with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
        json.dump(config_save, f, indent=2)
    
    print(f"\n✓ Saved training configuration to: training_config.json")

    print(f"\n✓ Training completed!")
    print(f"  Widths shape: {all_widths_pos.shape}")
    print(f"  PSFs shape: {all_psfs_pos.shape}")
    print(f"  Average loss (pos): {losses_arr[:, 0].mean():.4e}")
    print(f"  Average loss (neg): {losses_arr[:, 1].mean():.4e}")
    print(f"  Results saved to: {output_dir}")
    
    print(f"\n✓ To visualize results, run:")
    print(f"  python src/psf/visualize_psf_results.py")


if __name__ == "__main__":
    main()
