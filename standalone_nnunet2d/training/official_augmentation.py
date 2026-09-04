"""Confirmed 2D nnU-Net augmentation configuration and paired transforms."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy.ndimage import rotate, zoom, gaussian_filter
from standalone_nnunet2d.training.patch_sampler import crop_or_pad

from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

@dataclass(frozen=True)
class Official2DAugmentationConfig:
 rotation_radians: tuple[float,float]=(-np.pi,np.pi)
 rotation_probability: float=.2
 scaling_range: tuple[float,float]=(.7,1.4)
 scaling_probability: float=.2
 noise_probability: float=.1
 blur_probability: float=.2
 brightness_probability: float=.15
 contrast_probability: float=.15
 low_resolution_probability: float=.25
 gamma_invert_probability: float=.1
 gamma_probability: float=.3
 mirror_axes: tuple[int,int]=(0,1)

def mirror_pair(image: np.ndarray,label: np.ndarray,axes: tuple[int,...]) -> tuple[np.ndarray,np.ndarray]:
 if image.shape!=label.shape or image.ndim!=2: raise ValueError('image and label must be equal-shaped 2D arrays')
 return np.ascontiguousarray(np.flip(image,axes)),np.ascontiguousarray(np.flip(label,axes))

def rotate_pair(image: np.ndarray,label: np.ndarray,angle_degrees: float) -> tuple[np.ndarray,np.ndarray]:
 if image.shape!=label.shape or image.ndim!=2: raise ValueError('image and label must be equal-shaped 2D arrays')
 return rotate(image,angle_degrees,reshape=False,order=3,mode='constant',cval=0.0),rotate(label,angle_degrees,reshape=False,order=0,mode='constant',cval=-1)

def scale_pair(image: np.ndarray,label: np.ndarray,scale: float,patch_size: tuple[int,int]) -> tuple[np.ndarray,np.ndarray]:
 if scale<=0: raise ValueError('scale must be positive')
 scaled_image=zoom(image,scale,order=3); scaled_label=zoom(label,scale,order=0)
 return crop_or_pad(scaled_image,scaled_label,(scaled_image.shape[0]//2,scaled_image.shape[1]//2),patch_size)

def intensity_augment(image: np.ndarray,rng: np.random.Generator) -> np.ndarray:
 result=image.astype(np.float32,copy=True)
 if rng.random()<.1: result+=rng.normal(0,rng.uniform(0,.1),result.shape)
 if rng.random()<.2: result=gaussian_filter(result,rng.uniform(.5,1.))
 if rng.random()<.15: result*=rng.uniform(.75,1.25)
 if rng.random()<.15:
  mean=result.mean(); result=(result-mean)*rng.uniform(.75,1.25)+mean
 if rng.random()<.1: result=-result
 if rng.random()<.3:
  mean,std=result.mean(),result.std(); result=np.sign(result-mean)*np.abs(result-mean)**rng.uniform(.7,1.5); result=(result-result.mean())/(result.std()+1e-8)*std+mean
 return result.astype(np.float32)

def simulate_low_resolution(image: np.ndarray,scale: float) -> np.ndarray:
 if not .5<=scale<=1: raise ValueError('low-resolution scale must be in [0.5,1]')
 low=zoom(image,scale,order=1); restored=zoom(low,(image.shape[0]/low.shape[0],image.shape[1]/low.shape[1]),order=1); output=np.zeros_like(image,dtype=np.float32); h,w=min(image.shape[0],restored.shape[0]),min(image.shape[1],restored.shape[1]); output[:h,:w]=restored[:h,:w]; return output

def rotation_for_patch_size(patch_size: tuple[int, int]) -> tuple[float, float]:
    if len(patch_size) != 2:
        raise ValueError("2D patch_size must contain exactly two dimensions")
    if max(patch_size) / min(patch_size) > 1.5:
        limit = 15.0 / 360.0 * 2.0 * np.pi
    else:
        limit = 180.0 / 360.0 * 2.0 * np.pi
    return -limit, limit

def apply_official_2d_augmentation(image: np.ndarray,label: np.ndarray,rng: np.random.Generator,patch_size: tuple[int,int]=(512,512)) -> tuple[np.ndarray,np.ndarray]:
 if rng.random()<.2: image,label=rotate_pair(image,label,np.degrees(rng.uniform(-np.pi,np.pi)))
 if rng.random()<.2: image,label=scale_pair(image,label,rng.uniform(.7,1.4),patch_size)
 else: image,label=crop_or_pad(image,label,(image.shape[0]//2,image.shape[1]//2),patch_size)
 axes=tuple(axis for axis in (0,1) if rng.random()<.5)
 if axes: image,label=mirror_pair(image,label,axes)
 image=intensity_augment(image,rng)
 if rng.random()<.25: image=simulate_low_resolution(image,rng.uniform(.5,1.))
 return image,label


def apply_official_2d_batchgeneratorsv2(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: tuple[int, int],
    use_mask_for_norm: tuple[bool, ...] | list[bool] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the nnU-Net v2 2D training transform pipeline via batchgeneratorsv2."""
    if image.ndim != 2 or label.ndim != 2 or image.shape != label.shape:
        raise ValueError("image and label must be equal-shaped 2D arrays")
    data = torch.as_tensor(np.ascontiguousarray(image[None]).copy(), dtype=torch.float32)
    segmentation = torch.as_tensor(np.ascontiguousarray(label[None]).copy())
    rotation = rotation_for_patch_size(patch_size)
    transforms = ComposeTransforms([
        SpatialTransform(
            patch_size,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=0.2,
            rotation=rotation,
            p_scaling=0.2,
            scaling=(0.7, 1.4),
            p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False,
            border_mode_seg="constant",
            padding_value_seg=-1,
        ),
        RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True,
            ),
            apply_probability=0.1,
        ),
        RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.0),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5,
                benchmark=True,
            ),
            apply_probability=0.2,
        ),
        RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1,
            ),
            apply_probability=0.15,
        ),
        RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1,
            ),
            apply_probability=0.15,
        ),
        RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5,
            ),
            apply_probability=0.25,
        ),
        RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            ),
            apply_probability=0.1,
        ),
        RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            ),
            apply_probability=0.3,
        ),
        MirrorTransform(allowed_axes=(0, 1)),
        *([
            MaskImageTransform(
                apply_to_channels=[i for i, use in enumerate(use_mask_for_norm) if use],
                channel_idx_in_seg=0,
                set_outside_to=0,
            )
        ] if use_mask_for_norm is not None and any(use_mask_for_norm) else []),
        RemoveLabelTansform(-1, 0),
    ])
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    result = transforms(image=data, segmentation=segmentation)
    return (
        result["image"].detach().cpu().numpy()[0],
        result["segmentation"].detach().cpu().numpy()[0],
    )
