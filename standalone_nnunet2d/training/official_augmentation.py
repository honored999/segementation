"""Confirmed 2D nnU-Net augmentation configuration and paired transforms."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import rotate, zoom, gaussian_filter
from standalone_nnunet2d.training.patch_sampler import crop_or_pad

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

def apply_official_2d_augmentation(image: np.ndarray,label: np.ndarray,rng: np.random.Generator,patch_size: tuple[int,int]=(512,512)) -> tuple[np.ndarray,np.ndarray]:
 if rng.random()<.2: image,label=rotate_pair(image,label,np.degrees(rng.uniform(-np.pi,np.pi)))
 if rng.random()<.2: image,label=scale_pair(image,label,rng.uniform(.7,1.4),patch_size)
 else: image,label=crop_or_pad(image,label,(image.shape[0]//2,image.shape[1]//2),patch_size)
 axes=tuple(axis for axis in (0,1) if rng.random()<.5)
 if axes: image,label=mirror_pair(image,label,axes)
 image=intensity_augment(image,rng)
 if rng.random()<.25: image=simulate_low_resolution(image,rng.uniform(.5,1.))
 return image,label
