"""Foreground-aware 2D patch center sampling for formal Trainer alignment."""
from __future__ import annotations
import numpy as np

def sample_patch_center(label: np.ndarray, rng: np.random.Generator, *, oversample_foreground_percent: float=.33) -> tuple[int,int]:
 if label.ndim!=2 or not 0<=oversample_foreground_percent<=1: raise ValueError('expected 2D label and probability in [0,1]')
 foreground=np.argwhere(label>0)
 if len(foreground) and rng.random()<oversample_foreground_percent:
  y,x=foreground[rng.integers(len(foreground))]; return int(y),int(x)
 return int(rng.integers(label.shape[0])),int(rng.integers(label.shape[1]))

def crop_or_pad(image: np.ndarray, label: np.ndarray, center: tuple[int,int], patch_size: tuple[int,int]=(512,512)) -> tuple[np.ndarray,np.ndarray]:
 if image.shape!=label.shape or image.ndim!=2: raise ValueError('image and label must be equal-shaped 2D arrays')
 out_image=np.zeros(patch_size,dtype=image.dtype); out_label=np.zeros(patch_size,dtype=label.dtype); y,x=center; h,w=patch_size
 y0,x0=y-h//2,x-w//2; y1,x1=y0+h,x0+w; sy0,sx0=max(0,y0),max(0,x0); sy1,sx1=min(image.shape[0],y1),min(image.shape[1],x1); dy0,dx0=sy0-y0,sx0-x0
 out_image[dy0:dy0+sy1-sy0,dx0:dx0+sx1-sx0]=image[sy0:sy1,sx0:sx1]; out_label[dy0:dy0+sy1-sy0,dx0:dx0+sx1-sx0]=label[sy0:sy1,sx0:sx1]
 return out_image,out_label
