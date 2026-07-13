"""Explicit paired augmentation construction."""
from __future__ import annotations
def build_transforms(training: bool, image_size: tuple[int, int] | None = None):
    try: import albumentations as A
    except ImportError as error: raise RuntimeError("Albumentations is required for configured transforms; install requirements.txt on the server.") from error
    steps = []
    if image_size: steps.append(A.Resize(*image_size, interpolation=1, mask_interpolation=0))
    if training: steps += [A.HorizontalFlip(p=.5), A.ShiftScaleRotate(shift_limit=.05, scale_limit=.1, rotate_limit=10, interpolation=1, mask_interpolation=0, p=.8), A.RandomGamma((80,120),p=.3), A.GaussNoise(std_range=(0,.03),p=.3)]
    return A.Compose(steps)

