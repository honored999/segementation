# Configurable Augmentation and Slice Sampling Design

## Scope

Implement deterministic, caller-configured 2D augmentation and foreground-aware
slice-index selection for already loaded arrays. No official nnU-Net parameter
claim, data-cache generation, optimizer, or training run is included.

## Sampling

`select_slice_index(label_volume, rng, foreground_probability)` accepts a
`(z, y, x)` integer label volume and a caller-supplied NumPy generator. With
probability zero it chooses a uniformly random valid slice. With positive
probability it chooses uniformly among slices containing foreground when any
exist, otherwise it falls back to uniform sampling. The probability is checked
to be within `[0, 1]`; output is always an in-range Python integer.

## Augmentation

`augment_slice(image, label, rng, config)` accepts matched 2D arrays. It may
apply horizontal/vertical flips and a multiplicative intensity scale to image
only; geometric flips always apply identically to image and label. Labels are
never interpolated or intensity transformed. `AugmentationConfig()` defaults to
all probabilities/scales disabled, so the pipeline is unchanged until a caller
explicitly enables it.

## Tests and boundaries

Tests use seeded generators to verify reproducible foreground selection,
no-foreground fallback, bounds checking, synchronized flips, label integrity,
and identity defaults. The Dataset will optionally receive sampler/config
objects but continues to load just one case on demand. These values are local,
not source-verified official nnU-Net augmentation or oversampling defaults.
