# Reproduction Notes

## Directly supplied by `nnUNetPlans.json`

The 2D patch size, batch size, spacing, normalization, mask-for-normalization,
resampling names/kwargs, PlainConvUNet class name, eight stages, feature list,
kernels, strides, encoder/decoder convolution counts, convolution bias,
InstanceNorm2d settings, no-dropout setting, LeakyReLU inplace setting, and
`batch_dice=true` are read directly from the reference JSON.

## Fold and benchmark provenance

`splits_final.json` is used without random re-splitting: five folds each have
76 train and 19 validation cases, with 95 unique validation cases exactly once.
`summary.json` supplies the official 95-case foreground Dice baseline
0.731103738314918 and IoU baseline 0.5923877518050135.

## Defaults and outstanding confirmation

The plans omit `LeakyReLU.negative_slope`; the implementation uses 0.01, the
PyTorch `nn.LeakyReLU` default observed in the current `newconda` environment.
The `newconda` environment did not contain `nnunetv2` or
`dynamic_network_architectures` during the initial inspection. The requested
`nnunet5090` Conda environment was also unavailable at its declared location,
so official source could not be used to independently confirm the exact
deep-supervision head selection/order or trainer defaults. The current
implementation returns heads in full-to-low-resolution order: 512, 256, 128,
64, 32, 16, then 8 pixels.

Augmentation, foreground oversampling, inference mirroring, sliding-window
configuration, deep-supervision loss weights, optimization schedule, and exact
preprocessing mechanics are intentionally unimplemented until they are checked
against a compatible official source installation. Current differences from
official nnU-Net are therefore limited to this model-only foundation and the
explicitly deferred components above.

## Read-only data pipeline

The current data pipeline reads one explicitly requested image/label pair at a
time, uses the fixed supplied folds, resamples x/y to the plan spacing, applies
full-image Z-score normalization, and extracts a deterministic central axial
slice. It does not cache volumes, sample foreground slices, augment images, or
start training. Segmentation resampling uses linear interpolation followed by
rounding to preserve discrete labels; this is a practical interpretation of the
plan's segmentation `order=1` until the official preprocessing source is
available.
