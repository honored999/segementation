# Standalone nnU-Net 2D Implementation Plan

## Goal

Build a self-contained PyTorch reproduction of the supplied 2D nnU-Net plan
for Dataset501_StrokeLesion, then add source-verified training and evaluation
in later phases.

## References and confirmed configuration

`reference/nnUNetPlans.json` is the sole architecture source: 512x512 patches,
batch size 12, Z-score normalization, spacing 0.4892368018627167 mm, and an
eight-stage PlainConvUNet with features 32/64/128/256/512/512/512/512.
`splits_final.json` supplies the fixed five folds (76 training and 19 validation
cases each). `summary.json` supplies the 95-case official baseline Dice
0.731103738314918 and IoU 0.5923877518050135.

The LeakyReLU slope is not in the plans and is explicitly set to PyTorch's
documented default, 0.01. Deep-supervision loss weighting, augmentation,
foreground oversampling, mirror inference, sliding-window inference, and exact
trainer schedule remain unconfirmed.

## Phased development

1. Completed foundation: validate references, inspect environment, parse the
   2D model configuration, implement model blocks, and test tensor shapes.
2. Completed read-only data phase: bounded NIfTI loading, in-plane resampling,
   full-image normalization, fixed-fold dataset access, and synthetic tests.
3. Source-verified optimization phase: reproduce loss, deep-supervision
   weighting, augmentation, sampling, optimizer, scheduler, and checkpoints.
4. Evaluation phase: implement 2D inference, five-fold validation, metrics,
   out-of-fold summary, and comparison against the supplied baseline.
5. Extension seam: expose a first-layer/encoder/decoder factory so optical
   convolution and distillation components can replace those modules without
   changing configuration parsing or the engine contract.

## Data safety and intended workflow

All external Dataset501 locations are read-only. The project will use the
existing split files directly, never reshuffle cases, and write all generated
logs, checkpoints, predictions, and plots beneath `outputs/`. Future training
will run one fixed fold at a time, collect out-of-fold predictions for all 95
cases, and compare their per-case mean Dice/IoU with the official baseline.

## Expected reproduction variance

Differences can arise from unconfirmed trainer defaults, augmentation,
sampling, deep-supervision weights, resampling details, random seeds, hardware,
and library versions. No claim of matching the baseline is made before training
and five-fold evaluation are actually performed.
