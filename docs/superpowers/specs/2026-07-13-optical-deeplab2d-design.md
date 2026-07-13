# Optical DeepLab2D Design

## Goal

Add an independent, server-ready experiment module for binary segmentation of
2D CI-1 DWI lesions. It compares a trainable ideal single-layer optical
convolution front end against an electronic-only DeepLabV3+ baseline, without
altering the existing U-Net experiments or source data.

## Confirmed Dataset Facts

- Primary data root: `data/ci1_dwi_2d_dedup`.
- The manifest contains 2,445 image-mask pairs from 25 patients at D1, D2,
  D3, D7 and D14. All inspected images are 512 by 512 PNG files.
- The manifest's `patient` column is the authoritative patient identifier.
  It keeps all timepoints for the same person together; the hashed image
  filename must not be parsed for patient identity.
- The set includes 1,110 empty masks and 1,335 positive masks. Empty masks
  remain in the experiment.
- Existing U-Net training computes an ephemeral 80/20 patient split using
  seed 42 and does not persist a reusable split. This module therefore creates
  and persists its own five-fold patient-level splits using seed 2026.

## Module Boundaries

The new `optical_deeplab2d` package owns its data inspection, manifest
validation, patient splitting, normalization, augmentation, models, training,
evaluation, inference and comparison scripts. It reads image/mask paths from
the existing manifest and never writes into `data/` or existing `results/`.

`datasets` performs path validation, modality-preserving grayscale decoding,
binary mask conversion, configurable resize and paired transforms. `models`
contains the optical layer and the two model wrappers. `training` contains
losses, reproducibility, checkpointing and the training loop. `evaluation`
contains metrics, visualizations and optional post-processing. CLI entrypoints
only orchestrate these units.

## Data and Split Design

Samples are always image `[1,H,W]` and mask `[1,H,W]`, `float32`; network
input is DWI only. Image and mask paths must form a one-to-one pairing, share
spatial dimensions, be finite and nonempty. Invalid samples cause a named
error rather than silent skipping. Masks are binarized with `mask > 0` before
and after any resize or spatial transform.

The default retains native 512 by 512 spatial resolution. An explicit
`image_size` configuration may resize images with bilinear interpolation and
masks with nearest-neighbor interpolation. Normalization is configured and
recorded in checkpoints; the default uses robust percentile statistics fitted
on training images, never independent per-image min-max normalization.

`split.py` will group manifest rows by `patient`, produce reproducible five
folds at seed 2026, persist `splits_final.json`, and report train/validation
patients plus total, positive and negative slice counts. Every timepoint for
one patient must belong to the same fold. Training may reuse an existing
identical split file to ensure baseline fairness.

## Models

`HybridOpticalDeepLabV3Plus` has the exact front end:

`1 -> Conv2d(8, kernel 5, stride 1, padding 2, bias False) -> GroupNorm(4,8)
-> ReLU -> Conv2d(8,3, kernel 1, bias False) -> BatchNorm2d(3) -> ReLU`.

The first convolution is initialized with Kaiming normal and is an ideal,
signed, trainable optical convolution; no PSF decomposition, phase mapping,
RCWA, optical noise or fabrication constraints are included. The back end is
`segmentation_models_pytorch.DeepLabV3Plus` with a MobileNetV2 encoder,
ImageNet weights, output stride 16 and atrous rates `(12, 24, 36)` where the
installed SMP version supports it. Otherwise the module logs a clear warning
and falls back to ResNet18, recording the resolved encoder in all artifacts.

`ElectronicDeepLabV3Plus` repeats the grayscale input from one to three
channels, then applies the same DeepLabV3+ back end. Both models return
unthresholded logits resized to the original input spatial size. The model
never applies sigmoid internally.

## Training and Evaluation

`CombinedBCEDiceLoss` is `0.5 * BCEWithLogitsLoss + 0.5 * SoftDiceLoss`.
Dice is computed per sample then averaged, uses sigmoid logits and smooth
`1e-5`, and remains finite for empty masks. The BCE `pos_weight` is calculated
from training pixels, clipped to `[1,20]`, and persisted.

When both classes exist, the training loader uses `WeightedRandomSampler` to
approximate a 50/50 positive/negative slice composition; validation traverses
all validation rows. The two models share split, normalization, augmentation,
batch size, epochs, optimizer, scheduler, threshold, seed and metrics.

Defaults are seed 2026, 100 epochs, batch size 8, AdamW, encoder/new-layer
learning rates `1e-4`/`5e-4`, weight decay `1e-4`, AMP, gradient clipping 5,
early stopping patience 20 and `ReduceLROnPlateau` with factor 0.5/patience 6.
CUDA OOM is surfaced with actionable advice and no unannounced image resize.

Evaluation saves global-pixel, mean-per-image and merged-patient metrics for
Dice, IoU, precision, recall, specificity, false-positive pixels and lesion
areas. Patient-level metrics concatenate all slices per patient before metric
calculation and use the explicitly specified empty-mask Dice rules. The best
checkpoint is selected by mean patient Dice.

## Artifacts and Server Workflow

Runs write only to the supplied output directory and include `best.pt`,
`last.pt`, resolved configuration, split, logs, dataset/environment/model
reports, CSV/JSON metrics, selected visualizations and optical-kernel
statistics. Checkpoints capture states, resolved model metadata, threshold,
fold, seed, normalization, pos-weight, pairing rule and patient-ID rule.

`infer_image.py` loads every relevant behavior from the checkpoint and writes
probability, binary prediction, overlay, NumPy prediction and metadata. The
README documents setup, server commands, training/evaluation/inference and the
manifest-based patient-ID rule. `requirements.txt` declares SMP and
Albumentations because they are absent from local `newconda`.

Local verification is intentionally limited to imports, configuration,
small-tensor unit tests and selected data checks. Full 512-sized model tests,
100-step overfit and any formal training remain documented server procedures.

## Test Strategy

Tests are written before production code. They cover manifest pairing and
patient grouping, binary mask behavior, split leakage prevention, loss edge
cases, model output shapes on small tensors, optical-weight gradient health,
checkpoint metadata, patient metric aggregation and inference metadata use.
The documented server preflight additionally runs the mandated 512 by 512
shape/gradient checks and small-batch overfit test before formal training.
