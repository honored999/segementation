# Electronic DeepSeg Decoder Design

## Goal

Replace only the decoder of the electronic DeepLabV3+ baseline with the
modified U-Net decoder described by Zeineldin et al. (2020). The experiment
must remain entirely electronic: no optical convolution, optical adapter, or
optical parameters are present in the selected model.

## Fixed Scope

- Keep the existing ImageNet-initialized MobileNetV2 encoder (or the existing
  recorded encoder fallback) and DeepLab ASPP context module.
- Keep the existing DWI data pipeline, patient-level folds, loss, metrics,
  thresholding, checkpoint protocol, and training CLI.
- Add a new, explicit electronic model/configuration; do not repurpose the
  existing electronic DeepLabV3+ baseline or alter the hybrid optical model.
- Preserve unthresholded single-channel logits at the input spatial size.

## Architecture

The new model repeats the one-channel DWI image to three channels and feeds it
through the existing MobileNetV2 encoder. It takes the encoder feature maps at
1/2, 1/4, 1/8, and 1/16 input resolution. The 1/16 feature passes through a
DeepLab-style ASPP context module. A four-stage modified U-Net decoder then
recovers full resolution:

1. Bilinearly upsample the current decoder feature to the corresponding skip
   resolution.
2. Concatenate it with the encoder feature at that resolution; spatially align
   only when an odd input size creates a one-pixel mismatch.
3. Apply two sequential 3-by-3 convolution blocks, each `Conv2d ->
   BatchNorm2d -> ReLU`.

The decoder stage widths are 128, 64, 32, and 32. A final 1-by-1 convolution
produces one logit channel. This implements the paper's decoder changes:
batch normalization before each ReLU and a 32-channel base width, while using
the retained MobileNetV2 + ASPP encoder as the feature extractor.

## Component Boundaries

- A backbone factory exposes the SMP MobileNetV2 encoder and ASPP context
  separately, with the current `resnet18` fallback semantics retained.
- A reusable modified-U-Net decoder stage owns one upsample, one skip
  concatenation, and its two convolution-normalization-activation blocks.
- `ElectronicDeepSegDecoder` owns only the electronic input adaptation,
  encoder, ASPP, decoder, and output head.
- Model selection in `train.py` recognizes the new model type without changing
  the existing baseline and hybrid paths.

## Configuration and Reproducibility

Add a configuration named for the electronic DeepSeg decoder. It uses the
same seed, image settings, training schedule, and model encoder settings as
the electronic scratch smoke configuration so results are comparable. The
resolved configuration and checkpoint metadata record the new model type and
resolved encoder name.

## Tests and Acceptance Criteria

Tests are written before implementation and demonstrate that:

- a 1-channel, non-square tensor returns one logit channel at exactly the
  original spatial dimensions;
- all four decoder stages contain two Conv2d + BatchNorm2d + ReLU sequences;
- the decoder consumes real multi-scale encoder skips rather than a decoder
  output copied into every stage;
- gradients flow to the encoder, ASPP, and decoder parameters; and
- the pure-electronic model contains no optical module or parameter name.

The targeted tests and package test suite must pass in `newconda`. A smoke
training command remains a follow-up experiment rather than a unit-test claim.
