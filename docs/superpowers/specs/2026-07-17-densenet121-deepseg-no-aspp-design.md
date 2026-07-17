# DenseNet121 DeepSeg Without ASPP Design

## Goal

Add a paper-aligned electronic DeepSeg experiment that replaces the current
MobileNetV2 encoder with ImageNet-initialized DenseNet121 and removes the
DeepLabV3+ ASPP context module. It must be directly comparable with the
existing MobileNetV2 plus ASPP DeepSeg-decoder baseline on CI-1 DWI slices.

## Evidence and Scope

Zeineldin et al. (2020), *DeepSeg: deep neural network framework for
automatic brain tumor segmentation using magnetic resonance FLAIR images*,
reports DenseNet as the strongest tested encoder: DSC 0.839 in its training
cross-validation and 0.841 on the BraTS 2019 validation data. The paper's
DeepSeg architecture couples an encoder with a modified U-Net decoder and
same-scale skip connections; it does not include an ASPP module.

This work changes neither the existing `electronic_deepseg_decoder` model nor
its MobileNetV2 plus ASPP configuration. It adds one explicit experiment
configuration and model path, preserving the dataset, five-fold patient split,
loss, sampling, training schedule, evaluation, checkpointing, and inference
contracts already used by `optical_deeplab2d`.

## Architecture

The new `ElectronicDenseNetDeepSegDecoder` accepts a single-channel DWI image,
repeats it to three channels, and sends it through a
`segmentation_models_pytorch` DenseNet121 encoder with `depth=4`.
The deepest encoder feature map is passed directly to the existing four-stage
modified U-Net decoder: there is no ASPP, atrous convolution, or other
DeepLabV3+ context component. Each decoder stage bilinearly upsamples to its
real encoder skip's spatial shape, concatenates it with that skip, then
applies the existing two `Conv2d -> BatchNorm2d -> ReLU` refinement blocks.
The head returns one unthresholded logit channel at exactly the input spatial
size.

With four encoder downsampling operations, the deepest feature is at 1/16
resolution and the decoder consumes 1/8, 1/4, 1/2, and input-resolution
skips in order. Decoder input and stage widths are inferred from the DenseNet
encoder's reported `out_channels`, rather than using MobileNetV2-specific
channel constants. This keeps skip alignment and parameter dimensions correct
for the new encoder while retaining the 128, 64, 32, and 32 decoder widths
prescribed by the existing DeepSeg experiment.

## Configuration and Reproducibility

Add `configs/electronic_densenet121_deepseg_no_aspp_6gb_smoke.yaml`. It uses
`type: electronic_densenet121_deepseg_no_aspp`, `encoder_name: densenet121`,
and `encoder_weights: imagenet`; all data and training values are copied from
the current electronic DeepSeg smoke configuration. Its resolved configuration
and checkpoints record `encoder_name: densenet121` and
`context_module: none`.

The existing `electronic_deepseg_decoder` configuration remains unchanged and
continues to record its resolved encoder and ASPP-backed architecture. Runs
must use the same fold, seed, patient-level split, data root, epoch count,
batch size, loss, sampling and threshold as the baseline. The primary
comparison metric is mean patient Dice; global and mean-image Dice remain
supporting metrics.

## Error Handling

DenseNet121 creation must fail clearly if the installed SMP version cannot
construct it. It must not silently fall back to MobileNetV2, ResNet18, or any
other architecture, because that would invalidate the requested comparison.
Checkpoint metadata must be derived from the instantiated model so artifacts
cannot claim DenseNet121 when another model ran.

## Test Strategy

Tests are written before production code and prove that the new model:

- produces one logit channel at the exact input size for a non-square,
  odd-sized grayscale tensor;
- exposes `resolved_encoder == "densenet121"` and
  `context_module == "none"`;
- contains no ASPP module or parameter name;
- uses real multi-scale DenseNet skips and propagates finite gradients through
  the encoder and all decoder stages; and
- is selected by the new smoke configuration while the current MobileNetV2
  configuration remains unchanged.

The targeted model and CLI tests, followed by the full package suite, run in
`newconda`. A short server smoke run and the full five-fold training are
documented experiment procedures, not unit-test claims.
