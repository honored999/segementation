# Lite UPerNet Decoder Variants Design

Date: 2026-09-20
Status: Approved design pending implementation

## Goal

Add two independent A/B model variants without changing either existing model:

- `h2former_lite_upernet`
- `plain_conv_unet_lite_upernet`

Each variant keeps its corresponding encoder and replaces only the decoder with
a lightweight UPerNet-style decoder. Both comparisons use `single_output` so
that supervision mode is held constant within each A/B pair.

## Scientific and compatibility boundaries

- Preserve the existing patient-level five-fold split, data, sampling,
  augmentation, loss, optimizer, scheduler, batch size, validation, checkpoint
  selection, and inference semantics.
- Do not modify the existing `h2former` or `plain_conv_unet` construction paths,
  defaults, outputs, or checkpoint identities.
- Use distinct model identities for both new variants. Old and new checkpoints
  must be rejected before `load_state_dict` when their model identities differ.
- Engineering validation uses synthetic inputs only. No real medical data,
  training, preflight, or formal evaluation is part of this task.
- Parameter and FLOP measurements are engineering complexity evidence, not
  segmentation-performance evidence.

## Source fidelity

The ECCV 2018 UPerNet design specifies a Pyramid Pooling Module (PPM) on the
highest-level feature and a Feature Pyramid Network (FPN) that performs lateral,
top-down, and multi-scale fusion. The official implementation uses pool scales
`(1, 2, 3, 6)` and provides an `upernet_lite` configuration with 256 FPN
channels.

This project deliberately deviates from that implementation:

- FPN width is 64 channels.
- PPM scales are `(1, 2, 4)` so the 4x4 deepest nnU-Net feature does not request
  a larger 6x6 adaptive pooling output.
- The head performs binary stroke-lesion segmentation only; it does not
  reproduce UPerNet's multi-task scene/object/part/material heads.
- Backbone-specific feature selections and normalization preserve this
  project's existing encoder conventions.

These variants are therefore lightweight UPerNet-style decoders, not exact
numerical reproductions of the published network.

Primary references:

- https://arxiv.org/abs/1807.10221
- https://github.com/CSAILVision/unifiedparsing/blob/master/models/models.py

## Shared decoder

Introduce one reusable `LiteUPerDecoder` with explicit constructor inputs:

- four input-channel counts;
- output class count;
- `fpn_channels=64`;
- `pool_scales=(1, 2, 4)`;
- a normalization factory;
- interpolation mode fixed to bilinear with `align_corners=False`.

The decoder performs:

1. PPM on the deepest feature.
2. A 1x1 projection of each lateral feature to 64 channels.
3. Top-down interpolation to the exact lateral spatial size followed by
   elementwise addition.
4. A 3x3 refinement convolution for each pyramid level.
5. Interpolation of all four refined levels to the highest selected resolution.
6. Concatenation, 3x3 fusion to 64 channels, and a 1x1 two-class logits head.
7. Final interpolation to the exact input spatial size.

The first implementation uses ordinary convolutions rather than depthwise
separable convolutions. This limits simultaneous changes and provides a stable
reference before any later ultra-light ablation.

## H2Former variant

Keep the existing H2Former encoder and its four outputs:

| Level | Resolution for 512x512 input | Channels |
| --- | ---: | ---: |
| 1 | 256x256 | 64 |
| 2 | 128x128 | 128 |
| 3 | 64x64 | 256 |
| 4 | 32x32 | 512 |

Replace `decode4`, `decode3`, `decode2`, and `decode0` only in the new variant.
The decoder normalization is `BatchNorm2d`, matching the current H2Former
decoder convention. The output remains one `[B, 2, 512, 512]` logits tensor.

## PlainConvUNet variant

Keep all eight existing encoder stages. Supply encoder outputs with indices
`(1, 3, 5, 7)` to the shared decoder:

| Encoder index | Resolution for 512x512 input | Channels |
| --- | ---: | ---: |
| 1 | 256x256 | 64 |
| 3 | 64x64 | 256 |
| 5 | 16x16 | 512 |
| 7 | 4x4 | 512 |

Although only four outputs become lateral inputs, every encoder stage remains
on the sequential path to the deepest feature and receives gradients. The
decoder normalization is `InstanceNorm2d` with the epsilon and affine settings
from the existing nnU-Net plan. The output is one full-resolution logits tensor;
the new variant does not expose deep-supervision outputs.

## Model and checkpoint contracts

Register both names as explicit `single_output` model contracts in the model
factory and CLI choices. Resolved configuration and checkpoint metadata must
contain the exact new model name and `single_output` supervision mode.

Inference reconstructs the same architecture from checkpoint metadata. A
checkpoint from either baseline, the other Lite-UPerNet variant, or a different
supervision mode must fail identity validation before any state dictionary is
loaded.

## Validation and errors

`LiteUPerDecoder` rejects:

- a feature count other than four;
- non-BCHW features;
- mismatched batch dimensions;
- channels that differ from the declared input channels;
- feature maps not ordered from highest to lowest spatial resolution;
- a requested output size that is invalid.

Synthetic tests must cover:

- output shape `[B, 2, 512, 512]` for both variants;
- finite logits, loss, and gradients;
- gradients on representative early, intermediate, deepest, and decoder
  parameters;
- factory, CLI, resolved-config, inference, and checkpoint identities;
- rejection of old/new and cross-variant checkpoint mismatches before state
  loading;
- unchanged construction and output contracts for both baseline models;
- decoder and total parameter counts plus fixed-input FLOP reporting.

The Lite-UPerNet decoder parameter count must be lower than the corresponding
existing decoder. FLOP collection must use the same input shape and measurement
method for each comparison. These checks establish implementation and complexity
contracts only.

## Out of scope

- Real-data training, preflight, fold evaluation, or five-fold OOF evaluation.
- Changing encoders, losses, optimizers, schedules, data handling, or inference
  postprocessing.
- Importing the original UPerNet package or adding a new dependency.
- Loading baseline decoder weights into a Lite-UPerNet decoder.
- Depthwise-separable, attention-based, or additional decoder variants.

## Delegation and acceptance

All implementation, reviewer, fixer, and validator subagents for this task are
manually dispatched by the user. No worker may create nested subagents. The
main agent remains responsible for integration and evidence-based acceptance.

Implementation should be handled as one scoped worker lane, followed by an
independent Level 3 read-only review because the change affects model/checkpoint
contracts. Any required fixer and final validator are also manually dispatched.
