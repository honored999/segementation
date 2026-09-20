# Status

Updated: 2026-09-20

## Current state

- The active feature branch contains standalone 2D PlainConvUNet and H2Former
  model families with explicit model/supervision checkpoint identities.
- Existing H2Former uses single-output supervision; PlainConvUNet supports its
  default deep supervision and a matched single-output mode.

## Verified constraints

- Dataset501 remains the established DWI-only baseline.
- Preserve the existing patient-level five-fold split and original full-volume
  formal-evaluation semantics.
- Synthetic validation is engineering evidence only.

## Active design

- A design is approved for independent `h2former_lite_upernet` and
  `plain_conv_unet_lite_upernet` single-output variants. The written design is
  approved and the implementation plan is ready; implementation has not started.
