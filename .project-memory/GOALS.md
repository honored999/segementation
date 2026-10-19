# Goals

## Project goal

- Develop reproducible stroke-lesion segmentation models while preserving
  medical-data safety, fixed patient splits, and formal evaluation boundaries.

## Current milestone

- Add isolated lightweight UPerNet-style decoder variants for H2Former and
  PlainConvUNet without changing either baseline.

## Success criteria

- Each A/B pair changes only the decoder architecture under matched
  single-output supervision and existing training/data policies.
- New model and checkpoint identities cannot be confused with baseline models.
- Synthetic tests establish shape, gradient, compatibility, and complexity
  contracts before any real-data experiment is considered.

## Branch-local priority (2026-10-19)

- First screen fixed HD-BET3mm real-DWI preprocessing with ordinary-deep-supervision TopK10+early stopping on existing fold0 (76/19); use matched baseline and original-space full-volume metrics before any benefit claim.
- Preserve Dataset501/raw/splits/oldweights and pause expansion of failed gray GenSeg outputs. Original-DWI conditioning/residual or lesion-template synthesis remain candidate future designs after this preprocessing screen.