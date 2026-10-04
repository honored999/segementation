# Goals

## Branch-local milestone (unintegrated; 2026-10-03)

- Deliver manual plans-configured encoder selections for official UPerNet with reproducible checkpoint/predictor identity. Implementation and independent Level 3 review completed (PASS; P1 CLOSED). Branch remains unmerged; server/performance/formal evidence pending.

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
