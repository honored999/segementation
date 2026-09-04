# Single-Case Overfit Diagnostic Design

## Goal

Diagnose data/model/loss/gradient correctness by overfitting one explicit
non-empty lesion case; this is smoke-only and never a performance result.

## Contract

An explicit `--overfit-one-case CASE_ID` command uses only that case's axial
`(z,y,x)` slices, disables augmentation, fixes a seed, and runs 200 iterations
with smoke-only `SGD(0.01, 0.9, 0)`. Every 20 iterations it logs loss, full-case
Dice, GT voxels, and predicted voxels. It writes the final NIfTI and overlay
under `outputs/` with `smoke_run_only=true`.

## Failure Rule

If final Dice remains near the initial/random value, the command exits nonzero
and reports likely diagnostic categories: image/label pairing, axis order,
label values, loss/argmax dimension, gradient updates, interpolation, or
normalization. It does not proceed to fold validation.
