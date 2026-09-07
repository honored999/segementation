# Status

Updated: 2026-09-07

## Current state
- Dataset501 remains the established read-only DWI-only baseline with fixed patient-level folds.
- The isolated ADN transform implementation remains unchanged from baseline commit `f75a807`.
- A diagnostic-only NIfTI adapter, Fold 0 ADN training CLI, checkpoint contract, and single-case QC CLI are implemented.

## Verified capabilities
- Canonical model space is `[D,H,W]=[SI,AP,LR]` with positive LPS directions `+Z,+Y,+X`.
- Orientation uses a globally unique axis permutation with explicit finite, orthogonality, dominance, and margin gates; accepted inputs use only transpose and flips and retain reversible provenance.
- Training reads only Fold 0 train `_0000.nii.gz` images and uses the existing ADN alignment losses without lesion labels.
- Synthetic validation: `335 passed` in `standalone_nnunet2d/tests`; independent Level 3 re-review returned PASS for the hashed intended snapshot.

## Evidence boundary
- No real Dataset501 scan, training, QC run, clinical evaluation, or formal experimental result has been performed locally.
