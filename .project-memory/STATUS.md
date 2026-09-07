# Status

Updated: 2026-09-07

## Current state
- Dataset501 remains the established read-only DWI-only baseline with fixed patient-level folds.
- The isolated ADN transform implementation remains unchanged from baseline commit `f75a807`.
- A diagnostic-only acquisition-preserving NIfTI adapter, Fold 0 ADN training CLI, checkpoint contract, and single-case QC CLI are implemented.

## Verified capabilities
- Canonical model space is `[D,H,W]=[acquisition through-plane, acquisition in-plane non-LR, anatomical LR]`; D remains source voxel z and W remains source voxel x/y.
- Orientation requires finite orthogonal directions, a unique in-plane LR assignment with margin at least `0.20`, positive LPS X/Z signs for W/D, and a right-handed physical `[W,H,D]`; AP/SI obliquity is provenance only.
- Accepted inputs use only transpose and flips, retain exact reversible provenance, and never modify or fabricate NIfTI geometry.
- Training reads only Fold 0 train `_0000.nii.gz` images and uses the existing ADN alignment losses without lesion labels.
- Diagnostic model inputs are per-volume z-score normalized, then D-only constant-zero padded to minimum depth 16; odd padding is placed after (`13 -> 16` uses `1/2`), while D>=16 and all H/W sizes remain unchanged.
- Training checkpoints and QC record and validate the shared model-input padding contract; QC removes model-only padding before canonical-depth display or inverse mapping.
- Fresh synthetic validation: `358 passed` in `standalone_nnunet2d/tests` and `21 passed` in root `tests`; independent Level 3 re-review returned PASS after deterministic unpadding and checkpoint-contract blockers were fixed.

## Evidence boundary
- Server-reported Fold 0 orientation preflight covered 76 training images and exposed only the short-depth model-input blocker; the new padding implementation has not yet been rerun on the server.
- No real Dataset501 scan, training, QC run, or lesion-label access was performed locally; server preflight remains engineering evidence, not a formal experimental result.
