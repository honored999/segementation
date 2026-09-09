# Status

Updated: 2026-09-09

## Current state
- Dataset501 remains the established read-only DWI-only baseline with fixed patient-level folds.
- The isolated ADN transform implementation remains unchanged from baseline commit `f75a807`.
- A diagnostic-only acquisition-preserving NIfTI adapter, Fold 0 ADN training CLI, checkpoint contract, single-case QC CLI, fixed-grid transform loss-landscape CLI, and geometry-vs-loss comparison CLI are implemented.

## Verified capabilities
- Canonical model space is `[D,H,W]=[acquisition through-plane, acquisition in-plane non-LR, anatomical LR]`; D remains source voxel z and W remains source voxel x/y.
- Orientation requires finite orthogonal directions, a unique in-plane LR assignment with margin at least `0.20`, positive LPS X/Z signs for W/D, and a right-handed physical `[W,H,D]`; AP/SI obliquity is provenance only.
- Accepted inputs use only transpose and flips, retain exact reversible provenance, and never modify or fabricate NIfTI geometry.
- Training reads only Fold 0 train `_0000.nii.gz` images and uses the existing ADN alignment losses without lesion labels.
- Diagnostic model inputs are per-volume z-score normalized, then D-only constant-zero padded to minimum depth 16; odd padding is placed after (`13 -> 16` uses `1/2`), while D>=16 and all H/W sizes remain unchanged.
- Training checkpoints and QC record and validate the shared model-input padding contract; QC removes model-only padding before canonical-depth display or inverse mapping.
- The loss-landscape CLI scans 35 fixed `(rz, tx)` candidates per requested image without constructing the ADN encoder or loading checkpoints, and records per-case CSV, summary, and heatmap outputs.
- The geometry-vs-loss CLI estimates deterministic ellipse centroid/principal-axis pose on 25/50/75% acquisition slices, converts an explicit voxel-space forward content correction through `inverse(F)` to an `align_corners=False` sampler, and compares identity, optional ADN checkpoint prediction, and geometry alignment without labels or training.
- Whole-head geometry now uses a robust low threshold, largest component, hole filling and closing before binary PCA; QC displays its mask/contour, and summary flags rotations above 30 degrees without clamping. Transform and loss-comparison semantics are unchanged.
- Fresh synthetic validation: focused `20 passed`, all five ADN files `79 passed`, combined full `standalone_nnunet2d/tests tests` `402 passed`; independent Level 3 re-review PASS after fixing zero-border interior background noise.
- The fallback assumes approximately symmetric background noise with dim head signal above its noise floor. One-sided positive noise remains a known limitation; real-image pose reliability requires contour QC.

## Evidence boundary
- Server-reported Fold 0 orientation preflight covered 76 training images and exposed only the short-depth model-input blocker; the new padding implementation has not yet been rerun on the server.
- No real Dataset501 loss-landscape or geometry-vs-loss scan, training, QC run, or lesion-label access was performed locally; synthetic tests and server preflight remain engineering evidence, not formal experimental results.
