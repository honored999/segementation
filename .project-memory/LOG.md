# Log

## 2026-09-07 — Acquisition-preserving ADN LR canonicalization
- Changed: fixed D to source voxel z, assigned W only from a clear in-plane LR axis, made H right-handed, removed AP/SI dominance rejection, and updated reversible provenance/checkpoint/QC semantics.
- Validation: RED/GREEN synthetic coverage; fresh `341 passed` in `standalone_nnunet2d/tests` and `21 passed` in root `tests`; all-repository collection remains blocked by unrelated missing `nnunetv2` and a directory-dependent audit import.
- Review: independent Level 3 found an exact-zero source-z LPS Z blocker; a focused regression/minimal rejection fix was added and re-review returned PASS.
- Result: no real Dataset501 or labels were accessed; next step is a new-output server Fold 0 diagnostic preflight.

## 2026-09-07 — ADN real-data diagnostic workflow
- Changed: added strict array-only NIfTI canonicalization, label-free Fold 0 ADN training/checkpoints, and explicit single-case QC output.
- Validation: synthetic RED/GREEN coverage; final `standalone_nnunet2d/tests` result `335 passed`; Python compilation and diff checks passed.
- Review: independent Level 3 round 1 found a missing checkpoint-field validation; regression and minimal fix were added; round 2 returned PASS.
- Result: implementation is ready for a bounded server diagnostic, but no real data or formal experiment was run.
