# Log

## 2026-09-08 — ADN geometry-vs-loss diagnostic
- Changed: added an image-only CLI that estimates three-slice centroid/principal-axis geometry, applies an explicit voxel forward-content correction through inverse normalized sampling, and compares identity, optional ADN prediction, and geometry alignment with CSV/JSON/QC outputs.
- Validation: RED/GREEN synthetic coverage; fresh `370 passed` in `standalone_nnunet2d/tests` and `21 passed` in root `tests`.
- Review: independent Level 3 found a modulo-180 principal-axis median blocker; a focused `+89/-89` regression and deterministic axial-aware median fixed it, and re-review returned PASS.
- Result: ADN architecture/ranges/loss/training, Dataset501, labels, and split were unchanged; no real data or training was accessed.

## 2026-09-07 — ADN transform loss landscape diagnostic
- Changed: added an image-only CLI that applies the formal NIfTI preprocessing/padding chain, scans the fixed 7x5 `(rz, tx)` grid with existing transform/warp/loss functions, and writes per-case CSV, summary, and heatmap outputs.
- Validation: valid RED for the missing CLI; focused `3 passed`; affected ADN group `59 passed`; independent Level 3 review returned PASS.
- Result: no encoder, checkpoint, labels, split, real Dataset501 data, or training was accessed; real case conclusions remain pending a server-side diagnostic run.

## 2026-09-07 — Diagnostic short-depth model input padding
- Changed: after z-score normalization, diagnostic training/QC now pad only D<16 with deterministic symmetric constant zeros, validate the padding contract in checkpoints, and unpad QC outputs before canonical display.
- Validation: scoped worker RED/GREEN; fresh `358 passed` in `standalone_nnunet2d/tests` and `21 passed` in root `tests`.
- Review: independent Level 3 found malformed-unpadding and checkpoint-contract blockers; focused regressions/minimal fixes were added and re-review returned PASS.
- Result: `adn_transform.py`, canonicalization, NIfTI geometry, Dataset501, labels, and split were unchanged; no real data or training was accessed locally, and a server rerun remains pending.

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
