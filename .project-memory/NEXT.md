# Next

## Current focus
- Re-run the server-side Fold 0 diagnostic preflight with the shared model-input depth-padding contract before longer ADN training.

## Next actions
1. Run one diagnostic epoch with `python -m standalone_nnunet2d.tools.train_adn_alignment --dataset-dir <Dataset501> --splits-file standalone_nnunet2d/reference/splits_final.json --epochs 1 --lr <diagnostic-lr> --device cuda --output-dir <new-output-dir>`.
2. Confirm all 76 Fold 0 train images reach the model; verify case018/case037 record `13 -> 16` with pads `1/2`, case031 remains depth 18, and no H/W dimension changes.
3. Generate representative QC from the selected checkpoint and inspect the three four-panel views plus `summary.json`.
4. Record failures or observations without changing the fixed split, raw NIfTI files, ADN ranges, or segmentation pipeline.

## Blockers
- The new model-input padding path is synthetic-validated locally but pending server rerun on the 76 Fold 0 training images.
