# Next

## Current focus
- Re-run the server-side Fold 0 diagnostic preflight with the acquisition-preserving LR adapter before longer ADN training.

## Next actions
1. Run one diagnostic epoch with `python -m standalone_nnunet2d.tools.train_adn_alignment --dataset-dir <Dataset501> --splits-file standalone_nnunet2d/reference/splits_final.json --epochs 1 --lr <diagnostic-lr> --device cuda --output-dir <new-output-dir>`.
2. Confirm all Fold 0 train images canonicalize and inspect orientation rejection reasons/provenance before scheduling a longer diagnostic run.
3. Generate representative QC from the selected checkpoint and inspect the three four-panel views plus `summary.json`.
4. Record failures or observations without changing the fixed split, raw NIfTI files, ADN ranges, or segmentation pipeline.

## Blockers
- Real Dataset501 and GPU execution are server-only and were not validated locally.
