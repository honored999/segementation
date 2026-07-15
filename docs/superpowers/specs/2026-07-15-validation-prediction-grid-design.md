# Validation Prediction Grid Design

## Goal

Generate one compact, nnU-Net-style validation figure for a best checkpoint.
Each selected slice occupies one row with DWI, ground-truth mask and thresholded
prediction columns.

## Selection

The evaluator selects at most six validation slices deterministically: largest
positive ground-truth area, smallest positive ground-truth area, lowest Dice,
empty-mask slice with the most false-positive pixels, and two seed-fixed random
slices. Duplicate selections are removed and available unique rows are used.

## Figure and Metadata

The PNG is named `validation_predictions_best.png`. Every panel title includes
patient ID, timepoint, slice index, ground-truth area, predicted area and Dice.
DWI is grayscale; masks and predictions are binary white on black. The figure
is generated only during explicit best-checkpoint evaluation, not each epoch.

## Integration

`evaluate.py` keeps per-slice image/probability/target metadata while computing
metrics, invokes one focused visualization helper after CSV/JSON export, and
writes only inside the supplied evaluation output directory. A synthetic unit
test verifies deterministic unique selection and required three-column layout.
