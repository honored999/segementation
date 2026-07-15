# Random Six Prediction Grid Design

## Goal

Allow a fast local visual check without evaluating an entire validation fold.

## Interface

`evaluate.py` gains `--visualize-random N`. With this flag it deterministically
selects `N` validation manifest rows using seed 2026, performs inference only
for those rows, and writes `validation_predictions_random<N>.png` to the
provided output directory. The existing evaluation invocation remains complete
fold evaluation and metric export.

## Output

The figure uses one row per selected slice and DWI, GT Mask and Prediction
columns. Titles include patient/timepoint/slice ID, GT area, predicted area and
Dice. The random sample IDs are stored in `visualization_samples.json`.

## Verification

An isolated test checks deterministic, unique random selection. Local execution
is limited to six 512 by 512 forward passes; no training or full-fold metrics
run is permitted in this mode.
