# Fold 0 Short Smoke Design

## Goal

Run a bounded fold 0 end-to-end smoke workflow after single-case diagnostics
pass, without claiming formal nnU-Net performance.

## Limits

The workflow uses `splits_final.json` unchanged, two epochs, at most eight
training batches per epoch, and the first three fold-0 validation cases. Every
artifact contains `run_type=smoke_run_only`.

## Outputs

It writes `checkpoint_latest.pth` after each epoch and replaces
`checkpoint_best.pth` only when mean validation Dice improves. After loading
best state into a fresh model, it re-runs the bounded validation set and writes
case NIfTI predictions, metrics CSV, summary JSON, overlays, and best/worst
text. A checkpoint restore check compares argmax masks for the same input and
fails on any mismatch.

## Boundaries

No re-splitting, full-fold training, official optimizer claims, or formal
benchmark comparison is permitted. All generated files remain below a caller
selected smoke-only output root.
