# Segmentation Metrics and Cross-Validation Summary Design

## Scope

Implement pure NumPy/PyTorch-compatible metric functions for already-produced
binary segmentation masks and an out-of-fold JSON summary reader. This phase
does not run a model, read external NIfTI files, create predictions, or train.

## Metric contract

`binary_segmentation_metrics(prediction, reference)` accepts same-shaped,
integer/bool binary arrays. It computes true positives, false positives, false
negatives, true negatives, Dice, and IoU. If both masks contain no foreground,
Dice and IoU are defined as 1.0. If only one has foreground, both are 0.0.
Inputs with values outside `{0, 1}` or mismatched shapes raise clear errors.

`case_metric_record(case_id, prediction, reference)` produces a JSON-safe
dictionary with the case ID and metrics. No thresholding or argmax is hidden in
the API: callers must explicitly provide discrete masks.

## Cross-validation contract

`summarize_oof_cases(records)` requires exactly one record per case ID, returns
per-metric case means, and rejects duplicate/missing identity data. A comparison
helper reads the supplied `reference/summary.json` foreground Dice/IoU baseline
and reports the signed difference for a caller-provided OOF summary. It does
not claim equivalence unless actual 95-case predictions are supplied.

## Tests and safety

Tests cover perfect overlap, partial overlap, empty-mask behavior, invalid
inputs, JSON-safe records, duplicate-case rejection, mean aggregation, and
reference-baseline extraction. They use tiny arrays and reference JSON only;
they do not access external data paths or write outputs.

## Boundaries

The official baseline remains Dice `0.731103738314918` and IoU
`0.5923877518050135` from the supplied summary. This module will make later
five-fold evaluation auditable but will not report a reproduced score until
predictions have genuinely been generated and evaluated.
