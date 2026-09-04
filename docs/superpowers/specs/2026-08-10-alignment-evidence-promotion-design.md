# Alignment Evidence Promotion Design

## Goal

Permit new formal artifacts to use `official_aligned` only after the exact
transform report and the `repeat_oracle_stability_v1` inference report have
both passed. Existing artifacts remain unchanged and pending.

## Evidence boundary

The training CLI accepts two optional paths:

- `--transform-parity-report`
- `--inference-parity-report`

Supplying neither preserves the existing `official_alignment_pending` path.
Supplying only one is an error. Supplying both validates the reports before any
training output directory is created.

The transform report must be a single-oracle exact comparison with
`status="passed"`, `run_state="official_alignment_pending"`, zero image
tolerance, and passed image, label, mask, and manifest components.

The inference report must use `repeat_oracle_stability_v1`, contain at least
three distinct oracle roots, use zero image tolerance, have every component
passed, and report zero stable mismatches and zero unobserved standalone labels.

## Evidence record

Successful validation produces one JSON-safe record containing:

- schema version and policy name;
- `run_state="official_aligned"`;
- resolved source paths and SHA256 digests for both reports;
- the inference repeat count and policy;
- a compact snapshot of the report fields used by validation.

The record is embedded in the resolved training config and every formal
checkpoint. Its content contributes to the plan hash, so resume cannot silently
switch evidence.

## Propagation

Pending checkpoints still require `--allow-pending`. An `official_aligned`
checkpoint is accepted only when its embedded evidence record validates.
Prediction manifests copy the aligned state and evidence.

Fold validation passes the checkpoint state and evidence into the fold report.
OOF aggregation accepts either five pending reports or five aligned reports
with byte-equivalent evidence records. Mixed states, missing aligned evidence,
or different evidence records are errors. An aligned five-fold aggregate writes
`official_aligned`; a pending aggregate remains pending.

## Safety and compatibility

- No existing checkpoint or report is rewritten or promoted.
- No retry, tolerance, Dice threshold, morphology, or hidden fallback is added.
- Invalid JSON, missing fields, nonzero tolerance, failed components, stale
  policy, or malformed SHA256 fails before formal execution.
- The existing pending workflow remains available unchanged.
- Full-volume fold validation remains the source of OOF metrics; online
  validation is not used as official evidence.

## Verification

Each behavior is implemented test-first. Focused tests cover report validation,
checkpoint save/resume, prediction propagation, fold-report propagation,
five-fold aggregation, pending compatibility, and failure cases. Completion
requires the full pytest suite, `py_compile`, and `git diff --check`.
