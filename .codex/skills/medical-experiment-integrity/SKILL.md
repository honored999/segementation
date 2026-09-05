---
name: medical-experiment-integrity
description: Protect scientific validity and medical-data safety in stroke lesion segmentation experiments. Use when handling Dataset501/502/CI-1, raw medical images or labels, train/validation/test splits, OOF predictions, ROI derivation, preprocessing, formal metrics, preflight/training/evaluation runs, reproducibility metadata, leakage risks, or paper/spec reproduction.
---

# Medical experiment integrity

## Fixed project invariants

- Dataset501 is the established DWI-only baseline.
- Preserve the existing patient-level 5-fold `splits_final.json`.
- Do not regenerate or randomly replace established splits without explicit authorization.
- Formal segmentation metrics use original full-volume patient space unless the
  experiment protocol explicitly defines another space.
- Real images and labels are read-only.
- Store derived ROIs, crops, predictions, checkpoints, converted datasets, and
  reports separately from raw data.
- GT must never influence inference-time ROI or derived-input construction.
- Prediction-guided Stage 2 training must use out-of-fold Stage 1 predictions.

## Data leakage

Do not use test data for training, validation, hyperparameter selection, or
normalization statistics unless the explicit protocol requires it.

Preserve subject/patient grouping when group metadata exists.

Do not describe evaluation as patient-independent or equivalent unless the split
actually enforces that property.

Labels may be transformed with an image only when required as supervised targets
or geometry-aligned masks; they must not determine inference-time transforms,
candidate regions, input channels, or other features unless the protocol permits it.

## Raw-data safety

Do not overwrite, rename, convert in place, delete, or silently repair raw medical data.

Do not commit private patient data.

Derived data must have clear provenance and remain separable from the untouched source.

## Experiment classes

Distinguish:
- engineering smoke test;
- preflight;
- baseline experiment;
- tuning experiment;
- final evaluation.

A smoke test or preflight is not a formal result.

Do not aggregate incomplete/aborted runs into formal metrics.

Do not cherry-pick favorable runs.

Do not silently rerun poor outcomes with changed seeds/settings and treat them as
the same experiment.

If a formal run aborts:
- preserve it when practical;
- mark it aborted;
- exclude it from final aggregation;
- record the reason when known.

## Preflight

When a workflow needs preflight, implement an explicit preflight mode instead of
silently overriding formal configuration.

Preflight should use the same production paths/logic where appropriate and differ
only in documented engineering limits.

Mark preflight outputs as non-formal and ineligible for aggregation.

## Observation-only instrumentation

Logging, visualization, telemetry, and monitoring must not change model
parameters, optimizer, learning rate, random seed, split, preprocessing, epochs,
stopping, checkpoint selection, or metric semantics.

Optional telemetry failure must not block computation.

Do not swallow genuine computation failures such as CUDA OOM, forward/backward
errors, invalid data, or optimizer failures.

## External-source fidelity

For paper/spec/reference reproduction:
1. inspect the primary source;
2. inspect official documentation/code/data when available;
3. distinguish explicit source settings from assumptions;
4. document deliberate deviations;
5. mark unresolved details as unresolved.

Do not silently replace ambiguity with common practice.

## Reproducibility

For formal or important experiments, record enough information to reconstruct
the run, as relevant:
- Git commit;
- configuration snapshot;
- random seed;
- environment/device;
- dataset version;
- exclusions;
- split policy;
- timestamps;
- run identifier.

Generated metadata must not influence the experiment itself.

## Reporting

Clearly distinguish verified, partially verified, blocked, unresolved, and not started.

Do not claim numerical reproduction, benchmark success, data validation,
deployment readiness, or protocol confirmation until actually completed.

Synthetic engineering results must never be presented as real clinical or
server-side experimental results.
