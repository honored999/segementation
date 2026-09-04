# Server Preflight and Safe Dry-Run Design

## Goal

Provide an explicit command that determines whether a server can safely run a
future Dataset501 experiment without starting training, loading a dataset case,
or writing model artifacts.

## Scope

`tools/server_preflight.py` will expose a reusable inspection function and a
CLI. It will validate the supplied raw-data root, preprocessed-data root, and
results root as existing directories; load the local reference plan JSON; report
the selected PyTorch device, CUDA availability, GPU count/name, and the model
configuration's patch size and batch size. It will not recurse through input
directories or open NIfTI/NPZ files.

`train.py` will remain a formal-training-disabled entry point. A new
`dry_run.py` command will require explicit paths and run only the preflight
function. It must reject `--run` or any other flag that could be interpreted as
permission to train, keeping server validation separate from formal execution.

## Interface

The reusable `inspect_server_readiness(raw_root, preprocessed_root, results_root,
device)` function returns a JSON-safe dictionary with directory status,
environment facts, plan facts, and a boolean `ready`. Missing directories,
unavailable requested CUDA devices, and unreadable reference JSON produce a
`ready=False` result with a list of diagnostic messages rather than partial
success claims.

The CLI prints this dictionary as formatted JSON and exits with status 0 only
when `ready=True`; status 2 means a server prerequisite is missing. It never
creates the results directory, writes an output file, initializes a model, or
constructs a DataLoader.

## Error Handling

All supplied paths are normalized for reporting but are treated as read-only.
Malformed or unreadable reference JSON is reported as a diagnostic. A requested
`cuda` device is unavailable when PyTorch reports no CUDA support or the index
is outside the detected device count. The default device is `cuda` when
available and `cpu` otherwise; this decision is reported but does not allocate
GPU memory.

## Tests

Tests will use temporary empty directories and the repository's small reference
JSON. They will establish successful readiness for CPU, diagnose a missing
directory, reject unavailable CUDA, verify JSON-safe facts, and invoke the CLI
without reading a dataset case. No test will write under `outputs/` or start an
optimizer step.

## Deferred Work

This phase deliberately does not enable formal training. The next phases will
add opt-in execution configuration, single-case inference, and cross-validation
orchestration only after their own tested designs are approved.
