# Explicit Experiment Command Design

## Goal

Define one validated, reproducible command contract for a future server training
run while keeping its default behavior read-only and non-training.

## Scope

`experiment.py` will parse explicit raw, preprocessed, results, and output
directories; a fold index from 0 through 4; a positive epoch count; a device;
and an optional `--confirm-run` flag. Its configuration parser will produce a
frozen JSON-safe `ExperimentRequest` record and reject invalid fold/epoch values
before it can inspect data or create outputs.

Without `--confirm-run`, the command will call the existing server preflight,
print a JSON object containing the request plus its readiness report, and exit
0 only when readiness passes. It will not initialize a model, construct a
DataLoader, read cases, create output directories, or run an optimizer step.

With `--confirm-run`, the command will still not run local formal training in
this phase. It will return a distinct, documented exit status and a diagnostic
that formal execution has not yet been wired. This makes accidental training
impossible while reserving an explicit server-side consent boundary for the next
phase.

## Validation and Exit Codes

Argument syntax errors (missing paths, invalid fold, non-positive epoch) use
argparse exit status 2. A valid unconfirmed request exits 0 when preflight is
ready and 2 when it is not. A valid confirmed request exits 3 after printing a
JSON diagnostic that execution wiring is unavailable; it creates nothing.

The output directory must resolve beneath `standalone_nnunet2d/outputs/` to
avoid writing artifacts elsewhere. The command only validates this path in this
phase and never creates it.

## Tests

Tests use temporary empty directory roots and CPU. They verify a valid request
is JSON-safe, invalid fold/epoch values are rejected, the default command only
reports readiness, an outside output path is rejected, and `--confirm-run`
returns exit 3 without writing output. No test opens a dataset case or starts
an optimizer.

## Deferred Work

The next phase will bind the explicit request to fixed-fold DataLoaders,
`run_train_epoch`, validation, and safe checkpoints, then validate a server
smoke run with a deliberately small caller-selected epoch count.
