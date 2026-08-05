# Server Smoke Run Design

## Goal

Enable one explicitly confirmed server smoke run using one fixed-fold training
case, one validation case, and one epoch, while keeping local development free
of Dataset501 access.

## Contract

`experiment.py --confirm-run` will remain the sole execution gate. A new runner
will build `StrokeSliceDataset` instances from the request's raw root, select
only the first supplied train and validation case for the requested fold, create
single-item DataLoaders, construct the local model/loss/optimizer, run one
train epoch and one validation epoch, then save a versioned checkpoint and JSON
report below the request output root. The results/preprocessed roots are only
validated by preflight in this phase.

The smoke-only optimizer is `SGD(lr=0.01, momentum=0.9, weight_decay=0)`. Every
checkpoint, JSON report, and console payload will include `smoke_run_only=true`.
These values are engineering connectivity settings only: they are neither an
nnU-Net reproduction claim nor an original 2015 U-Net training configuration.
They must not be used for performance comparison. Formal optimizer and learning
rate policy remain deferred until verified against the corresponding official
code; the original U-Net paper's stated Caffe SGD momentum was 0.99, not 0.9.

## Local and Server Boundaries

Local tests use synthetic tensors, temporary paths, and tiny models; no test
constructs `StrokeSliceDataset`. The actual command only reads real NIfTI files
after `--confirm-run`; it writes only within `standalone_nnunet2d/outputs/`.
Missing raw `imagesTr`/`labelsTr`, unavailable CUDA, or checkpoint failures stop
the command with diagnostics and leave no claim of a completed run.

## Verification

Unit tests cover request-to-runner limits, one-epoch aggregation, output-path
confinement, and checkpoint metadata using synthetic components. Server handoff
will provide a single explicit command and expected artifacts, but no local
formal training execution.
