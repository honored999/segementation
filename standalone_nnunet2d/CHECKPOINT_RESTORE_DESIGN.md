# Checkpoint Restore Verification Design

## Goal

Verify that a smoke-only checkpoint restores model and optimizer state on the
server without performing another optimizer update.

## Contract

A CLI will require a checkpoint below `outputs/`, build the reference
PlainConvUNet and smoke-only SGD settings, call `load_checkpoint` with expected
`smoke_run_only=true`, and report metadata plus a no-grad forward output shape.
It will not construct datasets, read NIfTI files, write files, or call backward.
