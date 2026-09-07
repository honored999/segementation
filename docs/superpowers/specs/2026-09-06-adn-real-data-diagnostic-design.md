# ADN real-data diagnostic workflow

## Scope

This is a diagnostic-only bridge from one explicitly supplied Dataset501 DWI
NIfTI image to the existing canonical ADN transform. It does not scan a
dataset, read labels, resample voxels, alter NIfTI geometry, connect to
nnU-Net, or run a formal experiment. Synthetic fixtures are the only local
validation data.

## Geometry contract

The model array is always `[D,H,W] = [SI,AP,LR]`, with positive directions
`+Z,+Y,+X` in SimpleITK/LPS coordinates. The adapter reads one `NiftiVolume`
and performs only a NumPy transpose and per-axis flip. It chooses the
canonical-to-raw array-axis mapping by exhaustive global assignment, then
requires finite direction values, maximum orthogonality error `1e-4`, a unique
assignment, dominant absolute alignment at least `cos(20°)`, and a dominant to
second-best margin of `0.20`. Oblique directions within those gates are
accepted without interpolation. Raw spacing, origin, direction, array order,
mapping, labels, and an inverse provenance record are retained; restoration is
an exact inverse array operation.

## Diagnostic flow

The training CLI reads only `imagesTr/<case>_0000.nii.gz` for the cases in the
supplied fold-0 train list (defaulting to the repository reference split).
Each volume is canonicalized independently and normalized with its own finite
mean and standard deviation. It uses only the existing `alignment_losses()`
for flip, inverse reconstruction, and total loss. Batch size is fixed at the
default of one because volume shapes are not padded or resampled. Latest and
best checkpoints contain the ADN state, optimizer state, run/model contract,
normalization/split metadata, and machine-readable epoch history.

The QC CLI loads one explicit image and checkpoint, applies the same
canonicalization/normalization, and writes exactly three four-panel PNGs for
deterministic 25/50/75% depth slices plus `summary.json`. It is diagnostic
only, has isolated output-path checks, and never writes an input path.

## Checkpoint and safety contract

Checkpoint validation is performed before model state loading and rejects an
unknown format, missing fields, or a mismatched ADN contract. Training output
directories must not overlap the dataset directory and must not overwrite a
non-empty directory. Invalid orientation and path conditions raise explicit
errors with machine-recordable reasons. No project memory or existing ADN,
segmentation, loss, or dataset production file is changed.
