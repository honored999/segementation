# ADN real-data diagnostic workflow

## Scope

This is a diagnostic-only bridge from one explicitly supplied Dataset501 DWI
NIfTI image to the existing canonical ADN transform. It does not scan a
dataset, read labels, resample voxels, alter NIfTI geometry, connect to
nnU-Net, or run a formal experiment. Synthetic fixtures are the only local
validation data.

## Geometry contract

The model array uses the acquisition-preserving contract `[D,H,W] =
[acquisition through-plane, acquisition in-plane non-LR, anatomical LR]`.
`D` always comes from source voxel z (SimpleITK array axis 0) and is flipped,
if needed, toward LPS `+Z`; an exactly zero LPS Z component is explicitly
rejected because a sign flip cannot make it positive. `W` comes from the source voxel x/y axis with the
unique largest absolute LPS X component; voxel z as LR is rejected, as is an
LR best-to-second-best margin below `0.20`. `W` is flipped toward LPS `+X`.
`H` is the remaining source x/y axis and its flip is chosen only so the final
physical `[W,H,D]` vectors satisfy `(W x H) dot D > 0`.

The adapter requires nine finite direction values and maximum orthogonality
error `1e-4`. It does not impose AP/SI dominance or obliquity rejection gates,
and it adds no empirical LR absolute-angle threshold. AP/SI components and
obliquity are provenance only. The adapter performs only NumPy transpose and
flip operations: no interpolation, resampling, 3D deoblique, or replacement
NIfTI geometry is produced. Source spacing, origin, direction, array order,
axis sources, LR assignment measures, applied/inverse operations, and
orientation components are retained; restoration is an exact inverse array
operation.

## Diagnostic flow

The training CLI reads only `imagesTr/<case>_0000.nii.gz` for the cases in the
supplied fold-0 train list (defaulting to the repository reference split).
Each volume is canonicalized independently and normalized with its own finite
mean and standard deviation. After normalization, only model-input tensors with
canonical D below 16 receive deterministic constant-zero padding along D:
13 becomes 16 with widths `(1, 2)`, 15 becomes 16 with widths `(0, 1)`, and
D at least 16 is unchanged. H/W are never padded or cropped. This model-only
adaptation does not modify canonical arrays, provenance, or NIfTI geometry and
does not resample voxels. The shared padding metadata records the minimum
model depth, policy, and every padded training case with before/after depth and
pad widths. It uses only the existing `alignment_losses()` for flip, inverse
reconstruction, and total loss. Batch size remains fixed at the default of one
because volumes are processed independently. Latest and best checkpoints
contain the ADN state, optimizer state, run/model contract,
normalization/split/padding metadata, and machine-readable epoch history.

The QC CLI loads one explicit image and checkpoint, applies the same
canonicalization/normalization and the same model-input depth padding contract,
and writes exactly three four-panel PNGs for deterministic 25/50/75% slices of
the unpadded canonical depth plus `summary.json`. Model outputs are unpadded
before display or any canonical inverse mapping. It is diagnostic only, has
isolated output-path checks, and never writes an input path.
Within ADN, `W` is anatomical LR, `tx` is model-space LR translation, and `rz`
is acquisition/model in-plane rotation. These are not physical-space 3D rigid
registration claims.

## Checkpoint and safety contract

Checkpoint validation is performed before model state loading and rejects an
unknown format, missing fields, or a mismatched ADN contract. Training output
directories must not overlap the dataset directory and must not overwrite a
non-empty directory. Invalid orientation and path conditions raise explicit
errors with machine-recordable reasons. No project memory or existing ADN,
segmentation, loss, or dataset production file is changed.
