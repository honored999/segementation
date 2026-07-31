# Read-Only 2D Data Pipeline Design

## Scope

Implement the next project phase only: bounded NIfTI I/O, plan-driven 2D
resampling and normalization, fixed-fold case selection, and on-demand 2D slice
sampling. No formal training, predictions, result comparison, or data caching
is included.

## Inputs and safety

The pipeline accepts an explicit raw Dataset501 root. It must contain
`imagesTr/` and `labelsTr/`; the code raises a clear error if either is absent.
Every read is a single requested image/label pair. The implementation will not
write beneath the raw, preprocessed, or results locations; it will not scan all
image voxels, copy NIfTI files, or create NPZ caches.

## Components

- `nifti_io.py`: a small SimpleITK adapter returns an image array and spatial
  metadata for one NIfTI path, and writes only explicitly requested test files.
- `preprocessing.py`: uses the supplied 2D plan's spacing and interpolation
  orders. Image interpolation is cubic (`order=3`); segmentation interpolation
  is linear (`order=1`) and converted back to discrete integer labels. Image
  normalization is full-image Z-score because `use_mask_for_norm=false`.
- `dataset.py`: validates a named split from `splits_final.json`, resolves one
  case's image and label filenames, loads it on demand, and exposes sampled 2D
  slices. It will not enumerate or preload every NIfTI image.
- `sampling.py`: chooses a deterministic or supplied random axial slice index;
  foreground oversampling remains deliberately out of scope until verified.

## Interfaces and errors

`load_case(case_id)` returns a channel-first image and integer segmentation
with matched spatial shape. The case ID must be present in the requested fold;
missing files, invalid fold indices, mismatched image/label geometry, non-finite
inputs, and unsupported labels raise explicit exceptions. Tests use temporary
synthetic NIfTI files, so they do not depend on the unavailable external paths.

## Verification

Tests will demonstrate: SimpleITK metadata round-trip; interpolation output
shape and discrete labels; full-image Z-score; correct fixed-fold membership;
and on-demand sampled slice shapes. A bounded `inspect_dataset.py` command will
report only root/child-path availability and optionally one explicitly named
case, never bulk-scan the dataset.
