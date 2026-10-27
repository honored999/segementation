# CI-1 DWI conversion

Engineering conversion tooling only. Run only after independent static review and
an explicit decision to convert real data. No conversion/preflight is launched
by this change. Existing Dataset501, source data and fixed splits stay untouched.

PowerShell example (replace both absolute paths; output MUST NOT exist):

```powershell
& 'D:\Anaconda\envs\newconda\python.exe' non_teacher_student_files/make_nnunet_dataset_from_CI1.py `
  --ci1-root 'D:\source\CI-1' `
  --output-root 'D:\derived\CI1_DWI_high_new' `
  --b-value high
```

Both `--b-value` and `--output-root` are required. Choose `high` (largest known b
in the one selected DWI series), `low` (smallest), or a finite nonnegative numeric
value, e.g. `--b-value 800`. Exact requests never fall back. b uses the DICOM
DiffusionBValue header (s/mm²); no filename/InstanceNumber inference or private-tag
fallback. Existing direct Python reader/conversion helpers default to `high` only
for compatibility; the CLI has no automatic choice.

Selection reads headers, filters the whole series to ONE b, then checks the
selected b covers the union of source slice positions. Missing/unknown b,
ambiguous selected-b duplicates, multiple candidate DWI series, unsupported
multiframe files, or inconsistent geometry reject the case visibly. There is no
per-position substitution. Only the exact selected files are passed to the
SimpleITK DICOM reader; no intensity normalization, registration or resampling.
Before pixel reading, the selected unrounded 3D ImagePositionPatient sequence
must match a regular orthogonal grid along the normalized orientation normal.
The maximum Euclidean residual is 0.001 mm (to allow stored decimal rounding);
in-plane drift, shear and unsupported nonuniform 3D positions are rejected.

NIfTI labels containing `DWI` but no `ADC` take priority per patient/timepoint; `DWIADC` is eligible only when no pure DWI exists.
ADC-only/FLAIR files are excluded; `pred`, `output`, `deepisles`, and `xy_fixed`
paths remain forbidden. Pairing stays patient/timepoint based; `D1（2）` parsing
is unchanged. Byte-identical duplicate labels are deduplicated, retaining the
first candidate in priority/path sorted traversal order. Conflicting labels within the selected priority or multiple matching
source directories stop indexing with explicit candidate paths; resolve ambiguity manually. No arbitrary
filename or directory-size tie breaker is used for these conflicts.

Image and label size, spacing, origin and direction must agree (geometry
comparison tolerance 1e-5); mismatches are skipped, never repaired. Retained label
semantics: **every nonzero voxel becomes 1**, zero stays 0, saved as uint8 with
the label's own geometry. This does not establish any original class meaning.

The output path is resolved and rejected if equal to, inside, or an ancestor of
the input. Existing paths and `--clean` are refused. Use a fresh derived output
for each b choice; do not reuse a training dataset. `conversion_manifest.tsv`
records patient/timepoint, source label/directory, requested/actual b, series UID,
ordered selected DICOM paths, selected count, selection reason and status for successful cases only.
Rejected cases are written separately to `conversion_rejections.tsv` with
failure reasons; they are never mixed into the dataset manifest. Index ambiguity
stops before output creation. Console output also states the selected b or skip
reason. Do not treat zero successful cases or skips as a completed dataset.

This is not an independent-patient split audit or formal experimental evidence.
Repeated examinations must be grouped by patient before any new split decision;
this converter does not create or edit splits.
