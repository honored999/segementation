# CI-1 Raw Data Audit Design

## Goal and scope

Add a read-only audit tool for the non-standard CI-1 raw-data layout. The
tool inventories files and DICOM series, records metadata and geometry, and
computes auditable DWI/mask statistics only for the confirmed pairs returned
by `prepare_ci1_dwi_dataset.build_index()` with `match_status=matched`.

The audit does not convert data, train or infer, preprocess for nnU-Net,
resample or rewrite images, repair masks, or mutate any raw input. The only
files written by a successful invocation are the three result files listed in
[Outputs](#outputs). The local checkout may not contain the real CI-1 data;
the tool must never substitute synthetic or guessed data for a real audit.

The planned implementation is a standalone module at
`non_teacher_student_files/audit_ci1_raw_data.py`. It may reuse the existing
`prepare_ci1_dwi_dataset.build_index()` API for the confirmed DWI-to-mask
relation and may use read-only GDCM/SimpleITK primitives for series discovery.
It must not use filename-replacement helpers for ADC/FLAIR pairing and must
not import the sibling-worktree `standalone_nnunet2d` preprocessing module at
runtime.

## Command-line contract

The module is invoked as:

```powershell
conda run -n newconda python -m non_teacher_student_files.audit_ci1_raw_data `
  --ci1-root <CI-1-root> `
  --output-dir <audit-output-dir> `
  [--target-spacing-x-mm <x>] `
  [--target-spacing-y-mm <y>]
```

`--ci1-root` and `--output-dir` are required. Both paths are normalized with
`Path.resolve()` before any scan or write. The output path is rejected when it
is the input path or is contained anywhere below the input path, including
through symlink resolution. An existing output directory must be empty; an
absent directory may be created. This prevents an audit from silently
coexisting with unrelated output files.

The optional target spacings control the geometry-only XY simulation. When
both are omitted, the target is `0.4892368018627167` mm for both x and y
axes. Supplying only one is invalid. Values must be finite and positive. No
image or mask is written for either mode.

The command returns zero when the scan completes and writes a complete report,
including data findings such as unreadable series, unknown modality, unmatched
DWI, or geometry mismatch. It returns a non-zero code before writing results
for invalid arguments, an invalid input root, an unsafe/non-empty output
directory, an unrecoverable input enumeration failure, or a result-write
failure. Item-level findings are recorded in the CSV/JSON results and are not
silently skipped.

## Discovery and data lineage

### Inventory

Recursively enumerate regular files below `ci1_root` in deterministic
relative-path order. Inventory `.nii` and `.nii.gz` files as NIfTI candidates.
Inventory `.dcm` files and extensionless/other files that are accepted by a
read-only GDCM probe as DICOM candidates. Record relative path, file type,
byte size, and read status internally; the aggregate counts are written to
`dataset_summary.json`.

Also record directory/case candidates using the established CI-1 convention:
the first patient-directory level and a `D\\d+` timepoint token are candidate
identifiers, not independent proof of identity. The source of each candidate
must be recorded as a path rule. The `build_index()` result remains the only
formal DWI-to-mask pairing source.

### DWI-to-mask pairing

Call `build_index()` in read-only mode and preserve its returned paths and
status without re-discovering a replacement pair. For each row:

- `match_status=matched` is a confirmed DWI-to-mask pair for this audit;
- the case row records `pairing_status=confirmed`,
  `pairing_source=prepare_ci1_dwi_dataset.build_index`, and the original
  `match_status`;
- `match_status` other than `matched` (including `missing_dicom`) is an
  unmatched DWI and must not produce formal case statistics;
- duplicate or otherwise silently resolved candidates are not upgraded to a
  stronger status than the status returned by `build_index()`.

The audit must not use `find_adc_segmentation_path`,
`find_flair_segmentation_path`, filename substitution, basename similarity,
or directory proximity as a formal ADC/FLAIR pairing method. Existing
filename-based helpers are outside the formal data lineage for this tool.

The authoritative DWI identity is the unique mapping from the DWI source file
in a `build_index()` row with `match_status=matched` to one UID-isolated
`SeriesInstanceUID`. `modality_class` is independent audit evidence only;
`unknown` never cancels confirmed DWI lineage. A non-unique or unreadable
source-to-UID mapping is reported as an audit failure, not silently resolved.

### DICOM series isolation

Enumerate every GDCM series ID in each candidate directory and create one
series record per DICOM `SeriesInstanceUID` (`0020,000E`). A directory is not a
series and a patient/timepoint is not a series. For each GDCM file list, read
the UID from every readable file and require exactly one consistent UID before
using the series. Missing or inconsistent UIDs produce an auditable failed
series row and never cause files from different UIDs to be merged.

The series record retains the GDCM series ID, `SeriesInstanceUID`, directory,
sorted file list/count, and the metadata source. A matched `build_index()` DWI
row may be linked to its unique DWI series only when the UID-isolated series
and the returned DWI source are consistent. If no unique readable DWI series
can be established, the confirmed pair remains recorded but its case audit is
failed with blank derived statistics.

## Modality classification and metadata

Read metadata without changing pixel values. Every series row must contain a
column for, and must report when available, each of these requested public
DICOM fields:

- the internal case/patient identifier used by the audit, plus
  `SeriesInstanceUID` (`0020,000E`), `SeriesDescription` (`0008,103E`),
  `ProtocolName` (`0018,1030`), `SequenceName` (`0018,0024`), `ImageType`
  (`0008,0008`), and `Modality` (`0008,0060)`;
- `Manufacturer` (`0008,0070`), `ManufacturerModelName` (`0008,1090`),
  `Rows` (`0028,0010`), `Columns` (`0028,0011`), `PixelSpacing` (`0028,0030`),
  `SliceThickness` (`0018,0050`), and `SpacingBetweenSlices` (`0018,0088`);
- `ImagePositionPatient` (`0020,0032`) and `ImageOrientationPatient`
  (`0020,0037`);
- `RescaleSlope` (`0028,1053`), `RescaleIntercept` (`0028,1052`),
  `BitsAllocated` (`0028,0100`), `BitsStored` (`0028,0101`), and
  `PixelRepresentation` (`0028,0103`); and
- the public DICOM MR diffusion `b-value` (`0018,9087`) when generically
  readable.

The series CSV has a `metadata_field_status` JSON column whose entry for every
requested field is one of `present`, `absent`, `unreadable`, or `inconsistent`.
An absent field has a blank value plus the explicit `absent` status; it is
never silently inferred. Values are read from the deterministic representative
file and checked across readable files where series consistency is meaningful.
The audit supports only generically readable public DICOM attributes. It does
not implement a vendor-specific private-tag framework and does not infer a
requested field from private tags.

The semantic modality class is determined from DICOM metadata, not directory
names:

| Class | Deterministic metadata rule |
| --- | --- |
| `dwi` | normalized description/protocol text contains `dwi` and does not contain `adc` |
| `adc` | normalized description/protocol text contains `adc` |
| `flair` | normalized description/protocol text contains `flair` and does not contain `t1` |
| `unknown` | no unambiguous rule matches, metadata conflicts, or metadata is unavailable |

`classification_source` lists the actual tags used, such as
`dicom:SeriesDescription` or `dicom:ProtocolName`; `classification_confidence`
is `high` when the metadata rule is unique and the series is readable,
`medium` when a single keyword is present but complementary metadata is
missing, and `none` for `unknown` or conflicting metadata. The DICOM
`Modality` value (`MR`, for example) is reported separately and is not
silently treated as DWI, ADC, or FLAIR. Directory names may be recorded as
non-authoritative evidence but never determine the semantic class.

All metadata fields are retained as strings or JSON-encoded scalar/array
values. Missing tags are empty fields with an explicit metadata status; they
are not replaced with inferred values.

## ADC/FLAIR boundary

ADC and FLAIR are audited only at DICOM-series level. When a series shares a
path-derived patient/timepoint candidate with a confirmed DWI case, the
series row may record the candidate values, but it must contain:

```text
pairing_status=unverified
pairing_evidence=path_or_metadata_candidate_only; no_authoritative_pairing_table
```

The same `unverified` status is used when the series is merely a possible
companion with no authoritative pairing table. No ADC/FLAIR candidate may
populate any case-level field, including ADC/FLAIR intensity, lesion volume,
lesion statistics, or DWI-ADC pixel correspondence. `case_statistics.csv`
contains no formal ADC or FLAIR statistics columns. The series report may
contain DICOM intensity summaries and the unverified pairing evidence, but
those values are never promoted to case statistics.

## Geometry and physical-coordinate contract

Use SimpleITK's physical-image conventions consistently:

- `GetSize()`, `GetSpacing()`, and `GetOrigin()` are ordered `(x, y, z)`;
- `GetArrayFromImage()` returns a NumPy array ordered `(z, y, x)`;
- a NumPy voxel index `(z, y, x)` maps to the SimpleITK index `(x, y, z)`;
- physical coordinates use `origin + direction @ (index_xyz * spacing_xyz)`.

For every DWI/mask pair, compare size, spacing, origin, and direction before
any metric calculation. The default comparison tolerances are absolute
`1e-5 mm` for spacing, `1e-4 mm` for origin, and `1e-6` for each direction
matrix element; sizes must match exactly. A pair is `geometry_status=match`
only when all checks pass. The tool never calls an existing helper that
silently resamples a mask, including `read_mask_on_reference()`.

For a mismatch, record each failing geometry component, retain the source
geometry in the case row, set `audit_status=failed`, leave mask/lesion/DWI
comparison metrics blank, and continue auditing other cases. A mismatch is a
finding, not permission to align or resample images.

Every readable imaging series must also report its own 3D geometry in the
series result: `size_xyz`/shape, `spacing_xyz_mm`, `origin_xyz_mm`,
`direction_3x3`, slice count, field of view (`fov_xyz_mm`), in-plane x/y
spacing, and z spacing. The audit must calculate actual adjacent-slice
spacing from sorted slice `ImagePositionPatient` values by projecting each
adjacent IPP difference onto the slice normal derived from
`ImageOrientationPatient` (the normalized row/column cross product). Do not
derive it from only the IPP z-coordinate difference. Report whether that spacing is
uniform and the observed min/median/max, and report the discrepancy
between metadata z spacing (`SpacingBetweenSlices`, falling back to
`SliceThickness` only as an explicitly recorded fallback) and IPP-derived
spacing. A missing or unreadable geometry field is explicit and does not get
filled by a filename or directory inference.

For any DWI/ADC/FLAIR geometry comparison, the result is a series-level
candidate comparison only. Its status and mismatch fields must never create a
formal multi-modal case pairing, promote an ADC/FLAIR series, or enable
cross-modal pixel statistics. The only formal case relationship remains the
confirmed DWI-to-mask relationship returned by `build_index()`.

### DWI-mask comparison

For a confirmed pair, the DWI image returned by `build_index()` is the image
reference and the returned NIfTI mask is the binary lesion annotation. The
comparison is valid only when both objects are readable and the geometry check
passes. Record `dwi_mask_comparison_status=match` for a valid comparison,
including a valid empty mask, and `failed` for an unreadable object, invalid
UID-linked DWI series, or geometry mismatch. This comparison reports mask
occupancy and DWI intensities restricted by the mask; it does not compute a
prediction-vs-ground-truth score and does not use any ADC/FLAIR pixels.

## Intensity, mask, and lesion metrics

Every readable imaging series receives raw and reconstructed intensity
semantics. The record must state whether SimpleITK values are stored pixel
values or have been reconstructed using `RescaleSlope` and
`RescaleIntercept`, whether rescaling was applied, and which metadata supports
that statement. The tool must not apply an unrecorded vendor-specific
conversion. For every readable series, report finite/non-finite counts and,
for both all voxels and nonzero voxels, the statistics `min`, `p1`, `p5`,
`p25`, `median`, `p75`, `p95`, `p99`, `max`, `mean`, and `std`. Nonzero means
the exact predicate `value != 0` after the documented value semantics; a
series with no nonzero finite voxels has explicit empty/non-applicable
statistics. ADC nonzero statistics are mandatory in the series result.

Formal case-level DWI all-voxel and nonzero statistics are permitted only for
the readable DWI source of a `build_index()` row with `match_status=matched`.
ADC and FLAIR values remain series-level, retain
`pairing_status=unverified`, and never populate case statistics. A non-finite
DWI volume is a failed intensity audit; raw geometry and readable-series
metadata remain reportable and non-finite-dependent statistics are blank.

For a confirmed case whose DWI and mask pass the strict geometry check, DWI
lesion intensity statistics over finite voxels where `mask > 0` must include
`p5`, `p25`, `median`, `mean`, `p75`, and `p95`, together with finite voxel
count. For a geometry mismatch, set
`lesion_intensity_status=skipped_due_to_geometry_mismatch` and leave these
lesion statistics blank. Do not calculate ADC/FLAIR lesion case statistics or
DWI-ADC pixel-level statistics.

The mask is interpreted as binary with `mask > 0` for statistics; source mask
values and unique values are reported separately. No mask values are changed
on disk.

Report the following mask/lesion values:

- `mask_voxel_count` and `mask_volume_mm3`;
- `mask_volume_ml` as `mask_volume_mm3 / 1000`;
- foreground slice count and foreground slice ratio over the full z slice
  count;
- `component_count_26`;
- 3D bounding-box voxel size and physical millimetre size;
- centroid in voxel coordinates and physical coordinates;
- largest- and smallest-component voxel counts and physical volumes as
  descriptive values;
- index-space lesion min/max bounds in `(x, y, z)`; and
- physical-coordinate lesion min/max bounds in millimetres using the image
  origin, spacing, and direction.

Connected components use 26-connectivity in the `(z, y, x)` NumPy array,
including face, edge, and corner adjacency. Connectivity is fixed and
reported in the summary. Components are descriptive only: there is no
largest-component filtering, deletion, or mask repair.

For an empty mask, set `empty_mask=true`, foreground slice count and ratio to
zero, component count and all component/volume/bounding-box/centroid fields to
zero or the defined empty value, and `lesion_intensity_status=empty_mask`;
all lesion intensity and lesion coordinate fields are blank. The DWI
whole-volume intensity summary may still be reported. Empty masks are valid
audited cases and are not failures.

For rotated directions, compute physical lesion/world bounds by transforming
all eight index-space bounding-box corners with origin, spacing, and direction,
then taking component-wise min/max across those world points.

## Geometry-only XY simulation

The audit records the established `resample_inplane()` semantics without
importing that helper from the sibling worktree and without writing an image.
For an input with array shape `(z, y, x)` and spacing `(sx, sy, sz)` in
`(x, y, z)` order, and target `(tx, ty)`:

```text
simulated_x = round(x * sx / tx)
simulated_y = round(y * sy / ty)
simulated_z = z
simulated_spacing = (tx, ty, sz)
```

Origin and direction are preserved conceptually; z slice count and z spacing
are always preserved. The implementation must calculate and report the input
and simulated size/spacing, target values, and the formula version. It must
not call a sibling-worktree module, resample pixels, or save a simulated image
or mask. The simulation is geometry evidence only and does not alter any
case metric or pairing status.

The implementation and tests must verify actual SimpleITK/GDCM rescale
behavior, including whether returned values are stored or already transformed
by RescaleSlope/RescaleIntercept. Record the observed semantics and evidence;
never apply RescaleSlope/RescaleIntercept unconditionally a second time.

## Outputs

The output directory contains exactly these files:

1. `case_statistics.csv` — one row for every `build_index()` row with
   `match_status=matched`, including failed case audits with explicit status
   and blank unavailable metrics. It never contains ADC/FLAIR case columns.
2. `dicom_series_statistics.csv` — one row per UID-isolated DICOM series, including
   readable/unreadable status, metadata/classification evidence, geometry and
   intensity summaries, candidate identifiers, and ADC/FLAIR
   `pairing_status=unverified` evidence where applicable.
3. `dataset_summary.json` — schema/version, normalized inputs, protocol flags,
   inventory and result counts, geometry tolerances, XY simulation semantics,
   failure records, and the exact three relative result filenames.

Rows and lists are sorted deterministically by normalized relative path,
SeriesInstanceUID, and case ID as applicable. Arrays and structured values in
CSV cells use compact JSON with stable key ordering. Missing/not-applicable
values are empty CSV cells or JSON `null`, never guessed values.

Prepare all three report contents in an isolated temporary directory under the
requested output parent. Only after every write and validation succeeds,
atomically replace/rename the prepared files into the empty output directory.
On error, clean up temporary artifacts without touching `ci1_root`; exactly the
three named files may remain in a successful final output.

### `case_statistics.csv` schema

The header is fixed and includes at least:

```text
case_id,patient_id,timepoint,pairing_status,pairing_source,build_index_match_status,
dwi_source_path,mask_path,dwi_series_uid,dwi_series_path,audit_status,read_status,
error_code,error_message,geometry_status,geometry_mismatch_fields,
dwi_mask_comparison_status,
dwi_size_xyz,mask_size_xyz,dwi_spacing_xyz_mm,mask_spacing_xyz_mm,
dwi_origin_xyz_mm,mask_origin_xyz_mm,dwi_direction_3x3,mask_direction_3x3,
mask_unique_values,empty_mask,mask_voxel_count,mask_volume_mm3,mask_volume_ml,
component_connectivity,component_count_26,largest_component_voxels_26,
largest_component_volume_mm3_26,smallest_component_voxels_26,
smallest_component_volume_mm3_26,foreground_slice_count,foreground_slice_ratio,
bbox_voxel_size_xyz,bbox_physical_size_mm_xyz,centroid_voxel_xyz,
centroid_physical_xyz_mm,lesion_index_min_xyz,lesion_index_max_xyz,
lesion_physical_min_xyz_mm,lesion_physical_max_xyz_mm,
dwi_finite_voxel_count,dwi_nonfinite_voxel_count,dwi_all_min,dwi_all_max,
dwi_all_mean,dwi_all_std,dwi_all_median,dwi_all_p05,dwi_all_p95,
lesion_intensity_status,lesion_finite_voxel_count,lesion_min,lesion_max,
lesion_mean,lesion_std,lesion_median,lesion_p05,lesion_p25,lesion_p75,
lesion_p95,
xy_target_spacing_xy_mm,xy_simulated_size_xyz,xy_simulated_spacing_xyz_mm,
xy_z_preserved
```

`case_id`, `patient_id`, and `timepoint` retain the exact `build_index()`
values or its documented path-derived values. The pairing source/status and
all failure fields are mandatory even when metrics are unavailable.

### `dicom_series_statistics.csv` schema

The header is fixed and includes at least:

```text
internal_case_patient_id,series_instance_uid,gdcm_series_id,series_directory,
relative_file_count,
read_status,uid_status,metadata_status,error_code,error_message,
series_description,protocol_name,sequence_name,image_type,dicom_modality,
manufacturer,manufacturer_model_name,rows,columns,pixel_spacing,
slice_thickness,spacing_between_slices,image_position_patient,
image_orientation_patient,rescale_slope,rescale_intercept,bits_allocated,
bits_stored,pixel_representation,b_value,metadata_field_status,
modality_class,
classification_source,classification_confidence,classification_evidence,
candidate_patient_id,candidate_timepoint,candidate_id_source,
pairing_status,pairing_evidence,confirmed_dwi_case_id,
size_xyz,shape_zyx,spacing_xyz_mm,origin_xyz_mm,direction_3x3,slice_count,
fov_xyz_mm,inplane_spacing_xy_mm,z_spacing_metadata_mm,
z_spacing_metadata_source,z_spacing_ipp_min_mm,z_spacing_ipp_median_mm,
z_spacing_ipp_max_mm,z_spacing_ipp_uniform,
z_spacing_metadata_ipp_discrepancy_mm,pixel_type,voxel_count,
finite_voxel_count,nonfinite_voxel_count,intensity_value_semantics,
raw_value_semantics,reconstructed_value_semantics,rescale_applied,
intensity_metadata_source,all_min,all_p1,all_p5,all_p25,all_median,all_p75,
all_p95,all_p99,all_max,all_mean,all_std,nonzero_min,nonzero_p1,
nonzero_p5,nonzero_p25,nonzero_median,nonzero_p75,nonzero_p95,
nonzero_p99,nonzero_max,nonzero_mean,nonzero_std
```

`confirmed_dwi_case_id` is populated only for the UID-isolated DWI series
linked to a `build_index()` matched row. It is blank for ADC/FLAIR and
unknown series. ADC/FLAIR candidate fields never imply confirmation.

### `dataset_summary.json` schema

The top-level object contains these fixed keys:

```json
{
  "schema_version": 1,
  "tool": "ci1_raw_data_audit",
  "protocol": {
    "read_only": true,
    "formal_pairing_source": "prepare_ci1_dwi_dataset.build_index",
    "formal_pairing_status": "match_status=matched",
    "adc_flair_formal_pairing": false,
    "resampling": false,
    "xy_simulation_only": true,
    "connected_component_connectivity": 26
  },
  "inputs": {
    "ci1_root": "<normalized absolute path>",
    "output_dir": "<normalized absolute path>",
    "target_spacing_xy_mm": [0.4892368018627167, 0.4892368018627167],
    "geometry_tolerances": {
      "spacing_abs_mm": 1e-5,
      "origin_abs_mm": 1e-4,
      "direction_abs": 1e-6
    }
  },
  "inventory": {},
  "counts": {},
  "xy_simulation": {},
  "failures": [],
  "result_files": [
    "dicom_series_statistics.csv",
    "case_statistics.csv",
    "dataset_summary.json"
  ]
}
```

`inventory`, `counts`, and `xy_simulation` are populated with concrete numeric
values and deterministic formula/field names by the implementation; they may
not contain placeholders. `failures` contains objects with `scope`, stable
identifier, `error_code`, and concise message. It must not contain patient
pixel data.

## Failure and reporting rules

- **Unreadable NIfTI or DICOM:** preserve an inventory/series/case row with
  `read_status=unreadable`, an error code/message, and no derived metrics that
  require the unreadable object. Continue with independent items.
- **Unknown or conflicting modality:** preserve a series row with
  `modality_class=unknown`, `classification_confidence=none`, and evidence;
  do not assign it to a formal case or use it for ADC/FLAIR pairing.
- **Unmatched DWI:** preserve the `build_index()` status in summary/failure
  reporting and series output where available, but omit the DWI from formal
  `case_statistics.csv`.
- **Geometry mismatch:** keep the matched case row, mark it failed, list the
  mismatching fields, and leave all mask/lesion/intensity-comparison metrics
  blank. Never resample to make it pass.
- **UID failure or ambiguity:** keep separate series records and do not merge
  or promote any affected series to a formal DWI case source.
- **Output failure:** do not leave a partial report presented as complete;
  report the write failure through the process exit status. No raw input is
  modified as part of error handling.

## Test acceptance

The later implementation must add synthetic-only tests under
`non_teacher_student_files/tests/` and must not scan, open, convert, or depend
on real CI-1 data. Tests must make deterministic value assertions, not only
shape or successful-execution assertions. The minimum cases are:

1. Two synthetic DICOM series with distinct `SeriesInstanceUID` values in
   one directory remain two series rows and never merge.
2. SimpleITK `(x, y, z)` size/spacing and NumPy `(z, y, x)` arrays are
   asserted with explicit values, including physical-coordinate conversion.
3. A known binary mask and spacing produce the exact expected physical volume
   in mm³ and mL.
4. Shape, spacing, origin, and direction mismatches are each detected, with
   no resampled metrics produced.
5. XY simulation asserts the exact rounded x/y sizes while z count and z
   spacing remain unchanged; no image file is created.
6. An empty mask produces zero volume/components, blank lesion intensities,
   `empty_mask=true`, and a valid non-failing case audit.
7. Synthetic metadata assertions cover DWI/ADC/FLAIR/unknown classification,
   source and confidence, and the rule that ADC/FLAIR candidate evidence is
   series-only with `pairing_status=unverified`.
8. Output containment and deterministic row ordering are asserted. The test
   fixture never uses a real patient path or real CI-1 file.

## Explicit non-goals

- Converting CI-1 data into nnU-Net or any other training format.
- Training, inference, model evaluation, prediction generation, or
  experiment-result production.
- Writing resampled images, masks, cropped data, PNGs, plots, checkpoints, or
  any output outside the three named result files.
- Mutating, repairing, renaming, deleting, or overwriting raw CI-1 data.
- Treating directory names, file names, file counts, or metadata similarity as
  a replacement for the confirmed `build_index()` DWI-to-mask status.
- Formal ADC/FLAIR case pairing without an authoritative pairing table.
- ADC/FLAIR fields in `case_statistics.csv`, ADC/FLAIR lesion statistics, or
  DWI-ADC pixel-level analysis.
- Using an ADC/FLAIR candidate to derive an ROI, alter a mask, or change a
  DWI-to-mask case status.
- Cross-series registration, spatial interpolation, resampling, or geometry
  repair.
- Clinical validity claims, completeness claims, or formal segmentation
  benchmark claims based on an audit report alone.
- Importing the sibling-worktree `standalone_nnunet2d` implementation merely
  to perform the XY simulation.
## Approved requirement corrections (authoritative)

- The exact three output names are `dicom_series_statistics.csv`, `case_statistics.csv`, and `dataset_summary.json`.
- The default XY target is `(0.4892368018627167, 0.4892368018627167)` mm. Report raw and processed shapes/spacings, `xy_resampled`, X/Y scaling, Z spacing, and count preservation.
- The series output must retain every available requested DICOM item: internal case identifier; UID; description/protocol/sequence/image type/modality/manufacturer/model; rows/columns/pixel spacing/thickness/spacing between slices; image position/orientation; rescale slope/intercept; bit fields; and generic b-value when readable, otherwise blank. No vendor private-tag logic is allowed.
- Geometry adds FOV, slice count, X/Y/Z, Z uniformity, and metadata-vs-IPP discrepancy. Multimodal statuses remain unverified at series level only.
- Each readable series receives all-voxel and nonzero `min`, `p1`, `p5`, `p25`, `median`, `p75`, `p95`, `p99`, `max`, `mean`, and `std`, with stored-vs-rescaled semantics. ADC raw/nonzero statistics are series-level only.
- Confirmed `build_index` DWI-mask cases include foreground slice count/ratio, bounding box in voxel/mm coordinates, centroid in voxel/physical coordinates, and largest/smallest component volume; DWI lesion `p5`, `p25`, `median`, `mean`, and `p75`/`p95`, or `skipped_due_to_geometry_mismatch`.
- Summary distributions/counts cover modalities/unknown, rounded XY spacing, Z spacing/slice ranges, XY resample count/proportion, mismatches, lesion volume/slice/component summaries, confirmed DWI raw/nonzero/lesion, and series-only ADC raw/nonzero.
- ADC/FLAIR always remain series-only with `pairing_status=unverified` and never populate `case_statistics` fields, lesion metrics, or DWI-ADC pixel analysis.
