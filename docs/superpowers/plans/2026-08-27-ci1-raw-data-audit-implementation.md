# CI-1 Raw Data Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a synthetic-tested, read-only CI-1 raw-data audit command that uses only `build_index(match_status=matched)` for formal DWI-to-mask cases and writes exactly three auditable result files.

**Architecture:** Create one self-contained audit module and one synthetic-only test module. Reuse `prepare_ci1_dwi_dataset.build_index()` only for confirmed DWI lineage; independently enumerate UID-isolated GDCM series, read public metadata/pixels through SimpleITK, and never use filename-based ADC/FLAIR helpers. Build and validate all outputs in memory, stage them under the output parent, and atomically finalize only after all three are valid.

**Tech Stack:** Python 3.10+, `argparse`, `csv`, `json`, `math`, `pathlib`, `tempfile`, `shutil`, `dataclasses`, `numpy`, `SimpleITK/GDCM`, `unittest`/`pytest`.

---

## File map and implementation boundary

- Create `non_teacher_student_files/audit_ci1_raw_data.py`: CLI, path safety, inventory, GDCM discovery, metadata/classification, lineage, geometry, IPP spacing, intensity/rescale observation, mask metrics, XY simulation, schemas, summary, and atomic outputs.
- Create `non_teacher_student_files/tests/test_ci1_raw_data_audit.py`: synthetic-only tests; use temporary SimpleITK NIfTI/DICOM fixtures and patched `build_index`; never scan or reference real CI-1 data.
- Do not modify the design document, existing source/tests/configuration, data, generated outputs, or sibling worktrees.

Use these exact public names and constants:

```python
DEFAULT_TARGET_SPACING_XY_MM = (0.4892368018627167, 0.4892368018627167)
SPACING_TOLERANCE_MM = 1e-5
ORIGIN_TOLERANCE_MM = 1e-4
DIRECTION_TOLERANCE = 1e-6
RESULT_FILES = ("dicom_series_statistics.csv", "case_statistics.csv", "dataset_summary.json")

def normalize_and_validate_paths(ci1_root: Path, output_dir: Path) -> tuple[Path, Path]: ...
def parse_target_spacing(x_mm: float | None, y_mm: float | None) -> tuple[float, float]: ...
def enumerate_inventory(ci1_root: Path) -> list[dict]: ...
def discover_dicom_series(dicom_dir: Path) -> list[SeriesRecord]: ...
def classify_modality(metadata: dict[str, str]) -> tuple[str, str, str, str]: ...
def compare_geometry(image: sitk.Image, mask: sitk.Image) -> tuple[str, list[str]]: ...
def ipp_spacing(positions: list[tuple[float, float, float]], orientation: tuple[float, ...]) -> dict: ...
def summarize_intensity(values: np.ndarray) -> dict: ...
def mask_metrics(mask_array_zyx: np.ndarray, image: sitk.Image) -> dict: ...
def simulate_xy_geometry(size_xyz: tuple[int, int, int], spacing_xyz: tuple[float, float, float], target_xy: tuple[float, float]) -> dict: ...
def audit_dataset(ci1_root: Path, output_dir: Path, target_xy: tuple[float, float] = DEFAULT_TARGET_SPACING_XY_MM) -> dict: ...
def main(argv: list[str] | None = None) -> int: ...
```

Define `SeriesRecord` and `CaseRecord` with every output field represented. Use compact deterministic JSON for arrays/objects: `json.dumps(value, sort_keys=True, separators=(",", ":"))`; use an empty CSV cell and JSON `null` for unavailable values.

`CASE_COLUMNS` must be exactly the design order, including `foreground_slice_count`, `foreground_slice_ratio`, `bbox_voxel_size_xyz`, `bbox_physical_size_mm_xyz`, `centroid_voxel_xyz`, `centroid_physical_xyz_mm`, `lesion_index_min_xyz`, `lesion_index_max_xyz`, `lesion_physical_min_xyz_mm`, `lesion_physical_max_xyz_mm`, `largest_component_voxels_26`, `largest_component_volume_mm3_26`, `smallest_component_voxels_26`, `smallest_component_volume_mm3_26`, `lesion_p25`, and `lesion_p75`; use the complete header in design lines 356–374.

`SERIES_COLUMNS` must be exactly the design order in design lines 385–415, including all public DICOM fields, `metadata_field_status`, classification fields, candidate fields, UID/geometry fields, raw/nonzero intensity statistics, and rescale semantics.

### Task 1: Contracts, schemas, and target spacing

**Files:** Create both allowed files.

- [ ] **Step 1: Write failing tests.** Assert exact `RESULT_FILES`, default spacing, required case fields and absence of `adc_`/`flair_` case fields, compact sorted `encode_cell`, and `parse_target_spacing(None,None)`, paired-only flags, finite positive validation, and rejection of a single target flag.
- [ ] **Step 2: Run red.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -q`; expected FAIL because the new module/contracts do not exist.
- [ ] **Step 3: Implement minimally.** Import `json`, `math`, `argparse`, `csv`, `pathlib`, `tempfile`, `shutil`, `numpy`, `SimpleITK`; use `try: from .prepare_ci1_dwi_dataset import build_index` with an import fallback for direct test imports. Add complete constants, fixed schemas, records, `encode_cell`, and exact target validation.
- [ ] **Step 4: Run green.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -q`; expected visible PASS summary.

### Task 2: Read-only path safety and deterministic inventory

**Files:** Modify `non_teacher_student_files/audit_ci1_raw_data.py`; test `non_teacher_student_files/tests/test_ci1_raw_data_audit.py`.

- [ ] **Step 1: Write failing tests.** In a temporary root, assert `normalize_and_validate_paths` rejects equal/descendant output (including a symlink-resolved descendant), rejects a non-empty existing output, accepts a separate empty output, and `enumerate_inventory` returns regular files sorted by normalized POSIX relative path with type/size/read status and creates no files.
- [ ] **Step 2: Run red.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'path or inventory' -q`; expected FAIL on missing implementation.
- [ ] **Step 3: Implement minimally.** Resolve both paths before scanning; require input directory; reject `output == root` or successful `output.relative_to(root)`; require existing output directory empty. Enumerate only regular files with `rglob`, sort by `relative_to(root).as_posix()`, classify `.nii/.nii.gz`, `.dcm`, and extensionless/other files via read-only GDCM/SimpleITK probe, and capture per-item errors instead of writing or silently skipping.
- [ ] **Step 4: Run green.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'path or inventory' -q`; expected visible PASS.

### Task 3: UID-isolated GDCM discovery, metadata, modality, and rescale observation

**Files:** Modify the new module; test the new test module.

- [ ] **Step 1: Write failing synthetic tests.** Write two synthetic DICOM series with distinct `0020|000e` UIDs into one directory and assert two rows, no merge, sorted GDCM IDs/files, and UID consistency. Set public tags `0008|103e`, `0018|1030`, `0018|0024`, `0008|0008`, `0008|0060`, `0008|0070`, `0008|1090`, `0028|0010`, `0028|0011`, `0028|0030`, `0018|0050`, `0018|0088`, `0020|0032`, `0020|0037`, `0028|1053`, `0028|1052`, `0028|0100`, `0028|0101`, `0028|0103`, and `0018|9087`; assert metadata values/statuses and DWI/ADC/FLAIR/unknown classification, source, confidence, and evidence. Assert rescale fields record observed SimpleITK/GDCM semantics rather than a guessed second transformation.
- [ ] **Step 2: Run red.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'dicom or uid or modality or rescale' -q`; expected FAIL.
- [ ] **Step 3: Implement minimally.** For each candidate directory call `GetGDCMSeriesIDs`, sort IDs, call `GetGDCMSeriesFileNames`, sort files, read `0020|000e` from every readable file, and require one nonblank consistent UID per series. Keep failed UID rows separate. Read only public tags from a deterministic representative and compare meaningful fields across readable files; every requested field gets `present/absent/unreadable/inconsistent`. Classify from normalized public description/protocol text only: DWI contains `dwi` and not `adc`; ADC contains `adc`; FLAIR contains `flair` and not `t1`; otherwise unknown. Report actual classification tags/confidence/evidence; never use directory/file names, DICOM Modality alone, or private tags. Read pixels with `ImageSeriesReader` and explicitly record whether returned values are stored or already transformed; never unconditionally apply slope/intercept again.
- [ ] **Step 4: Run green.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'dicom or uid or modality or rescale' -q`; expected visible PASS.

### Task 4: Series geometry, projected IPP spacing, and intensity summaries

**Files:** Modify the new module; test the new test module.

- [ ] **Step 1: Write failing tests.** Assert SimpleITK size/spacing/origin/direction are XYZ while `GetArrayFromImage` is ZYX. Use oblique orientation and IPPs whose z difference is not the slice distance; assert `ipp_spacing` computes `abs(dot(delta_ipp, normalized(cross(row,column))))`, reports min/median/max/uniformity, and does not use z-only difference. Assert `SpacingBetweenSlices` is preferred and `SliceThickness` is an explicit fallback. Assert intensity summaries contain finite/nonfinite counts and all/nonzero `min,p1,p5,p25,median,p75,p95,p99,max,mean,std`, with explicit empty subset semantics.
- [ ] **Step 2: Run red.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'geometry or ipp or intensity' -q`; expected FAIL.
- [ ] **Step 3: Implement minimally.** Use XYZ for `GetSize/GetSpacing/GetOrigin`, ZYX for arrays; compute FOV as `size_xyz * spacing_xyz`. Sort slices by projected position, compute normal from normalized row/column cross product, and compare metadata z spacing with projected IPP spacing. Use `np.isfinite` and exact `value != 0` for nonzero; use fixed percentiles `(1,5,25,50,75,95,99)`. Record series geometry, pixel type, counts, raw/reconstructed/value semantics and metadata evidence.
- [ ] **Step 4: Run green.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'geometry or ipp or intensity' -q`; expected visible PASS.

### Task 5: Confirmed build_index lineage, strict geometry, and mask/lesion metrics

**Files:** Modify the new module; test the new test module.

- [ ] **Step 1: Write failing tests.** Patch `build_index` with matched and `missing_dicom` rows; assert only matched rows appear in case CSV. Test shape, spacing, origin, and direction mismatches independently, asserting ordered mismatch fields and blank derived metrics. For a known binary mask assert exact voxel count, mm3/ml volume, foreground slice count/ratio, 26-connected component count, largest/smallest component counts/volumes, index bbox, centroid, lesion p5/p25/median/mean/p75/p95, and rotated-direction physical bounds from all eight bbox corners. Test empty mask zero fields, `empty_mask=true`, `lesion_intensity_status=empty_mask`, blank lesion intensities, and valid non-failing status.
- [ ] **Step 2: Run red.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'case or lineage or mask or component or mismatch' -q`; expected FAIL.
- [ ] **Step 3: Implement minimally.** Call `build_index(ci1_root)` once and preserve its exact paths/status. Create a case row only for `match_status == "matched"`, setting `pairing_status="confirmed"` and `pairing_source="prepare_ci1_dwi_dataset.build_index"`; do not rediscover/replace labels and omit unmatched rows from case CSV while recording their status in summary/failures/series where applicable. Establish exactly one source→UID DWI mapping using resolved path/file membership; zero or multiple is failed, never resolved arbitrarily. A mapped series with `modality_class="unknown"` remains valid confirmed lineage evidence and cannot cancel it. Never import `find_adc_segmentation_path`, any FLAIR filename helper, or overlay helpers. Read mask directly, never `read_mask_on_reference` and never resample. Compare exact size and tolerances spacing `1e-5`, origin `1e-4`, direction `1e-6`. On mismatch retain source geometry, fail comparison, set `lesion_intensity_status="skipped_due_to_geometry_mismatch"`, and blank mask/lesion/DWI comparison metrics. For a match use `mask > 0`, preserve unique source values, voxel volume product, 26-connectivity, `(z,y,x)`→`(x,y,z)` conversion, physical transform `origin + direction @ (index_xyz * spacing_xyz)`, all eight bbox corners, and finite masked DWI lesion statistics.
- [ ] **Step 4: Run green.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'case or lineage or mask or component or mismatch' -q`; expected visible PASS.

### Task 6: Geometry-only XY simulation and ADC/FLAIR boundary

**Files:** Modify the new module; test the new test module.

- [ ] **Step 1: Write failing tests.** Assert `simulate_xy_geometry((7,5,3),(1,2,4),(0.6,1.25))` returns `(8,8,3)`, `(0.6,1.25,4)`, preserved z count/spacing, scale factors and formula version, and creates no image. Assert ADC/FLAIR series rows always have `pairing_status="unverified"`, exact evidence `path_or_metadata_candidate_only; no_authoritative_pairing_table`, blank `confirmed_dwi_case_id`, and no ADC/FLAIR case fields. Patch forbidden helpers to raise and ensure the new module never calls/imports them.
- [ ] **Step 2: Run red.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'simulation or adc or flair or pairing' -q`; expected FAIL.
- [ ] **Step 3: Implement minimally.** Use exactly `round(x*sx/tx)`, `round(y*sy/ty)`, `(simulated_x,simulated_y,z)`, `(tx,ty,sz)`; record raw/simulated geometry, target, scales, `xy_resampled`, z preservation, and formula version without allocating pixels or importing sibling-worktree code. Keep ADC/FLAIR series-only; candidate patient/timepoint and rationale are non-authoritative evidence only. Never populate case ADC/FLAIR fields, lesion metrics, ROI, or DWI-ADC correspondence; unknown/conflicting series remain unknown and unassigned.
- [ ] **Step 4: Run green.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'simulation or adc or flair or pairing' -q`; expected visible PASS.

### Task 7: Fixed outputs, summary, and atomic finalization

**Files:** Modify the new module; test the new test module.

- [ ] **Step 1: Write failing end-to-end tests.** With a temporary synthetic root and patched `build_index`, call `audit_dataset`; assert output contains exactly `dicom_series_statistics.csv`, `case_statistics.csv`, `dataset_summary.json`, exact headers, deterministic row order by normalized relative series path/UID/GDCM ID and case ID/patient/timepoint/source, matched-only case rows, series-only ADC/FLAIR unverified evidence, and summary flags/counts/distributions including modality/unknown, rounded XY, z/slice ranges, resample count/proportion, mismatches, lesion volumes/slices/components, confirmed DWI all/nonzero/lesion stats, ADC raw/nonzero series stats, actual target spacing, and rescale evidence. Inject a write/validation failure and assert no partial final report is presented.
- [ ] **Step 2: Run red.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'output or summary or atomic' -q`; expected FAIL.
- [ ] **Step 3: Implement minimally.** `audit_dataset` validates paths before scanning, calls inventory and `build_index` once, discovers every candidate DICOM directory/UID series, attaches only path-rule candidate IDs, computes series/case results, sorts all rows/lists, and populates `schema_version=1`, `tool`, protocol flags (`read_only`, formal source/status, `adc_flair_formal_pairing=false`, `resampling=false`, `xy_simulation_only=true`, connectivity 26), normalized inputs, actual target spacing defaulting to `[0.4892368018627167,0.4892368018627167]`, tolerances, concrete inventory/counts/XY/failures, and exact result names. Stage all files in `tempfile.mkdtemp(prefix="ci1_audit_", dir=output_dir.parent)`, write fixed CSV/JSON, validate headers/JSON/row counts/exact names, then atomically replace/rename into the prevalidated empty output directory. On error remove only temporary artifacts and leave raw input untouched.
- [ ] **Step 4: Run green.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'output or summary or atomic' -q`; expected visible PASS.

### Task 8: CLI contract

**Files:** Modify only the new module/test module.

- [ ] **Step 1: Write failing tests.** Assert `main(["--ci1-root", root, "--output-dir", output]) == 0` for a synthetic fixture and exactly three outputs; assert one target flag, unsafe/non-directory roots, and non-empty output return nonzero without final results.
- [ ] **Step 2: Run red.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'cli' -q`; expected FAIL.
- [ ] **Step 3: Implement minimally.** Add required `--ci1-root`/`--output-dir`, optional paired float flags, target validation, concise stderr errors and return code `2` for pre-output validation/write failures, `0` after complete output, and `if __name__ == "__main__": raise SystemExit(main())`.
- [ ] **Step 4: Run green.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -k 'cli' -q`; expected visible PASS.

### Task 9: Final validation, review, and one commit

**Files:** No files beyond the two allowed implementation/test files.

- [ ] **Step 1: Run focused suite.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py -q`; expected visible PASS summary and synthetic-only execution.
- [ ] **Step 2: Run relevant CI-1 synthetic suite.** `conda run -n newconda python -m pytest non_teacher_student_files/tests/test_ci1_raw_data_audit.py non_teacher_student_files/tests/test_make_nnunet_dataset_from_ci1.py non_teacher_student_files/tests/test_ci1_dwi_adc_mask_audit.py non_teacher_student_files/tests/test_dicom_slice_selection.py -q`; expected visible PASS summary; do not pass a real CI-1 path.
- [ ] **Step 3: Perform read-only review.** Inspect the two files for exact schemas/names, no ADC/FLAIR helper or `read_mask_on_reference`, no resampling/image writes/private tags, projected IPP spacing, no unconditional rescale, confirmed source→UID lineage, unknown-not-canceling-DWI, series-only unverified ADC/FLAIR, eight-corner rotated bounds, resolved containment, atomic staging, and no real-data paths.
- [ ] **Step 4: Run diff checks without staging.** `git diff --check -- non_teacher_student_files/audit_ci1_raw_data.py non_teacher_student_files/tests/test_ci1_raw_data_audit.py`; expected no whitespace errors. Inspect `git diff --` for only the two scoped files.
- [ ] **Step 5: Create exactly one final commit only after all checks pass.** `git add -- non_teacher_student_files/audit_ci1_raw_data.py non_teacher_student_files/tests/test_ci1_raw_data_audit.py; git commit -m "feat: add CI-1 raw data audit"`; expected exactly that subject, no intermediate commits, no design modification, no generated outputs.

## Requirement-to-task mapping

- Corrected filenames, default/actual target spacing, fixed schemas, summary protocol/counts: Tasks 1 and 7.
- Read-only behavior, safe containment, empty output, failure semantics, atomic finalization: Tasks 2, 7, and 8.
- Inventory and path-derived candidate identifiers: Tasks 2 and 7.
- `build_index(match_status=matched)` as sole formal DWI↔mask lineage, unique source→UID mapping, unknown metadata not overriding lineage: Task 5.
- No filename replacement or formal ADC/FLAIR pairing; series-only `unverified`; no case fields or lesion/pixel analysis: Task 6 and Task 9.
- UID-isolated GDCM discovery, consistency, public metadata, generic b-value, classification source/confidence/evidence, no private tags: Task 3.
- XYZ/ZYX conventions, strict geometry, no resampling: Tasks 4 and 5.
- Orientation-normal IPP spacing, z fallback, uniformity/discrepancy: Task 4.
- SimpleITK/GDCM rescale observation and no second unconditional application: Task 3 and Task 9.
- Series and confirmed DWI intensity summaries, lesion p25/p75: Tasks 4 and 5.
- Foreground slices, bbox/centroid, 26-components, largest/smallest volumes, physical eight-corner bounds, empty mask: Task 5.
- XY simulation, exact rounding, z preservation, no output image: Task 6.
- Synthetic acceptance tests, focused/relevant suites, review, diff checks, exact final commit: Tasks 1–9.

## Self-review

- [ ] Every corrected design requirement maps to at least one task above.
- [ ] The case schema explicitly includes foreground slice, bbox, centroid, largest/smallest components, and lesion p25/p75, and contains no ADC/FLAIR case statistics.
- [ ] Names, signatures, statuses, tolerances, filenames, and output schemas are consistent across tasks.
- [ ] No `TBD`, `TODO`, or unresolved implementation placeholder appears.
- [ ] Every behavior group has a concrete test sketch, exact red command, minimal implementation direction, and exact green command.
- [ ] The plan authorizes only the two new implementation/test files; the design remains unchanged.
- [ ] The final commit is singular and uses exactly `feat: add CI-1 raw data audit`.
