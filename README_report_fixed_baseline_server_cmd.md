# Frozen comparison cohort: server CMD commands

Updated 2026-10-03. User runs these on the training server after deploying the
reviewed code. Local validation uses synthetic CPU data and loader sentinels;
none of these real-data server commands has been run locally. Replace every
placeholder. Do not rerun training/prediction/HD95 or change existing splits.

## Generate selection once

Only the exact standard Dataset501 nnUNetTrainer / nnUNetPlans / 2d fold-0
baseline is accepted. The generator never loads a checkpoint. Existing saved
prediction checkpoint provenance stays UNKNOWN; best/final filenames prove
nothing about which weights produced those masks.

```bat
conda activate newconda
cd /d "<SERVER_REPORT_CODE_CHECKOUT>"
set "RESULTS=C:\lijialin\models3d\nnUNet\nnUNet_results"
set "BASELINE=%RESULTS%\Dataset501_StrokeLesion\nnUNetTrainer__nnUNetPlans__2d"
set "DWI=<ACTUAL_ORIGINAL_DWI_ROOT>"
set "GT=<ACTUAL_ORIGINAL_GT_ROOT>"
set "SPLITS=<EXISTING_DATASET501_SPLITS_FINAL_JSON>"
set "SEL=%RESULTS%\reports\comparison_fixed_cohort_v1\comparison_selection.json"
python generate_report_comparison_selection.py --baseline-model-dir "%BASELINE%" --fold 0 --splits-file "%SPLITS%" --images-dir "%DWI%" --labels-dir "%GT%" --prediction-dir "%BASELINE%\fold_0\validation" --baseline-metrics-file "%BASELINE%\fold_0\multi_metric_evaluation\case_metrics.csv" --output-json "%SEL%"
```

The derived parent is created after validation; an existing SEL is refused.
If newconda is unavailable, use the user's existing compatible environment;
no installation/upgrade is prescribed. Image collection is flat NIfTI with
`<case>_0000.nii[.gz]`, label/prediction `<case>.nii[.gz]`. Extra raw cases are
allowed, duplicates/missing validation cases are refused. Predictions and metrics
must match the entire fixed validation population, never an intersection.

Default size grouping uses original GT foreground count * product(mm spacing)
/1000 mL. NIfTI must explicitly declare mm, spacing must be positive and direction
orthonormal. Positive GT linear 1/3 and 2/3 quantiles define small V<=q1, medium
q1<V<=q2, large V>q2. Optional `--thresholds-ml <q1> <q2>` is the sole override.
Repeated thresholds/empty or undersized groups fail, without rank bucketing.
Good=max baseline Dice, bad=min among remaining; ID breaks ties. Display order
is large-good/bad, medium-good/bad, small-good/bad. Empty GT remains in full metrics.

Each entry freezes up to three distinct GT-positive slices, displayed ascending;
representative_slice is maximum GT area then minimum index independently. Axis
is SimpleITK array z/y/x, axis 0, 0-based; no anatomical-plane equivalence is claimed.
This is GT/baseline-guided post-hoc qualitative display, not random sampling or
formal evidence of improvement. GT never enters preprocessing/inference ROI.

SHA256 covers exact input file bytes and geometry. Recompression, header edits
or content edits require a new manifest. Root relocation uses current explicit
DWI/GT roots and relative filenames with identical contents. Semantic selection_id
covers strict UTF-8 canonical JSON (sorted keys, compact separators, no NaN),
excluding only selection_id and locator-only locations. All cases, slices,
volume/group/scores, geometry, source hashes, split hash and code hashes remain
included. Runtime roots do not modify the frozen source file.

## Standard baseline report

Ordinary official nnUNetTrainer additionally requires exact plans and loaded
PlainConvUNet, official class/module, SimpleITKIO and 2D inference. A different
architecture is rejected. Real checkpoint compatibility remains pending.

```bat
set "CP_BASE=<ACTUAL_BASELINE_FOLD_0_BEST_OR_FINAL_CHECKPOINT>"
set "OUT_BASE=%RESULTS%\reports\standard_fixed_cohort_v1"
python generate_nnunet_result_report.py --model-dir "%BASELINE%" --fold 0 --images-dir "%DWI%" --labels-dir "%GT%" --prediction-dir "%BASELINE%\fold_0\validation" --metrics-dir "%BASELINE%\fold_0\multi_metric_evaluation" --checkpoint "%CP_BASE%" --selection-json "%SEL%" --output-dir "%OUT_BASE%" --check
python generate_nnunet_result_report.py --model-dir "%BASELINE%" --fold 0 --images-dir "%DWI%" --labels-dir "%GT%" --prediction-dir "%BASELINE%\fold_0\validation" --metrics-dir "%BASELINE%\fold_0\multi_metric_evaluation" --checkpoint "%CP_BASE%" --selection-json "%SEL%" --output-dir "%OUT_BASE%" --device cuda
```

Standard and TopK10 saved prediction provenance defaults UNKNOWN. Only actual
user confirmation permits adding BOTH `--confirm-prediction-checkpoint` and
`--prediction-checkpoint-declaration "<ACTUAL ORIGINAL STATEMENT>"` to full mode.
This is USER CONFIRMED, not independently verified. A known historical TTA
`--historical-tta enabled|disabled` also requires that original statement.

## TopK10 report

```bat
set "nnUNet_extTrainer=%CD%\nnunet_ext_trainers"
set "TOPK=%RESULTS%\Dataset501_StrokeLesion\nnUNetTrainerTopK10__nnUNetPlans__2d"
set "TOPK_METRICS=<ACTUAL_TOPK10_METRICS_DIR>"
set "CP_TOPK=<ACTUAL_TOPK10_FOLD_0_BEST_OR_FINAL_CHECKPOINT>"
set "OUT_TOPK=%RESULTS%\reports\topk10_fixed_cohort_v1"
python generate_nnunet_result_report.py --model-dir "%TOPK%" --fold 0 --images-dir "%DWI%" --labels-dir "%GT%" --prediction-dir "%TOPK%\fold_0\validation" --metrics-dir "%TOPK_METRICS%" --checkpoint "%CP_TOPK%" --selection-json "%SEL%" --output-dir "%OUT_TOPK%" --check
python generate_nnunet_result_report.py --model-dir "%TOPK%" --fold 0 --images-dir "%DWI%" --labels-dir "%GT%" --prediction-dir "%TOPK%\fold_0\validation" --metrics-dir "%TOPK_METRICS%" --checkpoint "%CP_TOPK%" --selection-json "%SEL%" --output-dir "%OUT_TOPK%" --device cuda
```

## UPerNet report

```bat
set "UPER=%RESULTS%\Dataset501_StrokeLesion\nnUNetTrainerUPerNetTopK10EarlyStopping__nnUNetPlans__2d"
set "UPER_METRICS=<ACTUAL_UPERNET_METRICS_DIR>"
set "CP_UPER=<ACTUAL_UPERNET_FOLD_0_BEST_OR_FINAL_CHECKPOINT>"
set "OUT_UPER=%RESULTS%\reports\upernet_fixed_cohort_v1"
python generate_nnunet_result_report.py --model-dir "%UPER%" --fold 0 --images-dir "%DWI%" --labels-dir "%GT%" --prediction-dir "%UPER%\fold_0\validation" --metrics-dir "%UPER_METRICS%" --checkpoint "%CP_UPER%" --selection-json "%SEL%" --output-dir "%OUT_UPER%" --check
python generate_nnunet_result_report.py --model-dir "%UPER%" --fold 0 --images-dir "%DWI%" --labels-dir "%GT%" --prediction-dir "%UPER%\fold_0\validation" --metrics-dir "%UPER_METRICS%" --checkpoint "%CP_UPER%" --selection-json "%SEL%" --output-dir "%OUT_UPER%" --device cuda --prediction-checkpoint-declaration "<ACTUAL ORIGINAL SAVED-PREDICTION PROVENANCE STATEMENT>"
```

UPerNet full mode requires the real statement; never invent one from filenames.
A statement may accurately declare UNKNOWN. It stays USER DECLARED, unverified.

## Three standalone H2Former identities

Repeat this template separately for exactly `h2former`, `h2former_lite_upernet`,
and `h2former_lite_upernet_w128_ppm1236`, assigning each its actual run sources and
a fresh unique OUT_H2. W128-only and other variants are not supported. Checkpoint,
config and prediction manifest model identity must agree and return single_output.

```bat
set "RUN_H2=<ACTUAL_SUPPORTED_H2FORMER_RUN_DIR>"
set "PRED_H2=<ACTUAL_EXISTING_SOURCE_SPACE_PREDICTION_DIR>"
set "METRICS_H2=<ACTUAL_H2FORMER_FULL_METRICS_DIR>"
set "CP_H2=<ACTUAL_H2FORMER_CHECKPOINT>"
set "CONFIG_H2=<ACTUAL_RESOLVED_CONFIG_JSON>"
set "MANIFEST_H2=<ACTUAL_PREDICTION_MANIFEST_JSON>"
set "OUT_H2=%RESULTS%\reports\<H2FORMER_IDENTITY>_fixed_cohort_v1"
set "PENDING="
rem Set PENDING=--allow-pending only if you knowingly allow actual pending alignment.
python generate_nnunet_result_report.py --source standalone-h2former --model-dir "%RUN_H2%" --fold 0 --images-dir "%DWI%" --labels-dir "%GT%" --prediction-dir "%PRED_H2%" --metrics-dir "%METRICS_H2%" --manifest "%MANIFEST_H2%" --config "%CONFIG_H2%" --checkpoint "%CP_H2%" --selection-json "%SEL%" --output-dir "%OUT_H2%" %PENDING% --check
python generate_nnunet_result_report.py --source standalone-h2former --model-dir "%RUN_H2%" --fold 0 --images-dir "%DWI%" --labels-dir "%GT%" --prediction-dir "%PRED_H2%" --metrics-dir "%METRICS_H2%" --manifest "%MANIFEST_H2%" --config "%CONFIG_H2%" --checkpoint "%CP_H2%" --selection-json "%SEL%" --output-dir "%OUT_H2%" %PENDING% --device cuda
```

Manifest path contracts are unchanged; relocating raw roots may require a
separately correct current prediction manifest. Frozen selection itself must stay
unchanged. UNKNOWN source remains UNKNOWN unless BOTH actual confirmation and
original statement are supplied. --allow-pending permits explicit pending status,
not fabricated aligned evidence or inconsistent checkpoint/manifest identity.

## Compare and validate on server

`--check` verifies strict JSON/selection_id and metadata/file coverage only; no
voxel reads, model/checkpoint contents or outputs. Content/geometry remains
PENDING. Full mode verifies selected input/GT byte hashes, original geometry,
positive slices/volume and entire current prediction/metrics coverage before
model load. It never changes cases or labels when current Dice reverses.

```bat
findstr /b /c:"selection_id:" "%OUT_BASE%\report.txt" "%OUT_TOPK%\report.txt" "%OUT_UPER%\report.txt"
fc /b "%SEL%" "%OUT_BASE%\comparison_selection.json"
fc /b "%SEL%" "%OUT_TOPK%\comparison_selection.json"
fc /b "%SEL%" "%OUT_UPER%\comparison_selection.json"
rem Repeat findstr and fc /b for each actual H2Former output directory.
```

Inspect identical six-case/slice/representative/order content in all copies/TXT.
Native/stage maps use that representative and one shared diagnostic forward/window
per model. Official mapping explicitly rejects nonidentity transpose or slice-axis
resampling/cropped-out representative. H2Former source-axis mapping has no crop or
resampling, with its own full-volume z-score/padding/windows. Corresponding original
slice does not imply equal windows/channel semantics across models. Full metrics
CSV/PNG and summaries always cover the entire population including empty/unselected
cases. Publication fails on missing/corrupt main images or missing/altered copy/ID.
No report creates ppt/, layout_manifest, page PNGs or slide previews.

SERVER VALIDATION PENDING: real six-case grouping availability, mm units, byte
fingerprints/geometry, official checkpoint/Trainer and native-64x64 stages,
standalone identity/alignment, original-to-preprocessed slice/window mapping,
all real text/box layout and source provenance. Synthetic engineering passes do
not establish server compatibility. Independent Level 3 review remains for user
manual dispatch against the exact final snapshot. Historical V6/ppt is NOT TOUCHED.
