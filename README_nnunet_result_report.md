# Dataset501 fold 0 validation report

Run from this checkout on the server in Windows CMD. Replace the two raw-data
placeholders with the actual original DWI and ground-truth directories. The
images directory must contain one `CASE_0000.nii.gz` per validation case; the
labels and saved prediction directories use `CASE.nii.gz`. A single-channel
Dataset501 input is supported.

```bat
conda activate nnunet5090
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
set MODEL=C:\lijialin\models3d\nnUNet\nnUNet_results\Dataset501_StrokeLesion\nnUNetTrainerUPerNetTopK10EarlyStopping__nnUNetPlans__2d
python generate_nnunet_result_report.py --model-dir "%MODEL%" --fold 0 --images-dir "<ACTUAL_DWI_IMAGES_DIR>" --labels-dir "<ACTUAL_GT_LABELS_DIR>" --prediction-dir "%MODEL%\fold_0\validation" --metrics-dir "%MODEL%\fold_0\multi_metrics" --checkpoint "%MODEL%\fold_0\checkpoint_best.pth" --output-dir "<NEW_REPORT_OUTPUT_DIR>" --check
```

`--check` reads only metadata and filenames. It neither opens image voxels nor
loads a model. If the metadata checks pass, verify which checkpoint produced
the saved validation predictions from a training/validation log or other
auditable record. Replace `checkpoint_best.pth` if the evidence identifies
`checkpoint_final.pth`. Then run the same command without `--check` and add:

```bat
--prediction-checkpoint-declaration "USER DECLARED: <evidence file, command, checkpoint SHA256, validation time and TTA details>"
```

The script labels this statement as user-declared provenance; it cannot prove
the saved mask was produced by that checkpoint. Without it, a full report is
refused. Use a fresh output directory: existing targets and overlaps with any
source are refused. The output parent must exist. Complete output appears only
after every mode-specific file is written; an incomplete or failed run removes its private staging folder without leaving a report in the final output directory.
UPerNet outputs include `summary.png`, `encoder_stages_heatmap.png`, both native channel sheets, `architecture_overview.png`, the metrics CSV/PNG, UTF-8 `report.txt`, and complete `ppt/` pages with slide previews. UPerNet has no independent detail implementation. Full runs verify image,
GT and prediction headers for every metrics-covered case. The TXT records the
count. Only exact `image_reader_writer=SimpleITKIO` plans are supported.

The report is a **single fold 0 validation report**, not five-fold OOF or a
clinical conclusion. It reads the existing full-set case and summary metrics;
it preserves existing metrics; only missing TP/FP/FN are counted from saved masks. Empty GT cases have no selected positive
slices. The PNG feature magnitude and DWI overlay panels show `mean(abs(deepest encoder stage))` from
fresh raw-image preprocessing and no mirror TTA. Each patch is upsampled to
its own sliding window and fused with the nnU-Net Gaussian weights, then
mapped through inverse padding, resampling, cropping and transpose. These
diagnostic features are not the saved prediction's internal features, lesion
probabilities, attention, or explanations. The heatmap display is normalized
per slice and cannot compare absolute feature strength across cases.

## SERVER VALIDATION PENDING

- Confirm the server's installed `nnunetv2`, external Trainer source, plans and
  checkpoint load strictly with the script; local synthetic tests do not prove
  server compatibility.
- Establish saved prediction provenance: best/final, exact checkpoint hash,
  validation command, mirroring/TTA, tile step and Gaussian settings. The
  diagnostic feature run intentionally disables TTA; document any difference.
- Verify feature mapping on a known server case: original image/GT/prediction
  geometry, crop, transpose, resampling, sliding window, padding, and exported
  feature position. A report generated before this check remains provisional.
- Confirm source `case_metrics.csv` and `summary_metrics.json` cover the same
  full validation set, including F2 mode and missing-value meanings.

No local real checkpoint, raw patient image, or large model was loaded during
development. No training or validation prediction is run by this command.

## Visual revision and server review

The separate `architecture_overview.png` shows encoder stages, selected feature connections, PPM, FPN and output path from loaded model attributes. `summary.png` contains cases only for every supported identity. High and low Dice groups each show
three cases with up to three GT-positive slices. Green is TP, red FP, and blue
FN in the display overlay. The summary's inferno scale uses per-slice 1st-99th
percentile normalization; DWI+feature uses alpha 0.48. This scale is for display
only and is not comparable across cases. The native-channel sheet
(feature_channels.png) displays up to eight uniformly spaced channel IDs at the
unaltered low resolution, with nearest pixel display. It represents one local
preprocessed window selected as the first enumerated window containing the
first displayed slice center, without GT-dependent channel or window selection.
The TXT records window index, preprocessed bounds, actual sliding-window count,
feature shape and channel IDs. Nonidentity slice-axis transpose or slice-axis
resampling is rejected for this localization rather than silently mismapped.

Each native channel is displayed using its own exact minimum and maximum:
`(value - min) / (max - min)` in `[0,1]`. A constant channel displays uniformly
at zero. All channels share the same coolwarm color scale and colorbar; blue/red
indicate only relative low/high values, not negative/positive values. Native
colors cannot compare absolute activation between channels or cases and are not
lesion probabilities. Columns across the two layers have no guaranteed shared
meaning. This differs from the summary's DWI and feature-magnitude
panels, which each use per-slice 1st–99th percentile normalization to `[0,1]`
and display constant maps at zero.

On the server, first run the command above with --check. Confirm the metrics
coverage, checkpoint path and a new output directory. Then run the same
command without --check, adding the provenance declaration shown above.
Inspect the generated structure against the server's loaded encoder and decoder;
verify selected stage channels, relative sizes, PPM scales, FPN width and output
shape. Compare one case's original slice, crop/transpose, preprocessing window
bounds and native-channel window index. Check DWI, GT, saved prediction and
feature spatial alignment, plus the saved prediction's separate provenance.
Use a fresh independent V4 output directory. `feature_channels_64x64.png` shows
native `encoder.stages.3` channels from the same case, original/preprocessed
slice, and diagnostic window as `feature_channels.png`. The script verifies the
actual intermediate BCHW shape is 64x64 during the same no-TTA forward and
refuses other native sizes. The TXT records both modules, actual sizes, fixed
channel IDs and common window provenance. Check these facts on the server;
local synthetic tests do not establish the real intermediate shape. The
existing V3 report directory remains read-only. These server steps have not
been run locally.


## Original nnUNetTrainerTopK10 (Dataset501 DWI, 2D, fold 0)

This mode keeps the saved prediction checkpoint source **UNKNOWN** by default.
The explicit `checkpoint_final.pth` is loaded for new GT-free, no-TTA
diagnostic features only. The saved predictions are not replayed, and their
weights and historical TTA remain unverified. To record an explicit user
confirmation, add both `--confirm-prediction-checkpoint` and
`--prediction-checkpoint-declaration "<ORIGINAL USER STATEMENT
CONFIRMING FINAL AND TTA IF KNOWN>"`; the report labels it USER CONFIRMED,
not independently verified. If the statement establishes TTA, also add
`--historical-tta enabled` or `--historical-tta disabled`; otherwise it remains
unknown. The existing UPerNet mode still requires its
source declaration.

Run from this checkout on the server in Windows CMD, replacing the two input
placeholders and choosing a fresh, separate output directory whose parent
already exists:

```bat
conda activate nnunet5090
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
set MODEL=C:\lijialin\models3d\nnUNet\nnUNet_results\Dataset501_StrokeLesion\nnUNetTrainerTopK10__nnUNetPlans__2d
python generate_nnunet_result_report.py --model-dir "%MODEL%" --fold 0 --images-dir "<ACTUAL_DWI_IMAGES_DIR>" --labels-dir "<ACTUAL_GT_LABELS_DIR>" --prediction-dir "%MODEL%\fold_0\validation" --metrics-dir "%MODEL%\fold_0\multi_metric_evaluation" --checkpoint "%MODEL%\fold_0\checkpoint_final.pth" --output-dir "<NEW_INDEPENDENT_PARENT>\nnunet_topk10_fold0_report_split_arch" --check
```

After metadata-only `--check`, run the same command without `--check` on the
server. The script will verify every metrics-covered case's physical geometry,
restore the exact Trainer with strict state-dict loading, and inspect the actual
network. It supports the verified PlainConvUNet path and rejects unknown
architectures. The native 64x64 encoder stage is derived from patch size and
strides and confirmed in the actual no-TTA hook output; no resizing substitutes
for a missing stage. Training deep supervision is disabled by nnU-Net's predictor
for the inference main output, while checkpoint head parameters remain loaded.

In original TopK10 mode, `summary.png` contains the six case comparisons,
legend, feature scale and provenance note. `architecture_overview.png` shows
the module-level eight-stage/seven-step flow and both feature capture positions.
`architecture_detail.png` separates encoder stages, decoder operations, and
training versus inference heads into numbered panels. The architecture figures
read stage channels, strides, patch size and output classes from the restored
network. Both new PNGs are required for atomic delivery alongside
`feature_channels.png`, `feature_channels_64x64.png`, and UTF-8 `report.txt`.
The existing UPerNet figures remain; delivery now also requires the shared CSV/PNG metrics table.

**SERVER VALIDATION PENDING:** actual checkpoint/network compatibility, native
64x64 stage, geometry and feature alignment, report presentation, and saved
prediction provenance/TTA. Local small synthetic checks are engineering evidence
only.
# Standalone H2Former diagnostic reports

The same entry point supports `--source standalone-h2former`. Supply existing source-space predictions, their prediction manifest, existing multi-metric output, a resolved config, and the exact diagnostic checkpoint. The checkpoint identity must be `h2former`, `h2former_lite_upernet`, or `h2former_lite_upernet_w128_ppm1236`; the W128_PPM1236 factory checks FPN width 128, PPM scales (1,2,3,6), and PPM output width 128. The saved prediction checkpoint may remain `UNKNOWN`; diagnostic features from the named checkpoint are not a replay of saved predictions. To record a user assertion, supply both `--confirm-prediction-checkpoint` and `--prediction-checkpoint-declaration "..."`.

Windows CMD example for original H2Former **best** (set `RAW`, `OUT`, and verify every path before running):

```cmd
set ROOT=C:\lijialin\models3d\nnUNet\nnUNet_results\Dataset501_StrokeLesion\H2Former_fold0_bs4_adamw
python generate_nnunet_result_report.py --source standalone-h2former --model-dir "%ROOT%" --fold 0 --images-dir "%RAW%\imagesTr" --labels-dir "%RAW%\labelsTr" --prediction-dir "%ROOT%\full_volume_predictions_best\predictions" --metrics-dir "%ROOT%\full_volume_metrics_best" --manifest "%ROOT%\full_volume_predictions_best\prediction_manifest.json" --config "%ROOT%\resolved_config.json" --checkpoint "%ROOT%\checkpoint_best.pth" --output-dir "%OUT%\h2former_best_report" --check
```

For **latest**, pair `checkpoint_latest.pth` with `full_volume_predictions_latest` and `full_volume_metrics_latest`, and select another new output directory. Remove `--check` only after metadata checks and real server paths are verified. `--allow-pending` explicitly permits pending checkpoint and manifest status; the report labels that status.

Windows CMD skeleton for **W128_PPM1236** (fill the prediction, metric, and manifest paths from the actual server output; their existence is not assumed):

```cmd
set ROOT=C:\lijialin\models3d\nnUNet\nnUNet_results\H2Former_UPerNet_W128_PPM1236\fold_0
python generate_nnunet_result_report.py --source standalone-h2former --model-dir "%ROOT%" --fold 0 --images-dir "%RAW%\imagesTr" --labels-dir "%RAW%\labelsTr" --prediction-dir "<EXISTING_PREDICTIONS_DIRECTORY>" --metrics-dir "<EXISTING_METRICS_DIRECTORY>" --manifest "<EXISTING_PREDICTION_MANIFEST>" --config "%ROOT%\resolved_config.json" --checkpoint "%ROOT%\checkpoint_best.pth" --output-dir "%OUT%\h2former_w128_ppm1236_report" --check
```

Missing predictions or multi-metric output cause an error; this command never creates either. Full report generation checks checkpoint contents, source-space geometry for every case, manifest coverage, and model identity before loading the model. The report is a fold-0 diagnostic document, not five-fold evidence. Real server checkpoint, case geometry, provenance, and final figure validation remain pending until executed there.

Manifest checkpoint must be a nonempty object with explicit model identity parsed
by the strict production factory, matching the diagnostic checkpoint/config and
complete W128_PPM1236 contract. UNKNOWN weights provenance does not relax identity.
Manifest alignment uses the production validator: aligned needs valid evidence;
pending cannot carry evidence and still needs explicit `--allow-pending`.

Numbered architecture panels follow actual CNN/MS/Swin fusion and original skip
or Lite PPM/FPN routes with current model dimensions/channels/width/scales. Window
logits and source aggregation/unpadding are separate from diagnostic magnitude.
TXT records supplied parsed config, training `data_source.type` (or unknown), and
the separate raw-source diagnostic entry. Both native layer mappings include
slice, window index/count/padded coordinates, padding offsets, source/padded/window
sizes, stage/module, BCHW shape and channel IDs, without feature arrays. Miniature
local previews/smoke are synthetic engineering evidence; server validation PENDING.


## Shared full-evaluation metrics table

Both official Trainers (`nnUNetTrainerTopK10`,
`nnUNetTrainerUPerNetTopK10EarlyStopping`) and all three supported standalone
identities (`h2former`, `h2former_lite_upernet`,
`h2former_lite_upernet_w128_ppm1236`) now require `metrics_table.csv`,
`metrics_table.png` and the appended full table in `report.txt` for atomic
publication. Ordinary `nnUNetTrainer` and a separate W128-only identity remain
**pending adapters**, not supported by this change.

Every evaluated case is included, sorted by `case_id`, including missing Dice.
Six finite-Dice cases are still required for diagnostic selection. Existing
metrics and extra columns retain source values and CSV float precision.
Dice/IoU/F2/Recall remain 0..1; AVD is already percent, HD95 mm, LCD lesion-count
difference. Missing/nonfinite metrics: CSV `NA`, PNG/TXT `N/A`; invalid nonnumeric
metrics fail. Unknown extra columns are preserved and never averaged.
`row_type=case` distinguishes real cases from `case_macro_mean` (病例平均指标,
counts blank) and `all_case_voxel_total` (全病例 TP/FP/FN 总计, metrics blank).

Counts accept exact nonnegative integer strings/integers, never float conversion.
Lowercase `tp/fp/fn` are canonical; uppercase aliases must agree when both exist,
including missing-vs-present conflicts. Missing is never zero. Complete source
counts add no voxel reads. Supplied pred_voxels/gt_voxels and known nonzero-denominator
overlap definitions are checked. Each case with missing counts loads its saved
prediction/GT pair once, sequentially, requiring matching Size/Spacing/Origin/
Direction. All three counts are computed to cross-check existing fields; only
missing fields are filled. No resampling, inference, connected components, HD95
or full metric evaluation is invoked for counts. The original full-report
model diagnostic forwards still occur separately. Existing metrics are never
recomputed; only missing TP/FP/FN are counted from saved original-space masks.

Fallback requires official dataset.json background=0 plus one positive label,
or the existing validated standalone two-class config/manifest contract (0/1).
Actual masks must contain only those labels. Ignore, region, multiclass, negative,
fractional and nonfinite labels/contracts are refused. Source reuse adds no mask
inspection. Per-count `*_source` records `source_csv` or
`recomputed_from_saved_masks`; per-case label/geometry notes distinguish source
claims from fallback validation. Header checks do not prove source count values.
TP/FP/FN are full-volume voxel counts, with no spacing scaling or slice averaging.

Source aggregation must declare `macro average over cases`. n_cases, finite
valid_cases, optional case_ids and source means are checked against full rows.
Verified source means are retained; missing means use finite case-macro fallback,
with valid N and provenance. Unknown aggregation or contradicting means/coverage
(including Inf-valid-count disagreement) fail. Unknown f2_mode stays unknown;
paper/standard are not converted. Empty-overlap missing values are not filled
with 1. `--check` parses metadata/filenames and pending count coverage only:
no headers/voxels, checkpoint contents, model or table output. Input files and
old reports stay read-only; new files join the existing staging/publication guard.

### Server templates (user executed; SERVER VALIDATION PENDING)

Keep server environment `nnunet5090`. Replace every placeholder with verified
existing paths/statements; select a fresh absent output directory with existing
parent. First run --check, then remove only --check for full generation.
The earlier CMD examples also work and now emit the shared table.

CMD official UPerNet (full mode still requires a truthful declaration):

```cmd
conda activate nnunet5090
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
python generate_nnunet_result_report.py --model-dir "<EXISTING_UPERNET_MODEL_DIR>" --fold 0 --images-dir "<DWI_DIR>" --labels-dir "<GT_DIR>" --prediction-dir "<SAVED_PRED_DIR>" --metrics-dir "<MULTI_METRICS_DIR>" --checkpoint "<MODEL_DIR>\fold_0\checkpoint_best.pth" --prediction-checkpoint-declaration "<ORIGINAL_USER_STATEMENT>" --output-dir "<NEW_PARENT>\upernet_metrics_report" --check
```

PowerShell original TopK10 (saved weights provenance UNKNOWN by default):

```powershell
conda activate nnunet5090
$env:nnUNet_extTrainer = Join-Path $PWD 'nnunet_ext_trainers'
python generate_nnunet_result_report.py --model-dir '<EXISTING_TOPK10_MODEL_DIR>' --fold 0 --images-dir '<DWI_DIR>' --labels-dir '<GT_DIR>' --prediction-dir '<SAVED_PRED_DIR>' --metrics-dir '<MULTI_METRICS_DIR>' --checkpoint '<MODEL_DIR>\fold_0\checkpoint_final.pth' --output-dir '<NEW_PARENT>\topk10_metrics_report' --check
```

PowerShell standalone (each supported identity retains its exact pairing):

```powershell
conda activate nnunet5090
python generate_nnunet_result_report.py --source standalone-h2former --model-dir '<EXISTING_MODEL_DIR>' --fold 0 --images-dir '<DWI_DIR>' --labels-dir '<GT_DIR>' --prediction-dir '<SAVED_PRED_DIR>' --metrics-dir '<MULTI_METRICS_DIR>' --manifest '<EXISTING_PREDICTION_MANIFEST>' --config '<EXISTING_RESOLVED_CONFIG>' --checkpoint '<EXISTING_DIAGNOSTIC_CHECKPOINT>' --output-dir '<NEW_PARENT>\h2former_metrics_report' --check
```

Add --allow-pending only when explicitly accepting existing pending alignment.
UNKNOWN stays UNKNOWN unless both confirmation flag and a truthful original
statement are supplied. Identity/alignment guards remain mandatory.

SERVER VALIDATION PENDING: real count comparison, full-set geometry/labels,
source-summary semantics, checkpoint/manifest compatibility, actual all-case
long-table visual inspection. Local CPU synthetic tests and PNG inspection are
engineering evidence. No server commands were executed. Independent Level 3
review remains for user/main-coordinator manual dispatch.


### Optional per-case source definition declarations

`summary_metrics.json` aggregation declares only macro aggregation. `f2_mode`
does not establish case formulas, foreground or zero-denominator semantics.
Both production entries pass the source summary directly to the shared service.
Absent optional definitions default to `unknown`; original metrics are retained.
The report never rewrites source CSV/JSON to obtain evidence. A source producer
with actual definition evidence may supply these caller declarations:

```json
{
  "count_foreground": "positive",
  "metric_definitions": {
    "dice": {"formula": "2tp/(2tp+fp+fn)", "zero_denominator": "missing"},
    "iou": {"formula": "tp/(tp+fp+fn)", "zero_denominator": "missing"},
    "recall": {"formula": "tp/(tp+fn)", "zero_denominator": "missing"},
    "f2": {"formula": "5tp/(5tp+4fp+fn)", "zero_denominator": "missing"}
  }
}
```

Only these four metric keys and `formula`/`zero_denominator` fields are supported;
each metric can be omitted. Supported formulas are the exact strings above or
`unknown`. F2 also supports standard `5tp/(5tp+fp+4fn)`; its explicit formula
must agree with an existing paper/standard f2_mode declaration. Zero-denominator
values are `missing` or `unknown` (default); `missing` requires a known formula.
Known formulas validate only corresponding source metrics: finite nonzero-ratio
conflicts fail, and finite zero-denominator values fail when `missing` is declared.
Unknown empty semantics are never forced to NaN or both_empty=1. Metrics are not
recomputed or overwritten. `count_foreground` supports `positive` (>0) or
`unknown` (default), independently of formulas. Existing evaluator summaries lack
these fields and remain unknown. Declarations are recorded as caller-declared
source provenance, never independently verified evaluator identity. Saved-mask
binary label evidence proves mask interpretation only, not source metric formulas
or source foreground. Do not edit historical sources merely to pass validation.
Exact integers, aliases, coverage, pred_voxels=TP+FP, gt_voxels=TP+FN and fallback
conflict checks remain strict; complete source counts add no voxel reads.

Case missing metrics (empty/whitespace strings, None, NA, N/A, NaN, Inf) and macro
means with no valid values export CSV `NA`, PNG/TXT `N/A`. Structural blanks stay
blank: mean TP/FP/FN and total metric columns.

SERVER VALIDATION PENDING: real counts, geometry/labels, source summary and
definition evidence, checkpoint/manifest compatibility and full real-case long
table visual inspection. Local validation is synthetic CPU engineering evidence.


## All encoder stages and readable PPT pages (2026-09-30)

The five integrated identities are exactly:

| Entry | Encoder capture | Separate architecture |
| --- | --- | --- |
| nnUNetTrainerTopK10 | every actual encoder.stages output | overview + numbered detail |
| nnUNetTrainerUPerNetTopK10EarlyStopping | every encoder.stages output, including stages omitted by selected_feature_indices | overview + named encoder/PPM/FPN overview pages; no detail implementation |
| h2former | E0=decode2 input[1], E1=decode3 input[1], E2=decode4 input[1], E3=decode4 input[0] | overview + numbered detail |
| h2former_lite_upernet | decoder input[0][0..3] | overview + numbered detail |
| h2former_lite_upernet_w128_ppm1236 | decoder input[0][0..3] | overview + numbered detail |

`encoder_stages_heatmap.png` is titled **各级编码器特征强度** and shows
`mean(abs(channel))`, the actual padded model-input window and measured C×H×W.
Encoder numbering is **0-based**, distinct from diagram S1..S4 names.
Lite stage metadata distinguishes argument `input_index=0` from
`feature_index=stage`; captions read `decoder input[0][stage]`. Historical
native `input_index` remains a list index for compatibility, explicitly labeled
with `input_index_semantics` and `module_input_index=0`.
H2Former captures the fused CNN/MS features after Swin, directly as BCHW; no
CNN-only layer, decoder output, PPM feature or token reordering is substituted.
The original six selected cases and `slices[0]` are retained. Empty GT/native
rows explain the absence rather than substitute another case.

Every row's input/stages/native channels share one forward and one tile.
Official nnU-Net keeps the first enumerated window containing the preprocessed
slice center. H2Former keeps the first enumerated tile on its first displayed
slice. GT only chooses display slices, never model input, channel IDs or tile
positions. TXT records source/preprocessed slice, window bounds, padding,
crop/transpose/resampling, input normalization and stage modules/shapes.

Model-input normalization remains the nnU-Net plans preprocessor or H2Former
full-volume z-score. Stage/input **display** independently uses finite-only
1st-99th percentiles. Constants and all-invalid grids display zero; nonfinite
pixels display zero and metadata records invalid counts/status. Stable scaled
arithmetic handles extreme finite ranges. Nearest display of native grids does
not establish original-space alignment. This is neither attention, Grad-CAM,
lesion probability, saved prediction replay nor every channel's full content.
Independent scales cannot compare absolute strength across stages/cases.

Hooks reduce channels immediately. Only the representative window retains all
stage 2D grids, its input and at most eight channels from each of the two
existing native layers. Other windows retain only reduced deepest maps needed
by the original summary. There is no per-stage forward or extra full-volume
stage cache. Hooks are removed on success and exceptions; inference_mode/eval
is used, and model outputs are unchanged. Original deepest bilinear/Gaussian
fusion and inverse mapping remain in use.

Main sheets preserve all content with larger text and adjusted spacing. For
PPT use `ppt/`, rather than shrinking a complete long sheet onto one slide:

- `summary_high_01.png`, `summary_low_01.png`, etc.: one case, up to three slices, all six columns.
- `encoder_stages_high_01.png`, `encoder_stages_low_01.png`, etc.: up to two cases and four stage columns, repeating input for additional stage groups.
- `feature_channels_high_01.png`, `feature_channels_low_01.png`, and matching `feature_channels_64x64_*`: one case, all selected channels in a 2×4 layout. The historical 64x64 filename is retained; H2Former titles use measured sizes.
- `architecture_overview_01.png` and `architecture_detail_01.png`, etc.: complete numbered panels with named tensor connections; official UPerNet overview splits its encoder, PPM and FPN substructures.
- `metrics_table_01.png`, etc.: adaptive row pages and four-column groups, repeating case_id and row_type. Every original column, case, macro summary and voxel summary remains covered. CSV/TXT ordered records and formulas are unchanged.

Every main/page PNG has a corresponding `ppt/*_slide.png`: a real 1920×1080
preview, proportional fitting within an 1800×960 rectangle, without cropping or
stretching. `ppt/layout_manifest.json` records original dimensions, placement,
scale, renderer text bounds, effective pixel font sizes and content coverage.
The simple preset is 16×9 inches, 100 dpi, 14pt body / 22pt titles; effective
sizes depend on actual placement, not just nominal points. Complete main sheets
can be too dense for one slide and are **not** claimed to pass the PPT body-text
threshold. Pages target at least about 18px body and 28px titles. Full provenance
stays in TXT. A CJK font already installed on the host is required; no font or
PPTX dependency is installed automatically.

All new stage/main/page/preview files participate in private staging and
completeness validation before atomic publication. Existing targets and source
paths cannot be reused. --check stays metadata-only and creates no images.
No training/model/optimizer/seed/checkpoint selection, Dataset501 DWI-only data,
fixed patient five-fold splits, source-space formal metric contract or historical
UNKNOWN/USER CONFIRMED/USER DECLARED/pending status changes.

### Fresh-directory server commands (user runs these)

These are parameter templates, not claims about actual server paths or mask
provenance. Use an output parent that already exists and a new report directory.
For either exact official Trainer, set MODEL to its existing
Dataset501_StrokeLesion/<TRAINER>__nnUNetPlans__2d directory:

```bat
set MODEL=<ACTUAL_OFFICIAL_MODEL_DIR>
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
python generate_nnunet_result_report.py --model-dir "%MODEL%" --fold 0 --images-dir "<ACTUAL_DWI_IMAGES_DIR>" --labels-dir "<ACTUAL_GT_LABELS_DIR>" --prediction-dir "<ACTUAL_SAVED_PREDICTION_DIR>" --metrics-dir "<ACTUAL_METRICS_DIR>" --checkpoint "<ACTUAL_CHECKPOINT>" --output-dir "<NEW_ENCODER_STAGES_REPORT_DIR>" --check
```

Full official command: reuse these same parameters, remove --check and add
--device cuda. For official UPerNet add
`--prediction-checkpoint-declaration "<ACTUAL USER DECLARATION WITH EVIDENCE>"`.
Original TopK10 may remain UNKNOWN. Only if the user actually confirms its
prediction checkpoint add both --confirm-prediction-checkpoint and the actual
--prediction-checkpoint-declaration. Never invent a source declaration.

For any of the three supported standalone identities:

```bat
python generate_nnunet_result_report.py --source standalone-h2former --model-dir "<ACTUAL_STANDALONE_RUN_DIR>" --fold 0 --images-dir "<ACTUAL_DWI_IMAGES_DIR>" --labels-dir "<ACTUAL_GT_LABELS_DIR>" --prediction-dir "<ACTUAL_SAVED_PREDICTION_DIR>" --metrics-dir "<ACTUAL_METRICS_DIR>" --checkpoint "<ACTUAL_CHECKPOINT>" --config "<ACTUAL_RESOLVED_CONFIG_JSON>" --manifest "<ACTUAL_PREDICTION_MANIFEST_JSON>" --output-dir "<NEW_STANDALONE_ENCODER_STAGES_REPORT_DIR>" --check
```

Full standalone command: remove --check, add --device cuda, and use
--allow-pending only for a deliberately allowed pending checkpoint/manifest.
UNKNOWN remains UNKNOWN unless both explicit confirmation and an actual source
declaration are supplied. Generate on the server when no feature cache exists;
old PNGs cannot reconstruct the new encoder tensors.

**SERVER VALIDATION PENDING:** real checkpoint compatibility, feature source,
case geometry/window localization and real text layouts have not been verified
locally. Local evidence is tiny CPU synthetic engineering validation only.
Independent Level 3 review is a separate manual next step, bound to the final
uncommitted diff; implementation completion does not imply review PASS.
