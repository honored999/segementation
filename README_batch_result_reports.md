# Batch saved-result reports (Dataset501, frozen fold 0)

Run from this report checkout on the server in the existing working environment:

```bat
conda activate nnunet5090
python batch_generate_result_reports.py --results-roots "C:\lijialin\models3d\nnUNet\nnUNet_results" "C:\lijialin\models3d\nnUNet\nnUNet_results_ISLES508_pretrained_ES" --images-dir "<ORIGINAL_DWI_IMAGES_DIR>" --labels-dir "<ORIGINAL_GT_LABELS_DIR>" --selection-json "<EXISTING_COMPARISON_SELECTION_JSON>" --metrics-map "batch_report_metrics_map.example.json" --output-dir "C:\lijialin\models3d\nnUNet\batch_reports_dataset501_v1" --dry-run
```

Fill the raw/selection paths and edit a copy of the metrics-map example for actual
server paths. Remove `--dry-run` from the same command to generate saved reports.
The parent-owned `run_batch_saved_reports.cmd` is an alternative launcher; inspect
its configured paths first. No installation or environment upgrade is needed.

`--dry-run` prints JSON with metadata discovery, exclusions, metric associations,
and full 19-case coverage checks. It creates no output directory/files, reads no
voxels/checkpoints/models, and does not verify source content hashes or geometry.
Full mode verifies these before publishing each saved report. Use a fresh output
root beside the result roots; existing outputs and resolved overlap with result,
raw, selection-directory, explicit metric, or code roots are refused. Raw files,
existing reports, checkpoints and the selection source are never changed.

Discovery supports Dataset501 single-DWI official `Trainer__Plans__2d*` configs
with `dataset.json`, `plans.json`, and `fold_0/validation`. Saved-only reports do
not depend on the Trainer/model adapter. Standalone runs require an associated
`resolved_config.json`, `prediction_manifest.json`, and `predictions/`, with exact
identity `h2former`, `h2former_lite_upernet`, or
`h2former_lite_upernet_w128_ppm1236`. H2 dataset scope is never inferred from a
folder name: predictions/metrics must equal the full frozen population, current
DWI/GT sources must match, and explicit dataset declarations cannot conflict.
Pending H2 config/manifest alignment requires opt-in `--allow-pending`; it stays
pending in the report. Default runs do not opt in.

Reports, aborted/preflight/smoke/audit directories, Dataset508 and other dataset
branches, multimodal/non-DWI metadata, and unsupported standalone identities are
listed as exclusions. This does not claim every architecture has feature support.

Metric association uses specific candidate names, preserving complete snapshot
suffixes: `best_prediction -> best_metrics`,
`full_volume_predictions_best -> full_volume_metrics_best`, and
`full_volume_predictions_best1000 -> full_volume_metrics_best1000`. Official
candidates are `multi_metrics` and `multi_metric_evaluation` inside `fold_0`.
Across both roots, CSV absolute `niftipath`/`prediction_path`/`pred_path`/`pred_file`
references may identify an exact saved-prediction file set. Identical case IDs
alone never choose a metrics directory. Missing/ambiguous associations are skipped
with reasons; clinical metrics are never recomputed.

For external or ambiguous metrics use `--metrics-map <JSON>`:

```json
{
  "C:\\absolute\\run\\fold_0\\validation": "C:\\absolute\\external_metrics"
}
```

Keys must be exact absolute prediction directories; values must be existing
metric directories containing `case_metrics.csv` and `summary_metrics.json`.
Explicit mappings still require full population/summary checks; conflicting CSV
prediction references fail. The supplied example maps the external Stage6 and
NoStage7 directories under the ISLES508-pretrained result root, and the pretrained
TopK10ES `_counts` directory under the other/main result root. Edit it to match
actual paths. Names and caller mappings establish candidate associations only;
they do not independently prove saved-mask/checkpoint provenance.

Each successful job writes `report.txt`, `comparison_selection.json`,
`metrics_table.csv`, `metrics_table.txt`, `metrics_table.png`, and `summary.png`.
The table includes all 19 cases. The four-column summary shows DWI, GT, saved
prediction, and TP/FP/FN for the six frozen cases with three fixed slice slots
(blank if fewer slices were frozen). Full mode checks all scalar 3D volumes,
physical geometry/mm units, binary 0/1 masks, selected source hashes and frozen
slices. Clinical metrics retain source values; only missing TP/FP/FN are filled
from saved masks, with existing source counts cross-checked. No inference, HD95
re-evaluation, architecture, or fabricated features occur in saved-only mode.

Add `--features --device cuda` (or `cpu`) for sequential optional diagnostic
subprocesses using the unchanged existing generator. Exact official adapters:
`nnUNetTrainer`, `nnUNetTrainerTopK10`, and
`nnUNetTrainerUPerNetTopK10EarlyStopping`, all with `nnUNetPlans__2d`. Official
features use the existing `fold_0/checkpoint_best.pth` diagnostically, with saved
prediction provenance UNKNOWN. UPerNet receives a truthful UNKNOWN declaration.
TopK10ES, PlainConvStage6 and NoStage7 have no feature adapter here. H2 features
require an existing exact manifest checkpoint path and matching manifest
`checkpoint.sha256` (or `checkpoint.checkpoint_sha256`); ordinary manifests often
lack this proof, so historic mutable best/latest/snapshot pairing is skipped.
The batch never invents hashes or asserts user checkpoint confirmation.

Diagnostics go into each job's separate `feature_diagnostics/` directory. The
existing generator may include extra architecture images there. Unsupported or
failed features preserve the saved-only report and appear in `features`/status.
All jobs run sequentially. `batch_summary.csv`, `batch_summary.json`, `batch.log`,
and `logs/job_*.log` record exclusions, skips, failures and partial feature output.
Incomplete saved reports are marked `FAILED.txt`. Exit 0 means no job/scan/feature
errors; exit 1 includes skipped eligible jobs or failures, while ordinary scan
exclusions and unsupported feature skips are nonfatal. Invalid global arguments
or unsafe output boundaries exit 2. For another full run use a fresh output root.

Static source inspection only: no local tests, syntax compilation, image reads,
model/checkpoint loads or inference were run. Real server rendering and adapter
compatibility remain unverified. An installed CJK font is required by the reused
report_visuals helpers; no font/dependency is installed automatically.
