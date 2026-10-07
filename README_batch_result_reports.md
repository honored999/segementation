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


## Quick v2 feature backfill (SERVER VALIDATION PENDING)

Use the server report checkout containing this change and its existing nnunetv2
2.8.1 environment. All eight source directories in
`batch_report_trainer_extensions.example.json` are filled from user-provided
server locations. There are no EDIT placeholders. Filenames/dependencies still
require server verification; missing correct source files truthfully skip.
Original sibling source dependencies are used in place; never combine differing
mixins by copying files into one directory.

Run these commands in Windows **CMD**, from this server report checkout:

```bat
conda activate nnunet5090
run_batch_feature_backfill.cmd --dry-run
run_batch_feature_backfill.cmd --allow-pending
```

The launcher defaults to --features-only --final-h2-only --use-current-h2-checkpoints,
using the existing v2/root/selection paths and a fresh final output root.
`--allow-pending` permits pending H2 alignment only; it does not waive checkpoint
identity or SHA256 proof. Omit it when pending H2 diagnostics are not wanted.
Use the already working server environment if its name differs. Local checks
used newconda; no dependency installation/upgrade is needed.

Equivalent complete command (CMD):

```bat
python batch_generate_result_reports.py --features-only --final-h2-only --use-current-h2-checkpoints --existing-summary "C:\lijialin\models3d\nnUNet\batch_reports_dataset501_v2\batch_summary.csv" --images-dir "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion\imagesTr" --labels-dir "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion\labelsTr" --selection-json "C:\lijialin\models3d\nnUNet\nnUNet_results\reports\comparison_fixed_cohort_v1\comparison_selection.json" --trainer-extension-map "batch_report_trainer_extensions.example.json" --output-dir "C:\lijialin\models3d\nnUNet\batch_reports_dataset501_v2_features_final_v1" --device cuda --allow-pending
```

This reads saved `status=saved/feature_failed` job sources and metrics from the
existing batch_summary.csv, and checks each original report's frozen selection
copy against the supplied manifest. Unrun/failed/skipped saved reports are not
regenerated (including the three genuinely unrun reports). Existing successfully
recorded feature outputs are skipped if their required files exist. No scan,
full-19-mask reread, clinical metric recomputation, metric-table/count export,
or saved-report rewrite occurs. Full population filenames and saved metric rows
remain checked by the reused generator; frozen hashes and geometry are checked
for the six displayed cases. Features necessarily read those selected DWI/GT/
prediction volumes; GT stays outside preprocessing and model forward.

Output must be a new sibling of the original batch root. Existing or overlapping
output paths fail; repeated actual runs need a fresh sibling name. New diagnostics
are at `<fresh-root>/<job>/feature_diagnostics/`. Original reports, metrics and
selection are read-only. The new CSV includes `existing_report` for traceability;
`feature_saved` means the generator produced its required outputs,
`feature_skipped` means unsupported/missing source/weights or already existing
features, and `feature_failed` means attempted diagnostics failed. Feature skips
are nonfatal; exit 1 means attempted feature failure, exit 2 invalid global input.
A dry-run is a metadata plan, not proof of source/checkpoint/feature compatibility,
and reads no checkpoints/models/voxels. It writes no output.

Added diagnostic identities: Foreground50, Foreground50DetailRefine,
GroupedTopK10, UPerNetEarlyStopping, TopK10EarlyStopping,
UPerNetNoStage7TopK10EarlyStopping with `nnUNetPlans__2d`;
PlainConvDepthTopK10EarlyStopping with `nnUNetPlansPlainConvDepth__2d_stage5`
or `2d_stage6`; and UPerNetSelectedStagesTopK10EarlyStopping with
`nnUNetPlansUPerNetStages_s2345__2d`. All names include the `nnUNetTrainer`
prefix. Added identities require a real mapped source; `--trainer-extension-dir`
can provide one default directory, with per-Trainer map entries taking precedence.
Original three adapters retain their strict local fallback. Ordinary full-report
mode still rejects added identities. Direct diagnostic use also accepts
`--features-only --trainer-extension-dir <exact-directory>`.

Added identities validate exact class file identity, checkpoint Trainer/config/
fold and conservatively the full saved plans against current plans (including
inherited parents, transpose and reader fields). The real nnU-Net build loader
constructs the network. Runtime encoder count/metadata and s2345 selection are
checked, then existing hooks validate actual BCHW stage outputs. Official backfill
skips architecture diagrams, so changed decoders do not need fabricated diagrams.
Feature inverse geometry and normalization are unchanged. Configurations lacking
one unique native 64x64 intermediate stage are truthfully unsupported.

Default H2 reporting retains the exact manifest absolute checkpoint path and
matching SHA256 guard. The new explicitly authorized
`--use-current-h2-checkpoints` option works only with
`--features-only --final-h2-only`. It passes `--diagnostic-current-checkpoint`
to the reused standalone generator. Direct invocation requires the same explicit
features-only/final flags and rejects prediction checkpoint confirmation claims.

Final H2 policy retains exactly best1000/latest1000 for the preprocessed bs4 run,
current best/latest for AdamW and LiteUPer, and prefers W128
`full_volume_predictions_best` over `best_prediction`. These are distinct exports,
never averaged. Screening and other numeric suffixes are excluded; the policy
never finds the largest epoch or ranks by Dice. Official completed jobs remain.
The local v2 metadata yields 12 official + 7 H2 retained records and 7 H2
exclusions; three unrun official jobs remain unrun. Existing reports are preserved.

Current checkpoint resolution uses only `checkpoint_best.pth` or
`checkpoint_latest.pth` for the retained slot, directly in the model root or its
`checkpoints` child. A matching existing manifest locator within these candidates
may disambiguate; otherwise there must be exactly one candidate. Missing/conflicting
candidates skip. No recursive search or arbitrary filename substitution occurs.
The standalone entry repeats this guard before checkpoint reads. Real checkpoint
identity/config/alignment checks remain; pending requires `--allow-pending` and
stays pending. The current file's SHA256 is captured before reads and checked after loading,
after feature capture, and immediately before publication. A detected change
aborts publication and cleans staging; no large checkpoint copy is made.

Current diagnostics persist actual path/SHA256, original report/export association,
unchanged historical manifest checkpoint metadata, UNKNOWN saved-mask provenance,
and both alignment states in `report.txt` and `diagnostic_checkpoint.json`.
Historical snapshot path/hash may differ; this is explicitly marked CURRENT
DIAGNOSTIC, never proof that current weights generated saved predictions.

`final_report_allowlist.json` and `final_report_allowlist.csv` contain the 19
retained original report records (CSV has `report_basename` for PPT filtering);
`final_report_excluded.csv` gives the seven excluded records and reasons. The
JSON separately records unrun jobs and hashes the source summary. Copies are
created in fresh backfill output; the checked-in local metadata exports were
created from the local v2 CSV without reading image/model data. No PPT or PPT
scripts are edited. Real server weights/features/geometry/rendering still require
server validation; no feature completion or formal clinical result is claimed.
