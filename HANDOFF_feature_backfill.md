# HANDOFF — final v2 feature backfill (current diagnostic weights)

Worktree: `E:\study\研一\work14-图像分割\segementation\.worktrees\report-fixed-baseline-cohort`
Branch `codex/nnunet-result-report`; HEAD remains
`e115f0e34331a2a8a9059396c1cf0a619fd43c7b`.
Prior bounded implementation remains uncommitted. User explicitly authorized
this direct continuation. No agents, commit/push, main-checkout/canonical-memory
writes, PPT/script edits, or real image/model/checkpoint reads were performed.
Old report directories are preserved; only presentation/backfill selection changes.

Production files:
- `batch_generate_result_reports.py`: existing-summary features-only replay;
  --final-h2-only exact final export filter; --use-current-h2-checkpoints restricted
  to final features-only mode; per-Trainer map; final allowlist exports.
- `generate_nnunet_result_report.py`: previously reviewed encoder diagnostics,
  plus explicit standalone --diagnostic-current-checkpoint / --final-h2-only and
  original-report association CLI parameters. Existing inverse geometry and
  normalization remain unchanged.
- `standalone_h2former_report.py`: exact final H2 run/export policy and bounded
  current checkpoint slot resolver; direct-entry mode guards before checkpoint
  reads. Default exact historical manifest path+SHA guard remains unchanged.
  Existing strict model/config/alignment checks remain. Current diagnostic path/
  SHA, UNKNOWN saved-mask provenance, original report/export, unchanged historical
  checkpoint declaration and alignment states persist in report.txt and
  diagnostic_checkpoint.json. Hash captured before checkpoint reads; stability checked after model loading,
  after feature capture, and immediately before publication. Mutation aborts
  publication and staging cleanup remains active. No large checkpoint copy.

Other files:
- `tests/test_report_feature_backfill.py`: synthetic metadata/filter/guard tests.
- `README_batch_result_reports.md`: final/current diagnostic mode and boundaries.
- `batch_report_trainer_extensions.example.json`: all eight exact user-provided
  SERVER source directories filled, normal Windows separators, no EDIT placeholders.
- `run_batch_feature_backfill.cmd`: final mode flags and fresh final output root.
- `final_report_allowlist.json`, `final_report_allowlist.csv`,
  `final_report_excluded.csv`: actual local v2 summary metadata exports for PPT
  filtering. 19 retained (12 official + 7 H2), seven excluded with explicit reasons,
  three unrun recorded separately. No metric averages, model/data reads or PPT edits.
- this HANDOFF.

Final selection policy:
- H2Former_fold0_bs4_preprocessed: best1000 and latest1000, exact suffixes only.
- H2Former_fold0_bs4_adamw: current best/latest exports.
- H2Former_LiteUPerNet_AdamW_EarlyStop_bs4/fold_0: best/latest.
- H2Former_UPerNet_W128_PPM1236/fold_0: full_volume_predictions_best preferred
  over distinct best_prediction export, never averaged.
- Exclude best400/best600/best800/latest600/latest800, screening and W128 alternate
  best_prediction: seven exclusions. No numeric-largest or Dice-based selection.
- Three unrun official reports stay unrun; twelve completed official reports retained.

Current checkpoint resolution:
Exact checkpoint_best.pth or checkpoint_latest.pth in MODEL ROOT or its
checkpoints child only. Matching existing manifest locator may disambiguate within
these bounded candidates; otherwise exactly one existing candidate is required.
Missing/conflicting candidates truthfully skip. No recursive search or arbitrary
filename pairing. Current diagnostic mode explicitly waives historical weight
path/hash equality only, never rewrites manifest SHA or asserts that these weights
produced saved masks. Pending alignment requires --allow-pending and stays pending.
Source/config/checkpoint compatibility and actual feature rendering need SERVER RUN.

Actual local v2 metadata source:
`E:\study\研一\work14-图像分割\nnUNet_results\reports\batch_reports_dataset501_v2\batch_summary.csv`
Allowlist preserves original SERVER report paths, adds report_basename in retained
CSV for local PPT filtering, and stores source CSV SHA256. Original local/remote
reports and source CSV unchanged. Backfill creates copies of allowlists in its
fresh output root. Excluded and unrun entries have no diagnostic jobs.

Validation:
```powershell
D:\Anaconda\envs\newconda\python.exe -m pytest -q tests/test_report_feature_backfill.py -p no:cacheprovider
```
Final current result: **60 passed in 0.44s** (includes new mutation/stability guards). Earlier current iteration: 57 passed
in 0.47s. Prior iteration baseline evidence: three strict adapter/reader tests
passed; not rerun as a broad suite here. No full suite, real checkpoint/model
loading, actual image rendering, formal training or evaluation.
Resource preflight before final run: CPU sampled 17/18%, RAM 54.3%, GPU 11%,
VRAM 1001/6144 MiB. Lightweight serial metadata scope; no >=80% guard block.
newconda has no psutil, so existing Windows CIM used without installation.
Actual metadata allowlist generation confirmed retained=19, official=12, H2=7,
excluded=7, unrun=3. git diff --check passed.

SERVER CMD from updated report checkout, existing working environment:
```bat
conda activate nnunet5090
run_batch_feature_backfill.cmd --dry-run
run_batch_feature_backfill.cmd --allow-pending
```

Complete CMD:
```bat
python batch_generate_result_reports.py --features-only --final-h2-only --use-current-h2-checkpoints --existing-summary "C:\lijialin\models3d\nnUNet\batch_reports_dataset501_v2\batch_summary.csv" --images-dir "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion\imagesTr" --labels-dir "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion\labelsTr" --selection-json "C:\lijialin\models3d\nnUNet\nnUNet_results\reports\comparison_fixed_cohort_v1\comparison_selection.json" --trainer-extension-map "batch_report_trainer_extensions.example.json" --output-dir "C:\lijialin\models3d\nnUNet\batch_reports_dataset501_v2_features_final_v1" --device cuda --allow-pending
```
For default strict historical H2 mode omit --use-current-h2-checkpoints;
--final-h2-only can still filter final jobs. Every actual run requires a fresh
sibling output root. Skips are nonfatal; attempted generation failures return 1,
invalid global input returns 2. A dry-run does not hash/read checkpoint contents,
prove checkpoint availability, generate images or write outputs.

Final production SHA256 (independent Level 3 final PASS; all three exact hashes
confirmed twice and stable, no blockers):
```text
generate_nnunet_result_report.py 5C968453C547D3A224679CCFC44877C8A8FFAE6E4664552461926CA1B4B5F704
batch_generate_result_reports.py 4436A0C1E336BB38ABBD5D73037F2B5AD456EB4617265EDDB91AFD4279897B92
standalone_h2former_report.py 040ECF0240B8D470C12F7A0C7777069D7132478135A33D2726EB30B227AC8BDA
```

Residual limits: all eight paths are user-provided but source filenames/dependencies
and revisions still need server verification; current weights do not establish
historical mask provenance; conservative full-plans identity and unique native64x64
encoder stage requirements remain. No claim that 19 features were generated.
Server run pending; independent Level 3 final PASS confirmed for the exact
production snapshot below. All eight source map paths are filled.
No actual remaining feature failures have been measured locally: server generation
has not run, and 19 retained records do not mean 19 feature jobs succeeded. Final worktree has intended
uncommitted source/docs/tests/maps/allowlist artifacts only; no commit/push.

PPT status supplied by parent: 68 slides/19 records, all 13 changed pages visually
checked and 55 unchanged pages exact PNG hashes matched; retained IDs
1,3,4,5,6,8,9,10,11,12,13,14,16,17,18,22,26,27,29.
PPT and its claimed 55 unchanged-page hashes were not independently inspected
or modified by this implementation worker. Allowlist remains unchanged.
