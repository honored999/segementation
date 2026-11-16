# HANDOFF_C — independent Dataset509 curriculum

COMPLETED: C implementation and local CPU engineering checks only (2026-11-16).
Real C metadata/distribution, server verification, GPU preflight, training and
full-volume evaluation are NOT RUN. No deployment, commit/push, agents, dependency
installation, notifications or canonical-memory changes.

Worktree: `E:/study/研一/work14-图像分割/segementation/.worktrees/dataset509-quality`.
Branch `codex/dataset509-quality`, HEAD/base `5472832bcd077c7e98244a8f7016b7c6ad997428`.
External read-only LesionGen contract: `c498118e6e3b3e4bb7a1c2380bb777c13f601e04`.
All task additions/edits remain uncommitted/untracked. Existing root user changes,
old Q HANDOFF and accepted evidence are retained; no server Q snapshot was touched.

## Files and actual behavior

New under `experiments/dataset509_quality/`: `curriculum.py`, `data_c.py`,
`trainer_c.py`, `focused_checks_c.py`, `c.json`, `run_layout.py`, and preserved
`SOURCE_HASHES_Q_ACCEPTED_20261116.json` (verbatim pre-edit hash inventory).
Changed: `trainer.py` (lazy C factory branch), `run.py` (C sidecar/config guard),
`evaluate_run.py` / `server_job.py` (mode-specific names; default remains old Q),
`focused_checks.py` (unsupported-mode assertion c -> q+c only), README and current
SOURCE_HASHES.json. Old `data.py`, `quality.py`, `loss.py`, `trainer514.py`,
API contract and three original example configs are byte-identical to accepted
Q hashes. Existing Q HANDOFF hash is unchanged. New inventory binds all deliverables.

C is a separate Baseline subclass; `_build_loss` and `validation_step` are inherited
unchanged. It uses original mixed Dice+TopK10/DS, never SourceLoss or quality loss
weights. No Q+C implementation. Only training Z probabilities change.

## Fixed initial sampling protocol

- Area is native positive pixels * audited spacing_x_mm * spacing_y_mm, in mm²,
  NOT 3D volume. Geometry/affine/unit provenance, raw/native refs, split, plans,
  prepared pool, training reference, quality sidecar and C policy hashes are bound.
  Missing/invalid spacing or Z mapping fails; no guessed units.
- Four groups: median native real positive area (small/large) x training-valid
  positive contrast 25th percentile (low/non-low). Negative and unknown-contrast
  real layers remain separate categories. Training100 only; no validation pixels,
  loss, teacher, error rate or Q evaluation outcome sets groups/parameters.
- Group gain `min(2,sqrt(max_group_count/max(group_count,1)))`. Trusted generated
  candidate requires existing contrast-proxy quality >= .5 and known group;
  the proxy is NOT established quality detection. Difficult means low contrast
  OR group gain > 1. Generated weight = gain * (alpha if difficult else 1).
- `alpha=clip((epoch-20)/80,0,1)`, source synthetic mass `.10+.15*alpha`, conditional
  on nonempty candidates in the drawn case. Missing/zero synthetic mass safely
  falls back to real and logs reason; no cases removed or new images generated.
  Group-relative gains survive within-source normalization. No synthetic group
  support is manufactured when absent.
- Real random slots retain uniform real Z including negatives; real foreground
  slots use actual foreground-mass Z weights (uniform-real fallback if empty).
  Selected-plane actual foreground is rebased before ordinary upstream XY bbox,
  crop and augmentations. This differs explicitly from baseline stored foreground
  coordinate subsampling. Metadata never becomes image/label channels.
- workers=0 only, explicitly rejected otherwise. No background prefetch queue;
  current_epoch updates actual loader every epoch and stale batches fail. No LR,
  optimizer or epoch reset. Anonymous per-epoch source/group probabilities, realized
  exposures and fallback counts go to curriculum_exposure.jsonl.
- Confirmed historical config: seed12345, 2D/fold0, original508 weights only,
  batch12, 250 train/50 validation steps, max1000/min0 epochs, patience100,
  min_delta=.0001. All phases are within this budget; early stopping means actual
  updates can differ from prior runs, so this is not an equal-update causal ablation.

## Verification actually run

`D:/Anaconda/envs/newconda/python.exe -B experiments/dataset509_quality/focused_checks_c.py --lesiongen-repo E:/study/研一/work14-图像分割/CG-SAMR-reference/LesionGen`
exited 0 and printed C_ENGINEERING_CHECKS_PASS.

Covered: explicit mm² vs pixel units and invalid geometry; training-only groups;
phase/gain/normalization boundaries; 40,000 small categorical draws showing rare
synthetic group gain survives normalization; no-trusted/phase-wait/empty-foreground
fallbacks; complete 100-case/1593+478 small native fixture sidecar build; split
leakage/hash/no-overwrite rejection; official nnU-Net Dataset+Loader Z/source/group
and foreground consistency; actual crop/DA/deep-supervision batch metadata; live
SingleThreadedAugmenter epoch update and workers rejection; real C Trainer CPU
construction, inherited original loss finite backward, ordinary LR hook called
once at epoch100 without reset; old mode factory/layout identity and path guards.

Actual fixture synthetic exposure changed from .0933333 (start) to .2533333
(late), over 300 actual loader samples per stage. This is synthetic engineering
evidence, not real-cohort distribution or model training. No network initialize,
optimizer update or full Q-suite rerun. Unchanged Q evidence was reused.
AST/JSON, all authored text whitespace/conflict checks and git diff --check passed;
new files are untracked, so explicit file scans also cover them.

## Main-agent commands / next

Deploy into NEW code/output locations; never replace the existing server Q snapshot.
Fill c.local.json from c.json with existing pins. Build sidecar with that same
policy (curriculum_sidecar placeholder is not read during construction):

```
python curriculum.py --package <existing-native-package> --base <existing-prepared-514-base> --plans <existing-plans.json> --quality-sidecar <existing-quality.private.json> --config c.local.json --output <new-C-metadata-dir>/curriculum.private.json
python focused_checks_c.py --lesiongen-repo <pinned-LesionGen-repo>
python run.py --config c.local.json --stage preflight --gpu-uuid GPU-<available-uuid> --output <new-C-preflight-root>
python run.py --config c.local.json --stage train --gpu-uuid GPU-<available-uuid> --output <new-C-training-root>
```

Inspect real sidecar audit for thresholds/area units/frequencies/missing groups,
conditional source/group probabilities and fallback counts; do not claim proxy
validity from successful metadata construction. Reuse established patient-split
proof, check host/GPU resources and pick a currently available UUID. Main agent
owns acceptance and one isolated real GPU preflight before formal execution.

For existing controller/evaluation chain: NEW job_spec `mode: c`, filled config
`c.formal.json`; default subpaths `c_training` / `c_metrics`. Optional config_name,
training_name, metrics_name must be distinct leaf names. Populate source_files
from the NEW snapshot including C modules and run_layout; no old Q paths implied.
`python server_job.py --run-root <new-C-controller-root>` runs train then evaluate;
`python evaluate_run.py --run-root <new-C-controller-root>` evaluates completed
training only. Neither was executed here. Full original-space28 inference and
historical paper F0.5-equivalent seven-metric/supplemental semantics are unchanged.
