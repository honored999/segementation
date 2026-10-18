# HD-BET + GenSeg: G/D pretraining -> joint engineering trial

Status: commands reconciled with implemented CLI; independent review and publication tracked in the PRETRAIN review receipt.

This trial pretrains G/D on inner61, then initializes a fresh author U-Net and learns architecture using inner15. Dataset501 remains95; fixed outer76/19 and derived61/15 memberships stay unchanged. Outside19 remains existing development validation, and is not read for fitting/model selection. All real inputs use existing3mm targets. The old76-trained generator is provenance only. Do not use the completed cold-start joint_trial01 as pretrained input.

## Code synchronization

The current agreed workflow is local implementation/tests/independent review -> local commit/push -> data computer git pull --ff-only. Historical delivery documents mentioning server-first ZIP installation are superseded for this change.

On the data computer, Anaconda Prompt / CMD:

```bat
cd /d C:\lijialin\segementation\.worktrees\genseg-reference
git status --short --branch
git pull --ff-only
git rev-parse --short HEAD
```

If unrelated changes block pull, preserve them; do not reset/stash/overwrite them automatically. Use the existing working CUDA environment (previous commands used nnunet5090). Local engineering tests use newconda; do not replace a working data-computer environment merely to match its name.

```bat
conda activate nnunet5090
set "REPO=C:\lijialin\segementation\.worktrees\genseg-reference"
set "nnUNet_raw=C:\lijialin\models3d\nnUNet\nnUNet_raw"
set "nnUNet_preprocessed=C:\lijialin\models3d\nnUNet\nnUNet_preprocessed"
set "SOURCE=%nnUNet_preprocessed%\Dataset501_StrokeLesion\nnUNetPlans_2d"
set "BASELINE=C:\lijialin\genseg30_derived_trial01\generator_5000steps_seed20261014_attempt02"
set "TARGETS=C:\lijialin\HD-BET_outputs\genseg3mm_trial01\targets"
set "TARGET_SHA=23d25e2b2a6d7c7ca2a270690a9d4d35116ff4f834458b48a9049ff889894a56"
set "PRETRAIN=C:\lijialin\HD-BET_outputs\genseg_gd_pretrain_trial01"
set "JOINT=C:\lijialin\HD-BET_outputs\genseg_joint_trial02"
set "PYTHONPATH=%REPO%;%PYTHONPATH%"
cd /d "%REPO%"
```

Use new suffixes if either new output directory already exists, including failed/incomplete attempts. Never delete or overwrite old weights/targets/results. This trial's1+1 budget verifies the stage handoff; it cannot establish useful generated images or segmentation benefit.

## 1. Read-only source/target check

```bat
python -m experiments.genseg_joint.cli check --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%PRETRAIN%"
```

Expected: fixed61/15/19 roles, immutable target/source provenance, no neural model/CUDA construction and no patient image/lesion array reads.

## 2. One G/D pretraining update

```bat
python -m experiments.genseg_joint.cli pretrain --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%PRETRAIN%" --gpu 1 --steps 1 --l1-weight 100
```

Only inner61 G/D weights update; architecture tensors stay fixed and U-Net does not train. Use frozen BN, native planes/padding, existing SGD rates/loss masking. Resource protections stay active for CPU/RAM and selected physical GPU by UUID. An idle GPU1 is not blocked by unrelated GPU0 use. A resource refusal is not permission to disable guards or terminate other tasks.

Inspect the completion receipt, then extract the exact GAN checkpoint hash in CMD:

```bat
type "%PRETRAIN%\complete.json"
for /f %H in ('powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $env:PRETRAIN gan.pt)).Hash.ToLowerInvariant()"') do set "PRETRAIN_SHA=%H"
echo %PRETRAIN_SHA%
```

The for variable is single %H at the interactive CMD prompt; use double %%H if copying into a .bat file. Only a successful top-level complete=true phase receipt with matching checkpoint hash qualifies. Keep failed/incomplete directories; do not create completion markers manually or accept a freshly calculated hash as a reason to ignore a mismatched receipt.

## 3. One joint update from completed G/D pretraining

Audit the completed-pretrain handoff without loading neural weights first:

```bat
python -m experiments.genseg_joint.cli check --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%JOINT%" --pretrained-run "%PRETRAIN%" --pretrained-sha256 "%PRETRAIN_SHA%"
```

```bat
python -m experiments.genseg_joint.cli train --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%JOINT%" --gpu 1 --steps 1 --pretrain-steps 0 --l1-weight 100 --pretrained-run "%PRETRAIN%" --pretrained-sha256 "%PRETRAIN_SHA%"
```

Here --pretrain-steps0 remains the legacy inline-pretraining option: GAN pretraining already completed in its separate directory. The explicit pretrained-run initializes G/D plus fixed architecture; U-Net and outer-Adam state start fresh. Joint/data RNG initialization is explicitly recorded; this is phase handoff, not interrupted-run resume. The loader must reject incomplete/wrong-phase/wrong-hash/wrong-cohort or incompatible configuration before mutating models.

```bat
type "%JOINT%\complete.json"
type "%JOINT%\steps.jsonl"
```

Return the terminal result and both complete.json receipts (or failed.json and traceback). Check finite G/D training evidence, unchanged alpha during pretraining, exact pretraining checkpoint identity in the joint protocol, and finite nonzero architecture gradient/update during joint training. Local CPU tests do not establish target-PC CUDA handoff success. Do not interpret success as image QA, long-run training, full-volume evaluation or five-fold OOF evidence.

The optional check command audits default G/D rates and L1=100. This first1+1 trial uses those defaults. A future custom-rate experiment must match the pretrain and joint settings; do not interpret a default-only check refusal as permission to bypass identity validation.

## Longer experiments

No long budget is selected by this delivery. Decide explicit G/D pretraining and joint update counts after short phase-handoff/resource checks and train-only generated-image QA. Save new runs separately. Old76-pretrained and new61-pretrained results are not matched efficacy controls. A later performance comparison needs identical data/initialization/budgets and full-volume development evaluation, with the historical19 limitations stated.
