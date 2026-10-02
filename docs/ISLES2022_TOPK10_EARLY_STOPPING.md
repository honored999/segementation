# Dataset508 fold0: TopK10 + early stopping

## Scope and acceptance

Prepared for nnunetv2 2.8.1, Windows CMD, server environment `nnunet5090`.
User reports Dataset508_ISLES2022DWI conversion (250 cases) and 2d preprocessing
completed with exit code 0; nnUNetPlansFrom501 uses batch_size=12 and
patch_size=[512,512]. These are user-reported server facts, not local validation.
Only tiny CPU synthetic logic and isolated class discovery are validated locally.
Independent read-only Level3 review must pass before server integration or fold0.

## Composition and stopping

MRO: nnUNetTrainerTopK10EarlyStopping -> nnUNetTrainerEarlyStopping ->
nnUNetTrainerTopK10 -> official nnUNetTrainer -> object.
The new class overrides no methods. It inherits TopK10 `_build_loss` (Dice plus
hardest-10-percent CE, 1:1 weights, original deep supervision), official network,
optimizer/scheduler, sampling and augmentation, and existing early-stop control.
Maximum 1000 epochs; no patience accumulation through 300 completed epochs;
thereafter stop after 100 consecutive epochs without EMA foreground Dice
improvement of at least 1e-4. Sufficient improvement resets patience.
`checkpoint_best` still uses the official strict improvement rule independently
of the stopping min_delta. Checkpoints persist best monitored EMA, patience count
and triggered flag. A triggered checkpoint still invokes train-start/train-end
cleanup when resumed but enters no training epoch or optimizer step.
Full startup/cleanup integration is server-only validation.

## Planned server commands (not executed here)

Deploy the new Trainer and its existing sibling Trainer modules together in the
server checkout. The commands below use the current server checkout. Retain the
server's existing nnUNet_raw, nnUNet_preprocessed and nnUNet_results settings.

```bat
conda activate nnunet5090
set "PROJECT_DIR=C:\lijialin\segementation\.worktrees\isles2022-dwi-conversion"
set "nnUNet_extTrainer=%PROJECT_DIR%\nnunet_ext_trainers"
python -c "from importlib.metadata import version; print(version('nnunetv2')); assert version('nnunetv2') == '2.8.1'"
cd /d "%TEMP%"
python -B "%PROJECT_DIR%\nnunet_ext_trainers\tests\verify_external_topk10_early_stopping_discovery.py"
```

Before first launch, inspect Dataset508's own split and verify patient grouping,
no overlap, and actual fold0 counts (expected approximately 200 train / 50 val).
Never copy, regenerate or modify Dataset501's fixed splits. If Dataset508's
`splits_final.json` does not yet exist, resolve its own split/preflight on server
before accepting the first run; do not substitute Dataset501's split.
Run resource preflight before discovery/integration/training. Verify that the
new result directory is unused before first launch; archive no existing outputs.

First launch, using Dataset508's own fold0 and no pretrained initialization:

```bat
nnUNetv2_train 508 2d 0 -tr nnUNetTrainerTopK10EarlyStopping -p nnUNetPlansFrom501
```

Continue that same experiment with its own checkpoint (official 2.8.1 flag is
`--c`, not `-c`):

```bat
nnUNetv2_train 508 2d 0 -tr nnUNetTrainerTopK10EarlyStopping -p nnUNetPlansFrom501 --c
```

Expected independent result directory:

```text
%nnUNet_results%\Dataset508_ISLES2022DWI\nnUNetTrainerTopK10EarlyStopping__nnUNetPlansFrom501__2d\fold_0
```

Do not reuse the original TopK10 result directory. Plans reuse does not imply
checkpoint reuse. Later Dataset501 pretrained initialization is a separate
experiment requiring an explicit compatible checkpoint and protocol; it is not
`--c` continuation and is not part of this first Dataset508 trial.

## Local evidence and deferred work

New tests exercise real TopK10 loss on 1x2x4x5 CPU logits, weighted multi-output
loss, real MetaLogger EMA, warmup/patience/reset, official best selection, tiny
on-disk checkpoint serialization and terminal resume control with fail-fast
training sentinels. Base initialization dispatch is checked with a substituted
base initializer; this is not real production Trainer initialization. Optimizer
state is a tiny serialized fixture, not an optimizer training measurement.
Fresh subprocesses use the official resolver without pre-importing the new class
from both repository and unrelated ASCII temporary working directories.

Deferred to server after independent review: actual production network/Trainer
initialization; full deep-supervision/model integration; optimizer and scheduler
integration; startup/cleanup on resume; actual split/resource/batch-size checks;
512x512 training and real-case validation; actual fold0 run and checkpoints.
No local GPU calculation, benchmark, large checkpoint, real data, production
network or full repository suite is permitted. Fold0 online Dice is development
evidence, not formal full-volume five-fold OOF performance.
