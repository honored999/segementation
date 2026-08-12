# nnU-Net external Trainer experiments

`nnUNetTrainerForeground50` is an official nnU-Net v2 Trainer variant for the
fold-0 foreground-oversampling screen. It changes exactly one post-construction
instance attribute:

```python
self.oversample_foreground_percent = 0.50
```

It does not modify the plans, network, batch size, augmentation, loss,
optimizer, scheduler, epochs, data split, checkpoint selection, or inference.
It is an optimization variant, not an unmodified nnU-Net reproduction.

## Runtime requirements

The checked server runtime is `nnunetv2==2.8.1` in the `nnunet5090` Conda
environment. Keep this source directory available at the same path for initial
training, `--c` resume, `--val`, `--val_best`, and inference. Do not copy the
file into `site-packages`; use `nnUNet_extTrainer` instead.

From the worktree root in Windows `cmd.exe`:

```bat
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
python nnunet_ext_trainers\tests\test_trainer_foreground50.py
python nnunet_ext_trainers\tests\verify_external_discovery.py
python nnunet_ext_trainers\tests\verify_effective_oversampling.py "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json" "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion\dataset.json"
```

The last command must end with:

```text
EFFECTIVE_OVERSAMPLING_OK value=0.50
```

## Fold-0 screen

First confirm the official nnU-Net roots already point at the same Dataset501
raw/preprocessed/results locations used by the baseline. Do not alter the plans
or run preprocessing again. The following creates a new result directory; it
does not reuse a default-Trainer or standalone checkpoint:

```bat
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerForeground50 -p nnUNetPlans
```

The expected output directory is below the configured `nnUNet_results` root:

```text
Dataset501_StrokeLesion\nnUNetTrainerForeground50__nnUNetPlans__2d\fold_0
```

On an interrupted run, resume only the above custom Trainer run with exactly
the same extension source and path:

```bat
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerForeground50 -p nnUNetPlans --c
```

Never resume a default `nnUNetTrainer` checkpoint with this Trainer, and never
run this Trainer in the default Trainer's result directory.

## Full-volume decision rule

After training finishes, use the custom Trainer with the same external-path
setting for full-volume fold-0 validation:

```bat
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerForeground50 -p nnUNetPlans --val
```

Do not promote the strategy from training-log pseudo Dice. Evaluate the saved
argmax class-1 predictions for all 19 fold-0 cases as full 3D volumes: both
empty = 1, one empty = 0, otherwise `2TP / (2TP + FP + FN)`. Use the equal-case
macro mean. Compare it with the standalone fold-0 baseline `0.73225151385`;
only a score at least `0.74225151385` advances to a five-fold variant run.

## TopK10 fold-0 screen

`nnUNetTrainerTopK10` is a separate experiment. It retains the default
foreground oversampling value `0.33` and replaces the standard non-region loss
with the official `DC_and_topk_loss` using `k=10`. Do not combine it with
`nnUNetTrainerForeground50` in this first screening round.

To run it on physical GPU 1 from Windows `cmd.exe` while GPU 0 runs another
experiment:

```bat
set CUDA_VISIBLE_DEVICES=1
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerTopK10 -p nnUNetPlans
```

The log will say `Using device: cuda:0`; this is expected because physical GPU
1 is the only CUDA device visible to that process. Its independent result path
is:

```text
Dataset501_StrokeLesion\nnUNetTrainerTopK10__nnUNetPlans__2d\fold_0
```

To resume this experiment, keep `CUDA_VISIBLE_DEVICES=1`, the same external
Trainer source, and its own output directory:

```bat
set CUDA_VISIBLE_DEVICES=1
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerTopK10 -p nnUNetPlans --c
```

After it finishes, clear or change `CUDA_VISIBLE_DEVICES` as needed and run
full-volume validation with the same custom Trainer:

```bat
set CUDA_VISIBLE_DEVICES=1
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerTopK10 -p nnUNetPlans --val
```

Assess the same 19 full reconstructed 3D validation volumes using the stated
foreground case-macro Dice rule before comparing with the baseline.

## TopK20 fold-0 screen

`nnUNetTrainerTopK20` inherits the already validated TopK10 experiment and
changes only `TOPK_PERCENT` from 10 to 20. Foreground oversampling remains at
the default `0.33`; plans, network, batch size, augmentation, optimizer,
scheduler, epochs, deep supervision, split and inference remain unchanged.

Run it on physical GPU 1 from Windows `cmd.exe`:

```bat
cd /d C:\lijialin\segementation\.worktrees\standalone-nnunet2d
set CUDA_VISIBLE_DEVICES=1
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerTopK20 -p nnUNetPlans
```

Its independent result directory is:

```text
Dataset501_StrokeLesion\nnUNetTrainerTopK20__nnUNetPlans__2d\fold_0
```

Resume only this run with the same source and environment:

```bat
cd /d C:\lijialin\segementation\.worktrees\standalone-nnunet2d
set CUDA_VISIBLE_DEVICES=1
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerTopK20 -p nnUNetPlans --c
```

After training, run full-volume validation with the same Trainer:

```bat
cd /d C:\lijialin\segementation\.worktrees\standalone-nnunet2d
set CUDA_VISIBLE_DEVICES=1
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerTopK20 -p nnUNetPlans --val
```
