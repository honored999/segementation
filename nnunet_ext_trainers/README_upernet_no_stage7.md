# Dataset501 stage7 removal ablation

`nnUNetTrainerUPerNetNoStage7TopK10EarlyStopping` is a custom external Trainer
running inside the official `nnunetv2==2.8.1` framework. It inherits
`nnUNetTrainerUPerNetTopK10EarlyStopping`, including the official
`nnUNetTrainerNoDeepSupervision` lineage, and overrides only the network builder.
It preserves the current `plans_manager` / `configuration_manager` API used by
official training initialization and inference reconstruction.

The supported source configuration is the existing Dataset501 2D, 512×512,
eight-stage `PlainConvUNet` plan with strides `(1,1)` followed by seven `(2,2)`
entries. Incompatible architecture classes, stage metadata, stride geometry,
patch sizes, and deep supervision are rejected. Plans and configuration objects
are preserved; no planning or preprocessing is rerun.

The builder constructs the official plans network, retains its encoder layers
0–6 exactly, and removes stage7 from the returned model and encoder metadata.
The temporary official decoder and removed stage are not registered in the
returned model. Encoder forward and feature-map accounting traverse seven
stages. UPerNet selects `(1,3,5,6)`, giving native maps `256×256`, `64×64`,
`16×16`, `8×8` for a 512×512 input; baseline `(1,3,5,7)` ends at `4×4`.
PPM `(1,2,4)` and FPN width 128 remain inherited from the existing decoder.

Dice/TopK10, early stopping, optimizer, scheduler, batch size, augmentation,
patient splits, preprocessing, checkpoint selection, and inference/TTA policy
are inherited unchanged. Use the baseline's device and seed policy.

## Windows cmd.exe recipes (not executed by this implementation task)

Use the same existing Dataset501 raw/preprocessed roots and fixed
`splits_final.json` as the baseline. From this checkout, activate the verified
local environment and register the external source directory:

```bat
call D:\Anaconda\Scripts\activate.bat newconda
cd /d E:\study\研一\work14-图像分割\segementation\.worktrees\upernet-no-stage7
set "nnUNet_extTrainer=%CD%\nnunet_ext_trainers"
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerUPerNetNoStage7TopK10EarlyStopping -p nnUNetPlans
```

On the training host, use its corresponding checkout/environment paths. Keep
the baseline's configured `nnUNet_results` root. The new Trainer identity creates
the independent output path below that root:

```text
Dataset501_StrokeLesion\nnUNetTrainerUPerNetNoStage7TopK10EarlyStopping__nnUNetPlans__2d\fold_0
```

Require this experiment directory to be fresh for the initial ablation. Train
from scratch: do not resume another Trainer's checkpoint or supply pretrained
weights from another Trainer. Continue an interrupted run only with this same
Trainer, source, plans, fold, and output directory:

```bat
set "nnUNet_extTrainer=%CD%\nnunet_ext_trainers"
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerUPerNetNoStage7TopK10EarlyStopping -p nnUNetPlans --c
```

The reported final baseline uses `checkpoint_final.pth`. Match that policy for
the ablation with official full-volume validation of the final checkpoint:

```bat
set "nnUNet_extTrainer=%CD%\nnunet_ext_trainers"
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerUPerNetNoStage7TopK10EarlyStopping -p nnUNetPlans --val
```

Here `--val` selects the final checkpoint by default; do not add `--val_best`
for this comparison. Keep this external Trainer available for validation,
continuation, and inference. Local CPU synthetic tests establish engineering
behavior only. Real runtime validation and formal performance remain server
pending. Fold0 is a development screen; formal conclusions require the fixed
five-fold full-volume OOF evaluation.
