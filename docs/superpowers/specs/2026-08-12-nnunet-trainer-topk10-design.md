# nnU-Net Trainer TopK10 loss experiment

## Goal

Run one isolated official-nnU-Net v2 fold-0 experiment that tests whether
optimizing only the hardest 10 percent of cross-entropy pixels improves
stroke-lesion segmentation.

## Scope and boundary

The experiment runs through the external `nnUNetTrainerTopK10` subclass in the
verified server runtime `nnunetv2==2.8.1`. It leaves the standard Dice term,
foreground oversampling (0.33), plans, 2D network, batch size 12, augmentation,
optimizer, schedule, epochs, split, checkpoint policy and inference unchanged.

The sole behavior change is to use the official `DC_and_topk_loss` instead of
the base Trainer's Dice-plus-ordinary-CE loss, with `k=10`. It is a model
optimization variant, not an unmodified nnU-Net reproduction, and it must use
its own Trainer-named result directory.

## Evaluation and decision

Run only fold 0 first. Do not use training-log pseudo Dice as its result. After
the 19 validation cases receive complete-volume prediction, score foreground
class 1 after argmax per reconstructed 3D patient: both empty = 1, one empty =
0, otherwise `2TP/(2TP+FP+FN)`, and report the equal-case macro mean.

Compare to standalone fold-0 baseline 0.73225151385; advance only at or above
0.74225151385. Do not run five folds before this screen passes. TopK10 and
Foreground50 must remain separate experiments so their effects are attributable.

## Test-first verification

Create a no-pytest direct Python contract script before the Trainer module.
It must fail because `nnUNetTrainerTopK10` is unavailable. After minimal
implementation, it must verify inheritance, `TOPK_PERCENT == 10`, a real
constructed Trainer retains `oversample_foreground_percent == 0.33`, and its
`_build_loss()` returns the official `DC_and_topk_loss` with TopK CE `k == 10`.
Use a separate unpreloaded external-resolver script to prove discovery through
`nnUNet_extTrainer`.

## Failure handling

Use the installed 2.8.1 source as the interface authority. If the loss builder
or its deep-supervision wrapper differs, inspect it and make only the minimal
override needed to preserve base Trainer behavior except for loss type and
`k`. Do not install pytest, modify site-packages, or silently fall back to
ordinary CE.
