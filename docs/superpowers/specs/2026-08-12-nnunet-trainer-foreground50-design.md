# nnU-Net Trainer foreground sampling 50% experiment

## Goal

Run one isolated official-nnU-Net v2 fold-0 experiment that tests whether
increasing foreground-patch oversampling improves stroke-lesion segmentation.
The experiment is an optimization variant, not a reproduction of the unmodified
nnU-Net trainer or of the standalone baseline.

## Scope and boundary

The experiment runs in the server `nnunet5090` environment through a custom
`nnUNetTrainer` subclass. The subclass changes only
`oversample_foreground_percent` from the baseline `0.33` to `0.50`.

It must retain the Dataset501 fixed split, 2D configuration, plans, network,
batch size 12, augmentation, loss, optimizer, scheduler, epoch count, seed
policy, checkpoint selection, inference policy, and full-volume validation.
No standalone training or inference code is changed. The existing standalone
`official_aligned` artifacts remain the comparison baseline.

## Design

Create a uniquely named trainer class, `nnUNetTrainerForeground50`, inheriting
from the installed official `nnUNetTrainer`. Its sole behavior is to set
`oversample_foreground_percent = 0.50` during initialization. Keep the trainer
module in a separately versioned experiment extension directory so the official
installation is not edited. Configure `nnUNet_extTrainer` on Windows to make
the class importable for training, full-volume validation, inference, and
checkpoint resume.

Use a separate results output directory named for the trainer and fold, never
the original official or standalone output directory. Resume only with the
same trainer source, plans, configuration, fold, output directory, total epoch
count, and performance settings; otherwise start a fresh experiment directory.

## Evaluation and decision

After training, run full-volume inference for all 19 fold-0 validation cases.
Compute foreground class-1 Dice after argmax, reassembled per complete 3D
patient volume. Both-empty cases score 1, one-empty cases score 0, and the
reported value is the equal-case macro mean.

The screening result must name its field and cohort explicitly. It must not use
the online patch/global validation Dice from training logs as the final score.
Compare the result with the standalone fold-0 baseline 0.73225151385. Promote
the strategy only if it is at least 0.74225151385 (absolute improvement at
least +0.0100); otherwise stop or test the separately isolated 0.40 variant.
A successful fold-0 screen is only a variant-screening result. Five separately
trained folds and a 95-case OOF aggregate are required before claiming a
five-fold variant result.

## Test-first checks

Before the trainer is used, add a focused test that imports the extension and
asserts the trainer inherits from `nnUNetTrainer` and exposes exactly `0.50`.
Run it first and record the expected missing-module failure. Implement the
minimal trainer class, rerun the focused test, and then run the relevant
official-nnU-Net discovery/import check. Do not alter production code before
the failing test is observed.

## Failure handling

If the installed official version has a different initialization path or does
not read `oversample_foreground_percent` as an instance attribute, inspect that
version and adjust only the minimal override needed to make the same setting
effective. Record the installed nnU-Net version and the resolved trainer path
alongside the experiment output. If the custom Trainer is not importable during
validation or resume, stop rather than substituting the default Trainer.
