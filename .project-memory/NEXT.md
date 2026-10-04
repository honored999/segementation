# Next

Scope: branch-local codex/upernet-no-stage7; not canonical master memory.

## Current focus

- Configurable-depth ordinary PlainConvUNet Trainer implemented; independent exact-snapshot static review PASS. Prior NoStage7 UPerNet implementation remains preserved at f1b9ca8.
- Per user instruction, no new tests, imports, model construction or runtime validation were executed for the new Trainer. Earlier NoStage7 tests do not validate this new implementation.

## Next actions

1. Current source/README have independent static PASS and user-authorized branch commit/push. Update the matching server worktree from origin/codex/upernet-no-stage7 after delivery.
2. Deploy the new Trainer source to the training server and follow README_plainconv_depth.md to create the independent nnUNetPlansPlainConvDepth.json from the original source plans; reuse original data_identifier and fixed patient splits.
3. Select 2d_stage7 (all original stages), 2d_stage6 (remove stage7), or 2d_stage5 (remove stages6 and7). Use the same ordinary decoder, TopK10/early-stopping/single-output policy across these depth comparisons, with fresh per-configuration outputs.
4. Validate checkpoint_final with --val and compare original full-volume per-case Dice/Recall/Precision/HD95/FP/FN against the same case set. Fold0 is a development screen; formal conclusions require controlled five-fold OOF.

## Pending evidence

- Official runtime discovery/reconstruction, forward/checkpoint/CUDA compatibility and performance of the new Trainer are not locally validated.
- Original plans/preprocessed source data/fixed splits remain unchanged. No real data or training runs were performed by this local task.
