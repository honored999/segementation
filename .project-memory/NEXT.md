# Next

Scope: branch-local codex/upernet-no-stage7; not canonical master memory.

## Current focus

- Local NoStage7 Trainer engineering implementation and independent Level3 review accepted. Real experiment remains pending.

## Next actions

1. Deploy matching external Trainer source to the training host and confirm nnunetv2==2.8.1, official discovery, and actual 512x512/eight-stage source plans compatibility; perform CPU/RAM/GPU/VRAM resource preflight.
2. Start a fresh independent fold0 run using README_upernet_no_stage7.md. Preserve raw data, fixed patient splits and all other baseline training settings; do not resume or preload a baseline checkpoint.
3. Validate checkpoint_final using --val with the same inference policy as baseline; compare original full-volume Dice, Recall, Precision, FP/FN and small-lesion misses. Fold0 is a development screen; no gain claimed from heatmaps or synthetic tests.
4. Any new-Trainer report adapter support is a separate follow-up; current task covers official training/validation/inference reconstruction only.

## Pending evidence

- Real server runtime/checkpoint/CUDA compatibility and experimental performance have not been run locally.
- User authorized committing and pushing this isolated branch for a same-name server worktree; no master integration.
