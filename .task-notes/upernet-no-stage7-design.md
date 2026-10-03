# Approved stage7 removal ablation

Date: 2026-10-03
Worktree: E:/study/研一/work14-图像分割/segementation/.worktrees/upernet-no-stage7
Branch: codex/upernet-no-stage7
Base: 8a41ce3b0d4e2c980c9e9063a60b055b0a841d89

User approved removing only stage7 from the existing Dataset501 official-framework UPerNet TopK10 early-stopping baseline. Implement as a distinct external Trainer, retaining stages 0..6 and explicitly selecting (1,3,5,6). Native decoder inputs for 512x512: 256x256,64x64,16x16,8x8. Preserve PPM (1,2,4), FPN128, loss, optimizer, scheduler, batch size, augmentation, split, preprocessing, checkpoint selection and inference policy. The architecture override must not mutate original plans or include unused stage7 parameters. Initial experiment starts fresh, with its own output identity; no baseline-checkpoint resume or pretrained initialization.

Implementation owner: scoped leaf subagent, no nested delegation. Main owns independent review and acceptance. Only lightweight CPU synthetic tests and official external-discovery/reconstruction checks locally; no patient data/checkpoints, training, full suite or formal evaluation. Each test command requires fresh CPU/RAM/GPU/VRAM preflight and the 80 percent resource guard. Target runtime: newconda, nnunetv2==2.8.1.

Acceptance: final synthetic tests and affected trainer regressions pass; independent read-only reviewer PASS; exact files/digests/status inspected. Actual Dataset501 plan/checkpoint/inference/server compatibility and performance remain pending server checks. Fold0 is a development screen; promotion requires controlled five-fold/full-volume evidence. Shared split and source data remain read-only.

User subsequently authorized commit and push of codex/upernet-no-stage7 for a same-name server worktree. Integration into master remains outside scope. Original worktrees remain protected. Branch-local notes do not constitute canonical accepted project state.
