# Status

Updated: 2026-10-03
Scope: branch-local codex/upernet-no-stage7, baseline 8a41ce3b0d4e2c980c9e9063a60b055b0a841d89. Not merged into master; this is not canonical integration memory.

## Current verified engineering state

- Added external nnUNetTrainerUPerNetNoStage7TopK10EarlyStopping, inheriting the existing official-framework UPerNet/TopK10/early-stopping Trainer with an architecture-only override.
- Returned encoder contains stages 0..6, preserving their supplied configuration and initial weights; stage7 and corresponding metadata removed. UPerNet inputs fixed at (1,3,5,6), with native sizes 256,64,16,8 for 512x512 input; PPM (1,2,4) and FPN128 unchanged.
- Original plans/configuration, loss, early stopping, optimizer, scheduler, batch size, augmentation, split and inference/checkpoint policy remain unchanged. Distinct Trainer output identity; fresh initial training, same-Trainer continuation only.
- newconda nnunetv2==2.8.1: focused 10 passed, affected original UPerNet 25 passed. CPU synthetic only, including fresh official discovery, current builder signature, strict reconstruction and exact convolution accounting.
- Independent read-only Level3 review PASS on exact source/test/doc hashes recorded in .task-notes/validation-evidence.md. Reviewer did not implement or rerun tests.
- Main agent made coordination/memory edits only; implementation/tests delegated. User authorized branch commit/push; no master integration.

## Scientific boundaries and next work

- Dataset501 fixed patient-level five-fold split and original full-volume metrics remain protected; real source data read-only.
- No real checkpoint/data/CUDA validation, server training, fold evaluation or formal performance evidence produced by this task. 512x512 native geometry checked from strides; synthetic forward used 128x128 tiny channels.
- Deploy matching extension source to training host, verify actual plans/runtime/resources, and run a fresh fold0 development screen with final-checkpoint/full-volume comparison. Formal conclusions require controlled five-fold OOF evidence.
