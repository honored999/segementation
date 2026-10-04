# Status

Updated: 2026-10-04
Scope: branch-local codex/upernet-no-stage7, baseline 8a41ce3b0d4e2c980c9e9063a60b055b0a841d89. Not merged into master; this is not canonical integration memory.

## Current verified engineering state

- Added external nnUNetTrainerUPerNetNoStage7TopK10EarlyStopping, inheriting the existing official-framework UPerNet/TopK10/early-stopping Trainer with an architecture-only override.
- Returned encoder contains stages 0..6, preserving their supplied configuration and initial weights; stage7 and corresponding metadata removed. UPerNet inputs fixed at (1,3,5,6), with native sizes 256,64,16,8 for 512x512 input; PPM (1,2,4) and FPN128 unchanged.
- Original plans/configuration, loss, early stopping, optimizer, scheduler, batch size, augmentation, split and inference/checkpoint policy remain unchanged. Distinct Trainer output identity; fresh initial training, same-Trainer continuation only.
- newconda nnunetv2==2.8.1: focused 10 passed, affected original UPerNet 25 passed. CPU synthetic only, including fresh official discovery, current builder signature, strict reconstruction and exact convolution accounting.
- Independent read-only Level3 review PASS on exact source/test/doc hashes recorded in .task-notes/validation-evidence.md. Reviewer did not implement or rerun tests.
- Original NoStage7 implementation was delegated and committed/pushed as f1b9ca8. Main agent made coordination/memory edits only; no master integration.

## Scientific boundaries and next work

- Dataset501 fixed patient-level five-fold split and original full-volume metrics remain protected; real source data read-only.
- No real checkpoint/data/CUDA validation, server training, fold evaluation or formal performance evidence produced by this task. 512x512 native geometry checked from strides; synthetic forward used 128x128 tiny channels.
- Deploy matching extension source to training host, verify actual plans/runtime/resources, and run a fresh fold0 development screen with final-checkpoint/full-volume comparison. Formal conclusions require controlled five-fold OOF evidence.

## PlainConv depth follow-up (2026-10-04)

- User stopped the proposed NoStage6And7 UPerNet extension before any file was written. The replacement task is one external configurable-depth PlainConvUNet Trainer with the matching official UNetDecoder.
- Implemented control: zero-based encoder_last_stage in a saved inherited plans configuration; keep stages 0..last and last decoder upsampling levels. Separate derived plans/configuration identities reuse the original preprocessing data identifier and fixed splits.
- Keep existing TopK10, early stopping and single-output supervision. User explicitly requested no new tests or test executions; implementation completed, independent exact-snapshot static review PASS. No runtime or performance validation claimed for this replacement.

- User authorized committing and pushing the accepted new Trainer/README for server delivery. Production implementation was delegated; prior Trainer/Mixins/tests unchanged. Branch-local records only; canonical master memory untouched.
- Static reviewed SHA256: Trainer A3A0E50192C8B3E2C360C06353050CBA5E373C9B023042E944109B86146CCFEC; README E41AB426137CB2D366434A1799E2E487B2351E59C1C32AC361AAC4C9C762A212. No runtime acceptance implied.
