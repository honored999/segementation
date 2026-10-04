# Status

## 2026-10-04 server CPU results / CUDA probe preparation

- User reported server focused 61 passed and legacy 25 passed, exit 0, on Python 3.10.20/nnunetv2 2.8.1/PyTorch 2.11.0+cu128. Short evidence labels avoided a 279-character Windows synthetic log path failure; production code/tests unchanged. Full server evidence hashes not independently collected here.
- Task-local CUDA0 tiny FP32 probe added; real official initialize/loss/two forward-backward iterations/checkpoint identity, no initializer shim or optimizer step. Its CPU controls passed locally (2 passed). Server CUDA result PENDING. Resource guard refuses CUDA at >=80%, still querying all GPUs.
- Independent production-source review PASS/P1 CLOSED remains anchored to unchanged source. Branch unmerged; canonical memory untouched. DDP/AMP/compile/full-size/real-data/formal evidence deferred.

## 2026-10-04 reviewed branch delivery

- User authorized commit/push of this isolated branch; no merge authorized. Reviewed source/tests unchanged. Server download and initial resource-guarded CPU validation instructions added; server execution assigned to user.
- Fresh precommit external-Trainer CPU suite: 111 passed, exit 0; resources below 80%. Independent Level 3 PASS/P1 CLOSED remains valid for unchanged source. Delivery result is established by Git HEAD and remote branch verification. Server/CUDA/multi-rank/performance/formal evidence remains pending. Canonical memory untouched.

## 2026-10-04 branch-local independent re-review accepted

- Manual independent Level 3 re-review PASS; original P1 CLOSED, no new blocker. Report: .task-notes/upernet-selected-stages/re_review_level3_20261004_01a10481.md.
- Reviewed manifest SHA256: 9cb7266a68b87cf77ce125f86318e2e6db51863610044f0489d2cba0368f39ba. All 37 files and 24 ignored files matched at acceptance; both independent evidence files match all source hashes. Subsequent closeout edits are branch-local documentation only.
- Independent CPU synthetic DDP 6 passed; checkpoint/predictor/compatibility 9 passed, exit 0. Implementation focused 61 passed; affected external-Trainer suite 111 passed, exit 0. CPU DDP uses disclosed parent initializer shim.
- Implementation and independent review complete; uncommitted/unmerged, not integrated into canonical state. Server/CUDA/SyncBatchNorm/NCCL/multi-rank/multi-GPU/full-size performance/real-data/formal evaluation remain unverified. Protected worktrees clean and unchanged; canonical master memory untouched.

## Historical 2026-10-04 DDP fix before re-review

- Independent review on the original SHA256 snapshot returned BLOCKING (P1 unused trailing encoder parameters in default DDP). Its report and original manifest remain immutable evidence; re-review PENDING.
- Only new SelectedStages Trainer source was changed: conditionally rebuild DDP before forward with unused-parameter detection; load validated normalized weights through inherited loader. Full encoder, gradients, optimizer and old Trainer behavior retained.
- New tiny single-rank CPU Gloo/eager-compiled DDP regressions: 6 passed. Final focused 61 passed; affected external-Trainer suite 111 passed, all exit 0. Parent CUDA initialization is replaced by a CPU construction shim for these DDP fixtures, not claimed CUDA proof.
- Uncommitted/unmerged. Server/CUDA/SyncBatchNorm/multi-rank DDP/full-size feasibility/performance/formal evaluation NOT RUN. No raw data/splits/real checkpoints touched. Canonical master memory untouched.

## Historical 2026-10-03 implementation before review

- This checkout is codex/upernet-configurable-stages from 8a41ce3; changes are uncommitted, unmerged and independent review PENDING. Not canonical accepted state.
- New official nnU-Net UPerNet SelectedStages/TopK10/early-stop Trainer reads resolved plans; shared N-level decoder and new network extra_state/pre-load identity guard. Complete encoder and existing training/scientific policies retained.
- Synthetic CPU evidence: focused 55 passed, old official Trainer regression 25 passed, external-Trainer affected suite 105 passed. Actual official discovery/Trainer initialization/resume/predictor reconstruction and same-shape rejection tested.
- Server/CUDA/DDP execution/full-size feasibility/real-data performance NOT RUN. No real data/checkpoints/splits/preprocessing touched.
- master has no tracked .project-memory files at startup; canonical memory untouched.

## Retained baseline branch snapshot (2026-09-21; not revalidated in this task)

## Current state

- The active feature branch contains standalone 2D PlainConvUNet and H2Former
  model families with explicit model/supervision checkpoint identities.
- Existing H2Former uses single-output supervision; PlainConvUNet supports its
  default deep supervision and a matched single-output mode.
- Independent `h2former_lite_upernet` and `plain_conv_unet_lite_upernet`
  single-output variants are implemented at `53472ae`. Both use the shared
  lightweight PPM/FPN decoder while preserving the baseline model identities.

## Verified capabilities/results

- Fresh main-agent validation: standalone suite `424 passed in 47.67s`; root
  `tests/` suite `26 passed in 6.67s`.
- Independent Level 3 review of `bbe65b5..53472ae` returned PASS with no
  blocking findings.
- Synthetic complexity evidence shows both new decoders have fewer parameters
  than their corresponding baseline decoders. This is engineering evidence,
  not medical-performance evidence.

## Verified constraints

- Dataset501 remains the established DWI-only baseline.
- Preserve the existing patient-level five-fold split and original full-volume
  formal-evaluation semantics.
- Synthetic validation is engineering evidence only.

## Active work

- Lite-UPerNet implementation is accepted. No real-data training, preflight,
  fold evaluation, or formal five-fold OOF evaluation has been run.
