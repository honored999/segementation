# Status

Updated: 2026-09-21

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

## 2026-10-12 branch-local CKPD acceptance

On codex/nnunet-ckpd-finetune (base 414cdac), the opt-in ordinary 2D Dataset501 CKPD Trainer and Dataset508 train-only calibration CLI are implemented. Encoder rank-4 adapters on four ASR-selected convolutions train with the full decoder; baseline TopK10/early stopping, SGD/PolyLR, nnUNetPlans, fixed splits and full-volume evaluation remain the experiment contract. Merged inference weights and same-environment training-state continuation are supported; augmentation queues are excluded.

Fresh parent synthetic CPU validation: 51 passed; independent Level 3 review PASS. No real-data/CUDA/full-model training or formal evaluation. This is branch-local acceptance, not master integration. Source checkpoint for historical fine-tuning is unverified; use a same-checkpoint full-tuning control and independently verify cohort provenance.

- 2026-10-12 CKPD compatibility patch: source dataset.json name is optional as in official nnU-Net 2.8.1; absent/null accepted, explicit conflict rejected. Checkpoint metadata and plans/path identity guards remain. Fresh parent 30 synthetic CPU tests passed and independent focused review PASS; real server calibration/training still pending.

## 2026-10-13 CKPD checkpoint compatibility

Server reported calibration success and training completion; subsequent best-checkpoint validation was blocked by Python-only EMA scalar validation. Fixed compatibility with finite NumPy real scalars without altering checkpoint fields or training/selection semantics. Fresh parent affected CPU suite63 passed; focused independent review PASS. Real full-volume validation remains pending, and fold-0 evidence must remain separate from formal OOF. This state is branch-local.
