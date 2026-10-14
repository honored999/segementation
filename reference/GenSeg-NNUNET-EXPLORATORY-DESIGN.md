# Dataset501 fold0: 30% extra synthetic 2D training slices

Status: user-approved scope; implementation pending. User confirmed Dataset501 DWI-only on the training server, fold0 only, 30% measured by 2D slices (not patient counts or an arbitrary mixing probability).
Start: codex/genseg-reference, 656ab4f6f130f15def85d3b388c1a4b73c6da633. Existing untracked draft belongs to this task.

## Experiment contract

- Keep original fixed splits byte-for-byte; fold0 local reference has 76 train / 19 validation patients. Server split is authoritative and must be present, match the fixed reference/cohort, and have disjoint train/validation membership. Never regenerate splits.
- N counts ALL preprocessed z slices in the 76 train patients, including background slices. Add exactly ceil(0.30*N) synthetic image/label slices. Do not count validation/test slices or use them in generator fitting, scaling, source selection, previews, augmentation, or checkpoint tuning.
- Allocate synthetic counts per training patient by floor(0.30*z_count) plus largest-remainder increments until the global ceiling is reached. Source z selection is seeded and train-only.
- Preserve existing real-case uniform sampling: the train dataset still exposes the original 76 patient identifiers. Its load_case combines the original slices and that patient's generated slices in a TRAINING-ONLY sampling container. This appended axis is not physical 3D anatomy and must never be exported as a patient NIfTI or used for validation/inference geometry.
- Keep each patient plane shape and recompute foreground class_locations against the actual combined arrays, including synthetic slices. No stale z coordinates, no GT-derived inference input.
- Validation and final inference retain the official original dataset_class, original 19 IDs, original preprocessed properties, and raw full-volume space. Synthetic slices are visible only to the training dataset wrapper. Reject fold !=0, non-2D, cascades, non-single-channel/nonbinary data, missing/mismatched splits/manifests, and output paths inside raw/source data.
- New Trainer nnUNetTrainerGenSeg30EarlyStopping inherits nnUNetTrainerEarlyStopping and changes only training dataset assembly. Preserve network/loss/batch size/iterations/augmentation/foreground sampling/optimizer/scheduler/early stopping and checkpoint selection. Default plain early-stopping baseline; if the user's actual baseline is TopK10 or another variant, do not claim a matched comparison until that identity is confirmed.

## Generator and derived pool

- Reuse pinned upstream GenSeg generator/discriminator architecture; no edits inside reference/GenSeg. This is offline conditional-GAN augmentation, not full joint GenSeg reproduction.
- Read only original nnU-Net preprocessed float images and labels, never PNG/8bit/histogram-equalize preprocessing. Pad/unpad full 2D planes to a generator-compatible multiple without resampling. Explicit train-derived intensity scaling may be used for tanh generation, must be recorded and inverted to the original normalized representation; no validation statistics.
- Train generator on train patients only with a fixed seed; affine/pad/intensity settings and model width/epochs must be recorded as implementation assumptions. Default upstream 200 pretraining epochs is an exploration setting, not claimed paper reproduction.
- Preserve BOTH weights and conv_arch/upconv_arch tensors and verify round-trip output equality with a tiny engineering fixture. Freeze/checkpoint architecture assumptions explicitly; no accidental random architecture reset on reload. No Betty multi-level integration or imgaug dependency is required for this scope.
- Generate paired image/mask slices using recorded in-plane flips (whole paired labels/ignore masks), preserve binary/-1 semantics, finite floats, source patient and z, and exact source shapes. Ensure DWI lesion-label quality via saved preview grids and a required explicit quality approval before Trainer consumes the pool. Human QA cannot be inferred from a file merely existing.
- Pool manifest includes split hash, canonical reference split identity, train patient set, per-case original/synthetic slice counts, source folder/config/plans identity, generator checkpoint hash, transforms/seed/scaling, file hashes, completion status, and experiment classification. Never commit runtime patient IDs/data/weights/previews.
- Fail closed before medical-data reads for foreign/test provenance, manifest tampering or incomplete output. Fresh output only; no overwrite, deletion, inplace conversion, resume masquerading as fresh run, or source writes. Generated output only in a dedicated user-specified derived directory outside repository and protected inputs.

## Acceptance and run boundary

- Focused tiny synthetic tests: exact ceil30% allocation; train/val exclusion before load; missing split/reference mismatch; single-slice pool shapes; updated foreground coordinates; raw/output boundaries; tamper/incomplete/foreign provenance rejection; actual nnU-Net loader use of wrapper; checkpoint architecture round-trip with small widths on CPU only when resource headroom permits; validation dataset unchanged.
- Resource preflight before every test/smoke command: CPU/RAM/GPU/VRAM, downgrade at 80%. No real-data training locally, no dependency install, no GPU/full-suite test without justified headroom.
- Independent Level3 read-only review required for final diff, focusing leakage, sampling semantics, geometry, source safety, checkpoint identity and validation separation.
- Deliver concrete Windows/server commands for train generator -> generate + audit -> quality approve -> nnU-Net B fold0 -> original-space predict/evaluate with unchanged 19 patients. Explain where nnUNet_extTrainer and the pool env var are set.
- Report this as an exploratory fold0 trial; actual effect is unknown until server run. Keep baseline A reusable only after current Trainer/preprocessing/initialization/cohort provenance is confirmed. No automatic push or server training.

## Baseline identification (current verified local branches)

User could not find the old server command and requested local branch inspection. master, feat/h2former-stroke, codex/genseg-reference, codex/nnunet-exclude15-ab and codex/isles2022-dwi-conversion contain the same ordinary early-stopping Trainer blob ed19239b9e14f8765c0f79e42fca149c3929d5c6. The UPerNet-composition branch implements the same named role through a separate mixin; TopK10EarlyStopping is a distinct loss-changing variant. This trial selects the current ordinary nnUNetTrainerEarlyStopping implementation explicitly; no TopK10 or architecture changes. Historical server result comparability is still unverified.
