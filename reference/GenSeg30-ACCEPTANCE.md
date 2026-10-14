# GenSeg30 acceptance checklist (parent coordination)

- [ ] User unit is extra TWO-DIMENSIONAL slices: target ceil(0.30*N), N counts all train-only preprocessed slices. No 30% patient-count expansion and no arbitrary 20% mixing.
- [ ] Canonical fixed Dataset501 fold0 grouping, no split creation/replacement/fold fallback. Train/validation disjoint; external/test IDs unavailable to generator/source selector/scaling/QA.
- [ ] Generator and pool provenance match exact original train IDs, source data/config/plans and split hashes. Reject missing/tampered/incomplete/foreign artifacts before unsafe reads.
- [ ] Original 76 real train identifiers retained. Synthetic slices allocated by patient, global total exact. Dataset assembly preserves real arrays and correctly updates binary/-1 masks and foreground locations. Arrays stacked for TRAINING SAMPLING ONLY, never full-volume metric/export geometry.
- [ ] Validation/final inference remain 19 real patients through original nnUNet dataset and original geometry. No synthetic entries in do_split/perform_actual_validation.
- [ ] Generator import isolation, exact architecture checkpoint restore, finite float scale/pad inversion, labels binary/-1, shape maintained; no PNG/8bit/resize bridge.
- [ ] Fresh dedicated output paths; root/source/raw ancestor/descendant checks and symlink resolution. Refuse collisions/overwrites. No source decompression/unpack or raw mutations.
- [ ] Explicit human quality acknowledgement required; no automatic approval. QA feedback only for training-derived generated examples.
- [ ] Original EarlyStopping Trainer is unchanged. New Trainer's network/loss/optimization/epochs/checkpoint selection/augmentations/sampling contract inherited. Other baseline Trainer identity is not assumed equivalent.
- [ ] Tiny focused and affected tests have immediate CPU/RAM/GPU/VRAM preflight. No numerical performance/real-data claims from synthetic tests.
- [ ] Independent separate-context Level3 PASS on exact final snapshot and all blockers resolved.
- [ ] CLI/README usable from server checkout; preparation commands and nnUNet_extTrainer/pool env shown, baseline caveat and original-space evaluation explicit.
- [ ] User data/checkpoints/runtime manifests absent from source Git diff. Existing untracked proposal preserved/updated only by parent. Canonical master project memory untouched.
- [ ] Server data/access are not locally verified; actual generator training, QA, B fold0 training and A/B metrics remain pending until executed there.

## Final engineering acceptance (2026-10-14)

- Fresh guarded CPU validation: 74 passed, 4 dependency deprecation warnings, 93.31s. No real medical data or server training.
- Independent Level3 final PASS: all 15 file hashes match frozen snapshot SHA256 938e2b7c54bea00f04142161be65117041f07369c78801e5c574bf3731636906; no remaining blockers.
- Server compatibility, human image quality, CUDA/AMP/DDP training and matched full-volume metrics remain pending.
