# HD-BET + GenSeg: pretraining-to-joint protocol proposal

Date: 2026-10-18
Status: User approved alternative 1 on 2026-10-18. Implement G/D-only pretraining and completed-checkpoint handoff; first trial is 1+1 engineering updates. Long budgets and interrupted-run resume are excluded from this first implementation.
Inspected local checkout: E:/study/研一/work14-图像分割/segementation
Branch: codex/genseg-reference; starting commit: 22df4843502d4b78894566c8a00bd4d2fa5e5bdf; initially clean.
Current synchronization instruction: local changes, validation and independent review, then commit/push; data computer pulls with git pull --ff-only. This supersedes historical server-first delivery instructions. Never force-push or overwrite unrelated work.

## Cohort resolution

The user explicitly confirmed on 2026-10-18 that 94 was a typo: Dataset501 still has 95 cases. The checked-in reference dataset.json declares numTraining=95. Each of its five fixed folds contains 76 train and 19 validation cases, with no within-fold duplicates or overlap. Across the five validation lists there are 95 entries and 95 unique cases.

Reference splits_final.json SHA256: 02a84d777e4eb9f481c3a9d90529c1673c828e14f2699fd0e898596ce21574ed.

Keep the existing derived inner61 weight-training / inner15 architecture-optimization / external19 development-validation roles. Preserve memberships and the existing seed 20261018; do not generate a new partition. The actual trial01 protocol must be checked against these identities before a future run. This local machine cannot access C:/lijialin/HD-BET_outputs, so remote receipts and hashes below are user-reported, not reverified locally.

## Existing evidence and immutable inputs

- The user reports the native 512x512, physical GPU1, one-step cold-start joint pilot completed, with top-level complete=true and joint.pt SHA256 b6dca49de21b9d74ca4b01431814b3f8b0ee91c010e57e385d05a296fbfd5039. This is engineering evidence, not trained-model quality evidence.
- Reuse read-only targets C:/lijialin/HD-BET_outputs/genseg3mm_trial01/targets; manifest SHA256 23d25e2b2a6d7c7ca2a270690a9d4d35116ff4f834458b48a9049ff889894a56.
- The old generator_5000steps_seed20261014_attempt02 saw all outer76. It remains provenance-only and cannot initialize the new inner61 protocol. Do not overwrite it or the completed joint_trial01.
- Current CLI accepts pretrain_steps=0 only and has no resume loader. Saving RNG and optimizer state does not by itself implement resume.

## Source audit

Primary: Nature Communications 16:6486 (2025), DOI https://doi.org/10.1038/s41467-025-61754-6. Its Methods describe fixed-architecture GAN weight fitting, segmentation fitting and validation-driven architecture optimization; the reported experimental budget of 5000 iterations is not a verified DWI pretraining budget.

Pinned author code: reference/GenSeg, upstream 6480e178a5d44cb838f53f7951aae48138b8f5eb.

- running_files/train_end2end_jsrt.py explicitly loads G/D weights and creates a fresh UNet.
- running_files/train_pix2pix_lung.py actually loops range(200), although scripts/train_pix2pix_lung.sh passes n_epochs=10000. Neither setting is directly equivalent to DWI slice-update steps.
- README.md describes training/validation data use that does not establish the stricter inner61-only pretraining boundary required here. Our boundary is a deliberate project adaptation.
- models_pix2pix/pix2pix_model.py also contains a separate architecture-optimization method. Do not silently copy all upstream entrypoint behavior into the current repaired chain.

## Alternatives

1. Recommended: pretrain G/D only on inner61, then start a fresh U-Net and joint architecture optimization. Closest to the inspected author mainline initialization, with the smallest added phase.
2. Pretrain G/D and separately pretrain U-Net on inner61 real targets before joint optimization. Adds another training factor and budget; requires a separate experimental rationale.
3. Extend cold-start joint training without pretraining. Fewer phase transitions, but does not answer the intended pretraining-plus-joint experiment.

## Recommended phase contract (proposal)

1. Read-only check: audit reference/source memberships, targets hash and provenance; compare derived split and scaling policy with trial01. No outside19 arrays.
2. GAN pretraining: construct fresh G/D; train weights only on inner61, keeping both architecture tensors fixed. Use the existing valid-pixel GAN+L1 objective (L1=100), frozen BN, full native planes/padding and inner61-only scaling. Reuse current stateless SGD/rates initially to avoid changing optimizer and pretraining together; this remains an engineering deviation from the paper's Adam GAN optimizer. U-Net and alpha Adam do not update.
3. Pretraining receipt: save a phase-identified checkpoint with G/D weights, architecture, completed GAN step count, source/targets/split/scaling identities, optimizer policy/state, RNG and completion audit. Select a fixed-budget last-complete checkpoint; no inner15 or external19 checkpoint selection. Preview quality checks, if performed, use inner61 and are development decisions.
4. Joint initialization: load the exact same-inner61 GAN checkpoint, including its architecture tensors; initialize U-Net freshly with an explicitly recorded seed. Begin fresh outer-alpha Adam moments. Preserve G/D stateless SGD semantics and existing virtual-G, virtual-S and actual-update order. Record the pretraining checkpoint hash and phase boundary. No arbitrary old-checkpoint warm start.
5. Joint optimization: inner61 for G/D/S weights, inner15 for architecture hypergradients only; keep outer19 arrays excluded. Preserve existing upconv-only search, frozen BN, SGD virtual/ordinary updates and staged alpha Adam. This is still an engineering adaptation.
6. Run first a short pretrain-to-joint engineering handoff check (for example 1+1 updates), with explicit non-formal classification. Authorize a longer fixed budget only after correctness, selected-GPU resource behavior and checkpoint transitions are verified. Do not edit the old cold-start command to 5000 and call it this protocol.

## Checkpoint scope

Phase handoff and interrupted-run resume are different features. The minimum implementation supports a verified completed pretraining checkpoint feeding a new joint run. Full interrupted-run resume is optional scope and must be explicitly implemented/tested if selected; no resume claim from serialization alone. Use fresh output directories under C:/lijialin/HD-BET_outputs, preserve incomplete runs, and write completion receipts only after final audits. This proposal does not promise periodic recovery until implemented.

## Engineering acceptance before commit/push

Implementation must be delegated to a leaf worker with no nested agents. Independent Level3 review must inspect the exact final source/tests snapshot, phase data roles, checkpoint compatibility and differentiable joint chain. Run resource-preflighted focused/affected tests in newconda. Verify frozen architecture during GAN pretraining, no inner15/external19 array reads during that phase, same-cohort checkpoint checks before mutation, no live state changes on invalid handoff, phase counts/RNG semantics, and successful tiny actual-network transition. Preserve existing cold-start behavior and source/raw/split/old-weight boundaries. Document CUDA/real-data validation limits honestly. Commit/push only the accepted intended changes after reviewing Git status/diffs/checks; do not include datasets, patient identifiers or checkpoints.

## User-approved scope and later decisions

- Alternative 1 approved: G/D only, then fresh U-Net joint training.
- Set explicit GAN pretraining and joint update budgets for the later experiment; no inferred paper-default DWI budget.
- First implementation supports completed-pretraining handoff only; interrupted-run resume is excluded. Initial data-computer phase trial is1+1 updates after local acceptance.

External19 remains the existing fold0 development cohort. No final-test, five-fold OOF, image-quality or segmentation-benefit conclusion is established by this design.
