# TopK10 with Early Stopping Implementation Plan

> For the manually dispatched leaf worker: execute directly using executing-plans and test-validation. The user explicitly requires manual dispatch; do not create subagents, reviewers, fixers, validators, or other chats. User approved this design on 2026-10-02. No further design approval is required within this scope.

**Goal:** Add nnUNetTrainerTopK10EarlyStopping without changing the established TopK10 architecture, deep supervision, loss, optimizer, scheduler, sampling, or augmentation.

**Architecture:** Compose the existing nnUNetTrainerEarlyStopping and nnUNetTrainerTopK10 classes through cooperative inheritance. Reuse their checkpoint and loss implementations; do not duplicate the training loop or refactor existing Trainers.

**Tech Stack:** nnunetv2 2.8.1, PyTorch, pytest; local newconda Python. Server uses Windows CMD and nnunet5090.

## Ownership and snapshot

- Worktree: C:\Users\Windows\.codex\worktrees\isles2022-dwi-conversion\segementation
- Branch: codex/isles2022-dwi-conversion
- Starting HEAD: 0f0f1ed2f84a0d1e65c75a9b2ccb5277617a1894
- Starting tracked state: clean. This coordinator-owned plan is newly created under ignored docs/; do not modify it or mistake it for worker output.
- Protected checkout: E:\study\研一\work14-图像分割\segementation; never modify it.
- Read applicable AGENTS.md and medical-experiment-integrity, test-validation, resource-aware-testing.
- All commands request sandbox escalation; prefer D:\Anaconda\envs\newconda\python.exe.
- No staging, commits, pushes, dependency installs/upgrades, canonical project-memory edits, real-data access, preprocessing, or training.

## Approved protocol

- Trainer name: nnUNetTrainerTopK10EarlyStopping.
- Existing stopping semantics: MAX_EPOCHS=1000; MIN_TRAINING_EPOCHS=300; PATIENCE=100; MIN_DELTA=1e-4. Monitor validation ema_fg_dice. Do not count non-improving epochs during the first 300 completed epochs; count afterward, reset on sufficient improvement.
- Preserve official checkpoint_best selection separately from the stopping min_delta. Persist/restore stopping counters and triggered state; a stopped checkpoint must not perform extra optimizer steps when resumed.
- Preserve the original TopK10 deep-supervision behavior, Dice plus hardest-10-percent CE, and inherited architecture/optimizer/scheduler.
- Server preparation is complete: Dataset508_ISLES2022DWI, 250 cases; nnUNetPlansFrom501, 2d, batch_size=12, patch_size=[512,512]. This is user-reported server preprocessing evidence, not a local run.
- First source experiment: Dataset508 fold 0, expected approximately 200 training and 50 validation cases. Use its own split; never copy Dataset501 splits into Dataset508. Before training, inspect the actual split and class discovery on server.
- Keep Dataset501 raw data, nnUNetPlans, fixed five folds, and baseline outputs untouched. Source fold-0 online Dice is development evidence, not target performance or formal OOF.

## Allowed worker changes

Create only:
- nnunet_ext_trainers/nnUNetTrainerTopK10EarlyStopping.py
- nnunet_ext_trainers/tests/test_trainer_topk10_early_stopping.py
- nnunet_ext_trainers/tests/verify_external_topk10_early_stopping_discovery.py
- docs/ISLES2022_TOPK10_EARLY_STOPPING.md

Existing source, tests, converter, plans, and project memory are read-only. If a real integration defect cannot be resolved in these files, return concrete evidence rather than silently widening scope.

## Task 1: Focused RED and minimal composition

- [ ] Reuse fixture patterns from test_trainer_topk10.py and test_trainer_early_stopping.py. Add focused tests that initially fail because the new Trainer is missing.
- [ ] Cover effective production loss type/k/weights/deep-supervision behavior; do not only assert class constants or inheritance.
- [ ] Add the composition class. Expected minimal class shape is nnUNetTrainerTopK10EarlyStopping(nnUNetTrainerEarlyStopping, nnUNetTrainerTopK10), with no duplicated methods. Choose imports that work under the official external resolver, not only from repo cwd.
- [ ] Prove initialization reaches the base initializer once, enables the usual deep supervision, sets 1000 epochs, and initializes stopping state.

## Task 2: Runtime and checkpoint contracts

- [ ] Exercise actual production methods with tiny synthetic fixtures: warmup, patience boundary, improvement reset, best-checkpoint behavior, and terminal stopping.
- [ ] Use real nnU-Net 2.8.1 logging for at least one EMA monitor test, not exclusively a custom fake logger.
- [ ] Round-trip a tiny CPU checkpoint through inherited save/load. Check state, trainer_name, optimizer state, and next-epoch bookkeeping; exercise resumed stopped state with no extra training step.
- [ ] Verify inherited architecture, optimizer, scheduler, sampling, augmentation and deep supervision are preserved. Avoid constructing the real 512x512 network merely for this check.
- [ ] Fresh subprocess discovery must use recursive_find_trainer_class_by_name with nnUNet_extTrainer set to this worktree's nnunet_ext_trainers directory. Do not preload the new class. Also test from an unrelated ASCII temporary cwd.

## Task 3: Affected validation and documentation

- [ ] Before every test/subprocess verification, preflight CPU/RAM/GPU/VRAM; use CPU, ITK/BLAS single thread, tiny fixtures, disabled bytecode/cache, and unique ASCII D:\Temp basetemp. Do not install psutil if absent; use CIM.
- [ ] Run the new focused module, then that module together with test_trainer_topk10.py and test_trainer_early_stopping.py. Run the new isolated discovery script and existing TopK10 discovery check. Broaden only for a demonstrated affected surface.
- [ ] Document exact Windows CMD discovery, launch and resume commands, stopping semantics, and expected output path. Example launch after independent acceptance: nnUNetv2_train 508 2d 0 -tr nnUNetTrainerTopK10EarlyStopping -p nnUNetPlansFrom501. New name gives a separate output directory; never reuse the original TopK10 result directory.
- [ ] Mark the command as planned server execution, not executed evidence. Distinguish first launch from -c continuation and later Dataset501 pretrained initialization.
- [ ] Inspect final status, diff, whitespace (include untracked files with no-index), and SHA256 of all worker files. docs/ is ignored; report it, do not modify .gitignore or force-stage anything.

## Terminal HANDOFF

Return status COMPLETED/BLOCKED/FAILED; role; requested/observed model configuration; worktree/branch/start/end HEAD; changed files and final SHA256; design decisions and MRO; RED/GREEN/affected commands, results, resource preflight and skips; actual isolated resolver result; checkpoint/resume evidence; final Git status; protected/raw-data/project-memory state; deviations and unresolved issues. No important content after HANDOFF. Next action is independent read-only Level3 review, then server discovery/split/resource preflight and source fold-0 training.
## User clarification: local lightweight tests only (2026-10-02)

This explicit user constraint overrides any broader validation interpretation above.
- Local work is limited to lightweight CPU unit tests: tiny synthetic tensors, actual loss logic on tiny inputs, stopping-state transitions, tiny checkpoint serialization, and isolated class discovery.
- Do not build or initialize the actual PlainConvUNet or any production-size model locally. No 512x512 forward/backward, batch-size trial, optimizer/training run, real-case validation, GPU/CUDA execution, benchmark, throughput/latency measurement, or VRAM stress/profile test.
- Use minimal test fixtures for checkpoint and resume-state logic; do not initialize a complete Trainer network or run a real training loop to verify them. No large checkpoint files.
- Do not run the whole repository suite. Limit affected validation to lightweight cases in the new module and existing TopK10/EarlyStopping modules. If an existing case exceeds this scope, defer that case and record the reason; do not weaken it.
- Resource preflight is read-only monitoring, not a performance measurement; no GPU workload probe is permitted.
- All production-model integration and actual fold-0 training are server-only, after independent acceptance. Report local checks as lightweight synthetic engineering evidence, with any deferred integration checks explicitly listed.
