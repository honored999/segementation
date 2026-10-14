# GenSeg30 fold0 implementation plan

Goal: Generate ceil(30% of train-only preprocessed slices) paired synthetic DWI slices and add them only to current fold0 early-stopping nnU-Net training.
Architecture: a standalone offline preparation CLI and verified per-source synthetic pool; a training-only dataset wrapper preserves original case identifiers and the official validation path. Reuse pinned upstream generator architecture with complete checkpoint identity. No source/pipeline refactors.
Stack: Python/newconda, PyTorch, NumPy, nnunetv2==2.8.1; existing upstream GenSeg source only.
Owner: one normal leaf implementation worker; parent owns design/coordination/review. Worker must not delegate. Parent never duplicates worker source/tests.

## Tasks

1. experiments/genseg30/: implement train-only split/reference and path checks, exact allocation, pure helpers and synthetic regression tests. Follow TDD; preflight every execution. Keep runtime manifests outside source.
2. experiments/genseg30/: implement original preprocessed-array reading, pad/unpad and scaling, GenSeg import isolation, generator training/checkpoint/reload, offline pool generation, hash audit, quality review gate and portable CLI. No package installation or upstream edits.
3. nnunet_ext_trainers/nnUNetTrainerGenSeg30EarlyStopping.py: implement train-only wrapper and minimal inherited get_tr_and_val_datasets integration. Dataset identifiers remain real training IDs. Reject unsupported folds/configs; retain original validation and final inference. Test actual loader shape and foreground path.
4. experiments/genseg30/README.md: server commands from repo root using existing nnUNet paths; B has its own Trainer identity. Distinguish QA acknowledgement from quantitative benefit, paper mechanism from implementation assumptions, slice count from patient count, and training sampling arrays from patient volumes.
5. Return HANDOFF with exact files and resource-aware focused/affected test evidence. Parent identifies final snapshot, delegates independent Level3 review; worker fixes blockers, tests affected paths, reviewer rechecks exact final state.
6. Parent verifies scope/raw-data invariants and docs/CLI coherence, synchronizes branch-local project memory and reports runnable server instructions plus remaining access/baseline limits. No changes to canonical master memory or automatic push.
