# CKPD independent review record

Worktree: C:\Users\Windows\.codex\worktrees\nnunet-ckpd-finetune\segementation
Branch: codex/nnunet-ckpd-finetune
Starting commit: 414cdac3146d83feea8577f64624586eb431b5a0
Scope: ordinary 2D Dataset501 fold-0 CKPD private-cohort fine-tuning.

## First frozen snapshot: BLOCKING

Reviewer: independent read-only gpt-6-luna / max collaboration context; did not implement. All seven supplied file SHA256 values verified at start/end. The review ran no model/checkpoint/data tests and changed no files. The implementation worker reported 46 passing focused CPU synthetic checks, official external Trainer discovery and CLI help, but review identified gaps not covered by those tests.

Confirmed blockers:
1. Official --val_best reload on a live parametrized Trainer fails; standalone validation requires --val --val_best.
2. CUDA autocast lowers B @ A product precision despite FP32 factors, violating initialized reconstruction; preserve precision locally while retaining Conv AMP.
3. Source validation IDs also influence early-stopping checkpoint selection but were excluded from cohort overlap checks.
4. Source/target binding omits encoder normalization/activation/order semantics and calibration patch-size equality.
5. RNG state contents are not fully prevalidated; failures after live model/optimizer/logger mutations leave partially restored state.

Frozen first-snapshot SHA256:
- nnunet_ext_trainers/ckpd.py: 44ACCAB8138BF793A9F46481BC031F111099D4F1096042DFE3BCA69C32E656C7
- nnunet_ext_trainers/nnUNetTrainerCKPDTopK10EarlyStopping.py: 8D9A84859674DA1BCCA5AC0C6F95BFC5E92202424A07123C996187D4E2AC9831
- nnunet_ext_trainers/tests/test_ckpd.py: 8F1B6D8147EE5EDE28BFFAE811F65F39484A6467A12C3703C808586608DA35FB
- nnunet_ext_trainers/tests/test_trainer_ckpd.py: 4432C5FB621C520007AF0E34DBAAB1BC7CFC73877AB66C2832153CF798B6B4EC
- tools/calibrate_nnunet_ckpd.py: 578F4B35EA154D480B25B54D8FB8BE80A849CFC470A57D3F72DD68765AE64A66
- tools/tests/test_calibrate_nnunet_ckpd.py: 727E548D23F47D93ADCBEE4A3FE39EEC7A587CF86EE911DC4325B3C3D0AB6A5C
- docs/NNUNET_CKPD_FINETUNE.md: E0D97AE3AE3E58E17D5B5F4C5D460E0F850EFF95AE320CB990FCFD16C92E3266

## Repair/re-review status

The focused leaf fixer implemented the five repairs and regression tests; the parent made no production source/test edits. Independent read-only Level 3 re-review returned PASS with no confirmed blockers. The reviewer verified all seven final hashes and inspected the repaired runtime paths without running models or modifying files. Dataset501 documentation now uses nnUNetPlans; source calibration keeps nnUNetPlansFrom501.

Fresh parent verification in newconda: 51 passed, 5 upstream warnings in 22.23s across test_ckpd.py, test_trainer_ckpd.py, test_calibrate_nnunet_ckpd.py, and existing direct TopK10/early-stop/composed tests. Before this CPU-only one-thread run: CPU 25%, RAM 68% used / 5 GiB available, GPU 6% utilization / 1372 of 6144 MiB. CUDA was hidden from pytest. This is small synthetic engineering evidence. Installed runtime: nnunetv2 2.8.1, Torch 2.11.0+cu126, dynamic-network-architectures 0.4.4.

Final frozen SHA256:
- nnunet_ext_trainers/ckpd.py: b3ceb4e1a1aacae9d4ad8bc7e76c7086a3dce66b95abf36f92f5e1491ecd85dd
- nnunet_ext_trainers/nnUNetTrainerCKPDTopK10EarlyStopping.py: 26567d626d5219dab5b80249cb92cb89d2c6dee1551d45725d2aea8499dcdf5f
- tools/calibrate_nnunet_ckpd.py: 7211217f976edbf16ba97b57554606bb911af479732169f8d8845d474b000217
- nnunet_ext_trainers/tests/test_ckpd.py: 9555b6769cce9c4b6d1ba3e45a806ee802d7b77f52a6938e462cb8fca1b4a66c
- nnunet_ext_trainers/tests/test_trainer_ckpd.py: 3107ce2d167b992a65588c5cdc49df21c42ae0693e180ad045c19fdf17c54fbb
- tools/tests/test_calibrate_nnunet_ckpd.py: 7c2c8a77991cbeef66cebdc4d1aa29621cb2bcc49220c0d80fd9c1c0cdbbbef4
- docs/NNUNET_CKPD_FINETUNE.md: b750177f37d0eaab32714d757e758b4fc5fbd44a5ca218770e4c03691ae46baa

Same-environment continuation restores saved training/RNG state but does not serialize augmentation worker queues. Cross-device exact RNG continuation is not supported: a CPU-origin torch_cuda=None state does not restore CUDA RNG.

## Evidence limits

No local real data, production checkpoint/full model, CUDA execution or formal evaluation. Source cohort independence needs server-side provenance. Original six-run source checkpoint identity is unverified; a causal comparison requires a new same-weight full-finetuning control. Restoring RNG/model/optimizer/epoch state does not reproduce multiprocessing augmentation queues. Master is not changed or integrated by this task; checked-out project memory is branch-local.
Final pre-commit hygiene: removed one extra EOF LF from the Trainer and its test, with no logical changes; added .gitattributes LF rules for the six new Python paths to preserve byte identities on Windows. Parent performed only this minimal integration configuration plus coordination documents, not production Python/test implementation.

Independent final hygiene re-review: PASS remains. Adding one LF to each cleaned file exactly reproduces its previously reviewed SHA256; the reviewer verified the six exact LF rules. No logical source changes or additional tests.

## 2026-10-12 optional dataset name compatibility fix

Server evidence: plans.dataset_name is Dataset508_ISLES2022DWI and plans_name is nnUNetPlansFrom501, but dataset.json omits name. Official nnU-Net 2.8.1 generate_dataset_json accepts dataset_name=None and writes name only if provided. Calibrator now allows an absent/null name while rejecting an explicit conflicting name; exact checkpoint metadata, plans identity and run-folder guards remain intact. No dataset JSON was modified.

Three regression nodes: old code 2 failed / 1 passed; repaired calibrator file 10 passed. Fresh parent CPU calibration+CKPD helper+Trainer suite: 30 passed, 4 upstream warnings in 7.96s. Preflight CPU27%, RAM64% /6GiB available, GPU10% /1372 of6144MiB; pytest hides CUDA and uses one CPU thread. Independent focused review PASS; no real calibration/training.

Frozen patch SHA256:
- tools/calibrate_nnunet_ckpd.py: bfdc6bcc2e2fb4734a34923c8e8b75f91b54b20f846f0f32b120086db767c855
- tools/tests/test_calibrate_nnunet_ckpd.py: ae9b50e3ca632a2e64b301fcd977861237da1cee4366a6e015608797719f81e1

## 2026-10-13 NumPy EMA checkpoint compatibility fix

Server reported training done after early stopping, followed by --val_best load failure at _best_ema metadata validation. Official validation computes np.nanmean and MetaLogger retains numeric scalar type; CKPD saves _best_ema unchanged. Validator now accepts finite numbers.Real scalars (including NumPy), still rejects bool/np.bool_, NaN/Inf and non-real containers or values. No saved fields, optimizer, early-stopping or checkpoint selection semantics changed; no real checkpoint edited.

Leaf-fixer RED: 3 failing new regressions, 28 passing. GREEN affected Trainer file:40 passed. Fresh parent newconda CPU Trainer+early-stop+composed suite:63 passed,5 upstream warnings in26.95s. Preflight CPU19%,RAM58%/7GiB free,GPU12%/1028 of6144MiB; tests hid CUDA and used one CPU thread. Independent focused review PASS. Actual server full-volume validation still pending; recover using --val --val_best on the existing checkpoint, not continuation training.

Frozen source SHA256: Trainer898c7db8e129df2285a9f030000dc60dab4fde8c574afdecc353bb3afc43d191; test9765e0275a2cce194245d2cb6b6970d02ff685c3d23e76c2b6c47cc9bc39ef61. Implementation/tests delegated; parent handled review, documentation and Git only.
