# Fixed fold0 PlainConv depth pretrain / fine-tune

Double-click `run_plainconv_depth_pretrain_finetune.bat`, or call it from CMD.
Both success and failure pause so the final paths/error stay visible. Edit only
its top server settings as needed; confirmed defaults are nnunet5090, physical
GPU 1 and HD-BET trial01. Data augmentation uses nnUNet_n_proc_DA=2. conda.bat must be on PATH or configured explicitly.

Order: fresh Dataset508 ISLES pretraining, then full-parameter HD-BET3mm
Dataset501 fine-tuning from this run's checkpoint_best.pth, then the two original
full-volume metric evaluations (paper F2). Both stages use fold0, best validation,
existing Trainer TopK10/early stopping and deep supervision true. The inherited
architecture drops the last encoder stage and corresponding ordinary decoder
level via encoder_last_stage=n_stages-2. No weight remapping is needed.

Preparation reads JSON metadata with utf-8-sig (BOM tolerant) and checks filenames only. It compares resolved
2d configurations (including batch/patch/spacing/normalization/architecture while preserving each original
data_identifier and checking each prepared folder separately), dataset interfaces,
transpose and image_reader_writer. Dataset-specific foreground intensity statistics
are retained separately and are not required to match. It creates only
nnUNetPlansDropLastDS.json next to each source plan, retaining existing configs
and adding 2d_drop_last_ds. Existing identical derived plans can be reused;
conflicts fail. Original plans/splits are never edited. ORIGINAL_PP supplies the
original five-fold Dataset501 split; its HD-BET copy must match with fold0 76/19.
Prepared data must already exist; no preprocessing, planning or HD-BET runs.

Results use a fresh timestamp directory under OUTPUT_PARENT. Existing roots fail;
failed outputs are retained, with no retry/resume/deletion. run_receipt.json
records plan/split/Trainer and script hashes, environment and physical GPU UUID.
CUDA_VISIBLE_DEVICES uses that UUID. Resource gates sample CPU/RAM/GPU/VRAM before
preparation and immediately before each training. CPU/RAM-only gates also run
immediately before each metric evaluation. Any relevant reading >=80% or
unavailable fails. These
are startup observations, not a continuous limit or a prediction of training's
future memory usage. Optimizer/LR/stopping settings are left to Trainer;
nnUNet_compile is not set or modified.

After both trainings finish, validation prediction filenames must exactly match
their existing fold0 validation lists. Source GT comes from ISLES_RAW labelsTr;
target GT comes from ORIGINAL_RAW labelsTr, never the HD-BET/cropped GT. Existing
evaluator geometry checks remain authoritative. Both metric directories must
contain nonempty case_metrics.csv, summary_metrics.csv, summary_metrics.json
before the final two absolute output paths are printed.

## HANDOFF / verification boundary

Implementation is limited to this README, the BAT and
nnunet_ext_trainers/prepare_plainconv_depth_pipeline.py. Static source inspection
only. Tests: NOT_RUN. Model construction/import/weight loading: NOT_RUN.
Real data access: NOT_RUN. BAT/server execution/training/evaluation: NOT_RUN.
No installation, commit, push, original Trainer or project-memory changes.
Server metadata/dependencies/CLI behavior and runtime geometry remain unverified;
parent agent owns independent static review and any later execution acceptance.
