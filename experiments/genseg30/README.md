# Dataset501 fold0: 30% extra **2D slices** (offline exploration)

This prepares a runnable experiment, not a performance result. The training server, real-data training, human lesion/anatomy QA, and matched A/B metrics have not been executed by this worker. All local tests use tiny synthetic engineering fixtures. No dependency installation, dataset conversion, raw modification, split regeneration, or upstream edits are part of this implementation.

## Compatibility gate comes first

Use the existing `newconda` environment with `nnunetv2==2.8.1`, PyTorch, NumPy, Pillow and the already-present nnU-Net dependencies. The pinned source lives at `reference/GenSeg` (upstream commit `6480e178a5d44cb838f53f7951aae48138b8f5eb`). Run from the repository root and keep that folder on `PYTHONPATH` so external Trainer discovery can import `experiments.genseg30`.

**B training requires the ORIGINAL Dataset501 source to use Blosc2 `.b2nd`.** Official NPZ unpacking can create, repair or delete source `.npy` files, even when unpacked files already exist. The new Trainer keeps the official dataset class unchanged and refuses NPZ before inherited training starts. The `preflight` command checks this using metadata/filenames only, and `train` repeats that check BEFORE fitting the generator. The lower-level Python helpers support read-only NPZ for synthetic tests; that is not a claim that the B Trainer supports NPZ. No converter is supplied.

The source must be exactly `Dataset501_StrokeLesion`, original `nnUNetPlans.json` / `2d`, one `DWI` channel, binary labels 0/1 (preprocessed sentinel -1 retained), non-cascaded, fixed fold0 76 train / 19 validation. The repository's canonical split is `standalone_nnunet2d/reference/splits_final.json`, pinned to SHA256 `02a84d777e4eb9f481c3a9d90529c1673c828e14f2699fd0e898596ce21574ed`. `--raw-root` must agree with configured `nnUNet_raw`; actual imported nnU-Net results/preprocessed path values are used for Trainer boundary checks. Both output base and fold folder are checked against repository/raw/source/pool/generator paths BEFORE the parent constructor can create directories or write logs. The existing server split must agree in all five fold memberships. Its exact bytes/hash are recorded; no split is created or rewritten. Reject missing/foreign/test source IDs before loading source arrays.

Preflight is a compatibility check, not a resource or array-quality check. Before server training, check CPU, RAM, each GPU and VRAM; do not begin a heavy run at or projected above 80%. No full-size GPU or server experiment was validated locally.

## Windows CMD commands

Replace only the directory placeholders below with the actual server locations. `DERIVED`, the fresh generator run, the fresh pool, and results must be outside the repository, raw data, and source/preprocessed roots. Keep the original source read-only. Use a new result root for this trial; do not overwrite historical A or B results.

```bat
conda activate newconda
set "REPO=D:\segementation"
set "nnUNet_raw=D:\nnUNet_raw"
set "nnUNet_preprocessed=D:\nnUNet_preprocessed"
set "nnUNet_results=D:\nnUNet_results_genseg30_trial01"
set "DERIVED=D:\genseg30_derived_trial01"
set "SOURCE=%nnUNet_preprocessed%\Dataset501_StrokeLesion\nnUNetPlans_2d"
set "RUN=%DERIVED%\generator_5000steps_seed20261014"
set "POOL=%DERIVED%\pool_seed20261015"
set "nnUNet_extTrainer=%REPO%\nnunet_ext_trainers"
set "PYTHONPATH=%REPO%;%PYTHONPATH%"
cd /d "%REPO%"
python -m experiments.genseg30.cli preflight --source "%SOURCE%" --raw-root "%nnUNet_raw%"
```

Stop on a failed preflight. Do not run conversion or silently switch the experiment to another dataset/plans/backend. A successful result should report Blosc2, fixed fold0 76/19 and metadata-only compatibility.

The modest initial budget below is **5000 generator updates**, subject also to the 200 generator-epoch ceiling. This is a deliberate exploration setting, not a reproduction of the paper or a change to nnU-Net's budget. One update is one native full training slice (batch 1); `generator.json`, the tensor checkpoint, and the pool manifest record the requested limits, completed epochs and actual updates. A source with fewer than 5000 total updates in 200 passes finishes at the epoch limit. `--width 64 --downs 8` retains the upstream default width/eight-level U-Net; larger native planes/mixed upconvs can require substantial VRAM. Width, downsampling depth and both budgets are explicit generator-only assumptions; agree changes before comparing another trial. The CPU tests use width 1 / five levels and do not establish default-width GPU memory feasibility.

```bat
python -m experiments.genseg30.cli train --source "%SOURCE%" --raw-root "%nnUNet_raw%" --output "%RUN%" --epochs 200 --max-steps 5000 --width 64 --downs 8 --discriminator-width 64 --lr 0.00002 --l1-weight 100 --seed 20261014 --device cuda:0
python -m experiments.genseg30.cli export --source "%SOURCE%" --raw-root "%nnUNet_raw%" --run "%RUN%" --output "%POOL%" --seed 20261015 --device cuda:0
python -m experiments.genseg30.cli audit --source "%SOURCE%" --raw-root "%nnUNet_raw%" --pool "%POOL%"
```

`RUN` and `POOL` must not exist. Interrupted outputs remain incomplete and are not accepted; choose a new directory and preserve the failed run rather than deleting, overwriting or pretending to resume it. A completed generator produces `generator.pt` and `generator.json`; the checkpoint includes both G/D weights and BOTH global architecture tensors. The pool's `manifest.json` is written last, only after all arrays and preview grids exist and train-source hashes are rechecked.

**Human QA is a separate action.** Inspect the `_preview.png` grids for all generated slices: original training image, generated image, and generated image overlaid with the paired mask (lesion red, sentinel -1 yellow). Review lesion correspondence, plausible DWI/brain anatomy, missing/fabricated lesion appearance and failure/collapse on background conditions. All grids derive only from original training patients; do not use validation images/labels to judge, scale, tune or pick the generator. If quality is inadequate, leave the pool unapproved and report the failed exploratory trial. More training, another seed/width or other changes require a separately recorded trial.

Only after actually inspecting and accepting those grids, explicitly record the acknowledgement below. Supply the human reviewer name; there is no default, `--auto-approve`, or inferred approval from file existence. This records the human assertion and the exact manifest/preview hashes, not medical efficacy or proof that a human viewed a screen.

```bat
python -m experiments.genseg30.cli approve --source "%SOURCE%" --raw-root "%nnUNet_raw%" --pool "%POOL%" --reviewer "REPLACE_WITH_HUMAN_REVIEWER" --acknowledgement "I reviewed the training-only generated image-mask previews and approve this exploratory pool."
```

Copy the exact SHA256 printed by the completed export/audit into a separate run record and the variable below. Do not recompute/accept a new hash to bypass a mutation failure. Pool/source inputs must remain immutable for training.

```bat
set "GENSEG30_POOL=%POOL%"
set "GENSEG30_MANIFEST_SHA256=REPLACE_WITH_64_HEX_DIGITS_FROM_AUDIT"
python -m experiments.genseg30.cli audit --source "%SOURCE%" --raw-root "%nnUNet_raw%" --pool "%POOL%" --require-quality --manifest-sha256 "%GENSEG30_MANIFEST_SHA256%"
```

A uses the plain existing EarlyStopping Trainer. B inherits that exact class; its only change to training data assembly is through `get_tr_and_val_datasets`. Additional constructor/split/validation/checkpoint guards enforce source/output and resume integrity. Run with the same environment, plans, source, hardware/training options and fixed split. Baseline A also needs Blosc2 in this source-preserving trial; otherwise its inherited unpacking can write source files. Do not substitute TopK10, Foreground50, UPerNet or another Trainer and call it a matched comparison. The parent identified plain EarlyStopping as the minimal baseline; historical A metrics remain unverified for protocol matching (including preprocessing, initialization, augmentation, stopping/checkpoint selection and the 19-patient cohort).

```bat
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerEarlyStopping -p nnUNetPlans
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerGenSeg30EarlyStopping -p nnUNetPlans
```

Neither command caps nnU-Net epochs/iterations. B inherits network, loss, batch size, official online augmentation, 250 training iterations, optimizer/scheduler, foreground oversampling, maximum epochs, early stopping and best-checkpoint selection. No pretrained nnU-Net weights are introduced. Inherited EarlyStopping monitors EMA pseudo-Dice, which is distinct from final full-volume Dice. The B result folder stores an immutable `genseg30_pool_identity.json`; resuming with `--c` must use the same pool/hash/source. Checkpoint serialization itself remains inherited: an immutable pool receipt is attached under `init_args.genseg30_pool_identity`, which the parent serializer already saves. The guarded load requires the existing receipt, matching Trainer/source/configuration and embedded pool identity BEFORE delegating unchanged network/optimizer/EMA/early-stopping restoration. Missing/deleted receipts, copied checkpoints from another pool/Trainer, or a continuation without a loaded matching checkpoint fail; a receipt is created only for a fresh run without existing checkpoints. Ordinary numeric NumPy logger metadata is supported by restricted `weights_only` deserialization, without arbitrary pickle execution. DDP/CUDA/AMP and default-size training have not been exercised by these CPU tests.

Official training performs full-volume validation at the end. To run the inherited final prediction/validation explicitly on the same original fold0 19 patients using the best checkpoint for BOTH A and B:

```bat
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerEarlyStopping -p nnUNetPlans --val --val_best
nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerGenSeg30EarlyStopping -p nnUNetPlans --val --val_best
set "A_PRED=%nnUNet_results%\Dataset501_StrokeLesion\nnUNetTrainerEarlyStopping__nnUNetPlans__2d\fold_0\validation"
set "B_PRED=%nnUNet_results%\Dataset501_StrokeLesion\nnUNetTrainerGenSeg30EarlyStopping__nnUNetPlans__2d\fold_0\validation"
python evaluate_segmentation_metrics.py --pred-dir "%A_PRED%" --gt-dir "%nnUNet_raw%\Dataset501_StrokeLesion\labelsTr" --output-dir "%DERIVED%\metrics_A_original19"
python evaluate_segmentation_metrics.py --pred-dir "%B_PRED%" --gt-dir "%nnUNet_raw%\Dataset501_StrokeLesion\labelsTr" --output-dir "%DERIVED%\metrics_B_original19"
```

Validate that both prediction folders contain exactly the fixed 19 validation patient IDs and matching original geometry before accepting any paired metrics. `--val` first checks Dataset501 / 2d / fold0 / the existing fixed split and cohort, then delegates original full-volume data/properties through the original dataset class. Missing splits are refused, never regenerated; the exact server split ordering is retained. It does not require loading synthetic samples; it regenerates derived predictions in the corresponding validation output, so use the dedicated trial root above. Do not evaluate the training sampling containers, cropped generator inputs or preview PNGs. Do not aggregate this fold0 exploratory screen into a five-fold OOF result or describe the effect as known before the real server run.

On a Linux server, the Python/nnU-Net command arguments are identical. Replace CMD variable setup with `export REPO=/path/to/segementation`, `export nnUNet_raw=/path/to/raw`, `export nnUNet_preprocessed=/path/to/preprocessed`, `export nnUNet_results=/path/to/new_results`, `export nnUNet_extTrainer="$REPO/nnunet_ext_trainers"`, `export PYTHONPATH="$REPO:${PYTHONPATH:-}"`; use `/` and `$SOURCE`, `$RUN`, `$POOL`, `$DERIVED` and `$GENSEG30_MANIFEST_SHA256` in place of the CMD forms. Activate `newconda` and `cd "$REPO"` first. Do not change source/backend, count unit, fold or evaluation protocol when adapting shell syntax.

## Scientific and implementation assumptions

- This is **offline mask-conditioned GAN augmentation**, not joint multi-level GenSeg reproduction. No Betty, architecture search/validation optimization, end-to-end nnU-Net gradient feedback or imgaug is used. The pinned U-Net/PatchGAN and vanilla GAN + L1 objective are reused via isolated imports. Architecture logits retain seeded upstream initialization and stay frozen; conv logits currently are unused by upstream's plain downconv but are still saved/restored. All cached module arch-parameter lists are rebound to restored tensors. Separate generator instances retain separate module globals.
- Default generator fitting uses batch 1, BatchNorm, no dropout, constant Adam LR `2e-5`, betas `(0.5,0.999)`, L1 weight 100, no generator-side affine/online augmentation, last completed update selection and no validation tuning. The upstream example's 200 pretraining epochs were for a small example set; the 5000-update ceiling is an explicit additional assumption. The checkpoint records actual updates and exact settings.
- Scale ALL original training preprocessed float values by one train-only min/max range to tanh representation, then invert generated floats to the original normalized representation. Pad the bottom/right to a multiple of `2**downs`, unpad to each original H/W, and never resize/resample, histogram-equalize or use PNG/8-bit as model input. Ignore/padding pixels are masked for GAN image inputs and L1; original flipped image support is retained at exported -1 pixels. A degenerate constant-valued source or all-ignore training slice is rejected explicitly.
- Count `N` = ALL preprocessed z slices in the 76 training patients, including background. Add integer target `(3*N+9)//10`. Allocate `floor(3*Zp/10)` per patient then largest remainders, with lexical patient-ID tie breaks. Seeded source z selection uses replacement; in-plane x/y flips are recorded and applied to paired labels and sentinel masks. Labels are transformed original train targets, not independently generated masks. No affine or anatomical perturbation is silently substituted. Counted new slots are not guaranteed novel independent observations: repeated all-background conditioning can produce repeated or implausible images, which QA must reject when unsuitable.
- Original real patient identifiers remain 76. Every `load_case` builds `[1,Zp+Kp,H,W]` as a **training-only sampling container**. Appended z is not anatomical adjacency, spacing or a patient volume. It must never be exported as NIfTI, used as an inference ROI/input or used for full-volume metrics. Validation stays the untouched original 19-case dataset and geometry.
- The official loader draws cases with its original probabilities. Conditional on a selected patient, its existing random/foreground crop logic sees the combined array; foreground locations are regenerated using the official sampling routine, with coordinates `[channel,z,y,x]`. In the non-foreground-forced path the extra-slice chance conditional on patient p is `Kp/(Zp+Kp)`; this is not an arbitrary global 30% fake-slice mixing probability. Background slices are never removed from the count.
- Manifest/checkpoint/preview hashes, source-file inventory, canonical and server split hashes, original plans/dataset/source identity, source z/shape/counts, seed/transforms/scaling, actual generator budget and completion/classification are recorded. Runtime artifacts contain sensitive IDs/images and stay outside Git/source. Incomplete, missing, foreign or mismatched provenance is rejected before source dataset loads; QA is checked before B assembly.
- At wrapper startup, perform a cryptographic content audit and bind it to unchanged file identities before/after the audit. During each patient load, check global metadata and that patient's artifacts before/after reading using resolved path, size, mtime nanoseconds, device/inode/mode and a trustworthy metadata-change token. On Windows, `stat.st_ctime` is creation time; the code instead queries actual `FileBasicInfo.ChangeTime` through `GetFileInformationByHandleEx` on a read-only `FILE_READ_ATTRIBUTES` handle (also supports directory identities) and refuses if unavailable. On other platforms use `st_ctime_ns`. Same-size rewrites with restored mtime are covered by a regression. Reject mutations instead of rehashing all compressed source files every batch or silently updating trust. This requires immutable source/run/pool directories and does not authenticate hostile actors able to forge metadata and external hash records. No unbounded all-patient RAM cache is used; full native container concatenation and foreground regeneration may still add CPU/IO cost relative to the lazy baseline loader.

## Local verification

From repository root, use the existing environment; no full repository suite or real-data run:

```bat
python experiments/genseg30/tests/run_guarded.py -q experiments/genseg30/tests
python experiments/genseg30/tests/run_guarded.py -q nnunet_ext_trainers/tests/test_trainer_early_stopping.py
```

The runner samples CPU/RAM and GPU/VRAM before every invocation, refuses at 80%, sets one CPU thread and disables CUDA for its pytest child. Tiny actual upstream-network forward/backward and arch-checkpoint round-trip are engineering checks only. Focused tests cover exact counts, train/validation exclusion before reads, missing/mismatched split/reference, singleton shapes, foreground coordinates, output boundaries, tampering/incomplete/foreign provenance, explicit QA, the actual official loader and original validation class, no-op Blosc2 unpack, NPZ preflight rejection and unchanged inherited Trainer methods. Final independent Level3 review belongs to the parent and must inspect the exact final files; passing local tests is not that independent review.
