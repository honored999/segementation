# Joint Adam positive pilot: review before any server run

Strategy: `joint_adam_trainbn_positive_sgdproxy_v1`; see GENSEG-JOINT-REPAIR-PROTOCOL.md. This is a new controlled engineering combination. It changes the optimizer/BN protocol and decouples positive synthetic-mask sampling from real sampling. It is not a single-factor lesion filter experiment, exact differentiable Adam, or paper reproduction.

Use the existing server repair checkout `C:\lijialin\segementation\.worktrees\genseg-quality-repair` and `conda activate nnunet5090`. Verify branch/status/commit first; synchronize only after parent acceptance and manual independent Level3 review. Do not pull into the active reference checkout. Retain SOURCE, nnUNet_raw, BASELINE, TARGETS, TARGET_SHA from GENSEG-QUALITY-REPAIR-RUNBOOK.md.

Pinned upstream source remains 6480e178a5d44cb838f53f7951aae48138b8f5eb. Canonical CRLF-to-LF hashes and observed raw hashes are both retained; raw source mismatches are not waived. Local author `models_pix2pix/pix2pix_model.py` uses Adam G/D and architecture Adam with decay; author end-to-end S uses RMSprop. Our fresh S Adam, zero-decay staged upconv Adam, eval-BN virtual SGD proxy, independent positive synthesis and nested61/15 roles are deliberate local adaptations. Paper: https://www.nature.com/articles/s41467-025-61754-6 . No claim of numerical reproduction.

CMD variables (templates only; the approved 80-epoch budget is documented below, production preflight remains pending):

```bat
set "GD20=C:\lijialin\HD-BET_outputs\genseg_repair_gd_epoch20_trial01"
set "GD20_SHA=1547b42e65a7f21e8452d3509ce862f059884dd7e2bb82047eea7b33b7a5b75b"
set "JOINT=C:\lijialin\HD-BET_outputs\genseg_joint_adam_positive_pilot_trial01"
set "STRATEGY=joint_adam_trainbn_positive_sgdproxy_v1"
```

`gan.pt` is weights-only phase transition: audited20 epochs,970 valid planes/epoch,19400 G/D steps; copy G/D parameters, live BN counters/statistics and both alpha groups. Fresh seeded author UNet(1,1); fresh G/D/S Adam moments, fresh staged alpha Adam, fresh RNG and sampler. It is not resume. Do not load old S or single-slice diagnostic state. Existing quality receipts remain unchanged and learning_quality_passed remains false.

Check requires an explicit RAM cache budget in bytes; choose CACHE_BYTES after inspecting available RAM and manifest estimate. MODEL_RESERVE_BYTES must conservatively cover models/optimizers/higher-order copies. Separate LOADER_WORKSPACE_BYTES must reserve unknown external codec/pickle workspace; it must not be borrowed from the model reserve; OS free headroom8GiB and resource80% caps also apply. Unknown production peaks remain unknown. No auto disk-cache fallback. `psutil` is required; local newconda lacks it, server availability is unverified; fail closed without installing libraries.

```bat
python -m experiments.genseg_joint.joint_cli check --strategy "%STRATEGY%" --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%JOINT%" --gpu 1 --pretrained-run "%GD20%" --pretrained-sha256 "%GD20_SHA%" --cache-budget-bytes %CACHE_BYTES% --loader-workspace-bytes %LOADER_WORKSPACE_BYTES% --model-reserve-bytes %MODEL_RESERVE_BYTES%
```

Check hashes provenance/masks and estimates memory; it does not load patient image/label arrays or model tensors, create CUDA context, or create output. TargetDataset retains its existing mask audit. Training startup then actually loads and binds the RAM snapshot. The user approved 80 real-inner61 epochs; production resource preflight remains pending. The future command template is:

```bat
python -m experiments.genseg_joint.joint_cli train --strategy "%STRATEGY%" --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%JOINT%" --gpu 1 --pretrained-run "%GD20%" --pretrained-sha256 "%GD20_SHA%" --cache-budget-bytes %CACHE_BYTES% --loader-workspace-bytes %LOADER_WORKSPACE_BYTES% --model-reserve-bytes %MODEL_RESERVE_BYTES% --epochs 80
```

Only physicalGPU1 by default, bound once by UUID before CUDA initialization. One selected-card nvidia-smi query at a unit boundary; primed nonblocking CPU delta, immediate RAM and strict telemetry failures. No busy sleeps, empty_cache, retry or OOM continuation. Guard is a boundary external-load/headroom check, not a continuous GPU utilization limiter. Other cards do not gate this card.

Real sampling keeps all valid inner61 slices, including negatives, with old patient/plane rule and its RNG sequence. Independent seeded RNG uniformly samples inner61 positive planes and independent xy flips; label/condition/ignore stay aligned. No size stratification, small-lesion threshold, or demonstrated small-lesion improvement. Inner15 affects alpha through architecture loss only; it is not an independent holdout generalization result. Outer19 remains development validation and is never read as arrays here. Historical76-derived fill/preprocessing is disclosed, not refit.

Private virtual G/D/S use entry learned BN statistics in eval mode; virtual G/S each take one SGD proxy step. Both direct-alpha and virtual-G indirect routes remain. Actual staged new alpha is used for GAN G1/D3 trainBN gradients before either weight update; then Adam G and D; updated G eval/no_grad generates the positive S pair; actual S uses real+fake trainBN forwards, one Adam step. Initial counters restored19400/58200, fresh S0; increments1/3/2 per unit. No S-loss update to G. Failure records completed units and failure stage, poisons engine after live failure, never claims rollback or retries.

Startup/postcache/end/save audit full hashes; unit guards check file identity/stat and small source hashes. Private owned contiguous read-only arrays preserve the bound training input. Stat guards alone cannot detect content changed and restored between checks; final full hashes reject invalid runs. No inner-loop whole-volume decompression or full76 SHA. Cache owns one image/label copy; planes/normalization/augmentation allocate independent arrays. Budget failure stops.

`steps.jsonl`: host stage enqueue wall times, audit/resource/sample wall, completed unit wall, explicit unit-boundary CUDA synchronization wall, allocated/reserved peaks, real/fake/architecture source/augmentation and forward counts. Stage host times do not attribute CUDA kernel time exactly; asynchronous execution is charged to boundary sync and total unit wall. `timing.json` records startup/cache/log/checkpoint wall. Saved checkpoint timing is a pre-serialization snapshot; final serialization wall is in timing.json. No real throughput or speedup ratio measured. Compare future steps/sec and stage proportions under matching card/resource/cohort/unit budgets.

Preview is schema2 strict safe weights-only CPU load; only G constructed/restored. Set JOINT_SHA from complete.json and PREVIEW to a new independent directory:

```bat
python -m experiments.genseg_joint.joint_cli preview --strategy "%STRATEGY%" --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%PREVIEW%" --gpu 1 --run "%JOINT%" --checkpoint-sha256 "%JOINT_SHA%" --patient-limit 4
```

Fixed sorted inner61 representatives use existing lesion selection, real reference/mask/generated/overlay and common window; ignore=-1 excluded. Negative synthetic previews are zero, without dropping real-negative S training or concealing false-positive evaluation. A selected patient without positive planes is recorded as missing and skipped; preview fails only when no selected patient has an exportable positive plane. Legacy preview/diagnostic500 locks remain unchanged.

After a future approved production preflight, collect live BN/Adam/alpha state, actual positive fake counts, timing, RAM/VRAM peaks, fixed output roundtrip and image/segmentation quality. A successful unit is engineering evidence only; do not start long training or export30% synthetic data automatically.


B1/B2 follow-up: joint snapshot accepts audited Blosc2 source headers only; NPZ/NPY source backends and unaudited dtypes are explicitly refused before loading. check reports final_cache_estimated_bytes, max_case_load_transient_bytes, loader_workspace_reserve_bytes, model_reserve_bytes and startup_required_bytes. Startup required = final cache + maximum additional case transient + model reserve. Before each case the live resource guard reserves remaining final cache (subtracting actual cached nbytes), current case transient and model reserve; after each load it samples resources again. The explicit cache budget limits permanent owned arrays, not all RAM. External codec/pickle workspace and real production model peaks remain UNKNOWN, even with reserves.

Known limitation: nvidia-smi utilization at a unit boundary can include the preceding unit's own work and stop a high-throughput run at >=80%. Threshold, no-wait/no-retry policy unchanged. Observe this during a separately authorized short real preflight; no mitigation or server run is authorized here.


## Real epoch budget (2026-10-21)

The user approved 80 real-inner61 epochs. Production single-card resource preflight
is still incomplete; this implementation task does not launch server training.
Use `--epochs 80` as in the train command above; `--epochs` and `--steps` are
mutually exclusive, exactly one required. Existing `--steps` keeps its original random
patient/plane sampling. In epoch mode the existing EpochSampler traverses every
valid real inner61 plane once per epoch without replacement, including negatives;
all-ignore planes are excluded. N is measured from the in-memory snapshot, not
hardcoded. Current expected N=970 gives 80*970=77600 joint units.

Example progress: `Epoch 1/80 | 1/970 | Unit 1/77600 | S loss ... | Val loss ...`.
Epoch order has its own RNG; fake positive sampling remains independent and
inner15 keeps the original random sampling rule. Fake/inner15 coverage is not
an epoch guarantee. Epoch mode does not consume real draws from the shared old
real/validation RNG, so its validation sequence differs from steps mode.
`steps.jsonl` keeps detailed metrics and records epochs, total_valid_planes and
current epoch_progress; protocol records the same budget and final progress.
Only the final checkpoint is saved; interrupted runs cannot resume. No periodic
checkpoint or new training strategy is added. Existing production resource,
quality and audit gates remain required.
