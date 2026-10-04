# Official nnU-Net + UPerNet selected encoder stages

Implementation branch: `codex/upernet-configurable-stages`, base `8a41ce3`.
Status: manual independent Level 3 review PASS; original DDP P1 CLOSED; unmerged.
Reviewed source snapshot is retained in task notes. Server download/CPU validation:
`.task-notes/upernet-selected-stages/server_validation.md`.
Runtime inspected locally: `nnunetv2==2.8.1`, newconda. No dependency changes.

## Configuration

Use the distinct Trainer `nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping`.
Place the parameter beside `architecture` in the desired **resolved 2D configuration**:

```json
"upernet_feature_indices": [1, 3, 5, 6]
```

Never put it into `architecture.arch_kwargs`. Configuration inheritance is supported.
Indices refer to encoder stage outputs, starting at zero. Select 2..n_stages genuine integers
(not bool, float, string or null), strictly increasing, unique, within range. Adjacent selected
cumulative scales must increase on **both** spatial axes. Invalid values fail before model
allocation; actual feature BCHW/channels/batch/strict spatial descent are checked on forward.
No sorting, coercion, deduplication, fallback for invalid values or resizing fixes.

If the key is absent, the original automatic four-level selector is used. For the eight-stage
`[[1,1],[2,2],...]` reference this is `[1,3,5,7]`. Explicit `[1,3,5,7]`, `[1,3,5,6]`,
`[2,3,4,5]`, `[0,7]` and `[0,1,2,3,4,5,6,7]` are valid with that geometry.
Existing UPerNet Trainers continue to select automatically and keep their old checkpoint format.

The complete official PlainConvUNet encoder is retained and executed, including stages later
than the deepest selected stage. This is feature selection, not the NoStage7 experiment.
The existing decoder constructs N-1 top-down lateral/refinement branches, PPM `(1,2,4)` at
the deepest selected stage and `N*128` concatenation channels. FPN128, exact-size bilinear
interpolation with `align_corners=False`, raw single logits, loss, optimizer, scheduler, batch,
patch, augmentation, foreground sampling and early stopping remain inherited and unchanged.
New-model Conv2d accounting includes the **complete encoder** and selected decoder, using actual
convolution geometry on odd/anisotropic inputs. Legacy accounting is preserved; its encoder
library uses floor division and is not exact for odd strided-convolution shapes.

## Generate a separate plans copy

All commands below are operating instructions; no real plans/data/server runs were performed
in this task. Deploy these reviewed extension files together to the server.
Generate one new plans file per combination. Example in Windows PowerShell:

```powershell
$py = 'D:\Anaconda\envs\newconda\python.exe'
$env:nnUNet_extTrainer = 'E:\study\研一\work14-图像分割\segementation-upernet-configurable-stages\nnunet_ext_trainers'
# Set nnUNet_raw, nnUNet_preprocessed and nnUNet_results to your existing authorized roots.
$dataset = 'Dataset501_ISLES2022' # Replace with the exact existing dataset folder name.
$source = Join-Path $env:nnUNet_preprocessed "$dataset\nnUNetPlans.json"
$output = Join-Path $env:nnUNet_preprocessed "$dataset\nnUNetPlansUPerNetStages_s1356.json"
& $py "$env:nnUNet_extTrainer\create_upernet_stage_plans.py" `
  --source $source --output $output --configuration 2d --feature-indices 1 3 5 6
```

Source must be a current-format UTF-8 plans JSON with an unambiguous `plans_name`, resolved
architecture and original `data_identifier`. The script rejects duplicate JSON keys, missing
configuration/invalid inheritance, source/output same resolved path, an existing output and
incorrect output names. No `--force`. Output basename must be
`nnUNetPlansUPerNetStages_s<concatenated indices>.json`; `plans_name` is synchronized.
All other configuration fields and inheritance are preserved. The script only reads/writes
plans JSON and does not read images, labels, splits, predictions or checkpoints. Existing
preprocessed data are reused using unchanged `data_identifier`; do **not** rerun preprocessing.
Keep a run's saved plans and checkpoint immutable after training begins.

Examples:

| Feature indices | Derived plans name |
| --- | --- |
| 1 3 5 7 | nnUNetPlansUPerNetStages_s1357 |
| 1 3 5 6 | nnUNetPlansUPerNetStages_s1356 |
| 2 3 4 5 | nnUNetPlansUPerNetStages_s2345 |
| 0 7 | nnUNetPlansUPerNetStages_s07 |
| 0 1 2 3 4 5 6 7 | nnUNetPlansUPerNetStages_s01234567 |

## Training, continuation, validation and prediction

The following server/real-data commands are **not executed by this task**. Check CPU/RAM and
all GPU/VRAM resources before a separately authorized run. Begin a fresh result directory for
each combination; never resume a baseline or another combination's checkpoint.

```powershell
$trainer = 'nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping'
$plan = 'nnUNetPlansUPerNetStages_s1356'
$bin = Split-Path $py
& "$bin\Scripts\nnUNetv2_train.exe" 501 2d 0 -tr $trainer -p $plan
# Resume the exact same run:
& "$bin\Scripts\nnUNetv2_train.exe" 501 2d 0 -tr $trainer -p $plan --c
# Validate a saved checkpoint; do not combine --c and --val:
& "$bin\Scripts\nnUNetv2_train.exe" 501 2d 0 -tr $trainer -p $plan --val
# Example inference into a NEW derived-output directory:
& "$bin\Scripts\nnUNetv2_predict.exe" -i '<authorized imagesTs>' -o '<new predictions directory>' `
  -d 501 -c 2d -tr $trainer -p $plan -f 0 -chk checkpoint_final.pth
```

Results follow official naming:
`$nnUNet_results/<exact dataset>/$trainer__$plan__2d/fold_0/` (the literal `__` delimiters).
On Linux activate the existing server environment and use `nnUNetv2_train`/`nnUNetv2_predict`
with the same flags and `export nnUNet_extTrainer=/absolute/extension/path`.
For five-fold prediction use `-f 0 1 2 3 4` **only after** matching folds were actually trained.
Preserve that extension path/source, Trainer, plans identifier, configuration and fold identity
for `--c`, `--val` and predictor. The official training hook saves plans.json in each result root.

## Checkpoint identity and official predictor

Only the new model adds network state `_extra_state`: protocol version, effective indices,
number of levels/full encoder stages, selected scales/channels, classes, PPM and FPN128.
Its root load pre-hook validates exact primitive types and values **before** recursive parameter
copying. New Trainer loading validates this identity before calling the unchanged inherited
early-stop loader, so mismatches cannot overwrite weights, optimizer or logger state.
Missing identity is refused, including legacy UPerNet checkpoints with identical tensor shapes.
No `strict=False`, checkpoint conversion or training-loop copies are used.

Official predictor rebuilds from saved plans and calls `network.load_state_dict(network_weights)`
for the first fold and each subsequent fold. The same root hook therefore guards all those
calls; identity is not dependent on checkpoint top-level metadata. Different same-shape stage
combinations are rejected. A later invalid fold is rejected at its load in the official fold
loop, before its weights are copied (official initialization only loads the first fold).
CPU `torch.compile(..., backend='eager')` wrapper loading has synthetic coverage; CUDA compilation
and multi-rank/CUDA DDP execution are deferred, not claimed verified.

## DDP for selections before the final encoder stage

Trailing unselected stages still execute but do not reach the loss, so their parameters
naturally have `grad=None`. The new Trainer, after inherited compile/SyncBatchNorm/optimizer
initialization and before the first forward, rebuilds only its DDP wrapper with
`find_unused_parameters=True` when the deepest selected stage precedes the encoder's last
stage. The compiled module, parameter objects, optimizer, device ids/output device and
process group are retained; the old reducer is released first. Selections containing the
last stage retain the official default wrapper. No stages are removed/frozen and no dummy
gradients, loss changes or training-loop copies are used. Existing Trainers remain unchanged.

The new Trainer passes the validated normalized `network_weights` mapping to its inherited
loader, avoiding a `module.` prefix being retained when loading the inner DDP network. It
shallow-copies the checkpoint container and preserves the caller's dictionary.

Tiny single-rank CPU Gloo tests exercise the real new initialization integration, substituting
only the parent's CUDA-dependent setup with CPU compile-then-DDP construction. Plain and eager
compiled networks run two forward/backward iterations with a last-stage control, verify full
encoder execution and unused gradients/parameter/optimizer identity, and strict DDP checkpoint
round-trip/mismatch behavior. This is not actual official CUDA Trainer initialization, multi-rank
DDP, CUDA compilation, SyncBatchNorm execution or a performance result. Those remain unverified.

## Resources and scientific boundaries

Selecting stage0 fuses at input resolution. At 512x512, FPN128, batch one, the final eight-level
concat contains `8*128*512*512 = 268435456` elements (1 GiB in float32). The original four-level
stage1-highest concat is `4*128*256*256 = 33554432` elements (128 MiB), an 8x ratio for that tensor.
This is structural arithmetic, **not** a measured whole-network memory or speed claim. Activations,
autograd and optimizer need additional memory. Do not silently change patch/batch/FPN to make
the eight-level run fit; establish server feasibility under a separate authorized preflight.

Local evidence is tiny synthetic CPU forward/backward, Conv2d hook accounting (including
129x129, two-channel complete eight-stage encoder), single-rank CPU Gloo DDP, fresh official external discovery,
checkpoint round-trip/refusal and real official predictor synthetic directory reconstruction.
No patient data, real checkpoints, full-size eight-level training, CUDA/server, multi-rank DDP/performance
or formal medical metrics were tested. Known warnings originate in installed SciPy/PyTorch
and are recorded in validation evidence.
Dataset501 DWI-only baseline, fixed patient five-fold splits, original full-volume metric space,
raw images/labels read-only and inference GT exclusion remain unchanged. Fold0 is development
screening; scientific performance claims need controlled fixed-patient five-fold/full-volume OOF.
Prediction-guided Stage2 must continue to use OOF Stage1 predictions.

To reproduce engineering tests, use `.task-notes/upernet-selected-stages/run_validation.py`
with newconda and a unique evidence label. It samples CPU/RAM/all GPU/VRAM before each command,
sets serial CPU threads, isolates temporary files outside source and records exact commands,
exit codes, summaries and source hashes. At >=80%, it refuses a file/suite run until downgraded
to an exact tiny test node. It does not interrupt other processes.
