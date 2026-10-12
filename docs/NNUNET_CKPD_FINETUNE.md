# Dataset501 CKPD fine-tuning

This opt-in experiment adapts the ordinary 2D `PlainConvUNet` from a Dataset508
ISLES-pretrained checkpoint to Dataset501. The new Trainer inherits the existing
TopK10 loss and EMA early stopping. It freezes the encoder except for low-rank
CKPD adapters on four selected Conv2d kernels, trains the decoder, and keeps the
existing plans, preprocessing, augmentation, fold split, validation, SGD,
PolyLR, and checkpoint-selection behavior.

This is a Conv2d transfer adaptation of CKPD, not a complete reproduction of
the paper's FSCIL experiment. The approved paper reference is
[CKPD arXiv v2](https://arxiv.org/html/2501.05017v2), equations 1–5 and ASR
equation 7. The code selects the four layers with lowest `sigma[-rank] /
sigma[-1]` from descending singular values of `W C`. The current author
repository uses `sum(S_tail) / sum(S_all)` in its decomposition helper and has
a revised 2026 paper title; that later metric is intentionally not mixed into
this v2-pinned implementation.

The following are experiment assumptions because the paper does not specify
them for this nnU-Net adaptation: rank 4, four selected layers, eight source
training cases, one seeded axial slice and input-only random crop per case, at
most 256 output positions per case and layer, seed `20261012`, and relative
covariance damping `0.01`. The statistic is an uncentered second moment. For
each source case, sampled patch products are averaged over that case's sampled
positions, then case means are averaged equally; labels never choose cases,
slices, crops, or positions. Covariances are processed one layer at a time with
a default 512 MiB cap, and the script performs a host/GPU resource preflight
before loading the checkpoint or starting calibration.

The official nnU-Net 2.8.1 pretrained loader is used for both calibration and
the fresh training command. It checks matching names/shapes and excludes
`.seg_layers.` keys, so new segmentation heads retain nnU-Net initialization.
Fresh training verifies the supplied source checkpoint hash, calibration
artifact, loaded encoder fingerprint, and source plans/dataset/split hashes.
The artifact stores hashes for both complete Dataset508 fold-0 manifests. The
Trainer verifies both manifests and rejects any exact identifier overlap
between Dataset508 train plus validation and Dataset501 train plus validation;
calibration sampling remains restricted to Dataset508 training IDs. Before it
installs adapters, the Trainer also compares a normalized signature of the
resolved encoder operations and their order, along with the source calibration
`patch_size`, against the target configuration. Synthetic CPU regressions cover
source-validation-only overlap, encoder normalization/activation differences,
and patch-size mismatch. Different anonymization namespaces cannot prove
patient-level cohort independence; that still requires server-side provenance
from the dataset owner.

The source checkpoint path used for the earlier six Dataset501 fine-tuning runs
has not been verified here. Treat those runs as historical references; do not
claim their source weights match this CKPD run. A causal CKPD-versus-full-
fine-tuning comparison requires rerunning the full-fine-tuning baseline from
the exact same selected Dataset508 checkpoint, with the fixed Dataset501 split
and the same remaining protocol.

## Server calibration

Use the already configured Windows server Conda environment with
`nnunetv2==2.8.1`. First inspect available source checkpoints and choose the
intended Dataset508 fold-0 checkpoint explicitly; do not let a wildcard select
one implicitly. In PowerShell, the read-only discovery command is:

```powershell
$resultsRoots = Get-ChildItem -LiteralPath 'C:\lijialin\models3d\nnUNet' -Directory -Filter 'nnUNet_results*'
$trainerDirs = foreach ($resultsRoot in $resultsRoots) {
  $datasetRoot = Join-Path $resultsRoot.FullName 'Dataset508_ISLES2022DWI'
  if (Test-Path -LiteralPath $datasetRoot) {
    Get-ChildItem -LiteralPath $datasetRoot -Directory -Filter 'nnUNetTrainerTopK10EarlyStopping__*' -ErrorAction SilentlyContinue
  }
}
foreach ($trainerDir in $trainerDirs) {
  $foldPath = Join-Path $trainerDir.FullName 'fold_0'
  Get-ChildItem -LiteralPath $foldPath -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -in @('checkpoint_best.pth', 'checkpoint_final.pth') } |
    Select-Object -ExpandProperty FullName
}
```

Before calibration, ensure the intended pretrained checkpoint and the existing
Dataset508 prepared data/plans/split are the same experiment. The artifact
must be written outside both raw and preprocessed dataset roots. For example,
from the worktree root in `cmd.exe`:

```bat
conda activate nnunet5090
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
set nnUNet_compile=false
set "NNUNET_CKPD_SOURCE_CHECKPOINT=C:\path\to\selected-source-checkpoint.pth"
set "NNUNET_CKPD_CALIBRATION=C:\lijialin\models3d\nnUNet\ckpd-calibration\dataset508-fold0-r4-k4.pth"
python tools\calibrate_nnunet_ckpd.py ^
  --preprocessed-dir "%nnUNet_preprocessed%\Dataset508_ISLES2022DWI" ^
  --dataset-json "%nnUNet_raw%\Dataset508_ISLES2022DWI\dataset.json" ^
  --plans-json "%nnUNet_preprocessed%\Dataset508_ISLES2022DWI\nnUNetPlansFrom501.json" ^
  --source-checkpoint "%NNUNET_CKPD_SOURCE_CHECKPOINT%" ^
  --output "%NNUNET_CKPD_CALIBRATION%" ^
  --configuration 2d --fold 0 --cases 8 --rank 4 --layers 4 ^
  --max-positions 256 --max-covariance-mib 512 --seed 20261012 --device cuda
```

Select the source checkpoint according to the existing Dataset508 experiment record: `checkpoint_best.pth` follows its early-stopping selection, while `checkpoint_final.pth` is a different choice. Record the selected filename and hash; do not silently substitute one for the other. The command prints `CKPD_EARLY_RESOURCE_PREFLIGHT` before source metadata deserialization and `CKPD_RESOURCE_PREFLIGHT` after network construction but before the official weight loader. The early check reserves for checkpoint bytes; the detailed check estimates model, covariance, sample, activation, and checkpoint memory. Both enforce the 80% CPU/GPU headroom guard. The pinned official loader uses `torch.load` without `map_location`, so CPU calibration rejects checkpoints whose `init_args.device` declares CUDA. Explicit saved CUDA indices must be visible and match the requested device. Keep `--device cuda` for the normal GPU-origin source run.

The calibrator exits before writing an artifact if a resource guard fails, the covariance exceeds its explicit budget, the source split is absent/invalid, or source paths/IDs/metadata do not meet the contract. It refuses to overwrite an existing artifact. Save the printed artifact SHA256 with the experiment record.

## Fresh fold-0 training

Confirm the official roots point to the existing Dataset501 raw and
preprocessed directories and that its `splits_final.json` already exists and
contains fold 0. The Trainer checks this file before any `get_dataloaders` or
split-generation path and fails closed if it is missing, changes, or overlaps
internally. The result root below is intentionally separate; choose a new
empty path for each experiment and do not redirect a previous run into it.

```bat
conda activate nnunet5090
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
set nnUNet_compile=false
set "nnUNet_results=C:\lijialin\models3d\nnUNet\nnUNet_results_ISLES508_CKPD_ES"
set "NNUNET_CKPD_SOURCE_CHECKPOINT=C:\path\to\selected-source-checkpoint.pth"
set "NNUNET_CKPD_CALIBRATION=C:\lijialin\models3d\nnUNet\ckpd-calibration\dataset508-fold0-r4-k4.pth"
nnUNetv2_train Dataset501_StrokeLesion 2d 0 ^
  -tr nnUNetTrainerCKPDTopK10EarlyStopping -p nnUNetPlans --val_best ^
  -pretrained_weights "%NNUNET_CKPD_SOURCE_CHECKPOINT%"
```

The custom Trainer validates the calibration only after nnU-Net has loaded the
pretrained weights, then installs the adapters and rebuilds an SGD optimizer
over trainable parameters only. The resulting optimizer retains initial LR
`0.01`, momentum `0.99`, Nesterov, weight decay `3e-5`, and the inherited
PolyLR schedule. Encoder convolution biases and normalization parameters remain
frozen; encoder modules are returned to evaluation mode after nnU-Net calls
`network.train()` each epoch. DDP and `torch.compile` fail explicitly.

## Resume, validation, and inference

Resume only the CKPD run in its same result root and with the same extension
source. The checkpoint contains merged public `network_weights` for the normal
nnU-Net predictor plus a separate training state with each selected frozen
base kernel, A/B factors, optimizer, scheduler, RNG, split/plans identity, and
early-stopping state. Resume reconstructs the adapter topology from that file;
it does not read calibration, Dataset508, or the original source checkpoint.

```bat
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
set nnUNet_compile=false
set "nnUNet_results=C:\lijialin\models3d\nnUNet\nnUNet_results_ISLES508_CKPD_ES"
set NNUNET_CKPD_SOURCE_CHECKPOINT=
set NNUNET_CKPD_CALIBRATION=
nnUNetv2_train Dataset501_StrokeLesion 2d 0 ^
  -tr nnUNetTrainerCKPDTopK10EarlyStopping -p nnUNetPlans --c --val_best
```

Run nnU-Net full-volume validation with the same Trainer and result root. Use both `--val --val_best` to score the early-stopping-selected checkpoint_best. `--val_best` alone enters training; use `--val` alone only when the protocol explicitly calls for checkpoint_final:

```bat
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
set nnUNet_compile=false
set "nnUNet_results=C:\lijialin\models3d\nnUNet\nnUNet_results_ISLES508_CKPD_ES"
nnUNetv2_train Dataset501_StrokeLesion 2d 0 ^
  -tr nnUNetTrainerCKPDTopK10EarlyStopping -p nnUNetPlans --val --val_best
```

Standard inference also uses the same extension discovery setting and merged
`network_weights`; the predictor constructs the baseline PlainConvUNet and
loads a strict, ordinary state dictionary with all registered aliases and all
segmentation heads:

```bat
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
set nnUNet_compile=false
set nnUNet_results=C:\lijialin\models3d\nnUNet\nnUNet_results_ISLES508_CKPD_ES
nnUNetv2_predict -i "C:\path\to\Dataset501\imagesTs" ^
  -o "C:\lijialin\models3d\nnUNet\Dataset501_CKPD_predictions" ^
  -d Dataset501_StrokeLesion -c 2d -f 0 ^
  -tr nnUNetTrainerCKPDTopK10EarlyStopping -p nnUNetPlans
```

The saved Python/NumPy/Torch RNG, optimizer, scheduler, model, epoch, and
early-stopping state are restored for the next update. nnU-Net's multiprocessing
augmentation workers and their queued batches/cursors are not serialized by the
official Trainer, so a resumed real-data run cannot claim a bitwise identical
patient/augmentation stream even though its saved training state is restored.

## Evidence boundary

Tiny local CPU tests exercise decomposition, bounded patch extraction, shared
Conv2d aliases, decoder-only trainability, merged inference loading, artifact
provenance rejection, and checkpoint next-update equivalence. They do not load
full Dataset501/508 data or a production checkpoint. Before a server run,
confirm GPU/RAM headroom and cohort independence; calibration and training are
still server-only. Record calibration as preflight evidence. Any Dataset501
fold-0 full-volume result is development evidence, not five-fold OOF or a
clinical-performance claim.
