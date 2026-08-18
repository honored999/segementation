# Server project structure

## 1. Purpose

This document is an evidence-bounded map of the server-side acute ischemic
stroke lesion segmentation environment. It records confirmed paths and
historical path references so that local development can produce scripts that
are copied to the server without assuming that the local checkout contains the
server data.

It is intended to:

- identify resources that exist only on the server;
- help local Codex/agents write portable server commands;
- avoid treating a locally missing dataset as a server-side absence; and
- avoid hard-coding a local checkout path into a server script.

Status terminology used below:

- **Verified from local reference copy**: supported by a tracked local
  reference/configuration file.
- **Known from historical server output; not locally verified**: seen in a
  recorded server command, log, or user-supplied result, but not accessible
  from this checkout.
- **Pending server verification**: not established by available local evidence.

## 2. Environment overview

- Project: acute ischemic stroke lesion segmentation. **Known from historical
  server output; not locally verified.**
- Framework: nnU-Net v2. The recorded server runtime was `nnunetv2==2.8.1` in
  Conda environment `nnunet5090`. **Known from historical server output; not
  locally verified.**
- Primary baseline dataset: `Dataset501_StrokeLesion`. **Verified from local
  reference copy and historical server paths.**
- Primary experimental configuration: 2D nnU-Net. **Verified from local
  reference/planning material.**
- Baseline input: DWI only. **Verified from local reference/planning material.**
- Raw data, preprocessed arrays, checkpoints, and complete training outputs are
  server resources. **Known from historical server output; not locally
  verified.**

## 3. Known server directory layout

The following is a map of paths referenced by historical server commands. It
is not a directory listing. Ellipses mean unverified contents.

```text
C:\lijialin\models3d\nnUNet\                         [historical server path]
├── nnUNet_raw\
│   └── Dataset501_StrokeLesion\                       [historical path]
│       ├── imagesTr\                                  [required training-image layout]
│       ├── labelsTr\                                  [required training-label layout]
│       └── dataset.json                               [local reference exists]
│
├── nnUNet_preprocessed\
│   └── Dataset501_StrokeLesion\                       [historical path]
│       ├── nnUNetPlans.json                           [historical path]
│       ├── dataset.json                               [nnU-Net preprocessing artifact; pending server check]
│       ├── splits_final.json                          [historical path; local reference exists]
│       └── ...
│
└── nnUNet_results\                                   [nnU-Net result root; pending server check]
    └── Dataset501_StrokeLesion\                      [pending server check]
        └── <Trainer>__<Plans>__2d\                   [nnU-Net convention; exact folders pending]
```

Trainer names such as `nnUNetTrainerTopK10`,
`nnUNetTrainerTopK20`, and `nnUNetTrainerForeground50` are present in the
local custom-Trainer source. Their corresponding server result directories are
**historical references / pending server verification**, not locally observed
directories. The ResEnc experiment uses an `nnUNetResEncUNetMPlans` plans
identifier; its exact result-folder name must be checked on the server.

## 4. Dataset501

`Dataset501_StrokeLesion` is the DWI-only baseline dataset.

| Item | Status and source |
| --- | --- |
| Training cohort | 95 cases. **Verified from local reference copy** (`standalone_nnunet2d/reference/dataset.json`) and historical conversion-manifest evidence. |
| Images | `imagesTr` with DWI baseline input. **Verified from local reference/planning material.** |
| Labels | `labelsTr` with one lesion label per case. **Verified from local reference/planning material.** |
| Dataset metadata | `dataset.json`. **Verified from local reference copy.** |
| Split definition | Fixed `splits_final.json`, five patient-level folds. **Verified from local reference copy / historical training output.** |
| `imagesTs` | **Pending server verification.** No local evidence in scope establishes whether it is used. |
| Raw location | `C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion`. **Known from historical server output; not locally verified.** |
| Preprocessed location | `C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion`. **Known from historical server output; not locally verified.** |
| Results location | Under the server `nnUNet_results` root is expected by nnU-Net, but the exact Dataset501 location is **pending server verification**. |

## 5. Training result structure

The expected nnU-Net result layout is:

```text
Dataset501_StrokeLesion/
└── <Trainer>__<Plans>__2d/
    ├── fold_0/
    ├── fold_1/
    ├── fold_2/
    ├── fold_3/
    ├── fold_4/
    └── ...
```

This is a **nnU-Net convention**, not proof that all five server folders or
their contents exist. For each fold, later coarse-to-fine work needs to locate
the checkpoint, validation prediction, validation NIfTI, validation summary or
metrics, and training log. The concrete filenames and retention policy are
**pending server verification**; this document deliberately does not invent
them.

## 6. Known Trainer variants

| Trainer / configuration | Purpose supported by local source or historical result | Status | Server location |
| --- | --- | --- | --- |
| `nnUNetTrainer` | Default DWI baseline | Historical fold-0 result exists | Exact result path pending server verification |
| `nnUNetTrainerTopK10` | Top-k hard-pixel loss experiment | Local Trainer source; historical fold-0 result exists | Exact result path pending server verification |
| `nnUNetTrainerTopK20` | Top-k loss experiment | Local Trainer source; historical fold-0 result exists | Exact result path pending server verification |
| `nnUNetTrainerForeground50` | 50% foreground-oversampling experiment | Local Trainer source; historical fold-0 result exists | Exact result path pending server verification |
| ResEncUNetM plans | Residual-encoder architecture experiment | Historical fold-0 result exists | Exact result path pending server verification |

No table entry above asserts that a complete five-fold run exists. In
particular, fold-0 screening must not be represented as a five-fold OOF result.

## 7. Evaluation and analysis outputs

Historical analysis uses experiment-specific evaluation outputs, including the
following documented layout:

```text
trainer_eval/
└── <experiment>_stats/
    ├── case_metrics.csv
    ├── slice_metrics.csv
    └── overall_metrics.json
```

This is a **historical evaluation-output convention; current server directory
existence is not locally verifiable**. The available project evaluation
material distinguishes the following metrics:

- full-volume foreground Dice per patient, followed by a case-macro mean;
- global voxel Dice, pooled over all voxels;
- slice Dice; and
- positive-GT slice Dice, restricted to slices whose ground truth is non-empty.

For formal nnU-Net comparison, use reconstructed complete 3D volume
predictions, argmax post-processing, foreground class 1, both-empty Dice of 1,
one-empty Dice of 0, and equal weighting of the 95 cases. The final five-fold
field is `oof_summary.json -> foreground_mean -> Dice`; an online training
`Mean Validation Dice` is not the final comparison metric.

## 8. Local repository vs server

### Local checkout currently contains

The local project is expected to contain the standalone implementation,
reference dataset/split metadata, custom Trainer source, evaluation code, and
partial historical experiment information. The exact set of uncommitted local
files is intentionally not asserted here.

### Local checkout currently does not provide for this documentation task

- the full server `Dataset501_StrokeLesion/imagesTr` data;
- the full server `Dataset501_StrokeLesion/labelsTr` data;
- the full server `nnUNet_preprocessed` tree;
- a complete five-fold server checkpoint set;
- complete five-fold OOF prediction NIfTI files; or
- the server Conda/Python/CUDA runtime itself.

**Local absence must not be interpreted as server absence.**

## 9. Coarse-to-fine planned resources

The following is a plan, not an existing server structure:

```text
Stage 1: Dataset501 default DWI 2D nnU-Net
        ↓
5-fold OOF validation predictions
        ↓
Stage 2: ROI generation
        ↓
Dataset504_StrokeLesion_CoarseToFine
        ↓
default 2D nnU-Net
        ↓
ROI prediction
        ↓
restore to original volume
        ↓
full-volume evaluation
```

- `Dataset504_StrokeLesion_CoarseToFine` is **planned / not yet created**.
- Stage-2 ROIs must be generated from Stage-1 OOF predictions.
- Ground truth must not be used for formal ROI localization.

## 10. Paths that must remain configurable

Future scripts must accept these through CLI arguments, nnU-Net environment
variables, or a configuration file rather than hard-coding local paths:

- `nnUNet_raw`;
- `nnUNet_preprocessed`;
- `nnUNet_results`;
- Dataset501 location;
- Stage-1 OOF prediction directory;
- Dataset504 output location;
- restored-prediction output directory; and
- evaluation-output directory.

## 11. Verification checklist for next server session

- [ ] Confirm the actual server nnU-Net root.
- [ ] Confirm the Dataset501 raw path.
- [ ] Confirm the Dataset501 preprocessed path.
- [ ] Confirm the Dataset501 results path.
- [ ] Confirm whether default-Trainer checkpoints are complete for all five folds.
- [ ] Confirm whether default-Trainer validation NIfTI files were saved.
- [ ] Confirm whether all 95 OOF predictions are available.
- [ ] Confirm TopK10 five-fold status.
- [ ] Confirm TopK20 five-fold status.
- [ ] Confirm Foreground50 five-fold status.
- [ ] Confirm ResEnc five-fold status.
- [ ] Confirm the `trainer_eval` output directory and artifact names.
- [ ] Confirm the server Python and Conda environment.
- [ ] Confirm the server GPU and CUDA environment.
