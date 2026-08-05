# Standalone nnU-Net 2D Reproduction

This project is a pure-PyTorch foundation for reproducing the 2D `PlainConvUNet`
configuration of `Dataset501_StrokeLesion`. The standalone package does not
import `nnunetv2` or `dynamic_network_architectures` during normal use;
`oracle_capture.py` is the separate server-only, lazy-import exception.

Current progress: reference validation, environment inspection, JSON-driven
network construction, CPU model-shape tests, a read-only data pipeline, and
Dice-plus-cross-entropy deep-supervision loss modules, binary/OOF metrics,
optional caller-configured slice augmentation, a single explicit dry-run
optimization step, formal training/inference/evaluation entry points, and a
safe checkpoint interface are implemented. Formal run artifacts remain
`official_alignment_pending` until both required parity reports pass.

Run the checks from the worktree root (using the required environment):

```powershell
conda run -n newconda python standalone_nnunet2d/tools/inspect_environment.py
conda run -n newconda python standalone_nnunet2d/tools/inspect_reference.py
conda run -n newconda python standalone_nnunet2d/tools/inspect_dataset.py --raw-root C:\path\to\Dataset501_StrokeLesion
conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_documentation_contract.py -q
conda run -n newconda python -m pytest standalone_nnunet2d/tests -q
conda run -n newconda python -m py_compile standalone_nnunet2d/formal_train.py standalone_nnunet2d/predict.py standalone_nnunet2d/validate_cv.py standalone_nnunet2d/oracle_capture.py
```

The three external Dataset501 directories are strictly read-only. Do not copy
the NIfTI dataset, NNUNet outputs, large NPZ files, or model weights into this
repository; any future generated artifacts belong under `outputs/`.

`inspect_dataset.py` only checks paths unless a single `--case-id` is supplied.
It never scans all images. Data loading supports the read-only, on-demand slice
pipeline; formal training, prediction, and fold/OOF validation are explicit
commands documented below.

The loss module exposes foreground batch Dice and cross-entropy. Its
deep-supervision weights are caller-provided rather than asserted to match
unavailable official source defaults.

Metric helpers consume explicitly supplied binary masks and can aggregate
case-level OOF records. They do not create predictions or claim a reproduced
score; the bundled official 95-case Dice/IoU values are comparison baselines.

The data Dataset can opt into foreground-aware slice selection and synchronized
paired augmentation through explicit configuration. Both are disabled by
default. Formal training is explicit and all resulting artifacts remain pending
until the parity gate described below passes.

`engine.checkpoint.save_checkpoint` and `load_checkpoint` are caller-invoked
utilities, not a training loop. They only accept paths below
`standalone_nnunet2d/outputs/`, serialize a versioned model/optional-optimizer/
metadata payload, and never copy external model weights into the repository.

`engine.trainer.run_train_epoch` and `engine.validator.run_validation_epoch`
only consume caller-supplied tensor batches. They do not create a DataLoader,
schedule epochs, write checkpoints, or alter the disabled `train.py` command;
they are composable building blocks for a later, explicitly requested formal
run.

Before using a server, run the read-only preflight with explicit paths:

```powershell
conda run -n newconda python -m standalone_nnunet2d.dry_run `
  --raw-root C:\path\to\Dataset501_StrokeLesion `
  --preprocessed-root C:\path\to\Dataset501_StrokeLesion `
  --results-root C:\path\to\Dataset501_StrokeLesion `
  --device cuda
```

It prints JSON and exits 0 only if all directories, the requested device, and
the local reference plan are ready; it exits 2 otherwise. The command neither
creates directories nor reads cases, allocates a model, writes artifacts, or
starts training. Its `--run` flag is deliberately rejected.

`python -m standalone_nnunet2d.experiment` adds required `--output-root`,
`--fold 0..4`, and positive `--epochs` validation. Its explicit
`--confirm-run` path is a smoke run only; a smoke result is not an official
reproduction and remains `official_alignment_pending`.

## Documentation contract and result labels

The exact local verification commands are the documentation-contract test, the
full local test suite, and the four-file `py_compile` check shown above. A local
test pass, a smoke result, or an online `validation_dice` value is not an
official reproduction. Smoke results and online validation are not official
reproduction.

All formal training, prediction, fold-validation, and OOF artifacts are labeled
`official_alignment_pending` before alignment is established. The local parity
report also remains `official_alignment_pending` even when its comparison status
is `passed`. Only both a passed transform parity report and a passed inference
parity report permit a later artifact/result to be labeled `official_aligned`;
neither report alone is sufficient.

## Server oracle capture (server-only)

The server oracle capture command must run only in an environment where the
official `nnunetv2` installation is available. Do not run it in `newconda` and
do not treat its output as official until the matching local parity reports have
passed. The command supports `preprocess`, `sample`, `transform`,
`deep_supervision`, and `inference` and writes below the selected output root.

Capture a transform/preprocessing artifact on the server:

```powershell
conda run -n <server-env> python -m standalone_nnunet2d.oracle_capture `
  --mode preprocess `
  --raw-root C:\path\to\Dataset501_StrokeLesion `
  --preprocessed-root C:\path\to\Dataset501_StrokeLesion `
  --results-root C:\path\to\nnUNet_results `
  --output-root standalone_nnunet2d\outputs\oracle `
  --case-id case_0001 `
  --fold 0 `
  --seed 17
```

Capture the corresponding inference artifact on the server. The model folder
must point to the official trained model:

```powershell
conda run -n <server-env> python -m standalone_nnunet2d.oracle_capture `
  --mode inference `
  --raw-root C:\path\to\Dataset501_StrokeLesion `
  --preprocessed-root C:\path\to\Dataset501_StrokeLesion `
  --results-root C:\path\to\nnUNet_results `
  --output-root standalone_nnunet2d\outputs\oracle `
  --case-id case_0001 `
  --fold 0 `
  --seed 17 `
  --model-folder C:\path\to\nnUNet_results\Dataset501_StrokeLesion\... `
  --device cuda `
  --checkpoint-name checkpoint_best.pth
```

The server handoff order is: capture the transform artifact, capture the
inference artifact, run each local parity report, then run one local fold-0
prediction/validation export. No five-fold run starts before both parity reports
have status `passed`.

## Local parity reports

Run the transform parity report in `newconda` against the server artifact and a
standalone artifact produced from the same case, seed, and policy:

```powershell
conda run -n newconda python -m standalone_nnunet2d.tools.parity_report `
  --oracle-root standalone_nnunet2d\outputs\oracle\preprocess\case_0001 `
  --standalone-root standalone_nnunet2d\outputs\standalone\preprocess\case_0001 `
  --image-atol 1e-6 `
  --output standalone_nnunet2d\outputs\transform_parity_report.json
```

Run the inference parity report with the corresponding inference artifacts:

```powershell
conda run -n newconda python -m standalone_nnunet2d.tools.parity_report `
  --oracle-root standalone_nnunet2d\outputs\oracle\inference\case_0001 `
  --standalone-root standalone_nnunet2d\outputs\standalone\inference\case_0001 `
  --image-atol 1e-6 `
  --output standalone_nnunet2d\outputs\inference_parity_report.json
```

Both JSON reports must have `status: "passed"`. Their own `run_state` remains
`official_alignment_pending`; the pair is the gate that permits a later
`official_aligned` label.

## Fold-0 formal train, prediction, and validation

After the server artifacts and both local parity reports are available, run the
standalone fold-0 sequence in `newconda`. The formal train command requires the
explicit `--confirm-run` boundary and writes pending checkpoints:

```powershell
conda run -n newconda python standalone_nnunet2d\formal_train.py `
  --raw-root C:\path\to\Dataset501_StrokeLesion `
  --output-root standalone_nnunet2d\outputs\formal\fold_0 `
  --fold 0 `
  --device cuda:0 `
  --epochs 1000 `
  --confirm-run
```

Export the fold-0 validation cases from the best checkpoint. `--allow-pending`
is required because the checkpoint is not officially aligned yet:

```powershell
conda run -n newconda python standalone_nnunet2d\predict.py `
  --checkpoint standalone_nnunet2d\outputs\formal\fold_0\checkpoint_best.pth `
  --raw-root C:\path\to\Dataset501_StrokeLesion `
  --fold 0 `
  --output-root standalone_nnunet2d\outputs\formal\fold_0_predictions `
  --device cuda:0 `
  --allow-pending
```

Create the source-space fold-0 case report and prediction export:

```powershell
conda run -n newconda python standalone_nnunet2d\validate_cv.py fold `
  --checkpoint standalone_nnunet2d\outputs\formal\fold_0\checkpoint_best.pth `
  --raw-root C:\path\to\Dataset501_StrokeLesion `
  --fold 0 `
  --output-root standalone_nnunet2d\outputs\formal\crossval `
  --device cuda:0 `
  --allow-pending
```

This writes `fold_0_report.json` and `fold_0_case_metrics.csv` under the common
`crossval` directory, both still marked `official_alignment_pending`.

## Five-fold training and OOF sequence

Repeat the formal train, pending prediction, and pending fold validation
sequence for folds `1`, `2`, `3`, and `4`, changing only the fold number and the
training output directory. Keep all five validation reports in the same
`standalone_nnunet2d\outputs\formal\crossval` directory:

```powershell
conda run -n newconda python standalone_nnunet2d\formal_train.py --raw-root C:\path\to\Dataset501_StrokeLesion --output-root standalone_nnunet2d\outputs\formal\fold_<FOLD> --fold <FOLD> --device cuda:0 --epochs 1000 --confirm-run
conda run -n newconda python standalone_nnunet2d\predict.py --checkpoint standalone_nnunet2d\outputs\formal\fold_<FOLD>\checkpoint_best.pth --raw-root C:\path\to\Dataset501_StrokeLesion --fold <FOLD> --output-root standalone_nnunet2d\outputs\formal\fold_<FOLD>_predictions --device cuda:0 --allow-pending
conda run -n newconda python standalone_nnunet2d\validate_cv.py fold --checkpoint standalone_nnunet2d\outputs\formal\fold_<FOLD>\checkpoint_best.pth --raw-root C:\path\to\Dataset501_StrokeLesion --fold <FOLD> --output-root standalone_nnunet2d\outputs\formal\crossval --device cuda:0 --allow-pending
```

Replace `<FOLD>` with `1`, `2`, `3`, and `4` in order. After exactly five
reports exist, aggregate the 95 unique validation cases:

```powershell
conda run -n newconda python standalone_nnunet2d\validate_cv.py aggregate `
  --output-root standalone_nnunet2d\outputs\formal\crossval
```

The aggregate command requires `fold_0_report.json` through
`fold_4_report.json`, exactly 95 unique validation IDs, and writes
`oof_per_case_metrics.csv` plus `oof_summary.json`. The OOF summary remains
`official_alignment_pending` until both parity reports have passed.
