# GenSeg 3 mm DWI target comparison (Windows CMD)

This is an exploratory generator-image comparison. It does not train nnU-Net, approve synthetic samples, replace the original Dataset501, or establish a segmentation benefit.

## Verified original run

- Original code commit: 909df3ec131d9ac323576428e5b206a05f4fd8f6.
- Original generator: C:\lijialin\genseg30_derived_trial01\generator_5000steps_seed20261014_attempt02.
- Dataset501 original preprocessed data: C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans_2d.
- Fold0: 76 train / 19 validation. Source split, reference split membership, plans and dataset hashes match the original run.
- The working reference split hash is the expected CRLF hash. Git LF blob has a different expected hash; neither split needs rewriting. Preserve the existing modified-file status.
- Three train cases were checked: Dataset501 and Dataset503 DWI files/voxel arrays match exactly. New preparation nevertheless reads original Dataset501 train DWI directly.

## Deliberate target transformation

Brain masks are obtained from original native DWI with HD-BET 2.0.1, default TTA, followed by physical 3 mm in-plane expansion. No through-plane expansion. A predicted empty native plane retains the entire original plane, including extracranial signal. This retention mask is not an anatomical brain ground truth.

The mask is mapped using original reader geometry, transpose, crop bbox and segmentation resampling to the existing nnU-Net preprocessed grid. The generator target is a copy of the original normalized float image, with outside-mask values set to the ORIGINAL generator's training-global minimum. Original normalization, crop, label values, -1 semantics and slice counts remain unchanged. Existing generator min/max scaling is then calculated on the transformed targets and recorded. This is preprocessed-target masking, not raw-zero extraction followed by a newly fitted nnU-Net preprocessing pipeline.

The generator condition remains the lesion mask only. Brain extraction does not resolve the missing anatomical conditioning of empty lesion masks; generation quality may still be poor.

## Files and transfer

The delivery adds only experiments/genseg3mm/** and this runbook. It does not overwrite experiments/genseg30, Trainer, upstream reference, splits or model weights. Use the original nnunet5090 environment; no new dependency installation is part of this trial. The module reuses the existing pinned GenSeg30 implementation.

Extract the reviewed delivery ZIP into C:\lijialin\segementation\.worktrees\genseg-reference. Do not use forced extraction over a previously modified genseg3mm package. If a previous version exists, preserve it and reconcile explicitly.

## Command setup

```bat
conda activate nnunet5090
set "REPO=C:\lijialin\segementation\.worktrees\genseg-reference"
set "nnUNet_raw=C:\lijialin\models3d\nnUNet\nnUNet_raw"
set "nnUNet_preprocessed=C:\lijialin\models3d\nnUNet\nnUNet_preprocessed"
set "SOURCE=%nnUNet_preprocessed%\Dataset501_StrokeLesion\nnUNetPlans_2d"
set "BASELINE=C:\lijialin\genseg30_derived_trial01\generator_5000steps_seed20261014_attempt02"
set "TRIAL=C:\lijialin\genseg3mm_trial01"
set "PYTHONPATH=%REPO%;%PYTHONPATH%"
cd /d "%REPO%"
python -m experiments.genseg3mm.cli --help
```

Choose another fresh TRIAL name if prior outputs exist. No resume or overwrite is assumed. GPU identifiers below are physical nvidia-smi indices. A selected GPU is bound by UUID so resource checks and actual inference/training use the same device. CPU/RAM/GPU/VRAM need headroom; prior success on GPU0 does not authorize a busy GPU1.

## Execution sequence

Keep all outputs outside the repository, raw, preprocessed and original-generator directories. The following argument names match the new CLI. GPU1 is a suggested selection based on the earlier trial; the resource guard checks its current state.

First, read-only compatibility and resource checks:

```bat
python -m experiments.genseg3mm.cli prepare --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --output "%TRIAL%\targets" --gpu 1 --check-only
```

Only if that prints CHECK PASS, prepare all76 training cases:

```bat
python -m experiments.genseg3mm.cli prepare --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --output "%TRIAL%\targets" --gpu 1
```

Record the exact `manifest_sha256` printed by the completed prepare command. Set it below; do not recompute a replacement after a tampering error. This is an integrity receipt, not a quality approval.

```bat
set "TARGET_SHA=REPLACE_WITH_64_HEX_CHARACTERS_PRINTED_BY_PREPARE"
python -m experiments.genseg3mm.cli train --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TRIAL%\targets" --targets-sha256 "%TARGET_SHA%" --output "%TRIAL%\generator_5000steps_seed20261014" --gpu 1
```

After completion at5000 actual updates:

```bat
python -m experiments.genseg3mm.cli preview --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TRIAL%\targets" --targets-sha256 "%TARGET_SHA%" --run "%TRIAL%\generator_5000steps_seed20261014" --output "%TRIAL%\preview" --gpu 1
```

Copy the final `%TRIAL%\preview` directory back for visual review. Retain generator/target manifests and logs in their original locations, as later validation binds those paths and identities.

- Preparation processes only fixed training patients, deriving and mapping retention masks. It never uses labels to repair masks.
- Training starts from the same seed/architecture and reuses recorded baseline settings, including max_steps=5000 and epochs ceiling200. Do not load old weights and claim retraining on the new targets.
- Preview uses identical training-only lesion conditions for both old and new generators. Real source, transformed target, old output, new output and lesion overlay are distinct panels. No heldout images or quality assertions are used to select a model.
- If interrupted, retain the aborted directory/logs and select a new output. Do not manually fabricate completion markers or edit hashes.

## Reading the images

Check actual brain anatomy, repeated stripes on background conditions, image-label lesion correspondence and fabricated bright regions outside labels. A visually appealing image alone is insufficient. The same-patient derivatives are not new independent patients. This task does not create a segmentation-training pool or remove its human quality gate.

Formal whole-volume performance and five-fold OOF claims remain out of scope. Case005 was a brain-extraction development example and is not used for fold0 generator fitting or its training-only preview.

## Evidence boundary

Local checks cover small synthetic geometry/provenance/adapter behavior and static CLI availability. Real target-computer HD-BET preparation, full-size GPU memory, actual 5000-update training, generated-image quality and segmentation performance require execution there.

## Interpretation of background-only conditions

The current generator is deterministic at evaluation with dropout disabled. Identical all-zero lesion conditions of identical padded shape must produce identical images for a fixed checkpoint. Several source slices with that condition cannot be expected to produce different anatomy. The preview includes distinct background source slices where available to expose this limitation; improvement in skull/background rendering is a separate question from anatomical diversity. This trial does not add slice-position, brain-mask or image conditioning.

