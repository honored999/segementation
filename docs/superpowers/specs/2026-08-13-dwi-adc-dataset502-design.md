# Dataset502/503 multimodal nnU-Net input design

## Goal

Create independently auditable two- and three-channel nnU-Net datasets from the
existing single-channel Dataset501 DWI dataset and the original CI-1 DICOM
series. The new dataset tests whether ADC improves 2D stroke-lesion
segmentation without modifying Dataset501 or its existing experiments.

## Modalities and output datasets

The builder supports the ordered modality selections `DWI ADC` and `DWI ADC
FLAIR`. It reads each modality only from the matching DICOM series in the
manifest's source directory. The selected list determines the nnU-Net channel
numbers and the target dataset's `channel_names`.

```text
Dataset502_StrokeLesion_DWI_ADC
  _0000 = DWI, _0001 = ADC

Dataset503_StrokeLesion_DWI_ADC_FLAIR
  _0000 = DWI, _0001 = ADC, _0002 = FLAIR
```

Each dataset is built and planned separately. A two-channel run cannot later be
converted in place into a three-channel run because plans, preprocessed arrays,
checkpoint shapes and inference inputs depend on channel count.

## Data contract

The source Dataset501 contains 95 DWI cases and binary labels. Its
`conversion_manifest.tsv` maps every case ID to a CI-1 patient, timepoint and
source DICOM directory. Dataset502 retains those exact case IDs and labels:

```text
caseNNN_0000.nii.gz = copied Dataset501 DWI
caseNNN_0001.nii.gz = ADC read from the manifest's DICOM directory
caseNNN_0002.nii.gz = FLAIR read from the manifest's DICOM directory, when selected
caseNNN.nii.gz      = copied Dataset501 label
```

The DICOM reader selects only a series whose description identifies the
requested modality. It never uses a CI-1 file merely because its name contains
`ADC` or `FLAIR`: these files can be masks or inconsistent historical exports.

## Audit before build

The construction command is audit-only by default. For every manifest row it
must: resolve existing DWI and label paths; read every selected non-DWI DICOM
series; resample it to DWI as reference with linear interpolation; and verify
exact output size, spacing, origin and direction against DWI. It writes a
TSV/JSON report with source paths, series descriptions, original and reference
geometry, and pass/fail status.

The audit passes only when there are exactly 95 unique case IDs, every source
file exists, every DICOM directory yields one readable series for every
selected modality, all resampled images match their DWI reference geometry,
and no duplicate output path exists. A failure prevents build.

## Explicit build

Only `--build` after a passing audit may write Dataset502. The output root must
not already exist, avoiding overwrite or partial reuse. Build copies each DWI
and label without altering pixels, writes each audited resampled modality to
its assigned channel number, and writes matching dataset.json channel names,
binary labels, `numTraining: 95`, and `SimpleITKIO`. It also writes the audit
report and a new provenance manifest containing source and output SHA256
digests.

## Follow-on server workflow

Each Dataset502/503 target must receive a fresh fingerprint, plans and
preprocessing. Its split file is copied from Dataset501 only after checking it
contains exactly the same 95 case IDs. The first training screen uses a 2D
default Trainer on fold 0; if full-volume 19-case validation is convincingly
better, test the corresponding TopK10 variant separately. New results are
multimodal experiments, not Dataset501/standalone official-aligned
reproductions.

## Tests

Unit tests must establish that an incomplete manifest audit fails before any
build action, that a valid synthetic two-case audit creates the two-channel
dataset only with `--build`, that channel 0 and labels are byte-identical
copies, that ADC and FLAIR output adopt DWI geometry, and that two- and
three-channel dataset JSON files accurately declare their channels. A focused
test must be run and observed failing before the builder implementation is
written.
