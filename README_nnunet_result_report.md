# Dataset501 fold 0 validation report

Run from this checkout on the server in Windows CMD. Replace the two raw-data
placeholders with the actual original DWI and ground-truth directories. The
images directory must contain one `CASE_0000.nii.gz` per validation case; the
labels and saved prediction directories use `CASE.nii.gz`. A single-channel
Dataset501 input is supported.

```bat
conda activate nnunet5090
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
set MODEL=C:\lijialin\models3d\nnUNet\nnUNet_results\Dataset501_StrokeLesion\nnUNetTrainerUPerNetTopK10EarlyStopping__nnUNetPlans__2d
python generate_nnunet_result_report.py --model-dir "%MODEL%" --fold 0 --images-dir "<ACTUAL_DWI_IMAGES_DIR>" --labels-dir "<ACTUAL_GT_LABELS_DIR>" --prediction-dir "%MODEL%\fold_0\validation" --metrics-dir "%MODEL%\fold_0\multi_metrics" --checkpoint "%MODEL%\fold_0\checkpoint_best.pth" --output-dir "<NEW_REPORT_OUTPUT_DIR>" --check
```

`--check` reads only metadata and filenames. It neither opens image voxels nor
loads a model. If the metadata checks pass, verify which checkpoint produced
the saved validation predictions from a training/validation log or other
auditable record. Replace `checkpoint_best.pth` if the evidence identifies
`checkpoint_final.pth`. Then run the same command without `--check` and add:

```bat
--prediction-checkpoint-declaration "USER DECLARED: <evidence file, command, checkpoint SHA256, validation time and TTA details>"
```

The script labels this statement as user-declared provenance; it cannot prove
the saved mask was produced by that checkpoint. Without it, a full report is
refused. Use a fresh output directory: existing targets and overlaps with any
source are refused. The output parent must exist. Complete output appears only
after all three files are written; an incomplete or failed run removes its private staging folder without leaving a report in the final output directory.
The outputs are `summary.png`, `feature_channels.png`, and UTF-8 `report.txt`. Full runs verify image,
GT and prediction headers for every metrics-covered case. The TXT records the
count. Only exact `image_reader_writer=SimpleITKIO` plans are supported.

The report is a **single fold 0 validation report**, not five-fold OOF or a
clinical conclusion. It reads the existing full-set case and summary metrics;
it never recomputes or changes them. Empty GT cases have no selected positive
slices. The PNG feature magnitude and DWI overlay panels show `mean(abs(deepest encoder stage))` from
fresh raw-image preprocessing and no mirror TTA. Each patch is upsampled to
its own sliding window and fused with the nnU-Net Gaussian weights, then
mapped through inverse padding, resampling, cropping and transpose. These
diagnostic features are not the saved prediction's internal features, lesion
probabilities, attention, or explanations. The heatmap display is normalized
per slice and cannot compare absolute feature strength across cases.

## SERVER VALIDATION PENDING

- Confirm the server's installed `nnunetv2`, external Trainer source, plans and
  checkpoint load strictly with the script; local synthetic tests do not prove
  server compatibility.
- Establish saved prediction provenance: best/final, exact checkpoint hash,
  validation command, mirroring/TTA, tile step and Gaussian settings. The
  diagnostic feature run intentionally disables TTA; document any difference.
- Verify feature mapping on a known server case: original image/GT/prediction
  geometry, crop, transpose, resampling, sliding window, padding, and exported
  feature position. A report generated before this check remains provisional.
- Confirm source `case_metrics.csv` and `summary_metrics.json` cover the same
  full validation set, including F2 mode and missing-value meanings.

No local real checkpoint, raw patient image, or large model was loaded during
development. No training or validation prediction is run by this command.

## Visual revision and server review

The summary shows the encoder stages, selected feature connections, PPM, FPN and
output path from the loaded model attributes. High and low Dice groups each show
three cases with up to three GT-positive slices. Green is TP, red FP, and blue
FN in the display overlay. The summary's inferno scale uses per-slice 1st-99th
percentile normalization; DWI+feature uses alpha 0.48. This scale is for display
only and is not comparable across cases. The native-channel sheet
(feature_channels.png) displays up to eight uniformly spaced channel IDs at the
unaltered low resolution, with nearest pixel display. It represents one local
preprocessed window selected as the first enumerated window containing the
first displayed slice center, without GT-dependent channel or window selection.
The TXT records window index, preprocessed bounds, actual sliding-window count,
feature shape and channel IDs. Nonidentity slice-axis transpose or slice-axis
resampling is rejected for this localization rather than silently mismapped.

Each native 4×4 channel is displayed using its own exact minimum and maximum:
`(value - min) / (max - min)` in `[0,1]`. A constant channel displays uniformly
at zero. All channels share the same coolwarm color scale and colorbar; blue/red
indicate only relative low/high values, not negative/positive values. Native
colors cannot compare absolute activation between channels or cases and are not
lesion probabilities. This differs from the summary's DWI and feature-magnitude
panels, which each use per-slice 1st–99th percentile normalization to `[0,1]`
and display constant maps at zero.

On the server, first run the command above with --check. Confirm the metrics
coverage, checkpoint path and a new output directory. Then run the same
command without --check, adding the provenance declaration shown above.
Inspect the generated structure against the server's loaded encoder and decoder;
verify selected stage channels, relative sizes, PPM scales, FPN width and output
shape. Compare one case's original slice, crop/transpose, preprocessing window
bounds and native-channel window index. Check DWI, GT, saved prediction and
feature spatial alignment, plus the saved prediction's separate provenance.
The existing report directory remains read-only. These server steps have not
been run locally.
