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
after both files are written; a failed run removes its private staging folder.
The outputs are `summary.png` and UTF-8 `report.txt`. Full runs verify image,
GT and prediction headers for every metrics-covered case. The TXT records the
count. Only exact `image_reader_writer=SimpleITKIO` plans are supported.

The report is a **single fold 0 validation report**, not five-fold OOF or a
clinical conclusion. It reads the existing full-set case and summary metrics;
it never recomputes or changes them. Empty GT cases have no selected positive
slices. The PNG feature panels show `mean(abs(deepest encoder stage))` from
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
