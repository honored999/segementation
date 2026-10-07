@echo off
setlocal
cd /d "%~dp0"
set "ROOT=C:\lijialin\models3d\nnUNet"
python batch_generate_result_reports.py ^
  --features-only --final-h2-only --use-current-h2-checkpoints ^
  --existing-summary "%ROOT%\batch_reports_dataset501_v2\batch_summary.csv" ^
  --images-dir "%ROOT%\nnUNet_raw\Dataset501_StrokeLesion\imagesTr" ^
  --labels-dir "%ROOT%\nnUNet_raw\Dataset501_StrokeLesion\labelsTr" ^
  --selection-json "%ROOT%\nnUNet_results\reports\comparison_fixed_cohort_v1\comparison_selection.json" ^
  --trainer-extension-map "%~dp0batch_report_trainer_extensions.example.json" ^
  --output-dir "%ROOT%\batch_reports_dataset501_v2_features_final_v1" ^
  --device cuda %*
set "BACKFILL_RESULT=%ERRORLEVEL%"
echo Feature backfill exit code: %BACKFILL_RESULT%
exit /b %BACKFILL_RESULT%
