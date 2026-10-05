@echo off
setlocal
cd /d "%~dp0"
set "ROOT=C:\lijialin\models3d\nnUNet"
set "OUT=%ROOT%\batch_reports_dataset501_v1"
set "nnUNet_extTrainer=%~dp0nnunet_ext_trainers"
python batch_generate_result_reports.py ^
  --results-roots "%ROOT%\nnUNet_results" "%ROOT%\nnUNet_results_ISLES508_pretrained_ES" ^
  --images-dir "%ROOT%\nnUNet_raw\Dataset501_StrokeLesion\imagesTr" ^
  --labels-dir "%ROOT%\nnUNet_raw\Dataset501_StrokeLesion\labelsTr" ^
  --selection-json "%ROOT%\nnUNet_results\reports\comparison_fixed_cohort_v1\comparison_selection.json" ^
  --metrics-map "%~dp0batch_report_metrics_map.example.json" ^
  --output-dir "%OUT%" --device cuda --features %*
set "BATCH_RESULT=%ERRORLEVEL%"
echo Batch exit code: %BATCH_RESULT%
echo Output directory: %OUT%
exit /b %BATCH_RESULT%
