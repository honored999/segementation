@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Editable server settings. Requires conda on PATH (or set CONDA_BAT).
set "ENV_NAME=nnunet5090"
set "CONDA_BAT=conda.bat"
set "PHYSICAL_GPU=1"
set "nnUNet_n_proc_DA=2"
set "ISLES_RAW=C:\lijialin\models3d\nnUNet\nnUNet_raw"
set "ISLES_PP=C:\lijialin\models3d\nnUNet\nnUNet_preprocessed"
set "HDBET_ROOT=C:\lijialin\HD-BET_outputs\nnunet_hdbet3mm_isles508_finetune_fold0_trial01"
set "ORIGINAL_RAW=C:\lijialin\models3d\nnUNet\nnUNet_raw"
set "ORIGINAL_PP=C:\lijialin\models3d\nnUNet\nnUNet_preprocessed"
set "OUTPUT_PARENT=C:\lijialin\HD-BET_outputs"
set "REPO=%~dp0"
set "STAMP="
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "STAMP=%%T"
if not defined STAMP goto fail
set "ROOT=%OUTPUT_PARENT%\plainconv_drop_last_ds_topk10_pipeline_%STAMP%"
set "HELPER=%REPO%nnunet_ext_trainers\prepare_plainconv_depth_pipeline.py"
python "%HELPER%" prepare --root "%ROOT%" --source-raw "%ISLES_RAW%" --source-pp "%ISLES_PP%" --target-raw "%HDBET_ROOT%\nnUNet_raw" --target-pp "%HDBET_ROOT%\nnUNet_preprocessed" --original-raw "%ORIGINAL_RAW%" --original-pp "%ORIGINAL_PP%" --gpu %PHYSICAL_GPU%
if errorlevel 1 goto fail
set "CUDA_VISIBLE_DEVICES="
set /p CUDA_VISIBLE_DEVICES=<"%ROOT%\gpu_uuid.txt"
if errorlevel 1 goto fail
if not defined CUDA_VISIBLE_DEVICES goto fail
set "nnUNet_results=%ROOT%\nnUNet_results"
set "nnUNet_extTrainer=%REPO%nnunet_ext_trainers"
set "TRAINER=nnUNetTrainerPlainConvDepthTopK10EarlyStopping"
set "PLANS=nnUNetPlansDropLastDS"
set "CONFIG=2d_drop_last_ds"
set "RUN=%TRAINER%__%PLANS%__%CONFIG%"
set "SOURCE_FOLD=%nnUNet_results%\Dataset508_ISLES2022DWI\%RUN%\fold_0"
set "TARGET_FOLD=%nnUNet_results%\Dataset501_StrokeLesion\%RUN%\fold_0"
set "nnUNet_raw=%ISLES_RAW%"
set "nnUNet_preprocessed=%ISLES_PP%"
python "%HELPER%" guard --gpu %PHYSICAL_GPU%
if errorlevel 1 goto fail
call nnUNetv2_train Dataset508_ISLES2022DWI %CONFIG% 0 -tr %TRAINER% -p %PLANS% --val_best
if errorlevel 1 goto fail
if not exist "%SOURCE_FOLD%\checkpoint_best.pth" goto fail
set "nnUNet_raw=%HDBET_ROOT%\nnUNet_raw"
set "nnUNet_preprocessed=%HDBET_ROOT%\nnUNet_preprocessed"
python "%HELPER%" guard --gpu %PHYSICAL_GPU%
if errorlevel 1 goto fail
call nnUNetv2_train Dataset501_StrokeLesion %CONFIG% 0 -tr %TRAINER% -p %PLANS% -pretrained_weights "%SOURCE_FOLD%\checkpoint_best.pth" --val_best
if errorlevel 1 goto fail
if not exist "%TARGET_FOLD%\checkpoint_best.pth" goto fail
python "%HELPER%" predictions --split "%ISLES_PP%\Dataset508_ISLES2022DWI\splits_final.json" --pred "%SOURCE_FOLD%\validation"
if errorlevel 1 goto fail
python "%HELPER%" predictions --split "%HDBET_ROOT%\nnUNet_preprocessed\Dataset501_StrokeLesion\splits_final.json" --pred "%TARGET_FOLD%\validation"
if errorlevel 1 goto fail
python "%HELPER%" guard --gpu %PHYSICAL_GPU% --cpu-only
if errorlevel 1 goto fail
python "%REPO%evaluate_segmentation_metrics.py" --pred-dir "%SOURCE_FOLD%\validation" --gt-dir "%ISLES_RAW%\Dataset508_ISLES2022DWI\labelsTr" --output-dir "%ROOT%\evaluation_isles508_best_fold0" --f2-mode paper
if errorlevel 1 goto fail
python "%HELPER%" guard --gpu %PHYSICAL_GPU% --cpu-only
if errorlevel 1 goto fail
python "%REPO%evaluate_segmentation_metrics.py" --pred-dir "%TARGET_FOLD%\validation" --gt-dir "%ORIGINAL_RAW%\Dataset501_StrokeLesion\labelsTr" --output-dir "%ROOT%\evaluation_hdbet501_best_fold0" --f2-mode paper
if errorlevel 1 goto fail
python "%HELPER%" finish --root "%ROOT%"
if errorlevel 1 goto fail
pause
exit /b 0
:fail
echo FAILED. Existing outputs are preserved. No automatic resume. Root: %ROOT%
pause
exit /b 1
