@echo off
setlocal
call conda activate nnunet5090
if errorlevel 1 exit /b 1
set "PYTHONHASHSEED=0"
set "CONFIG=%~dp0experiment_config.cv70.json"
if not "%~1"=="" set "CONFIG=%~1"
for %%F in (0 1 2 3 4) do (
    python "%~dp0run_experiment.py" --config "%CONFIG%" --cv70-fold %%F --seed 0 --check-only
    if errorlevel 1 exit /b 1
)
for %%F in (0 1 2 3 4) do (
    python "%~dp0run_experiment.py" --config "%CONFIG%" --cv70-fold %%F --seed 0
    if errorlevel 1 exit /b 1
)
python "%~dp0cv70.py" aggregate --config "%CONFIG%" --seed 0
exit /b %ERRORLEVEL%
