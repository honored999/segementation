@echo off
setlocal
call conda activate nnunet5090
if errorlevel 1 exit /b 1
python "%~dp0run_experiment.py" --arm A --seed 0 --check-only
if errorlevel 1 exit /b 1
python "%~dp0run_experiment.py" --arm B --seed 0 --check-only
exit /b %ERRORLEVEL%