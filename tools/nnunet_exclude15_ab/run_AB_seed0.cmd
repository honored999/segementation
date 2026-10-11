@echo off
setlocal
call conda activate nnunet5090
if errorlevel 1 exit /b 1
set "PYTHONHASHSEED=0"
set "SEED=0"
python "%~dp0run_experiment.py" --arm A --seed %SEED% --check-only
if errorlevel 1 exit /b 1
python "%~dp0run_experiment.py" --arm B --seed %SEED% --check-only
if errorlevel 1 exit /b 1
python "%~dp0run_experiment.py" --arm A --seed %SEED%
if errorlevel 1 exit /b 1
python "%~dp0run_experiment.py" --arm B --seed %SEED%
if errorlevel 1 exit /b 1
python "%~dp0compare_AB.py" %SEED%
exit /b %ERRORLEVEL%