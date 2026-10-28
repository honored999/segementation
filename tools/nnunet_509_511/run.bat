@echo off
setlocal
if /I "%CONDA_DEFAULT_ENV%"=="nnunet5090" goto ready
if not exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    echo ERROR: Required activate.bat is missing.
    exit /b 1
)
call "C:\ProgramData\anaconda3\Scripts\activate.bat" nnunet5090
if errorlevel 1 (
    echo ERROR: Cannot activate nnunet5090.
    exit /b 1
)
:ready
if /I not "%CONDA_DEFAULT_ENV%"=="nnunet5090" (
    echo ERROR: nnunet5090 is not active.
    exit /b 1
)
python "%~dp0workflow.py" %*
set "WORKFLOW_EXIT=%ERRORLEVEL%"
endlocal & exit /b %WORKFLOW_EXIT%
