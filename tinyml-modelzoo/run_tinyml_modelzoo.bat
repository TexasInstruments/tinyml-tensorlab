@echo off
REM Tiny ML ModelZoo Training Wrapper for Windows
REM Delegates training to tinyml-modelmaker
REM
REM Usage:
REM   run_tinyml_modelzoo.bat examples\hello_world\config.yaml
REM   run_tinyml_modelzoo.bat C:\path\to\config.yaml

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM ModelMaker location (sibling directory)
set "MODELMAKER_DIR=%SCRIPT_DIR%\..\tinyml-modelmaker"
set "RUN_SCRIPT=%MODELMAKER_DIR%\tinyml_modelmaker\run_tinyml_modelmaker.py"

REM Check if modelmaker exists
if not exist "%RUN_SCRIPT%" (
    echo Error: ModelMaker not found at %RUN_SCRIPT%
    echo Make sure tinyml-modelmaker is installed alongside tinyml-modelzoo
    exit /b 1
)

REM Check arguments
if "%~1"=="" (
    echo Tiny ML ModelZoo Training Wrapper
    echo.
    echo Usage: %~nx0 ^<config_file^> [additional_args...]
    echo.
    echo Examples:
    echo   %~nx0 examples\hello_world\config.yaml
    echo   %~nx0 examples\motor_bearing_fault\config.yaml
    echo.
    echo Available example configs:
    if exist "%SCRIPT_DIR%\examples" (
        for /r "%SCRIPT_DIR%\examples" %%f in (*.yaml) do (
            set "filepath=%%f"
            set "relpath=!filepath:%SCRIPT_DIR%\=!"
            echo   !relpath!
        )
    )
    exit /b 1
)

set "CONFIG_FILE=%~1"

REM Check if path is absolute (starts with drive letter or UNC)
echo %CONFIG_FILE% | findstr /r "^[A-Za-z]:" >nul 2>&1
if errorlevel 1 (
    echo %CONFIG_FILE% | findstr /r "^\\\\" >nul 2>&1
    if errorlevel 1 (
        REM Relative path - check if it exists relative to current dir
        if exist "%CONFIG_FILE%" (
            set "CONFIG_FILE=%CD%\%CONFIG_FILE%"
        ) else if exist "%SCRIPT_DIR%\%CONFIG_FILE%" (
            REM Check if relative to script directory
            set "CONFIG_FILE=%SCRIPT_DIR%\%CONFIG_FILE%"
        ) else (
            echo Error: Config file not found: %CONFIG_FILE%
            exit /b 1
        )
    )
)

REM Verify config exists
if not exist "%CONFIG_FILE%" (
    echo Error: Config file not found: %CONFIG_FILE%
    exit /b 1
)

echo Tiny ML ModelZoo Training
echo ========================================
echo Config:      %CONFIG_FILE%
echo ModelMaker:  %RUN_SCRIPT%
echo ========================================
echo.

REM Run training via modelmaker
cd /d "%MODELMAKER_DIR%"
python "%RUN_SCRIPT%" "%CONFIG_FILE%" %*
