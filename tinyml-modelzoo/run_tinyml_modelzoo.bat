@echo off
REM Tiny ML ModelZoo Training Wrapper for Windows
REM Delegates training to tinyml-modelmaker, installed as a Python package
REM (wheel or editable install) - it does NOT need to be cloned as a sibling
REM directory.
REM
REM Usage:
REM   run_tinyml_modelzoo.bat examples\hello_world\config.yaml
REM   run_tinyml_modelzoo.bat C:\path\to\config.yaml

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM tinyml_modelmaker is a dependency of this repo's pyproject.toml, so
REM `pip install -e .` here pulls it in (and transitively tinyverse +
REM torchmodelopt) automatically.
python -c "import tinyml_modelmaker" >nul 2>&1
if errorlevel 1 (
    echo Error: tinyml_modelmaker is not installed in this Python environment.
    echo Run: pip install -e "%SCRIPT_DIR%"
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
echo Config: %CONFIG_FILE%
echo ========================================
echo.
echo Output artifacts will be written under .\data\projects relative to
echo the directory you ran this script from (override with common.projects_path
echo in the config).
echo.

REM Run training via modelmaker. Invoked as a module (-m), not by script path -
REM run_tinyml_modelmaker.py has a chdir('..') special-case for the latter that
REM assumes a repo-checkout layout and would land in the wrong directory here.
python -m tinyml_modelmaker.run_tinyml_modelmaker "%CONFIG_FILE%"
