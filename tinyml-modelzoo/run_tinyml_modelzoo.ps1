<#
.SYNOPSIS
    Tiny ML ModelZoo Training Wrapper for PowerShell.
    Delegates training to tinyml-modelmaker, installed as a Python package
    (wheel or editable install) - it does NOT need to be cloned as a sibling
    directory.

.EXAMPLE
    ./run_tinyml_modelzoo.ps1 examples/generic_timeseries_classification/config.yaml
#>

param(
    [Parameter(Position = 0)]
    [string]$ConfigFile,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# tinyml_modelmaker is a dependency of this repo's pyproject.toml, so
# `pip install -e .` here pulls it in (and transitively tinyverse +
# torchmodelopt) automatically.
python -c "import tinyml_modelmaker" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "tinyml_modelmaker is not installed in this Python environment. Run: pip install -e `"$ScriptDir`""
    exit 1
}

if (-not $ConfigFile) {
    Write-Host "Tiny ML ModelZoo Training Wrapper"
    Write-Host ""
    Write-Host "Usage: run_tinyml_modelzoo.ps1 <config_file> [additional_args...]"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  run_tinyml_modelzoo.ps1 examples/hello_world/config.yaml"
    Write-Host "  run_tinyml_modelzoo.ps1 examples/motor_bearing_fault/config.yaml"
    Write-Host ""
    Write-Host "Available example configs:"
    $examplesDir = Join-Path $ScriptDir "examples"
    if (Test-Path $examplesDir) {
        Get-ChildItem -Path $examplesDir -Filter "*.yaml" -Recurse | ForEach-Object {
            Write-Host "  $($_.FullName.Substring($ScriptDir.Length + 1))"
        }
    }
    exit 1
}

if (-not [System.IO.Path]::IsPathRooted($ConfigFile)) {
    if (Test-Path $ConfigFile) {
        $ConfigFile = (Resolve-Path $ConfigFile).Path
    } elseif (Test-Path (Join-Path $ScriptDir $ConfigFile)) {
        $ConfigFile = Join-Path $ScriptDir $ConfigFile
    } else {
        Write-Error "Config file not found: $ConfigFile"
        exit 1
    }
}

if (-not (Test-Path $ConfigFile)) {
    Write-Error "Config file not found: $ConfigFile"
    exit 1
}

Write-Host "Tiny ML ModelZoo Training"
Write-Host "========================================"
Write-Host "Config: $ConfigFile"
Write-Host "========================================"
Write-Host ""
Write-Host "Output artifacts will be written under ./data/projects relative to"
Write-Host "the directory you ran this script from (override with common.projects_path"
Write-Host "in the config)."
Write-Host ""

# Run training via modelmaker. Invoked as a module (-m), not by script path -
# run_tinyml_modelmaker.py has a chdir('..') special-case for the latter that
# assumes a repo-checkout layout and would land in the wrong directory here.
python -m tinyml_modelmaker.run_tinyml_modelmaker $ConfigFile @ExtraArgs
