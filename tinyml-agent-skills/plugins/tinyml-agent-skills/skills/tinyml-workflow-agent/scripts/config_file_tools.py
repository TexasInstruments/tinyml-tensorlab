"""
Tools for assembling the final config.yaml and running tinyml-modelzoo.
"""

import os
import subprocess
from typing import Optional, Dict, Any

import yaml

_DEFAULT_MODELZOO_PATH = os.path.expanduser("~/tinyml-tensorlab/tinyml-modelzoo")
_DEFAULT_EXAMPLES_DIR = os.path.join(_DEFAULT_MODELZOO_PATH, "examples")
_DEFAULT_RUN_SCRIPT = os.path.join(_DEFAULT_MODELZOO_PATH, "run_tinyml_modelzoo.sh")


def _read_yaml_source(yaml_str: Optional[str], yaml_file: Optional[str], section_name: str) -> tuple:
    """
    Resolve a section's YAML from either an inline string or a file path.
    Returns (yaml_content_or_None, error_or_None).
    Prefers yaml_file over yaml_str when both are provided.
    """
    if yaml_file:
        yaml_file = os.path.expanduser(yaml_file)
        if not os.path.isfile(yaml_file):
            return None, f"{section_name}: yaml_file '{yaml_file}' does not exist."
        with open(yaml_file) as f:
            return f.read(), None
    return yaml_str, None


def generate_complete_config_file(
    task_name: str,
    tinyml_base_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    # Inline YAML strings (legacy / fallback)
    common_yaml: Optional[str] = None,
    dataset_yaml: Optional[str] = None,
    feature_extraction_yaml: Optional[str] = None,
    training_yaml: Optional[str] = None,
    testing_yaml: Optional[str] = None,
    compilation_yaml: Optional[str] = None,
    # File-based YAML paths (preferred — avoids JSON escaping issues)
    common_yaml_file: Optional[str] = None,
    dataset_yaml_file: Optional[str] = None,
    feature_extraction_yaml_file: Optional[str] = None,
    training_yaml_file: Optional[str] = None,
    testing_yaml_file: Optional[str] = None,
    compilation_yaml_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: Assemble all generated YAML sections into a single config.yaml and save it.

    Call this after all section YAMLs have been generated (Steps 2-10 in the skill workflow).

    PREFERRED USAGE — pass yaml_file paths produced by --save-yaml in earlier steps:
        {
          "task_name": "motor_fault_demo",
          "tinyml_base_path": "/home/user/tinyml-tensorlab",
          "common_yaml_file": "/tmp/tinyml_work/common.yaml",
          "dataset_yaml_file": "/tmp/tinyml_work/dataset.yaml",
          "feature_extraction_yaml_file": "/tmp/tinyml_work/feat_ext.yaml",
          "training_yaml_file": "/tmp/tinyml_work/training.yaml",
          "testing_yaml_file": "/tmp/tinyml_work/testing.yaml",
          "compilation_yaml_file": "/tmp/tinyml_work/compilation.yaml"
        }

    FALLBACK USAGE — pass inline yaml strings (more error-prone due to JSON escaping):
        Use common_yaml, dataset_yaml, etc. instead of *_file variants.

    File-based variants take precedence over inline strings when both are provided.

    Output path resolution (first match wins):
        1. output_dir — explicit override
        2. {tinyml_base_path}/tinyml-modelzoo/examples/{task_name}/
        3. ~/tinyml-tensorlab/tinyml-modelzoo/examples/{task_name}/ (default)

    Returns:
        success, yaml_file_path, config (dict), errors, warnings
    """
    errors = []
    warnings = []

    # Resolve each section — prefer file over inline string
    section_order = [
        ("common",                            common_yaml,                common_yaml_file),
        ("dataset",                           dataset_yaml,               dataset_yaml_file),
        ("data_processing_feature_extraction", feature_extraction_yaml,   feature_extraction_yaml_file),
        ("training",                          training_yaml,              training_yaml_file),
        ("testing",                           testing_yaml,               testing_yaml_file),
        ("compilation",                       compilation_yaml,           compilation_yaml_file),
    ]

    merged: Dict[str, Any] = {}

    for section_name, yaml_str, yaml_file in section_order:
        content, err = _read_yaml_source(yaml_str, yaml_file, section_name)
        if err:
            errors.append(err)
            continue
        if not content:
            continue
        try:
            parsed = yaml.safe_load(content)
            if not isinstance(parsed, dict):
                errors.append(f"{section_name}: expected YAML dict, got {type(parsed).__name__}.")
                continue
            if section_name in parsed:
                merged[section_name] = parsed[section_name]
            else:
                merged[section_name] = parsed
                warnings.append(f"{section_name}: YAML not wrapped in section key — used as-is.")
        except yaml.YAMLError as e:
            errors.append(f"{section_name}: failed to parse YAML — {e}")

    if errors:
        return {
            "success": False,
            "yaml_file_path": None,
            "config": None,
            "errors": errors,
            "warnings": warnings,
        }

    if not merged:
        return {
            "success": False,
            "yaml_file_path": None,
            "config": None,
            "errors": ["No YAML sections provided. Pass at least common_yaml_file and dataset_yaml_file."],
            "warnings": [],
        }

    # Resolve output directory
    if output_dir:
        out_dir = os.path.expanduser(output_dir)
    elif tinyml_base_path:
        modelzoo_path = os.path.join(os.path.expanduser(tinyml_base_path), "tinyml-modelzoo")
        examples_dir = os.path.join(modelzoo_path, "examples")
        safe_task = task_name.replace(" ", "_").replace("/", "_")
        out_dir = os.path.join(examples_dir, safe_task)
    else:
        safe_task = task_name.replace(" ", "_").replace("/", "_")
        out_dir = os.path.join(_DEFAULT_EXAMPLES_DIR, safe_task)

    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "config.yaml")

    try:
        with open(config_path, "w") as f:
            yaml.dump(merged, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except OSError as e:
        return {
            "success": False,
            "yaml_file_path": None,
            "config": merged,
            "errors": [f"Failed to write config.yaml: {e}"],
            "warnings": warnings,
        }

    return {
        "success": True,
        "yaml_file_path": config_path,
        "config": merged,
        "errors": [],
        "warnings": warnings,
    }


def run_example(
    config_yaml_path: str,
    tinyml_base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: Run tinyml-modelzoo with the given config.yaml.

    Executes: bash {tinyml_base_path}/tinyml-modelzoo/run_tinyml_modelzoo.sh <config_yaml_path>

    Call this only after generate_complete_config_file has succeeded and the user
    confirms they want to start training/compilation.

    Args:
        config_yaml_path: Absolute path to the config.yaml file to run.
            Use yaml_file_path from generate_complete_config_file.
        tinyml_base_path: Path to tinyml-tensorlab root. Defaults to ~/tinyml-tensorlab.

    Returns:
        success, returncode, stdout, stderr, errors
    """
    config_yaml_path = os.path.expanduser(config_yaml_path)

    if not os.path.isfile(config_yaml_path):
        return {
            "success": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "errors": [f"config.yaml not found: {config_yaml_path}"],
        }

    if tinyml_base_path:
        run_script = os.path.join(
            os.path.expanduser(tinyml_base_path), "tinyml-modelzoo", "run_tinyml_modelzoo.sh"
        )
    else:
        run_script = _DEFAULT_RUN_SCRIPT

    if not os.path.isfile(run_script):
        return {
            "success": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "errors": [
                f"run_tinyml_modelzoo.sh not found at {run_script}",
                "Verify tinyml_base_path is correct, or run manually: "
                f"bash {run_script} {config_yaml_path}",
            ],
        }

    try:
        result = subprocess.run(
            ["bash", run_script, config_yaml_path],
            capture_output=True,
            text=True,
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "errors": [] if result.returncode == 0 else [
                f"Script exited with code {result.returncode}. Check stderr for details."
            ],
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "errors": [f"Failed to launch run script: {e}"],
        }
