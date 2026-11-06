#!/usr/bin/env python3
"""
Config writer for the setup skill.
Called by setup skill at Step 5 to write ~/.tinyml-agent-skills/.env.
"""

from pathlib import Path
from typing import Dict


ENV_FILE = Path.home() / ".tinyml-agent-skills" / ".env"


def save_config(
    tinyml_modelzoo_path: str,
    tinyml_modelmaker_path: str = "",
    tinyml_tinyverse_path: str = "",
    tinyml_modelopt_path: str = "",
    modelzoo_install_mode: str = "repo",
    venv_python: str = "",
    **kwargs,
) -> Dict:
    """
    Write all config vars to ~/.tinyml-agent-skills/.env.
    Called at end of setup to persist the full configuration.
    Uses pathlib for cross-platform compatibility.
    """
    if modelzoo_install_mode not in ("repo", "wheel"):
        return {"success": False, "errors": [f"Invalid modelzoo_install_mode '{modelzoo_install_mode}'. Use 'repo' or 'wheel'."]}
    try:
        ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        # repo:  TINYML_MODELZOO_PATH = git checkout root
        # wheel: TINYML_MODELZOO_PATH = site-packages dir containing tinyml_modelzoo/
        config = {
            "IS_REPO_SETUP": "1",
            "TINYML_MODELZOO_PATH": tinyml_modelzoo_path,
            "MODELZOO_INSTALL_MODE": modelzoo_install_mode,
        }
        if venv_python:
            config["VENV_PYTHON"] = venv_python
        if tinyml_modelmaker_path:
            config["TINYML_MODELMAKER_PATH"] = tinyml_modelmaker_path
        if tinyml_tinyverse_path:
            config["TINYML_TINYVERSE_PATH"] = tinyml_tinyverse_path
        if tinyml_modelopt_path:
            config["TINYML_MODELOPT_PATH"] = tinyml_modelopt_path
        with open(ENV_FILE, "w") as f:
            for k, v in config.items():
                f.write(f"{k}={v}\n")
        return {
            "success": True,
            "env_file": str(ENV_FILE),
            "config": config,
        }
    except Exception as e:
        return {"success": False, "errors": [str(e)]}
