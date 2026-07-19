#!/usr/bin/env python3
"""
Update mode setup for tinyml-workflow-agent.
Called by setup skill during Step 2.
Writes UPDATE_MODE to .env file.
"""

import json
from pathlib import Path
from typing import Dict, Optional


SKILL_ROOT = Path(__file__).parent.parent.parent / "tinyml-workflow-agent"
ENV_FILE = Path.home() / ".tinyml-agent-skills" / ".env"


def _read_plugin_json() -> Dict:
    # plugin.json is at <skills_root>/../.claude-plugin/plugin.json
    plugin_file = SKILL_ROOT.parent.parent / ".claude-plugin" / "plugin.json"
    if plugin_file.exists():
        try:
            with open(plugin_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def get_current_version() -> str:
    data = _read_plugin_json()
    return data.get("version", "unknown")


def _read_env_vars() -> Dict:
    """Read all vars from .env file."""
    vars_dict = {}
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, val = line.split("=", 1)
                            vars_dict[key.strip()] = val.strip()
        except IOError:
            pass
    return vars_dict


def _write_env_var(key: str, value: str):
    """Update or add a key=value pair in .env file."""
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)

    vars_dict = _read_env_vars()
    vars_dict[key] = value

    with open(ENV_FILE, "w") as f:
        for k, v in vars_dict.items():
            f.write(f"{k}={v}\n")


def save_config(tinyml_base_path: str, docs_path: str, update_mode: str, pinned_version: str = "") -> Dict:
    """
    Write all config vars to ~/.tinyml-agent-skills/.env.
    Called at end of setup to persist the full configuration.
    Uses pathlib for cross-platform compatibility.
    """
    try:
        ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "IS_REPO_SETUP": "1",
            "TINYML_BASE_PATH": tinyml_base_path,
            "TINYML_TENSORLAB_DOCS_PATH": docs_path,
            "UPDATE_MODE": update_mode,
            "UPDATE_PINNED_VERSION": pinned_version or "",
        }
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


def set_update_mode(mode: str) -> Dict:
    """
    Store user's chosen update mode in .env.
    mode: "pinned" or "auto"
    """
    if mode not in ("pinned", "auto"):
        return {"success": False, "errors": [f"Invalid mode '{mode}'. Use 'pinned' or 'auto'."]}

    current_version = get_current_version()
    _write_env_var("UPDATE_MODE", mode)

    if mode == "pinned":
        _write_env_var("UPDATE_PINNED_VERSION", current_version)
    else:
        _write_env_var("UPDATE_PINNED_VERSION", "")

    return {
        "success": True,
        "mode": mode,
        "pinned_version": current_version if mode == "pinned" else None,
        "current_version": current_version,
        "message": (
            f"Pinned to v{current_version}. Skill will not auto-update."
            if mode == "pinned"
            else "Auto-update enabled. Skill will check for updates on each session."
        ),
    }
