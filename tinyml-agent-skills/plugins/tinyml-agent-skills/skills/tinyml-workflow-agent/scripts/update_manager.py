#!/usr/bin/env python3
"""
Update manager for tinyml-workflow-agent runtime.
Called by tinyml-workflow-agent during session start.
Reads UPDATE_MODE from .env, checks commits behind origin, updates via git_pull_all.sh.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional


SKILL_ROOT = Path(__file__).parent.parent
ENV_FILE = Path.home() / ".tinyml-agent-skills" / ".env"


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
            print(f"[setup] .env loaded from: {ENV_FILE}", file=__import__('sys').stderr)
        except IOError as e:
            print(f"[setup] ERROR reading .env from {ENV_FILE}: {e}", file=__import__('sys').stderr)
    else:
        print(f"[setup] ERROR: .env NOT FOUND at {ENV_FILE}", file=__import__('sys').stderr)
        print("[setup] This file must be created by /tinyml-agent-skills:setup first", file=__import__('sys').stderr)
    return vars_dict


def _get_tinyml_modelzoo_path() -> Optional[str]:
    return _read_env_vars().get("TINYML_MODELZOO_PATH")


def get_current_version() -> str:
    plugin_file = SKILL_ROOT.parent.parent / ".claude-plugin" / "plugin.json"
    if plugin_file.exists():
        try:
            with open(plugin_file) as f:
                return json.load(f).get("version", "unknown")
        except (json.JSONDecodeError, IOError):
            pass
    return "unknown"


def get_update_status() -> Dict:
    """
    Returns current update config + current version from .env.
    Called at session start to check if UPDATE_MODE was initialized by setup.
    """
    env_vars = _read_env_vars()
    mode = env_vars.get("UPDATE_MODE")

    if not ENV_FILE.exists():
        return {
            "success": False,
            "initialized": False,
            "mode": None,
            "current_version": get_current_version(),
            "error": f".env file not found at {ENV_FILE}",
            "hint": "Run /tinyml-agent-skills:setup first to create the .env file with all required configuration.",
            "env_file_path": str(ENV_FILE),
            "scripts_dir": str(SKILL_ROOT / "scripts"),
        }

    return {
        "success": True,
        "initialized": mode is not None,
        "mode": mode,
        "current_version": get_current_version(),
        "env_file_path": str(ENV_FILE),
        "scripts_dir": str(SKILL_ROOT / "scripts"),
    }


def check_updates() -> Dict:
    """
    Check if tinyml-modelzoo is behind origin by comparing commit counts.
    Only runs when mode is 'auto'.
    """
    env_vars = _read_env_vars()
    mode = env_vars.get("UPDATE_MODE")

    if not mode:
        return {"success": False, "errors": ["Update mode not set. Run setup first."]}

    if mode != "auto":
        return {"success": True, "update_available": False, "reason": "pinned_version"}

    base = _get_tinyml_modelzoo_path()
    if not base:
        return {"success": False, "errors": ["TINYML_MODELZOO_PATH not set in .env. Run setup first."]}

    repo_root = Path(base).expanduser()
    if not repo_root.exists():
        return {"success": False, "errors": [f"Path does not exist: {repo_root}"]}

    try:
        fetch = subprocess.run(
            ["git", "fetch"],
            cwd=str(repo_root),
            capture_output=True, text=True, timeout=30,
        )
        if fetch.returncode != 0:
            return {"success": False, "errors": [f"git fetch failed: {fetch.stderr.strip()}"]}

        for ref in ("origin/HEAD", "origin/main"):
            behind = subprocess.run(
                ["git", "rev-list", "--count", f"HEAD..{ref}"],
                cwd=str(repo_root),
                capture_output=True, text=True, timeout=10,
            )
            if behind.returncode == 0:
                commits_behind = int(behind.stdout.strip())
                current = get_current_version()
                if commits_behind == 0:
                    return {"success": True, "update_available": False, "reason": "already_latest", "current": current}
                return {
                    "success": True,
                    "update_available": True,
                    "current": current,
                    "commits_behind": commits_behind,
                    "message": f"tinyml-modelzoo is {commits_behind} commit(s) behind origin/main.",
                }

        return {"success": False, "errors": ["Could not determine commit distance from origin."]}

    except subprocess.TimeoutExpired:
        return {"success": False, "errors": ["Timed out fetching from remote."]}
    except Exception as e:
        return {"success": False, "errors": [str(e)]}


def do_update(tinyml_modelzoo_path: str) -> Dict:
    """
    Update tinyml-modelzoo via git pull.
    """
    repo_root = Path(tinyml_modelzoo_path).expanduser()

    if not repo_root.exists():
        return {"success": False, "errors": [f"Path does not exist: {repo_root}"]}

    try:
        result = subprocess.run(
            ["git", "pull"],
            cwd=str(repo_root),
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return {"success": False, "errors": [result.stderr.strip() or result.stdout.strip()]}

        return {
            "success": True,
            "message": "tinyml-modelzoo updated successfully.",
            "output": result.stdout.strip(),
            "new_version": get_current_version(),
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "errors": ["git pull timed out."]}
    except Exception as e:
        return {"success": False, "errors": [str(e)]}
