#!/usr/bin/env python3
"""
Locate an existing tinyml-modelzoo install: a git repo checkout, or a pip-installed
package (wheel) in any Python environment on this machine.
Called by the setup skill during Step 3.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ENV_FILE = Path.home() / ".tinyml-agent-skills" / ".env"
MANAGED_VENV = Path.home() / ".tinyml-agent-skills" / "venv"

# Directories never worth descending into while hunting for venvs
_PRUNE = {
    ".git", ".cache", ".local", ".npm", ".nvm", ".cargo", ".rustup", "node_modules",
    "Trash", "site-packages", "__pycache__", "snap", ".vscode", ".vscode-server", ".claude",
}
_VENV_SEARCH_DEPTH = 5

_PROBE = (
    "import tinyml_modelzoo, os; "
    "f = os.path.dirname(os.path.abspath(tinyml_modelzoo.__file__)); "
    "print(f); print(getattr(tinyml_modelzoo, '__version__', 'unknown'))"
)


def _venv_python(venv: Path) -> Path:
    win = venv / "Scripts" / "python.exe"
    return win if win.exists() else venv / "bin" / "python3"


def _read_env() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def is_modelzoo_repo(path: str) -> bool:
    p = Path(path).expanduser()
    return (p / "tinyml_modelzoo").is_dir() and (
        (p / "run_tinyml_modelzoo.sh").exists() or (p / "run_tinyml_modelzoo.bat").exists()
    )


def _find_repos(extra: List[str]) -> List[str]:
    home = Path.home()
    cands = list(extra) + [
        str(home / "tinyml-modelzoo"),
        str(home / "projects" / "tinyml-modelzoo"),
        str(home / "tinyml-tensorlab" / "tinyml-modelzoo"),
        "/opt/tinyml-modelzoo",
        os.getcwd(),
        os.path.join(os.getcwd(), "tinyml-modelzoo"),
    ]
    seen, found = set(), []
    for c in cands:
        if not c:
            continue
        c = os.path.realpath(os.path.expanduser(c))
        if c not in seen and is_modelzoo_repo(c):
            seen.add(c)
            found.append(c)
    return found


def _find_venv_pythons() -> List[str]:
    """Walk $HOME (bounded depth, pruned) for virtualenvs; return their python binaries."""
    home = str(Path.home())
    base_depth = home.rstrip(os.sep).count(os.sep)
    pythons: List[str] = []
    for root, dirs, files in os.walk(home):
        if "pyvenv.cfg" in files:
            py = _venv_python(Path(root))
            if py.exists():
                pythons.append(str(py))
            dirs[:] = []
            continue
        if root.count(os.sep) - base_depth >= _VENV_SEARCH_DEPTH:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in _PRUNE]
    return pythons


def _probe_python(py: str) -> Optional[Dict]:
    try:
        r = subprocess.run([py, "-c", _PROBE], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or len(lines) < 2:
        return None
    pkg_dir = lines[0]
    root = os.path.dirname(pkg_dir)  # parent of tinyml_modelzoo/
    return {
        "python": py,
        "package_dir": pkg_dir,
        "modelzoo_root": root,  # site-packages for a wheel; repo root for an editable install
        "version": lines[1],
        "is_repo_backed": is_modelzoo_repo(root),
    }


def detect_modelzoo(extra_repo_paths: Optional[List[str]] = None, deep_search: bool = True) -> Dict:
    """
    Returns:
        repos:          git checkouts of tinyml-modelzoo (contain run_tinyml_modelzoo.sh)
        wheels:         Pythons that import a non-repo (pip/wheel) tinyml_modelzoo; each has
                        python, package_dir, modelzoo_root (site-packages), version
        editable_repos: Pythons whose tinyml_modelzoo is an editable install of a repo
    """
    env = _read_env()
    extra = list(extra_repo_paths or [])
    if env.get("MODELZOO_INSTALL_MODE", "repo") == "repo" and env.get("TINYML_MODELZOO_PATH"):
        extra.append(env["TINYML_MODELZOO_PATH"])
    repos = _find_repos(extra)

    pythons: List[str] = []
    if env.get("VENV_PYTHON"):
        pythons.append(env["VENV_PYTHON"])
    managed = _venv_python(MANAGED_VENV)
    if managed.exists():
        pythons.append(str(managed))
    pythons.append(sys.executable)
    for name in ("python3", "python"):
        w = shutil.which(name)
        if w:
            pythons.append(w)
    for r in repos:
        py = _venv_python(Path(r) / ".venv")
        if py.exists():
            pythons.append(str(py))
    if deep_search:
        pythons.extend(_find_venv_pythons())

    seen, wheels, editable = set(), [], []
    for py in pythons:
        if not py or not os.path.exists(py):
            continue
        key = os.path.abspath(py)  # venv pythons symlink to system python: dedupe on venv path
        if key in seen:
            continue
        seen.add(key)
        info = _probe_python(py)
        if not info:
            continue
        if info["is_repo_backed"]:
            editable.append(info)
            if info["modelzoo_root"] not in repos:
                repos.append(info["modelzoo_root"])
        else:
            wheels.append(info)

    uniq, dirs_seen = [], set()
    for w in wheels:
        if w["package_dir"] not in dirs_seen:
            dirs_seen.add(w["package_dir"])
            uniq.append(w)

    return {
        "success": True,
        "found": bool(repos or uniq),
        "repos": repos,
        "wheels": uniq,
        "editable_repos": editable,
    }


def check_wheel_install(venv_python: str) -> Dict:
    """Verify a wheel install in venv_python has every package the workflow needs."""
    probe = (
        "import json, os\n"
        "out = {}\n"
        "for m in ('tinyml_modelzoo','tinyml_modelmaker','tinyml_tinyverse','tinyml_torchmodelopt'):\n"
        "    try:\n"
        "        mod = __import__(m)\n"
        "        out[m] = {'path': os.path.dirname(os.path.abspath(mod.__file__)), 'version': getattr(mod, '__version__', 'unknown')}\n"
        "    except Exception as e:\n"
        "        out[m] = {'error': str(e)}\n"
        "print(json.dumps(out))\n"
    )
    try:
        r = subprocess.run([venv_python, "-c", probe], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"success": False, "errors": [str(e)]}
    if r.returncode != 0:
        return {"success": False, "errors": [r.stderr.strip()]}
    pkgs = json.loads(r.stdout.strip().splitlines()[-1])
    missing = [m for m, v in pkgs.items() if "error" in v]
    run_script = None
    if "error" not in pkgs["tinyml_modelmaker"]:
        cand = os.path.join(pkgs["tinyml_modelmaker"]["path"], "run_tinyml_modelmaker.py")
        if os.path.isfile(cand):
            run_script = cand
        else:
            missing.append("run_tinyml_modelmaker.py")
    return {
        "success": not missing,
        "packages": pkgs,
        "modelzoo_root": os.path.dirname(pkgs["tinyml_modelzoo"]["path"]) if "error" not in pkgs["tinyml_modelzoo"] else None,
        "run_script": run_script,
        "errors": [f"Missing: {m}" for m in missing],
    }
