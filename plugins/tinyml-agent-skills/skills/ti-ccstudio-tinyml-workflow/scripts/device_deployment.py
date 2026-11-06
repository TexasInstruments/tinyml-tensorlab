"""
Device deployment helper tools for tinyml-modelzoo.

  check_sdk_installation  - locate the device-family SDK and its AI examples
  find_run_artifacts      - locate and validate ModelMaker outputs (mod.a, tvmgen_default.h, test_vector.c, user_input_config.h)

Importing, building and flashing the project are done through the IDE's own tools (see SKILL.md Step 14).
See references/device_deployment_guide.md for the deployment walkthrough.
SDK family -> device mapping is discovered dynamically via check_sdk_installation.
"""

import os
import glob
import json
import shutil
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any, List, Tuple
import stat
from pathlib import Path

try:
    from .constants import CLASSIFICATION_TASKS, ANOMALY_TASKS, REGRESSION_TASKS, FORECASTING_TASKS
except ImportError:
    from constants import CLASSIFICATION_TASKS, ANOMALY_TASKS, REGRESSION_TASKS, FORECASTING_TASKS

_DEFAULT_TINYML_BASE     = os.path.expanduser("~/tinyml-modelzoo")
_DEFAULT_MODELMAKER_RUNS = os.path.join(_DEFAULT_TINYML_BASE, "data", "projects")


# ── Device family + SDK metadata ───────────────────────────────────────────────

# Maps each device ID to its family name
DEVICE_FAMILY: Dict[str, str] = {
    # C2000 F28x — uses C2000Ware SDK
    "F280013": "c2000", "F280015": "c2000", "F28003": "c2000", "F28004": "c2000",
    "F2837":   "c2000", "F28P55":  "c2000", "F28P65": "c2000",
    # C2000 F29x — uses F29H85X-SDK (separate from C2000Ware)
    "F29H85": "f29", "F29P58": "f29", "F29P32": "f29",
    # MSPM0 — uses MSPM0 SDK
    "MSPM0G3507": "mspm0", "MSPM0G3519": "mspm0", "MSPM0G5187": "mspm0",
    # MSPM33 — uses MSPM33 SDK
    "MSPM33C32": "mspm33", "MSPM33C34": "mspm33",
    # AM13 — uses AM13E2X-SDK
    "AM13E2": "am13",
    # AM26x — uses MCU+ SDK
    "AM263": "am26x", "AM263P": "am26x", "AM261": "am26x",
    # SimpleLink F3 (CC27xx) — uses SimpleLink LP F3 SDK
    "CC2755": "simplelink_f3",
    # SimpleLink F2 (CC13xx) — uses SimpleLink LP F2 SDK
    "CC1352": "simplelink_f2", "CC1354": "simplelink_f2",
    # SimpleLink Wi-Fi (CC35x) — uses SimpleLink Wi-Fi SDK
    "CC35X1": "simplelink_wifi",
}

# Per-family SDK information: name, install search globs, AI examples subpath, download URL
SDK_INFO: Dict[str, Dict] = {
    "c2000": {
        "name":         "C2000Ware",
        "install_globs": [
            os.path.expanduser("~/ti/c2000ware_*"),
            os.path.expanduser("~/ti/C2000Ware_*"),
            "/opt/ti/c2000ware_*",
            "/opt/ti/C2000Ware_*",
            os.path.expanduser("~/ti/ccs*/c2000/C2000Ware_*"),
            "C:/ti/c2000ware_*",
            "C:/ti/C2000Ware_*",
            "C:/ti/ccs*/c2000/C2000Ware_*",
        ],
        "ai_examples_subpath": "libraries/ai/examples",
        "download_url": "https://www.ti.com/tool/C2000WARE",
    },
    "mspm0": {
        "name":         "MSPM0 SDK",
        "install_globs": [
            os.path.expanduser("~/ti/mspm0_sdk_*"),
            "/opt/ti/mspm0_sdk_*",
            "C:/ti/mspm0_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/MSPM0-SDK",
    },
    "mspm33": {
        "name":         "MSPM33 SDK",
        "install_globs": [
            os.path.expanduser("~/ti/mspm33_sdk_*"),
            "/opt/ti/mspm33_sdk_*",
            "C:/ti/mspm33_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/download/MSPM33-SDK",
    },
    "f29": {
        "name":         "C2000 SDK (F29H85X)",
        "install_globs": [
            os.path.expanduser("~/ti/f29h85_sdk_*"),
            "/opt/ti/f29h85_sdk_*",
            "C:/ti/f29h85_sdk_*",
        ],
        "ai_examples_subpath": "libraries/ai/examples",
        "download_url": "https://www.ti.com/tool/download/F29H85X-SDK/",
    },
    "am13": {
        "name":         "AM13E2X SDK",
        "install_globs": [
            os.path.expanduser("~/ti/am13e230x_sdk_*"),
            "/opt/ti/am13e230x_sdk_*",
            "C:/ti/am13e230x_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/AM13E2X-SDK",
    },
    "am26x": {
        "name":         "MCU+ SDK (AM26x)",
        "install_globs": [
            os.path.expanduser("~/ti/mcu_plus_sdk_am263x_*"),
            os.path.expanduser("~/ti/mcu_plus_sdk_*"),
            "/opt/ti/mcu_plus_sdk_*",
            "C:/ti/mcu_plus_sdk_am263x_*",
            "C:/ti/mcu_plus_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/MCU-PLUS-SDK-AM263X",
    },
    "simplelink_f3": {
        "name":         "SimpleLink LP F3 SDK",
        "install_globs": [
            os.path.expanduser("~/ti/simplelink_lowpower_f3_sdk_*"),
            "/opt/ti/simplelink_lowpower_f3_sdk_*",
            "C:/ti/simplelink_lowpower_f3_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/SIMPLELINK-LOWPOWER-F3-SDK",
    },
    "simplelink_f2": {
        "name":         "SimpleLink LP F2 SDK",
        "install_globs": [
            os.path.expanduser("~/ti/simplelink_lowpower_f2_sdk_*"),
            os.path.expanduser("~/ti/simplelink_cc13xx_cc26xx_sdk_*"),
            "/opt/ti/simplelink_lowpower_f2_sdk_*",
            "/opt/ti/simplelink_cc13xx_cc26xx_sdk_*",
            "C:/ti/simplelink_lowpower_f2_sdk_*",
            "C:/ti/simplelink_cc13xx_cc26xx_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/SIMPLELINK-LOWPOWER-F2-SDK",
    },
    "simplelink_wifi": {
        "name":         "SimpleLink Wi-Fi SDK",
        "install_globs": [
            os.path.expanduser("~/ti/simplelink_wifi_sdk_*"),
            "/opt/ti/simplelink_wifi_sdk_*",
            "C:/ti/simplelink_wifi_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/SIMPLELINK-WIFI-SDK",
    },
}


def _get_device_family(device_id: str) -> Optional[str]:
    """Return SDK family name for a device ID, case-insensitive."""
    return DEVICE_FAMILY.get(device_id) or DEVICE_FAMILY.get(device_id.upper())


def _find_sdk_root(family: str) -> Optional[str]:
    """Search common install locations for an SDK by family. Returns newest match."""
    info = SDK_INFO.get(family, {})
    matches = []
    for pattern in info.get("install_globs", []):
        matches += glob.glob(pattern)
    if not matches:
        return None
    # Sort by basename only so parent directory path doesn't affect version comparison.
    # TI zero-pads version strings (e.g. mspm0_sdk_2_10_00_04), so lexicographic
    # sort of the folder name reliably picks the highest version.
    return max(matches, key=os.path.basename)


def _sdk_ai_examples_path(family: str, sdk_root: str) -> str:
    """Return the AI examples directory within an SDK root."""
    subpath = SDK_INFO.get(family, {}).get("ai_examples_subpath", "examples")
    return os.path.join(sdk_root, subpath)


# ── Tool: check_sdk_installation ───────────────────────────────────────────────

def check_sdk_installation(
    target_device: str,
    sdk_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: Check whether the device-specific SDK is installed and locate its AI examples.

    Call this before importing a template project to detect missing SDKs early.

    Args:
        target_device: Target MCU (e.g., 'F28P55', 'MSPM0G3507', 'AM263').
        sdk_path: Optional explicit path to the SDK root. If provided, skips
            auto-detection and validates only the given path.

    Returns:
        Dict with:
        - family: device family name
        - sdk_name: human-readable SDK name
        - sdk_root: resolved SDK root path (or None)
        - ai_examples_path: path to AI examples within the SDK (or None)
        - found: True if SDK was located
        - errors: empty if found, else guidance message with download URL
    """
    family = _get_device_family(target_device)
    if not family:
        return {
            "found": False,
            "family": None,
            "sdk_name": None,
            "sdk_root": None,
            "ai_examples_path": None,
            "errors": [f"Unknown device '{target_device}'. Cannot determine required SDK."],
        }

    info = SDK_INFO[family]
    sdk_name = info["name"]

    if sdk_path:
        sdk_root = os.path.expanduser(sdk_path)
        if not os.path.isdir(sdk_root):
            return {
                "found": False,
                "family": family,
                "sdk_name": sdk_name,
                "sdk_root": sdk_root,
                "ai_examples_path": None,
                "errors": [f"Provided sdk_path does not exist: {sdk_root}"],
            }
    else:
        sdk_root = _find_sdk_root(family)

    if not sdk_root:
        searched = info.get("install_globs", [])
        return {
            "found": False,
            "family": family,
            "sdk_name": sdk_name,
            "sdk_root": None,
            "ai_examples_path": None,
            "errors": [
                f"{sdk_name} not found. Searched: {searched}",
                f"Download from: {info['download_url']}",
                f"After installing, pass sdk_path='<install_root>' when importing a template.",
            ],
        }

    ai_examples = _sdk_ai_examples_path(family, sdk_root)
    ai_exists = os.path.isdir(ai_examples)

    return {
        "found": True,
        "family": family,
        "sdk_name": sdk_name,
        "sdk_root": sdk_root,
        "ai_examples_path": ai_examples if ai_exists else None,
        "ai_examples_exists": ai_exists,
        "errors": [] if ai_exists else [
            f"SDK found at {sdk_root} but AI examples path not found: {ai_examples}. "
            "Ensure the SDK version includes AI library examples."
        ],
    }


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _runs_path(tinyml_base_path: Optional[str]) -> str:
    # tinyml_base_path is now the modelzoo root; output goes to {modelzoo}/data/projects/
    base = os.path.expanduser(tinyml_base_path) if tinyml_base_path else _DEFAULT_TINYML_BASE
    return os.path.join(base, "data", "projects")


def _run_base(tinyml_base_path, task_type, run_id, model_id) -> str:
    return os.path.join(_runs_path(tinyml_base_path), task_type, "run", run_id, model_id)


def _golden_dir(tinyml_base_path, task_type, run_id, model_id, quantization) -> str:
    sub = "quantization" if quantization else "base"
    return os.path.join(_run_base(tinyml_base_path, task_type, run_id, model_id), "training", sub, "golden_vectors")


# ── Tool A: find_run_artifacts ─────────────────────────────────────────────────

def find_run_artifacts(
    tinyml_base_path: str,
    task_type: str,
    run_id: str,
    model_id: str,
    quantization: bool,
) -> Dict[str, Any]:
    """
    Tool: Locate and validate all ModelMaker output files needed for the project.

    Call this BEFORE copying artifacts into the project to confirm training/compilation
    completed successfully and all 4 required files exist.

    Args:
        tinyml_base_path: Root of tinyml-modelzoo clone (output goes to {modelzoo}/data/projects/).
        task_type: Task type (e.g., 'motor_fault').
        run_id: Run directory name from training output (e.g., '20240115_143022').
        model_id: Model name (matches model_name from training config).
        quantization: True if quantized training was performed.

    Returns:i_examples_subpath": "examples",
        Dict with required_files (name → {path, exists}), missing list,
        artifacts_dir, golden_dir, and success flag.

    How to find run_id and model_id:
        ls {tinyml_base_path}/data/projects/{task_type}/run/
    """
    run_dir   = _run_base(tinyml_base_path, task_type, run_id, model_id)
    art_dir   = os.path.join(run_dir, "compilation", "artifacts")
    gold_dir  = _golden_dir(tinyml_base_path, task_type, run_id, model_id, quantization)

    required = {
        "mod.a":             os.path.join(art_dir,  "mod.a"),
        "tvmgen_default.h":  os.path.join(art_dir,  "tvmgen_default.h"),
        "test_vector.c":     os.path.join(gold_dir, "test_vector.c"),
        "user_input_config.h": os.path.join(gold_dir, "user_input_config.h"),
    }

    extra_artifacts: List[str] = []
    if os.path.isdir(art_dir):
        extra_artifacts = sorted(os.listdir(art_dir))

    file_status = {name: {"path": path, "exists": os.path.exists(path)}
                   for name, path in required.items()}
    missing     = [name for name, s in file_status.items() if not s["exists"]]

    return {
        "success":       len(missing) == 0,
        "run_dir":       run_dir,
        "artifacts_dir": art_dir,
        "golden_dir":    gold_dir,
        "required_files": file_status,
        "extra_artifacts": extra_artifacts,
        "missing": missing,
        "errors":  [f"Missing: {n} (expected: {file_status[n]['path']})" for n in missing],
        "hint":    (
            "Check that training and compilation completed successfully. "
            "The run_id is the timestamp directory under "
            f"{_runs_path(tinyml_base_path)}/{task_type}/run/"
        ) if missing else "",
    }
