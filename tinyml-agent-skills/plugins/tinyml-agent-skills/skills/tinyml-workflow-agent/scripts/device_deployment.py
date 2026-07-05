"""
Device deployment tools for tinyml-tensorlab.

Workflow (Steps 13A–13D in SKILL.md):
  A. find_run_artifacts       — locate and validate ModelMaker outputs (mod.a, tvmgen_default.h, test_vector.c, user_input_config.h)
  B. create_ccs_project       — create CCS project from SDK template and auto-copy artifacts + golden vectors
  C. build_ccs_project        — headless build via CCS Eclipse launcher
  D. flash_ccs_project        — flash compiled .out to device via dslite

See references/device_deployment_guide.md for full deployment walkthrough.
See assets/deployment_sdk_reference.md for device family → SDK mapping.
"""

import os
import glob
import shutil
import subprocess
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any, List, Tuple
import stat
from pathlib import Path

_DEFAULT_TINYML_BASE     = os.path.expanduser("~/tinyml-tensorlab")
_DEFAULT_MODELMAKER_RUNS = os.path.join(_DEFAULT_TINYML_BASE, "tinyml-modelmaker", "data", "projects")


# ── Device family + SDK metadata ───────────────────────────────────────────────

# Maps each device ID to its family name
DEVICE_FAMILY: Dict[str, str] = {
    # C2000 (F28x / F29x) — uses C2000Ware SDK
    "F280013": "c2000", "F280015": "c2000", "F28003": "c2000", "F28004": "c2000",
    "F2837":   "c2000", "F28P55":  "c2000", "F28P65": "c2000", "F29H85": "c2000",
    "F29P58":  "c2000", "F29P32":  "c2000",
    # MSPM0 — uses MSPM0 SDK
    "MSPM0G3507": "mspm0", "MSPM0G3519": "mspm0", "MSPM0G5187": "mspm0",
    # MSPM33 — uses MSPM33 SDK
    "MSPM33C32": "mspm33", "MSPM33C34": "mspm33",
    # AM13 — uses MCU+ SDK
    "AM13E2": "am13",
    # AM26x — uses MCU+ SDK
    "AM263": "am26x", "AM263P": "am26x", "AM261": "am26x",
    # Connectivity (CC27xx / CC13xx / CC35x) — uses SimpleLink SDK
    "CC2755": "simplelink", "CC1352": "simplelink", "CC1354": "simplelink", "CC35X1": "simplelink",
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
        ],
        "ai_examples_subpath": "libraries/ai/examples",
        "download_url": "https://www.ti.com/tool/C2000WARE",
    },
    "mspm0": {
        "name":         "MSPM0 SDK",
        "install_globs": [
            os.path.expanduser("~/ti/mspm0_sdk_*"),
            "/opt/ti/mspm0_sdk_*",
        ],
        "ai_examples_subpath": "examples",  # adjust if SDK ships AI examples elsewhere
        "download_url": "https://www.ti.com/tool/MSPM0-SDK",
    },
    "mspm33": {
        "name":         "MSPM33 SDK",
        "install_globs": [
            os.path.expanduser("~/ti/mspm33_sdk_*"),
            "/opt/ti/mspm33_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/download/MSPM33-SDK",
    },
    "am13": {
        "name":         "MCU+ SDK (AM13x)",
        "install_globs": [
            os.path.expanduser("~/ti/mcu_plus_sdk_am263x_*"),
            os.path.expanduser("~/ti/mcu_plus_sdk_*"),
            "/opt/ti/mcu_plus_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/MCU-PLUS-SDK-AM263X",
    },
    "am26x": {
        "name":         "MCU+ SDK (AM26x)",
        "install_globs": [
            os.path.expanduser("~/ti/mcu_plus_sdk_am263x_*"),
            os.path.expanduser("~/ti/mcu_plus_sdk_*"),
            "/opt/ti/mcu_plus_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/MCU-PLUS-SDK-AM263X",
    },
    "simplelink": {
        "name":         "SimpleLink Low Power SDK",
        "install_globs": [
            os.path.expanduser("~/ti/simplelink_cc13xx_cc26xx_sdk_*"),
            os.path.expanduser("~/ti/simplelink_lowpower_f3_sdk_*"),
            "/opt/ti/simplelink_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/SIMPLELINK-LOWPOWER-F3-SDK",
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
    # Return the lexicographically last match (usually highest version)
    return sorted(matches)[-1] if matches else None


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

    Call this before create_ccs_project to detect missing SDKs early.

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
                f"After installing, pass sdk_path='<install_root>' to create_ccs_project.",
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
    base = os.path.expanduser(tinyml_base_path) if tinyml_base_path else _DEFAULT_TINYML_BASE
    return os.path.join(base, "tinyml-modelmaker", "data", "projects")


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
    Tool: Locate and validate all ModelMaker output files needed for CCS project.

    Call this BEFORE creating the CCS project to confirm training/compilation
    completed successfully and all 4 required files exist.

    Args:
        tinyml_base_path: Root of tinyml-tensorlab checkout.
        task_type: Task type (e.g., 'motor_fault').
        run_id: Run directory name from training output (e.g., '20240115_143022').
        model_id: Model name (matches model_name from training config).
        quantization: True if quantized training was performed.

    Returns:
        Dict with required_files (name → {path, exists}), missing list,
        artifacts_dir, golden_dir, and success flag.

    How to find run_id and model_id:
        ls {tinyml_base_path}/tinyml-modelmaker/data/projects/{task_type}/run/
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

# ── Tool B: build_ccs_project ──────────────────────────────────────────────────

def build_ccs_project(
    ccs_project_path: str,
    ccs_install_path: str,
    workspace_path: Optional[str] = None,
    build_type: str = "full",
) -> Dict[str, Any]:
    """
    Tool: Build a CCS project headlessly using the Eclipse launcher.

    Requires CCS 12.x or later. Runs the CCS headless build application.

    Args:
        ccs_project_path: Absolute path to the CCS project folder.
        ccs_install_path: CCS installation root (e.g., '/opt/ti/ccs1260').
        workspace_path: Eclipse workspace directory. Defaults to /tmp/ccs_ws_<pid>.
        build_type: 'full' or 'incremental'. Default: 'full'.

    Returns:
        Dict with success, returncode, stdout, stderr, out_file (if found), errors.

    If this fails, open CCS manually and run Project → Build Project (Ctrl+B).
    """
    project_path = os.path.abspath(os.path.expanduser(ccs_project_path))
    ccs_path     = os.path.expanduser(ccs_install_path)

    # Locate the headless launcher (name varies by CCS version)
    candidates = [
        os.path.join(ccs_path, "eclipse", "ccstudio"),
        os.path.join(ccs_path, "eclipse", "ccstudio.exe"),
        os.path.join(ccs_path, "eclipse", "eclipse"),
    ]
    launcher = next((c for c in candidates if os.path.isfile(c)), None)
    if not launcher:
        return {
            "success": False,
            "stdout": "", "stderr": "",
            "errors": [
                f"CCS launcher not found. Tried: {candidates}. "
                "Verify ccs_install_path points to the CCS installation root."
            ],
        }

    ws = workspace_path or f"/tmp/ccs_ws_{os.getpid()}"
    cmd = [
        launcher,
        "--launcher.suppressErrors",
        "-noSplash",
        "-data", ws,
        "-application", "com.ti.ccstudio.apps.projectBuild",
        "-ccs.projects", project_path,
        "-ccs.buildType", build_type,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        success = result.returncode == 0

        # Find .out file
        out_file = None
        for config in ("Debug", "Release"):
            project_name = os.path.basename(project_path)
            candidate = os.path.join(project_path, config, f"{project_name}.out")
            if os.path.isfile(candidate):
                out_file = candidate
                break

        return {
            "success":     success,
            "returncode":  result.returncode,
            "stdout":      result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout,
            "stderr":      result.stderr[-4000:] if len(result.stderr) > 4000 else result.stderr,
            "out_file":    out_file,
            "errors": [] if success else [
                f"Build failed (exit code {result.returncode}). "
                "Check stderr above. Common causes: missing include paths, linker errors, "
                "or missing CCS device support for the target."
            ],
            "next_step": f"Flash the device: run flash_ccs_project with out_file='{out_file}'." if out_file else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "", "errors": ["Build timed out (10 min). Try building manually in CCS."]}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": "", "errors": [str(e)]}


# ── Tool C: flash_ccs_project ──────────────────────────────────────────────────

def flash_ccs_project(
    ccs_project_path: str,
    ccs_install_path: str,
    project_name: Optional[str] = None,
    ccxml_path: Optional[str] = None,
    out_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: Flash a compiled CCS project to the connected target device via dslite.

    Connect the device via USB/JTAG before calling this.

    Args:
        ccs_project_path: Absolute path to the CCS project folder.
        ccs_install_path: CCS installation root (e.g., '/opt/ti/ccs1260').
        project_name: Name of the compiled binary (defaults to folder name).
        ccxml_path: Path to the device .ccxml target config file.
            If not provided, auto-searched in the project folder.
            Use the LaunchPad variant (e.g., TMS320F28P550SJ9_LaunchPad.ccxml).
        out_file: Explicit path to the .out binary. If not provided, searched
            in project/Debug/<project_name>.out.

    Returns:
        Dict with success, returncode, stdout, stderr, out_file, ccxml, errors.

    Equivalent CCS manual step: Run → Flash Project.
    """
    project_path = os.path.abspath(os.path.expanduser(ccs_project_path))
    ccs_path     = os.path.expanduser(ccs_install_path)
    pname        = project_name or os.path.basename(project_path)

    # Locate .out file
    if out_file:
        binary = os.path.expanduser(out_file)
    else:
        for config in ("Debug", "Release"):
            candidate = os.path.join(project_path, config, f"{pname}.out")
            if os.path.isfile(candidate):
                binary = candidate
                break
        else:
            return {
                "success": False, "stdout": "", "stderr": "",
                "errors": [
                    f"Compiled binary not found. Expected at: "
                    f"{project_path}/Debug/{pname}.out\n"
                    "Build the project first (run build_ccs_project or Ctrl+B in CCS).",
                ],
            }

    # Locate dslite
    dslite_candidates = [
        os.path.join(ccs_path, "ccs_base", "DebugServer", "bin", "dslite.sh"),
        os.path.join(ccs_path, "ccs_base", "DebugServer", "bin", "dslite.bat"),
    ]
    dslite = next((d for d in dslite_candidates if os.path.isfile(d)), None)
    if not dslite:
        return {
            "success": False, "stdout": "", "stderr": "",
            "errors": [f"dslite not found. Tried: {dslite_candidates}. Verify ccs_install_path."],
        }

    # Locate .ccxml
    if ccxml_path:
        ccxml = os.path.expanduser(ccxml_path)
    else:
        # Search project root and CCS/ subdirectory
        search_dirs = [project_path, os.path.join(project_path, "CCS")]
        ccxml_files = []
        for d in search_dirs:
            if os.path.isdir(d):
                ccxml_files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".ccxml")]

        if not ccxml_files:
            return {
                "success": False, "stdout": "", "stderr": "",
                "errors": [
                    "No .ccxml target config file found in project. "
                    "Provide ccxml_path explicitly. "
                    "Use the LaunchPad variant (e.g., TMS320F28P550SJ9_LaunchPad.ccxml). "
                    "In CCS: right-click the .ccxml → Set as Active Target Configuration first."
                ],
            }

        # Prefer LaunchPad variant
        launchpad = [f for f in ccxml_files if "LaunchPad" in os.path.basename(f) or "launchpad" in os.path.basename(f).lower()]
        ccxml = launchpad[0] if launchpad else ccxml_files[0]

    if not os.path.isfile(ccxml):
        return {
            "success": False, "stdout": "", "stderr": "",
            "errors": [f"ccxml file not found: {ccxml}"],
        }

    cmd = ["bash", dslite, "--mode", "flash", "--ccxml", ccxml, binary]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        success = result.returncode == 0
        return {
            "success":    success,
            "returncode": result.returncode,
            "stdout":     result.stdout,
            "stderr":     result.stderr,
            "out_file":   binary,
            "ccxml":      ccxml,
            "errors": [] if success else [
                f"Flash failed (exit code {result.returncode}). "
                "Check stderr. Common causes: device not connected, wrong .ccxml, "
                "USB drivers missing, or device in wrong state."
            ],
            "next_step": (
                "Flash succeeded. In CCS debug perspective: click Resume (F8) to run, "
                "set a breakpoint after the inference call, and check test_result == 1 "
                "in the Watch window to verify model inference is correct."
            ) if success else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "", "errors": ["Flash timed out. Check device connection."]}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": "", "errors": [str(e)]}


# ── Artifact timestamp validation ──────────────────────────────────────

def _get_file_mtime(path: str) -> float:
    """Get file modification time. Returns 0.0 if file doesn't exist."""
    try:
        return os.path.getmtime(path)
    except (OSError, FileNotFoundError):
        return 0.0

def _validate_copied_artifacts(src_dir: str, dst_dir: str) -> Tuple[bool, List[str]]:
    """
    Validate that files copied from src_dir to dst_dir have matching creation timestamps.

    Purpose: Ensure artifacts/golden vectors were copied from the actual run,
    not from a template or stale cache. Timestamp mismatch indicates:
    - Files came from template (old timestamp)
    - Copy operation silently failed and fell back to template
    - Files are stale/outdated

    Args:
        src_dir: Source directory (ModelMaker run artifacts or golden_vectors)
        dst_dir: Destination directory (CCS project)

    Returns:
        (success: bool, errors: List[str])
        - success=True if all files match within 2s tolerance (for filesystem granularity)
        - success=False if any file missing or timestamps diverge significantly
    """
    errors = []
    tolerance_sec = 0.5  # Allow 0.5s clock skew for filesystem operations. Larger gaps indicate stale/template files.

    if not os.path.isdir(src_dir):
        return (False, [f"Source directory does not exist: {src_dir}"])

    if not os.path.isdir(dst_dir):
        return (False, [f"Destination directory does not exist: {dst_dir}"])

    # Walk all files in destination and check they have matching source
    for root, dirs, files in os.walk(dst_dir):
        for fname in files:
            dst_file = os.path.join(root, fname)
            # Reconstruct relative path from destination root
            rel_path = os.path.relpath(dst_file, dst_dir)
            src_file = os.path.join(src_dir, rel_path)

            src_mtime = _get_file_mtime(src_file)
            dst_mtime = _get_file_mtime(dst_file)

            if src_mtime == 0.0:
                errors.append(
                    f"Source file missing: {src_file} "
                    f"(but exists in destination: {dst_file}). "
                    f"Artifacts may have been copied from template, not from actual run."
                )
            elif abs(src_mtime - dst_mtime) > tolerance_sec:
                errors.append(
                    f"Timestamp mismatch for {rel_path}: "
                    f"source={src_mtime:.1f}, dest={dst_mtime:.1f} "
                    f"(diff={abs(src_mtime - dst_mtime):.1f}s). "
                    f"File may be from template or stale cache."
                )

    return (len(errors) == 0, errors)


# ── create_ccs_project (template-based) ────────────────────────────────

class DeviceDeployer:
    """Template-based CCS project creator. Uses SDK AI example templates."""

    def __init__(self, run_id, task_type, quantization, model_id, tinyml_base_path=None):
        self.run_id       = run_id
        self.quantization = quantization
        self.model_id     = model_id
        self.task_type    = task_type
        runs_path         = _runs_path(tinyml_base_path)
        self.artifacts_dir   = os.path.join(runs_path, task_type, "run", run_id, model_id, "compilation", "artifacts")
        self.golden_dir      = _golden_dir(tinyml_base_path, task_type, run_id, model_id, quantization)

    def create_new_ccs_project(self, project_name, device_type, ccs_templates_path):
        """
        Args:
            ccs_templates_path: Absolute path to the SDK's AI examples directory
                (e.g., C2000Ware_6.xx/libraries/ai/examples).
                Must be resolved by the caller using check_sdk_installation or explicit sdk_path.
                Project will be created as a sibling in this directory.

        Returns:
            Tuple[str, Dict[str, Any]]: (project_path, validation_result)
            validation_result has keys: timestamp_validation_passed (bool), validation_errors (List[str])
        """
        templates_path = os.path.expanduser(ccs_templates_path)

        task_normalized    = self.task_type.lower().replace(" ", "_")
        task_type_base_dir = os.path.join(templates_path, task_normalized)
        task_type_dev_dir  = os.path.join(task_type_base_dir, device_type)

        if not os.path.exists(task_type_dev_dir):
            raise FileNotFoundError(
                f"Template not found: {task_type_dev_dir}\n"
                f"Searched SDK AI examples at: {templates_path}\n"
                "Verify the SDK is installed and ccs_templates_path points to its AI examples folder."
            )

        new_project_path = os.path.join(templates_path, project_name)
        if os.path.exists(new_project_path):
            raise FileExistsError(f"Project already exists: {new_project_path}")

        os.makedirs(templates_path, exist_ok=True)
        device_folder = os.path.join(new_project_path, device_type)
        shutil.copytree(task_type_dev_dir, device_folder)

        app_main_src = os.path.join(task_type_base_dir, "application_main.c")
        if os.path.exists(app_main_src):
            shutil.copy2(app_main_src, os.path.join(new_project_path, "application_main.c"))

        ccs_dir       = os.path.join(device_folder, "CCS")
        old_spec      = os.path.join(ccs_dir, f"{device_type}_{task_normalized}.projectspec")
        new_spec      = os.path.join(ccs_dir, f"{device_type}_{project_name}.projectspec")

        if os.path.exists(old_spec):
            self._update_projectspec(old_spec, new_spec, project_name, task_normalized)
            os.remove(old_spec)
        else:
            raise FileNotFoundError(f"Projectspec not found: {old_spec}")

        # Copy compilation artifacts and the 4 required ModelMaker files
        artifacts_copied = self._copy_artifacts(device_folder)
        golden_copied = self._copy_golden_files(device_folder)

        # Validate timestamps to ensure artifacts came from actual run, not template
        validation_result = {"timestamp_validation_passed": True, "validation_errors": []}

        if artifacts_copied:
            artifacts_dst = os.path.join(device_folder, "artifacts")
            passed, errors = _validate_copied_artifacts(self.artifacts_dir, artifacts_dst)
            if not passed:
                validation_result["timestamp_validation_passed"] = False
                validation_result["validation_errors"].extend([f"Artifacts: {e}" for e in errors])

        if golden_copied:
            golden_dst = device_folder
            passed, errors = _validate_copied_artifacts(self.golden_dir, golden_dst)
            if not passed:
                validation_result["timestamp_validation_passed"] = False
                validation_result["validation_errors"].extend([f"Golden vectors: {e}" for e in errors])

        return new_project_path, validation_result

    def _update_projectspec(self, old_path, new_path, project_name, task_normalized):
        tree = ET.parse(old_path)
        root = tree.getroot()
        for elem in root.iter():
            if elem.text and isinstance(elem.text, str):
                elem.text = elem.text.replace(task_normalized, project_name)
            for k, v in elem.attrib.items():
                if isinstance(v, str):
                    elem.set(k, v.replace(task_normalized, project_name))
        tree.write(new_path, encoding="utf-8", xml_declaration=False)

    def _copy_artifacts(self, project_path):
        """Copy compiled artifacts from ModelMaker run. Returns True if copy succeeded."""
        src = os.path.expanduser(self.artifacts_dir)
        dst = os.path.join(project_path, "artifacts")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        if os.path.exists(src):
            shutil.copytree(src, dst)
            return True
        return False

    def _copy_golden_files(self, project_root):
        """Copy test_vector.c and user_input_config.h from ModelMaker run. Returns True if any files copied."""
        src = os.path.expanduser(self.golden_dir)
        if not os.path.isdir(src):
            return False
        files_copied = False
        for fname in ("test_vector.c", "user_input_config.h"):
            src_file = os.path.join(src, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, os.path.join(project_root, fname))
                files_copied = True
        return files_copied


def create_ccs_project(
    project_name: str,
    device_type: str,
    run_id: str,
    task_type: str,
    quantization: bool,
    model_id: str,
    target_device: str,
    tinyml_base_path: Optional[str] = None,
    sdk_path: Optional[str] = None,
    ccs_templates_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: Create a CCS project from a device-family SDK template with compiled artifacts.

    Creates the project as a sibling in the SDK's AI examples directory.
    Automatically detects which SDK is required based on target_device (device family),
    searches common installation paths, and returns a helpful error with download URL
    if the SDK is not installed.

    Args:
        project_name: Name for the new CCS project folder (created in ai/examples/).
        device_type: CCS device variant string (e.g., 'f28p55x', 'f28p65x', 'mspm0g3507').
        run_id: Run directory name from training output.
        task_type: Task type string (e.g., 'motor_fault').
        quantization: True if quantized model artifacts should be used.
        model_id: Model name from training artifacts.
        target_device: Canonical device ID (e.g., 'F28P55', 'MSPM0G3507') — used to
            determine device family and required SDK.
        tinyml_base_path: Path to tinyml-tensorlab root. Defaults to ~/tinyml-tensorlab.
        sdk_path: Optional explicit path to SDK root (overrides auto-detection).
            E.g., '/home/user/ti/c2000ware_6.00.00.00'
        ccs_templates_path: Optional explicit path to SDK AI examples directory
            (overrides both sdk_path and auto-detection).

    Returns:
        Dict with success, project_path, family, sdk_name, errors.
        Errors include SDK download URL if SDK not found.
        On success, call build_ccs_project and flash_ccs_project to complete deployment.
    """
    # Resolve templates path: explicit > sdk_path+subpath > auto-detect by family
    if ccs_templates_path:
        templates_path = os.path.expanduser(ccs_templates_path)
        sdk_check = {"found": True, "family": None, "sdk_name": "custom", "sdk_root": None}
    else:
        sdk_check = check_sdk_installation(target_device, sdk_path)
        if not sdk_check["found"]:
            return {
                "success":      False,
                "project_path": None,
                "family":       sdk_check.get("family"),
                "sdk_name":     sdk_check.get("sdk_name"),
                "errors":       sdk_check["errors"],
            }
        if not sdk_check.get("ai_examples_path"):
            return {
                "success":      False,
                "project_path": None,
                "family":       sdk_check["family"],
                "sdk_name":     sdk_check["sdk_name"],
                "errors":       sdk_check["errors"],
            }
        templates_path = sdk_check["ai_examples_path"]

    try:
        deployer = DeviceDeployer(run_id, task_type, quantization, model_id, tinyml_base_path)
        project_path, validation_result = deployer.create_new_ccs_project(project_name, device_type, templates_path)

        result = {
            "success":      True,
            "project_path": project_path,
            "family":       sdk_check.get("family"),
            "sdk_name":     sdk_check.get("sdk_name"),
            "sdk_root":     sdk_check.get("sdk_root"),
            "errors":       [],
            "timestamp_validation_passed": validation_result["timestamp_validation_passed"],
            "validation_warnings": validation_result["validation_errors"],
            "next_step":    f"Build the project: run build_ccs_project with ccs_project_path='{project_path}'.",
        }

        # If timestamp validation failed, add critical warning but don't fail (user can manually verify)
        if not validation_result["timestamp_validation_passed"]:
            result["errors"].extend(validation_result["validation_errors"])
            result["critical_warning"] = (
                "TIMESTAMP VALIDATION FAILED: Copied artifacts or golden vectors do not match source timestamps. "
                "This may indicate files were copied from template or cache, not from actual run. "
                "VERIFY MANUALLY before proceeding to build: compare artifact mtimes in source and destination."
            )

        return result
    except FileNotFoundError as e:
        family  = sdk_check.get("family")
        sdk_info = SDK_INFO.get(family, {}) if family else {}
        errors = [str(e)]
        if sdk_info.get("download_url"):
            errors.append(
                f"If this is an SDK installation issue, ensure {sdk_info['name']} "
                f"is fully installed: {sdk_info['download_url']}"
            )
        return {"success": False, "project_path": None, "family": family, "errors": errors}
    except FileExistsError as e:
        return {"success": False, "project_path": None, "errors": [str(e)]}
    except Exception as e:
        return {"success": False, "project_path": None, "errors": [f"Unexpected error: {e}"]}
