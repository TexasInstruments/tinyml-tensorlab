"""
Dataset format validation and reorganization tools.

Required dataset layouts per task type (from config_creation_guide.md / BYOD documentation):

CLASSIFICATION (generic_timeseries_classification, motor_fault, ecg_classification,
                arc_fault, blower_imbalance, pir_detection):
    dataset_root/
    ├── classes/
    │   ├── <class_A>/       ← folder name = class label
    │   │   ├── sample_1.csv
    │   │   └── sample_2.csv
    │   └── <class_B>/
    │       └── sample_3.csv
    └── metadata.json        ← optional
    ZIP: classes/ must be at ZIP top level (no extra nesting directory).

ANOMALY DETECTION (generic_timeseries_anomalydetection):
    dataset_root/
    ├── classes/
    │   ├── Normal/              ← EXACT name, case-sensitive (training data)
    │   │   ├── normal_1.csv
    │   │   └── normal_2.csv
    │   └── Anomaly/             ← EXACT name, case-sensitive (evaluation only, optional)
    │       ├── fault_1.csv
    │       └── fault_2.csv
    └── annotations/             ← optional, auto-generated after windowing

REGRESSION (generic_timeseries_regression):
    dataset_root/
    ├── files/               ← MUST be named "files"
    │   ├── signal_1.csv
    │   └── signal_2.csv
    ├── annotations.json     ← optional
    └── metadata.json        ← optional

FORECASTING (generic_timeseries_forecasting):
    Identical layout to regression.
    Target column set via target_variables config param (not implied by position).
"""

import os
import io
import re
import csv
import random
import shutil
import zipfile
from typing import Optional, Dict, List, Tuple, Any, Set
from pathlib import Path
from constants import *
import glob

# ── Expected structure descriptions (matches config_creation_guide.md) ───────
_EXPECTED_STRUCTURES = {
    "classification": (
        "dataset_root/\n"
        "├── classes/\n"
        "│   ├── <class_A>/\n"
        "│   │   ├── sample_1.csv\n"
        "│   │   └── sample_2.csv\n"
        "│   └── <class_B>/\n"
        "│       └── sample_3.csv\n"
        "└── metadata.json  ← optional\n"
        "ZIP rule: classes/ must be at top level of the ZIP (no extra nesting)."
    ),
    "anomaly_detection": (
        "dataset_root/\n"
        "├── classes/\n"
        "│   ├── Normal/         ← EXACT name, case-sensitive (used for training)\n"
        "│   │   ├── normal_1.csv\n"
        "│   │   └── normal_2.csv\n"
        "│   └── Anomaly/        ← EXACT name, case-sensitive (evaluation only, optional)\n"
        "│       ├── fault_1.csv\n"
        "│       └── fault_2.csv\n"
        "└── annotations/        ← optional, auto-generated after windowing\n"
        "Normal/ and Anomaly/ must be inside classes/ (not at the dataset root)."
    ),
    "regression": (
        "dataset_root/\n"
        "├── files/          ← MUST be named 'files'\n"
        "│   ├── signal_1.csv\n"
        "│   └── signal_2.csv\n"
        "├── annotations.json  ← optional\n"
        "└── metadata.json     ← optional"
    ),
    "forecasting": (
        "dataset_root/\n"
        "├── files/          ← MUST be named 'files'\n"
        "│   ├── signal_1.csv\n"
        "│   └── signal_2.csv\n"
        "├── annotations.json  ← optional\n"
        "└── metadata.json     ← optional\n"
        "Target column set via target_variables config param (not implied by position)."
    ),
}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _get_task_family(task_type: str) -> str:
    if task_type in ANOMALY_TASKS:
        return "anomaly_detection"
    if task_type in FORECASTING_TASKS:
        return "forecasting"
    if task_type in REGRESSION_TASKS:
        return "regression"
    return "classification"


def _is_supported(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_EXTENSIONS


def _safe_label(label: str) -> str:
    """Convert class label string to a safe directory name."""
    safe = re.sub(r"[^\w\-]", "_", label.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe[:64] or "unknown"


def _list_data_files(dir_path: str) -> List[str]:
    """Immediate supported data files in dir_path (non-recursive)."""
    return sorted(f for f in os.listdir(dir_path) if _is_supported(f))


# ── ZIP introspection ─────────────────────────────────────────────────────────

def _norm(p: str) -> str:
    return p.replace("\\", "/")


def _zip_top_level(zip_path: str) -> Tuple[Set[str], Set[str]]:
    """Return (top_level_dirs, top_level_files) in a ZIP."""
    dirs: Set[str] = set()
    files: Set[str] = set()
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            name = _norm(name)
            parts = name.rstrip("/").split("/")
            if len(parts) == 1:
                if name.endswith("/"):
                    dirs.add(parts[0])
                else:
                    files.add(parts[0])
            elif len(parts) >= 2:
                dirs.add(parts[0])
    return dirs, files


def _zip_list_under(zip_path: str, prefix: str) -> Tuple[List[str], List[str]]:
    """Return (immediate_subdirs, immediate_files) directly under prefix in ZIP."""
    prefix = _norm(prefix).rstrip("/") + "/"
    subdirs: Set[str] = set()
    files: List[str] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            name = _norm(name)
            if not name.startswith(prefix):
                continue
            remainder = name[len(prefix):]
            if not remainder:
                continue
            parts = remainder.rstrip("/").split("/")
            if len(parts) == 1:
                if remainder.endswith("/"):
                    subdirs.add(parts[0])
                else:
                    files.append(parts[0])
            elif len(parts) >= 2:
                subdirs.add(parts[0])
    return sorted(subdirs), files


def _zip_detect_nesting(zip_path: str) -> Optional[str]:
    """If all ZIP entries share one top-level wrapper dir, return that dir name."""
    top_dirs, top_files = _zip_top_level(zip_path)
    if top_files:
        return None
    if len(top_dirs) == 1:
        return list(top_dirs)[0]
    return None


def _extract_zip(zip_path: str, dest_dir: str) -> str:
    """Extract ZIP to dest_dir, normalising Windows separators. Returns dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            member.filename = _norm(member.filename)
            z.extract(member, dest_dir)
    return dest_dir

# ── Sub-validators ────────────────────────────────────────────────────────────

def _check_classification(path: str, is_zip: bool) -> Tuple[List[str], List[str], bool, str]:
    """Returns (errors, warnings, can_auto_reorganize, reorganize_hints)."""
    errors, warnings = [], []

    if is_zip:
        top_dirs, top_files = _zip_top_level(path)

        if CLASSIFICATION_DATA_DIR not in top_dirs:
            nesting = _zip_detect_nesting(path)
            if nesting:
                inner_dirs, _ = _zip_list_under(path, nesting)
                if CLASSIFICATION_DATA_DIR in inner_dirs:
                    errors.append(
                        f"ZIP has extra nesting: '{nesting}/classes/' found. "
                        "classes/ must be at ZIP top level (not inside a wrapper folder)."
                    )
                    return errors, warnings, True, ""
            errors.append(
                "ZIP missing top-level 'classes/' directory."
            )
            return errors, warnings, False, (
                "Provide a ZIP where classes/ sits at the top level. "
                "Do not wrap it in an extra folder like 'dataset_name/classes/'."
            )

        class_dirs, class_files = _zip_list_under(path, CLASSIFICATION_DATA_DIR)
        if class_files:
            warnings.append(
                f"Files found directly in classes/ (e.g. {class_files[:2]}). "
                "Data files should be inside class subdirectories."
            )
        if not class_dirs:
            errors.append(
                "classes/ exists but contains no class subdirectories. "
                "Create one subdirectory per class (folder name = class label)."
            )
            return errors, warnings, False, (
                "Create subdirectories inside classes/, one per class. "
                "Move data files into the appropriate class subdirectory."
            )
        if len(class_dirs) < 2:
            warnings.append(
                f"Only 1 class found ('{class_dirs[0]}'). "
                "Classification needs ≥2 classes for meaningful training."
            )
        for cls in class_dirs:
            _, cls_files = _zip_list_under(path, f"{CLASSIFICATION_DATA_DIR}/{cls}")
            if not any(_is_supported(f) for f in cls_files):
                errors.append(
                    f"classes/{cls}/ contains no supported data files (.csv/.txt/.npy/.pkl)."
                )

    else:
        classes_dir = os.path.join(path, CLASSIFICATION_DATA_DIR)
        if not os.path.isdir(classes_dir):
            # Check if CSVs exist at root level or in subdirs with a different parent name
            root_csvs = _list_data_files(path)
            root_subdirs = [
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")
            ]
            if root_csvs:
                errors.append(
                    "No 'classes/' directory found. "
                    "CSV files are at the root level — they need to be inside "
                    "classes/<class_name>/ subdirectories."
                )
                return errors, warnings, True, (
                    "Provide label_column so files can be split by class label, "
                    "OR manually create classes/<class_name>/ subdirectories."
                )
            if root_subdirs:
                errors.append(
                    f"No 'classes/' directory. Found subdirectories: {root_subdirs}. "
                    "If these subdirectories represent classes, rename the parent "
                    "structure so that classes/ contains them."
                )
                return errors, warnings, True, ""
            errors.append("No 'classes/' directory and no recognisable data structure found.")
            return errors, warnings, False, "Check that input_data_path points to the correct directory."

        class_dirs = sorted(
            d for d in os.listdir(classes_dir)
            if os.path.isdir(os.path.join(classes_dir, d)) and not d.startswith(".")
        )
        if not class_dirs:
            # Files might be flat inside classes/
            flat_csvs = _list_data_files(classes_dir)
            if flat_csvs:
                errors.append(
                    f"classes/ contains CSV files directly (e.g. {flat_csvs[:2]}) "
                    "instead of class subdirectories. "
                    "Each class needs its own subdirectory inside classes/."
                )
                return errors, warnings, True, (
                    "Provide label_column to split CSVs into class subdirectories."
                )
            errors.append("classes/ is empty — no class subdirectories or files found.")
            return errors, warnings, False, "Add data files in class subdirectories under classes/."

        if len(class_dirs) < 2:
            warnings.append(
                f"Only 1 class found ('{class_dirs[0]}'). "
                "Classification needs ≥2 classes."
            )
        for cls in class_dirs:
            cls_path = os.path.join(classes_dir, cls)
            if not any(_is_supported(f) for f in os.listdir(cls_path)):
                errors.append(
                    f"classes/{cls}/ contains no supported data files (.csv/.txt/.npy/.pkl)."
                )

    return errors, warnings, len(errors) == 0, ""


def _check_anomaly(path: str, is_zip: bool) -> Tuple[List[str], List[str], bool, str]:
    """
    Validate anomaly detection dataset structure.
    Expected: Normal/ and Anomaly/ inside classes/ at dataset root.
    """
    errors, warnings = [], []
    can_fix = False

    if is_zip:
        top_dirs, _ = _zip_top_level(path)

        # Root-level Normal/Anomaly is the wrong layout — fixable
        if any(d.lower() == "normal" for d in top_dirs):
            errors.append(
                "ZIP has Normal/Anomaly at top level but they must be inside 'classes/'. "
                "Restructure so Normal/ and Anomaly/ are inside classes/ in the ZIP."
            )
            return errors, warnings, True, ""

        if CLASSIFICATION_DATA_DIR not in top_dirs:
            nesting = _zip_detect_nesting(path)
            if nesting:
                inner_dirs, _ = _zip_list_under(path, nesting)
                if CLASSIFICATION_DATA_DIR in inner_dirs:
                    errors.append(
                        f"ZIP has extra nesting: '{nesting}/classes/' found. "
                        "classes/ must be at ZIP top level (not inside a wrapper folder)."
                    )
                    return errors, warnings, True, ""
            errors.append("ZIP missing top-level 'classes/' directory.")
            return errors, warnings, False, (
                "Create classes/Normal/ and classes/Anomaly/ directories in the ZIP."
            )

        inner_dirs, _ = _zip_list_under(path, CLASSIFICATION_DATA_DIR)
        inner_lower = {d.lower(): d for d in inner_dirs}

        for expected in [ANOMALY_NORMAL_CLASS, ANOMALY_ANOMALY_CLASS]:
            if expected in inner_dirs:
                continue
            lower = expected.lower()
            if lower in inner_lower:
                errors.append(
                    f"Found 'classes/{inner_lower[lower]}/' but must be exactly "
                    f"'classes/{expected}/' (case-sensitive)."
                )
                can_fix = True
            elif expected == ANOMALY_ANOMALY_CLASS:
                warnings.append(
                    "No 'classes/Anomaly/' folder found. "
                    "Model can still train on Normal/ data only, "
                    "but anomaly detection performance cannot be evaluated without anomaly samples."
                )
            else:
                errors.append(
                    f"Missing 'classes/{expected}/' folder (exact name required)."
                )

        if ANOMALY_NORMAL_CLASS in inner_dirs:
            _, normal_files = _zip_list_under(path, f"{CLASSIFICATION_DATA_DIR}/{ANOMALY_NORMAL_CLASS}")
            if not any(_is_supported(f) for f in normal_files):
                errors.append(f"'classes/{ANOMALY_NORMAL_CLASS}/' contains no supported data files.")

    else:
        classes_dir = os.path.join(path, CLASSIFICATION_DATA_DIR)
        root_dirs = {
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")
        }

        # Root-level Normal/Anomaly is the wrong layout — fixable
        if any(d.lower() == "normal" for d in root_dirs):
            errors.append(
                "Found Normal/Anomaly at dataset root but they must be inside 'classes/'. "
                "Move Normal/ and Anomaly/ into a classes/ directory."
            )
            return errors, warnings, True, ""

        if not os.path.isdir(classes_dir):
            root_csvs = _list_data_files(path)
            if root_csvs:
                errors.append(
                    "No 'classes/' directory found. "
                    "CSV files are at root — need to be inside classes/Normal/ or classes/Anomaly/."
                )
                return errors, warnings, True, (
                    "Provide normal_pattern and anomaly_pattern regex strings "
                    "to sort files into classes/Normal/ and classes/Anomaly/."
                )
            errors.append("No 'classes/' directory and no recognisable data structure found.")
            return errors, warnings, False, "Check that input_data_path points to the correct directory."

        inner = {
            d for d in os.listdir(classes_dir)
            if os.path.isdir(os.path.join(classes_dir, d)) and not d.startswith(".")
        }
        inner_lower = {d.lower(): d for d in inner}

        for expected in [ANOMALY_NORMAL_CLASS, ANOMALY_ANOMALY_CLASS]:
            if expected in inner:
                continue
            lower = expected.lower()
            if lower in inner_lower:
                errors.append(
                    f"Found 'classes/{inner_lower[lower]}/' but must be exactly "
                    f"'classes/{expected}/' (case-sensitive)."
                )
                can_fix = True
            elif expected == ANOMALY_ANOMALY_CLASS:
                warnings.append(
                    "No 'classes/Anomaly/' folder found. "
                    "Model can still train on Normal/ data only, "
                    "but anomaly detection performance cannot be evaluated without anomaly samples."
                )
            else:
                errors.append(
                    f"Missing 'classes/{expected}/' folder (exact name required)."
                )

        normal_path = os.path.join(classes_dir, ANOMALY_NORMAL_CLASS)
        if os.path.isdir(normal_path) and not any(_is_supported(f) for f in os.listdir(normal_path)):
            errors.append(f"'classes/{ANOMALY_NORMAL_CLASS}/' contains no supported data files.")

    hints = "" if can_fix else (
        "Provide normal_pattern and anomaly_pattern (Python regex strings) "
        "matching filenames of normal and anomaly samples. "
        "Example: normal_pattern='normal_.*', anomaly_pattern='fault_.*'"
    )
    return errors, warnings, can_fix, hints if errors else ""


def _check_regression(path: str, is_zip: bool) -> Tuple[List[str], List[str], bool, str]:
    errors, warnings = [], []

    if is_zip:
        top_dirs, top_files = _zip_top_level(path)
        if FILES_DATA_DIR not in top_dirs:
            if CLASSIFICATION_DATA_DIR in top_dirs:
                errors.append(
                    "ZIP has 'classes/' directory but regression/forecasting requires 'files/'. "
                    "Rename the data directory to 'files/' and remove class subdirectories."
                )
                return errors, warnings, True, ""

            supported_at_root = [f for f in top_files if _is_supported(f)]
            if supported_at_root:
                errors.append(
                    f"ZIP has data files at top level (e.g. {supported_at_root[:2]}) "
                    "but needs them inside a 'files/' subdirectory."
                )
                return errors, warnings, True, ""

            errors.append(
                f"ZIP missing top-level 'files/' directory. "
                "Regression/forecasting data must be in a 'files/' subdirectory."
            )
            return errors, warnings, False, "Create a 'files/' directory at ZIP top level and place all CSV files inside it."

        _, data_files = _zip_list_under(path, FILES_DATA_DIR)
        if not any(_is_supported(f) for f in data_files):
            errors.append(f"'{FILES_DATA_DIR}/' directory contains no supported data files.")

        ann_exists = any(_norm(n).startswith(f"{ANNOTATION_DIR}/") for n in
                         __import__("zipfile").ZipFile(path).namelist())
        if not ann_exists:
            warnings.append(
                "No annotations/ directory in ZIP. "
                "ModelMaker will auto-generate train/val splits using split_factor from config."
            )
        else:
            _, ann_files = _zip_list_under(path, ANNOTATION_DIR)
            if "instances_train_list.txt" not in ann_files:
                warnings.append("annotations/ present but missing instances_train_list.txt.")
            if "instances_val_list.txt" not in ann_files:
                warnings.append("annotations/ present but missing instances_val_list.txt.")

    else:
        files_dir = os.path.join(path, FILES_DATA_DIR)
        if not os.path.isdir(files_dir):
            root_csvs = _list_data_files(path)
            root_subdirs = [
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")
                and d not in {CLASSIFICATION_DATA_DIR, ANNOTATION_DIR}
            ]
            if root_csvs:
                errors.append(
                    f"Data files found at root level (e.g. {root_csvs[:2]}) "
                    "but regression/forecasting requires them inside a 'files/' subdirectory."
                )
                return errors, warnings, True, ""
            if root_subdirs:
                errors.append(
                    f"Found subdirectories {root_subdirs} but no 'files/' directory. "
                    "Rename the data subdirectory to 'files/'."
                )
                return errors, warnings, True, ""
            errors.append(
                "Missing 'files/' directory. "
                "Regression/forecasting datasets must place all CSV files in a 'files/' subdirectory."
            )
            return errors, warnings, False, "Create a 'files/' directory and move all CSV files into it."

        if not any(_is_supported(f) for f in os.listdir(files_dir)):
            errors.append(f"'{FILES_DATA_DIR}/' directory contains no supported data files.")

        ann_dir = os.path.join(path, ANNOTATION_DIR)
        if not os.path.isdir(ann_dir):
            warnings.append(
                "No annotations/ directory found. "
                "ModelMaker will auto-generate train/val splits using split_factor from config. "
                "Call reorganize_dataset to generate annotation files explicitly."
            )
        else:
            if not os.path.exists(os.path.join(ann_dir, "instances_train_list.txt")):
                warnings.append("annotations/ present but missing instances_train_list.txt.")
            if not os.path.exists(os.path.join(ann_dir, "instances_val_list.txt")):
                warnings.append("annotations/ present but missing instances_val_list.txt.")

    return errors, warnings, len(errors) == 0, ""


# ── Tool 1: validate_dataset_format ─────────────────────────────────────────

def validate_dataset_format(
    input_data_path: str,
    task_type: str,
) -> Dict[str, Any]:
    """
    Tool: Validate that the dataset at input_data_path matches the required BYOD
    format for the given task_type.

    Call this tool immediately after the user provides input_data_path, BEFORE
    generating the dataset config section. If validation fails (is_valid=False),
    call reorganize_dataset to fix the layout, then continue with config generation
    using the new_input_data_path returned by reorganize_dataset.

    Supported input: local directory path or local .zip file.
    Remote URLs cannot be validated — skip validation and proceed with config generation.

    Args:
        input_data_path: Local path to dataset directory or .zip file.
        task_type: Task type string (e.g., 'motor_fault', 'generic_timeseries_classification',
                   'generic_timeseries_forecasting', 'generic_timeseries_anomalydetection').

    Returns:
        Dict with:
        - is_valid: True if format is correct, False if issues found
        - format_type: format family inferred from task_type
          ('classification' | 'anomaly_detection' | 'regression' | 'forecasting')
        - expected_structure: multiline string showing the required directory layout
          — ALWAYS show this to the user when validation fails
        - issues: list of errors blocking correct data loading
        - warnings: list of non-fatal issues (ModelMaker may still work)
        - can_auto_reorganize: True if reorganize_dataset can fix all issues automatically
        - reorganize_hints: instructions for the user if can_auto_reorganize is False
          (e.g., which parameters to provide to reorganize_dataset)
        - success: True if the validation check itself completed without internal errors
    """
    if input_data_path.startswith("http://") or input_data_path.startswith("https://"):
        return {
            "is_valid": None,
            "format_type": _get_task_family(task_type),
            "expected_structure": _EXPECTED_STRUCTURES[_get_task_family(task_type)],
            "issues": [],
            "warnings": ["Remote URL provided — format validation skipped. Ensure dataset matches expected structure."],
            "can_auto_reorganize": False,
            "reorganize_hints": "Download the dataset locally to run format validation.",
            "success": True,
        }

    if not os.path.exists(input_data_path):
        return {
            "is_valid": False,
            "format_type": _get_task_family(task_type),
            "expected_structure": _EXPECTED_STRUCTURES[_get_task_family(task_type)],
            "issues": [f"Path does not exist: {input_data_path}"],
            "warnings": [],
            "can_auto_reorganize": False,
            "reorganize_hints": "Verify the path and try again.",
            "success": True,
        }

    is_zip = input_data_path.lower().endswith(".zip")
    family = _get_task_family(task_type)

    try:
        if family == "classification":
            errors, warnings, can_fix, hints = _check_classification(input_data_path, is_zip)
        elif family == "anomaly_detection":
            errors, warnings, can_fix, hints = _check_anomaly(input_data_path, is_zip)
        else:
            errors, warnings, can_fix, hints = _check_regression(input_data_path, is_zip)
    except Exception as e:
        return {
            "is_valid": False,
            "format_type": family,
            "expected_structure": _EXPECTED_STRUCTURES[family],
            "issues": [f"Validation failed with internal error: {e}"],
            "warnings": [],
            "can_auto_reorganize": False,
            "reorganize_hints": "",
            "success": False,
        }

    return {
        "is_valid": len(errors) == 0,
        "format_type": family,
        "expected_structure": _EXPECTED_STRUCTURES[family],
        "issues": errors,
        "warnings": warnings,
        "can_auto_reorganize": can_fix if errors else True,
        "reorganize_hints": hints if errors else "",
        "success": True,
    }