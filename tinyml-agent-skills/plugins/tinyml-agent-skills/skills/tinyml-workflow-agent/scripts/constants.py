# All valid task types and their required target_module
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv

# Load .env from ~/.tinyml-agent-skills/ explicitly to support execution from any directory
_env_file = Path.home() / ".tinyml-agent-skills" / ".env"
load_dotenv(_env_file)                                                                                                                                                          
# Note: Feature extraction presets, transforms, augmenters, and task recommendations
# have been moved to schema.yaml and are accessed via feature_schema module.
# This keeps constants.py focused on device support and task mappings.

TASK_TYPE_TO_MODULE = {
    # Timeseries generic tasks
    "generic_timeseries_classification": "timeseries",
    "generic_timeseries_regression": "timeseries",
    "generic_timeseries_forecasting": "timeseries",
    "generic_timeseries_anomalydetection": "timeseries",
    # Timeseries specialized tasks
    "motor_fault": "timeseries",
    "ecg_classification": "timeseries",
    "arc_fault": "timeseries",
    "blower_imbalance": "timeseries",
    "pir_detection": "timeseries",
    # Vision tasks
    "image_classification": "vision",
}


### Dataset section constants

SPLIT_TYPES = [
    "amongst_files",
    "within_files"
]

# ─── Context paths for the agent to read when answering user questions ────────
# Paths are local — resolved relative to this file's skill root.
import os as _os
_SKILL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_REFS_DIR  = _os.path.join(_SKILL_DIR, "references")
_ASSETS_DIR = _os.path.join(_SKILL_DIR, "assets")

TINYML_MODELZOO_PATH = _os.environ.get("TINYML_MODELZOO_PATH")

if not TINYML_MODELZOO_PATH:
    raise RuntimeError(
        "Environment variable TINYML_MODELZOO_PATH must be set. "
        "Run: /tinyml-agent-skills:setup"
    )

CONTEXT_PATHS = {
    "fe_transforms_doc":  _os.path.join(_REFS_DIR, "FE_and_Data_Processing_Transforms", "FE_transforms.md"),
    "data_proc_doc":      _os.path.join(_REFS_DIR, "FE_and_Data_Processing_Transforms", "Data_processing_transforms.md"),
    "examples_dir":       _os.path.join(TINYML_MODELZOO_PATH, "examples"),
}


# ── Constants ────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".npy", ".pkl"}
CLASSIFICATION_DATA_DIR = "classes"
FILES_DATA_DIR = "files"
ANNOTATION_DIR = "annotations"
ANOMALY_NORMAL_CLASS = "Normal"
ANOMALY_ANOMALY_CLASS = "Anomaly"

CLASSIFICATION_TASKS = {
    "generic_timeseries_classification",
    "motor_fault",
    "ecg_classification",
    "arc_fault",
    "blower_imbalance",
    "pir_detection",
}
ANOMALY_TASKS = {"generic_timeseries_anomalydetection"}
REGRESSION_TASKS = {"generic_timeseries_regression"}
FORECASTING_TASKS = {"generic_timeseries_forecasting"}


EXPECTED_STRUCTURES = {
    "classification": (
        "dataset_root/\n"
        "├── classes/\n"
        "│   ├── <class_A>/\n"
        "│   │   ├── class_A.csv\n"
        "│   └── <class_B>/\n"
        "│       └── class_B.csv\n"
        "└── metadata.json  ← optional\n"
        "ZIP rule: classes/ must be at top level of the ZIP (no extra nesting)."
    ),
    "anomaly_detection": (
        "dataset_root/\n"
        "├── classes/\n"
        "│   ├── Normal/         ← EXACT name, case-sensitive (used for training)\n"
        "│   │   ├── normal.csv\n"
        "│   └── Anomaly/        ← EXACT name, case-sensitive (evaluation only, optional)\n"
        "│       ├── fault/anomaly.csv\n"
        "└── annotations/        ← optional, auto-generated after windowing\n"
        "Normal/ and Anomaly/ must be inside classes/ (not at the dataset root)."
    ),
    "regression": (
        "dataset_root/\n"
        "├── files/          ← MUST be named 'files'\n"
        "│   ├── signal.csv\n"
        "├── annotations.json  ← optional\n"
        "└── metadata.json     ← optional"
    ),
    "forecasting": (
        "dataset_root/\n"
        "├── files/          ← MUST be named 'files'\n"
        "│   ├── signal.csv\n"
        "├── annotations.json  ← optional\n"
        "└── metadata.json     ← optional\n"
        "Target column set via target_variables config param (not implied by position)."
    ),
}