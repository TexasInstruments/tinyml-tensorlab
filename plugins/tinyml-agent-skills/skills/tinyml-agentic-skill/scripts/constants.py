# All valid task types and their required target_module
from typing import Dict
from dotenv import load_dotenv
load_dotenv()                                                                                                                                                          
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

# Device support matrix for specialized tasks
DEVICE_TASK_SUPPORT = {
    "arc_fault": ["F280013", "F280015", "F28003", "F28004", "F2837", "F28P55", "F28P65", "MSPM0G3507", "MSPM0G3519", "MSPM0G5187", "MSPM33C32", "F29H85", "AM13E2", "AM263"],
    "ecg_classification": ["MSPM0G3507", "MSPM0G5187", "MSPM0G3519"],
    "motor_fault": ["F280013", "F280015", "F28003", "F28004", "F2837", "F28P55", "F28P65", "MSPM0G3507", "MSPM0G3519", "MSPM0G5187", "MSPM33C32", "F29H85", "AM13E2", "AM263"],
    "blower_imbalance": ["F280013", "F280015", "F28003", "F28004", "F2837", "F28P55", "F28P65", "F29H85", "MSPM33C32", "AM13E2", "AM263"],
    "pir_detection": ["CC2755", "CC1352", "CC1354", "CC35X1", "MSPM0G5187", "MSPM0G3507", "MSPM0G3519", "MSPM33C32"],
    "image_classification": ["F280013", "F280015", "F28003", "F28004", "F2837", "F28P55", "F28P65", "F29H85", "F29P58", "F29P32"],
}

# All supported target devices
ALL_TARGET_DEVICES = [
    # C2000 DSP Family
    "F280013", "F280015", "F28003", "F28004", "F2837", "F28P55", "F28P65", "F29H85", "F29P58", "F29P32",
    # MSPM0 Family
    "MSPM0G3507", "MSPM0G3519", "MSPM0G5187",
    # MSPM33 Family
    "MSPM33C32", "MSPM33C34",
    # AM13 Family
    "AM13E2",
    # AM26x Family
    "AM263", "AM263P", "AM261",
    # Connectivity Devices
    "CC2755", "CC1352", "CC1354", "CC35X1",
]

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

TINYML_TENSORLAB_PATH = _os.environ.get("TINYML_BASE_PATH")
TINYML_DOCS_PATH = _os.environ.get("TINYML_TENSORLAB_DOCS_PATH")

CONTEXT_PATHS = {
    # Feature extraction documentation (local .rst) -> to be changed
    "feat_ext_docs":    _os.path.join(TINYML_DOCS_PATH, "source", "features", "feature_extraction.rst"),
    # Timeseries module constants — presets, feat_ext transforms (local .md)
    "feat_ext_presets": _os.path.join(_ASSETS_DIR, "timeseries_data_proc_feat_ext_consts.md"),
    # Data processing transforms + augmenters reference
    "basic_transforms": _os.path.join(TINYML_DOCS_PATH, "source", "features", "feature_extraction.rst"),
    "augmenters_doc":   _os.path.join(TINYML_DOCS_PATH, "source", "features", "feature_extraction.rst"),
    # Reference example configs
    "examples_dir":     _os.path.join(TINYML_TENSORLAB_PATH, "tinyml-modelzoo", "examples"),
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