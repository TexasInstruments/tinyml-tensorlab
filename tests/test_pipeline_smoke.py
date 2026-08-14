"""Tier 2 — Pipeline Smoke Tests.

Runs 1-epoch training for each task type using synthetic datasets.
These tests validate the full pipeline: config parsing → data loading →
feature extraction → model creation → training → evaluation → ONNX export.

Marked with @pytest.mark.smoke — run with: pytest -m smoke

Note on data handling:
- Classification: DatasetHandling auto-creates train/val/test splits ✅
- Regression/Forecasting: DatasetHandling creates file_list.txt only —
  we run a two-phase approach: (1) dataset-only to populate file_list,
  (2) auto-create train/val splits, (3) run with training enabled.
- Anomaly Detection: No device preset exists — marked xfail.
"""

import copy
import csv
import os
import random
import shutil
import sys
from glob import glob
from pathlib import Path

import pytest

# Force QAT quantization ops to CPU on Apple Silicon MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(task_type, model_name, data_path, **overrides):
    """Build a minimal config dict for a 1-epoch smoke test.

    Quantization is disabled by default because QAT segfaults on Apple Silicon
    MPS (PyTorch fake_quantize ops not supported on MPS device).
    """
    config = {
        "common": {
            "task_type": task_type,
            "target_device": "F28P55",
        },
        "dataset": {
            "enable": True,
            "dataset_name": task_type,
            "input_data_path": data_path,
        },
        "data_processing_feature_extraction": {
            "feature_extraction_name": "default",
        },
        "training": {
            "enable": True,
            "model_name": model_name,
            "batch_size": 16,
            "training_epochs": 1,
            "num_gpus": 0,
            "quantization": 0,  # Disable QAT for MPS compatibility
        },
        "testing": {"enable": False},
        "compilation": {"enable": False, "compile_preset_name": "default_preset"},
    }
    # Apply overrides
    for section, values in overrides.items():
        if section in config:
            config[section].update(values)
        else:
            config[section] = values
    return config


def _auto_create_splits(project_base, task_type):
    """Create train/val/test split files from file_list.txt.

    DatasetHandling auto-creates these for classification but not for
    regression/forecasting. This mimics the same behavior.
    """
    ann_dir = Path(project_base) / "dataset" / "annotations"
    if not ann_dir.is_dir():
        return False

    # Already has splits? Skip.
    if list(ann_dir.glob("*train*_list.txt")):
        return True

    file_list = ann_dir / "file_list.txt"
    if not file_list.is_file():
        return False

    files = [line.strip() for line in file_list.read_text().splitlines() if line.strip()]
    if not files:
        return False

    # 70/15/15 split
    n = len(files)
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15))
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    (ann_dir / "instances_train_list.txt").write_text("\n".join(train_files))
    (ann_dir / "instances_val_list.txt").write_text("\n".join(val_files))
    if test_files:
        (ann_dir / "instances_test_list.txt").write_text("\n".join(test_files))

    return True


def _run_smoke_twophase(config):
    """Run main() in two phases for tasks that need annotation fixup.

    Phase 1: Run with training disabled to set up dataset structure
    Phase 2: Auto-create train/val splits, then run with training enabled
    """
    from tinyml_modelmaker.run_tinyml_modelmaker import main

    task_type = config["common"]["task_type"]
    # Determine where the project data will be stored
    project_base = os.path.join("data", "projects", task_type)

    # Phase 1: Dataset-only run
    dataset_config = copy.deepcopy(config)
    dataset_config["training"]["enable"] = False
    main(dataset_config)

    # Phase 2: Create annotation splits if needed
    _auto_create_splits(project_base, task_type)

    # Phase 3: Full run with training enabled
    # Disable dataset re-processing since data is already in place
    train_config = copy.deepcopy(config)
    train_config["dataset"]["enable"] = False
    result = main(train_config)
    return result


def _run_smoke(config):
    """Run modelmaker main() directly (for classification which auto-splits)."""
    from tinyml_modelmaker.run_tinyml_modelmaker import main
    return main(config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestClassificationSmoke:
    """1-epoch classification with synthetic 3-class data."""

    def test_classification_trains(self, synthetic_classification_data, tmp_path):
        config = _make_config(
            task_type="generic_timeseries_classification",
            model_name="CLS_1k_NPU",
            data_path=synthetic_classification_data,
            data_processing_feature_extraction={
                "feature_extraction_name": "Generic_256Input_RAW_256Feature_1Frame",
                "variables": 1,
            },
        )

        result = _run_smoke(config)
        assert result is True, "Classification smoke test failed"


@pytest.mark.smoke
class TestRegressionSmoke:
    """1-epoch regression with synthetic continuous data."""

    def test_regression_trains(self, synthetic_regression_data, tmp_path):
        config = _make_config(
            task_type="generic_timeseries_regression",
            model_name="REGR_1k",
            data_path=synthetic_regression_data,
            data_processing_feature_extraction={
                "feature_extraction_name": "Generic_256Input_RAW_256Feature_1Frame",
                "variables": 2,
            },
        )

        result = _run_smoke_twophase(config)
        assert result is True, "Regression smoke test failed"


@pytest.mark.smoke
@pytest.mark.xfail(
    reason="generic_timeseries_anomalydetection has no device preset in compilation config"
)
class TestAnomalyDetectionSmoke:
    """1-epoch anomaly detection with synthetic normal/anomaly data."""

    def test_anomalydetection_trains(self, synthetic_anomaly_data, tmp_path):
        config = _make_config(
            task_type="generic_timeseries_anomalydetection",
            model_name="AD_1k",
            data_path=synthetic_anomaly_data,
            data_processing_feature_extraction={
                "feature_extraction_name": "Generic_256Input_RAW_256Feature_1Frame",
                "variables": 1,
            },
        )

        result = _run_smoke(config)
        assert result is True, "Anomaly detection smoke test failed"


@pytest.mark.smoke
class TestForecastingSmoke:
    """1-epoch forecasting with synthetic time series."""

    def test_forecasting_trains(self, synthetic_forecasting_data, tmp_path):
        config = _make_config(
            task_type="generic_timeseries_forecasting",
            model_name="FCST_LSTM8",
            data_path=synthetic_forecasting_data,
            data_processing_feature_extraction={
                "data_proc_transforms": ["SimpleWindow"],
                "frame_size": 32,
                "stride_size": 0.5,
                "forecast_horizon": 2,
                "variables": 1,
                "target_variables": [0],
            },
        )

        result = _run_smoke_twophase(config)
        assert result is True, "Forecasting smoke test failed"
