"""Shared fixtures for tinyml-modelmaker tests.

Provides:
- pytest markers for component and smoke test tiers
- Synthetic dataset generators for pipeline smoke tests
"""

import os
import csv
import random
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers for test tiers."""
    config.addinivalue_line("markers", "component: Tier 1 component tests (fast, no training)")
    config.addinivalue_line("markers", "smoke: Tier 2 pipeline smoke tests (1-epoch, synthetic data)")


# ---------------------------------------------------------------------------
# Synthetic dataset generators
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_classification_data(tmp_path):
    """Create a minimal 3-class classification dataset.

    Structure:
        tmp_path/
            classes/
                class_a/  (30 CSV files)
                class_b/  (30 CSV files)
                class_c/  (30 CSV files)
            annotations/
                file_list.txt  (all files)
    Each CSV has 256 rows (samples) × 1 column (single-channel sensor data).
    """
    random.seed(42)
    classes_dir = tmp_path / "classes"
    all_files = []

    for label in ["class_a", "class_b", "class_c"]:
        class_dir = classes_dir / label
        class_dir.mkdir(parents=True)
        for i in range(30):
            filename = f"{label}_{i:03d}.csv"
            filepath = class_dir / filename
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                for _ in range(256):
                    writer.writerow([random.gauss(0, 1)])
            all_files.append(f"classes/{label}/{filename}")

    # Create annotations directory with file list
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(parents=True)
    (ann_dir / "file_list.txt").write_text("\n".join(all_files))

    return str(tmp_path)


@pytest.fixture
def synthetic_regression_data(tmp_path):
    """Create a minimal regression dataset.

    Structure:
        tmp_path/
            files/  (60 CSV files — flat layout for regression)
            annotations/
                instances_train_list.txt
                instances_val_list.txt
    Each CSV has 256 rows × 2 columns (input feature, target value).
    """
    random.seed(42)
    files_dir = tmp_path / "files"
    files_dir.mkdir(parents=True)
    all_files = []

    for i in range(60):
        filename = f"sample_{i:03d}.csv"
        filepath = files_dir / filename
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            for _ in range(256):
                x = random.gauss(0, 1)
                y = 2.0 * x + random.gauss(0, 0.1)
                writer.writerow([x, y])
        all_files.append(f"files/{filename}")

    # Create annotations with train/val split (48 train, 12 val)
    # File names must match GenericTSDataset glob patterns: *train*_list.txt, *val*_list.txt
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(parents=True)
    (ann_dir / "instances_train_list.txt").write_text("\n".join(all_files[:48]))
    (ann_dir / "instances_val_list.txt").write_text("\n".join(all_files[48:]))

    return str(tmp_path)


@pytest.fixture
def synthetic_anomaly_data(tmp_path):
    """Create a minimal anomaly detection dataset.

    Structure:
        tmp_path/
            classes/
                normal/  (50 CSV files)
                anomaly/ (10 CSV files)
    """
    random.seed(42)
    classes_dir = tmp_path / "classes"

    for label, count, mu in [("normal", 50, 0.0), ("anomaly", 10, 5.0)]:
        class_dir = classes_dir / label
        class_dir.mkdir(parents=True)
        for i in range(count):
            filepath = class_dir / f"{label}_{i:03d}.csv"
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                for _ in range(256):
                    writer.writerow([random.gauss(mu, 1)])

    return str(tmp_path)


@pytest.fixture
def synthetic_forecasting_data(tmp_path):
    """Create a minimal forecasting dataset.

    Structure:
        tmp_path/
            files/  (60 CSV files — flat layout for forecasting)
            annotations/
                instances_train_list.txt
                instances_val_list.txt
    Each CSV has 64 rows × 1 column (univariate time series).
    """
    random.seed(42)
    files_dir = tmp_path / "files"
    files_dir.mkdir(parents=True)
    all_files = []

    for i in range(60):
        filename = f"series_{i:03d}.csv"
        filepath = files_dir / filename
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            val = random.gauss(0, 1)
            for _ in range(64):
                val += random.gauss(0, 0.3)
                writer.writerow([val])
        all_files.append(f"files/{filename}")

    # Create annotations with train/val split (48 train, 12 val)
    # File names must match GenericTSDataset glob patterns: *train*_list.txt, *val*_list.txt
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(parents=True)
    (ann_dir / "instances_train_list.txt").write_text("\n".join(all_files[:48]))
    (ann_dir / "instances_val_list.txt").write_text("\n".join(all_files[48:]))

    return str(tmp_path)


@pytest.fixture
def modelzoo_examples_dir():
    """Return the path to the modelzoo examples directory.

    Returns None if not found (not all environments will have modelzoo).
    """
    this_dir = Path(__file__).resolve().parent
    modelmaker_dir = this_dir.parent
    tensorlab_dir = modelmaker_dir.parent
    examples_dir = tensorlab_dir / "tinyml-modelzoo" / "examples"
    if examples_dir.is_dir():
        return examples_dir
    return None
