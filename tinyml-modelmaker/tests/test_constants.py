"""Tests for timeseries and vision constants modules."""

import pytest

from tinyml_modelmaker.ai_modules.timeseries import constants as ts_constants
from tinyml_modelmaker.ai_modules.vision import constants as vis_constants


class TestTimeseriesConstants:
    """Verify structural invariants of the timeseries constants module."""

    def test_task_types_list_not_empty(self):
        assert len(ts_constants.TASK_TYPES) > 0

    def test_all_task_types_mapped_to_category(self):
        for task_type in ts_constants.TASK_TYPES:
            assert task_type in ts_constants.TASK_TYPE_TO_CATEGORY, (
                f"{task_type} missing from TASK_TYPE_TO_CATEGORY"
            )

    def test_task_categories_list_not_empty(self):
        assert len(ts_constants.TASK_CATEGORIES) > 0

    def test_target_devices_all_includes_base_and_additional(self):
        for dev in ts_constants.TARGET_DEVICES:
            assert dev in ts_constants.TARGET_DEVICES_ALL
        for dev in ts_constants.TARGET_DEVICES_ADDITIONAL:
            assert dev in ts_constants.TARGET_DEVICES_ALL

    def test_training_device_constants(self):
        assert ts_constants.TRAINING_DEVICE_CPU == "cpu"
        assert ts_constants.TRAINING_DEVICE_CUDA == "cuda"
        assert ts_constants.TRAINING_DEVICE_MPS == "mps"
        assert ts_constants.TRAINING_DEVICE_GPU == ts_constants.TRAINING_DEVICE_CUDA

    def test_training_backend_constant(self):
        assert ts_constants.TRAINING_BACKEND_TINYML_TINYVERSE == "tinyml_tinyverse"

    def test_data_dir_constants(self):
        assert ts_constants.DATA_DIR_CLASSES == "classes"
        assert ts_constants.DATA_DIR_FILES == "files"


class TestGetTaskCategory:
    """Tests for get_task_category()."""

    def test_known_task_type(self):
        cat = ts_constants.get_task_category(ts_constants.TASK_TYPE_MOTOR_FAULT)
        assert cat == ts_constants.TASK_CATEGORY_TS_CLASSIFICATION

    def test_task_category_passed_through(self):
        cat = ts_constants.get_task_category(ts_constants.TASK_CATEGORY_TS_REGRESSION)
        assert cat == ts_constants.TASK_CATEGORY_TS_REGRESSION

    def test_unknown_task_type_defaults(self):
        cat = ts_constants.get_task_category("unknown_task")
        assert cat == ts_constants.TASK_CATEGORY_TS_CLASSIFICATION


class TestGetDefaultDataDir:
    """Tests for get_default_data_dir_for_task()."""

    def test_classification_returns_classes(self):
        assert (
            ts_constants.get_default_data_dir_for_task(
                ts_constants.TASK_CATEGORY_TS_CLASSIFICATION
            )
            == ts_constants.DATA_DIR_CLASSES
        )

    def test_regression_returns_files(self):
        assert (
            ts_constants.get_default_data_dir_for_task(
                ts_constants.TASK_CATEGORY_TS_REGRESSION
            )
            == ts_constants.DATA_DIR_FILES
        )

    def test_forecasting_returns_files(self):
        assert (
            ts_constants.get_default_data_dir_for_task(
                ts_constants.TASK_CATEGORY_TS_FORECASTING
            )
            == ts_constants.DATA_DIR_FILES
        )

    def test_anomaly_returns_classes(self):
        assert (
            ts_constants.get_default_data_dir_for_task(
                ts_constants.TASK_CATEGORY_TS_ANOMALYDETECTION
            )
            == ts_constants.DATA_DIR_CLASSES
        )

    def test_unknown_category_returns_classes(self):
        assert ts_constants.get_default_data_dir_for_task("something_else") == ts_constants.DATA_DIR_CLASSES

    def test_vision_returns_classes(self):
        # Vision module always returns 'classes'
        assert vis_constants.get_default_data_dir_for_task("image_classification") == "classes"


class TestGetSkipNormalizeAndOutputInt:
    """Tests for get_skip_normalize_and_output_int()."""

    def test_no_quantization(self):
        skip, out_int = ts_constants.get_skip_normalize_and_output_int(
            ts_constants.TASK_CATEGORY_TS_CLASSIFICATION, 0, False
        )
        assert skip is False
        assert out_int is False

    def test_quantized_classification(self):
        skip, out_int = ts_constants.get_skip_normalize_and_output_int(
            ts_constants.TASK_CATEGORY_TS_CLASSIFICATION, 1, False
        )
        assert skip is True
        assert out_int is True

    def test_quantized_regression(self):
        skip, out_int = ts_constants.get_skip_normalize_and_output_int(
            ts_constants.TASK_CATEGORY_TS_REGRESSION, 1, False
        )
        assert skip is True
        assert out_int is False

    def test_quantized_regression_partial(self):
        skip, out_int = ts_constants.get_skip_normalize_and_output_int(
            ts_constants.TASK_CATEGORY_TS_REGRESSION, 1, True
        )
        assert skip is False  # partial quantization disables skip_normalize for regression
        assert out_int is False

    def test_quantized_forecasting(self):
        skip, out_int = ts_constants.get_skip_normalize_and_output_int(
            ts_constants.TASK_CATEGORY_TS_FORECASTING, 2, False
        )
        assert skip is True
        assert out_int is False


class TestVisionConstants:
    """Verify structural invariants of the vision constants module."""

    def test_task_types_not_empty(self):
        assert len(vis_constants.TASK_TYPES) > 0

    def test_target_devices_all(self):
        for dev in vis_constants.TARGET_DEVICES:
            assert dev in vis_constants.TARGET_DEVICES_ALL

    def test_training_backend_constant(self):
        assert vis_constants.TRAINING_BACKEND_TINYML_TINYVERSE == "tinyml_tinyverse"
