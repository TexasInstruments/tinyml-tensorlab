"""Tier 3 — Cross-Device Validation Tests.

Validates device-specific configuration is correct without running training.
Covers Tests 15-17, 19 from the test analysis:

  Test 15: NPU device config — hard NPU devices set type=hard in compilation
  Test 16: Non-NPU device config — soft NPU devices set type=soft
  Test 17: MSPM0 task support — matches tinyml-modelzoo model target_devices, not a hand-typed list
  Test 19: Compilation profile correctness — all devices have valid profiles

Marked with @pytest.mark.device — run with: pytest -m device
"""

import pytest

from tinyml_modelmaker.ai_modules.timeseries import constants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Devices with hardware NPU (type=hard in base compilation)
HARD_NPU_DEVICES = ["F28P55", "MSPM0G5187", "MSPM33C34", "AM13E2"]

# Devices with soft-only NPU
SOFT_NPU_DEVICES = [
    "F280013", "F280015", "F28003", "F28004", "F2837", "F28P65",
    "F29H85", "F29P58", "F29P32",
    "MSPM0G3507", "MSPM0G3519",
    "MSPM33C32",
    "AM263", "AM263P", "AM261",
    "CC2755", "CC1352", "CC1354", "CC35X1",
]

# MSPM0 devices under test — supports classification and, per model data, the other generic tasks too
MSPM0_CLASSIFICATION_ONLY = ["MSPM0G3507", "MSPM0G3519", "MSPM0G5187"]

# Non-classification task types
NON_CLASSIFICATION_TASKS = [
    constants.TASK_TYPE_GENERIC_TS_REGRESSION,
    constants.TASK_TYPE_GENERIC_TS_FORECASTING,
    constants.TASK_TYPE_GENERIC_TS_ANOMALYDETECTION,
]


# ---------------------------------------------------------------------------
# Test 15: NPU device config — hard NPU compilation flags
# ---------------------------------------------------------------------------

@pytest.mark.device
class TestHardNPUDeviceConfig:
    """Devices with hardware NPU should have type=hard in base compilation."""

    @pytest.mark.parametrize("device", HARD_NPU_DEVICES)
    def test_hard_npu_flag(self, device):
        """Hard NPU devices should have 'has_hard_npu': True."""
        profile = constants._DEVICE_PROFILES[device]
        assert profile["has_hard_npu"] is True, (
            f"Device {device} should have has_hard_npu=True"
        )

    @pytest.mark.parametrize("device", HARD_NPU_DEVICES)
    def test_hard_npu_base_compilation(self, device):
        """Hard NPU devices should have 'type=hard' in base compilation target."""
        profile = constants._DEVICE_PROFILES[device]
        base = profile["compilation_base"]
        assert "type=hard" in base["target"], (
            f"Device {device} base compilation should have type=hard, "
            f"got: {base['target']}"
        )

    @pytest.mark.parametrize("device", HARD_NPU_DEVICES)
    def test_hard_npu_has_soft_fallback(self, device):
        """Hard NPU devices should also have a soft NPU fallback compilation."""
        profile = constants._DEVICE_PROFILES[device]
        assert "compilation_soft" in profile, (
            f"Device {device} should have compilation_soft fallback"
        )
        soft = profile["compilation_soft"]
        assert "type=soft" in soft["target"], (
            f"Device {device} soft fallback should have type=soft"
        )


# ---------------------------------------------------------------------------
# Test 16: Non-NPU device config — soft NPU compilation flags
# ---------------------------------------------------------------------------

@pytest.mark.device
class TestSoftNPUDeviceConfig:
    """Devices without hardware NPU should use type=soft compilation."""

    @pytest.mark.parametrize("device", SOFT_NPU_DEVICES)
    def test_soft_npu_flag(self, device):
        """Soft NPU devices should have 'has_hard_npu': False."""
        profile = constants._DEVICE_PROFILES[device]
        assert profile["has_hard_npu"] is False, (
            f"Device {device} should have has_hard_npu=False"
        )

    @pytest.mark.parametrize("device", SOFT_NPU_DEVICES)
    def test_soft_npu_base_compilation(self, device):
        """Soft NPU devices should NOT have 'type=hard' in base compilation."""
        profile = constants._DEVICE_PROFILES[device]
        base = profile["compilation_base"]
        assert "type=hard" not in base["target"], (
            f"Device {device} base compilation should NOT have type=hard, "
            f"got: {base['target']}"
        )


# ---------------------------------------------------------------------------
# Test 17: MSPM0 classification-only
# ---------------------------------------------------------------------------

@pytest.mark.device
class TestMSPM0TaskSupport:
    """MSPM0 device/task support is derived from tinyml-modelzoo model target_devices
    (constants.TASK_DESCRIPTIONS), not a hand-typed per-device task_types list.
    MSPM0G3507/3519/5187 actually have models targeting all four generic timeseries
    tasks, not just classification — this used to be asserted otherwise via a stale
    _DEVICE_PROFILES['task_types'] field that had no real consumer and disagreed with
    the model data; that field has been removed.
    """

    @pytest.mark.parametrize("device", MSPM0_CLASSIFICATION_ONLY)
    def test_mspm0_supports_classification(self, device):
        """MSPM0 devices should support generic timeseries classification."""
        devices = constants.TASK_DESCRIPTIONS[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION]["target_devices"]
        assert device in devices, (
            f"MSPM0 device {device} should support generic_timeseries_classification"
        )

    @pytest.mark.parametrize("device", MSPM0_CLASSIFICATION_ONLY)
    @pytest.mark.parametrize("task_type", NON_CLASSIFICATION_TASKS)
    def test_mspm0_supports_non_classification_tasks(self, device, task_type):
        """MSPM0 devices also have models for regression/anomaly/forecasting."""
        devices = constants.TASK_DESCRIPTIONS[task_type]["target_devices"]
        assert device in devices, (
            f"MSPM0 device {device} should support {task_type} "
            f"(a model in tinyml-modelzoo targets it)"
        )

    @pytest.mark.parametrize("device", MSPM0_CLASSIFICATION_ONLY)
    def test_mspm0_no_regression_compilation_override(self, device):
        """MSPM0 devices use the base compilation config, no dedicated regression override."""
        profile = constants._DEVICE_PROFILES[device]
        assert "compilation_regression" not in profile, (
            f"MSPM0 device {device} should not have a compilation_regression override"
        )


# ---------------------------------------------------------------------------
# Test 19: Compilation profile correctness — all devices
# ---------------------------------------------------------------------------

@pytest.mark.device
class TestCompilationProfileCorrectness:
    """Validate all device profiles have correct structure."""

    def test_all_devices_have_profiles(self):
        """Every device in TARGET_DEVICES should have a profile."""
        for device in constants.TARGET_DEVICES:
            assert device in constants._DEVICE_PROFILES, (
                f"Device {device} missing from _DEVICE_PROFILES"
            )

    @pytest.mark.parametrize("device", constants.TARGET_DEVICES)
    def test_profile_has_base_compilation(self, device):
        """Every device profile should have compilation_base."""
        profile = constants._DEVICE_PROFILES[device]
        assert "compilation_base" in profile, (
            f"Device {device} missing compilation_base"
        )

    @pytest.mark.parametrize("device", constants.TARGET_DEVICES)
    def test_profile_has_npu_flag(self, device):
        """Every device profile should declare has_hard_npu."""
        profile = constants._DEVICE_PROFILES[device]
        assert "has_hard_npu" in profile, (
            f"Device {device} missing has_hard_npu flag"
        )

    @pytest.mark.parametrize("device", constants.TARGET_DEVICES)
    def test_compilation_base_has_target(self, device):
        """Base compilation config should have a 'target' key."""
        profile = constants._DEVICE_PROFILES[device]
        base = profile["compilation_base"]
        assert "target" in base, (
            f"Device {device} compilation_base missing 'target' key"
        )

    @pytest.mark.parametrize("device", constants.TARGET_DEVICES)
    def test_compilation_base_has_cross_compiler(self, device):
        """Base compilation config should have a 'cross_compiler' key."""
        profile = constants._DEVICE_PROFILES[device]
        base = profile["compilation_base"]
        assert "cross_compiler" in base, (
            f"Device {device} compilation_base missing 'cross_compiler' key"
        )

    @pytest.mark.parametrize("device", constants.TARGET_DEVICES)
    def test_cross_compiler_options_exist(self, device):
        """Every device should have cross-compiler options defined."""
        assert device in constants._CROSS_COMPILER_OPTIONS, (
            f"Device {device} missing from _CROSS_COMPILER_OPTIONS"
        )



# ---------------------------------------------------------------------------
# Additional: Task type ↔ category mapping consistency
# ---------------------------------------------------------------------------

@pytest.mark.device
class TestTaskCategoryMapping:
    """Validate task type to category mapping is complete and consistent."""

    def test_all_task_types_have_category(self):
        """Every task type should map to a category."""
        for task_type in constants.TASK_TYPES:
            category = constants.TASK_TYPE_TO_CATEGORY.get(task_type)
            assert category is not None, (
                f"Task type {task_type} has no category mapping"
            )

    def test_all_categories_used(self):
        """Every defined category should be referenced by at least one task type."""
        used_categories = set(constants.TASK_TYPE_TO_CATEGORY.values())
        for cat in constants.TASK_CATEGORIES:
            assert cat in used_categories, (
                f"Category {cat} defined but never used by any task type"
            )

    def test_data_dir_convention_classification(self):
        """Classification tasks should use 'classes' data dir."""
        data_dir = constants.get_default_data_dir_for_task(
            constants.TASK_CATEGORY_TS_CLASSIFICATION
        )
        assert data_dir == "classes"

    def test_data_dir_convention_regression(self):
        """Regression tasks should use 'files' data dir."""
        data_dir = constants.get_default_data_dir_for_task(
            constants.TASK_CATEGORY_TS_REGRESSION
        )
        assert data_dir == "files"

    def test_data_dir_convention_forecasting(self):
        """Forecasting tasks should use 'files' data dir."""
        data_dir = constants.get_default_data_dir_for_task(
            constants.TASK_CATEGORY_TS_FORECASTING
        )
        assert data_dir == "files"

    def test_data_dir_convention_anomaly(self):
        """Anomaly detection tasks should use 'classes' data dir."""
        data_dir = constants.get_default_data_dir_for_task(
            constants.TASK_CATEGORY_TS_ANOMALYDETECTION
        )
        assert data_dir == "classes"


# ---------------------------------------------------------------------------
# Quantization flag consistency
# ---------------------------------------------------------------------------

@pytest.mark.device
class TestQuantizationFlags:
    """Validate skip_normalize / output_int matrix is correct."""

    @pytest.mark.parametrize("task_category", constants.TASK_CATEGORIES)
    def test_float_mode_no_normalize(self, task_category):
        """Quantization=0 (float) should set skip_normalize=False, output_int=False."""
        skip, output = constants.get_skip_normalize_and_output_int(
            task_category, quantization=0, auto_quantization=False
        )
        assert skip is False
        assert output is False

    def test_classification_quant_sets_output_int(self):
        """Classification with quantization should set output_int=True."""
        skip, output = constants.get_skip_normalize_and_output_int(
            constants.TASK_CATEGORY_TS_CLASSIFICATION,
            quantization=1, auto_quantization=False,
        )
        assert skip is True
        assert output is True

    def test_regression_quant_no_output_int(self):
        """Regression with quantization should set output_int=False."""
        skip, output = constants.get_skip_normalize_and_output_int(
            constants.TASK_CATEGORY_TS_REGRESSION,
            quantization=1, auto_quantization=False,
        )
        assert skip is True
        assert output is False

    def test_forecasting_quant_no_output_int(self):
        """Forecasting with quantization should set output_int=False."""
        skip, output = constants.get_skip_normalize_and_output_int(
            constants.TASK_CATEGORY_TS_FORECASTING,
            quantization=1, auto_quantization=False,
        )
        assert skip is True
        assert output is False

    def test_auto_quant_regression_override(self):
        """Auto quantization for regression should set skip_normalize=False."""
        skip, output = constants.get_skip_normalize_and_output_int(
            constants.TASK_CATEGORY_TS_REGRESSION,
            quantization=1, auto_quantization=True,
        )
        assert skip is False
        assert output is False
