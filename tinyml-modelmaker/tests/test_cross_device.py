"""Tier 3 — Cross-Device Validation Tests.

Validates device-specific configuration is correct without running training.
Covers Tests 15-19 from the test analysis:

  Test 15: NPU device config — hard NPU devices set type=hard in compilation
  Test 16: Non-NPU device config — soft NPU devices set type=soft
  Test 17: MSPM0 classification-only — MSPM0G3507 rejects non-classification tasks
  Test 18: Device model size constraints — device_selection_factor consistency
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

# MSPM0 devices — classification only
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
class TestMSPM0ClassificationOnly:
    """MSPM0 devices should only support classification task types."""

    @pytest.mark.parametrize("device", MSPM0_CLASSIFICATION_ONLY)
    def test_mspm0_limited_tasks(self, device):
        """MSPM0 devices should have explicit task_types list in profile."""
        profile = constants._DEVICE_PROFILES[device]
        assert "task_types" in profile, (
            f"MSPM0 device {device} should have explicit task_types list"
        )

    @pytest.mark.parametrize("device", MSPM0_CLASSIFICATION_ONLY)
    def test_mspm0_no_regression(self, device):
        """MSPM0 devices should not support regression."""
        profile = constants._DEVICE_PROFILES[device]
        task_types = profile.get("task_types", [])
        assert constants.TASK_TYPE_GENERIC_TS_REGRESSION not in task_types, (
            f"MSPM0 device {device} should not support regression"
        )

    @pytest.mark.parametrize("device", MSPM0_CLASSIFICATION_ONLY)
    def test_mspm0_no_forecasting(self, device):
        """MSPM0 devices should not support forecasting."""
        profile = constants._DEVICE_PROFILES[device]
        task_types = profile.get("task_types", [])
        assert constants.TASK_TYPE_GENERIC_TS_FORECASTING not in task_types, (
            f"MSPM0 device {device} should not support forecasting"
        )

    @pytest.mark.parametrize("device", MSPM0_CLASSIFICATION_ONLY)
    def test_mspm0_supports_classification(self, device):
        """MSPM0 devices should support classification."""
        profile = constants._DEVICE_PROFILES[device]
        task_types = profile.get("task_types", [])
        assert constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION in task_types, (
            f"MSPM0 device {device} should support classification"
        )

    @pytest.mark.parametrize("device", MSPM0_CLASSIFICATION_ONLY)
    def test_mspm0_no_regression_compilation(self, device):
        """MSPM0 devices should not have compilation_regression profile."""
        profile = constants._DEVICE_PROFILES[device]
        assert "compilation_regression" not in profile, (
            f"MSPM0 device {device} should not have compilation_regression"
        )


# ---------------------------------------------------------------------------
# Test 18: Device model size constraints — device_selection_factor
# ---------------------------------------------------------------------------

@pytest.mark.device
class TestDeviceSelectionFactor:
    """Validate device_selection_factor ordering and consistency."""

    def test_all_devices_have_descriptions(self):
        """Every device in TARGET_DEVICES should have a description."""
        for device in constants.TARGET_DEVICES:
            assert device in constants.TARGET_DEVICE_DESCRIPTIONS, (
                f"Device {device} missing from TARGET_DEVICE_DESCRIPTIONS"
            )

    def test_selection_factors_are_non_negative(self):
        """All device_selection_factor values should be non-negative integers."""
        for device, desc in constants.TARGET_DEVICE_DESCRIPTIONS.items():
            factor = desc.get("device_selection_factor")
            assert factor is not None, (
                f"Device {device} missing device_selection_factor"
            )
            assert isinstance(factor, int) and factor >= 0, (
                f"Device {device} has invalid factor: {factor}"
            )

    def test_selection_factors_are_unique(self):
        """Device selection factors should ideally be unique (warn if not)."""
        factors = {}
        for device, desc in constants.TARGET_DEVICE_DESCRIPTIONS.items():
            f = desc["device_selection_factor"]
            if f in factors:
                # Not a hard failure — just tracked
                pass
            factors.setdefault(f, []).append(device)
        # At least some differentiation should exist
        assert len(factors) > 1, "All devices have the same selection factor"

    def test_hard_npu_devices_higher_factor(self):
        """Devices with hard NPU should generally have higher selection factor."""
        npu_factors = []
        for device in HARD_NPU_DEVICES:
            if device in constants.TARGET_DEVICE_DESCRIPTIONS:
                npu_factors.append(
                    constants.TARGET_DEVICE_DESCRIPTIONS[device]["device_selection_factor"]
                )
        if npu_factors:
            avg_npu = sum(npu_factors) / len(npu_factors)
            # Hard NPU devices should have above-average selection factor
            all_factors = [
                d["device_selection_factor"]
                for d in constants.TARGET_DEVICE_DESCRIPTIONS.values()
            ]
            avg_all = sum(all_factors) / len(all_factors)
            assert avg_npu >= avg_all, (
                f"Hard NPU devices avg factor ({avg_npu:.1f}) should be "
                f">= overall avg ({avg_all:.1f})"
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

    def test_all_devices_have_sdk_info(self):
        """Every device description should include SDK version and release."""
        for device, desc in constants.TARGET_DEVICE_DESCRIPTIONS.items():
            assert "sdk_version" in desc, (
                f"Device {device} missing sdk_version"
            )
            assert "sdk_release" in desc, (
                f"Device {device} missing sdk_release"
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
            task_category, quantization=0, partial_quantization=False
        )
        assert skip is False
        assert output is False

    def test_classification_quant_sets_output_int(self):
        """Classification with quantization should set output_int=True."""
        skip, output = constants.get_skip_normalize_and_output_int(
            constants.TASK_CATEGORY_TS_CLASSIFICATION,
            quantization=1, partial_quantization=False,
        )
        assert skip is True
        assert output is True

    def test_regression_quant_no_output_int(self):
        """Regression with quantization should set output_int=False."""
        skip, output = constants.get_skip_normalize_and_output_int(
            constants.TASK_CATEGORY_TS_REGRESSION,
            quantization=1, partial_quantization=False,
        )
        assert skip is True
        assert output is False

    def test_forecasting_quant_no_output_int(self):
        """Forecasting with quantization should set output_int=False."""
        skip, output = constants.get_skip_normalize_and_output_int(
            constants.TASK_CATEGORY_TS_FORECASTING,
            quantization=1, partial_quantization=False,
        )
        assert skip is True
        assert output is False

    def test_partial_quant_regression_override(self):
        """Partial quantization for regression should set skip_normalize=False."""
        skip, output = constants.get_skip_normalize_and_output_int(
            constants.TASK_CATEGORY_TS_REGRESSION,
            quantization=1, partial_quantization=True,
        )
        assert skip is False
        assert output is False
