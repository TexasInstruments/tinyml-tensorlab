"""Tier 1 — Model Registry Component Tests.

Validates that model descriptions are complete, properly structured,
and correctly filtered by task type and device.
"""

import pytest

from tinyml_modelmaker.ai_modules.timeseries import training, constants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GENERIC_TASK_TYPES = [
    constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION,
    constants.TASK_TYPE_GENERIC_TS_REGRESSION,
    constants.TASK_TYPE_GENERIC_TS_ANOMALYDETECTION,
    constants.TASK_TYPE_GENERIC_TS_FORECASTING,
]

ALL_TASK_TYPES = list(constants.TASK_TYPE_TO_CATEGORY.keys())

REQUIRED_DESCRIPTION_KEYS = {"common", "training"}
REQUIRED_TRAINING_KEYS = {"training_backend", "model_training_id"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.component
class TestModelDescriptions:
    """Every registered model must have a well-formed description dict."""

    def test_registry_is_not_empty(self):
        """Global model registry should contain models."""
        all_models = training.get_model_descriptions()
        assert len(all_models) > 0, "Model registry is empty"

    @pytest.mark.parametrize("task_type", GENERIC_TASK_TYPES)
    def test_models_exist_per_task_type(self, task_type):
        """Each generic task type should have at least one model."""
        models = training.get_model_descriptions(task_type=task_type)
        assert len(models) > 0, f"No models registered for task_type={task_type}"

    def test_every_model_has_required_keys(self):
        """All model descriptions must contain 'common' and 'training' sections."""
        all_models = training.get_model_descriptions()
        for name, desc in all_models.items():
            for key in REQUIRED_DESCRIPTION_KEYS:
                assert key in desc, f"Model '{name}' missing required key '{key}'"

    def test_every_model_has_training_backend(self):
        """All models must specify a training_backend."""
        all_models = training.get_model_descriptions()
        for name, desc in all_models.items():
            tr = desc.get("training", {})
            assert "training_backend" in tr, (
                f"Model '{name}' missing 'training.training_backend'"
            )

    def test_every_model_has_model_training_id(self):
        """All models must specify a model_training_id."""
        all_models = training.get_model_descriptions()
        for name, desc in all_models.items():
            tr = desc.get("training", {})
            assert "model_training_id" in tr, (
                f"Model '{name}' missing 'training.model_training_id'"
            )

    def test_every_model_has_task_type(self):
        """All models must specify common.task_type."""
        all_models = training.get_model_descriptions()
        for name, desc in all_models.items():
            common = desc.get("common", {})
            assert "task_type" in common, (
                f"Model '{name}' missing 'common.task_type'"
            )

    def test_every_model_task_type_is_known(self):
        """common.task_type must be a recognized task type string."""
        all_models = training.get_model_descriptions()
        for name, desc in all_models.items():
            task_type = desc.get("common", {}).get("task_type")
            assert task_type in ALL_TASK_TYPES, (
                f"Model '{name}' has unknown task_type='{task_type}'"
            )


@pytest.mark.component
class TestModelFiltering:
    """get_model_descriptions() filtering by task_type and target_device."""

    def test_classification_filter_returns_only_classification(self):
        """Filtering by classification should only return classification models."""
        task = constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION
        models = training.get_model_descriptions(task_type=task)
        for name, desc in models.items():
            assert desc["common"]["task_type"] == task, (
                f"Model '{name}' leaked through filter: "
                f"expected task_type={task}, got {desc['common']['task_type']}"
            )

    def test_regression_filter_returns_only_regression(self):
        task = constants.TASK_TYPE_GENERIC_TS_REGRESSION
        models = training.get_model_descriptions(task_type=task)
        for name, desc in models.items():
            assert desc["common"]["task_type"] == task

    def test_anomalydetection_filter_returns_only_anomalydetection(self):
        task = constants.TASK_TYPE_GENERIC_TS_ANOMALYDETECTION
        models = training.get_model_descriptions(task_type=task)
        for name, desc in models.items():
            assert desc["common"]["task_type"] == task

    def test_forecasting_filter_returns_only_forecasting(self):
        task = constants.TASK_TYPE_GENERIC_TS_FORECASTING
        models = training.get_model_descriptions(task_type=task)
        for name, desc in models.items():
            assert desc["common"]["task_type"] == task

    def test_device_filter_f28p55(self):
        """Filtering by F28P55 should return only models that support it."""
        device = "F28P55"
        models = training.get_model_descriptions(target_device=device)
        assert len(models) > 0, f"No models found for device {device}"
        for name, desc in models.items():
            devices = desc.get("training", {}).get("target_devices", [])
            assert device in devices, (
                f"Model '{name}' does not list {device} in target_devices"
            )

    def test_device_filter_mspm0g3507(self):
        """MSPM0G3507 should return models (primarily classification)."""
        device = "MSPM0G3507"
        models = training.get_model_descriptions(target_device=device)
        assert len(models) > 0, f"No models found for device {device}"

    def test_combined_task_and_device_filter(self):
        """Combined filter should be the intersection."""
        task = constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION
        device = "F28P55"
        models = training.get_model_descriptions(task_type=task, target_device=device)
        for name, desc in models.items():
            assert desc["common"]["task_type"] == task
            assert device in desc["training"]["target_devices"]


@pytest.mark.component
class TestModelLookup:
    """get_model_description() single-model lookup."""

    def test_known_model_returns_dict(self):
        """Looking up a known model name should return its description."""
        # Find any model name from the registry
        all_models = training.get_model_descriptions()
        name = next(iter(all_models))
        desc = training.get_model_description(name)
        assert desc is not None
        assert isinstance(desc, dict)

    def test_unknown_model_returns_none(self):
        """Looking up a nonexistent model should return None."""
        desc = training.get_model_description("TOTALLY_FAKE_MODEL_XYZ")
        assert desc is None

    def test_lookup_result_matches_registry(self):
        """Lookup result should match the registry entry."""
        all_models = training.get_model_descriptions()
        for name in list(all_models.keys())[:5]:  # spot-check first 5
            desc = training.get_model_description(name)
            assert desc is not None
            assert desc["training"]["model_training_id"] == all_models[name]["training"]["model_training_id"]


@pytest.mark.component
class TestDeviceConstants:
    """Verify device constant lists are populated and consistent."""

    def test_target_devices_not_empty(self):
        assert len(constants.TARGET_DEVICES) > 0

    def test_all_task_types_mapped(self):
        """Every TASK_TYPE constant should be in TASK_TYPE_TO_CATEGORY."""
        for tt in ALL_TASK_TYPES:
            assert tt in constants.TASK_TYPE_TO_CATEGORY, (
                f"Task type '{tt}' not in TASK_TYPE_TO_CATEGORY mapping"
            )

    def test_target_devices_are_strings(self):
        for device in constants.TARGET_DEVICES:
            assert isinstance(device, str), f"Device {device!r} is not a string"
