"""Tier 1 — Config YAML Validation Tests.

Validates that all example configs in tinyml-modelzoo/examples/ parse correctly,
contain required keys, and reference valid task types and models.
"""

import os
from pathlib import Path

import pytest
import yaml

from tinyml_modelmaker.ai_modules.timeseries import training, constants


# ---------------------------------------------------------------------------
# Discover all example YAML configs
# ---------------------------------------------------------------------------

def _find_example_configs():
    """Find all config*.yaml files in the modelzoo examples directory."""
    this_dir = Path(__file__).resolve().parent
    modelmaker_dir = this_dir.parent
    tensorlab_dir = modelmaker_dir.parent
    examples_dir = tensorlab_dir / "tinyml-modelzoo" / "examples"

    if not examples_dir.is_dir():
        return []

    configs = []
    for yaml_path in sorted(examples_dir.rglob("config*.yaml")):
        # Use relative path from examples_dir as the test ID
        rel = yaml_path.relative_to(examples_dir)
        configs.append(pytest.param(yaml_path, id=str(rel)))

    return configs


EXAMPLE_CONFIGS = _find_example_configs()

# Known valid task types (timeseries + vision)
KNOWN_TASK_TYPES = set(constants.TASK_TYPE_TO_CATEGORY.keys()) | {"image_classification"}

# Required top-level keys in every config
REQUIRED_SECTIONS = {"common", "training"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.component
class TestExampleConfigsParse:
    """Every example config YAML must parse without error."""

    @pytest.mark.skipif(not EXAMPLE_CONFIGS, reason="modelzoo examples not found")
    @pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS)
    def test_config_parses(self, config_path):
        """Config file should be valid YAML."""
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"Config did not parse as a dict: {config_path}"


@pytest.mark.component
class TestExampleConfigsStructure:
    """Config files must have required sections and valid references."""

    @pytest.mark.skipif(not EXAMPLE_CONFIGS, reason="modelzoo examples not found")
    @pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS)
    def test_has_required_sections(self, config_path):
        """Config must have 'common' and 'training' sections."""
        with open(config_path) as f:
            data = yaml.safe_load(f)

        for section in REQUIRED_SECTIONS:
            assert section in data, (
                f"Config {config_path.name} missing required section '{section}'"
            )

    @pytest.mark.skipif(not EXAMPLE_CONFIGS, reason="modelzoo examples not found")
    @pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS)
    def test_has_task_type(self, config_path):
        """Config must specify common.task_type."""
        with open(config_path) as f:
            data = yaml.safe_load(f)
        common = data.get("common", {})
        assert "task_type" in common, (
            f"Config {config_path.name} missing 'common.task_type'"
        )

    @pytest.mark.skipif(not EXAMPLE_CONFIGS, reason="modelzoo examples not found")
    @pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS)
    def test_task_type_is_valid(self, config_path):
        """common.task_type must be a recognized task type."""
        with open(config_path) as f:
            data = yaml.safe_load(f)
        task_type = data.get("common", {}).get("task_type")
        if task_type is not None:
            assert task_type in KNOWN_TASK_TYPES, (
                f"Config {config_path.name} has unknown task_type='{task_type}'"
            )

    @pytest.mark.skipif(not EXAMPLE_CONFIGS, reason="modelzoo examples not found")
    @pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS)
    def test_has_model_name(self, config_path):
        """Config must specify training.model_name."""
        with open(config_path) as f:
            data = yaml.safe_load(f)
        tr = data.get("training", {})
        assert "model_name" in tr, (
            f"Config {config_path.name} missing 'training.model_name'"
        )

    @pytest.mark.skipif(not EXAMPLE_CONFIGS, reason="modelzoo examples not found")
    @pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS)
    def test_has_target_device(self, config_path):
        """Config must specify common.target_device."""
        with open(config_path) as f:
            data = yaml.safe_load(f)
        common = data.get("common", {})
        assert "target_device" in common, (
            f"Config {config_path.name} missing 'common.target_device'"
        )


@pytest.mark.component
class TestExampleConfigsModelReferences:
    """Model names in configs should exist in the model registry."""

    @pytest.mark.skipif(not EXAMPLE_CONFIGS, reason="modelzoo examples not found")
    @pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS)
    def test_model_name_exists_in_registry(self, config_path):
        """training.model_name should be a registered model."""
        with open(config_path) as f:
            data = yaml.safe_load(f)

        model_name = data.get("training", {}).get("model_name")
        if model_name is None:
            pytest.skip("No model_name in config")

        # Skip NAS configs where model_name is a search placeholder
        if "NAS" in model_name.upper():
            pytest.skip(f"NAS model placeholder: {model_name}")

        task_type = data.get("common", {}).get("task_type")

        # Vision models have a separate registry
        if task_type == "image_classification":
            pytest.skip("Vision model registry not tested here")

        desc = training.get_model_description(model_name)
        assert desc is not None, (
            f"Config references model '{model_name}' which is not in the registry"
        )
