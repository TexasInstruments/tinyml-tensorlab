"""Tests for misc_utils: resolve_paths, resolve_run_name, symlink helpers."""

import os
import re
import tempfile

import pytest

from tinyml_modelmaker.utils import misc_utils
from tinyml_modelmaker.utils.config_dict import ConfigDict


class TestResolvePaths:
    """Tests for the extracted resolve_paths() function."""

    @staticmethod
    def _make_params(**overrides):
        """Build a minimal params ConfigDict for testing."""
        base = dict(
            common=dict(
                projects_path="./projects",
                project_path=None,
                project_run_path=None,
                run_name="{date-time}/{model_name}",
                target_device="F28P55",
            ),
            dataset=dict(
                input_data_path="./data/input",
                input_annotation_path=None,
                dataset_name="",
                dataset_path=None,
                extract_path=None,
            ),
            training=dict(
                enable=True,
                model_name="TestModel",
                train_output_path=None,
                training_path=None,
                training_path_quantization=None,
                model_packaged_path=None,
                quantization=0,  # NO_QUANTIZATION
            ),
            compilation=dict(
                enable=False,
                compile_output_path=None,
                compilation_path=None,
                model_packaged_path=None,
            ),
        )
        for section, vals in overrides.items():
            base[section].update(vals)
        return ConfigDict(base)

    def test_default_paths(self):
        params = self._make_params()
        target_devices = ["F28P55", "F280015"]
        misc_utils.resolve_paths(params, target_devices)

        assert os.path.isabs(params.common.projects_path)
        assert params.dataset.dataset_name == "input"  # basename of input_data_path
        assert params.dataset.dataset_path.endswith(os.path.join("input", "dataset"))
        assert "run" in params.common.project_run_path
        assert "training" in params.training.training_path

    def test_train_output_path(self, tmp_path):
        params = self._make_params(
            training=dict(train_output_path=str(tmp_path / "output")),
        )
        target_devices = ["F28P55"]
        misc_utils.resolve_paths(params, target_devices)

        assert params.dataset.dataset_path == os.path.join(str(tmp_path / "output"), "dataset")
        assert params.training.training_path.endswith("training_base")
        assert params.common.project_run_path == str(tmp_path / "output")

    def test_invalid_target_device(self):
        params = self._make_params(common=dict(target_device="INVALID"))
        with pytest.raises(ValueError, match="must be set to one of"):
            misc_utils.resolve_paths(params, ["F28P55"])

    def test_dataset_name_fallback(self):
        params = self._make_params(
            dataset=dict(input_data_path="/some/path/my_dataset.zip", dataset_name=""),
        )
        misc_utils.resolve_paths(params, ["F28P55"])
        assert params.dataset.dataset_name == "my_dataset"

    def test_compile_output_path(self, tmp_path):
        params = self._make_params(
            training=dict(enable=False),
            compilation=dict(enable=True, compile_output_path=str(tmp_path / "compile_out")),
        )
        misc_utils.resolve_paths(params, ["F28P55"])
        assert params.compilation.compilation_path == str(tmp_path / "compile_out")
        assert params.compilation.model_packaged_path.endswith("_F28P55.zip")


class TestResolveRunName:
    def test_date_time_placeholder(self):
        result = misc_utils.resolve_run_name("{date-time}/model", "MyModel")
        # Should have replaced {date-time} with a timestamp
        assert "{date-time}" not in result
        assert re.match(r"\d{8}-\d{6}/model", result)

    def test_model_name_placeholder(self):
        result = misc_utils.resolve_run_name("run/{model_name}", "MyModel")
        assert result == "run/MyModel"

    def test_empty_run_name(self):
        assert misc_utils.resolve_run_name("", "Model") == ""
        assert misc_utils.resolve_run_name(None, "Model") == ""

    def test_both_placeholders(self):
        result = misc_utils.resolve_run_name("{date-time}/{model_name}", "TestNet")
        assert "TestNet" in result
        assert "{date-time}" not in result


class TestSimplifyDict:
    def test_valid_dict(self):
        d = {"a": 1, "b": {"c": [1, 2, 3]}}
        result = misc_utils.simplify_dict(d)
        assert result == {"a": 1, "b": {"c": [1, 2, 3]}}

    def test_invalid_input(self):
        with pytest.raises(TypeError, match="must be of type dict"):
            misc_utils.simplify_dict("not a dict")
