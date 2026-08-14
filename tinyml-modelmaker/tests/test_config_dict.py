"""Tests for ConfigDict."""

import os
import tempfile

import pytest
import yaml

from tinyml_modelmaker.utils.config_dict import ConfigDict


class TestConfigDict:
    def test_from_dict(self):
        d = {"a": 1, "b": {"c": 2}}
        cfg = ConfigDict(d)
        assert cfg.a == 1
        assert cfg.b.c == 2

    def test_from_yaml(self, tmp_path):
        data = {"x": 10, "nested": {"y": 20}}
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml.dump(data))
        cfg = ConfigDict(str(yaml_file))
        assert cfg.x == 10
        assert cfg.nested.y == 20

    def test_invalid_extension_raises(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="unrecognized file type"):
            ConfigDict(str(txt_file))

    def test_invalid_input_raises(self):
        with pytest.raises(TypeError, match="got invalid input"):
            ConfigDict(12345)

    def test_update(self):
        cfg = ConfigDict({"a": 1, "b": 2})
        cfg.update({"b": 3, "c": 4})
        assert cfg.b == 3
        assert cfg.c == 4

    def test_none_input(self):
        cfg = ConfigDict(None)
        # Should create an empty config without error
        assert isinstance(cfg, ConfigDict)

    def test_constructor_args_deep_merge_nested_dict(self):
        default = dict(training=dict(a=1, b=2))
        user = dict(training=dict(b=99))
        cfg = ConfigDict(default, user)
        assert dict(cfg.training) == {"a": 1, "b": 99}

    def test_constructor_args_path_string_is_loaded_and_merged(self, tmp_path):
        """A .yaml path passed as a positional arg *after* `input` (e.g.
        ConfigDict(default_params, user_yaml_path), the shape init_params
        uses) must actually be read and merged -- not silently discarded
        because it's a string rather than a dict/ConfigDict."""
        default = dict(training=dict(a=1, b=2))
        user_file = tmp_path / "user.yaml"
        user_file.write_text(yaml.dump(dict(training=dict(b=99))))
        cfg = ConfigDict(default, str(user_file))
        assert dict(cfg.training) == {"a": 1, "b": 99}

    def test_constructor_args_bad_extension_raises(self, tmp_path):
        default = dict(training=dict(a=1))
        bad_file = tmp_path / "user.txt"
        bad_file.write_text("a: 1")
        with pytest.raises(ValueError, match="unrecognized file type"):
            ConfigDict(default, str(bad_file))
