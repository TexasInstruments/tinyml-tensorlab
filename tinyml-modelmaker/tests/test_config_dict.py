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
