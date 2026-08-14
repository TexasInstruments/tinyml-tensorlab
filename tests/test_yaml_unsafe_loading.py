"""Regression test for unsafe yaml.CLoader usage.

yaml.CLoader is the C-accelerated equivalent of the unsafe yaml.Loader (not
yaml.SafeLoader) -- it can instantiate arbitrary Python objects (and execute
code) via !!python/object/apply:... tags. mdcl_utils.read_yaml_file(),
image_dataset.py, and timeseries_dataset.py all used it to parse config files.
This test covers the standalone, directly-testable call site
(mdcl_utils.read_yaml_file); the dataset-class call sites use the identical
one-line substitution (yaml.load(Loader=yaml.CLoader) -> yaml.safe_load) and
were verified by code review rather than duplicating this test against the
much heavier Dataset class fixtures.
"""
import os
import tempfile

import pytest
import yaml

from tinyml_tinyverse.common.utils.mdcl_utils import read_yaml_file


def test_read_yaml_file_parses_normal_yaml():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = os.path.join(tmp_dir, "config.yaml")
        with open(yaml_path, "w") as fp:
            fp.write("key: value\nnested:\n  a: 1\n  b: 2\n")

        data = read_yaml_file(yaml_path)

    assert data == {"key": "value", "nested": {"a": 1, "b": 2}}


def test_read_yaml_file_rejects_unsafe_python_object_tags(capsys):
    """A malicious config using !!python/object/apply: would execute arbitrary
    code under the unsafe Loader; safe_load must refuse to construct it."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = os.path.join(tmp_dir, "malicious.yaml")
        with open(yaml_path, "w") as fp:
            fp.write("payload: !!python/object/apply:builtins.print [UNSAFE DESERIALIZATION EXECUTED]\n")

        with pytest.raises(yaml.constructor.ConstructorError):
            read_yaml_file(yaml_path)

    assert "UNSAFE DESERIALIZATION EXECUTED" not in capsys.readouterr().out
