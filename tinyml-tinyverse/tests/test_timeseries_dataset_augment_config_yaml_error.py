"""Regression test for a crash the yaml.safe_load() fix (PR #26) exposed in
BaseGenericTSDataset.__init__ (timeseries_dataset.py).

The try/except around parsing augment_config caught yaml.YAMLError and logged
it, but never assigned augment_config in the except branch, then
unconditionally fell through to `augment_config.items()` right after the
try/except -- raising UnboundLocalError for any YAML that fails to parse
(e.g. a malicious !!python/object/apply: tag, which safe_load correctly
refuses via ConstructorError -- a YAMLError subclass). An empty YAML file hit
the same crash a different way: safe_load() returns None on empty input, so
`None.items()` raised AttributeError. Both are now handled: the except
branch sets augment_config = {} on a parse error, and `or {}` covers the
None-on-empty-file case.
"""
import os
import tempfile

from tinyml_tinyverse.common.datasets.timeseries_dataset import BaseGenericTSDataset


def _make_dataset(yaml_path):
    return BaseGenericTSDataset(subset=None, dataset_dir="/unused", augment_config=yaml_path, transforms=[])


def test_augment_config_yaml_error_does_not_crash():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = os.path.join(tmp_dir, "malicious.yaml")
        with open(yaml_path, "w") as fp:
            fp.write("payload: !!python/object/apply:builtins.print [UNSAFE]\n")

        dataset = _make_dataset(yaml_path)

    assert dataset.augment_pipeline == []


def test_augment_config_empty_yaml_does_not_crash():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = os.path.join(tmp_dir, "empty.yaml")
        open(yaml_path, "w").close()

        dataset = _make_dataset(yaml_path)

    assert dataset.augment_pipeline == []


def test_augment_config_non_mapping_root_does_not_crash():
    """Valid YAML whose root is a list (or scalar) parses fine but would
    crash the `.items()` call that builds the pipeline -- must degrade to
    no augmenters instead."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = os.path.join(tmp_dir, "list_root.yaml")
        with open(yaml_path, "w") as fp:
            fp.write("- AddNoise\n- Crop\n")

        dataset = _make_dataset(yaml_path)

    assert dataset.augment_pipeline == []
