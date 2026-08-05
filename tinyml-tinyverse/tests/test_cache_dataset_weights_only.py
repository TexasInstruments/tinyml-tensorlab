"""Regression tests for two --cache-dataset bugs in load_data().

1. Both cache-hit branches called torch.load(cache_path) with no
   weights_only argument. The cached object is a (dataset, datadir) tuple
   where dataset is one of this project's own Dataset subclasses -- not the
   kind of type torch's weights_only=True default (PyTorch 2.6+)
   allowlists, so any run reusing a --cache-dataset cache failed with
   UnpicklingError: Weights only load failed.

2. An independent peer review of the fix for (1) caught a second, more
   serious bug _get_cache_path(datadir) exposed: it computed the SAME
   cache path for the training dataset and the validation dataset (both
   derived only from datadir), and only the training branch ever wrote its
   cache -- the validation write-back was dead code, commented out with
   "TODO: Add utils and uncomment the if block". On the very first run
   with --cache-dataset enabled, the validation cache-hit check
   (os.path.exists(cache_path)) became true immediately after the training
   branch's write (same path), so dataset_test silently became a
   torch.load() of the TRAINING dataset -- not a crash, just meaningless
   validation metrics (train==val) from that point on. Fixed by giving
   _get_cache_path a `tag` parameter ('train'/'val') folded into the hash,
   producing distinct paths, and by enabling the validation cache
   write-back that was dead code.
"""
import os
import tempfile
from argparse import Namespace
from unittest.mock import patch

import torch

from tinyml_tinyverse.common.utils import utils


class _FakeDataset:
    """Stands in for a real Dataset subclass -- the point is only that it's a
    custom class instance, not the specific dataset implementation. Needs
    __len__ since load_data() wraps the loaded dataset in a
    RandomSampler/SequentialSampler afterward."""

    def __init__(self, tag):
        self.tag = tag

    def __len__(self):
        return 4


class _StubPreparable:
    """Stands in for what dataset_loader(subset, dataset_dir=..., **kwargs)
    returns -- load_data() immediately calls .prepare(**kwargs) on it."""

    def __init__(self, dataset):
        self._dataset = dataset

    def prepare(self, **kwargs):
        return self._dataset


def _fake_cache_path(tmp_dir):
    def _get(datadir, tag='train', args=None):
        return os.path.join(tmp_dir, f"cache_{tag}.pt")
    return _get


def test_get_cache_path_differs_between_train_and_val_for_the_same_datadir():
    train_path = utils._get_cache_path("/some/datadir", tag='train')
    val_path = utils._get_cache_path("/some/datadir", tag='val')
    assert train_path != val_path


def test_get_cache_path_differs_when_dataset_config_changes():
    """The cached object is the fully PREPARED dataset -- transforms and
    loader settings are baked in ("Attention, as the transforms are also
    cached!"). A run with a different transform pipeline against the same
    datadir must therefore miss the cache, not silently reuse a dataset
    prepared under the old configuration."""
    args_a = Namespace(feat_ext_transform=["MFCC"], dataset_loader="GenericTSDataset")
    args_b = Namespace(feat_ext_transform=["RAW"], dataset_loader="GenericTSDataset")

    path_a = utils._get_cache_path("/some/datadir", tag='train', args=args_a)
    path_a_again = utils._get_cache_path("/some/datadir", tag='train', args=args_a)
    path_b = utils._get_cache_path("/some/datadir", tag='train', args=args_b)

    assert path_a == path_a_again  # stable for an equivalent configuration
    assert path_a != path_b  # any config change invalidates


def test_save_cache_atomically_publishes_only_a_complete_file(tmp_path):
    """A reader that sees cache_path exist must never read a partial file:
    the write goes to a .tmp sibling and is published via os.replace."""
    cache_path = str(tmp_path / "sub" / "cache_train.pt")

    utils._save_cache_atomically((_FakeDataset("payload"), "/some/datadir"), cache_path)

    assert os.path.exists(cache_path)
    assert not os.path.exists(cache_path + ".tmp")
    dataset, _ = torch.load(cache_path, weights_only=False)
    assert dataset.tag == "payload"


def test_load_data_reads_distinct_cached_train_and_val_datasets_without_weights_only_error():
    with tempfile.TemporaryDirectory() as tmp_dir:
        get_cache_path = _fake_cache_path(tmp_dir)
        torch.save((_FakeDataset("cached-train"), "/some/datadir"),
                   get_cache_path("/some/datadir", tag='train'))
        torch.save((_FakeDataset("cached-val"), "/some/datadir"),
                   get_cache_path("/some/datadir", tag='val'))

        args = Namespace(cache_dataset=True, dataset_loader="unused",
                          distributed=False, loader_type="regression")

        with patch.object(utils, "_get_cache_path", side_effect=get_cache_path):
            dataset, dataset_test, train_sampler, test_sampler = utils.load_data(
                "/some/datadir", args, dataset_loader_dict={}
            )

    assert dataset.tag == "cached-train"
    assert dataset_test.tag == "cached-val"


def test_load_data_writes_a_validation_cache_when_missing():
    """The validation cache write-back was dead code, so --cache-dataset
    never actually cached the validation dataset even after a successful
    training-cache write. Confirm the val cache file gets written now."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        get_cache_path = _fake_cache_path(tmp_dir)
        train_cache_path = get_cache_path("/some/datadir", tag='train')
        val_cache_path = get_cache_path("/some/datadir", tag='val')
        torch.save((_FakeDataset("cached-train"), "/some/datadir"), train_cache_path)
        assert not os.path.exists(val_cache_path)

        def _fake_loader(subset, dataset_dir, **kwargs):
            return _StubPreparable(_FakeDataset(f"loaded-{subset}"))

        args = Namespace(cache_dataset=True, dataset_loader="fake", dataset="local",
                          data_path="/some/datadir", distributed=False, loader_type="regression")

        with patch.object(utils, "_get_cache_path", side_effect=get_cache_path):
            dataset, dataset_test, _, _ = utils.load_data(
                "/some/datadir", args, dataset_loader_dict={"fake": _fake_loader}
            )

        assert dataset.tag == "cached-train"
        assert dataset_test.tag == "loaded-val"
        assert os.path.exists(val_cache_path)
