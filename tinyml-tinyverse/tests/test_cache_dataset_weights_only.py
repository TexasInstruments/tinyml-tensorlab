"""Regression test for load_data()'s --cache-dataset torch.load() crash.

Both cache-hit branches in load_data() (training data and validation data)
called torch.load(cache_path) with no weights_only argument. The cached
object is a (dataset, datadir) tuple where dataset is one of this project's
own Dataset subclasses -- not the kind of type torch's weights_only=True
default (PyTorch 2.6+) allowlists, so any second run using --cache-dataset
(after the first run wrote the cache) failed with
UnpicklingError: Weights only load failed.
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


def test_load_data_reads_a_cached_dataset_without_weights_only_error():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_path = os.path.join(tmp_dir, "cache.pt")
        torch.save((_FakeDataset("cached"), "/some/datadir"), cache_path)

        args = Namespace(cache_dataset=True, dataset_loader="unused",
                          distributed=False, loader_type="regression")

        with patch.object(utils, "_get_cache_path", return_value=cache_path):
            dataset, dataset_test, train_sampler, test_sampler = utils.load_data(
                "/some/datadir", args, dataset_loader_dict={}
            )

    assert isinstance(dataset, _FakeDataset)
    assert dataset.tag == "cached"
    assert isinstance(dataset_test, _FakeDataset)
