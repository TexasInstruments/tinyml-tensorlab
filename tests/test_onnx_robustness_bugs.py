"""Regression tests for two test_onnx.py robustness bugs.

1. timeseries_anomalydetection/test_onnx.py's get_reconstruction_errors_stats()
   built its DataLoader with `pin_memory=True if args.gpu > 0 else False`, but
   the script's argparser (common/test_onnx_base.py) only ever defines
   `--gpus` (plural), never `--gpu`. Every call raised:
     AttributeError: 'Namespace' object has no attribute 'gpu'
   before any model loading or data processing even started. The same
   file's main() had a related (non-crashing but still wrong) variant:
   `pin_memory=True if gpu > 0 else False`, gating on the DDP rank
   parameter rather than device type. Both are now unconditional
   `pin_memory=True`, matching every sibling test_onnx.py.

2. audio_classification/test_onnx.py never imported or called
   shutdown_data_loaders() on its DataLoader, unlike every one of its five
   sibling test_onnx.py scripts (image_classification, timeseries_
   classification, timeseries_forecasting, timeseries_regression,
   timeseries_anomalydetection), leaking DataLoader worker processes /
   POSIX semaphores whenever --workers > 0.
"""
import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tinyml_tinyverse.references.audio_classification import test_onnx as audio_test_onnx
from tinyml_tinyverse.references.timeseries_anomalydetection import test_onnx as anomaly_test_onnx


class _FakeAnomalyDataset(torch.utils.data.Dataset):
    """(raw, data, label) tuples matching what utils.collate_fn expects, with
    data shaped (C, H, W) = (1, 4, 4) so the per-sample reconstruction-error
    reduction (dim=(1, 2, 3) over an unsqueezed (1, C, H, W) tensor) has real
    dimensions to reduce over."""

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return torch.tensor(idx), torch.zeros(1, 4, 4), torch.tensor(0)


class _FakeAudioDataset(torch.utils.data.Dataset):
    classes = ["a", "b"]

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return torch.tensor(idx), torch.zeros(1, 4), torch.tensor(0)


def _fake_ort_sess():
    sess = MagicMock()
    sess.run.return_value = [np.zeros((1, 1, 4, 4), dtype=np.float32)]
    return sess


def test_get_reconstruction_errors_stats_does_not_crash_without_args_gpu():
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = Namespace(
            output_dir=tmp_dir, lis=None, DEBUG=False, seed=0, device="cpu",
            data_path="/fake/data", batch_size=2, workers=0,
            gpus=1,  # note: no `gpu` attribute -- matches the real argparser
            model_path="/fake/model.onnx", generic_model=True,
        )
        fake_dataset = _FakeAnomalyDataset()

        with patch.object(anomaly_test_onnx, "prepare_transforms"), \
             patch.object(anomaly_test_onnx.utils, "load_data",
                          return_value=(fake_dataset, fake_dataset, None, None)), \
             patch.object(anomaly_test_onnx, "load_onnx_model",
                          return_value=(_fake_ort_sess(), "input", "output")):
            mean, std = anomaly_test_onnx.get_reconstruction_errors_stats(args)

    assert torch.is_tensor(mean)
    assert torch.is_tensor(std)


def test_audio_test_onnx_shuts_down_data_loader_even_when_model_load_fails():
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = Namespace(
            output_dir=tmp_dir, lis=None, DEBUG=False, seed=0, device="cpu",
            data_path="/fake/data", batch_size=2, workers=0,
            model_path="/fake/model.onnx", generic_model=True,
            distributed=False, nn_for_feature_extraction=False,
        )
        fake_dataset = _FakeAudioDataset()

        with patch.object(audio_test_onnx.utils, "init_distributed_mode"), \
             patch.object(audio_test_onnx, "prepare_transforms"), \
             patch.object(audio_test_onnx.utils, "load_data",
                          return_value=(fake_dataset, fake_dataset, None, None)), \
             patch.object(audio_test_onnx, "load_onnx_model", side_effect=RuntimeError("boom")), \
             patch.object(audio_test_onnx, "shutdown_data_loaders") as mock_shutdown:
            with pytest.raises(RuntimeError):
                audio_test_onnx.main(0, args)

        mock_shutdown.assert_called_once()
