"""Regression test for a crash introduced by this session's own H2D/MPS fix.

timeseries_anomalydetection/train.py's get_reconstruction_errors_stats()
gates non_blocking transfers with `device.type == 'cuda'` (the fix applied
across the codebase to stop unsafe non_blocking=True on non-CUDA devices).
Before that fix, non_blocking was hardcoded True and never touched `device`
beyond passing it straight to `.to(device, ...)`, which tolerates a plain
device string ('cuda') just fine.

The fix exposed a latent bug in the caller: main() called this function
with `args.device` -- the raw argparse string ('cuda' by default, never
converted to a torch.device anywhere in this file) -- instead of `device`,
the torch.device already constructed by setup_training_environment() and
in scope in main(). `device.type` on a plain str raises AttributeError, so
this crashed on every anomaly-detection training run, right after export,
while calculating the detection threshold -- not an edge case, the normal
path.

Fixed by passing the existing local `device` (torch.device) instead of
`args.device` (str) at the call site.

Two tests:
1. A function-contract test characterizing get_reconstruction_errors_stats()
   directly: it works with a real torch.device and crashes with a raw
   device string -- this is the exact defect surface and locks in the
   function's contract against future regressions.
2. A main()-level test that actually drives the real call site (heavily
   mocking every other dependency, with args.start_epoch == args.epochs so
   the training loop is skipped and get_reconstruction_errors_stats is
   reached quickly) and asserts what main() actually passes as the device
   argument -- this is the one that genuinely exercises the fixed line,
   since test 1 alone would pass identically whether or not the call site
   itself were fixed.
"""
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tinyml_tinyverse.references.timeseries_anomalydetection import train as anomaly_train


class _FakeAnomalyDataset(torch.utils.data.Dataset):
    """(raw, data, label) tuples matching what utils.collate_fn expects, with
    data shaped (C, H, W) = (1, 4, 4) so the per-sample reconstruction-error
    reduction (dim=(1, 2, 3) over an unsqueezed (1, C, H, W) tensor) has real
    dimensions to reduce over."""

    classes = ["a", "b"]
    X = np.zeros((4, 3, 4), dtype=np.float32)

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return torch.tensor(idx), torch.zeros(1, 4, 4), torch.tensor(0)


def _fake_ort_sess():
    sess = MagicMock()
    sess.run.return_value = [np.zeros((1, 1, 4, 4), dtype=np.float32)]
    return sess


def _fake_data_loader():
    return torch.utils.data.DataLoader(
        _FakeAnomalyDataset(), batch_size=2, collate_fn=anomaly_train.utils.collate_fn)


def test_get_reconstruction_errors_stats_works_with_a_real_torch_device():
    """The fixed call site now passes a real torch.device -- confirm the
    function actually works with one (device.type resolves normally)."""
    with patch.object(anomaly_train.ort, "InferenceSession", return_value=_fake_ort_sess()):
        mean, std = anomaly_train.get_reconstruction_errors_stats(
            generic_model=True, model_path="/fake/model.onnx",
            device=torch.device("cpu"), data_loader=_fake_data_loader(),
        )

    assert torch.is_tensor(mean)
    assert torch.is_tensor(std)


def test_get_reconstruction_errors_stats_crashes_if_given_a_raw_device_string():
    """Characterizes the exact regression the fix closes: this is what the
    pre-fix call site (`get_reconstruction_errors_stats(..., args.device,
    ...)`, args.device being the unconverted argparse string) actually
    passed, and it crashes immediately on `device.type`."""
    with patch.object(anomaly_train.ort, "InferenceSession", return_value=_fake_ort_sess()):
        with pytest.raises(AttributeError, match="'str' object has no attribute 'type'"):
            anomaly_train.get_reconstruction_errors_stats(
                generic_model=True, model_path="/fake/model.onnx",
                device="cpu", data_loader=_fake_data_loader(),
            )


def test_main_passes_a_torch_device_not_the_raw_args_device_string():
    """Drives the real main() (heavily mocked elsewhere) far enough to reach
    the actual call site and captures what it passes. This is the test that
    genuinely exercises the fixed line -- the two tests above only
    characterize the callee's contract and would pass unchanged whether or
    not main()'s call site itself were fixed."""
    args = Namespace(
        quantization=True, ondevice_training=False, model='dummy', model_config=None,
        model_spec=None, dual_op=False, output_int=True, auto_quantization=False,
        weight_bitwidth=8, activation_bitwidth=8, epochs=1, start_epoch=1,  # zero iterations
        quantization_method='QAT', distributed=False, apex=False, print_freq=10,
        output_dir='/tmp/fake-output', autoquant_tolerance_anomaly=0.1,
        opset_version=17, generic_model=True, gen_golden_vectors=False,
        device='cuda',  # the raw argparse string that must NOT reach get_reconstruction_errors_stats
    )
    dataset = _FakeAnomalyDataset()
    anomaly_train.dataset_load_state['dataset'] = dataset
    anomaly_train.dataset_load_state['dataset_test'] = dataset
    anomaly_train.dataset_load_state['train_sampler'] = None
    anomaly_train.dataset_load_state['test_sampler'] = None

    real_device = torch.device("cpu")
    fake_loaders = ([1, 2], [1, 2])

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            anomaly_train, "setup_training_environment", return_value=(anomaly_train.getLogger("test"), real_device)))
        stack.enter_context(patch.object(anomaly_train, "prepare_transforms"))
        stack.enter_context(patch.object(anomaly_train, "create_data_loaders", return_value=fake_loaders))
        stack.enter_context(patch.object(anomaly_train.models, "get_model", return_value=torch.nn.Identity()))
        stack.enter_context(patch.object(anomaly_train, "log_model_summary"))
        stack.enter_context(patch.object(anomaly_train, "load_pretrained_weights", side_effect=lambda model, a, l: model))
        stack.enter_context(patch.object(anomaly_train, "handle_export_only", return_value=False))
        stack.enter_context(patch.object(anomaly_train, "move_model_to_device"))
        stack.enter_context(patch.object(anomaly_train, "compile_model_if_enabled", side_effect=lambda model, a, l: model))
        stack.enter_context(patch.object(anomaly_train.utils, "quantization_wrapped_model", side_effect=lambda model, *a, **kw: model))
        stack.enter_context(patch.object(anomaly_train, "setup_optimizer_and_scheduler", return_value=(MagicMock(), MagicMock())))
        stack.enter_context(patch.object(
            anomaly_train, "setup_distributed_model", side_effect=lambda model, a, d: (model, model, None)))
        stack.enter_context(patch.object(anomaly_train, "resume_from_checkpoint"))
        stack.enter_context(patch.object(anomaly_train, "get_amp_context", return_value=MagicMock()))
        stack.enter_context(patch.object(anomaly_train, "get_grad_scaler", return_value=None))
        stack.enter_context(patch.object(anomaly_train.utils, "export_model"))
        stack.enter_context(patch.object(anomaly_train, "log_training_time"))
        stack.enter_context(patch.object(anomaly_train, "shutdown_data_loaders"))
        mock_get_stats = stack.enter_context(patch.object(
            anomaly_train, "get_reconstruction_errors_stats",
            return_value=(torch.tensor(0.0), torch.tensor(0.0))))

        anomaly_train.main(0, args)

    mock_get_stats.assert_called_once()
    passed_device = mock_get_stats.call_args[0][2]
    assert passed_device is real_device, (
        f"main() passed {passed_device!r} (type {type(passed_device).__name__}) as the device "
        "argument -- expected the real torch.device from setup_training_environment(), not "
        "args.device (a raw string)."
    )
