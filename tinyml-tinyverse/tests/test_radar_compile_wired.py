"""Regression test: radar_classification.train.main() must call
compile_model_if_enabled, matching the pattern already used in the 4
timeseries_* reference scripts (PR #22). Without this, --compile-model is
silently a no-op for radar training regardless of what the caller requests.

A prior version of this test only did an `inspect.getsource` substring
check for "compile_model_if_enabled(" in main()'s source. A peer review
found that pattern would pass under several real regressions: the call
deleted (but the string surviving in a comment), the call's return value
not reassigned to `model`, the call moved to the wrong position, or the
call placed in a dead/unreachable branch. This test instead patches
compile_model_if_enabled and drives the real main() (mocking everything
else, following the harness in test_radar_quant_only_no_crash.py), then
asserts on the actual call: it happens exactly once, with the pre-wrap
model and the expected input_shape kwarg, and its return value is what
flows into setup_distributed_model -- not silently discarded.
"""
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import torch

from tinyml_tinyverse.references.radar_classification import train as radar_train


class _FakeDataset:
    classes = ['a', 'b']
    inverse_label_map = {0: 'a', 1: 'b'}
    X = torch.zeros((4, 8))
    Y = torch.zeros((4,), dtype=torch.long)

    def __getitem__(self, i):
        return self.X[i], self.X[i], self.Y[i]

    def __len__(self):
        return 4


def test_main_calls_compile_model_if_enabled_and_uses_its_return_value():
    radar_train.dataset_load_state['dataset'] = None
    radar_train.dataset_load_state['dataset_test'] = None
    radar_train.dataset_load_state['train_sampler'] = None
    radar_train.dataset_load_state['test_sampler'] = None

    args = Namespace(
        quantization=True, data_path='/fake', output_dir='/tmp/fake-radar-compile-wired',
        gof_test=False, frame_size='None', dont_train_just_feat_ext='False',
        load_saved_model='None', nas_enabled='False', generic_model=True,
        model='LINEAR_4L_PC', model_config=None, model_spec=None, dual_op=False,
        output_int=True, quantization_method='QAT', weight_bitwidth=8,
        activation_bitwidth=8, epochs=1, start_epoch=0, label_smoothing=0.0,
        distributed=False, apex=False, print_freq=10, opset_version=17,
        gen_golden_vectors=False, DEBUG=False,
        file_level_classification_log='/tmp/fake-radar-compile-wired/file_level.log',
    )

    fake_dataset = _FakeDataset()
    # Each "batch" must be subscriptable like (raw, features, target) -- main()'s
    # export step does `next(iter(data_loader_test))[1]` to get an example input.
    fake_batch = (torch.zeros((1, 8)), torch.zeros((1, 8)), torch.zeros((1,), dtype=torch.long))
    fake_loaders = ([fake_batch], [fake_batch])

    pre_wrap_model = torch.nn.Linear(8, 2)
    # Distinct object returned by compile_model_if_enabled, so we can prove it --
    # not the pre-wrap model -- is what continues into setup_distributed_model.
    compiled_model_sentinel = torch.nn.Sequential(pre_wrap_model)

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            radar_train, "setup_training_environment",
            return_value=(radar_train.getLogger("test"), torch.device("cpu"))))
        stack.enter_context(patch.object(radar_train, "prepare_transforms"))
        stack.enter_context(patch.object(
            radar_train, "load_datasets",
            return_value=(fake_dataset, fake_dataset, None, None)))
        stack.enter_context(patch.object(radar_train, "create_data_loaders", return_value=fake_loaders))
        stack.enter_context(patch.object(radar_train.models, "get_model", return_value=pre_wrap_model))
        stack.enter_context(patch.object(radar_train, "log_model_summary"))
        stack.enter_context(patch.object(radar_train, "load_pretrained_weights", side_effect=lambda m, a, l: m))
        stack.enter_context(patch.object(radar_train, "handle_export_only", return_value=False))
        # Shared recorder so we can assert move_model_to_device runs BEFORE
        # compile_model_if_enabled. move_model_to_device mutates the model
        # in-place (model.to(device)) and has no captured return value in
        # production code, so a bare no-op mock has no observable side
        # effect that distinguishes call order -- side_effect here is what
        # makes the order observable.
        call_order = []
        stack.enter_context(patch.object(
            radar_train, "move_model_to_device",
            side_effect=lambda *a, **kw: call_order.append("move_model_to_device")))
        mock_compile = stack.enter_context(patch.object(
            radar_train, "compile_model_if_enabled",
            side_effect=lambda *a, **kw: (call_order.append("compile_model_if_enabled"), compiled_model_sentinel)[1]))
        mock_setup_distributed = stack.enter_context(patch.object(
            radar_train, "setup_distributed_model", side_effect=lambda m, a, d: (m, m, None)))
        stack.enter_context(patch.object(
            radar_train, "setup_optimizer_and_scheduler", return_value=(MagicMock(), MagicMock())))
        stack.enter_context(patch.object(radar_train, "resume_from_checkpoint"))
        stack.enter_context(patch.object(radar_train.utils, "quantization_wrapped_model", side_effect=lambda m, *a, **kw: m))
        stack.enter_context(patch.object(radar_train.utils, "train_one_epoch_classification"))
        stack.enter_context(patch.object(
            radar_train.utils, "evaluate_classification",
            return_value=(1.0, 1.0, 1.0, {}, [], [])))
        stack.enter_context(patch.object(radar_train, "save_checkpoint"))
        stack.enter_context(patch.object(radar_train.utils, "save_on_master"))
        stack.enter_context(patch.object(radar_train.utils, "print_file_level_classification_summary"))
        stack.enter_context(patch.object(radar_train.utils, "export_model"))
        stack.enter_context(patch.object(radar_train, "log_training_time"))

        radar_train.main(0, args)

    # Called exactly once, with the pre-wrap model as the first positional arg.
    mock_compile.assert_called_once()
    call_args, call_kwargs = mock_compile.call_args
    assert call_args[0] is pre_wrap_model, (
        "compile_model_if_enabled's first positional arg should be the model "
        "returned by models.get_model() (pre-setup_distributed_model, "
        "pre-NeuralNetworkWithPreprocess-wrap), but got something else."
    )

    # input_shape kwarg must be the actual expected value, not merely present.
    expected_input_shape = (1,) + fake_dataset.X.shape[1:]
    assert call_kwargs.get("input_shape") == expected_input_shape, (
        f"expected input_shape={expected_input_shape!r}, got {call_kwargs.get('input_shape')!r}"
    )

    # The compiled/wrapped return value must be what flows into
    # setup_distributed_model -- not silently discarded.
    mock_setup_distributed.assert_called_once()
    setup_distributed_call_args = mock_setup_distributed.call_args[0]
    assert setup_distributed_call_args[0] is compiled_model_sentinel, (
        "setup_distributed_model should receive compile_model_if_enabled's return "
        "value, but received something else -- the compiled model's return value "
        "looks like it was discarded (model = compile_model_if_enabled(...) not "
        "assigned back to `model`)."
    )

    # Order matters: compile must run on a model that's already on its
    # target device, not before.
    assert call_order == ["move_model_to_device", "compile_model_if_enabled"], (
        f"Expected move_model_to_device to run before compile_model_if_enabled, got order: {call_order}"
    )
