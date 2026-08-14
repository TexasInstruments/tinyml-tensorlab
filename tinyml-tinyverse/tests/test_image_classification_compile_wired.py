"""Regression test: image_classification.train.main() must call
compile_model_if_enabled, matching the timeseries_* pattern (PR #22).
Without this, --compile-model is silently a no-op for image classification
training regardless of what the caller requests.

A prior version of this test only did an `inspect.getsource` substring
check for "compile_model_if_enabled(" in main()'s source. A peer review
found that pattern would pass under several real regressions: the call
deleted (but the string surviving in a comment), the call's return value
not reassigned to `model`, the call moved to the wrong position, or the
call placed in a dead/unreachable branch. This test instead patches
compile_model_if_enabled and drives the real main() (reusing the mock
harness from test_train_best_epoch_bugs_vision_audio.py), then asserts on
the actual call: it happens exactly once, with the pre-wrap model and the
expected input_shape kwarg, and its return value is what flows into
setup_distributed_model -- not silently discarded.
"""
import os
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from tinyml_tinyverse.references.image_classification import train as img_train


def _fake_data_loaders():
    item = (torch.tensor(0), torch.zeros(1, 3, 4), torch.tensor(0))
    return [item], [item]


def _base_classification_args(tmp_dir, **overrides):
    tmp_dir = str(tmp_dir)
    args = Namespace(
        quantization=True, dont_train_just_feat_ext='False', load_saved_model='None',
        nas_enabled='False', generic_model=False, nn_for_feature_extraction=False,
        output_int=True, auto_quantization=False, distributed=False, gen_golden_vectors=False,
        model='dummy', model_config=None, model_spec=None, dual_op=False,
        label_smoothing=0.0, apex=False, print_freq=10, quantization_method='QAT',
        epochs=3, start_epoch=3,  # loop runs zero iterations, like a fully-resumed checkpoint
        output_dir=tmp_dir, weight_bitwidth=8, activation_bitwidth=8,
        autoquant_tolerance_classification=0.1, opset_version=17, device='cpu',
        file_level_classification_log=os.path.join(tmp_dir, 'file_level.log'), DEBUG=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class _FakeImageClassificationDataset:
    classes = ["a", "b"]
    X = np.zeros((4, 3, 4), dtype=np.float32)
    inverse_label_map = {0: "a", 1: "b"}


def _dataset_load_state_patch(train_module, dataset):
    """patch.dict so the module-global dataset_load_state mutation is undone
    after each test instead of leaking into later tests in the session."""
    return patch.dict(train_module.dataset_load_state, {
        'dataset': dataset, 'dataset_test': dataset,
        'train_sampler': None, 'test_sampler': None,
    })


def test_main_calls_compile_model_if_enabled_and_uses_its_return_value(tmp_path):
    args = _base_classification_args(tmp_path)
    dataset = _FakeImageClassificationDataset()

    pre_wrap_model = torch.nn.Identity()
    # Distinct object returned by compile_model_if_enabled, so we can prove it --
    # not the pre-wrap model -- is what continues into setup_distributed_model.
    compiled_model_sentinel = torch.nn.Sequential(pre_wrap_model)

    with ExitStack() as stack:
        stack.enter_context(_dataset_load_state_patch(img_train, dataset))
        stack.enter_context(patch.object(
            img_train, "setup_training_environment",
            return_value=(img_train.getLogger("test"), torch.device("cpu"))))
        stack.enter_context(patch.object(img_train, "prepare_transforms"))
        stack.enter_context(patch.object(img_train, "create_data_loaders", return_value=_fake_data_loaders()))
        stack.enter_context(patch.object(img_train.models, "get_model", return_value=pre_wrap_model))
        stack.enter_context(patch.object(img_train, "load_pretrained_weights", side_effect=lambda model, a, _logger: model))
        stack.enter_context(patch.object(img_train, "handle_export_only", return_value=False))
        stack.enter_context(patch.object(img_train, "move_model_to_device"))
        mock_compile = stack.enter_context(patch.object(
            img_train, "compile_model_if_enabled", return_value=compiled_model_sentinel))
        mock_setup_distributed = stack.enter_context(patch.object(
            img_train, "setup_distributed_model", side_effect=lambda model, a, d: (model, model, None)))
        stack.enter_context(patch.object(img_train, "setup_optimizer_and_scheduler", return_value=(MagicMock(), MagicMock())))
        stack.enter_context(patch.object(img_train, "resume_from_checkpoint"))
        stack.enter_context(patch.object(img_train.utils, "quantization_wrapped_model", side_effect=lambda model, *a, **kw: model))
        stack.enter_context(patch.object(img_train.utils, "export_model"))
        stack.enter_context(patch.object(img_train, "log_training_time"))
        stack.enter_context(patch.object(img_train, "shutdown_data_loaders"))
        img_train.main(0, args)

    # Called exactly once, with the pre-wrap model as the first positional arg.
    mock_compile.assert_called_once()
    call_args, call_kwargs = mock_compile.call_args
    assert call_args[0] is pre_wrap_model, (
        "compile_model_if_enabled's first positional arg should be the model "
        "returned by models.get_model() (pre-setup_distributed_model), but got "
        "something else."
    )

    # input_shape kwarg must be the actual expected value, not merely present.
    expected_input_shape = (1,) + dataset.X.shape[1:]
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
