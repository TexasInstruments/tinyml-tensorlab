"""Regression tests for two image_classification/train.py and
audio_classification/train.py post-loop bugs.

1. audio_classification/train.py logged `best['f1']` under the
   "AUC ROC Score" label instead of the `best['auc']` value that was
   actually computed and stored in `best` on every improving epoch.
   image_classification/train.py already logged this correctly.

2. When `for epoch in range(args.start_epoch, args.epochs)` runs zero
   iterations -- e.g. `--resume` pointed at a checkpoint that already
   satisfies `--epochs`, a real and expected use case for re-running only
   the post-training export/compile steps -- both scripts' post-loop
   "Log best epoch results" section read `best['predictions']` /
   `best['ground_truth']`, which are only ever assigned inside the loop
   body, crashing with KeyError (best = dict(accuracy=0.0, f1=0,
   conf_matrix=dict(), epoch=None) has no 'predictions'/'ground_truth'/
   'auc' keys until an improving epoch runs). Both now guard the whole
   block on `best['epoch'] is not None`.

These tests drive each script's real main() through a full but heavily
mocked model/data pipeline (mocking every heavy helper imported from
common/train_base.py and common/models.py) with args.start_epoch ==
args.epochs, so the training loop body never executes -- reproducing
exactly the "--resume to an already-completed checkpoint" scenario -- and
assert main() completes without raising.
"""
import os
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from tinyml_tinyverse.references.audio_classification import train as audio_train
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


def test_image_classification_train_main_survives_zero_iteration_resume(tmp_path):
    """image_classification/train.py already logged best['auc'] correctly
    (no mislabel bug here) -- only the empty-loop KeyError crash applies."""
    args = _base_classification_args(tmp_path)

    with ExitStack() as stack:
        stack.enter_context(_dataset_load_state_patch(img_train, _FakeImageClassificationDataset()))
        stack.enter_context(patch.object(
            img_train, "setup_training_environment",
            return_value=(img_train.getLogger("test"), torch.device("cpu"))))
        stack.enter_context(patch.object(img_train, "prepare_transforms"))
        stack.enter_context(patch.object(img_train, "create_data_loaders", return_value=_fake_data_loaders()))
        stack.enter_context(patch.object(img_train.models, "get_model", return_value=torch.nn.Identity()))
        stack.enter_context(patch.object(img_train, "load_pretrained_weights", side_effect=lambda model, a, _logger: model))
        stack.enter_context(patch.object(img_train, "handle_export_only", return_value=False))
        stack.enter_context(patch.object(img_train, "move_model_to_device"))
        stack.enter_context(patch.object(
            img_train, "setup_distributed_model", side_effect=lambda model, a, d: (model, model, None)))
        stack.enter_context(patch.object(img_train, "setup_optimizer_and_scheduler", return_value=(MagicMock(), MagicMock())))
        stack.enter_context(patch.object(img_train, "resume_from_checkpoint"))
        stack.enter_context(patch.object(img_train.utils, "quantization_wrapped_model", side_effect=lambda model, *a, **kw: model))
        stack.enter_context(patch.object(img_train.utils, "export_model"))
        stack.enter_context(patch.object(img_train, "log_training_time"))
        stack.enter_context(patch.object(img_train, "shutdown_data_loaders"))
        img_train.main(0, args)


class _FakeAudioClassificationDataset:
    classes = ["a", "b"]
    X = np.zeros((4, 3, 4), dtype=np.float32)
    inverse_label_map = {0: "a", 1: "b"}


def _base_audio_args(tmp_dir, **overrides):
    tmp_dir = str(tmp_dir)
    args = Namespace(
        quantization=True, dont_train_just_feat_ext='False', load_saved_model='None',
        nas_enabled=False, generic_model=False, nn_for_feature_extraction=False,
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


def _patch_audio_common_pipeline(stack, dataset):
    stack.enter_context(_dataset_load_state_patch(audio_train, dataset))
    stack.enter_context(patch.object(
        audio_train, "setup_training_environment",
        return_value=(audio_train.getLogger("test"), torch.device("cpu"))))
    stack.enter_context(patch.object(audio_train, "prepare_transforms"))
    stack.enter_context(patch.object(audio_train, "create_data_loaders", return_value=_fake_data_loaders()))
    stack.enter_context(patch.object(audio_train.models, "get_model", return_value=torch.nn.Identity()))
    stack.enter_context(patch.object(audio_train, "load_pretrained_weights", side_effect=lambda model, a, _logger: model))
    stack.enter_context(patch.object(audio_train, "handle_export_only", return_value=False))
    stack.enter_context(patch.object(audio_train, "move_model_to_device"))
    stack.enter_context(patch.object(
        audio_train, "setup_distributed_model", side_effect=lambda model, a, d: (model, model, None)))
    stack.enter_context(patch.object(audio_train, "setup_optimizer_and_scheduler", return_value=(MagicMock(), MagicMock())))
    stack.enter_context(patch.object(audio_train, "resume_from_checkpoint"))
    stack.enter_context(patch.object(audio_train.utils, "quantization_wrapped_model", side_effect=lambda model, *a, **kw: model))
    stack.enter_context(patch.object(audio_train.utils, "export_model"))
    stack.enter_context(patch.object(audio_train, "log_training_time"))


def test_audio_classification_train_main_survives_zero_iteration_resume(tmp_path):
    args = _base_audio_args(tmp_path)

    with ExitStack() as stack:
        _patch_audio_common_pipeline(stack, _FakeAudioClassificationDataset())
        audio_train.main(0, args)


def test_audio_classification_train_logs_auc_not_f1_as_auc_roc_score(caplog, tmp_path):
    args = _base_audio_args(tmp_path, epochs=1, start_epoch=0)

    avg_conf_matrix = [[2, 0], [0, 2]]
    fake_predictions = torch.zeros(4)
    fake_ground_truth = torch.zeros(4)
    evaluate_return = (99.0, 11.0, 77.0, avg_conf_matrix, fake_predictions, fake_ground_truth)

    with ExitStack() as stack:
        _patch_audio_common_pipeline(stack, _FakeAudioClassificationDataset())
        stack.enter_context(patch.object(audio_train.utils, "train_one_epoch_classification"))
        stack.enter_context(patch.object(audio_train.utils, "evaluate_classification", return_value=evaluate_return))
        stack.enter_context(patch.object(audio_train, "save_checkpoint", return_value={}))
        stack.enter_context(patch.object(audio_train.utils, "save_on_master"))
        stack.enter_context(patch.object(audio_train.utils, "print_file_level_classification_summary"))
        with caplog.at_level("INFO"):
            audio_train.main(0, args)

    auc_lines = [r.message for r in caplog.records if "AUC ROC Score" in r.message]
    assert auc_lines, "expected an 'AUC ROC Score' log line"
    assert "77.000" in auc_lines[0], f"expected the auc value (77.0) in: {auc_lines[0]}"
    assert "11.000" not in auc_lines[0], f"AUC ROC Score line wrongly logged the f1 value: {auc_lines[0]}"
