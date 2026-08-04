"""Regression tests for two timeseries_classification/train.py and
timeseries_forecasting/train.py post-loop bugs.

1. timeseries_classification/train.py logged `best['f1']` under the
   "AUC ROC Score" label instead of the `best['auc']` value that was
   actually computed and stored in `best` on every improving epoch.

2. When `for epoch in range(args.start_epoch, args.epochs)` runs zero
   iterations -- e.g. `--resume` pointed at a checkpoint that already
   satisfies `--epochs`, a real and expected use case for re-running only
   the post-training export/compile steps -- the post-loop "Log best epoch
   results" section read dict keys that are only ever assigned inside the
   loop body, crashing:
     - timeseries_classification/train.py: KeyError on best['predictions']
       (best = dict(accuracy=0.0, f1=0, conf_matrix=dict(), epoch=None) has
       no 'predictions'/'ground_truth' keys until an improving epoch runs).
       Now guards the whole block on `best['epoch'] is not None`.
     - timeseries_forecasting/train.py: TypeError (None subscript) on
       best_epoch_values['true_values'][:, :, idx] (true_values/predictions
       are pre-populated with None, not omitted). Now guards the whole
       block on `best_epoch_values['true_values'] is not None`.

(timeseries_regression and timeseries_anomalydetection were already safe --
their `best` dicts only ever hold scalars with safe initial values, so they
have no test here.)

These tests drive each script's real main() through a full but heavily
mocked model/data pipeline (mocking every heavy helper imported from
common/train_base.py and common/models.py) with args.start_epoch ==
args.epochs, so the training loop body never executes -- reproducing
exactly the "--resume to an already-completed checkpoint" scenario -- and
assert main() completes without raising.
"""
import os
import tempfile
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from tinyml_tinyverse.references.timeseries_classification import train as tsc_train
from tinyml_tinyverse.references.timeseries_forecasting import train as forecast_train


class _FakeTSClassificationDataset:
    classes = ["a", "b"]
    X = np.zeros((4, 3, 4), dtype=np.float32)
    inverse_label_map = {0: "a", 1: "b"}


def _base_classification_args(**overrides):
    args = Namespace(
        quantization=True, dont_train_just_feat_ext='False', load_saved_model='None',
        nas_enabled='False', generic_model=False, nn_for_feature_extraction=False,
        output_int=True, auto_quantization=False, distributed=False, gen_golden_vectors=False,
        model='dummy', model_config=None, model_spec=None, dual_op=False,
        label_smoothing=0.0, apex=False, print_freq=10, quantization_method='QAT',
        epochs=3, start_epoch=3,  # loop runs zero iterations, like a fully-resumed checkpoint
        output_dir='/tmp/fake-output', weight_bitwidth=8, activation_bitwidth=8,
        autoquant_tolerance_classification=0.1, opset_version=17, device='cpu',
        file_level_classification_log='/tmp/fake-output/file_level.log', DEBUG=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _fake_data_loaders():
    item = (torch.tensor(0), torch.zeros(1, 3, 4), torch.tensor(0))
    return [item], [item]


def _set_dataset_load_state(dataset):
    tsc_train.dataset_load_state['dataset'] = dataset
    tsc_train.dataset_load_state['dataset_test'] = dataset
    tsc_train.dataset_load_state['train_sampler'] = None
    tsc_train.dataset_load_state['test_sampler'] = None


def _patch_common_pipeline(stack):
    """Mocks every heavy helper main() calls before/around the epoch loop,
    common to both classification tests below."""
    stack.enter_context(patch.object(
        tsc_train, "setup_training_environment",
        return_value=(tsc_train.getLogger("test"), torch.device("cpu"))))
    stack.enter_context(patch.object(tsc_train, "prepare_transforms"))
    stack.enter_context(patch.object(tsc_train, "create_data_loaders", return_value=_fake_data_loaders()))
    stack.enter_context(patch.object(tsc_train.models, "get_model", return_value=torch.nn.Identity()))
    stack.enter_context(patch.object(tsc_train, "load_pretrained_weights", side_effect=lambda model, a, l: model))
    stack.enter_context(patch.object(tsc_train, "handle_export_only", return_value=False))
    stack.enter_context(patch.object(tsc_train, "move_model_to_device"))
    stack.enter_context(patch.object(tsc_train, "compile_model_if_enabled", side_effect=lambda model, a, l, **kw: model))
    stack.enter_context(patch.object(
        tsc_train, "setup_distributed_model", side_effect=lambda model, a, d: (model, model, None)))
    stack.enter_context(patch.object(tsc_train, "setup_optimizer_and_scheduler", return_value=(MagicMock(), MagicMock())))
    stack.enter_context(patch.object(tsc_train, "resume_from_checkpoint"))
    stack.enter_context(patch.object(tsc_train, "get_amp_context", return_value=MagicMock()))
    stack.enter_context(patch.object(tsc_train, "get_grad_scaler", return_value=None))
    stack.enter_context(patch.object(tsc_train.utils, "quantization_wrapped_model", side_effect=lambda model, *a, **kw: model))
    stack.enter_context(patch.object(tsc_train.utils, "export_model"))
    stack.enter_context(patch.object(tsc_train, "log_training_time"))
    stack.enter_context(patch.object(tsc_train, "shutdown_data_loaders"))


def test_classification_train_main_survives_zero_iteration_resume():
    args = _base_classification_args()
    _set_dataset_load_state(_FakeTSClassificationDataset())

    with ExitStack() as stack:
        _patch_common_pipeline(stack)
        tsc_train.main(0, args)


def test_classification_train_logs_auc_not_f1_as_auc_roc_score(caplog):
    """Runs the loop for exactly one (improving) epoch so best['auc'] and
    best['f1'] are set to distinct values, then asserts the "AUC ROC Score"
    log line reports the auc value, not the f1 value."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = _base_classification_args(
            epochs=1, start_epoch=0, output_dir=tmp_dir,
            file_level_classification_log=os.path.join(tmp_dir, "file_level.log"),
        )
        _set_dataset_load_state(_FakeTSClassificationDataset())

        avg_conf_matrix = [[2, 0], [0, 2]]
        fake_predictions = torch.zeros(4)
        fake_ground_truth = torch.zeros(4)
        # f1 and auc are deliberately far apart so a mislabeled log line is unmistakable.
        evaluate_return = (99.0, 11.0, 77.0, avg_conf_matrix, fake_predictions, fake_ground_truth)

        with ExitStack() as stack:
            _patch_common_pipeline(stack)
            stack.enter_context(patch.object(tsc_train.utils, "train_one_epoch_classification"))
            stack.enter_context(patch.object(tsc_train.utils, "evaluate_classification", return_value=evaluate_return))
            stack.enter_context(patch.object(tsc_train, "save_checkpoint", return_value={}))
            stack.enter_context(patch.object(tsc_train.utils, "save_on_master"))
            stack.enter_context(patch.object(tsc_train.utils, "print_file_level_classification_summary"))
            with caplog.at_level("INFO"):
                tsc_train.main(0, args)

    auc_lines = [r.message for r in caplog.records if "AUC ROC Score" in r.message]
    assert auc_lines, "expected an 'AUC ROC Score' log line"
    assert "77.000" in auc_lines[0], f"expected the auc value (77.0) in: {auc_lines[0]}"
    assert "11.000" not in auc_lines[0], f"AUC ROC Score line wrongly logged the f1 value: {auc_lines[0]}"


class _FakeForecastingDataset:
    X = np.zeros((4, 3, 4), dtype=np.float32)
    Y = None
    header_row = [["temp"]]


def test_forecasting_train_main_survives_zero_iteration_resume():
    """timeseries_forecasting/train.py crashed with TypeError (None subscript)
    rather than KeyError, since best_epoch_values pre-populates 'true_values'
    and 'predictions' with None instead of omitting the keys."""
    args = Namespace(
        quantization=True, dont_train_just_feat_ext='False', gen_golden_vectors=False,
        model='dummy', model_config=None, model_spec=None, dual_op=False,
        forecast_horizon=2, auto_quantization=False, distributed=False,
        weight_bitwidth=8, activation_bitwidth=8, autoquant_tolerance_forecasting=0.1,
        epochs=3, start_epoch=3,  # loop runs zero iterations, like a fully-resumed checkpoint
        output_dir='/tmp/fake-output', opset_version=17, device='cpu',
        quantization_method='QAT', output_int=None, generic_model=False,
    )
    dataset = _FakeForecastingDataset()
    forecast_train.dataset_load_state['dataset'] = dataset
    forecast_train.dataset_load_state['dataset_test'] = dataset
    forecast_train.dataset_load_state['train_sampler'] = None
    forecast_train.dataset_load_state['test_sampler'] = None

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            forecast_train, "setup_training_environment",
            return_value=(forecast_train.getLogger("test"), torch.device("cpu"))))
        stack.enter_context(patch.object(forecast_train, "prepare_transforms"))
        stack.enter_context(patch.object(forecast_train, "generate_golden_vector_dir"))
        stack.enter_context(patch.object(forecast_train, "create_data_loaders", return_value=_fake_data_loaders()))
        stack.enter_context(patch.object(forecast_train.models, "get_model", return_value=torch.nn.Identity()))
        stack.enter_context(patch.object(forecast_train, "log_model_summary"))
        stack.enter_context(patch.object(forecast_train, "load_pretrained_weights", side_effect=lambda model, a, l: model))
        stack.enter_context(patch.object(forecast_train, "handle_export_only", return_value=False))
        stack.enter_context(patch.object(forecast_train, "move_model_to_device"))
        stack.enter_context(patch.object(forecast_train, "compile_model_if_enabled", side_effect=lambda model, a, l, **kw: model))
        stack.enter_context(patch.object(forecast_train.utils, "quantization_wrapped_model", side_effect=lambda model, *a, **kw: model))
        stack.enter_context(patch.object(forecast_train, "setup_optimizer_and_scheduler", return_value=(MagicMock(), MagicMock())))
        stack.enter_context(patch.object(
            forecast_train, "setup_distributed_model", side_effect=lambda model, a, d: (model, model, None)))
        stack.enter_context(patch.object(forecast_train, "resume_from_checkpoint"))
        stack.enter_context(patch.object(forecast_train, "export_trained_model"))
        stack.enter_context(patch.object(forecast_train, "log_training_time"))
        stack.enter_context(patch.object(forecast_train, "shutdown_data_loaders"))
        forecast_train.main(0, args)
