"""Regression test: timeseries_classification.train.main() must not crash when
args.quantization is set and dataset_load_state's cache is empty (the
run_quant_train_only config: a standalone quantized run with no preceding
float-training call in the same process to populate the cache).

Identical bug and fix to radar_classification/train.py (see
tests/test_radar_quant_only_no_crash.py): pre-fix, `if args.quantization:`
always took the cache-reuse branch regardless of whether the cache had ever
been populated, so a lone quantized run crashed with
`AttributeError: 'NoneType' object has no attribute 'classes'` reading
`dataset.classes` from the never-populated `dataset_load_state['dataset']`.

Fake dataset shape and mock pattern for driving this module's real main()
are adapted from tests/test_train_best_epoch_bugs_timeseries.py, which
already exercises tsc_train.main() end-to-end (zero-iteration-loop
"--resume" scenario) with quantization=True and a *populated* cache -- this
test instead drives it with an *empty* cache, so it must fall through to a
real (here, mocked) `load_datasets` call.
"""
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from tinyml_tinyverse.references.timeseries_classification import train as tsc_train


class _FakeTSClassificationDataset:
    classes = ["a", "b"]
    X = np.zeros((4, 3, 4), dtype=np.float32)
    Y = np.zeros((4,), dtype=np.int64)
    inverse_label_map = {0: "a", 1: "b"}


def _fake_data_loaders():
    item = (torch.tensor(0), torch.zeros(1, 3, 4), torch.tensor(0))
    return [item], [item]


def test_main_does_not_crash_when_quantization_cache_is_empty():
    tsc_train.dataset_load_state['dataset'] = None
    tsc_train.dataset_load_state['dataset_test'] = None
    tsc_train.dataset_load_state['train_sampler'] = None
    tsc_train.dataset_load_state['test_sampler'] = None

    args = Namespace(
        quantization=True, data_path='/fake', output_dir='/tmp/fake-tsc-quant-only',
        gof_test=False, frame_size='None', dont_train_just_feat_ext='False',
        load_saved_model='None', nas_enabled='False', generic_model=False,
        nn_for_feature_extraction=False, model='dummy', model_config=None,
        model_spec=None, dual_op=False, label_smoothing=0.0, apex=False,
        print_freq=10, quantization_method='QAT', output_int=True,
        auto_quantization=False, distributed=False, gen_golden_vectors=False,
        epochs=3, start_epoch=3,  # loop runs zero iterations -- avoids needing to
        # mock train_one_epoch_classification/evaluate_classification/save_checkpoint,
        # matching the existing zero-iteration-resume pattern in
        # test_train_best_epoch_bugs_timeseries.py.
        weight_bitwidth=8, activation_bitwidth=8,
        autoquant_tolerance_classification=0.1, opset_version=17, device='cpu',
        file_level_classification_log='/tmp/fake-tsc-quant-only/file_level.log',
        DEBUG=False,
    )

    fake_dataset = _FakeTSClassificationDataset()
    fake_loaders = _fake_data_loaders()

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            tsc_train, "setup_training_environment",
            return_value=(tsc_train.getLogger("test"), torch.device("cpu"))))
        stack.enter_context(patch.object(tsc_train, "prepare_transforms"))
        stack.enter_context(patch.object(
            tsc_train, "load_datasets",
            return_value=(fake_dataset, fake_dataset, None, None)))
        stack.enter_context(patch.object(tsc_train.utils, "plot_feature_components_graph"))
        stack.enter_context(patch.object(tsc_train, "create_data_loaders", return_value=fake_loaders))
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

        # Should not raise. Pre-fix: AttributeError on dataset_load_state['dataset'] is None.
        tsc_train.main(0, args)
