"""Regression test: radar_classification.train.main() must not crash when
args.quantization is set and dataset_load_state's cache is empty (the
run_quant_train_only config: a standalone quantized run with no preceding
float-training call in the same process to populate the cache)."""
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


def test_main_does_not_crash_when_quantization_cache_is_empty():
    radar_train.dataset_load_state['dataset'] = None
    radar_train.dataset_load_state['dataset_test'] = None
    radar_train.dataset_load_state['train_sampler'] = None
    radar_train.dataset_load_state['test_sampler'] = None

    args = Namespace(
        quantization=True, data_path='/fake', output_dir='/tmp/fake-radar-quant-only',
        gof_test=False, frame_size='None', dont_train_just_feat_ext='False',
        load_saved_model='None', nas_enabled='False', generic_model=True,
        model='LINEAR_4L_PC', model_config=None, model_spec=None, dual_op=False,
        output_int=True, quantization_method='QAT', weight_bitwidth=8,
        activation_bitwidth=8, epochs=1, start_epoch=0, label_smoothing=0.0,
        distributed=False, apex=False, print_freq=10, opset_version=17,
        gen_golden_vectors=False, DEBUG=False,
        file_level_classification_log='/tmp/fake-radar-quant-only/file_level.log',
    )

    fake_dataset = _FakeDataset()
    # Each "batch" must be subscriptable like (raw, features, target) -- main()'s
    # export step does `next(iter(data_loader_test))[1]` to get an example input.
    fake_batch = (torch.zeros((1, 8)), torch.zeros((1, 8)), torch.zeros((1,), dtype=torch.long))
    fake_loaders = ([fake_batch], [fake_batch])

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            radar_train, "setup_training_environment",
            return_value=(radar_train.getLogger("test"), torch.device("cpu"))))
        stack.enter_context(patch.object(radar_train, "prepare_transforms"))
        stack.enter_context(patch.object(
            radar_train, "load_datasets",
            return_value=(fake_dataset, fake_dataset, None, None)))
        stack.enter_context(patch.object(radar_train, "create_data_loaders", return_value=fake_loaders))
        stack.enter_context(patch.object(radar_train.models, "get_model", return_value=torch.nn.Linear(8, 2)))
        stack.enter_context(patch.object(radar_train, "log_model_summary"))
        stack.enter_context(patch.object(radar_train, "load_pretrained_weights", side_effect=lambda m, a, l: m))
        stack.enter_context(patch.object(radar_train, "handle_export_only", return_value=False))
        stack.enter_context(patch.object(radar_train, "move_model_to_device"))
        stack.enter_context(patch.object(radar_train, "compile_model_if_enabled", side_effect=lambda m, a, l, **kw: m))
        stack.enter_context(patch.object(
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

        # Should not raise. Pre-fix: AttributeError on dataset_load_state['dataset'] is None.
        radar_train.main(0, args)
