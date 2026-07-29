"""Regression tests for: copy.deepcopy(model) crashing at export time with
`NotImplementedError: ('{} cannot be pickled', '_MultiProcessingDataLoaderIter')`.

Root cause: when auto-quantization is enabled, TinyMLQConfigType embeds the raw
calibration/eval DataLoader objects into the qconfig dict (for its one-time
Hessian-based bitwidth search). TinyMLQuantFxBaseModule.__init__ then stores
that whole dict as self.qconfig_type, permanently pinning the DataLoaders into
the wrapped model's object graph even though the search has already consumed
them. If those loaders were built with persistent_workers=True (the default
whenever num_workers > 0) and have been iterated at least once, they carry a
live, unpicklable _MultiProcessingDataLoaderIter in their own __dict__ — which
copy.deepcopy(model) (used by export_model() before ONNX export) then trips
over.
"""
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tinyml_torchmodelopt.quantization.tinpu.quant_fx import TINPUTinyMLQATFxModule


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class _TinyDataset(Dataset):
    def __init__(self, n=8):
        self.x = torch.randn(n, 4)
        self.y = torch.randint(0, 2, (n,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _make_persistent_loader():
    # num_workers=1 + persistent_workers=True reproduces the exact DataLoader
    # configuration that made the crash reachable in practice.
    return DataLoader(_TinyDataset(), batch_size=4, num_workers=1, persistent_workers=True)


def _make_qconfig_type(calibration_loader, eval_loader):
    # Mirrors exactly what TinyMLQConfigType.__init__ produces when
    # auto_quantization=True (common.py:169-179), minus the 'auto_quantization'
    # dict flag itself — set to False here purely so quant_base.py's
    # _prepare_quantization_config() skips the real (slow, unrelated) Hessian
    # bitwidth search and goes straight to what this test actually targets:
    # whether the wrapper module retains the dataloader references afterward.
    return {
        'weight': {'bitwidth': 8, 'qscheme': torch.per_channel_symmetric, 'power2_scale': True,
                   'range_max': None, 'fixed_range': False},
        'activation': {'bitwidth': 8, 'qscheme': torch.per_tensor_symmetric, 'power2_scale': True,
                       'range_max': None, 'fixed_range': False},
        'auto_quantization': False,
        'calibration_dataloader': calibration_loader,
        'eval_dataloader': eval_loader,
    }


def test_wrapper_drops_dataloader_references_after_construction():
    calibration_loader = _make_persistent_loader()
    eval_loader = _make_persistent_loader()
    qconfig_type = _make_qconfig_type(calibration_loader, eval_loader)

    model = TINPUTinyMLQATFxModule(
        _TinyModel(), total_epochs=1, qconfig_type=qconfig_type, example_inputs=torch.randn(1, 4),
    )

    assert 'calibration_dataloader' not in model.qconfig_type
    assert 'eval_dataloader' not in model.qconfig_type


def test_callers_own_qconfig_type_dict_is_not_mutated():
    """self.qconfig_type = qconfig_type (a reference assignment) means the
    wrapper's dict and the caller's dict are the same object unless
    explicitly copied. Popping keys from self.qconfig_type must not remove
    them from the caller's own dict too -- a caller that constructs
    qconfig_type once and reuses it (e.g. across a retry, or for logging
    after construction) would otherwise see it silently emptied by a
    constructor call it doesn't own."""
    calibration_loader = _make_persistent_loader()
    eval_loader = _make_persistent_loader()
    qconfig_type = _make_qconfig_type(calibration_loader, eval_loader)

    TINPUTinyMLQATFxModule(
        _TinyModel(), total_epochs=1, qconfig_type=qconfig_type, example_inputs=torch.randn(1, 4),
    )

    assert 'calibration_dataloader' in qconfig_type
    assert 'eval_dataloader' in qconfig_type
    assert qconfig_type['calibration_dataloader'] is calibration_loader


def test_deepcopy_survives_persistent_worker_dataloaders():
    calibration_loader = _make_persistent_loader()
    eval_loader = _make_persistent_loader()
    qconfig_type = _make_qconfig_type(calibration_loader, eval_loader)

    model = TINPUTinyMLQATFxModule(
        _TinyModel(), total_epochs=1, qconfig_type=qconfig_type, example_inputs=torch.randn(1, 4),
    )

    try:
        # Simulate a completed training epoch: iterating a persistent_workers
        # DataLoader caches a live _MultiProcessingDataLoaderIter on the
        # loader itself (this is what made the bug reachable in real runs).
        for _ in calibration_loader:
            pass
        for _ in eval_loader:
            pass

        # This is what export_model() does right before ONNX export.
        copy.deepcopy(model)
    finally:
        calibration_loader._iterator = None
        eval_loader._iterator = None
