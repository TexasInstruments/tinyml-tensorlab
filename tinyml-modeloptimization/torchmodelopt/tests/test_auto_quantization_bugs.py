"""Regression tests for two bugs in Hessian-based auto-quantization:

1. run_auto_quantization() crashed with a bare TypeError (None * int) instead
   of failing gracefully when the Hessian sensitivity search couldn't run (no
   calibration_dataloader, or sample inputs/targets/criterion unavailable --
   a path callers like audio_classification/train.py already anticipate and
   try to degrade gracefully from).
2. apply_mixed_precision() mutated the caller's qconfig_dict in place, which
   is frequently the same dict object retained as self.qconfig_type on the
   live model wrapper.
"""
import torch
import torch.nn as nn
from torch.ao.quantization import QConfigMapping

from tinyml_torchmodelopt.quantization.base.fx.auto_quantization import run_auto_quantization
from tinyml_torchmodelopt.quantization.base.fx.qconfig_types import apply_mixed_precision, get_default_qconfig


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def test_run_auto_quantization_falls_back_when_sensitivities_unavailable():
    """No 'inputs'/'targets'/'criterion' in qconfig_dict -> compute_hessian_sensitivity
    returns ({}, {}) -> optimal_bitwidth stays None. Must fall back to the
    caller's already-built qconfig_mapping, not crash on None * total_params."""
    model = _TinyModel()
    fallback_mapping = QConfigMapping().set_global(get_default_qconfig())
    qconfig_dict = {"weight": {"bitwidth": 8}, "activation": {"bitwidth": 8}}

    result = run_auto_quantization(
        model, qconfig_dict, fallback_mapping,
        get_default_qconfig_fn=get_default_qconfig,
        apply_mixed_precision_fn=apply_mixed_precision,
    )

    assert result is fallback_mapping


def test_run_auto_quantization_falls_back_when_no_calibration_dataloader():
    """inputs/targets/criterion present (so sensitivities *could* compute) but
    no calibration_dataloader -- the binary search branch is also skipped, so
    optimal_bitwidth still ends up None via a different path than the test
    above. Must not crash either."""
    model = _TinyModel()
    fallback_mapping = QConfigMapping().set_global(get_default_qconfig())
    qconfig_dict = {
        "weight": {"bitwidth": 8}, "activation": {"bitwidth": 8},
        "inputs": torch.randn(4, 4), "targets": torch.zeros(4, dtype=torch.long),
        "criterion": nn.CrossEntropyLoss(),
        "calibration_dataloader": None,
    }

    result = run_auto_quantization(
        model, qconfig_dict, fallback_mapping,
        get_default_qconfig_fn=get_default_qconfig,
        apply_mixed_precision_fn=apply_mixed_precision,
    )

    assert result is fallback_mapping


def test_apply_mixed_precision_does_not_mutate_callers_qconfig_dict():
    qconfig_mapping = QConfigMapping().set_global(get_default_qconfig())
    qconfig_dict = {"weight": {"bitwidth": 8}, "activation": {"bitwidth": 8}}
    original_weight_bitwidth = qconfig_dict["weight"]["bitwidth"]

    apply_mixed_precision(qconfig_mapping, qconfig_dict, {4: ["fc"]})

    assert qconfig_dict["weight"]["bitwidth"] == original_weight_bitwidth
    assert qconfig_dict["activation"]["bitwidth"] == original_weight_bitwidth


def test_apply_mixed_precision_still_applies_the_requested_bitwidth_per_layer():
    """Regression safety: the mutation fix must not break the actual mixed-
    precision assignment -- set_module_name must still be called with a
    qconfig built at the requested bitwidth for each layer."""
    qconfig_mapping = QConfigMapping().set_global(get_default_qconfig())

    result = apply_mixed_precision(
        qconfig_mapping, {"weight": {"bitwidth": 8}, "activation": {"bitwidth": 8}}, {32: ["fc"]}
    )

    # bit_width == 32 takes the "disable quantization for this layer" path.
    assert result.module_name_qconfigs["fc"] is None
