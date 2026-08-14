"""Regression test for: compile_model=1 combined with FX-based quantization
(quantization=1 or 2) crashing at prepare_qat_fx time with
`RuntimeError: Detected that you are using FX to symbolically trace a
dynamo-optimized function. This is not supported at the moment.`

Root cause: torch.compile() wraps the model in torch._dynamo.OptimizedModule
before quantization_wrapped_model() runs prepare_qat_fx on it. FX symbolic
tracing cannot trace a dynamo-optimized module at all. This is a different
incompatibility than the ONNX/TorchScript export tracing issue (fixed
separately in export_model()) -- unwrapping right before quantization would
work mechanically but would mean compile provides no benefit for the rest
of a quantized run's training, so this fix skips compiling entirely
whenever quantization is enabled.
"""
from unittest.mock import patch

import torch
import torch.nn as nn


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class _FakeArgs:
    def __init__(self, compile_model, quantization):
        self.compile_model = compile_model
        self.quantization = quantization


def _get_logger():
    import logging
    return logging.getLogger("test_compile_skipped_under_quantization")


def test_compile_skipped_when_quantization_enabled():
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1, quantization=2)
    with patch('torch.compile') as mock_compile:
        result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    # Must be the ORIGINAL model, not compiled -- an OptimizedModule here
    # would go on to crash prepare_qat_fx's FX symbolic trace.
    assert result is model
    # Proves the skip happens BEFORE compilation is attempted, not that
    # compilation happened to fail or get discarded afterward.
    mock_compile.assert_not_called()


def test_compile_skip_logs_why(caplog):
    import logging
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1, quantization=2)
    logger = _get_logger()
    with caplog.at_level(logging.INFO, logger=logger.name):
        compile_model_if_enabled(model, args, logger, input_shape=(1, 4))
    assert any('quantization' in record.message.lower() for record in caplog.records)


def test_compile_skipped_when_quantization_is_ptq_mode():
    """quantization=1 (generic PTQ/QAT) hits the same FX-trace incompatibility
    as quantization=2 (TINPU) -- both go through prepare_qat_fx."""
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1, quantization=1)
    result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    assert result is model


def test_compile_still_happens_for_float_training():
    """Zero behavior change for the common case: quantization=0 (float
    training) still compiles exactly as before."""
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1, quantization=0)
    result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    assert result is not model  # genuinely compiled
    out = result(torch.rand(1, 4))
    assert out.shape == (1, 2)


def test_compile_disabled_and_quantization_enabled_is_still_a_noop():
    """When compile_model=0, quantization doesn't matter -- nothing changes."""
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=0, quantization=2)
    result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    assert result is model
