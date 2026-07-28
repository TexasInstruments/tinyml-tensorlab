from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import pytest


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class _FakeArgs:
    def __init__(self, compile_model):
        self.compile_model = compile_model


def _get_logger():
    import logging
    return logging.getLogger("test_compile_warmup_fallback")


def test_compile_disabled_returns_original_model():
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=0)
    result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    assert result is model


def test_compile_success_with_warmup_returns_compiled_model():
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1)
    result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    # torch.compile wraps in an OptimizedModule; on CPU with a trivial model
    # this should succeed genuinely (no mocking needed — real compile on a
    # tiny CPU model is fast and should not fail).
    assert result is not None
    out = result(torch.rand(1, 4))
    assert out.shape == (1, 2)


def test_warmup_failure_falls_back_to_original_model():
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1)

    class _BrokenCompiledModel(nn.Module):
        def forward(self, x):
            raise RuntimeError("simulated Inductor/Triton compile failure")

    with patch('torch.compile', return_value=_BrokenCompiledModel()):
        result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))

    # Must fall back to the ORIGINAL model, not the broken compiled one.
    assert result is model
    out = result(torch.rand(1, 4))
    assert out.shape == (1, 2)


def test_wrap_time_failure_falls_back_to_original_model():
    """torch.compile() itself raising (not just the warmup forward) is still caught."""
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1)

    with patch('torch.compile', side_effect=RuntimeError("simulated wrap-time failure")):
        result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))

    assert result is model


def test_no_input_shape_skips_warmup_but_still_compiles():
    """Backward compatibility: callers that don't pass input_shape get the
    old behavior (compile attempted, no warmup, no new fallback coverage)."""
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1)
    result = compile_model_if_enabled(model, args, _get_logger())  # no input_shape
    assert result is not None


def test_warmup_restores_original_training_mode():
    """The warmup pass must not leave the model stuck in eval mode."""
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    model.train()
    args = _FakeArgs(compile_model=1)
    result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    assert result.training is True
