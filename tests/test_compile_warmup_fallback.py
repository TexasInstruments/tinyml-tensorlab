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
    """Regression test for: torch.compile's OptimizedModule wraps the
    original model BY REFERENCE (shares its parameters/state), so calling
    .eval() on the compiled wrapper also flips the ORIGINAL model's
    .training flag. If the warmup forward pass then raises, the original
    model must still come back with its training mode restored — not
    stuck in eval() because the restore step got skipped on the failure
    path.

    A plain mock that returns an unrelated standalone nn.Module does NOT
    reproduce this, because it doesn't share state with the original model
    the way OptimizedModule does. To actually catch a regression of this
    bug, this test wraps the original model as a genuine child submodule
    (registered via a real nn.Module attribute) so that calling
    .train()/.eval() on the wrapper recurses into and mutates the original
    model's .training flag too — exactly like OptimizedModule._orig_mod.
    """
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    model.train()  # known starting state
    args = _FakeArgs(compile_model=1)

    class _SharedStateBrokenCompiledModel(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            # Registering the original model as a real submodule means
            # nn.Module.train()/.eval() on this wrapper recurses into it,
            # mutating wrapped.training too — mirroring how
            # torch._dynamo.OptimizedModule wraps the original model by
            # reference via self._orig_mod.
            self._orig_mod = wrapped

        def forward(self, x):
            raise RuntimeError("simulated Inductor/Triton compile failure")

    broken = _SharedStateBrokenCompiledModel(model)

    with patch('torch.compile', return_value=broken):
        result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))

    # Must fall back to the ORIGINAL model, not the broken compiled one.
    assert result is model
    # Must be restored to its pre-call training state, not left in eval()
    # mode from the failed warmup pass — this is the exact defect a naive
    # mock (that doesn't share state/reference with the original model)
    # would fail to catch.
    assert model.training is True
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
