"""Regression test for: export_model() crashing on a torch.compile-wrapped
model with `RuntimeError: Detected that you are using FX to torch.jit.trace
a dynamo-optimized function. This is not supported at the moment.`

Root cause: torch.compile() wraps a model in torch._dynamo.OptimizedModule,
which exposes the original module at ._orig_mod. Neither torch.onnx.export
nor torch.jit.trace (both used inside export_model, depending on whether
quantization is enabled) can trace a dynamo-optimized module directly.
"""
import os
import tempfile

import torch
import torch.nn as nn

from tinyml_tinyverse.common.utils.utils import export_model, unwrap_compiled_submodules


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class _WrapperModel(nn.Module):
    """Mirrors NeuralNetworkWithPreprocess: a compiled submodule nested one
    level below the top-level model, not at the top level itself -- the
    exact shape that broke a single-level `getattr(model, '_orig_mod', ...)`
    unwrap in real timeseries_classification runs."""
    def __init__(self, inner):
        super().__init__()
        self.model = inner

    def forward(self, x):
        return self.model(x)


def test_export_model_handles_compiled_model_float_path():
    """The non-quantized (else) branch: torch.onnx.export must not choke on
    a compiled model."""
    model = _TinyModel()
    compiled_model = torch.compile(model, backend='aot_eager')
    # Trigger real compilation so we're testing an actual OptimizedModule,
    # not just the lazy uncalled wrapper.
    compiled_model(torch.rand(1, 4))

    with tempfile.TemporaryDirectory() as tmpdir:
        export_model(compiled_model, input_shape=(1, 4), output_dir=tmpdir, quantization=0)
        assert os.path.exists(os.path.join(tmpdir, 'model.onnx'))


def test_export_model_uncompiled_model_still_works():
    """Backward compatibility: an ordinary, uncompiled model must still
    export exactly as before (the getattr fallback is a no-op)."""
    model = _TinyModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        export_model(model, input_shape=(1, 4), output_dir=tmpdir, quantization=0)
        assert os.path.exists(os.path.join(tmpdir, 'model.onnx'))


def test_export_model_handles_nested_compiled_submodule():
    """Reproduces the real timeseries_classification failure: the compiled
    model isn't the top-level module -- it's nested one level inside a
    wrapper (NeuralNetworkWithPreprocess.model), because compile happens
    before that wrapping is applied. A single-level unwrap misses this."""
    inner = _TinyModel()
    compiled_inner = torch.compile(inner, backend='aot_eager')
    compiled_inner(torch.rand(1, 4))
    wrapped = _WrapperModel(compiled_inner)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_model(wrapped, input_shape=(1, 4), output_dir=tmpdir, quantization=0)
        assert os.path.exists(os.path.join(tmpdir, 'model.onnx'))


def test_unwrap_compiled_submodules_top_level():
    model = _TinyModel()
    compiled = torch.compile(model, backend='aot_eager')
    compiled(torch.rand(1, 4))
    result = unwrap_compiled_submodules(compiled)
    assert result is model


def test_unwrap_compiled_submodules_nested():
    inner = _TinyModel()
    compiled_inner = torch.compile(inner, backend='aot_eager')
    compiled_inner(torch.rand(1, 4))
    wrapped = _WrapperModel(compiled_inner)
    result = unwrap_compiled_submodules(wrapped)
    assert result is wrapped
    assert result.model is inner


def test_unwrap_compiled_submodules_noop_when_nothing_compiled():
    model = _TinyModel()
    result = unwrap_compiled_submodules(model)
    assert result is model
