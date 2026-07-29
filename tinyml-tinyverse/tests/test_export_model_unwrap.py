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

from tinyml_tinyverse.common.utils.utils import export_model


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


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
