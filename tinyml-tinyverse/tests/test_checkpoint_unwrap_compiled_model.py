"""Regression test for: checkpoints saved from a torch.compile-wrapped model
carrying _orig_mod.-prefixed keys, which the float->quantization weight
transfer path (load_weights.py) cannot match -- it falls back to
strict=False and silently discards the entire float-trained result. No
exception is raised; the pipeline reports success while quietly retraining
from random init.

Root cause: setup_distributed_model sets model_without_ddp = model when not
using DDP, so when compile_model_if_enabled succeeded upstream,
model_without_ddp IS the torch._dynamo.OptimizedModule wrapper. state_dict()
on it emits every key prefixed _orig_mod.
"""
import torch
import torch.nn as nn

from tinyml_tinyverse.references.common.train_base import save_checkpoint, resume_from_checkpoint


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class _FakeOptimizer:
    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass


class _FakeScheduler:
    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass


class _FakeArgs:
    def __init__(self, resume):
        self.resume = resume


def test_save_checkpoint_strips_orig_mod_prefix_from_compiled_model():
    model = _TinyModel()
    compiled_model = torch.compile(model, backend='aot_eager')
    compiled_model(torch.rand(1, 4))  # trigger real compilation

    checkpoint = save_checkpoint(
        compiled_model, _FakeOptimizer(), _FakeScheduler(), epoch=0, args=_FakeArgs(resume=None),
    )
    keys = list(checkpoint['model'].keys())
    assert keys, "checkpoint has no keys at all"
    assert not any(k.startswith('_orig_mod.') for k in keys), keys


def test_save_checkpoint_uncompiled_model_unaffected():
    """Backward compatibility: an ordinary, uncompiled model's checkpoint
    keys are unchanged (no _orig_mod. prefix ever existed to strip)."""
    model = _TinyModel()
    checkpoint = save_checkpoint(
        model, _FakeOptimizer(), _FakeScheduler(), epoch=0, args=_FakeArgs(resume=None),
    )
    assert set(checkpoint['model'].keys()) == set(model.state_dict().keys())


def test_checkpoint_round_trips_into_a_fresh_uncompiled_model():
    """The actual failure mode: save from a compiled model, load into the
    (uncompiled) model used for the next training phase, and confirm the
    real trained weights -- not random-init defaults -- are what land."""
    source = _TinyModel()
    with torch.no_grad():
        source.linear.weight.fill_(3.14)
    compiled_source = torch.compile(source, backend='aot_eager')
    compiled_source(torch.rand(1, 4))

    checkpoint = save_checkpoint(
        compiled_source, _FakeOptimizer(), _FakeScheduler(), epoch=0, args=_FakeArgs(resume=None),
    )

    target = _TinyModel()  # fresh, randomly initialized, NOT compiled
    assert not torch.allclose(target.linear.weight, torch.full_like(target.linear.weight, 3.14))
    target.load_state_dict(checkpoint['model'], strict=True)  # must not need strict=False
    assert torch.allclose(target.linear.weight, torch.full_like(target.linear.weight, 3.14))


def test_resume_from_checkpoint_symmetric_with_compiled_model():
    """resume_from_checkpoint (the --resume path) must be able to load a
    checkpoint saved by save_checkpoint back into a still-compiled model,
    using the same unwrap on both sides."""
    import tempfile
    import os

    source = _TinyModel()
    with torch.no_grad():
        source.linear.weight.fill_(2.71)
    compiled_source = torch.compile(source, backend='aot_eager')
    compiled_source(torch.rand(1, 4))

    checkpoint = save_checkpoint(
        compiled_source, _FakeOptimizer(), _FakeScheduler(), epoch=5, args=_FakeArgs(resume=None),
    )

    fresh = _TinyModel()
    compiled_fresh = torch.compile(fresh, backend='aot_eager')
    compiled_fresh(torch.rand(1, 4))

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint.pth')
        torch.save(checkpoint, ckpt_path)
        args = _FakeArgs(resume=ckpt_path)
        args.device = 'cpu'
        resume_from_checkpoint(compiled_fresh, _FakeOptimizer(), _FakeScheduler(), None, args)

    assert torch.allclose(fresh.linear.weight, torch.full_like(fresh.linear.weight, 2.71))
