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
from argparse import Namespace

import torch
import torch.nn as nn

from tinyml_tinyverse.references.common.train_base import save_checkpoint, resume_from_checkpoint
from tinyml_tinyverse.common.utils.utils import ExponentialMovingAverage
from tinyml_tinyverse.common.utils.load_weights import load_weights


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


class _NotOnTheSafeGlobalsAllowlist:
    """Standin for an attacker-controlled class with a malicious __reduce__;
    the actual payload doesn't matter for the pickle-rejection test below,
    only that torch.load refuses to construct instances of arbitrary,
    non-allowlisted classes. Must be module-level, not defined inside the
    test function -- pickle cannot serialize local/nested classes at all,
    which would make the test fail for an unrelated reason before it ever
    reached the weights_only check it's meant to exercise."""
    def __reduce__(self):
        return (self.__class__, ())


def _fill(model, value):
    with torch.no_grad():
        model.linear.weight.fill_(value)
        model.linear.bias.fill_(value)
    return model


def _assert_all_params_equal(model, value):
    """Check every parameter, not just .weight -- a bug that only corrupted
    .bias handling would otherwise pass undetected."""
    for name, param in model.named_parameters():
        assert torch.allclose(param, torch.full_like(param, value)), \
            f"{name} was not correctly transferred (expected all {value})"


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
    source = _fill(_TinyModel(), 3.14)
    compiled_source = torch.compile(source, backend='aot_eager')
    compiled_source(torch.rand(1, 4))

    checkpoint = save_checkpoint(
        compiled_source, _FakeOptimizer(), _FakeScheduler(), epoch=0, args=_FakeArgs(resume=None),
    )

    target = _TinyModel()  # fresh, randomly initialized, NOT compiled
    assert not torch.allclose(target.linear.weight, torch.full_like(target.linear.weight, 3.14))
    target.load_state_dict(checkpoint['model'], strict=True)  # must not need strict=False
    _assert_all_params_equal(target, 3.14)


def test_checkpoint_round_trips_through_the_real_load_weights_consumer():
    """Same scenario as test_checkpoint_round_trips_into_a_fresh_uncompiled_model,
    but through the ACTUAL production consumer of these checkpoints --
    load_weights.load_weights(), used for the float->quantization --weights
    transfer in timeseries_base.py -- rather than a raw load_state_dict call.
    This is the function whose silent strict=False fallback originally
    masked the bug (it printed a yellow warning and continued with 100% of
    weights discarded, no exception, pipeline reported success)."""
    source = _fill(_TinyModel(), 6.28)
    compiled_source = torch.compile(source, backend='aot_eager')
    compiled_source(torch.rand(1, 4))

    checkpoint = save_checkpoint(
        compiled_source, _FakeOptimizer(), _FakeScheduler(), epoch=0, args=_FakeArgs(resume=None),
    )

    target = _TinyModel()  # fresh, randomly initialized, NOT compiled
    assert not torch.allclose(target.linear.weight, torch.full_like(target.linear.weight, 6.28))
    load_weights(target, checkpoint['model'], state_dict_name=None)
    _assert_all_params_equal(target, 6.28)


def test_resume_from_checkpoint_symmetric_with_compiled_model():
    """resume_from_checkpoint (the --resume path) must be able to load a
    checkpoint saved by save_checkpoint back into a still-compiled model,
    using the same unwrap on both sides."""
    import tempfile
    import os

    source = _fill(_TinyModel(), 2.71)
    compiled_source = torch.compile(source, backend='aot_eager')
    compiled_source(torch.rand(1, 4))

    checkpoint = save_checkpoint(
        # Namespace, not _FakeArgs: this ends up as checkpoint['args'], which
        # torch.load's weights_only safety check must unpickle -- only
        # argparse.Namespace is allowlisted, matching real production usage.
        compiled_source, _FakeOptimizer(), _FakeScheduler(), epoch=5, args=Namespace(),
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

    _assert_all_params_equal(fresh, 2.71)


def test_save_checkpoint_strips_orig_mod_prefix_from_compiled_ema():
    """ExponentialMovingAverage (AveragedModel) deep-copies its source model
    into self.module -- so when the source was already compiled, the
    OptimizedModule wrapper ends up nested at model_ema.module._orig_mod,
    not at model_ema._orig_mod itself. A top-level unwrap can't reach it."""
    model = _TinyModel()
    compiled_model = torch.compile(model, backend='aot_eager')
    compiled_model(torch.rand(1, 4))
    model_ema = ExponentialMovingAverage(compiled_model, decay=0.99)

    checkpoint = save_checkpoint(
        compiled_model, _FakeOptimizer(), _FakeScheduler(), epoch=0,
        args=_FakeArgs(resume=None), model_ema=model_ema,
    )
    keys = list(checkpoint['model_ema'].keys())
    assert keys, "ema checkpoint has no keys at all"
    assert not any('_orig_mod' in k for k in keys), keys


def test_resume_from_checkpoint_symmetric_with_compiled_ema():
    """The EMA analogue of test_resume_from_checkpoint_symmetric_with_compiled_model:
    a checkpoint saved from a compiled model+EMA must load back into a fresh
    compiled model+EMA, restoring the real EMA weight values."""
    import tempfile
    import os

    source = _TinyModel()
    with torch.no_grad():
        source.linear.weight.fill_(1.5)
    compiled_source = torch.compile(source, backend='aot_eager')
    compiled_source(torch.rand(1, 4))
    source_ema = ExponentialMovingAverage(compiled_source, decay=0.99)
    with torch.no_grad():
        for p in source_ema.module.parameters():
            p.fill_(1.5)

    checkpoint = save_checkpoint(
        # Namespace, not _FakeArgs: see the identical note in
        # test_resume_from_checkpoint_symmetric_with_compiled_model.
        compiled_source, _FakeOptimizer(), _FakeScheduler(), epoch=3,
        args=Namespace(), model_ema=source_ema,
    )

    fresh = _TinyModel()
    compiled_fresh = torch.compile(fresh, backend='aot_eager')
    compiled_fresh(torch.rand(1, 4))
    fresh_ema = ExponentialMovingAverage(compiled_fresh, decay=0.99)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint.pth')
        torch.save(checkpoint, ckpt_path)
        args = _FakeArgs(resume=ckpt_path)
        args.device = 'cpu'
        resume_from_checkpoint(compiled_fresh, _FakeOptimizer(), _FakeScheduler(), fresh_ema, args)

    for name, param in fresh_ema.module.named_parameters():
        assert torch.allclose(param, torch.full_like(param, 1.5)), \
            f"{name} was not correctly transferred (expected all 1.5)"


def test_resume_from_checkpoint_ema_compiled_save_uncompiled_load():
    """The actual production shape: EMA saved from a compiled source (keys
    stripped at save time), then resumed into a run where EMA is NOT
    compiled -- e.g. a later quantization phase that (per the separate
    skip-compile-under-quantization fix) never compiles at all. The old
    load-side code's top-level getattr couldn't reach EMA's nested wrapper,
    so this asymmetric direction is the real discriminating case."""
    import tempfile
    import os

    source = _TinyModel()
    with torch.no_grad():
        source.linear.weight.fill_(4.2)
    compiled_source = torch.compile(source, backend='aot_eager')
    compiled_source(torch.rand(1, 4))
    source_ema = ExponentialMovingAverage(compiled_source, decay=0.99)
    with torch.no_grad():
        for p in source_ema.module.parameters():
            p.fill_(4.2)

    checkpoint = save_checkpoint(
        # Namespace, not _FakeArgs: see the identical note in
        # test_resume_from_checkpoint_symmetric_with_compiled_model.
        compiled_source, _FakeOptimizer(), _FakeScheduler(), epoch=3,
        args=Namespace(), model_ema=source_ema,
    )

    fresh = _TinyModel()  # NOT compiled this time
    fresh_ema = ExponentialMovingAverage(fresh, decay=0.99)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint.pth')
        torch.save(checkpoint, ckpt_path)
        args = _FakeArgs(resume=ckpt_path)
        args.device = 'cpu'
        resume_from_checkpoint(fresh, _FakeOptimizer(), _FakeScheduler(), fresh_ema, args)

    for name, param in fresh_ema.module.named_parameters():
        assert torch.allclose(param, torch.full_like(param, 4.2)), \
            f"{name} was not correctly transferred (expected all 4.2)"


def test_resume_from_checkpoint_loads_old_format_checkpoint_with_orig_mod_keys():
    """Backward compatibility: a checkpoint written before this fix existed
    (or by any other torch.compile-using caller) has raw _orig_mod.-prefixed
    keys in checkpoint['model'] -- resume_from_checkpoint must still load it
    into a plain, uncompiled model, not raise a strict key-mismatch error."""
    import tempfile
    import os

    old_style_checkpoint = {
        'model': {'_orig_mod.linear.weight': torch.full((2, 4), 8.5),
                   '_orig_mod.linear.bias': torch.full((2,), 8.5)},
        'optimizer': {},
        'lr_scheduler': {},
        'epoch': 7,
    }

    target = _TinyModel()  # NOT compiled -- the real shape once a compile-hardening fix skips compile for this phase

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint.pth')
        torch.save(old_style_checkpoint, ckpt_path)
        args = _FakeArgs(resume=ckpt_path)
        args.device = 'cpu'
        resume_from_checkpoint(target, _FakeOptimizer(), _FakeScheduler(), None, args)

    assert torch.allclose(target.linear.weight, torch.full_like(target.linear.weight, 8.5))
    assert torch.allclose(target.linear.bias, torch.full_like(target.linear.bias, 8.5))
    assert args.start_epoch == 8


def test_resume_from_checkpoint_rejects_untrusted_pickle_payload():
    """Security regression guard: resume_from_checkpoint must NOT accept
    arbitrary pickled objects wholesale (i.e. must not silently be
    weights_only=False in spirit). Only the one non-tensor type the
    checkpoint legitimately needs (argparse.Namespace, for checkpoint['args'])
    is allowlisted -- anything else in the pickle stream must still be
    rejected by torch's weights_only safety check."""
    import tempfile
    import os

    malicious_checkpoint = {
        'model': _TinyModel().state_dict(),
        'optimizer': {},
        'lr_scheduler': {},
        'epoch': 0,
        'payload': _NotOnTheSafeGlobalsAllowlist(),
    }

    target = _TinyModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint.pth')
        torch.save(malicious_checkpoint, ckpt_path)
        args = _FakeArgs(resume=ckpt_path)
        args.device = 'cpu'
        try:
            resume_from_checkpoint(target, _FakeOptimizer(), _FakeScheduler(), None, args)
            raised = False
        except Exception:
            raised = True

    assert raised, (
        "resume_from_checkpoint accepted a pickle payload containing a "
        "non-allowlisted class -- the weights_only safety check is not "
        "actually restricting unpickling."
    )
