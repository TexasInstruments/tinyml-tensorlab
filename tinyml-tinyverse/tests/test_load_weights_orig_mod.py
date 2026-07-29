"""Regression tests for load_weights() handling of torch.compile's
_orig_mod. wrapper-artifact prefix, as a defense-in-depth complement to the
save-side fix in train_base.py's save_checkpoint/resume_from_checkpoint
(which now write checkpoints without the prefix in the first place).

This covers the case load_weights() is the actual consumer for: the
float->quantization --weights transfer
(timeseries_base.py -> load_weights.load_weights(..., state_dict_name='model')),
including checkpoints that might still carry _orig_mod. keys from an older,
unpatched save, or from any other compile-using caller not covered by the
train_base.py fix.
"""
import copy

import torch
import torch.nn as nn

from tinyml_tinyverse.common.utils.load_weights import load_weights


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def _filled(value):
    model = _TinyModel()
    with torch.no_grad():
        model.linear.weight.fill_(value)
        model.linear.bias.fill_(value)
    return model


def _assert_all_params_equal(model, value):
    """Check every parameter, not just .weight -- a bug that only corrupted
    .bias handling (a differently-shaped tensor, so a distinct code path
    through key matching) would otherwise pass undetected."""
    for name, param in model.named_parameters():
        assert torch.allclose(param, torch.full_like(param, value)), \
            f"{name} was not correctly transferred (expected all {value})"


def test_existing_module_prefix_alignment_still_works():
    """Backward compatibility: the pre-existing 'module.' (DDP) prefix
    realignment must be unaffected by the new _orig_mod. handling."""
    source = _filled(7.0)
    data = {f'module.{k}': v for k, v in source.state_dict().items()}

    target = _TinyModel()
    load_weights(target, data, state_dict_name=None)

    _assert_all_params_equal(target, 7.0)


def test_orig_mod_prefix_in_checkpoint_data_is_stripped():
    """A checkpoint saved by older, unpatched code (or any other
    torch.compile-using caller) with raw _orig_mod. keys must still load
    correctly into a plain, uncompiled model."""
    source = _filled(5.5)
    data = {f'_orig_mod.{k}': v for k, v in source.state_dict().items()}

    target = _TinyModel()
    load_weights(target, data, state_dict_name=None)

    _assert_all_params_equal(target, 5.5)


def test_loading_into_a_currently_compiled_model():
    """The live model being loaded INTO is itself currently torch.compile-
    wrapped (its own state_dict keys carry _orig_mod.), and the checkpoint
    data does not (the real shape produced by the already-fixed
    save_checkpoint). Data must remap onto the model's actual current key
    names, not fail to match them."""
    source = _filled(2.25)
    data = copy.deepcopy(source.state_dict())  # no prefix, as save_checkpoint now writes

    target = _TinyModel()
    compiled_target = torch.compile(target, backend='aot_eager')
    compiled_target(torch.rand(1, 4))  # trigger real compilation

    load_weights(compiled_target, data, state_dict_name=None)

    _assert_all_params_equal(target, 2.25)


def test_orig_mod_in_both_data_and_live_model():
    """Both sides compiled: checkpoint data has stale _orig_mod. keys (e.g.
    an old unpatched save) AND the live model being loaded into is itself
    currently compiled. Must still transfer correctly."""
    source = _filled(9.0)
    data = {f'_orig_mod.{k}': v for k, v in source.state_dict().items()}

    target = _TinyModel()
    compiled_target = torch.compile(target, backend='aot_eager')
    compiled_target(torch.rand(1, 4))

    load_weights(compiled_target, data, state_dict_name=None)

    _assert_all_params_equal(target, 9.0)
