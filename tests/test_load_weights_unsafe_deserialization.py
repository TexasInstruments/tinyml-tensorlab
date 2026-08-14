"""Regression tests for load_weights()'s torch.load() deserialization safety.

load_weights() previously called torch.load(..., weights_only=False), fully disabling
PyTorch's deserialization safety check on a file that can be fetched from a URL with
no integrity check (data_utils.download_url). An untrusted/spoofed "pretrained"
checkpoint would get arbitrary code execution during unpickling.
"""
import argparse
import os
import pickle
import tempfile

import pytest
import torch
import torch.nn as nn

from tinyml_tinyverse.common.utils.load_weights import load_weights


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


class _NotOnTheSafeGlobalsAllowlist:
    """Stand-in for an arbitrary class an attacker's payload might reference.

    Not a real exploit gadget -- the point is only that torch.load() must refuse to
    unpickle a type it doesn't know is safe, not that this specific class is harmful.
    """

    def __reduce__(self):
        # If this ever actually executes during unpickling, the safety check failed.
        return (print, ("UNSAFE DESERIALIZATION EXECUTED",))


def test_load_weights_loads_a_legitimate_state_dict():
    model = _TinyModel()
    source = _TinyModel()
    with tempfile.TemporaryDirectory() as tmp_dir:
        weights_path = os.path.join(tmp_dir, "weights.pth")
        torch.save(source.state_dict(), weights_path)

        load_weights(model, weights_path)

    for p1, p2 in zip(model.parameters(), source.parameters()):
        assert torch.equal(p1, p2)


def test_load_weights_loads_a_checkpoint_with_namespace_under_args():
    """Backward-compat: this project's own save_checkpoint() (train_base.py) writes
    checkpoint['args'] as an argparse.Namespace. Such a checkpoint is a plausible
    'pretrained' input to load_weights() too, and must still load -- not just a raw
    state_dict file."""
    model = _TinyModel()
    source = _TinyModel()
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = os.path.join(tmp_dir, "checkpoint.pth")
        torch.save(
            {"state_dict": source.state_dict(), "args": argparse.Namespace(epochs=10)},
            checkpoint_path,
        )

        load_weights(model, checkpoint_path)

    for p1, p2 in zip(model.parameters(), source.parameters()):
        assert torch.equal(p1, p2)


def test_load_weights_rejects_untrusted_pickle_payload(capsys):
    """With weights_only=False (the pre-fix behavior), unpickling this payload
    would *succeed* -- the class is importable from this test module -- and only
    fail later for an unrelated reason (treating the reconstructed object as a
    dict), which a bare `pytest.raises(Exception)` can't tell apart from a real
    security rejection. Assert on the actual security property instead: the
    reduce-callable (standing in for arbitrary code) must never run, and the
    failure must be attributable to the deserialization safety check itself."""
    model = _TinyModel()
    with tempfile.TemporaryDirectory() as tmp_dir:
        payload_path = os.path.join(tmp_dir, "malicious.pth")
        torch.save({"state_dict": _NotOnTheSafeGlobalsAllowlist()}, payload_path)

        with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
            load_weights(model, payload_path)

    # The strongest possible check: the payload's code must never have run.
    assert "UNSAFE DESERIALIZATION EXECUTED" not in capsys.readouterr().out
