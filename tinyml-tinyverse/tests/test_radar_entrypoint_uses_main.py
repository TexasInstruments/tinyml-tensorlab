"""Regression test: radar_classification.train.run() must dispatch to main(),
not main_debug() (a leftover notebook-parity harness that silently skips
quantization, AMP, and torch.compile for every radar training run)."""
from unittest.mock import patch

from tinyml_tinyverse.references.radar_classification import train as radar_train


def test_run_dispatches_to_main_not_main_debug():
    with patch.object(radar_train, "run_distributed") as mock_run_distributed:
        fake_args = object()
        radar_train.run(fake_args)

    mock_run_distributed.assert_called_once()
    dispatched_fn, dispatched_args = mock_run_distributed.call_args[0]

    assert dispatched_fn is radar_train.main, (
        f"run() dispatched to {dispatched_fn.__name__!r}, expected 'main'. "
        "main_debug() never applies quantization_wrapped_model, "
        "compile_model_if_enabled/apply_hardware_defaults, or "
        "resume_from_checkpoint -- wiring run() to it silently drops quantized "
        "training and hardware acceleration for every radar run."
    )
    assert dispatched_args is fake_args
