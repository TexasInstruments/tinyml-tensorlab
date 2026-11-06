"""Regression test: audio_classification.train.main() must call
compile_model_if_enabled, matching the timeseries_* pattern (PR #22)."""
import inspect

from tinyml_tinyverse.references.audio_classification import train as audio_train


def test_main_calls_compile_model_if_enabled():
    main_source = inspect.getsource(audio_train.main)
    assert "compile_model_if_enabled(" in main_source, (
        "main() does not call compile_model_if_enabled -- torch.compile/AMP "
        "hardware acceleration would silently not apply to audio classification training."
    )
