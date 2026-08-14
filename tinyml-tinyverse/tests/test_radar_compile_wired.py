"""Regression test: radar_classification.train.main() must call
compile_model_if_enabled, matching the pattern already used in the 4
timeseries_* reference scripts (PR #22). Without this, --compile-model is
silently a no-op for radar regardless of what the caller requests."""
import inspect

from tinyml_tinyverse.references.radar_classification import train as radar_train


def test_main_calls_compile_model_if_enabled():
    main_source = inspect.getsource(radar_train.main)
    assert "compile_model_if_enabled(" in main_source, (
        "main() does not call compile_model_if_enabled -- torch.compile/AMP "
        "hardware acceleration would silently not apply to radar training."
    )
