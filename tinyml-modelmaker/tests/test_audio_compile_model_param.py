"""Regression test: audio's modelmaker params must carry a compile_model
field, and apply_hardware_defaults must be invoked so it can auto-enable
compile on CUDA -- matching the pattern timeseries already has. Without
this, --compile-model is unreachable from the modelmaker (product) path
even though tinyml-tinyverse's audio_classification.train.main() already
supports it."""
from unittest.mock import patch

from tinyml_modelmaker.ai_modules.audio.params import init_params
from tinyml_modelmaker.ai_modules.audio.training.tinyml_tinyverse.audio_base import BaseAudioModelTraining


def test_init_params_carries_compile_model_field():
    # apply_hardware_defaults auto-raises compile_model to 1 when CUDA is
    # available -- mock it False so this test asserts the static default,
    # not whatever value the CI/dev machine's hardware happens to produce.
    with patch('torch.cuda.is_available', return_value=False):
        params = init_params()
    assert hasattr(params.training, "compile_model"), (
        "audio's params.training has no compile_model field -- "
        "apply_hardware_defaults can't act on it (it's hasattr-guarded), "
        "and the field never reaches the --compile-model argv flag."
    )
    assert params.training.compile_model == 0, (
        "compile_model must default to 0 (matching timeseries) -- "
        "apply_hardware_defaults is what conditionally raises it, not the static default."
    )


def test_build_common_train_argv_includes_compile_model():
    """Regression test: audio_base.py's train argv must include --compile-model,
    sourced from params.training.compile_model -- otherwise the field added
    above has nowhere to go and remains dead."""
    params = init_params()
    params.training.compile_model = 1
    # dataset_path/data_dir default to None; _build_common_train_argv joins
    # them with os.path.join (unlike other fields, not wrapped in an
    # f-string), so they must be populated here the way the real pipeline
    # populates them after dataset preparation, before this test can reach
    # the --compile-model wiring under test.
    params.dataset.dataset_path = "/tmp/fake_dataset"
    params.dataset.data_dir = "data"

    class _Dummy(BaseAudioModelTraining):
        train_module = None
        test_module = None

    instance = object.__new__(_Dummy)
    instance.params = params

    argv = instance._build_common_train_argv(device="cpu", distributed=0)
    assert "--compile-model" in argv, "argv builder never emits --compile-model"
    idx = argv.index("--compile-model")
    assert argv[idx + 1] == "1", (
        f"--compile-model should carry params.training.compile_model's value (1), got {argv[idx + 1]!r}"
    )
