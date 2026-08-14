from unittest.mock import patch
import pytest
import yaml


def test_init_params_auto_enables_on_cuda():
    """init_params with no training overrides auto-enables both flags on CUDA."""
    from tinyml_modelmaker.ai_modules.timeseries.params import init_params
    user_config = dict(common=dict(task_category='timeseries_classification'))
    with patch('torch.cuda.is_available', return_value=True):
        params = init_params(user_config)
    assert params.training.compile_model == 1
    assert params.training.native_amp is True


def test_init_params_respects_native_amp_false_override():
    """Explicit native_amp: false in user config is not overridden.

    Mirrors the real production call pattern in run_tinyml_modelmaker.py:
    ModelRunner.init_params() is called with zero arguments (so hardware
    defaults auto-enable via apply_hardware_defaults with an empty
    explicitly_set), and the user's config is merged in afterward via
    ConfigDict.update() -- a deep merge -- not passed into the ConfigDict
    constructor. (The constructor's *args merge path has a separate,
    pre-existing shallow-merge bug that this test must avoid triggering.)
    """
    from tinyml_modelmaker.ai_modules.timeseries.params import init_params
    user_config = dict(
        common=dict(task_category='timeseries_classification'),
        training=dict(native_amp=False),
    )
    with patch('torch.cuda.is_available', return_value=True):
        params = init_params()
    params.update(user_config)
    assert params.training.native_amp is False
    assert params.training.compile_model == 1  # still auto-enabled


def test_init_params_no_change_without_cuda():
    """Without CUDA, both flags stay at defaults."""
    from tinyml_modelmaker.ai_modules.timeseries.params import init_params
    user_config = dict(common=dict(task_category='timeseries_classification'))
    with patch('torch.cuda.is_available', return_value=False):
        params = init_params(user_config)
    assert params.training.compile_model == 0
    assert params.training.native_amp is False


def test_init_params_respects_explicit_settings_from_yaml_path(tmp_path):
    """Regression test for a bug found by an independent analysis of this
    function (mmcli's ANALYSIS-cuda-auto-defaults.md, finding F-1) and
    reproduced here: init_params's explicitly-set-key detection used to
    check `isinstance(args[0], dict)`, so a YAML file passed by *path*
    (not a dict) always looked like "the user set nothing" -- silently
    overriding an explicit compile_model: 0 / native_amp: false on any
    CUDA machine. Also proves the deeper bug (F-1a): passing a path as a
    positional arg to init_params used to silently discard the whole file
    -- ConfigDict's *args merge loop only merged dict/ConfigDict values,
    so a string arg (a YAML path) was dropped entirely, not just
    misclassified as non-explicit. model_name here is the canary: if the
    file were still being silently discarded, it would come back None
    (the hardcoded default) instead of the value actually written below."""
    from tinyml_modelmaker.ai_modules.timeseries.params import init_params
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "training": {
            "model_name": "my_explicit_model",
            "compile_model": 0,
            "native_amp": False,
        },
    }))
    with patch('torch.cuda.is_available', return_value=True):
        params = init_params(str(config_file))
    assert params.training.model_name == "my_explicit_model", (
        "the YAML's own content was lost -- init_params(path) must load "
        "the file, not just avoid crashing on it"
    )
    assert params.training.compile_model == 0, (
        "explicit compile_model: 0 from a YAML-path config was silently overridden"
    )
    assert params.training.native_amp is False, (
        "explicit native_amp: false from a YAML-path config was silently overridden"
    )


def test_init_params_still_auto_enables_for_yaml_path_with_no_opinion(tmp_path):
    """The fix for the above must not make YAML-path configs opt out of
    the auto-enable policy entirely -- a YAML that doesn't mention
    compile_model/native_amp at all should still get them auto-enabled on
    CUDA, exactly like an equivalent dict config already does."""
    from tinyml_modelmaker.ai_modules.timeseries.params import init_params
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"training": {"model_name": "some_model"}}))
    with patch('torch.cuda.is_available', return_value=True):
        params = init_params(str(config_file))
    assert params.training.compile_model == 1
    assert params.training.native_amp is True


def test_init_params_raises_on_nonexistent_yaml_path():
    """A genuinely bad path must fail loudly, not silently no-op -- the
    same way ConfigDict(bad_path) already behaves when the path is passed
    as the constructor's `input` instead of as a later positional arg."""
    from tinyml_modelmaker.ai_modules.timeseries.params import init_params
    with patch('torch.cuda.is_available', return_value=True):
        with pytest.raises(FileNotFoundError):
            init_params('/nonexistent/path/to/config.yaml')


def test_init_params_does_not_crash_on_none_first_arg():
    """None is a documented valid first positional argument (an empty
    config) and must still be handled without error."""
    from tinyml_modelmaker.ai_modules.timeseries.params import init_params
    with patch('torch.cuda.is_available', return_value=True):
        params = init_params(None)
    assert params.training.compile_model == 1
    assert params.training.native_amp is True
