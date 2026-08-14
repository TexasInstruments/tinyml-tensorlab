from unittest.mock import patch
import pytest


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


def test_init_params_does_not_crash_on_non_dict_first_arg():
    """ConfigDict documents a YAML path string (or None) as valid first
    positional input, not just a dict. init_params's own explicitly-set-key
    detection must not assume args[0] is dict-like and crash instead."""
    from tinyml_modelmaker.ai_modules.timeseries.params import init_params
    with patch('torch.cuda.is_available', return_value=True):
        # A bogus path is fine here -- ConfigDict's *args merge loop only
        # merges dict/ConfigDict values and silently ignores strings, so
        # this never actually attempts to read the file. The point of this
        # test is solely that passing a string doesn't crash init_params's
        # own explicitly-set-key detection before ConfigDict is even built.
        params = init_params('/nonexistent/path/to/config.yaml')
    assert params.training.compile_model == 1
    assert params.training.native_amp is True


def test_init_params_does_not_crash_on_none_first_arg():
    from tinyml_modelmaker.ai_modules.timeseries.params import init_params
    with patch('torch.cuda.is_available', return_value=True):
        params = init_params(None)
    assert params.training.compile_model == 1
    assert params.training.native_amp is True
