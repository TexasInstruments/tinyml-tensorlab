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
