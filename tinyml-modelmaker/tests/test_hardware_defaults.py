from unittest.mock import patch
import pytest


def _make_params():
    """Build a minimal ConfigDict with the two training flags at their defaults."""
    from tinyml_modelmaker.utils.config_dict import ConfigDict
    return ConfigDict(dict(training=dict(compile_model=0, native_amp=False)))


def test_auto_enables_both_on_cuda_with_no_explicit_keys():
    from tinyml_modelmaker.utils.hardware_defaults import apply_hardware_defaults
    params = _make_params()
    with patch('torch.cuda.is_available', return_value=True):
        apply_hardware_defaults(params, set())
    assert params.training.compile_model == 1
    assert params.training.native_amp is True


def test_respects_explicit_native_amp_false():
    from tinyml_modelmaker.utils.hardware_defaults import apply_hardware_defaults
    params = _make_params()
    with patch('torch.cuda.is_available', return_value=True):
        apply_hardware_defaults(params, {'native_amp'})
    assert params.training.compile_model == 1   # still auto-enabled
    assert params.training.native_amp is False  # explicit choice respected


def test_respects_explicit_compile_model_zero():
    from tinyml_modelmaker.utils.hardware_defaults import apply_hardware_defaults
    params = _make_params()
    with patch('torch.cuda.is_available', return_value=True):
        apply_hardware_defaults(params, {'compile_model'})
    assert params.training.compile_model == 0   # explicit choice respected
    assert params.training.native_amp is True   # still auto-enabled


def test_no_change_without_cuda():
    from tinyml_modelmaker.utils.hardware_defaults import apply_hardware_defaults
    params = _make_params()
    with patch('torch.cuda.is_available', return_value=False):
        apply_hardware_defaults(params, set())
    assert params.training.compile_model == 0
    assert params.training.native_amp is False


def test_safe_when_params_lacks_flags():
    """hasattr guards: calling on a params dict without compile_model/native_amp must not raise."""
    from tinyml_modelmaker.utils.config_dict import ConfigDict
    from tinyml_modelmaker.utils.hardware_defaults import apply_hardware_defaults
    params = ConfigDict(dict(training=dict(batch_size=32)))
    with patch('torch.cuda.is_available', return_value=True):
        apply_hardware_defaults(params, set())  # must not raise
    assert params.training.batch_size == 32


def test_logs_when_it_actually_changes_a_value(caplog):
    """F-3: the policy must announce itself when it fires, so an effective
    config that differs from the requested one isn't silently unexplained."""
    from tinyml_modelmaker.utils.hardware_defaults import apply_hardware_defaults
    params = _make_params()
    with patch('torch.cuda.is_available', return_value=True):
        with caplog.at_level('INFO'):
            apply_hardware_defaults(params, set())
    assert 'compile_model' in caplog.text
    assert 'native_amp' in caplog.text


def test_does_not_log_when_nothing_changes(caplog):
    """No log noise when CUDA is unavailable, or the user already set both
    flags -- only log on an actual mutation."""
    from tinyml_modelmaker.utils.hardware_defaults import apply_hardware_defaults

    params = _make_params()
    with patch('torch.cuda.is_available', return_value=False):
        with caplog.at_level('INFO'):
            apply_hardware_defaults(params, set())
    assert caplog.text == ''

    params2 = _make_params()
    with patch('torch.cuda.is_available', return_value=True):
        with caplog.at_level('INFO'):
            apply_hardware_defaults(params2, {'compile_model', 'native_amp'})
    assert caplog.text == ''
