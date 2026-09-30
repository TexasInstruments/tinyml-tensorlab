import logging

import torch

from .config_dict import ConfigDict

logger = logging.getLogger(__name__)


def explicit_training_keys(*args) -> set:
    """Union of 'training'-section keys explicitly present across all
    positional config args passed to init_params -- dicts, ConfigDicts, or
    .yaml path strings.

    Mirrors ConfigDict's own arg resolution (see
    ConfigDict.resolve_config_value) so a YAML-path config is inspected the
    same way a dict config is. Checking only `isinstance(args[0], dict)`
    made every setting in a path-based config look unset, so
    apply_hardware_defaults would silently override explicit choices (e.g.
    compile_model: 0) written to a YAML file passed by path.
    """
    keys = set()
    for value in args:
        if isinstance(value, str):
            value = ConfigDict.resolve_config_value(value)
        #
        if isinstance(value, (dict, ConfigDict)):
            keys |= set((value.get('training') or {}).keys())
        #
    #
    return keys


def apply_hardware_defaults(params, explicitly_set: set) -> None:
    """Auto-enable compile_model and native_amp when CUDA is available.

    Skips fields present in explicitly_set — those are deliberate user
    choices and must not be overridden. hasattr guards keep this safe for
    any params object that doesn't carry these fields. Logs an INFO line
    whenever it actually changes a value, so a run's effective config is
    never silently different from what was requested with no trace of why.
    """
    if not torch.cuda.is_available():
        return
    # Respect num_gpus=0 — user explicitly opted out of GPU acceleration.
    num_gpus = getattr(params.training, 'num_gpus', None)
    if num_gpus is not None and int(num_gpus) == 0:
        return
    if 'compile_model' not in explicitly_set and hasattr(params.training, 'compile_model'):
        if getattr(params.training, 'compile_model', 0) == 0:
            params.training.compile_model = 1
            logger.info(
                "CUDA detected: auto-enabling training.compile_model (0 -> 1); "
                "not specified in config. Set it explicitly to override."
            )
        #
    #
    if 'native_amp' not in explicitly_set and hasattr(params.training, 'native_amp'):
        if not getattr(params.training, 'native_amp', False):
            params.training.native_amp = True
            logger.info(
                "CUDA detected: auto-enabling training.native_amp (False -> True); "
                "not specified in config. Set it explicitly to override."
            )
        #
    #
