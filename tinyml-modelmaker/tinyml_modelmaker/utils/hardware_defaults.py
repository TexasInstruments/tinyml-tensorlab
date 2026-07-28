import torch


def apply_hardware_defaults(params, explicitly_set: set) -> None:
    """Auto-enable compile_model and native_amp when CUDA is available.

    Skips fields present in explicitly_set — those are deliberate user
    choices from the YAML config and must not be overridden.
    hasattr guards keep this safe for params that don't carry these fields
    yet (vision, audio — Phase 2).
    """
    if not torch.cuda.is_available():
        return
    if 'compile_model' not in explicitly_set and hasattr(params.training, 'compile_model'):
        if getattr(params.training, 'compile_model', 0) == 0:
            params.training.compile_model = 1
    if 'native_amp' not in explicitly_set and hasattr(params.training, 'native_amp'):
        if not getattr(params.training, 'native_amp', False):
            params.training.native_amp = True
