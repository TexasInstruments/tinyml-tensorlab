"""
Feature extraction schema loader and accessor.

Provides centralized access to feature extraction definitions (presets, transforms,
augmenters, and task recommendations) loaded from schema.yaml. This replaces
hardcoded dictionaries in constants.py and tools/feature_extraction.py.
"""

import yaml
import os
from typing import Dict, List, Optional, Any

_SCHEMA = None
_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.yaml")


def _load_schema() -> Dict[str, Any]:
    """Load schema.yaml once and cache it."""
    global _SCHEMA
    if _SCHEMA is None:
        with open(_SCHEMA_PATH, "r") as f:
            _SCHEMA = yaml.safe_load(f)
    return _SCHEMA


_CATALOG = None   # preset name -> config, from the installed tinyml_modelmaker (set by use_modelmaker_catalog)
_MODULE_TRANSFORMS = set()   # every feat_ext_transform step used by any preset of the module (any task type)


def _require_catalog() -> Dict[str, Any]:
    if _CATALOG is None:
        raise RuntimeError("Preset catalog not loaded: call use_modelmaker_catalog(task_type) first.")
    return _CATALOG


def use_modelmaker_catalog(task_type: str) -> str:
    """Serve presets from the installed tinyml_modelmaker for this task type's module.

    Reads FEATURE_EXTRACTION_PRESET_DESCRIPTIONS in ai_modules/<module>/constants.py (timeseries, radar,
    vision, ...) and keeps only presets registered for task_type, so every option is a real preset of the
    installed version. schema.yaml only supplements them (descriptions and defaults modelmaker leaves unset).
    Returns a short description of the source that is now active.
    """
    global _CATALOG, _MODULE_TRANSFORMS
    try:
        import importlib
        from constants import TASK_TYPE_TO_MODULE
        module = TASK_TYPE_TO_MODULE[task_type]
        mm = importlib.import_module(f"tinyml_modelmaker.ai_modules.{module}.constants")
        described = mm.FEATURE_EXTRACTION_PRESET_DESCRIPTIONS
    except Exception as e:
        _CATALOG = None
        raise RuntimeError(
            f"Cannot load the preset catalog for task_type '{task_type}' from the installed tinyml_modelmaker "
            f"({type(e).__name__}: {e}). tinyml_modelmaker is required: check the task type, or re-run "
            "/tinyml-agent-skills:setup if the package is missing."
        ) from e
    schema_presets = _load_schema().get("presets", {})
    catalog = {}
    _MODULE_TRANSFORMS = set()
    for name, entry in described.items():
        _MODULE_TRANSFORMS.update(entry.get("data_processing_feature_extraction", {}).get("feat_ext_transform") or [])
        task_types = entry.get("common", {}).get("task_type")
        if not isinstance(task_types, (list, tuple, set)):
            task_types = [task_types]
        if task_type not in task_types:
            continue
        fe = entry.get("data_processing_feature_extraction", {})
        cfg = dict(schema_presets.get(name, {}))
        # Carry every value the preset sets (frame_size, feature_size_per_frame, num_frame_concat, stacking,
        # feat_ext_transform, data_proc_transforms, min_bin, analysis_bandwidth, frame_skip, sampling_rate, ...):
        # the arithmetic in fe_rules.py needs them, not just the five sizes.
        for key, value in fe.items():
            if value is not None:
                cfg[key] = value
        cfg["variables"] = fe.get("variables", cfg.get("variables", 1))
        if "description" not in cfg:
            transforms = cfg.get("feat_ext_transform") or []
            if any("PIR" in t for t in transforms):
                kind = "PIR"
            elif any("Q15" in t for t in transforms):
                kind = "fixed-point (Q15) FFT"
            elif "RAW_FE" in transforms:
                kind = "RAW"
            elif any("FFT" in t for t in transforms):
                kind = "FFT"
            else:
                kind = "custom"
            sizes = ", ".join(f"{cfg[key]} {label}" for key, label in (("frame_size", "samples/frame"), ("feature_size_per_frame", "features/frame"), ("num_frame_concat", "frames")) if cfg.get(key))
            cfg["description"] = f"{kind} preset from tinyml_modelmaker {module} constants.py ({sizes or 'sizes set by the user'})."
        cfg.setdefault("use_case", "")
        catalog[name] = cfg
    _CATALOG = catalog
    return f"installed tinyml_modelmaker ai_modules.{module}.constants ({len(catalog)} presets for {task_type})"


def known_transform_names() -> List[str]:
    """Transform names a custom feat_ext_transform may use: those in schema.yaml plus any used by a preset of
    the installed modelmaker module (training silently ignores names it does not know)."""
    from fe_rules import NO_EFFECT
    _require_catalog()
    # Some presets list names no dataset class acts on (e.g. FE_DECIMATE, FE_Q15); they are not usable steps.
    return sorted((set(list_all_transforms()) | _MODULE_TRANSFORMS) - NO_EFFECT)


def get_preset(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a feature extraction preset by name.

    Args:
        name: Preset name (e.g., 'Generic_1024Input_FFTBIN_64Feature_8Frame')

    Returns:
        Preset config dict, or None if not found.
    """
    return _require_catalog().get(name)


def list_all_presets() -> List[str]:
    """Get list of all available preset names."""
    return list(_require_catalog())


def get_transform(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a transform definition by name.

    Args:
        name: Transform name (e.g., 'FFT_FE', 'BINNING', 'SimpleWindow')

    Returns:
        Transform definition dict, or None if not found.
    """
    schema = _load_schema()
    return schema.get("transforms", {}).get(name)


def list_all_transforms() -> List[str]:
    """Get list of all available transform names."""
    schema = _load_schema()
    return list(schema.get("transforms", {}).keys())


def list_transforms_by_category(category: str) -> List[str]:
    """
    Get transform names by category.

    Args:
        category: e.g., 'feature_extraction', 'data_proc', 'feature_extraction_q15'

    Returns:
        List of transform names in that category.
    """
    schema = _load_schema()
    transforms = schema.get("transforms", {})
    return [
        name for name, cfg in transforms.items()
        if cfg.get("category") == category
    ]


def get_augmenter(name: str) -> Optional[Dict[str, Any]]:
    """
    Get an augmenter definition by name.

    Args:
        name: Augmenter name (e.g., 'AddNoise', 'Crop')

    Returns:
        Augmenter definition dict, or None if not found.
    """
    schema = _load_schema()
    return schema.get("augmenters", {}).get(name)


def list_all_augmenters() -> List[str]:
    """Get list of all available augmenter names."""
    schema = _load_schema()
    return list(schema.get("augmenters", {}).keys())


def get_task_recommendations(task_type: str) -> Optional[Dict[str, Any]]:
    """
    Get task-specific feature extraction recommendations.

    Args:
        task_type: Task type name (e.g., 'motor_fault', 'arc_fault')

    Returns:
        Task recommendations dict, or None if not found.
    """
    schema = _load_schema()
    return schema.get("task_recommendations", {}).get(task_type)


def list_all_tasks() -> List[str]:
    """Get list of all task types with recommendations."""
    schema = _load_schema()
    return list(schema.get("task_recommendations", {}).keys())


def validate_feat_ext_pipeline(feat_ext_transform: List[str]) -> tuple[bool, List[str]]:
    """
    Validate a feature extraction transform pipeline.

    Args:
        feat_ext_transform: List of transform names in order.

    Returns:
        (is_valid, error_messages). If is_valid is True, error_messages is empty.
    """
    if not feat_ext_transform:
        return True, []

    errors = []
    all_transforms = list_all_transforms()

    for i, transform_name in enumerate(feat_ext_transform):
        if transform_name not in all_transforms:
            errors.append(f"Step {i}: Unknown transform '{transform_name}'")

    return len(errors) == 0, errors


def get_stacking_mode(mode: str) -> Optional[Dict[str, Any]]:
    """
    Get stacking mode definition.

    Args:
        mode: Stacking mode name (e.g., '2D1', '1D')

    Returns:
        Mode definition dict, or None if not found.
    """
    schema = _load_schema()
    return schema.get("stacking_modes", {}).get(mode)


def list_all_stacking_modes() -> List[str]:
    """Get list of available stacking modes."""
    schema = _load_schema()
    return list(schema.get("stacking_modes", {}).keys())
