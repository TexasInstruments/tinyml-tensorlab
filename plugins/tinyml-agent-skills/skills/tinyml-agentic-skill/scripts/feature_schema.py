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


def get_preset(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a feature extraction preset by name.

    Args:
        name: Preset name (e.g., 'Generic_1024Input_FFTBIN_64Feature_8Frame')

    Returns:
        Preset config dict, or None if not found.
    """
    schema = _load_schema()
    return schema.get("presets", {}).get(name)


def list_all_presets() -> List[str]:
    """Get list of all available preset names."""
    schema = _load_schema()
    return list(schema.get("presets", {}).keys())


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
