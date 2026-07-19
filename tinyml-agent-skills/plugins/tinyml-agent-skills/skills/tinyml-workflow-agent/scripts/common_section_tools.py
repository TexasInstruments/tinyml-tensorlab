from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import re
import json
import yaml
from constants import *

@dataclass
class CommonSectionConfig:
    """Validated common section configuration"""
    task_type: str
    target_device: str
    target_module: Optional[str] = None  # Can be inferred from task_type if not provided
    run_name: Optional[str] = None

    def __post_init__(self):
        """Infer target_module from task_type if not provided"""
        if self.target_module is None:
            self.target_module = TASK_TYPE_TO_MODULE.get(self.task_type)

        # Set default run_name if not provided
        if self.run_name is None:
            self.run_name = "{date-time}/{model_name}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for YAML serialization"""
        return {
            "target_module": self.target_module,
            "task_type": self.task_type,
            "target_device": self.target_device,
            "run_name": self.run_name,
        }

    def to_yaml_string(self) -> str:
        """Convert to YAML format string"""
        lines = [
            "common:",
            f"  target_module: '{self.target_module}'",
            f"  task_type: '{self.task_type}'",
            f"  target_device: '{self.target_device}'",
            f"  run_name: '{self.run_name}'",
        ]
        return "\n".join(lines)


class CommonSectionValidator:
    """Validates common section configuration"""

    VALID_MODULES = list(set(TASK_TYPE_TO_MODULE.values()))  # ['timeseries', 'vision']

    @staticmethod #This needs to handle new types of tasks as well -> some way to figure out what matches task it matches closest, ask clarifying questions about the task itself then determine 
    def validate_task_type(task_type: str) -> Tuple[bool, Optional[str]]:
        """Validate task_type parameter""" 
        if task_type not in TASK_TYPE_TO_MODULE:
            valid_types = sorted(TASK_TYPE_TO_MODULE.keys())
            return False, f"Invalid task_type '{task_type}'. Valid options:\n  " + "\n  ".join(valid_types)
        return True, None

    @staticmethod
    def validate_target_device(device: str) -> Tuple[bool, Optional[str]]:
        """Validate target_device parameter"""
        if device not in ALL_TARGET_DEVICES:
            return False, f"Invalid target_device '{device}'. Valid options: {sorted(ALL_TARGET_DEVICES)}"
        return True, None

    @staticmethod
    def validate_run_name(run_name: str) -> Tuple[bool, Optional[str]]:
        """Validate run_name format"""
        # Allow alphanumeric, hyphens, underscores, slashes, and placeholders like {date-time}
        valid_chars = r"^[\w\-\{}\./]+$"
        if not re.match(valid_chars, run_name):
            return False, f"Invalid run_name '{run_name}'. Use alphanumeric, -, _, /, or placeholders like {{date-time}}, {{model_name}}"
        return True, None

    @classmethod
    def validate(cls, task_type: str, target_device: str, run_name: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate all parameters for common section.
        Returns: (is_valid, list_of_error_messages)
        """
        errors = []

        is_valid, error = cls.validate_task_type(task_type)
        if not is_valid:
            errors.append(error)

        is_valid, error = cls.validate_target_device(target_device)
        if not is_valid:
            errors.append(error)

        if run_name:
            is_valid, error = cls.validate_run_name(run_name)
            if not is_valid:
                errors.append(error)

        return len(errors) == 0, errors


def validate_common_section(
    task_type: str,
    target_device: str,
    target_module: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: Validate common section parameters.

    This tool takes structured parameters (not natural language) and validates them.
    The calling agent is responsible for understanding NL input and extracting parameters.

    Args:
        task_type: One of the valid task types (e.g., "generic_timeseries_classification", "motor_fault")
        target_device: One of the supported MCUs (e.g., "F28P55", "MSPM0G3507")
        target_module: Optional, will be inferred from task_type if not provided
        run_name: Optional, defaults to "{date-time}/{model_name}"

    Returns:
        Dict with keys:
            - success: bool indicating if validation passed
            - config: CommonSectionConfig dict if valid
            - errors: List of error messages if invalid
            - inferred_module: The inferred target_module if it was not provided
    """
    is_valid, errors = CommonSectionValidator.validate(task_type, target_device, run_name)

    if not is_valid:
        return {
            "success": False,
            "config": None,
            "errors": errors,
            "inferred_module": TASK_TYPE_TO_MODULE.get(task_type),
        }

    # Create and return the config
    config = CommonSectionConfig(
        task_type=task_type,
        target_device=target_device,
        target_module=target_module,
        run_name=run_name,
    )

    return {
        "success": True,
        "config": config.to_dict(),
        "errors": [],
        "inferred_module": config.target_module,
    }


def generate_common_section_yaml(
    task_type: str,
    target_device: str,
    target_module: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: Generate YAML for the common section.

    Validates parameters and returns YAML string ready to write to config file.

    Args:
        task_type: One of the valid task types
        target_device: One of the supported MCUs
        target_module: Optional, will be inferred from task_type
        run_name: Optional, defaults to "{date-time}/{model_name}"

    Returns:
        Dict with keys:
            - success: bool
            - yaml: YAML string if valid
            - config: Config dict if valid
            - errors: Error messages if invalid
    """
    is_valid, errors = CommonSectionValidator.validate(task_type, target_device, run_name)

    if not is_valid:
        return {
            "success": False,
            "yaml": None,
            "config": None,
            "errors": errors,
        }

    config = CommonSectionConfig(
        task_type=task_type,
        target_device=target_device,
        target_module=target_module,
        run_name=run_name,
    )

    return {
        "success": True,
        "yaml": config.to_yaml_string(),
        "config": config.to_dict(),
        "errors": [],
    }


def list_supported_values(parameter_type: str) -> Dict[str, Any]:
    """
    Tool: List all valid values for common section parameters.

    Useful for agents to understand what values are valid for each parameter.

    Args:
        parameter_type: One of: "task_type", "target_device", "target_module"

    Returns:
        Dict with valid values and descriptions
    """
    if parameter_type == "task_type":
        task_types = {}
        for task, module in TASK_TYPE_TO_MODULE.items():
            task_types[task] = {
                "target_module": module,
                "device_restrictions": task in DEVICE_TASK_SUPPORT,
            }
            if task in DEVICE_TASK_SUPPORT:
                task_types[task]["supported_devices"] = DEVICE_TASK_SUPPORT[task]
        return {
            "parameter_type": "task_type",
            "values": task_types,
            "total_count": len(task_types),
        }

    elif parameter_type == "target_device":
        return {
            "parameter_type": "target_device",
            "values": sorted(ALL_TARGET_DEVICES),
            "total_count": len(ALL_TARGET_DEVICES),
        }

    elif parameter_type == "target_module":
        return {
            "parameter_type": "target_module",
            "values": ["timeseries", "vision"],
            "note": "target_module is usually inferred from task_type",
        }

    else:
        return {"error": f"Unknown parameter_type: {parameter_type}"}