from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import re
import json
import yaml
import os
import io
import csv
import zipfile
from constants import SPLIT_TYPES

@dataclass
class DatasetSectionConfig:
    """Validated common section configuration"""
    enable: bool
    dataset_name: str
    input_data_path: str
    split_type: Optional[str] = None
    split_factor: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for YAML serialization, excluding None values"""
        result = {
            "enable": self.enable,
            "dataset_name": self.dataset_name,
            "input_data_path": self.input_data_path,
        }
        if self.split_type is not None:
            result["split_type"] = self.split_type
        if self.split_factor is not None:
            result["split_factor"] = self.split_factor
        return result

    def to_yaml_string(self) -> str:
        """Convert to YAML format string, excluding None values"""
        lines = [
            "dataset:",
            f"  enable: {str(self.enable).lower()}",
            f"  dataset_name: '{self.dataset_name}'",
            f"  input_data_path: '{self.input_data_path}'",
        ]
        if self.split_type is not None:
            lines.append(f"  split_type: '{self.split_type}'")
        if self.split_factor is not None:
            lines.append(f"  split_factor: {self.split_factor}")
        return "\n".join(lines)


class DatasetSectionValidator:
    """Validation for dataset section"""

    @staticmethod
    def validate_split_type(split_type: str) -> Tuple[bool, Optional[str]]:
        """Validate dataset split type"""
        if split_type not in SPLIT_TYPES:
            valid_types = sorted(SPLIT_TYPES)
            return False, f"Invalid task_type '{split_type}'. Valid options:\n  " + "\n  ".join(valid_types)

        return True, None

    @staticmethod
    def validate_split_factor(split_factor: List[float]) -> Tuple[bool, Optional[str]]:
        """Validate data split factor"""

        # 1. split_factor must have 3 values - train, val, and test split
        if len(split_factor) != 3:
            return False, f"Expected 3 split values for train, val, and test sets, got {len(split_factor)} values"

        # 2. split_factor must add up to 1
        train_split = split_factor[0]
        val_split = split_factor[1]
        test_split = split_factor[2]

        if train_split + test_split + val_split != 1:
            return False, "Train split, validation split, and test split must sum to 1"

        return True, None

    @classmethod
    def validate(
        cls,
        split_type: Optional[str],
        split_factor: Optional[List[float]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate all parameters for dataset section.
        Only validates split_type and split_factor if they are provided (not None).
        Returns: (is_valid, list_of_error_messages)
        """
        errors = []

        if split_type is not None:
            is_valid, error = cls.validate_split_type(split_type)
            if not is_valid:
                errors.append(error)

        if split_factor is not None:
            is_valid, error = cls.validate_split_factor(split_factor)
            if not is_valid:
                errors.append(error)

        return len(errors) == 0, errors


def validate_dataset_section(
    enable: bool,
    dataset_name: str,
    input_data_path: str,
    task_type: Optional[str] = None,
    split_type: Optional[str] = None,
    split_factor: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Validates and, where possible, auto-corrects the dataset directory layout
    to match the required BYOD format for the given task_type.

    Returns effective_input_data_path — the path that MUST be used as input_data_path
    in generate_dataset_section_yaml. This may differ from input_data_path if the
    dataset was automatically reorganised to the correct format.

    If effective_input_data_path is None, the dataset format has issues that require
    additional information (e.g. label_column for classification). Communicate the
    format_validation.reorganize_hints to the user.

    split_type and split_factor are optional. Defaults are in
    tinyml_modelmaker/ai_modules/{target_module}/params.py.
    """
    from dataset_format_tools import validate_dataset_format

    is_valid, errors = DatasetSectionValidator.validate(split_type, split_factor)

    if not is_valid:
        return {
            "success": False,
            "config": None,
            "effective_input_data_path": None,
            "format_validation": None,
            "errors": errors,
            "warnings": [],
        }

    # ── Dataset format validation (local paths only) ──────────────────────
    format_validation = None
    effective_path = input_data_path
    format_warnings: List[str] = []

    is_local = not input_data_path.startswith(("http://", "https://"))
    if is_local and task_type:
        format_validation = validate_dataset_format(input_data_path, task_type)

        if not format_validation["is_valid"] and format_validation.get("issues"):
            return {
                "success": False,
                "config": None,
                "effective_input_data_path": None,
                "format_validation": format_validation,
                "errors": [
                    "Dataset format is incorrect."
                    "Show the user the expected_structure and issues from format_validation."
                ],
                "warnings": [],
            }

    config = DatasetSectionConfig(
        enable=enable,
        dataset_name=dataset_name,
        input_data_path=effective_path,
        split_type=split_type,
        split_factor=split_factor
    )

    return {
        "success": True,
        "config": config.to_dict(),
        "effective_input_data_path": effective_path,
        "format_validation": format_validation,
        "errors": [],
        "warnings": format_warnings,
    }


def generate_dataset_section_yaml(
    enable: bool,
    dataset_name: str,
    input_data_path: str,
    split_type: Optional[str] = None,
    split_factor: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Tool: Generate YAML for the dataset section.

    IMPORTANT: input_data_path here must be the effective_input_data_path returned
    by validate_dataset_section — NOT the raw path the user provided. If the dataset
    was reorganised, effective_input_data_path will point to the corrected directory
    and that corrected path is what must appear in the generated YAML.

    Validates parameters and returns YAML string ready to write to config file.
    split_type and split_factor are optional. If not provided, users should refer to
    tinyml_modelmaker/ai_modules/{target_module}/params.py for default values.
    """
    is_valid, errors = DatasetSectionValidator.validate(split_type, split_factor)

    if not is_valid:
        return {
            "success": False,
            "yaml": None,
            "config": None,
            "errors": errors,
        }

    config = DatasetSectionConfig(
        enable=enable,
        dataset_name=dataset_name,
        input_data_path=input_data_path,
        split_type=split_type,
        split_factor=split_factor
    )

    return {
        "success": True,
        "yaml": config.to_yaml_string(),
        "config": config.to_dict(),
        "errors": [],
    }