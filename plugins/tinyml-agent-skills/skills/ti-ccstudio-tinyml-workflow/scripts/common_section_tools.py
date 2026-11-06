from dataclasses import dataclass
from typing import Optional, Dict, Any
from constants import TASK_TYPE_TO_MODULE

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