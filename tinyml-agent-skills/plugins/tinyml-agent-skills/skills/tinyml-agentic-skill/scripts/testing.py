from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import yaml


@dataclass
class TestingSectionConfig:
    """Validated testing section configuration."""
    enable: bool = True
    skip_train: bool = False
    device_inference: bool = False
    test_data: Optional[str] = None
    model_path: Optional[str] = None

    def to_dict(self) -> Dict:
        result: Dict[str, Any] = {"enable": self.enable}
        if self.skip_train:
            result["skip_train"] = self.skip_train
        if self.device_inference:
            result["device_inference"] = self.device_inference
        if self.test_data is not None:
            result["test_data"] = self.test_data
        if self.model_path is not None:
            result["model_path"] = self.model_path
        return result

    def to_yaml_string(self) -> str:
        section = {"testing": self.to_dict()}
        return yaml.dump(section, default_flow_style=False, sort_keys=False).rstrip()


class TestingSectionValidator:

    @staticmethod
    def validate_skip_train_model_path(skip_train: bool, model_path: Optional[str]) -> Tuple[bool, Optional[str]]:
        if skip_train and not model_path:
            return False, "model_path is required when skip_train=True — the system needs a pre-trained model to test."
        return True, None

    @classmethod
    def validate(
        cls,
        enable: bool,
        skip_train: bool,
        model_path: Optional[str],
    ) -> Tuple[bool, List[str]]:
        errors = []
        ok, err = cls.validate_skip_train_model_path(skip_train, model_path)
        if not ok:
            errors.append(err)
        return len(errors) == 0, errors


def validate_testing_section(
    enable: bool = True,
    skip_train: bool = False,
    device_inference: bool = False,
    test_data: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: Validate parameters for the 'testing' section of tiny ML config.yaml.

    **Usage**
    Call this to check parameters before generating YAML.
    The agent should ask the user:
    1. Do you want to skip training and test an existing model? (skip_train)
       - If yes, ask for model_path to the pre-trained model.
    2. Do you want to run inference on the actual target device? (device_inference)
       - This is optional; most users leave it False.
    3. Do you have a custom test dataset different from the training data? (test_data)
       - Optional. If not provided, the system splits test data from the main dataset.

    All parameters except enable are optional and default to the values shown.
    When in doubt, leave everything at defaults — the pipeline runs full train+test.

    **Parameters**
    - enable: True to run testing, False to skip the entire testing phase.
    - skip_train: True to bypass training and test a pre-existing model (requires model_path).
    - device_inference: True to run inference on the actual connected target device.
    - test_data: Optional path to a separate test dataset.
    - model_path: Path to a pre-trained model file (required when skip_train=True).

    **Returns**
    Validation result without YAML. Use generate_testing_section_yaml to get YAML.
    """
    is_valid, errors = TestingSectionValidator.validate(enable, skip_train, model_path)
    if not is_valid:
        return {"success": False, "config": None, "errors": errors}

    config = TestingSectionConfig(
        enable=enable,
        skip_train=skip_train,
        device_inference=device_inference,
        test_data=test_data,
        model_path=model_path,
    )
    return {"success": True, "config": config.to_dict(), "errors": []}


def generate_testing_section_yaml(
    enable: bool = True,
    skip_train: bool = False,
    device_inference: bool = False,
    test_data: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: Generate YAML for the 'testing' section of tiny ML config.yaml.

    **Usage**
    Call this after confirming testing preferences with the user.
    Only pass parameters the user explicitly specified — omitted optional params
    will use system defaults from tinyml_modelmaker/ai_modules/{target_module}/params.py.

    For a standard full-pipeline run (train → test → compile), the minimal config is:
        testing: {}
    or equivalently: enable=True, all others at default.

    **Returns**
    - yaml: ready-to-use YAML string for the testing section
    - config: dict representation
    - errors: validation errors (empty if success)
    """
    is_valid, errors = TestingSectionValidator.validate(enable, skip_train, model_path)
    if not is_valid:
        return {"success": False, "yaml": None, "config": None, "errors": errors}

    config = TestingSectionConfig(
        enable=enable,
        skip_train=skip_train,
        device_inference=device_inference,
        test_data=test_data,
        model_path=model_path,
    )
    return {
        "success": True,
        "yaml": config.to_yaml_string(),
        "config": config.to_dict(),
        "errors": [],
    }
