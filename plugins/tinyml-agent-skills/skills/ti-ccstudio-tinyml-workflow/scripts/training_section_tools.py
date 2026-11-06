from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import re
import json
import yaml
from constants import *

# NPU-capable devices — require quantization: 2 for NPU deployment
_NPU_DEVICES = {
    "F28P55",
    "MSPM0G5187",
    "AM13E2",
    "CC2745R10_Q1"
}

_VALID_QUANTIZATION_MODES = {0, 1, 2}
_VALID_QUANTIZATION_METHODS = {"PTQ", "QAT"}
_VALID_BITWIDTHS = {2, 4, 8}
_VALID_NAS_OPTIMIZATION_MODES = {"Memory", "Compute"}
_VALID_NAS_MODEL_SIZES = {"s", "m", "l", "xl", "xxl"}


@dataclass
class TrainingSectionConfig:
    """Validated training section configuration"""
    enable: bool
    model_name: str
    batch_size: Optional[int] = None
    training_epochs: Optional[int] = None
    num_gpus: Optional[int] = None
    learning_rate: Optional[float] = None

    # Quantization params
    quantization: Optional[int] = None                          # 0=float, 1=standard, 2=TI NPU-optimized
    quantization_method: Optional[str] = None                  # 'PTQ' or 'QAT'
    quantization_weight_bitwidth: Optional[int] = None         # 2, 4, or 8
    quantization_activation_bitwidth: Optional[int] = None     # 2, 4, or 8

    # NAS params (preset mode)
    nas_enabled: Optional[bool] = None
    nas_epochs: Optional[int] = None
    nas_optimization_mode: Optional[str] = None                # 'Memory' or 'Compute'
    nas_model_size: Optional[str] = None                       # 's', 'm', 'l', 'xl', 'xxl'

    # NAS customization (advanced — only if not using nas_model_size)
    nas_nodes_per_layer: Optional[int] = None
    nas_layers: Optional[int] = None
    nas_init_channels: Optional[int] = None
    nas_init_channel_multiplier: Optional[int] = None
    nas_fanout_concat: Optional[int] = None

    def to_dict(self) -> Dict:
        result = {
            "enable": self.enable,
            "model_name": self.model_name,
        }
        for attr in (
            "batch_size", "training_epochs", "num_gpus", "learning_rate",
            "quantization", "quantization_method",
            "quantization_weight_bitwidth", "quantization_activation_bitwidth",
            "nas_enabled", "nas_epochs", "nas_optimization_mode", "nas_model_size",
            "nas_nodes_per_layer", "nas_layers", "nas_init_channels",
            "nas_init_channel_multiplier", "nas_fanout_concat",
        ):
            val = getattr(self, attr)
            if val is not None:
                result[attr] = val
        return result

    def to_yaml_string(self) -> str:
        lines = [
            "training:",
            f"  enable: {str(self.enable).lower()}",
            f"  model_name: '{self.model_name}'",
        ]
        for attr, fmt in (
            ("batch_size",                      "{}"),
            ("training_epochs",                 "{}"),
            ("num_gpus",                        "{}"),
            ("learning_rate",                   "{}"),
            ("quantization",                    "{}"),
            ("quantization_method",             "'{}'"),
            ("quantization_weight_bitwidth",    "{}"),
            ("quantization_activation_bitwidth","{}"),
            ("nas_enabled",                     None),   # bool — handled below
            ("nas_epochs",                      "{}"),
            ("nas_optimization_mode",           "'{}'"),
            ("nas_model_size",                  "'{}'"),
            ("nas_nodes_per_layer",             "{}"),
            ("nas_layers",                      "{}"),
            ("nas_init_channels",               "{}"),
            ("nas_init_channel_multiplier",     "{}"),
            ("nas_fanout_concat",               "{}"),
        ):
            val = getattr(self, attr)
            if val is None:
                continue
            if attr == "nas_enabled":
                lines.append(f"  nas_enabled: {str(val).lower()}")
            else:
                lines.append(f"  {attr}: {fmt.format(val)}")
        return "\n".join(lines)


class TrainingSectionValidator:

    @staticmethod
    def validate_model_name(model_name: str) -> Tuple[bool, Optional[str]]:
        if not model_name or not model_name.strip():
            return False, "model_name cannot be empty"
        return True, None

    @staticmethod
    def validate_batch_size(batch_size: Optional[int]) -> Tuple[bool, Optional[str]]:
        if batch_size is None:
            return True, None
        if batch_size <= 0:
            return False, "batch_size must be > 0"
        return True, None

    @staticmethod
    def validate_training_epochs(training_epochs: Optional[int]) -> Tuple[bool, Optional[str]]:
        if training_epochs is None:
            return True, None
        if training_epochs <= 0:
            return False, "training_epochs must be > 0"
        return True, None

    @staticmethod
    def validate_num_gpus(num_gpus: Optional[int]) -> Tuple[bool, Optional[str]]:
        if num_gpus is None:
            return True, None
        if num_gpus < 0:
            return False, "num_gpus cannot be negative"
        return True, None

    @staticmethod
    def validate_learning_rate(learning_rate: Optional[float]) -> Tuple[bool, Optional[str]]:
        if learning_rate is None:
            return True, None
        if learning_rate <= 0:
            return False, "learning_rate must be > 0"
        return True, None

    @staticmethod
    def validate_quantization(
        quantization: Optional[int],
        quantization_method: Optional[str],
        quantization_weight_bitwidth: Optional[int],
        quantization_activation_bitwidth: Optional[int],
    ) -> Tuple[bool, List[str]]:
        errors = []
        if quantization is None:
            return True, []
        if quantization not in _VALID_QUANTIZATION_MODES:
            errors.append(f"quantization must be 0, 1, or 2 (got {quantization}). "
                          "0=float32, 1=standard PyTorch, 2=TI NPU-optimized.")
            return False, errors
        if quantization > 0:
            if quantization_method and quantization_method not in _VALID_QUANTIZATION_METHODS:
                errors.append(f"quantization_method must be 'PTQ' or 'QAT' (got '{quantization_method}').")
            if quantization_weight_bitwidth is not None and quantization_weight_bitwidth not in _VALID_BITWIDTHS:
                errors.append(f"quantization_weight_bitwidth must be 2, 4, or 8 (got {quantization_weight_bitwidth}).")
            if quantization_activation_bitwidth is not None and quantization_activation_bitwidth not in _VALID_BITWIDTHS:
                errors.append(f"quantization_activation_bitwidth must be 2, 4, or 8 (got {quantization_activation_bitwidth}).")
        return len(errors) == 0, errors

    @staticmethod
    def validate_nas(
        nas_enabled: Optional[bool],
        nas_epochs: Optional[int],
        nas_optimization_mode: Optional[str],
        nas_model_size: Optional[str],
    ) -> Tuple[bool, List[str]]:
        errors = []
        if not nas_enabled:
            return True, []
        if nas_epochs is not None and nas_epochs <= 0:
            errors.append("nas_epochs must be > 0.")
        if nas_optimization_mode and nas_optimization_mode not in _VALID_NAS_OPTIMIZATION_MODES:
            errors.append(f"nas_optimization_mode must be 'Memory' or 'Compute' (got '{nas_optimization_mode}').")
        if nas_model_size and nas_model_size not in _VALID_NAS_MODEL_SIZES:
            errors.append(f"nas_model_size must be one of {sorted(_VALID_NAS_MODEL_SIZES)} (got '{nas_model_size}').")
        return len(errors) == 0, errors

    @classmethod
    def validate(
        cls,
        enable: bool,
        model_name: str,
        batch_size: Optional[int] = None,
        training_epochs: Optional[int] = None,
        num_gpus: Optional[int] = None,
        learning_rate: Optional[float] = None,
        quantization: Optional[int] = None,
        quantization_method: Optional[str] = None,
        quantization_weight_bitwidth: Optional[int] = None,
        quantization_activation_bitwidth: Optional[int] = None,
        nas_enabled: Optional[bool] = None,
        nas_epochs: Optional[int] = None,
        nas_optimization_mode: Optional[str] = None,
        nas_model_size: Optional[str] = None,
    ) -> Tuple[bool, List[str]]:
        errors = []

        ok, err = cls.validate_model_name(model_name)
        if not ok:
            errors.append(err)
        ok, err = cls.validate_batch_size(batch_size)
        if not ok:
            errors.append(err)
        ok, err = cls.validate_training_epochs(training_epochs)
        if not ok:
            errors.append(err)
        ok, err = cls.validate_num_gpus(num_gpus)
        if not ok:
            errors.append(err)
        ok, err = cls.validate_learning_rate(learning_rate)
        if not ok:
            errors.append(err)
        ok, errs = cls.validate_quantization(
            quantization, quantization_method,
            quantization_weight_bitwidth, quantization_activation_bitwidth,
        )
        errors.extend(errs)
        ok, errs = cls.validate_nas(nas_enabled, nas_epochs, nas_optimization_mode, nas_model_size)
        errors.extend(errs)

        return len(errors) == 0, errors


def get_training_recommendations(
    target_device: str,
    dataset_size_bucket: Optional[str] = None,
    num_gpus: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tool: Get context-aware recommendations for quantization and NAS.

    Call this BEFORE asking the user about quantization and NAS.
    Use the returned recommendations to explain options and suggest defaults.

    Args:
        target_device: Target MCU (e.g., 'F28P55', 'MSPM0G3507').
        dataset_size_bucket: Optional. 'tiny'|'small'|'medium'|'large' from analyze_dataset_for_model_guidance.
        num_gpus: Optional. Number of GPUs available (0 or None = no GPU).

    Returns:
        quantization recommendations and plain-English explanations.
    """
    device_has_npu = target_device in _NPU_DEVICES
    has_gpu = num_gpus is not None and num_gpus > 0

    # ── Quantization recommendation ───────────────────────────────────────────
    if device_has_npu:
        recommended_quantization = 2
        quantization_reason = (
            f"Device '{target_device}' has a hardware NPU. "
            "quantization: 2 (TI NPU-optimized) is required to use the NPU at inference time. "
            "Without quantization, the model runs on CPU only."
        )
    else:
        recommended_quantization = 1
        quantization_reason = (
            f"Device '{target_device}' has no hardware NPU. "
            "quantization: 1 (standard PyTorch) gives smaller model size and faster CPU inference. "
            "quantization: 0 (no quantization) is also valid if model size is not a concern."
        )

    quantization_options = {
        0: "Float32 training — no quantization. Largest model, slowest inference. Use only when evaluating accuracy first.",
        1: "Standard PyTorch quantization — general-purpose, works on all devices. 4× size reduction typical.",
        2: "TI NPU-optimized quantization — required for NPU acceleration. Best performance on NPU-equipped devices.",
    }

    autoquant_explanation = (
        "When quantization: 1 or 2 is selected, Automatic Mixed Precision (AMP) is enabled by default. "
        "AMP automatically assigns per-layer bit widths (2, 4, 8, or 32) using Hessian-aware sensitivity analysis, "
        "balancing model size and accuracy without manual tuning. No PTQ/QAT or bit width configuration needed — "
        "the system handles this optimally for your task and dataset."
    )

    return {
        "success": True,
        "device_has_npu": device_has_npu,
        # Quantization
        "quantization": {
            "recommended_mode": recommended_quantization,
            "reason": quantization_reason,
            "options": quantization_options,
            "autoquant_explanation": autoquant_explanation,
        }
    }


def validate_training_section(
    enable: bool,
    model_name: str,
    batch_size: Optional[int] = None,
    training_epochs: Optional[int] = None,
    num_gpus: Optional[int] = None,
    learning_rate: Optional[float] = None,
    quantization: Optional[int] = None,
    quantization_method: Optional[str] = None,
    quantization_weight_bitwidth: Optional[int] = None,
    quantization_activation_bitwidth: Optional[int] = None,
    nas_enabled: Optional[bool] = None,
    nas_epochs: Optional[int] = None,
    nas_optimization_mode: Optional[str] = None,
    nas_model_size: Optional[str] = None,
    nas_nodes_per_layer: Optional[int] = None,
    nas_layers: Optional[int] = None,
    nas_init_channels: Optional[int] = None,
    nas_init_channel_multiplier: Optional[int] = None,
    nas_fanout_concat: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tool: Validate training section parameters including quantization and NAS.

    enable and model_name are required. All other parameters are optional.
    Defaults are loaded from tinyml_modelmaker/ai_modules/{target_module}/params.py at runtime.
    """
    is_valid, errors = TrainingSectionValidator.validate(
        enable, model_name, batch_size, training_epochs, num_gpus, learning_rate,
        quantization, quantization_method, quantization_weight_bitwidth,
        quantization_activation_bitwidth, nas_enabled, nas_epochs,
        nas_optimization_mode, nas_model_size,
    )

    if not is_valid:
        return {"success": False, "config": None, "errors": errors}

    config = TrainingSectionConfig(
        enable=enable, model_name=model_name,
        batch_size=batch_size, training_epochs=training_epochs,
        num_gpus=num_gpus, learning_rate=learning_rate,
        quantization=quantization, quantization_method=quantization_method,
        quantization_weight_bitwidth=quantization_weight_bitwidth,
        quantization_activation_bitwidth=quantization_activation_bitwidth,
        nas_enabled=nas_enabled, nas_epochs=nas_epochs,
        nas_optimization_mode=nas_optimization_mode, nas_model_size=nas_model_size,
        nas_nodes_per_layer=nas_nodes_per_layer, nas_layers=nas_layers,
        nas_init_channels=nas_init_channels,
        nas_init_channel_multiplier=nas_init_channel_multiplier,
        nas_fanout_concat=nas_fanout_concat,
    )
    return {"success": True, "config": config.to_dict(), "errors": []}


def generate_training_section_yaml(
    enable: bool,
    model_name: str,
    batch_size: Optional[int] = None,
    training_epochs: Optional[int] = None,
    num_gpus: Optional[int] = None,
    learning_rate: Optional[float] = None,
    quantization: Optional[int] = None,
    quantization_method: Optional[str] = None,
    quantization_weight_bitwidth: Optional[int] = None,
    quantization_activation_bitwidth: Optional[int] = None,
    nas_enabled: Optional[bool] = None,
    nas_epochs: Optional[int] = None,
    nas_optimization_mode: Optional[str] = None,
    nas_model_size: Optional[str] = None,
    nas_nodes_per_layer: Optional[int] = None,
    nas_layers: Optional[int] = None,
    nas_init_channels: Optional[int] = None,
    nas_init_channel_multiplier: Optional[int] = None,
    nas_fanout_concat: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tool: Generate YAML for the training section.

    Validates all parameters and returns the YAML string ready to write to config file.
    Only enable and model_name are required; everything else is optional.

    Quantization guidance:
      - NPU devices require quantization=2 (TI NPU-optimized) for NPU acceleration
      - Non-NPU devices: use quantization=1 (standard) or 0 (no quantization)
      - QAT recommended over PTQ for accuracy retention
      - Start with quantization_weight_bitwidth=8 and quantization_activation_bitwidth=8

    NAS guidance:
      - Requires GPU (num_gpus >= 1); impractical on CPU
      - Use nas_model_size preset ('s' or 'm' to start) instead of manual customization
      - nas_optimization_mode: 'Memory' for flash/RAM constrained, 'Compute' for latency
    """
    is_valid, errors = TrainingSectionValidator.validate(
        enable, model_name, batch_size, training_epochs, num_gpus, learning_rate,
        quantization, quantization_method, quantization_weight_bitwidth,
        quantization_activation_bitwidth, nas_enabled, nas_epochs,
        nas_optimization_mode, nas_model_size,
    )

    if not is_valid:
        return {"success": False, "yaml": None, "config": None, "errors": errors}

    config = TrainingSectionConfig(
        enable=enable, model_name=model_name,
        batch_size=batch_size, training_epochs=training_epochs,
        num_gpus=num_gpus, learning_rate=learning_rate,
        quantization=quantization, quantization_method=quantization_method,
        quantization_weight_bitwidth=quantization_weight_bitwidth,
        quantization_activation_bitwidth=quantization_activation_bitwidth,
        nas_enabled=nas_enabled, nas_epochs=nas_epochs,
        nas_optimization_mode=nas_optimization_mode, nas_model_size=nas_model_size,
        nas_nodes_per_layer=nas_nodes_per_layer, nas_layers=nas_layers,
        nas_init_channels=nas_init_channels,
        nas_init_channel_multiplier=nas_init_channel_multiplier,
        nas_fanout_concat=nas_fanout_concat,
    )
    return {
        "success": True,
        "yaml": config.to_yaml_string(),
        "config": config.to_dict(),
        "errors": [],
    }
