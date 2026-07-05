from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import os
import yaml

# ─── Preset definitions ───────────────────────────────────────────────────────

COMPILE_PRESETS = {
    "default_preset": {
        "label": "Default (best available)",
        "description": (
            "Uses the best available execution path for the target device. "
            "Automatically enables hardware NPU acceleration if the device has one. "
            "Best choice for most tasks."
        ),
    },
    "forced_soft_npu_preset": {
        "label": "Forced Software NPU",
        "description": (
            "Disables hardware NPU and forces all computation to software (CPU) path. "
            "Recommended for anomaly detection and forecasting tasks where the NPU "
            "execution path may not produce optimal results. Also use when you want "
            "to validate the software inference path."
        ),
    },
    "compress_npu_layer_data": {
        "label": "Compressed NPU (optimised for space)",
        "description": (
            "Hardware NPU execution with compressed layer data to reduce on-device memory. "
            "Use on NPU-equipped devices with tight SRAM constraints "
            "(e.g., MSPM0G family)."
        ),
    },
}

# Devices that have a hardware NPU and therefore support all three presets
NPU_DEVICES = {
    "F28P55", "F28P65",           # C2000 with TI-NNPU
    "MSPM0G5187",                  # MSPM0 with TI-NPU
    "MSPM33C34",                   # MSPM33 with TI-NPU
    "CC2755", "CC35X1",            # Connectivity with TI-NPU
    "AM13E2",                      # AM13 with TI-NPU
}

# Tasks where forced_soft_npu_preset is the recommended default
SOFT_NPU_PREFERRED_TASKS = {
    "generic_timeseries_anomalydetection",
    "generic_timeseries_forecasting",
}

# Context paths — resolved relative to this file's skill root (no hardcoded user paths)
_SKILL_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFS_DIR   = os.path.join(_SKILL_DIR, "references")
_ASSETS_DIR = os.path.join(_SKILL_DIR, "assets")

CONTEXT_PATHS = {
    "npu_guidelines":        os.path.join(_REFS_DIR,   "docs", "source", "devices", "npu_guidelines.rst"),
    "compilation_constants": os.path.join(_ASSETS_DIR, "timeseries_module_constants.md"),
    "timeseries_params":     os.path.join(_ASSETS_DIR, "timeseries_default_params.md"),
    "vision_params":         os.path.join(_ASSETS_DIR, "vision_default_params.md"),
}


# ─── Config dataclass ─────────────────────────────────────────────────────────

@dataclass
class CompilationSectionConfig:
    """Validated compilation section configuration."""
    enable: bool = True
    compile_preset_name: Optional[str] = None
    model_path: Optional[str] = None
    compile_output_path: Optional[str] = None
    keep_libc_files: bool = False

    def to_dict(self) -> Dict:
        result: Dict[str, Any] = {"enable": self.enable}
        if self.compile_preset_name is not None:
            result["compile_preset_name"] = self.compile_preset_name
        if self.model_path is not None:
            result["model_path"] = self.model_path
        if self.compile_output_path is not None:
            result["compile_output_path"] = self.compile_output_path
        if self.keep_libc_files:
            result["keep_libc_files"] = self.keep_libc_files
        return result

    def to_yaml_string(self) -> str:
        section = {"compilation": self.to_dict()}
        return yaml.dump(section, default_flow_style=False, sort_keys=False).rstrip()


# ─── Validator ────────────────────────────────────────────────────────────────

class CompilationSectionValidator:

    @staticmethod
    def validate_preset(
        compile_preset_name: Optional[str],
        target_device: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        if compile_preset_name is None:
            return True, None
        if compile_preset_name not in COMPILE_PRESETS:
            valid = ", ".join(f"'{p}'" for p in COMPILE_PRESETS)
            return False, f"Invalid compile_preset_name '{compile_preset_name}'. Valid options: {valid}."
        if compile_preset_name in ("forced_soft_npu_preset", "compress_npu_layer_data"):
            if target_device and target_device not in NPU_DEVICES:
                return False, (
                    f"Preset '{compile_preset_name}' is only available for devices with a hardware NPU "
                    f"({', '.join(sorted(NPU_DEVICES))}). Device '{target_device}' does not have a hardware NPU."
                )
        return True, None

    @classmethod
    def validate(
        cls,
        enable: bool,
        compile_preset_name: Optional[str],
        target_device: Optional[str],
        model_path: Optional[str],
    ) -> Tuple[bool, List[str]]:
        errors = []
        ok, err = cls.validate_preset(compile_preset_name, target_device)
        if not ok:
            errors.append(err)
        return len(errors) == 0, errors


# ─── Tool 1: Recommendations ──────────────────────────────────────────────────

def get_compilation_preset_recommendations(
    task_type: str,
    target_device: str,
    quantization_mode: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tool: Get a recommendation for the compile_preset_name.

    Takes quantization_mode from the training section into account — this is the
    primary driver of which preset to use on NPU-equipped devices.

    Call this BEFORE asking the user about compilation settings and AFTER the
    training section (Step 8) so quantization_mode is known.

    Args:
        task_type: Task type (e.g., 'motor_fault', 'generic_timeseries_anomalydetection').
        target_device: Target MCU (e.g., 'F28P55', 'MSPM0G3507').
        quantization_mode: Quantization mode chosen in training section.
            0 = float32 (no quantization)
            1 = standard PyTorch quantization
            2 = TI NPU-optimized quantization  ← required for NPU hardware
            None = unknown / not set

    Returns:
        recommended_preset, recommendation_reason, device_has_npu, available_presets,
        preset_details, context_paths.
    """
    device_has_npu    = target_device in NPU_DEVICES
    task_prefers_soft = task_type in SOFT_NPU_PREFERRED_TASKS

    # Available presets depend on device NPU capability
    available = list(COMPILE_PRESETS.keys()) if device_has_npu else ["default_preset", "forced_soft_npu_preset"]

    # ── Preset selection logic ────────────────────────────────────────────────
    if not device_has_npu:
        # No NPU → always use default (software path)
        recommended = "default_preset"
        reason = f"Device '{target_device}' has no hardware NPU. Software inference path used."

    elif quantization_mode is not None and quantization_mode != 2:
        # NPU device but model is NOT TI-NPU-quantized → must force software path.
        # The NPU requires quantization: 2 (TI NPU-optimized) to function.
        # Using default_preset with a non-TI-quantized model would silently fall back
        # to CPU anyway, but forced_soft_npu_preset makes this explicit and avoids
        # unexpected behavior.
        recommended = "forced_soft_npu_preset"
        reason = (
            f"Device '{target_device}' has an NPU, but quantization_mode={quantization_mode} "
            "was chosen for training. The NPU requires quantization: 2 (TI NPU-optimized) "
            "to accelerate inference. Using 'forced_soft_npu_preset' to ensure the model "
            "runs on the CPU path. To use the NPU, re-train with quantization: 2."
        )

    elif task_prefers_soft:
        # NPU device + TI quantization, but task type works better on software path
        recommended = "forced_soft_npu_preset"
        reason = (
            f"'{task_type}' tasks produce better results on the software inference path "
            "even on NPU-equipped devices. Using 'forced_soft_npu_preset'."
        )

    else:
        # NPU device + quantization: 2 (or unknown) + standard task → use NPU
        recommended = "default_preset"
        reason = (
            f"Device '{target_device}' has an NPU and quantization: 2 was selected. "
            "The 'default_preset' will use the hardware NPU for accelerated inference. "
            "Use 'compress_npu_layer_data' instead if flash/SRAM is tight."
        )

    preset_details = [
        {
            "name": p,
            "label": COMPILE_PRESETS[p]["label"],
            "description": COMPILE_PRESETS[p]["description"],
            "available_for_device": p in available,
        }
        for p in COMPILE_PRESETS
    ]

    return {
        "success": True,
        "recommended_preset": recommended,
        "recommendation_reason": reason,
        "device_has_npu": device_has_npu,
        "available_presets": available,
        "preset_details": preset_details,
        "context_paths": CONTEXT_PATHS,
    }


# ─── Tool 2: Validate ─────────────────────────────────────────────────────────

def validate_compilation_section(
    enable: bool = True,
    compile_preset_name: Optional[str] = None,
    target_device: Optional[str] = None,
    model_path: Optional[str] = None,
    compile_output_path: Optional[str] = None,
    keep_libc_files: bool = False,
) -> Dict[str, Any]:
    """
    Tool: Validate parameters for the 'compilation' section of tiny ML config.yaml.

    **Usage**
    The agent should ask the user:
    1. Do you want to compile the trained model for the target device? (enable)
       - Almost always Yes (True) — skip only for pure training experiments.
    2. Which compilation preset? (compile_preset_name)
       - Call get_compilation_preset_recommendations first to get the right suggestion.
       - If user doesn't have a preference, use the recommended preset.
    3. Do you want a custom output path for compiled artifacts? (compile_output_path)
       - Optional. Default output goes to the run directory.
    4. Keep libc files in the output? (keep_libc_files)
       - Only needed for advanced integration scenarios. Default: False.

    For BYOM (Bring Your Own Model) compilation, model_path points to the ONNX model.

    **Returns**
    Validation result without YAML.
    """
    is_valid, errors = CompilationSectionValidator.validate(
        enable, compile_preset_name, target_device, model_path
    )
    if not is_valid:
        return {"success": False, "config": None, "errors": errors}

    config = CompilationSectionConfig(
        enable=enable,
        compile_preset_name=compile_preset_name,
        model_path=model_path,
        compile_output_path=compile_output_path,
        keep_libc_files=keep_libc_files,
    )
    return {"success": True, "config": config.to_dict(), "errors": []}


# ─── Tool 3: Generate YAML ────────────────────────────────────────────────────

def generate_compilation_section_yaml(
    enable: bool = True,
    compile_preset_name: Optional[str] = None,
    target_device: Optional[str] = None,
    model_path: Optional[str] = None,
    compile_output_path: Optional[str] = None,
    keep_libc_files: bool = False,
) -> Dict[str, Any]:
    """
    Tool: Generate YAML for the 'compilation' section of tiny ML config.yaml.

    **Usage**
    Call this after the user confirms their compilation settings.
    Only pass parameters the user explicitly specified — omitted optional params
    will use system defaults from tinyml_modelmaker/ai_modules/{target_module}/params.py.

    For a standard run, a minimal config is often sufficient:
        compilation: {}
    or with just a preset:
        compilation:
          compile_preset_name: forced_soft_npu_preset

    **Returns**
    - yaml: ready-to-use YAML string for the compilation section
    - config: dict representation
    - errors: validation errors (empty if success)
    """
    is_valid, errors = CompilationSectionValidator.validate(
        enable, compile_preset_name, target_device, model_path
    )
    if not is_valid:
        return {"success": False, "yaml": None, "config": None, "errors": errors}

    config = CompilationSectionConfig(
        enable=enable,
        compile_preset_name=compile_preset_name,
        model_path=model_path,
        compile_output_path=compile_output_path,
        keep_libc_files=keep_libc_files,
    )
    return {
        "success": True,
        "yaml": config.to_yaml_string(),
        "config": config.to_dict(),
        "errors": [],
    }
