from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Union
import os
import io
import csv
import zipfile
import yaml
from constants import CONTEXT_PATHS
import feature_schema
# ─── Output shape computation ─────────────────────────────────────────────────

def compute_output_shape(
    feature_extraction_name: Optional[str],
    feat_ext_transform: Optional[List[str]],
    frame_size: Optional[int],
    feature_size_per_frame: Optional[int],
    num_frame_concat: Optional[int],
    stacking: Optional[str],
    variables: Optional[int],
    task_type: Optional[str] = None,
    forecast_horizon: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analytically compute the feature extraction output shape from config params.

    Returns a dict with:
        shape: tuple describing the output tensor shape (excluding batch dim)
        description: human-readable explanation
        total_features: total number of float values per sample
    """
    # Resolve variables
    num_vars = variables if isinstance(variables, int) else (len(variables) if isinstance(variables, list) else 1)

    # Forecasting: output is raw (frame_size, variables) — no feat_ext applied
    if task_type and "forecasting" in task_type:
        if frame_size is None:
            return {"shape": None, "description": "Cannot compute: frame_size required for forecasting.", "total_features": None}
        fh = forecast_horizon or 1
        shape = (frame_size, num_vars)
        return {
            "shape": shape,
            "description": f"Forecasting input: {frame_size} time steps x {num_vars} variable(s). Forecast horizon: {fh}.",
            "total_features": frame_size * num_vars,
        }

    # Named preset: look up shape from schema
    if feature_extraction_name:
        fs = frame_size
        fspf = feature_size_per_frame
        nfc = num_frame_concat
        stk = stacking
        cfg = feature_schema.get_preset(feature_extraction_name)
        if cfg:
            fs = frame_size or cfg.get("frame_size")
            fspf = feature_size_per_frame or cfg.get("feature_size_per_frame")
            nfc = num_frame_concat or cfg.get("num_frame_concat")
            stk = stacking or cfg.get("stacking")

        if feature_extraction_name == "Custom_Default" or not fspf:
            # Raw data: shape = (frame_size, variables)
            if fs is None:
                return {"shape": None, "description": "Cannot compute: frame_size required for Custom_Default.", "total_features": None}
            shape = (fs, num_vars)
            return {
                "shape": shape,
                "description": f"Custom_Default (no feat ext): raw window of {fs} x {num_vars} variable(s).",
                "total_features": fs * num_vars,
            }

        return _compute_shape_from_params(fs, fspf, nfc, stk, num_vars, feature_extraction_name)

    # Custom pipeline: use explicit params
    if feat_ext_transform or feature_size_per_frame:
        if not frame_size:
            return {"shape": None, "description": "Cannot compute: frame_size required.", "total_features": None}

        fspf = feature_size_per_frame
        nfc = num_frame_concat or 1
        stk = stacking or "2D1"

        if feat_ext_transform and "RAW_FE" in feat_ext_transform:
            fspf = fspf or frame_size

        if not fspf:
            if feat_ext_transform and ("FFT_FE" in feat_ext_transform or "FFT_Q15" in feat_ext_transform):
                fspf = frame_size // 2  # rough estimate: pos half of FFT
            else:
                fspf = frame_size

        return _compute_shape_from_params(frame_size, fspf, nfc, stk, num_vars, "custom pipeline")

    # Fallback: no feat_ext, raw window
    if frame_size:
        return {
            "shape": (frame_size, num_vars),
            "description": f"No feature extraction: raw window of {frame_size} × {num_vars} variable(s).",
            "total_features": frame_size * num_vars,
        }

    return {"shape": None, "description": "Insufficient parameters to compute output shape.", "total_features": None}


def _compute_shape_from_params(
    frame_size: Optional[int],
    feature_size_per_frame: int,
    num_frame_concat: int,
    stacking: Optional[str],
    num_vars: int,
    source_name: str,
) -> Dict[str, Any]:
    total = feature_size_per_frame * num_frame_concat
    if stacking == "2D1":
        shape = (feature_size_per_frame, num_frame_concat, num_vars)
        desc = f"{source_name}: {feature_size_per_frame} features × {num_frame_concat} frames × {num_vars} channel(s) [2D1 stacking]."
    elif stacking == "1D":
        shape = (total, num_vars)
        desc = f"{source_name}: {total} features ({feature_size_per_frame}×{num_frame_concat}) × {num_vars} channel(s) [1D stacking]."
    else:
        shape = (total * num_vars,)
        desc = f"{source_name}: {total * num_vars} total features (flat)."
    return {"shape": shape, "description": desc, "total_features": total * num_vars}


# ─── Data sampling ────────────────────────────────────────────────────────────

def _load_data_sample(data_path: str, max_rows: int = 5) -> Tuple[bool, Optional[List[str]], Optional[str]]:
    """
    Load a small sample from a local CSV or ZIP file to inspect column names.

    Returns: (success, column_names_or_None, error_message_or_None)
    """
    if not os.path.exists(data_path):
        return False, None, f"Path does not exist: {data_path}"

    try:
        if data_path.lower().endswith(".csv"):
            with open(data_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            if header is None:
                return False, None, "CSV file is empty."
            return True, header, None

        if data_path.lower().endswith(".zip"):
            with zipfile.ZipFile(data_path, "r") as z:
                csv_files = [n for n in z.namelist() if n.lower().endswith(".csv") and not os.path.basename(n).startswith(".")]
                if not csv_files:
                    return False, None, "No CSV files found in ZIP archive."
                with z.open(csv_files[0]) as f:
                    reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                    header = next(reader, None)
            if header is None:
                return False, None, f"First CSV in ZIP ({csv_files[0]}) is empty."
            return True, header, None

        # Try reading as plain CSV regardless of extension
        with open(data_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header is None:
            return False, None, "File appears empty."
        return True, header, None

    except Exception as e:
        return False, None, str(e)


def _count_non_time_columns(columns: List[str]) -> int:
    """Estimate number of data columns after removing time/timestamp columns."""
    time_keywords = {"time", "timestamp", "date", "datetime", "t", "index", "idx", "sample"}
    return sum(1 for c in columns if c.strip().lower() not in time_keywords)


# ─── Tool 1: Recommendations ──────────────────────────────────────────────────

def get_data_proc_feat_ext_recommendations(
    task_type: str,
    prefer_fft: bool,
    need_full_spectrum: bool,
    need_temporal_ctx: bool,
    shortest_sample_len: int,
    variables: Optional[int] = None,
    sampling_rate: Optional[int] = None,
    new_sr: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tool: Get intelligently ranked recommendations for data processing & feature extraction.

    Args:
        task_type: Task type (e.g., 'motor_fault', 'arc_fault', 'generic_timeseries_classification').
        shortest_sample_len: The SHORTEST sample length (rows) measured in the dataset.
            Presets with frame_size > shortest_sample_len are excluded — those samples
            would be silently skipped during training, causing data loss.
        variables: Number of sensor channels. Filters to compatible presets.
        sampling_rate: Original data sampling rate in Hz. Triggers DownSample suggestions.
        new_sr: Target sampling rate if downsampling desired.
        prefer_fft: True = prefer FFT presets, False = prefer RAW.

    Returns:
        ranked_presets_detailed: all compatible presets ranked by score (frame_size <= shortest_sample_len enforced)
        required_data_proc, optional_data_proc_transforms, downsample_suggestion, context_paths
    """
    try:

        task_recs = feature_schema.get_task_recommendations(
            task_type
        )

        from preset_ranker import PresetRanker

        # === INTELLIGENT RANKING ===
        ranker = PresetRanker()
        ranked_presets = ranker.rank_presets(
            task_type=task_type,
            variables=variables,
            sampling_rate=sampling_rate,
            prefer_fft=prefer_fft,
            need_full_spectrum=need_full_spectrum,
            need_temporal_ctx=need_temporal_ctx,
            shortest_sample_len=shortest_sample_len
        )


        # === DOWNSAMPLING SUGGESTION ===
        downsample_suggestion = None
        if sampling_rate and sampling_rate > 10000:
            suggested_new_sr = new_sr or (sampling_rate // 100)
            downsample_suggestion = (
                f"Your sampling rate ({sampling_rate} Hz) is high. Consider adding 'DownSample' "
                f"with new_sr={suggested_new_sr} to reduce data size and focus on relevant frequencies."
            )

        # === OPTIONAL TRANSFORMS ===
        optional_with_desc = []
        required_data_proc = []

        if task_recs:
            optional_transforms = task_recs.get("optional_data_proc", [])

            optional_with_desc = []
            for t in optional_transforms:
                aug_def = feature_schema.get_augmenter(t)
                transform_def = feature_schema.get_transform(t)
                desc = (aug_def or transform_def or {}).get("description", "")
                optional_with_desc.append({"name": t, "description": desc})

            required_data_proc = [
                {
                    "transform": t,
                    "reason": task_recs.get("required_reason", "Required for this task type."),
                }
                for t in task_recs.get("required_data_proc", [])
            ]

        return {
            "success": True,
            "task_type": task_type,
            "required_data_proc": required_data_proc,
            "optional_data_proc_transforms": optional_with_desc,
            "downsample_suggestion": downsample_suggestion,
            "ranked_presets_detailed": ranked_presets,
            "preset_ranking_factors": {
                "task_type": task_type,
                "variables": variables,
                "prefer_fft": prefer_fft,
            },
            "all_preset_names": feature_schema.list_all_presets(),
            "custom_pipeline_option": (
                "Instead of a preset, you can specify feat_ext_transform as a list of steps "
                "(e.g., ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT']) "
                "along with frame_size, feature_size_per_frame, num_frame_concat, and stacking. "
                "Use feature_extraction_name='Custom_<YourName>' to identify a custom pipeline."
            ),
            "context_paths": {
                "fe_transforms_reference": CONTEXT_PATHS["fe_transforms_doc"],
                "data_proc_reference":     CONTEXT_PATHS["data_proc_doc"],
                "examples_directory":      CONTEXT_PATHS["examples_dir"],
            },
        }
    except Exception as e:
        print(e)
        raise


# ─── Tool 2: Validate data shape ──────────────────────────────────────────────

def validate_feat_ext_data_shape(
    data_path: str,
    task_type: str,
    variables: Union[int, List],
    data_proc_transforms: Optional[List[str]] = None,
    frame_size: Optional[int] = None,
    stride_size: Optional[float] = None,
    feature_extraction_name: Optional[str] = None,
    feat_ext_transform: Optional[List[str]] = None,
    feature_size_per_frame: Optional[int] = None,
    num_frame_concat: Optional[int] = None,
    stacking: Optional[str] = None,
    model_name: Optional[str] = None,
    forecast_horizon: Optional[int] = None,
    sampling_rate: Optional[int] = None,
    new_sr: Optional[int] = None,
    target_variables: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Tool: Validate that the data shape produced by the configured transforms matches
    what the selected model expects.

    Steps:
    1. Loads a sample of the data from data_path (local CSV or ZIP only).
       For remote URLs, column validation is skipped but shape computation still runs.
    2. Checks that the actual number of data columns is compatible with `variables`.
    3. Analytically computes the feat_ext output shape from the provided parameters.
    4. If model_name is provided, looks for a matching example in tinyml-modelzoo
       and compares that example's feat_ext config against the provided config.
    5. Returns all findings with warnings and errors.

    Args:
        data_path: Local path to CSV or ZIP file (or URL — column check skipped for URLs).
        task_type: Task type string (e.g., 'motor_fault', 'generic_timeseries_forecasting').
        variables: Number of sensor channels, or list of column indices/names.
        data_proc_transforms: List of data processing transforms.
        frame_size: Number of samples per window.
        stride_size: Window stride as a fraction of frame_size.
        feature_extraction_name: Named preset (e.g., 'Generic_1024Input_FFTBIN_64Feature_8Frame').
        feat_ext_transform: Custom feature extraction pipeline steps.
        feature_size_per_frame: Output features per frame (for custom pipelines).
        num_frame_concat: Number of frames concatenated per sample.
        stacking: '2D1', '1D', or None.
        model_name: Model name from training section (used to find matching example).
        forecast_horizon: Number of future timesteps for forecasting tasks.
        sampling_rate: Original sampling rate (Hz).
        new_sr: Target sampling rate after downsampling.
        target_variables: Column indices/names to forecast (forecasting tasks only).

    Returns:
        Dict with:
        - data_columns_found: actual CSV column names (if readable)
        - data_column_count: total columns in data
        - non_time_column_count: estimated data columns after removing time columns
        - variables_requested: the variables param value
        - variables_check: 'ok', 'warning', or 'error' with a message
        - computed_output_shape: analytically computed feat_ext output shape
        - model_example_config: feat_ext config from the matching modelzoo example (if found)
        - shape_consistency: 'ok', 'warning', or 'error' with a message
        - param_validation: list of parameter validation results
        - warnings: list of non-fatal issues
        - errors: list of fatal issues
        - success: True if no errors (warnings are acceptable)
    """
    errors = []
    warnings = []
    param_validation = []

    # ── 1. Variables normalisation ────────────────────────────────────────────
    if isinstance(variables, list):
        num_vars = len(variables)
    elif isinstance(variables, int):
        num_vars = variables
    else:
        num_vars = 1
        warnings.append("variables not specified; defaulting to 1.")

    # ── 2. Data column check ──────────────────────────────────────────────────
    data_columns_found = None
    data_column_count = None
    non_time_column_count = None
    variables_check = {"status": "skipped", "message": "Data path is a URL or not local — column check skipped."}

    is_url = data_path.startswith("http://") or data_path.startswith("https://")
    if not is_url:
        ok, columns, err = _load_data_sample(data_path)
        if not ok:
            warnings.append(f"Could not read data sample: {err}")
        else:
            data_columns_found = columns
            data_column_count = len(columns)
            non_time_column_count = _count_non_time_columns(columns)

            if non_time_column_count >= num_vars:
                variables_check = {
                    "status": "ok",
                    "message": f"Data has {non_time_column_count} non-time columns; variables={num_vars} is compatible.",
                }
            else:
                msg = (
                    f"Data has only {non_time_column_count} non-time columns "
                    f"but variables={num_vars} was requested."
                )
                errors.append(msg)
                variables_check = {"status": "error", "message": msg}

    # ── 3. Parameter validation ───────────────────────────────────────────────
    is_forecasting = "forecasting" in task_type

    # SimpleWindow required for forecasting
    if is_forecasting:
        if not data_proc_transforms or "SimpleWindow" not in data_proc_transforms:
            errors.append("'SimpleWindow' must be in data_proc_transforms for forecasting tasks.")
            param_validation.append({
                "param": "data_proc_transforms",
                "status": "error",
                "message": "SimpleWindow is required for forecasting.",
            })
        if frame_size is None:
            errors.append("frame_size is required for forecasting tasks.")
            param_validation.append({"param": "frame_size", "status": "error", "message": "Required for forecasting."})
        if forecast_horizon is None:
            warnings.append("forecast_horizon not specified; defaults to 1.")
            param_validation.append({"param": "forecast_horizon", "status": "warning", "message": "Not specified; defaults to 1."})

    # SimpleWindow + frame_size consistency
    if data_proc_transforms and "SimpleWindow" in data_proc_transforms:
        if frame_size is None:
            errors.append("frame_size is required when SimpleWindow is in data_proc_transforms.")
            param_validation.append({"param": "frame_size", "status": "error", "message": "Required with SimpleWindow."})
        if stride_size is None:
            warnings.append("stride_size not specified; defaults will be used (check params.py).")
            param_validation.append({"param": "stride_size", "status": "warning", "message": "Not specified; defaults apply."})

    # DownSample consistency
    if data_proc_transforms and "DownSample" in data_proc_transforms:
        if sampling_rate is None:
            errors.append("sampling_rate is required when DownSample is in data_proc_transforms.")
            param_validation.append({"param": "sampling_rate", "status": "error", "message": "Required for DownSample."})
        if new_sr is None:
            errors.append("new_sr is required when DownSample is in data_proc_transforms.")
            param_validation.append({"param": "new_sr", "status": "error", "message": "Required for DownSample."})
        elif sampling_rate and new_sr and new_sr >= sampling_rate:
            errors.append(f"new_sr ({new_sr}) must be less than sampling_rate ({sampling_rate}).")
            param_validation.append({"param": "new_sr", "status": "error", "message": "new_sr must be less than sampling_rate."})

    # Preset vs custom pipeline
    if feature_extraction_name and not feature_extraction_name.startswith("Custom_"):
        preset_cfg = feature_schema.get_preset(feature_extraction_name)
        if not preset_cfg:
            warnings.append(
                f"Preset '{feature_extraction_name}' is not in the known presets list. "
                "It may be valid — check tinyml-modelmaker constants.py for full list."
            )

    if feature_extraction_name and not feature_extraction_name.startswith("Custom_"):
        preset_cfg = feature_schema.get_preset(feature_extraction_name)
        if preset_cfg:
            preset_vars = preset_cfg.get("variables")
            if preset_vars and num_vars != preset_vars:
                warnings.append(
                    f"Preset '{feature_extraction_name}' is designed for {preset_vars} variable(s), "
                    f"but variables={num_vars} is configured. This may still work but is non-standard."
                )
                param_validation.append({
                    "param": "variables",
                    "status": "warning",
                    "message": f"Preset expects {preset_vars} variable(s), got {num_vars}.",
                })

    # ── 4. Compute output shape ───────────────────────────────────────────────
    shape_info = compute_output_shape(
        feature_extraction_name=feature_extraction_name,
        feat_ext_transform=feat_ext_transform,
        frame_size=frame_size,
        feature_size_per_frame=feature_size_per_frame,
        num_frame_concat=num_frame_concat,
        stacking=stacking,
        variables=num_vars,
        task_type=task_type,
        forecast_horizon=forecast_horizon,
    )

    # ── 5. Find matching modelzoo example for the selected model ──────────────
    model_example_config = None
    shape_consistency = {"status": "skipped", "message": "No model_name provided; skipping example comparison."}

    if model_name:
        from model_selection_tools import ExampleFinder
        examples_root = ExampleFinder.find_examples_path()
        if examples_root and os.path.isdir(examples_root):
            for item in os.listdir(examples_root):
                example_dir = os.path.join(examples_root, item)
                config_path = os.path.join(example_dir, "config.yaml")
                if not os.path.exists(config_path):
                    continue
                try:
                    with open(config_path) as f:
                        cfg = yaml.safe_load(f)
                    ex_model = (cfg.get("training") or {}).get("model_name")
                    if ex_model == model_name:
                        model_example_config = cfg.get("data_processing_feature_extraction", {})
                        # Compare feat_ext_name or frame_size
                        ex_fe_name = model_example_config.get("feature_extraction_name")
                        if ex_fe_name and feature_extraction_name and ex_fe_name != feature_extraction_name:
                            warnings.append(
                                f"The closest modelzoo example for '{model_name}' uses "
                                f"feature_extraction_name='{ex_fe_name}', but you selected "
                                f"'{feature_extraction_name}'. Both may work, but verify shape compatibility."
                            )
                            shape_consistency = {
                                "status": "warning",
                                "message": f"Example uses '{ex_fe_name}'; your config uses '{feature_extraction_name}'.",
                            }
                        else:
                            shape_consistency = {
                                "status": "ok",
                                "message": f"Feature extraction config matches example for model '{model_name}'.",
                            }
                        break
                except Exception:
                    continue

        if model_example_config is None and model_name:
            shape_consistency = {
                "status": "warning",
                "message": f"No example found for model '{model_name}' in modelzoo. Cannot compare config.",
            }

    return {
        "success": len(errors) == 0,
        "data_columns_found": data_columns_found,
        "data_column_count": data_column_count,
        "non_time_column_count": non_time_column_count,
        "variables_requested": variables,
        "variables_check": variables_check,
        "computed_output_shape": shape_info,
        "model_example_config": model_example_config,
        "shape_consistency": shape_consistency,
        "param_validation": param_validation,
        "warnings": warnings,
        "errors": errors,
    }


# ─── Config dataclass and YAML generator ──────────────────────────────────────

@dataclass
class DataProcFeatExtConfig:
    """Validated data_processing_feature_extraction section configuration."""
    data_proc_transforms: Optional[List[str]] = None
    sampling_rate: Optional[int] = None
    new_sr: Optional[int] = None
    frame_size: Optional[int] = None
    stride_size: Optional[float] = None
    feature_extraction_name: Optional[str] = None
    variables: Optional[Union[int, List]] = None
    feat_ext_transform: Optional[List[str]] = None
    feature_size_per_frame: Optional[int] = None
    num_frame_concat: Optional[int] = None
    stacking: Optional[str] = None
    normalize_bin: Optional[bool] = None
    offset: Optional[int] = None
    scale: Optional[float] = None
    frame_skip: Optional[int] = None
    log_mul: Optional[float] = None
    log_base: Optional[Union[int, str]] = None
    log_threshold: Optional[float] = None
    forecast_horizon: Optional[int] = None
    target_variables: Optional[List] = None

    def to_dict(self) -> Dict:
        result = {}
        if self.data_proc_transforms is not None:
            result["data_proc_transforms"] = self.data_proc_transforms
        if self.sampling_rate is not None:
            result["sampling_rate"] = self.sampling_rate
        if self.new_sr is not None:
            result["new_sr"] = self.new_sr
        if self.frame_size is not None:
            result["frame_size"] = self.frame_size
        if self.stride_size is not None:
            result["stride_size"] = self.stride_size
        if self.forecast_horizon is not None:
            result["forecast_horizon"] = self.forecast_horizon
        if self.target_variables is not None:
            result["target_variables"] = self.target_variables
        if self.feature_extraction_name is not None:
            result["feature_extraction_name"] = self.feature_extraction_name
        if self.variables is not None:
            result["variables"] = self.variables
        if self.feat_ext_transform is not None:
            result["feat_ext_transform"] = self.feat_ext_transform
        if self.feature_size_per_frame is not None:
            result["feature_size_per_frame"] = self.feature_size_per_frame
        if self.num_frame_concat is not None:
            result["num_frame_concat"] = self.num_frame_concat
        if self.stacking is not None:
            result["stacking"] = self.stacking
        if self.normalize_bin is not None:
            result["normalize_bin"] = int(self.normalize_bin)
        if self.offset is not None:
            result["offset"] = self.offset
        if self.scale is not None:
            result["scale"] = self.scale
        if self.frame_skip is not None:
            result["frame_skip"] = self.frame_skip
        if self.log_mul is not None:
            result["log_mul"] = self.log_mul
        if self.log_base is not None:
            result["log_base"] = self.log_base
        if self.log_threshold is not None:
            result["log_threshold"] = self.log_threshold
        return result

    def to_yaml_string(self) -> str:
        d = self.to_dict()
        section = {"data_processing_feature_extraction": d}
        return yaml.dump(section, default_flow_style=False, sort_keys=False).rstrip()


# ─── Tool 4: Generate YAML ────────────────────────────────────────────────────

def generate_feat_ext_section_yaml(
    task_type: str,
    variables: Optional[Union[int, List]] = None,
    data_proc_transforms: Optional[List[str]] = None,
    sampling_rate: Optional[int] = None,
    new_sr: Optional[int] = None,
    frame_size: Optional[int] = None,
    stride_size: Optional[float] = None,
    feature_extraction_name: Optional[str] = None,
    feat_ext_transform: Optional[List[str]] = None,
    feature_size_per_frame: Optional[int] = None,
    num_frame_concat: Optional[int] = None,
    stacking: Optional[str] = None,
    normalize_bin: Optional[bool] = None,
    offset: Optional[int] = None,
    scale: Optional[float] = None,
    frame_skip: Optional[int] = None,
    log_mul: Optional[float] = None,
    log_base: Optional[Union[int, str]] = None,
    log_threshold: Optional[float] = None,
    forecast_horizon: Optional[int] = None,
    target_variables: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Tool: Generate YAML for the 'data_processing_feature_extraction' config section.

    Validates all parameters and returns ready-to-use YAML.

    Only pass parameters that the user explicitly specified or that were agreed upon.
    Optional parameters omitted from the call will be excluded from the YAML output,
    and system defaults will apply at runtime.

    Args:
        task_type: Required for validation (e.g., 'motor_fault', 'generic_timeseries_forecasting').
        variables: Number of sensor channels, or list of column indices/names.
        data_proc_transforms: e.g., ['SimpleWindow', 'DownSample'].
        sampling_rate: Original sampling rate (Hz). Required if DownSample is used.
        new_sr: Target sampling rate. Required if DownSample is used.
        frame_size: Samples per window. Required if SimpleWindow is used.
        stride_size: Window stride as fraction (0.0–1.0). E.g., 0.1 = 10% overlap.
        feature_extraction_name: Named preset or 'Custom_<name>'.
        feat_ext_transform: Custom pipeline steps (used with Custom_* presets).
        feature_size_per_frame: Output features per frame (custom pipelines).
        num_frame_concat: Number of frames concatenated per sample.
        stacking: '2D1' or '1D'.
        normalize_bin: Whether to normalize frequency bins.
        offset: Frame offset for overlap control.
        scale: Signal scale factor.
        frame_skip: Number of frames to skip between samples.
        log_mul: Logarithm multiplier (e.g., 20 for dB).
        log_base: Logarithm base (e.g., 10 or 'e').
        log_threshold: Minimum value to avoid log(0).
        forecast_horizon: Future timesteps to predict (forecasting only).
        target_variables: Columns to predict (forecasting only).

    Returns:
        Dict with:
        - success: bool
        - yaml: YAML string ready to write to config file
        - config: dict representation of the section
        - computed_output_shape: analytically computed feature shape
        - errors: list of validation error strings
        - warnings: list of non-fatal issue strings
    """
    errors = []
    warnings = []

    is_forecasting = "forecasting" in task_type

    # Validate required params
    if is_forecasting:
        if not data_proc_transforms or "SimpleWindow" not in data_proc_transforms:
            errors.append("'SimpleWindow' must be in data_proc_transforms for forecasting tasks.")
        if frame_size is None:
            errors.append("frame_size is required for forecasting tasks.")
        if forecast_horizon is None:
            warnings.append("forecast_horizon not specified; will default to 1.")
        if target_variables is None:
            warnings.append("target_variables not specified; model will predict all variables by default.")

    if data_proc_transforms and "SimpleWindow" in data_proc_transforms and frame_size is None:
        errors.append("frame_size is required when SimpleWindow is in data_proc_transforms.")

    if data_proc_transforms and "DownSample" in data_proc_transforms:
        if sampling_rate is None:
            errors.append("sampling_rate is required when DownSample is in data_proc_transforms.")
        if new_sr is None:
            errors.append("new_sr is required when DownSample is in data_proc_transforms.")
        elif sampling_rate and new_sr and new_sr >= sampling_rate:
            errors.append(f"new_sr ({new_sr}) must be less than sampling_rate ({sampling_rate}).")

    if feature_extraction_name and feature_extraction_name not in feature_schema.list_all_presets() and not feature_extraction_name.startswith("Custom_"):
        warnings.append(
            f"'{feature_extraction_name}' is not in the known presets list. "
            "Verify the name against tinyml-modelmaker constants.py."
        )

    if errors:
        return {"success": False, "yaml": None, "config": None, "computed_output_shape": None, "errors": errors, "warnings": warnings}

    config = DataProcFeatExtConfig(
        data_proc_transforms=data_proc_transforms,
        sampling_rate=sampling_rate,
        new_sr=new_sr,
        frame_size=frame_size,
        stride_size=stride_size,
        feature_extraction_name=feature_extraction_name,
        variables=variables,
        feat_ext_transform=feat_ext_transform,
        feature_size_per_frame=feature_size_per_frame,
        num_frame_concat=num_frame_concat,
        stacking=stacking,
        normalize_bin=normalize_bin,
        offset=offset,
        scale=scale,
        frame_skip=frame_skip,
        log_mul=log_mul,
        log_base=log_base,
        log_threshold=log_threshold,
        forecast_horizon=forecast_horizon,
        target_variables=target_variables,
    )

    num_vars = variables if isinstance(variables, int) else (len(variables) if isinstance(variables, list) else None)
    shape_info = compute_output_shape(
        feature_extraction_name=feature_extraction_name,
        feat_ext_transform=feat_ext_transform,
        frame_size=frame_size,
        feature_size_per_frame=feature_size_per_frame,
        num_frame_concat=num_frame_concat,
        stacking=stacking,
        variables=num_vars,
        task_type=task_type,
        forecast_horizon=forecast_horizon,
    )

    return {
        "success": True,
        "yaml": config.to_yaml_string(),
        "config": config.to_dict(),
        "computed_output_shape": shape_info,
        "errors": [],
        "warnings": warnings,
    }
