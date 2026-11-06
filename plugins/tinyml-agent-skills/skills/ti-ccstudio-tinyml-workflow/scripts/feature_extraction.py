from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Union
import os
import io
import csv
import zipfile
import yaml
from constants import CONTEXT_PATHS
import feature_schema
import fe_rules as fr
# ─── Output shape computation ─────────────────────────────────────────────────

def _shape_description(source: str, shape, info: Dict[str, Any], num_frame_concat: int, num_vars: int) -> str:
    per_frame = info.get("per_frame")
    parts = f"{source}: model input per sample = {shape}"
    if per_frame is not None:
        parts += f" ({per_frame} features/frame x {num_frame_concat} frame(s)"
        parts += f" x {num_vars} variable(s) end to end)" if shape and shape[0] == 1 and num_vars > 1 else ")"
    return f"{parts}. {info.get('note', '')}".strip()


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
    min_bin: Optional[int] = None,
    window_count: Optional[int] = None,
    chunk_size: Optional[int] = None,
    stride_size: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Analytically compute the model input shape per sample (excluding the batch dim), as the dataset
    produces it: (channels, width, 1). 2D1 stacking: (V, features_per_frame*num_frame_concat, 1);
    1D: (1, features_per_frame*num_frame_concat*V, 1); no feature extraction: (V, frame_size, 1).

    Returns a dict with: shape, description, total_features.
    """
    num_vars = variables if isinstance(variables, int) else (len(variables) if isinstance(variables, list) else 1)

    # Forecasting runs no feature extraction: the model sees (V, frame_size, 1) windows.
    if task_type and "forecasting" in task_type:
        if frame_size is None:
            return {"shape": None, "description": "Cannot compute: frame_size required for forecasting.", "total_features": None}
        fh = forecast_horizon or 1
        return {
            "shape": (num_vars, frame_size, 1),
            "description": f"Forecasting input: {num_vars} variable(s) x {frame_size} time steps (no feature extraction). Forecast horizon: {fh}.",
            "total_features": frame_size * num_vars,
        }

    # Resolve parameters: explicit values win over the preset's (config values override presets in modelmaker).
    source, transforms = "custom pipeline", list(feat_ext_transform or [])
    F, B, C, stk = frame_size, feature_size_per_frame, num_frame_concat, stacking
    m, wc, cs, st = min_bin, window_count, chunk_size, stride_size
    if feature_extraction_name:
        cfg = feature_schema.get_preset(feature_extraction_name)
        if cfg:
            source = feature_extraction_name
            transforms = list(feat_ext_transform) if feat_ext_transform is not None else list(cfg.get("feat_ext_transform") or [])
            F = frame_size or cfg.get("frame_size")
            B = feature_size_per_frame or cfg.get("feature_size_per_frame")
            C = num_frame_concat or cfg.get("num_frame_concat")
            stk = stacking or cfg.get("stacking")
            m = min_bin if min_bin is not None else cfg.get("min_bin", 1)
            wc, cs, st = wc or cfg.get("window_count"), cs or cfg.get("chunk_size"), st if st is not None else cfg.get("stride_size")
    if not F:
        return {"shape": None, "description": "Cannot compute: frame_size required.", "total_features": None}
    C = C or 1
    info = fr.output_shape(transforms, F, C, num_vars, stk or "2D1", B, 1 if m is None else m, wc or 1, cs or 1, st)
    shape = info["shape"]
    if shape is None:
        return {"shape": None, "description": f"{source}: {info['note']}.", "total_features": None}
    total = 1
    for dim in shape:
        total *= dim
    return {"shape": shape, "description": _shape_description(source, shape, info, C, num_vars), "total_features": total}


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
    """Data columns modelmaker keeps: it drops a column only when its first-row text contains 'time'
    (case-insensitive). A header-less file has numbers there, so nothing is dropped. Names such as
    'index' or 'sample' are NOT dropped and would count as variables."""
    return sum(1 for c in columns if "time" not in str(c).lower())


def _custom_pipeline_issues(feature_extraction_name, feat_ext_transform):
    """(errors, warnings) for a custom pipeline: unknown transform steps and an unregistered Custom_ name."""
    errors, warnings = [], []
    if feat_ext_transform:
        known = feature_schema.known_transform_names()
        unknown = [t for t in feat_ext_transform if t not in known]
        if unknown:
            errors.append(
                f"Unknown feat_ext_transform step(s) {unknown}: training silently ignores names it does not know. "
                f"Use these: {', '.join(known)}."
            )
    if (feature_extraction_name and feature_extraction_name.startswith("Custom_")
            and feature_extraction_name not in feature_schema.list_all_presets()):
        warnings.append(
            f"'{feature_extraction_name}' is not a registered preset, so modelmaker ignores the name. "
            "For a custom pipeline leave feature_extraction_name unset."
        )
    return errors, warnings


def _data_proc_issues(data_proc_transforms, sampling_rate, new_sr):
    """(errors, warnings) for data_proc_transforms: wrong spelling (silently ignored) and Downsample inputs."""
    errors, warnings = [], []
    for name in data_proc_transforms or []:
        if name != "SimpleWindow" and name != "Downsample" and name.lower() in ("downsample", "simplewindow"):
            errors.append(f"'{name}' is misspelled: the exact names are 'Downsample' and 'SimpleWindow'; anything else is silently ignored.")
        elif name not in ("SimpleWindow", "Downsample"):
            warnings.append(f"data_proc_transforms entry '{name}' is not acted on by the dataset (only 'SimpleWindow' and 'Downsample' are).")
    if data_proc_transforms and "Downsample" in data_proc_transforms:
        if sampling_rate is None:
            errors.append("sampling_rate is required when Downsample is in data_proc_transforms.")
        if new_sr is None:
            errors.append("new_sr is required when Downsample is in data_proc_transforms.")
        elif sampling_rate and new_sr > sampling_rate:
            errors.append(f"new_sr ({new_sr}) must not exceed sampling_rate ({sampling_rate}): the step would be 0 and the file is skipped.")
        elif sampling_rate and new_sr:
            k = int(sampling_rate // new_sr)
            if sampling_rate % new_sr:
                warnings.append(f"sampling_rate/new_sr = {sampling_rate / new_sr:.3g} is not an integer: every {k}-th row is kept, "
                                f"so the real rate is {sampling_rate / k:.6g} Hz, not {new_sr}.")
            if k == 1:
                warnings.append("Downsample with new_sr == sampling_rate keeps every row; drop it.")
    return errors, warnings


def _required_rows(transforms, frame_size, num_frame_concat, data_proc, sampling_rate, new_sr, stride_size,
                   task_type=None, forecast_horizon=None):
    """(min_rows, explanation) a file needs to yield one sample with these settings, or (None, '')."""
    if not frame_size:
        return None, ""
    k = fr.downsample_factor(data_proc, sampling_rate, new_sr)
    if task_type and "forecasting" in task_type:
        return (frame_size + (forecast_horizon or 1) - 1) * k + 1, f"frame_size {frame_size} + forecast_horizon {forecast_horizon or 1}" + (f" after Downsample x{k}" if k > 1 else "")
    stride = fr.window_stride_rows(frame_size, stride_size) if fr.uses_simple_window(data_proc) else None
    need = fr.min_rows_required(frame_size, num_frame_concat or 1, k, stride, has_fe=bool(transforms))
    why = f"frame_size {frame_size} x num_frame_concat {num_frame_concat or 1}"
    if stride is not None and (num_frame_concat or 1) > 1 and transforms:
        why = f"frame_size {frame_size} + (num_frame_concat {num_frame_concat} - 1) x SimpleWindow stride {stride}"
    if k > 1:
        why += f", after Downsample x{k}"
    return need, why


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
    target_device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: Get ranked recommendations for data processing & feature extraction.

    Args:
        task_type: Task type (e.g., 'motor_fault', 'arc_fault', 'generic_timeseries_classification').
        shortest_sample_len: Row count of the SHORTEST file in the dataset. A file yields a sample only if it has at
            least frame_size * num_frame_concat rows (after Downsample), so presets needing more than this are
            excluded: their short files would be skipped silently.
        variables: Number of sensor channels. Filters to compatible presets.
        sampling_rate: Original data sampling rate in Hz. Triggers Downsample suggestions and Hz figures.
        new_sr: Target sampling rate if downsampling is desired.
        prefer_fft: True = prefer FFT presets, False = prefer RAW.
        target_device: e.g. 'MSPM0G5187' or 'F28P55'. MSPM0 devices only run the Q15 fixed-point chain; other
            devices only the float chain. Omit to skip the device filter.

    Returns:
        ranked_presets_detailed (each with min_rows_required, input_shape, preset_notes),
        excluded_presets (each with the reason), required_data_proc, optional_data_proc_transforms,
        downsample_suggestion, frame_size_rule, context_paths
    """
    try:
        catalog_source = feature_schema.use_modelmaker_catalog(task_type)
        task_recs = feature_schema.get_task_recommendations(task_type)

        from preset_ranker import PresetRanker

        ranker = PresetRanker()
        ranked_presets, excluded_presets = ranker.rank_with_exclusions(
            task_type=task_type, variables=variables, sampling_rate=sampling_rate,
            prefer_fft=prefer_fft, need_full_spectrum=need_full_spectrum,
            need_temporal_ctx=need_temporal_ctx, shortest_sample_len=shortest_sample_len,
            target_device=target_device,
        )

        # === DOWNSAMPLING SUGGESTION ===
        downsample_suggestion = None
        if sampling_rate and sampling_rate > 10000:
            suggested_new_sr = new_sr or (sampling_rate // 100)
            k = max(int(sampling_rate // suggested_new_sr), 1)
            downsample_suggestion = (
                f"Your sampling rate ({sampling_rate} Hz) is high. Consider data_proc_transforms: ['Downsample'] "
                f"(exact spelling) with sampling_rate={sampling_rate}, new_sr={suggested_new_sr} (keeps every {k}-th row). "
                f"Rows shrink by {k}: the shortest file becomes {-(-shortest_sample_len // k)} rows, so re-run this tool with "
                f"shortest_sample_len={-(-shortest_sample_len // k)} and sampling_rate={suggested_new_sr}. Use an integer ratio."
            )

        # === OPTIONAL TRANSFORMS ===
        optional_with_desc = []
        required_data_proc = []

        if task_recs:
            for t in task_recs.get("optional_data_proc", []):
                aug_def = feature_schema.get_augmenter(t)
                transform_def = feature_schema.get_transform(t)
                desc = (aug_def or transform_def or {}).get("description", "")
                optional_with_desc.append({"name": t, "description": desc})

            required_data_proc = [
                {"transform": t, "reason": task_recs.get("required_reason", "Required for this task type.")}
                for t in task_recs.get("required_data_proc", [])
            ]

        # === ROWS-PER-FILE RULE ===
        excluded_names = {e["name"] for e in excluded_presets}
        no_sized_preset = not any(p.get("frame_size") for p in ranked_presets)
        frame_size_rule = (
            f"shortest_sample_len={shortest_sample_len} rows. A file produces a sample only if it has at least "
            f"frame_size x num_frame_concat rows (after Downsample; with SimpleWindow: frame_size + (num_frame_concat-1) x stride). "
            f"Choose frame_size x num_frame_concat <= {shortest_sample_len}. A frame that merely fits once is not enough "
            f"when num_frame_concat > 1: that file is still skipped silently. Presets that need more are in excluded_presets."
        )
        if no_sized_preset:
            frame_size_rule += (
                f" No ranked preset has a frame_size. Use a Custom_ pipeline with an explicit frame_size (a power of two) "
                f"and num_frame_concat such that frame_size x num_frame_concat <= {shortest_sample_len}."
            )

        return {
            "success": True,
            "task_type": task_type,
            "target_device": target_device,
            "device_family": fr.device_family(target_device),
            "preset_catalog_source": catalog_source,
            "shortest_sample_len": shortest_sample_len,
            "max_frame_size": shortest_sample_len,   # upper bound when num_frame_concat == 1; see frame_size_rule
            "frame_size_rule": frame_size_rule,
            "excluded_presets": excluded_presets,
            "required_data_proc": required_data_proc,
            "optional_data_proc_transforms": optional_with_desc,
            "downsample_suggestion": downsample_suggestion,
            "ranked_presets_detailed": ranked_presets,
            "preset_ranking_factors": {
                "task_type": task_type,
                "variables": variables,
                "prefer_fft": prefer_fft,
                "target_device": target_device,
            },
            "all_preset_names": [n for n in feature_schema.list_all_presets() if n not in excluded_names],
            "custom_pipeline_option": (
                "Instead of a preset, you can specify feat_ext_transform as a list of steps "
                "(e.g., ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT']) "
                "along with frame_size, feature_size_per_frame, num_frame_concat, stacking, "
                "log_mul/log_base/log_threshold (with LOG_DB: defaults 20, 10, 1e-100 apply if unset; setting them is safer), and data_proc_transforms (e.g. ['SimpleWindow']). "
                "Leave feature_extraction_name unset: a made-up name is not a registered preset and modelmaker ignores it."
            ),
            "context_paths": {
                "selection_rules": CONTEXT_PATHS["selection_rules_doc"],
                "arithmetic":      CONTEXT_PATHS["arithmetic_doc"],
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
    shortest_sample_len: Optional[int] = None,
    target_device: Optional[str] = None,
    min_bin: Optional[int] = None,
    analysis_bandwidth: Optional[int] = None,
    frame_skip: Optional[int] = None,
    offset: Optional[int] = None,
    log_threshold: Optional[float] = None,
    normalize_bin: Optional[bool] = None,
    dc_remove: Optional[bool] = None,
    log_mul: Optional[float] = None,
    log_base: Optional[Union[int, str]] = None,
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
        shortest_sample_len: Rows of the shortest file. Enforces: rows >= frame_size * num_frame_concat
            (after Downsample; with SimpleWindow frame_size + (num_frame_concat-1) * stride).
        target_device: e.g. 'MSPM0G5187', 'F28P55': checks the chain exists in that device's library.
        min_bin, analysis_bandwidth, frame_skip, offset, log_mul, log_base, log_threshold, normalize_bin, dc_remove:
            only needed to check a custom pipeline; presets carry their own values.

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
    feature_schema.use_modelmaker_catalog(task_type)
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
            step = fr.window_stride_rows(frame_size, None) if frame_size else None
            msg = ("stride_size not specified: the default 0.01 gives a step of "
                   f"{step} row(s) for frame_size {frame_size}" + (" (windows overlap almost completely, so the dataset becomes huge and "
                   "near-duplicate)" if step and frame_size and step < frame_size // 4 else "") +
                   ". Set stride_size explicitly (1.0 = no overlap, 0.5 = 50% overlap).")
            warnings.append(msg)
            param_validation.append({"param": "stride_size", "status": "warning", "message": msg})

    # Downsample consistency (and misspellings such as 'DownSample', which training silently ignores)
    dp_errors, dp_warnings = _data_proc_issues(data_proc_transforms, sampling_rate, new_sr)
    for msg in dp_errors:
        errors.append(msg)
        param_validation.append({"param": "data_proc_transforms", "status": "error", "message": msg})
    warnings.extend(dp_warnings)

    # Preset vs custom pipeline
    if feature_extraction_name and not feature_extraction_name.startswith("Custom_"):
        preset_cfg = feature_schema.get_preset(feature_extraction_name)
        if not preset_cfg:
            valid = ", ".join(feature_schema.list_all_presets())
            errors.append(
                f"'{feature_extraction_name}' is not a preset registered for task_type '{task_type}' in the installed "
                f"tinyml_modelmaker. Choose one of: {valid}, or use a Custom_ pipeline."
            )

    if feature_extraction_name and not feature_extraction_name.startswith("Custom_"):
        preset_cfg = feature_schema.get_preset(feature_extraction_name)
        if preset_cfg:
            preset_vars = preset_cfg.get("variables")
            if preset_vars and preset_vars > num_vars:
                msg = (f"Preset '{feature_extraction_name}' needs {preset_vars} channels but only {num_vars} "
                       "variable(s) are configured. Choose a preset with fewer channels or use more variables.")
                errors.append(msg)
                param_validation.append({"param": "variables", "status": "error", "message": msg})
            elif preset_vars and num_vars != preset_vars:
                warnings.append(
                    f"Preset '{feature_extraction_name}' is designed for {preset_vars} variable(s), "
                    f"but variables={num_vars} is configured. This may still work but is non-standard."
                )
                param_validation.append({
                    "param": "variables",
                    "status": "warning",
                    "message": f"Preset expects {preset_vars} variable(s), got {num_vars}.",
                })

    custom_errors, custom_warnings = _custom_pipeline_issues(feature_extraction_name, feat_ext_transform)
    errors.extend(custom_errors)
    warnings.extend(custom_warnings)

    # Effective settings: explicit values override the preset's (modelmaker applies the config last)
    named = feature_extraction_name if feature_extraction_name and not feature_extraction_name.startswith("Custom_") else None
    pcfg = (feature_schema.get_preset(named) if named else None) or {}

    def eff(explicit, key, default=None):
        return explicit if explicit is not None else pcfg.get(key, default)

    eff_transforms = list(feat_ext_transform) if feat_ext_transform is not None else list(pcfg.get("feat_ext_transform") or [])
    eff_dp = data_proc_transforms if data_proc_transforms is not None else (pcfg.get("data_proc_transforms") or [])
    eff_F = frame_size or pcfg.get("frame_size")
    eff_C = num_frame_concat or pcfg.get("num_frame_concat") or 1
    eff_min_bin = eff(min_bin, "min_bin", 1)

    # Rows per file: frame_size * num_frame_concat, not just frame_size
    need, why = _required_rows(eff_transforms, eff_F, eff_C, eff_dp, eff(sampling_rate, "sampling_rate"),
                               eff(new_sr, "new_sr"), eff(stride_size, "stride_size"), task_type, forecast_horizon)
    rows_check = None
    if need is not None:
        rows_check = {"min_rows_required": need, "because": why, "shortest_sample_len": shortest_sample_len}
        if shortest_sample_len and need > shortest_sample_len:
            msg = (f"Files need at least {need} rows ({why}) but the shortest file has {shortest_sample_len}: shorter files "
                   f"are skipped silently. Reduce frame_size or num_frame_concat until the requirement is <= {shortest_sample_len}.")
            errors.append(msg)
            param_validation.append({"param": "frame_size", "status": "error", "message": msg})

    pipe_errors, pipe_warnings = fr.pipeline_issues(
        task_type, eff_transforms, frame_size=eff_F, feature_size_per_frame=eff(feature_size_per_frame, "feature_size_per_frame"),
        num_frame_concat=eff_C, frame_skip=eff(frame_skip, "frame_skip"), analysis_bandwidth=eff(analysis_bandwidth, "analysis_bandwidth"),
        min_bin=eff_min_bin, offset=eff(offset, "offset"), stacking=eff(stacking, "stacking"), variables=num_vars,
        log_threshold=eff(log_threshold, "log_threshold"), using_preset=bool(pcfg),
        normalize_bin=eff(normalize_bin, "normalize_bin"), dc_remove=eff(dc_remove, "dc_remove"))
    errors.extend(pipe_errors)
    warnings.extend(pipe_warnings)
    conflict = fr.device_conflict(target_device, eff_transforms)
    if conflict:
        errors.append(f"Device {target_device}: {conflict}.")

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
        min_bin=min_bin,
        stride_size=stride_size,
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
        "rows_check": rows_check,
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
    min_bin: Optional[int] = None
    analysis_bandwidth: Optional[int] = None
    q15_scale_factor: Optional[int] = None
    dc_remove: Optional[bool] = None

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
        if self.min_bin is not None:
            result["min_bin"] = self.min_bin
        if self.analysis_bandwidth is not None:
            result["analysis_bandwidth"] = self.analysis_bandwidth
        if self.q15_scale_factor is not None:
            result["q15_scale_factor"] = self.q15_scale_factor
        if self.dc_remove is not None:
            result["dc_remove"] = bool(self.dc_remove)
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
    target_device: Optional[str] = None,
    min_bin: Optional[int] = None,
    analysis_bandwidth: Optional[int] = None,
    q15_scale_factor: Optional[int] = None,
    dc_remove: Optional[bool] = None,
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
        data_proc_transforms: e.g., ['SimpleWindow', 'Downsample'] (exact spelling; anything else is silently ignored).
        sampling_rate: Original sampling rate (Hz). Required if Downsample is used.
        new_sr: Target sampling rate (<= sampling_rate, integer ratio). Required if Downsample is used.
        frame_size: Rows per frame. Required if SimpleWindow is used. A file needs frame_size * num_frame_concat rows.
        stride_size: SimpleWindow step as a FRACTION of frame_size (step = int(frame_size*stride_size) rows; 0.25 = 75% overlap, 1.0 = no overlap).
        feature_extraction_name: Named preset or 'Custom_<name>'.
        feat_ext_transform: Custom pipeline steps (used with Custom_* presets).
        feature_size_per_frame: Output features per frame (custom pipelines).
        num_frame_concat: Number of frames concatenated per sample.
        stacking: '2D1' or '1D'.
        normalize_bin: Whether to normalize frequency bins.
        offset: Leave 0. Non-zero re-frames the series from shifted starts and is broken with num_frame_concat > 1.
        scale: Every value is divided by it before any other processing (e.g. 4096.0 for 12-bit ADC counts).
        frame_skip: Keep every N-th window (only with num_frame_concat >= 2). Leave 1 unless needed.
        log_mul: Logarithm multiplier (e.g., 20 for dB).
        log_base: Logarithm base (e.g., 10 or 'e').
        log_threshold: Added before the log to avoid log(0). If unset with LOG_DB, released tinyverse uses 1e-100 (log_mul 20, log_base 10); older source checkouts crash. Set it explicitly.
        forecast_horizon: Future timesteps to predict (forecasting only).
        target_variables: Columns to predict (forecasting only).
        target_device: e.g. 'MSPM0G5187': checks the chain exists in that device's feature-extraction library.
        min_bin, analysis_bandwidth, q15_scale_factor, dc_remove: advanced feature-extraction keys for custom pipelines.

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
    feature_schema.use_modelmaker_catalog(task_type)

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

    dp_errors, dp_warnings = _data_proc_issues(data_proc_transforms, sampling_rate, new_sr)
    errors.extend(dp_errors)
    warnings.extend(dp_warnings)

    if feature_extraction_name and feature_extraction_name not in feature_schema.list_all_presets() and not feature_extraction_name.startswith("Custom_"):
        warnings.append(
            f"'{feature_extraction_name}' is not in the known presets list. "
            "Verify the name against tinyml-modelmaker constants.py."
        )

    custom_errors, custom_warnings = _custom_pipeline_issues(feature_extraction_name, feat_ext_transform)
    errors.extend(custom_errors)
    warnings.extend(custom_warnings)

    named = feature_extraction_name if feature_extraction_name and not feature_extraction_name.startswith("Custom_") else None
    pcfg = (feature_schema.get_preset(named) if named else None) or {}
    eff_transforms = list(feat_ext_transform) if feat_ext_transform is not None else list(pcfg.get("feat_ext_transform") or [])
    num_vars_chk = variables if isinstance(variables, int) else (len(variables) if isinstance(variables, list) else None)

    def eff(explicit, key, default=None):
        return explicit if explicit is not None else pcfg.get(key, default)

    pipe_errors, pipe_warnings = fr.pipeline_issues(
        task_type, eff_transforms, frame_size=eff(frame_size, "frame_size"),
        feature_size_per_frame=eff(feature_size_per_frame, "feature_size_per_frame"),
        num_frame_concat=eff(num_frame_concat, "num_frame_concat"), frame_skip=eff(frame_skip, "frame_skip"),
        analysis_bandwidth=eff(analysis_bandwidth, "analysis_bandwidth"), min_bin=eff(min_bin, "min_bin", 1),
        offset=eff(offset, "offset"), stacking=eff(stacking, "stacking"), variables=num_vars_chk,
        log_threshold=eff(log_threshold, "log_threshold"), using_preset=bool(pcfg),
        normalize_bin=eff(normalize_bin, "normalize_bin"), dc_remove=eff(dc_remove, "dc_remove"))
    errors.extend(pipe_errors)
    warnings.extend(pipe_warnings)
    conflict = fr.device_conflict(target_device, eff_transforms)
    if conflict:
        errors.append(f"Device {target_device}: {conflict}.")

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
        min_bin=min_bin,
        analysis_bandwidth=analysis_bandwidth,
        q15_scale_factor=q15_scale_factor,
        dc_remove=dc_remove,
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
        min_bin=min_bin,
        stride_size=stride_size,
    )

    return {
        "success": True,
        "yaml": config.to_yaml_string(),
        "config": config.to_dict(),
        "computed_output_shape": shape_info,
        "errors": [],
        "warnings": warnings,
    }
