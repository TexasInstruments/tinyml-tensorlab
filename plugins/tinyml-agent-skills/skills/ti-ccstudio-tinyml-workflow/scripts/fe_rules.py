"""Feature-extraction arithmetic and pipeline rules (pure Python, no modelmaker import).

Mirrors tinyml_tinyverse ``BaseGenericTSDataset`` (classification). See
references/FE_and_Data_Processing_Transforms/arithmetic.md for the derivation. tests/test_fe_rules.py
compares these functions with the real dataset class.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

Q15_TRANSFORMS = {"TO_Q15", "FFT_Q15", "Q15_SCALE", "Q15_MAG", "BIN_Q15", "PIR_FE_Q15"}
# Float spectral steps: the MSPM0 feature-extraction library only implements the Q15 chain.
FLOAT_SPECTRAL_TRANSFORMS = {"FFT_FE", "BINNING", "LOG_DB", "WINDOWING"}
SCALAR_FEATURES = {"KURT_FE", "ENT_FE", "ZCR_FE", "DOM_FE", "SLOPE_FE"}   # only meaningful inside PIR_FE
NO_DEVICE_LIBRARY = {"L2_NORM_ROW", "FFT_COL", "RANGE_NORMALIZE"} | SCALAR_FEATURES
FFT_STEPS = {"FFT_FE", "FFT_Q15"}
# Steps the anomaly-detection dataset class actually executes; everything else is silently ignored.
AD_SUPPORTED = {"WINDOWING", "RAW_FE", "FFT_FE", "NORMALIZE", "FFT_POS_HALF", "ABS", "DC_REMOVE",
                "BINNING", "LOG_DB", "CONCAT"}
# Names that look valid but that no dataset class acts on.
NO_EFFECT = {"HAAR", "HADAMARD", "RANGE_NORMALIZE", "ECG_NORMALIZE", "FE_DECIMATE", "FE_Q15", "NORMALIZE_ROW"}
# Application-specific preset prefixes: appropriate only for that application's data.
APPLICATION_PREFIXES = ("HandGesture", "FallDetection", "WiFiCSI", "ECG", "PIRDetection", "ArcFault",
                        "MotorFault")


def device_family(device: Optional[str]) -> Optional[str]:
    """'mspm0' (Q15 fixed-point library), 'other' (float library), or None when unknown."""
    if not device:
        return None
    return "mspm0" if str(device).upper().startswith("MSPM0") else "other"


def device_conflict(device: Optional[str], transforms: Iterable[str]) -> Optional[str]:
    """Reason the transform chain cannot run on the device's feature-extraction library, else None.
    The MSPM0 library only has the Q15 fixed-point chain; the other devices only the float chain."""
    ts = set(transforms or [])
    family = device_family(device)
    if family == "mspm0" and ts & FLOAT_SPECTRAL_TRANSFORMS:
        return (f"float FFT chain {sorted(ts & FLOAT_SPECTRAL_TRANSFORMS)} is not implemented by the MSPM0 "
                "feature-extraction library, which only has the Q15 fixed-point chain")
    if family == "other" and ts & Q15_TRANSFORMS:
        return (f"fixed-point Q15 chain {sorted(ts & Q15_TRANSFORMS)} is only implemented for MSPM0 devices, "
                f"not {device}")
    return None


def window_stride_rows(frame_size: int, stride_size: Optional[float]) -> int:
    """Stride in rows that SimpleWindow uses: int(F * stride_size), falling back to F when that is 0."""
    stride = int((frame_size or 0) * (stride_size if stride_size is not None else 0.01))
    return stride if stride > 0 else frame_size


def uses_simple_window(data_proc: Optional[Iterable[str]]) -> bool:
    return bool(data_proc) and "SimpleWindow" in list(data_proc)


def downsample_factor(data_proc: Optional[Iterable[str]], sampling_rate, new_sr) -> int:
    """k = int(sampling_rate // new_sr) when Downsample is requested, else 1 (0 would be an invalid step)."""
    if data_proc and "Downsample" in list(data_proc) and sampling_rate and new_sr:
        return int(sampling_rate // new_sr)
    return 1


# ─── per-frame length and model input shape ──────────────────────────────────

def per_frame_length(transforms: Iterable[str], frame_size: int, feature_size_per_frame: Optional[int] = None,
                     min_bin: Optional[int] = 1) -> Optional[int]:
    """Length of one frame's feature vector after the fixed-order chain; None if it is not a vector
    (PIR) or cannot be computed."""
    t = set(transforms or [])
    if t & {"PIR_FE", "PIR_FE_Q15"} or t & SCALAR_FEATURES:
        return None
    n = frame_size
    if "FFT_Q15" in t:
        n = 2 * frame_size
    if "Q15_MAG" in t:
        n = frame_size // 2 + 1
    if "FFT_POS_HALF" in t:
        n = frame_size // 2 + (min_bin if min_bin is not None else 1)
    if "DC_REMOVE" in t:
        n -= 1
    if t & {"BINNING", "BIN_Q15"}:
        if not feature_size_per_frame:
            return None
        n = feature_size_per_frame
    return n


def output_shape(transforms: Iterable[str], frame_size: int, num_frame_concat: int = 1, variables: int = 1,
                 stacking: str = "2D1", feature_size_per_frame: Optional[int] = None,
                 min_bin: Optional[int] = 1, window_count: int = 1, chunk_size: int = 1,
                 stride_size: Optional[float] = None) -> Dict[str, Any]:
    """Model input shape per sample as the dataset produces it: (channels, width, 1)
    (PIR: (V, windows, features); FFT_COL: (V//2, 1, width)). Returns {'shape', 'per_frame', 'note'}."""
    t = set(transforms or [])
    V, C = variables or 1, num_frame_concat or 1
    if not t:
        return {"shape": (V, frame_size, 1), "per_frame": frame_size, "note": "no feature extraction: raw windows"}
    if "PIR_FE_Q15" in t:
        return {"shape": (V, 25, 25), "per_frame": None, "note": "PIR_FE_Q15 is hard-coded to 25 windows x 25 features"}
    if "PIR_FE" in t:
        stride = int(frame_size * (stride_size or 0))
        chunks = ((frame_size + 7) - window_count * stride) // max(chunk_size, 1)
        return {"shape": (V, window_count, 16 + chunks + 5), "per_frame": None,
                "note": "PIR_FE: 16 FFT bins + kurtosis chunks + zcr, slope, 2 dominant freqs, entropy"}
    p = per_frame_length(t, frame_size, feature_size_per_frame, min_bin)
    if p is None:
        return {"shape": None, "per_frame": None, "note": "cannot compute (feature_size_per_frame missing, or a scalar feature used alone)"}
    wl = p * C
    if "FFT_COL" in t:
        return {"shape": (V // 2, 1, wl), "per_frame": p, "note": "FFT_COL: channels become V//2"}
    if stacking == "1D":
        return {"shape": (1, wl * V, 1), "per_frame": p, "note": "1D stacking: variables laid end to end"}
    return {"shape": (V, wl, 1), "per_frame": p, "note": "2D1 stacking: one channel per variable"}


# ─── how many frames / windows a file gives ──────────────────────────────────

def min_rows_required(frame_size: int, num_frame_concat: int = 1, downsample: int = 1,
                      window_stride: Optional[int] = None, has_fe: bool = True) -> Optional[int]:
    """Fewest raw rows a file needs to yield at least one sample (shorter files are skipped silently).
    window_stride: rows if SimpleWindow is used, else None. None is returned when no sample can ever form
    (no SimpleWindow and no feature extraction)."""
    C = max(num_frame_concat or 1, 1)
    if not has_fe:
        if window_stride is None:
            return None
        rp = frame_size
    elif window_stride is not None:
        rp = frame_size + (C - 1) * window_stride
    else:
        rp = frame_size * C
    return (rp - 1) * max(downsample, 1) + 1


def fit(R: int, frame_size: int, num_frame_concat: int = 1, frame_skip: int = 1, variables: int = 1,
        stacking: str = "2D1", transforms: Iterable[str] = (), feature_size_per_frame: Optional[int] = None,
        analysis_bandwidth: int = 1, min_bin: Optional[int] = 1, downsample: int = 1,
        window_stride: Optional[int] = None, **shape_kwargs) -> Dict[str, Any]:
    """Predict what the classification dataset produces for ONE file with R rows."""
    F, C, S = frame_size, max(num_frame_concat or 1, 1), max(frame_skip or 1, 1)
    t = list(transforms or [])
    Rp = math.ceil(R / max(downsample, 1))
    if window_stride is not None:
        if Rp - F + 1 <= 0:
            return {"skipped": f"{Rp} rows after downsampling < frame_size {F}"}
        frames = math.ceil((Rp - F + 1) / window_stride)
    else:
        frames = Rp // F
    if not t:   # no feature extraction: every SimpleWindow window is a sample
        if window_stride is None:
            return {"skipped": "no feature extraction and no SimpleWindow: the data has no windows"}
        return {"frames": frames, "windows": frames, "input_shape": (variables, F, 1), "rows_per_window": F}
    if frames < C:
        return {"skipped": f"{frames} frames < num_frame_concat {C}"}
    windows = frames if C == 1 else sum(1 for i in range(C - 1, frames) if i % S == 0)
    if set(t) & {"BINNING", "BIN_Q15"}:
        if not feature_size_per_frame or F // 2 // feature_size_per_frame // max(analysis_bandwidth, 1) < 1:
            return {"skipped": "bin_size < 1 (feature_size_per_frame * analysis_bandwidth > frame_size/2)"}
    shape = output_shape(t, F, C, variables, stacking, feature_size_per_frame, min_bin, **shape_kwargs)
    return {"frames": frames, "windows": windows, "input_shape": shape["shape"], "rows_per_window": F * C}


# ─── pipeline rules ──────────────────────────────────────────────────────────

def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def pipeline_issues(task_type: str, transforms: Optional[List[str]], *, frame_size: Optional[int] = None,
                    feature_size_per_frame: Optional[int] = None, num_frame_concat: Optional[int] = None,
                    frame_skip: Optional[int] = None, analysis_bandwidth: Optional[int] = None,
                    min_bin: Optional[int] = 1, offset: Optional[int] = None, stacking: Optional[str] = None,
                    variables: Optional[int] = None, log_threshold=None, using_preset: bool = False,
                    normalize_bin=None, dc_remove=None) -> Tuple[List[str], List[str]]:
    """(errors, warnings) for a feature-extraction pipeline. Only things training would silently
    ignore, crash on, or deploy differently."""
    errors: List[str] = []
    warnings: List[str] = []
    t = list(transforms or [])
    ts = set(t)
    task = task_type or ""
    C = num_frame_concat or 1
    S = frame_skip or 1
    A = analysis_bandwidth or 1

    for name in t:
        if name in NO_EFFECT:
            # a stock preset that lists such a name is a modelmaker quirk the user cannot fix: warn only
            (warnings if using_preset else errors).append(f"'{name}' has no effect: no dataset class acts on it.")
    if "forecasting" in task and t:
        errors.append("Forecasting does not run feature extraction: feat_ext_transform is ignored. "
                      "Use only data_proc_transforms (SimpleWindow).")
        return errors, warnings
    if "anomaly" in task:
        ignored = sorted(ts - AD_SUPPORTED - set(NO_EFFECT))
        if ignored:
            warnings.append(f"Anomaly detection silently ignores {ignored}; only the float FFT/RAW chain runs there.")
    if "regression" in task:
        if ts & {"PIR_FE", "PIR_FE_Q15"} or stacking == "1D":
            errors.append("Regression supports neither PIR_FE nor stacking '1D' (its output shape handling is 3-D only).")
        if t and (C > 1 or S > 1):
            warnings.append("Regression with num_frame_concat>1 or frame_skip>1: X can end up shorter than Y with no "
                            "check. Keep both at 1 or verify len(X) == len(Y).")

    if not t:
        if C > 1:
            warnings.append(f"num_frame_concat={C} has no effect without feature extraction.")
        return errors, warnings

    if C > 1 and "CONCAT" not in ts:
        errors.append(f"num_frame_concat={C} needs 'CONCAT' in feat_ext_transform (without it no samples are produced).")
    if S > 1:
        warnings.append(f"frame_skip={S} thins windows only when num_frame_concat>=2 and leaves X_raw un-thinned; use 1 unless needed.")
    if offset and C > 1:
        errors.append("offset>0 with num_frame_concat>1 is broken in the dataset code; set offset: 0.")
    if ts & SCALAR_FEATURES and not ts & {"PIR_FE", "PIR_FE_Q15"}:
        errors.append(f"{sorted(ts & SCALAR_FEATURES)} only work inside PIR_FE; used alone the output shape is invalid.")
    if "FFT_FE" in ts and "FFT_Q15" in ts:
        errors.append("FFT_FE (float) and FFT_Q15 (fixed-point) cannot be mixed; choose one chain.")
    if ts & NO_DEVICE_LIBRARY - NO_EFFECT:
        warnings.append(f"{sorted(ts & NO_DEVICE_LIBRARY - NO_EFFECT)}: no device feature-extraction library support found; "
                        "training works but verify the device can reproduce it.")

    if ts & FFT_STEPS and frame_size and not _is_pow2(frame_size):
        errors.append(f"frame_size {frame_size} is not a power of two; the device FFT needs one (FE_FFT_STAGES = log2).")
    if "LOG_DB" in ts:
        if ts & {"FFT_FE"} and not ts & {"ABS"}:
            errors.append("LOG_DB after FFT_FE needs ABS first (log of a complex spectrum).")
        if not using_preset and log_threshold is None:
            warnings.append("log_threshold is not set: released tinyverse 1.5.0 then uses log_threshold=1e-100 "
                            "(with log_mul=20, log_base=10), but an older source checkout crashes with a TypeError. "
                            "Set log_mul, log_base and log_threshold explicitly to be safe and to keep the values visible.")
    if ts & {"BINNING", "BIN_Q15"}:
        B = feature_size_per_frame
        if not B:
            errors.append("BINNING needs feature_size_per_frame.")
        elif frame_size:
            half = frame_size // 2
            if B * A > half:
                errors.append(f"feature_size_per_frame*analysis_bandwidth = {B * A} > frame_size/2 = {half}: bin_size would be 0.")
            else:
                if half % (B * A):
                    warnings.append(f"frame_size/2 = {half} is not divisible by feature_size_per_frame*analysis_bandwidth = {B * A}; "
                                    "the top bins are silently ignored.")
                before = per_frame_length(ts - {"BINNING", "BIN_Q15"}, frame_size, None, min_bin)
                bin_size = half // B // A
                off = min_bin or 0
                if before is not None and off + B * bin_size > before:
                    warnings.append(f"min_bin={min_bin}: the last band covers {before - off - (B - 1) * bin_size} bin(s) but is "
                                    f"divided by bin_size={bin_size} (offset applied after DC_REMOVE). Small, but the device must match.")
        if "BIN_Q15" in ts and normalize_bin in (0, False):
            warnings.append("normalize_bin is false but the MSPM0 device always normalises BIN_Q15; set normalize_bin: true.")
    if "RAW_FE" in ts and dc_remove in (0, False):
        warnings.append("RAW_FE with dc_remove false: Python keeps the mean but the device always removes it.")
    if stacking and stacking not in ("1D", "2D1"):
        warnings.append(f"stacking '{stacking}' is not '1D' or '2D1'; the code treats any other value as 2D1.")
    return errors, warnings
