"""Preset ranking for feature extraction recommendations.

Hard gates (a preset failing one is excluded, with the reason): device library family, channel count,
minimum rows per file (frame_size * num_frame_concat, after Downsample/SimpleWindow), bin sanity.
Soft score (0-10): variables match, FFT/RAW preference, frame fit margin, use case, minus a penalty for
application-specific presets. All arithmetic lives in fe_rules.py. Decisions use the preset's values, never its name.
"""

from typing import Any, Dict, List, Optional, Tuple

import fe_rules as fr
from feature_schema import list_all_presets, get_preset

# Words in a task type that say nothing about the application.
_GENERIC_TASK_WORDS = {"generic", "timeseries", "classification", "detection", "regression", "forecasting",
                       "anomalydetection"}
# Application prefix -> keyword in the task type that makes that application appropriate.
_APPLICATION_TASK_KEYWORD = {"ArcFault": "arc", "MotorFault": "motor", "ECG": "ecg", "PIRDetection": "pir",
                             "HandGesture": None, "FallDetection": None, "WiFiCSI": None}


class PresetRanker:
    """Rank and score feature extraction presets."""

    def __init__(self):
        self.presets: Dict[str, Dict[str, Any]] = {}
        for name in list_all_presets():
            cfg = get_preset(name)
            if cfg:
                self.presets[name] = cfg

    # ── public API ────────────────────────────────────────────────────────────

    def rank_presets(self, task_type: str, prefer_fft: bool, need_full_spectrum: bool, need_temporal_ctx: bool,
                     shortest_sample_len: int, variables: Optional[int] = None, sampling_rate: Optional[int] = None,
                     target_device: Optional[str] = None) -> List[Dict[str, Any]]:
        """Ranked, compatible presets (highest score first)."""
        return self.rank_with_exclusions(task_type, prefer_fft, need_full_spectrum, need_temporal_ctx,
                                         shortest_sample_len, variables, sampling_rate, target_device)[0]

    def rank_with_exclusions(self, task_type: str, prefer_fft: bool, need_full_spectrum: bool,
                             need_temporal_ctx: bool, shortest_sample_len: int, variables: Optional[int] = None,
                             sampling_rate: Optional[int] = None, target_device: Optional[str] = None
                             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """(ranked, excluded). Every excluded entry carries the reason."""
        ranked, excluded = [], []
        for name, cfg in self.presets.items():
            score, reasons, excl, details = self._evaluate(
                name, cfg, task_type, variables, prefer_fft, need_full_spectrum, need_temporal_ctx,
                shortest_sample_len, sampling_rate, target_device)
            if excl:
                excluded.append({"name": name, "frame_size": cfg.get("frame_size"), "reason": excl})
                continue
            shape = details.get("input_shape")
            total = 1
            for dim in (shape or ()):
                total *= dim
            ranked.append({
                "name": name,
                "score": score,
                "reasoning": reasons,
                "frame_size": cfg.get("frame_size"),
                "feature_size_per_frame": cfg.get("feature_size_per_frame"),
                "num_frame_concat": cfg.get("num_frame_concat"),
                "total_features": total if shape else None,
                "variables": cfg.get("variables"),
                "stacking": cfg.get("stacking"),
                "use_case": cfg.get("use_case", ""),
                "description": cfg.get("description", ""),
                **details,
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked, excluded

    # ── scoring ───────────────────────────────────────────────────────────────

    def _evaluate(self, name, cfg, task_type, variables, prefer_fft, need_full_spectrum, need_temporal_ctx,
                  shortest, sampling_rate, target_device):
        """Returns (score, reasons, exclusion_reason_or_None, details)."""
        transforms = list(cfg.get("feat_ext_transform") or [])
        ts = set(transforms)
        data_proc = cfg.get("data_proc_transforms") or []
        F = cfg.get("frame_size")
        B = cfg.get("feature_size_per_frame")
        C = cfg.get("num_frame_concat") or 1
        A = cfg.get("analysis_bandwidth") or 1
        m = cfg.get("min_bin", 1)
        preset_vars = cfg.get("variables", 1) or 1
        score, reasons, details = 0.0, [], {}

        # === 1. Device library (hard) ===
        family = fr.device_family(target_device)
        conflict = fr.device_conflict(target_device, ts)
        if conflict:
            return 0.0, [], conflict, {}
        if family:
            reasons.append(f"device library matches ({'Q15 fixed-point' if ts & fr.Q15_TRANSFORMS else 'float or none'} on {target_device})")

        # === 2. Variables ===
        if variables is not None:
            if preset_vars == variables:
                score += 3
                reasons.append(f"variables match ({variables})")
            elif preset_vars == 1 and not ts & {"PIR_FE", "PIR_FE_Q15", "FFT_COL"}:
                score += 1.5   # the 1-channel chain is applied to every selected channel
                reasons.append(f"variables flexible (preset=1, dataset={variables}); the chain runs on each channel")
            else:
                return 0.0, [], f"preset is built for {preset_vars} channel(s); {variables} selected", {}
        else:
            score += 0.75
            reasons.append("no variable constraint")

        # === 3. Rows per file (hard) ===
        k = fr.downsample_factor(data_proc, cfg.get("sampling_rate"), cfg.get("new_sr"))
        stride = fr.window_stride_rows(F, cfg.get("stride_size")) if (F and fr.uses_simple_window(data_proc)) else None
        need = fr.min_rows_required(F, C, k, stride, has_fe=bool(transforms)) if F else None
        if need is not None:
            details["min_rows_required"] = need
            if need > shortest:
                why = f"frame_size {F} x num_frame_concat {C}"
                if k > 1:
                    why += f" after Downsample x{k}"
                return 0.0, [], f"needs >= {need} rows per file ({why}); shortest file has {shortest}: shorter files would be skipped silently", {}
            score += 3
            reasons.append(f"every file has enough rows ({shortest} >= {need}: frame_size {F} x num_frame_concat {C})")
            if shortest >= 4 * need:
                score += 0.5
                reasons.append("comfortable margin: many windows per file")
        if F and ts & {"BINNING", "BIN_Q15"} and B and B * A > F // 2:
            return 0.0, [], f"feature_size_per_frame*analysis_bandwidth = {B * A} > frame_size/2 = {F // 2}", {}

        # === 4. FFT vs RAW preference, from the preset's transforms ===
        kind = self._get_transform_type(transforms)
        multi_frame = C >= 2 and "CONCAT" in ts
        if prefer_fft is not None:
            if prefer_fft and kind in ("FFT", "FFT_Q15"):
                score += 1
                reasons.append("FFT preferred since the pattern is in frequency content")
                if not need_full_spectrum and ts & {"BINNING", "BIN_Q15"}:
                    score += 0.25
                    reasons.append("binning used since the full spectrum is not needed")
                if need_full_spectrum and not ts & {"BINNING", "BIN_Q15"}:
                    score += 0.25
                    reasons.append("full spectrum kept (no binning)")
                if need_temporal_ctx and multi_frame:
                    score += 0.25
                    reasons.append(f"{C}-frame preset since temporal context is needed")
            elif not prefer_fft and kind == "RAW":
                score += 1
                reasons.append("raw waveform preferred")
                if need_temporal_ctx and multi_frame:
                    score += 0.5
                    reasons.append(f"{C}-frame preset since temporal context is needed")

        # === 5. Application-specific presets are for that application's data ===
        for prefix, keyword in _APPLICATION_TASK_KEYWORD.items():
            if name.startswith(prefix) and not (keyword and keyword in task_type):
                score -= 2
                reasons.append(f"application-specific preset ({prefix}): choose it only if the data is that application")
                break

        # === 6. Use case ===
        use_case = (cfg.get("use_case") or "").lower()
        words = [w for w in task_type.split("_") if w not in _GENERIC_TASK_WORDS]
        if use_case and any(w in use_case for w in words):
            score += 1
            reasons.append(f"use case matches ({cfg.get('use_case')})")

        # === details for the caller: shape, durations, resolution, problems with the preset itself ===
        n_vars = variables or preset_vars
        shape = fr.output_shape(transforms, F, C, n_vars, cfg.get("stacking") or "2D1", B, m,
                                cfg.get("window_count") or 1, cfg.get("chunk_size") or 1, cfg.get("stride_size")) if F else {"shape": None}
        details["input_shape"] = shape["shape"]
        if F:
            details["rows_per_window"] = F * C
            fs = (cfg.get("new_sr") if k > 1 and cfg.get("new_sr") else None) or sampling_rate
            if fs and not ts & {"PIR_FE", "PIR_FE_Q15"}:
                details["window_seconds"] = round(F * C / fs, 6)
                details["fft_bin_width_hz"] = round(fs / F, 6) if ts & fr.FFT_STEPS else None
        errs, warns = fr.pipeline_issues(task_type, transforms, frame_size=F, feature_size_per_frame=B,
                                         num_frame_concat=C, analysis_bandwidth=A, min_bin=m, stacking=cfg.get("stacking"),
                                         using_preset=True, normalize_bin=cfg.get("normalize_bin"),
                                         dc_remove=cfg.get("dc_remove"))
        if errs or warns:
            details["preset_notes"] = errs + warns
        if family:
            details["device_family"] = family
        return max(0.0, min(10.0, score)), reasons, None, details

    def _get_transform_type(self, feat_ext_transform: List[str]) -> str:
        """Classify a transform list as FFT, FFT_Q15, RAW, PIR or CUSTOM."""
        if not feat_ext_transform:
            return "CUSTOM"
        t = set(feat_ext_transform)
        if t & {"PIR_FE", "PIR_FE_Q15"}:
            return "PIR"
        if t & {"FFT_FE", "FFT_Q15", "FFT_POS_HALF"}:
            return "FFT_Q15" if t & fr.Q15_TRANSFORMS else "FFT"
        if "RAW_FE" in t:
            return "RAW"
        return "CUSTOM"
