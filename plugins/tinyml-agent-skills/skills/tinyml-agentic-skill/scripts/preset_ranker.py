"""Intelligent preset ranking engine for feature extraction recommendations."""

from typing import Dict, List, Optional, Tuple, Any
from feature_schema import (
    list_all_presets, get_preset, get_task_recommendations,
    list_transforms_by_category
)


class PresetRanker:
    """Rank and score feature extraction presets based on multi-factor criteria."""

    def __init__(self):
        """Initialize ranker with all available presets."""
        self.presets = {}
        for name in list_all_presets():
            cfg = get_preset(name)
            if cfg:
                self.presets[name] = cfg

    def rank_presets(
        self,
        task_type: str,
        prefer_fft: bool,
        need_full_spectrum: bool,
        need_temporal_ctx: bool,
        min_sample_or_seq_length: int,
        variables: Optional[int] = None,
        sampling_rate: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Intelligently rank all compatible presets for a task.

        Args:
            task_type: Task type (e.g., 'motor_fault', 'arc_fault')
            variables: Number of sensor channels (e.g., 1, 3)
            sampling_rate: Original sampling rate (Hz)
            prefer_fft: True = prefer FFT presets, False = prefer RAW

        Returns:
            Sorted list of dicts with name, score, reasoning, metadata.
        """
        try:
            ranked = []

            for preset_name, preset_cfg in self.presets.items():

                score, reasons = self._score_preset(
                    preset_name, preset_cfg, task_type, variables, 
                    prefer_fft, need_full_spectrum, need_temporal_ctx, min_sample_or_seq_length 
                )

                if score > 0:  # Only include compatible presets
                    total_features = self._compute_total_features(preset_cfg)

                    ranked.append({
                        "name": preset_name,
                        "score": score,
                        "reasoning": reasons,
                        "frame_size": preset_cfg.get("frame_size"),
                        "feature_size_per_frame": preset_cfg.get("feature_size_per_frame"),
                        "num_frame_concat": preset_cfg.get("num_frame_concat"),
                        "total_features": total_features,
                        "variables": preset_cfg.get("variables"),
                        "stacking": preset_cfg.get("stacking"),
                        "use_case": preset_cfg.get("use_case", ""),
                        "description": preset_cfg.get("description", ""),
                    })

            ranked.sort(key=lambda x: x["score"], reverse=True)
            return ranked
        except Exception as e:
            print(e)
            raise

    def _score_preset(
        self,
        preset_name: str,
        preset_cfg: Dict,
        task_type: str,
        variables: Optional[int],
        prefer_fft: bool,
        need_full_spectrum: bool,
        need_temporal_ctx: bool,
        min_sample_or_seq_length: int,
    ) -> Tuple[float, List[str]]:
        """
        Score a single preset. Returns (score: 0-10, reasoning: list[str]).

        Scoring factors (normalized to max 10):
          - Variables match: +3 (exact) or +1.5 (flexible)
          - FFT vs Raw Preset:
            - If freq content and FFT - +1
                - If full spectrum not needed + FFTBIN - +0.5
            - If not freq content and RAW - +1
                - If need temporal context and multi-frame preset - + 0.5
          - Frame size: +3 (appropriate)
          - Use case: +1 (matches)

        """
        try:
            score = 0.0
            reasons = []

            preset_vars = preset_cfg.get("variables", 1)
            transform_type = self._get_transform_type(
                preset_cfg.get("feat_ext_transform", [])
            )
            preset_use_case = preset_cfg.get("use_case", "").lower()

            # === 1. Variables match (+3 exact, +1.5 flexible, 0 no match) ===
            if variables is not None:
                if preset_vars == variables:
                    score += 3
                    reasons.append(f"variables match ({variables})")
                elif preset_vars == 1 or variables == 1:
                    score += 1.5
                    reasons.append(f"variables flexible (preset={preset_vars}, request={variables})")
                else:
                    score += 0
                    return score, reasons  # Incompatible
            else:
                score += 0.75  # Generic award if no constraint
                reasons.append("no variable constraint")

            # === 2. Preference (fft vs raw) ===
            if prefer_fft is not None:
                if prefer_fft and transform_type in ("FFT", "FFT_Q15"):
                    score += 1
                    reasons.append("FFT preferred since data pattern is in frequency content")
                    if not need_full_spectrum and "FFTBIN" in preset_name:
                        score += 0.25
                        reasons.append("Binning used since full spectrum not needed")
                    if need_temporal_ctx and any(f"{i}Frame" in preset_name for i in range(2, 10)):
                        score += 0.25
                        reasons.append("multi-frame preset chosen since temporal context needed")
                elif not prefer_fft and transform_type == "RAW":
                    score += 1
                    reasons.append("raw waveform preferred")
                    if need_temporal_ctx and any(f"{i}Frame" in preset_name for i in range(2, 10)):
                        score += 0.5
                        reasons.append("multi-frame preset chosen since temporal context needed")

            # === 3. Frame size appropriateness (+3) === -> can be expanded later to a multi-step scorer that checks optimal framesize
            frame_size = preset_cfg.get("frame_size")
            if frame_size and frame_size <= min_sample_or_seq_length:
                score += 3
                reasons.append(f"adequate frame size ({frame_size})")

            # === 4. Use case alignment (+1 if matches) ===
            if any(task_keyword in preset_use_case
                for task_keyword in task_type.split("_")):
                score += 1
                reasons.append(f"use case matches ({preset_use_case})")

            return max(0.0, min(10.0, score)), reasons
        except Exception as e:
            print(e)
            raise

    def _get_transform_type(self, feat_ext_transform: List[str]) -> str:
        """Classify feat_ext_transform list as FFT, RAW, PIR, Q15, or CUSTOM."""
        if not feat_ext_transform:
            return "CUSTOM"

        transform_set = set(feat_ext_transform)
        has_fft = any(t in transform_set for t in ("FFT_FE", "FFT_Q15", "FFT_POS_HALF"))
        has_raw = "RAW_FE" in transform_set
        has_pir = "PIR_FE" in transform_set or "PIR_FE_Q15" in transform_set
        has_q15 = any(t in transform_set for t in ("FFT_Q15", "Q15_SCALE", "BIN_Q15"))

        if has_pir:
            return "PIR"
        if has_fft:
            return "FFT_Q15" if has_q15 else "FFT"
        if has_raw:
            return "RAW"
        return "CUSTOM"

    def _compute_total_features(self, preset_cfg: Dict) -> Optional[int]:
        """Compute total output features: feature_size_per_frame × num_frame_concat × variables."""
        fs = preset_cfg.get("feature_size_per_frame")
        nfc = preset_cfg.get("num_frame_concat")
        v = preset_cfg.get("variables", 1)

        if fs is None or nfc is None:
            return None
        return fs * nfc * v
