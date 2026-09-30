"""
prepare_dataset.py  —  v2

Full pipeline: raw EdgeAI Studio CSV → segmented 8820 Hz WAVs ready for TensorLab.

HOW THIS MATCHES THE C FIRMWARE
---------------------------------
Step 1  extract int16   (raw >> 10, clamp to [-32768, 32767])
        Firmware ISR:   int32_t s32 = word >> 10; clamp; sample = (int16_t)s32

Step 2  deinterleave L channel   (even indices: 0, 2, 4, ...)
        Firmware ISR:   reads L word, waits for R, discards R

Step 3  decimate by 5   (every 5th L sample → 44100/5 = 8820 Hz)
        Firmware ISR:   dec increments per L word; fires when dec >= 5

Step 4  segment into 2-second clips (17640 samples @ 8820 Hz)
        Firmware:       100 frames × 176 samples = 17600-sample sliding window
        50% overlap (stride = 1s) doubles the clip count.

AUGMENTATION STRATEGY  (v2)
----------------------------
silence/white-noise are used ONLY as noise bank, never as "other" clips.
cough files: [1.0, 0.15] — spans RMS ~500-15000, overlapping "other"
  forcing the model to use spectral shape, not energy, to discriminate.
other files: [1.0] only — no augmentation.

USAGE
-----
  python prepare_dataset.py --data-dir /path/to/csvs --output-dir /path/to/output

  Or via environment variables:
    DATA_DIR=/path/to/csvs OUTPUT_DIR=/path/to/output python prepare_dataset.py
"""

import argparse
import os
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE_FIFO = 88200
SAMPLE_RATE_L    = 44100
SAMPLE_RATE_OUT  = 8820
DECIMATE         = 5

CLIP_SAMPLES   = int(SAMPLE_RATE_OUT * 2)   # 17640 = 2 s @ 8820 Hz
STRIDE_SAMPLES = CLIP_SAMPLES // 2          # 8820 = 1 s stride (50% overlap)

RNG = np.random.default_rng(42)             # fixed seed for reproducibility

# Gain levels applied to cough clips only.
# 0.15 gain → RMS ~550-1160 when mixed with noise at 10 dB SNR.
COUGH_GAIN_LEVELS = [1.0, 0.15]


# ── Audio processing helpers ──────────────────────────────────────────────────
def csv_to_decimated(csv_path: Path) -> np.ndarray:
    """Load an EdgeAI CSV and return 8820 Hz L-channel int16 audio."""
    raw    = pd.read_csv(csv_path).iloc[:, 0].values.astype("int32")
    audio  = np.clip(raw >> 10, -32768, 32767).astype("int16")
    l_chan = audio[0::2]
    return l_chan[::DECIMATE]


def build_noise_bank(noise_sources: list[Path]) -> np.ndarray:
    """Concatenate all noise sources into a single 8820 Hz int16 array."""
    parts = []
    missing = [p for p in noise_sources if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Noise source file(s) not found:\n"
            + "\n".join(f"  {p}" for p in missing)
        )
    for p in noise_sources:
        parts.append(csv_to_decimated(p))
    return np.concatenate(parts) if parts else np.zeros(CLIP_SAMPLES, dtype="int16")


def mix_with_noise(clip: np.ndarray, noise_bank: np.ndarray,
                   gain: float, snr_db: float = 10.0) -> np.ndarray:
    """
    Scale clip by gain, then add noise so that SNR = snr_db relative
    to the scaled signal.  Returns int16 clamped result.
    """
    sig = clip.astype(np.float64) * gain

    n = len(sig)
    if len(noise_bank) > n:
        start = int(RNG.integers(0, len(noise_bank) - n))
        noise = noise_bank[start:start + n].astype(np.float64)
    else:
        reps  = n // len(noise_bank) + 1
        noise = np.tile(noise_bank.astype(np.float64), reps)[:n]

    sig_rms   = np.sqrt(np.mean(sig ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noise *= target_noise_rms / noise_rms

    return np.clip(sig + noise, -32768, 32767).astype("int16")


# ── WAV writer ────────────────────────────────────────────────────────────────
def write_wav(path: Path, clip: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE_OUT)
        wf.writeframes(struct.pack(f"<{CLIP_SAMPLES}h", *clip.tolist()))


# ── Per-file processor ────────────────────────────────────────────────────────
def process_file(csv_path: Path, class_name: str, out_dir: Path,
                 clip_offset: int, noise_bank: np.ndarray) -> int:
    """
    Segment one CSV into 2-second clips.  Cough files are augmented with
    COUGH_GAIN_LEVELS; all other files use gain=1.0 only.
    Returns base clip count (gain=1.0 only).
    """
    gain_levels = COUGH_GAIN_LEVELS if class_name == "cough" else [1.0]
    print(f"  {csv_path.name}  (offset {clip_offset:04d})  gains={gain_levels}")

    decimated = csv_to_decimated(csv_path)
    clips_written = 0

    start = 0
    while start + CLIP_SAMPLES <= len(decimated):
        raw_clip = decimated[start: start + CLIP_SAMPLES]

        for g_idx, gain in enumerate(gain_levels):
            if gain == 1.0:
                clip = raw_clip.copy()
            else:
                clip = mix_with_noise(raw_clip, noise_bank, gain, snr_db=10)

            idx      = clip_offset + clips_written + g_idx * 10000
            out_path = out_dir / f"{class_name}_{idx:05d}.wav"
            write_wav(out_path, clip)

        clips_written += 1
        start         += STRIDE_SAMPLES

    return clips_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert EdgeAI Studio CSVs to segmented 8820 Hz training WAVs (v2)."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", ""),
        help="Directory containing the raw CSV recordings (or set DATA_DIR env var).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", ""),
        help="Directory where the output WAV dataset will be written (or set OUTPUT_DIR env var).",
    )
    args = parser.parse_args()

    if not args.data_dir:
        parser.error(
            "DATA_DIR is not set. Pass --data-dir /path/to/csvs "
            "or set the DATA_DIR environment variable."
        )
    if not args.output_dir:
        parser.error(
            "OUTPUT_DIR is not set. Pass --output-dir /path/to/output "
            "or set the OUTPUT_DIR environment variable."
        )

    DATA_DIR   = Path(args.data_dir)
    OUTPUT_DIR = Path(args.output_dir)

    if not DATA_DIR.is_dir():
        sys.exit(f"ERROR: --data-dir does not exist: {DATA_DIR}")

    # ── Class map ─────────────────────────────────────────────────────────────
    CLASS_MAP = {
        DATA_DIR / "coughing_1787056684961.csv":                "cough",
        DATA_DIR / "coughing_2_1787155987537.csv":              "cough",
        DATA_DIR / "coughing_female_1787158883351.csv":         "cough",
        DATA_DIR / "hello_1787049513302.csv":                   "other",
        DATA_DIR / "meeting_1787155577802.csv":                 "other",
        DATA_DIR / "people_talking_chatting_1787155418511.csv": "other",
    }

    # Noise sources used for cough augmentation (NOT added as "other" clips).
    NOISE_SOURCES = [
        DATA_DIR / "silence_1787056345178.csv",
        DATA_DIR / "blank_white_noise_1787154693435.csv",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("CSV → 8820 Hz segmented training dataset  (v2, cough augmentation only)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Fail early if any CLASS_MAP file is missing.
    missing_class = [p for p in CLASS_MAP if not p.exists()]
    if missing_class:
        sys.exit(
            "ERROR: The following class CSV file(s) are missing:\n"
            + "\n".join(f"  {p}" for p in missing_class)
            + "\nFix the paths or provide the missing recordings before continuing."
        )

    print("\nBuilding noise bank for augmentation ...")
    noise_bank = build_noise_bank(NOISE_SOURCES)
    rms_nb = int(np.sqrt(np.mean(noise_bank.astype(np.float64) ** 2)))
    print(f"  Noise bank: {len(noise_bank):,} samples  RMS={rms_nb}")

    class_names = set(CLASS_MAP.values())

    for class_name in class_names:
        class_dir = OUTPUT_DIR / class_name
        class_dir.mkdir(exist_ok=True)
        for old in class_dir.glob("*.wav"):
            old.unlink()

    class_offsets: dict    = {}
    class_base_clips: dict = {}

    for csv_path, class_name in CLASS_MAP.items():
        print(f"\n[{class_name}]")
        out_dir = OUTPUT_DIR / class_name
        offset  = class_offsets.get(class_name, 0)
        n       = process_file(csv_path, class_name, out_dir, offset, noise_bank)
        class_offsets[class_name]    = offset + n
        class_base_clips[class_name] = class_base_clips.get(class_name, 0) + n

    print("\n" + "=" * 60)
    print("Dataset summary")
    print("=" * 60)
    total = 0
    for class_name in sorted(class_names):
        wavs = list((OUTPUT_DIR / class_name).glob("*.wav"))
        base = class_base_clips.get(class_name, 0)
        aug  = len(wavs) - base
        suffix = f" ({base} original + {aug} augmented)" if aug else ""
        print(f"  {class_name:6s}: {len(wavs):4d} clips{suffix}")
        total += len(wavs)
    print(f"\n  Total: {total} clips across {len(set(CLASS_MAP.values()))} classes")

    print(f"\nTensorLab config:")
    print(f"  input_data_path : {OUTPUT_DIR.parent.parent}")
    print(f"  data_dir        : {'/'.join(OUTPUT_DIR.parts[-2:])}")
    print(f"  sampling_rate   : 8820")
    print(f"  num_classes     : 2")
    print(f"  class_names     : [\"cough\", \"other\"]")


if __name__ == "__main__":
    main()
