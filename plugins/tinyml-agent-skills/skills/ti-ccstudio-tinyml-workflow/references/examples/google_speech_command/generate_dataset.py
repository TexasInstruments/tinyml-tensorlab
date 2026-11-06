import os
import shutil
from pathlib import Path

import torchaudio
from scipy.io import wavfile
from pydub import AudioSegment
import numpy as np
from tqdm import tqdm


KNOWN_LABELS = [
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes",
]

FINAL_LABELS = KNOWN_LABELS + ["_unknown_", "_silence_"]

BACKGROUND_NOISE_DIR_NAME = "_background_noise_"
SAMPLE_RATE = 16000


def copy_wav(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, dst_path)


def create_silence_samples(background_noise_dir: Path, silence_dir: Path):
    silence_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for wav_path in background_noise_dir.glob("*.wav"):
        audio = AudioSegment.from_wav(wav_path)
        audio_samples = np.array(audio.get_array_of_samples())

        for start in range(0, len(audio_samples) - SAMPLE_RATE, SAMPLE_RATE // 2):
            segment = audio_samples[start:start + SAMPLE_RATE]
            output_path = silence_dir / f"{wav_path.stem}_{start:06d}.wav"
            wavfile.write(output_path, SAMPLE_RATE, segment.astype(np.int16))
            count += 1

    return count


def prepare_speechcommands_class_folders(root=".", output_dir=None, force=False):
    root = Path(root)

    print("Downloading SpeechCommands if needed...")

    torchaudio.datasets.SPEECHCOMMANDS(
        root=str(root),
        url="speech_commands_v0.02",
        folder_in_archive="SpeechCommands",
        download=True,
    )
    print("Download finished.")
    raw_dir = root / "SpeechCommands" / "speech_commands_v0.02"

    if output_dir is None:
        output_root = root / "SpeechCommands" / "classes"
    else:
        output_root = Path(output_dir)

    if output_root.exists():
        if force:
            shutil.rmtree(output_root)
        else:
            print(f"Output already exists: {output_root}")
            return output_root

    for label in FINAL_LABELS:
        (output_root / label).mkdir(parents=True, exist_ok=True)

    print("Creating class-folder dataset...")

    copied_count = 0
    unknown_count = 0

    for label_dir in tqdm(list(raw_dir.iterdir()), desc="Processing labels"):
        if not label_dir.is_dir():
            continue

        label = label_dir.name

        if label == BACKGROUND_NOISE_DIR_NAME:
            continue

        target_label = label if label in KNOWN_LABELS else "_unknown_"

        for wav_path in label_dir.glob("*.wav"):
            if target_label == "_unknown_":
                dst_name = f"{label}_{wav_path.name}"
                unknown_count += 1
            else:
                dst_name = wav_path.name

            dst_path = output_root / target_label / dst_name
            copy_wav(wav_path, dst_path)
            copied_count += 1

    background_noise_dir = raw_dir / BACKGROUND_NOISE_DIR_NAME
    silence_count = 0

    if background_noise_dir.exists():
        print("Creating _silence_ samples from background noise...")
        silence_count = create_silence_samples(
            background_noise_dir=background_noise_dir,
            silence_dir=output_root / "_silence_",
        )

    print("\nDone.")
    print(f"Output dataset: {output_root.resolve()}")
    print(f"Known/unknown wavs copied: {copied_count}")
    print(f"Unknown-class wavs: {unknown_count}")
    print(f"Silence wavs created: {silence_count}")

    print("\nFinal structure:")
    print(f"{output_root}/")
    for label in FINAL_LABELS:
        count = len(list((output_root / label).glob('*.wav')))
        print(f"  {label}/  {count} wavs")

    return output_root


if __name__ == "__main__":
    prepare_speechcommands_class_folders(
        root=".",
        output_dir=None,
        force=False,
    )