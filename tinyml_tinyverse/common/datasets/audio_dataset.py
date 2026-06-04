import os
from glob import glob
from pathlib import Path
from logging import getLogger
from ast import literal_eval

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm


def _str2num_if_possible(value):
    if isinstance(value, str):
        value_strip = value.strip()

        if value_strip in ("None", "none"):
            return None
        if value_strip in ("True", "true"):
            return True
        if value_strip in ("False", "false"):
            return False

        try:
            return literal_eval(value_strip)
        except Exception:
            return value

    return value


def _normalize_transform_list(value):
    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()

        if value in ("", "None", "[]"):
            return []

        if value.startswith("[") or value.startswith("("):
            try:
                parsed = literal_eval(value)
                return _normalize_transform_list(parsed)
            except Exception:
                value = value.strip("[]()'\" ")
                return [value] if value else []

        if "," in value:
            output = []
            for item in value.split(","):
                output.extend(_normalize_transform_list(item))
            return output

        return [value.strip("'\" ")]

    if isinstance(value, np.ndarray):
        return _normalize_transform_list(value.tolist())

    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(_normalize_transform_list(item))
        return output

    return [str(value)]


class GoogleSpeechCommandsDataset(Dataset):
    """
    Google Speech Commands class-folder dataset loader.

    Expected layout:
        <dataset_dir>/
            down/
            go/
            left/
            no/
            off/
            on/
            right/
            stop/
            up/
            yes/
            _unknown_/

    Returns:
        X_raw[index], X[index], Y[index]

    X_raw:
        Raw processed waveform tensor, shape [1, 16000]

    X:
        Feature tensor.
        MFCC -> [1, time_frames, n_mfcc]
        LPC -> [1, time_frames, nlpc]
        RAW -> [1, 1, 16000]
    """

    _logger_name = "root.GoogleSpeechCommandsDataset"

    DEFAULT_LABELS = [
        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes",
        "_silence_",
        "_unknown_",
    ]

    def __init__(self, subset=None, dataset_dir=None, **kwargs):
        super().__init__()

        self.logger = getLogger(self._logger_name)

        self.subset = subset or "training"
        self._path = dataset_dir

        self.classes = []
        self.label_map = {}
        self.inverse_label_map = {}

        self.X_raw = []
        self.X = []
        self.Y = []

        self.file_paths = []
        self.file_names = []

        self.feature_extraction_params = {}
        self.audio_preprocessing_params = {}
        self.preprocessing_flags = []

        for key, value in kwargs.items():
            setattr(self, key, _str2num_if_possible(value))

        self.sampling_rate = int(getattr(self, "sampling_rate", 16000))
        self.audio_duration_ms = int(getattr(self, "audio_duration_ms", 1000))
        self.n_audio = int(self.sampling_rate * self.audio_duration_ms / 1000)

        self.audio_feature = getattr(self, "audio_feature", "MFCC")
        self.audio_feature = str(self.audio_feature).upper()

        self.feat_ext_transform = _normalize_transform_list(
            getattr(self, "feat_ext_transform", [])
        )

        self.data_proc_transforms = _normalize_transform_list(
            getattr(self, "data_proc_transforms", [])
        )

        self.n_mfcc = int(getattr(self, "n_mfcc", 10))
        self.n_mels = int(getattr(self, "n_mels", 40))
        self.frame_length_ms = int(getattr(self, "frame_length_ms", 30))
        self.frame_step_ms = int(getattr(self, "frame_step_ms", 20))

        self.nlpc = int(getattr(self, "nlpc", 14))
        self.lpc_order = int(getattr(self, "lpc_order", 14))

        self.normalize_audio = bool(getattr(self, "normalize_audio", True))
        self.mono = bool(getattr(self, "mono", True))

        self._walker = self._load_file_list(self.subset, kwargs)
        self.classes = self._get_classes()

        self.label_map = {
            class_name: class_index
            for class_index, class_name in enumerate(self.classes)
        }

        self.inverse_label_map = {
            class_index: class_name
            for class_name, class_index in self.label_map.items()
        }

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sampling_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": int(self.sampling_rate * self.frame_length_ms / 1000),
                "hop_length": int(self.sampling_rate * self.frame_step_ms / 1000),
                "n_mels": self.n_mels,
                "center": False,
            },
        )

    def _load_file_list(self, subset, kwargs):
        """
        Prefer annotation list if available, otherwise discover wav files directly.
        Mirrors GenericImageDataset-style behavior.
        """

        def load_list(kwargs_list_key, file_pattern):
            joined_path = os.path.join(
                os.path.dirname(self._path),
                "annotations",
                file_pattern,
            )

            candidates = glob(joined_path)

            if kwargs.get(kwargs_list_key):
                list_to_load = kwargs.get(kwargs_list_key)
            elif candidates:
                list_to_load = candidates[0]
            else:
                return None

            walker = []

            with open(list_to_load) as fileobj:
                for line in fileobj:
                    line = line.strip()
                    if line:
                        walker.append(os.path.join(self._path, line))

            return walker

        walker = None

        if subset in ["training", "train"]:
            walker = load_list("training_list", "*train*_list.txt")
        elif subset in ["testing", "test"]:
            walker = (
                load_list("testing_list", "*test*_list.txt")
                or load_list("testing_list", "*file*_list.txt")
            )
        elif subset in ["validation", "val"]:
            walker = load_list("validation_list", "*val*_list.txt")

        if walker is not None:
            return walker

        all_paths = []
        all_paths.extend(glob(os.path.join(self._path, "*", "*.wav")))
        all_paths.extend(glob(os.path.join(self._path, "*", "*.WAV")))

        all_paths = sorted(all_paths)

        if not all_paths:
            raise FileNotFoundError(
                f"No wav files found under {self._path}/<class>/*.wav"
            )

        return all_paths

    def _get_classes(self):
        discovered_classes = sorted(
            set([Path(datafile).parent.name for datafile in self._walker])
        )

        preferred_classes = [
            class_name
            for class_name in self.DEFAULT_LABELS
            if class_name in discovered_classes
        ]

        extra_classes = [
            class_name
            for class_name in discovered_classes
            if class_name not in preferred_classes
        ]

        classes = preferred_classes + sorted(extra_classes)

        if not classes:
            raise FileNotFoundError(f"No classes found under: {self._path}")

        return classes

    def _load_audio(self, file_path):
        waveform, source_sampling_rate = torchaudio.load(file_path)

        if self.mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if source_sampling_rate != self.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=source_sampling_rate,
                new_freq=self.sampling_rate,
            )

        waveform = waveform.squeeze(0).to(torch.float32)

        if waveform.numel() < self.n_audio:
            waveform = F.pad(waveform, (0, self.n_audio - waveform.numel()))
        else:
            waveform = waveform[:self.n_audio]

        if self.normalize_audio:
            max_value = torch.max(torch.abs(waveform)).item()
            if max_value > 0:
                waveform = waveform / max_value

        return waveform.unsqueeze(0)

    def _extract_features(self, raw_audio):
        if self.audio_feature == "MFCC":
            return self._extract_mfcc(raw_audio)

        if self.audio_feature == "LPC":
            return self._extract_lpc(raw_audio)

        if self.audio_feature == "RAW":
            return raw_audio.unsqueeze(0)

        raise ValueError(f"Unsupported audio_feature: {self.audio_feature}")

    def _extract_mfcc(self, raw_audio):
        mfcc = self.mfcc_transform(raw_audio)

        # torchaudio MFCC output: [channel, n_mfcc, time]
        # DSCNN-friendly output: [channel, time, n_mfcc]
        mfcc = mfcc.transpose(1, 2)

        return mfcc.to(torch.float32)

    def _extract_lpc(self, raw_audio):
        audio_np = raw_audio.squeeze(0).detach().cpu().numpy()

        frame_length = int(self.sampling_rate * self.frame_length_ms / 1000)
        frame_step = int(self.sampling_rate * self.frame_step_ms / 1000)

        frames = self._frame_audio(
            audio_np=audio_np,
            frame_length=frame_length,
            frame_step=frame_step,
        )

        lpc_features = self._compute_lpc_features(frames)

        return torch.tensor(lpc_features, dtype=torch.float32).unsqueeze(0)

    def _frame_audio(self, audio_np, frame_length, frame_step):
        if len(audio_np) < frame_length:
            audio_np = np.pad(audio_np, (0, frame_length - len(audio_np)))

        num_frames = 1 + int(np.ceil((len(audio_np) - frame_length) / frame_step))
        padded_length = (num_frames - 1) * frame_step + frame_length

        if padded_length > len(audio_np):
            audio_np = np.pad(audio_np, (0, padded_length - len(audio_np)))

        frames = []

        for start in range(0, padded_length - frame_length + 1, frame_step):
            frames.append(audio_np[start:start + frame_length])

        frames = np.stack(frames, axis=0)

        window = np.hamming(frame_length)
        frames = frames * window[None, :]

        return frames

    def _compute_lpc_features(self, frames):
        """
        Placeholder LPC feature extractor.

        Replace this with production LPC:
            cacorr
            c_levinson
            frq_resp
            log10f_fast
        """

        num_frames = frames.shape[0]
        features = np.zeros((num_frames, self.nlpc), dtype=np.float32)

        for frame_index in range(num_frames):
            frame = frames[frame_index].astype(np.float64)
            energy = np.mean(frame * frame)

            if energy <= 1e-12:
                features[frame_index, :] = 0.0
            else:
                features[frame_index, :] = np.log10(energy + 1e-6)

        return features

    def _prepare_audio_variables(self):
        self.preprocessing_flags = []

        self.audio_preprocessing_params["AUDIO_sampling_rate"] = self.sampling_rate
        self.audio_preprocessing_params["AUDIO_DURATION_MS"] = self.audio_duration_ms
        self.audio_preprocessing_params["AUDIO_NUM_SAMPLES"] = self.n_audio
        self.audio_preprocessing_params["AUDIO_NUM_CLASSES"] = len(self.classes)

        if self.audio_feature == "MFCC":
            self.preprocessing_flags.append("AUDIO_MFCC")
            self.audio_preprocessing_params["AUDIO_N_MFCC"] = self.n_mfcc
            self.audio_preprocessing_params["AUDIO_N_MELS"] = self.n_mels
            self.audio_preprocessing_params["AUDIO_FRAME_LENGTH_MS"] = self.frame_length_ms
            self.audio_preprocessing_params["AUDIO_FRAME_STEP_MS"] = self.frame_step_ms

        elif self.audio_feature == "LPC":
            self.preprocessing_flags.append("AUDIO_LPC")
            self.audio_preprocessing_params["AUDIO_NLPC"] = self.nlpc
            self.audio_preprocessing_params["AUDIO_LPC_ORDER"] = self.lpc_order
            self.audio_preprocessing_params["AUDIO_FRAME_LENGTH_MS"] = self.frame_length_ms
            self.audio_preprocessing_params["AUDIO_FRAME_STEP_MS"] = self.frame_step_ms

        elif self.audio_feature == "RAW":
            self.preprocessing_flags.append("AUDIO_RAW")

    def _prepare_feature_extraction_variables(self):
        self.feature_extraction_params.update(self.audio_preprocessing_params)

        if isinstance(self.X, torch.Tensor):
            self.feature_extraction_params["FE_STACKING_CHANNELS"] = self.X.shape[1]

            if self.X.ndim == 4:
                self.feature_extraction_params["FE_STACKING_FRAME_WIDTH"] = self.X.shape[3]
                self.feature_extraction_params["FE_HL"] = self.X.shape[2]
            elif self.X.ndim == 3:
                self.feature_extraction_params["FE_STACKING_FRAME_WIDTH"] = self.X.shape[2]
                self.feature_extraction_params["FE_HL"] = 1

        self.feature_extraction_params["FE_NN_OUT_SIZE"] = len(self.classes)

    def prepare(self, **kwargs):
        if not self._walker:
            raise FileNotFoundError(
                f"No wav files found under {self._path}/<class>/*.wav"
            )

        self.logger.info(
            f"[GoogleSpeechCommandsDataset] Found {len(self._walker)} wav files"
        )

        for file_path in tqdm(
            self._walker,
            desc=f"Loading {self.subset} audio",
            unit="wav",
        ):
            label_name = Path(file_path).parent.name

            raw_audio = self._load_audio(file_path)
            feature_tensor = self._extract_features(raw_audio)

            self.X_raw.append(raw_audio)
            self.X.append(feature_tensor)
            self.Y.append(self.label_map[label_name])
            self.file_paths.append(file_path)
            self.file_names.append(file_path)

        if not len(self.X):
            raise Exception("Aborting run as the audio dataset loaded is empty.")

        self.X_raw = torch.stack(self.X_raw)
        self.X = torch.stack(self.X)
        self.Y = torch.tensor(self.Y, dtype=torch.long)

        self.file_names = np.array(self.file_names)

        self._prepare_audio_variables()
        self._prepare_feature_extraction_variables()

        self.logger.info(
            f"Prepared GoogleSpeechCommandsDataset with {len(self.Y)} samples "
            f"and {len(self.classes)} classes"
        )
        self.logger.info(f"Classes: {self.classes}")
        self.logger.info(f"X_raw shape: {self.X_raw.shape}")
        self.logger.info(f"X shape: {self.X.shape}")
        self.logger.info(f"Y shape: {self.Y.shape}")

        return self

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X_raw[index], self.X[index], self.Y[index]
