import os
import sys
from glob import glob
from pathlib import Path
from logging import getLogger
from ast import literal_eval

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import cmsisdsp as dsp
from torch.utils.data import Dataset
from tqdm import tqdm


# ── Firmware-matched LPC constants ────────────────────────────────────────────
# 70 non-linearly spaced evaluation frequencies (Hz) matching lpc.c frqvec table
_LPC_FRQVEC = np.array([
     112,  136,  161,  187,  212,  239,  266,  294,  322,  351,
     380,  411,  442,  473,  505,  538,  572,  606,  641,  677,
     714,  751,  790,  829,  869,  910,  951,  994, 1038, 1082,
    1128, 1174, 1222, 1270, 1320, 1371, 1423, 1476, 1530, 1585,
    1641, 1699, 1758, 1819, 1880, 1943, 2008, 2073, 2141, 2209,
    2279, 2351, 2424, 2499, 2576, 2654, 2734, 2815, 2899, 2984,
    3071, 3160, 3251, 3344, 3439, 3536, 3635, 3737, 3840, 3946,
], dtype=np.float64)

# Q15 fixed-point Hamming window for 264 samples (8820 Hz, 30 ms frame)
# Exact match to HAMMING_FX table in lpc_constants.h
_LPC_HAMMING_FX_264 = np.array([
      2621,   2626,   2639,   2660,   2690,   2729,   2776,   2832,
      2896,   2969,   3050,   3139,   3237,   3343,   3457,   3579,
      3709,   3848,   3994,   4148,   4310,   4479,   4656,   4840,
      5032,   5231,   5437,   5650,   5870,   6097,   6330,   6570,
      6816,   7068,   7327,   7591,   7861,   8137,   8418,   8704,
      8996,   9292,   9594,   9900,  10210,  10525,  10844,  11166,
     11493,  11823,  12156,  12492,  12832,  13174,  13519,  13866,
     14215,  14567,  14920,  15275,  15631,  15988,  16346,  16705,
     17065,  17425,  17785,  18145,  18505,  18864,  19223,  19580,
     19937,  20292,  20646,  20999,  21349,  21697,  22043,  22387,
     22728,  23066,  23401,  23732,  24060,  24385,  24706,  25023,
     25335,  25643,  25947,  26246,  26540,  26829,  27113,  27391,
     27664,  27931,  28193,  28448,  28697,  28940,  29177,  29407,
     29630,  29847,  30056,  30259,  30454,  30642,  30823,  30996,
     31162,  31320,  31470,  31612,  31746,  31873,  31991,  32101,
     32203,  32296,  32381,  32458,  32527,  32587,  32638,  32681,
     32715,  32741,  32758,  32767,  32767,  32758,  32741,  32715,
     32681,  32638,  32587,  32527,  32458,  32381,  32296,  32203,
     32101,  31991,  31873,  31746,  31612,  31470,  31320,  31162,
     30996,  30823,  30642,  30454,  30259,  30056,  29847,  29630,
     29407,  29177,  28940,  28697,  28448,  28193,  27931,  27664,
     27391,  27113,  26829,  26540,  26246,  25947,  25643,  25335,
     25023,  24706,  24385,  24060,  23732,  23401,  23066,  22728,
     22387,  22043,  21697,  21349,  20999,  20646,  20292,  19937,
     19580,  19223,  18864,  18505,  18145,  17785,  17425,  17065,
     16705,  16346,  15988,  15631,  15275,  14920,  14567,  14215,
     13866,  13519,  13174,  12832,  12492,  12156,  11823,  11493,
     11166,  10844,  10525,  10210,   9900,   9594,   9292,   8996,
      8704,   8418,   8137,   7861,   7591,   7327,   7068,   6816,
      6570,   6330,   6097,   5870,   5650,   5437,   5231,   5032,
      4840,   4656,   4479,   4310,   4148,   3994,   3848,   3709,
      3579,   3457,   3343,   3237,   3139,   3050,   2969,   2896,
      2832,   2776,   2729,   2690,   2660,   2639,   2626,   2621,
], dtype=np.int16)


def _lpc_hamming_fx(window_len: int) -> np.ndarray:
    """Q15 Hamming window. Uses exact pre-computed table for 264-sample frames."""
    if window_len == 264:
        return _LPC_HAMMING_FX_264
    return np.round(np.hamming(window_len) * 32768).astype(np.int16)


def _lpc_log10f_fast(x: np.ndarray) -> np.ndarray:
    """Polynomial log2 approximation matching C firmware log2f_approx(), converted to log10."""
    x    = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    x    = np.abs(x)
    F, E = np.frexp(x)
    Y  = 1.23149591368684  * F
    Y += -4.11852516267426
    Y *= F
    Y += 6.02197014179219
    Y *= F
    Y += -3.13396450166353
    Y += E
    return sign * Y * 0.3010299956639812   # log2 → log10


def _lpc_levinson(r: np.ndarray) -> tuple:
    """Levinson-Durbin recursion matching C firmware levinson() / scikits.talkbox c_levinson."""
    order = len(r) - 1
    a   = np.zeros(order + 1, dtype=np.float64)
    tmp = np.zeros(order,     dtype=np.float64)
    k   = np.zeros(order,     dtype=np.float64)
    a[0] = 1.0
    err  = r[0]
    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k[i - 1]  = -acc / err
        a[i]      = k[i - 1]
        tmp[:order] = a[:order]
        for j in range(1, i):
            a[j] += k[i - 1] * tmp[i - j]
        err *= 1.0 - k[i - 1] ** 2
    return a, err


def _lpc_preprocess(audio_np: np.ndarray, window_len: int, frame_len: int) -> np.ndarray:
    """
    Prepend overlap zeros, clip to int16 range, frame, and apply Q15 Hamming window.
    Parametric port of firmware_lpc.preprocess() — works for any sample rate.
    Returns int64 array shape (num_frames, window_len).
    """
    overlap = window_len - frame_len
    if overlap < 0:
        raise ValueError(
            f"frame_len ({frame_len} samples) must not exceed window_len ({window_len} samples). "
            f"Reduce frame_step_ms or increase frame_length_ms."
        )
    y = np.concatenate([np.zeros(overlap, dtype=np.float64), audio_np])
    y = np.clip(y, -32768, 32767)
    n = y.shape[0]
    num_frames = max(1, 1 + (n - window_len) // frame_len)
    padded = num_frames * frame_len + overlap
    if padded > n:
        y = np.concatenate([y, np.zeros(padded - n, dtype=np.float64)])
    frames = np.stack([
        y[i * frame_len: i * frame_len + window_len]
        for i in range(num_frames)
    ])
    hamming_fx = _lpc_hamming_fx(window_len)
    frames = (frames.astype(np.int64) * hamming_fx.astype(np.int64) + 16384) // 32768
    return frames.astype(np.int64)


def _lpc_compute_features(frames: np.ndarray, fs: float, lpc_order: int = 10) -> np.ndarray:
    """
    LPC feature extraction matching C firmware get_lpc() / lpc_70.py lpc().

    Pipeline: autocorrelation → normalize → noise-scale (1/1.1) →
    Levinson-Durbin → freqz at 70 bins → 10*log10f_fast.

    Args:
        frames:    int64 array (num_frames, window_len) from _lpc_preprocess()
        fs:        sample rate in Hz (used for frequency normalisation)
        lpc_order: LPC model order (default 10, matches firmware LPC_ORDER)
    Returns:
        float32 array (num_frames, 70) in dB
    """
    _LPC_ORDER = lpc_order
    nfrm, window_len = frames.shape
    f = frames.astype(np.float64)

    # Autocorrelation, lags 0..order
    r = np.zeros((nfrm, _LPC_ORDER + 1), dtype=np.float64)
    for lag in range(_LPC_ORDER + 1):
        r[:, lag] = np.sum(f[:, :window_len - lag] * f[:, lag:], axis=1)

    # Normalize so r[:,0] == 1
    acorr_max = r[:, 0].copy()
    acorr_max[acorr_max == 0.0] = 0.001
    r /= acorr_max[:, None]

    # Noise scaling — matches LPC_ACORR_NOISE_SCALE = 1/1.1 in C firmware
    r[:, 1:] *= (1.0 / 1.1)

    # Levinson-Durbin per frame
    a_lpc = np.zeros((nfrm, _LPC_ORDER + 1), dtype=np.float64)
    G_vec = np.zeros(nfrm, dtype=np.float64)
    for i in range(nfrm):
        a, _ = _lpc_levinson(r[i])
        a_lpc[i] = a
        G_vec[i] = np.sqrt(max(np.dot(a, r[i]), 1e-30))

    # Frequency response magnitude at 70 evaluation points
    lags   = np.arange(1, _LPC_ORDER + 1, dtype=np.float64)
    angles = -2.0 * np.pi * _LPC_FRQVEC[:, None] * lags[None, :] / fs
    z_real = np.cos(angles)   # (70, 10)
    z_imag = np.sin(angles)   # (70, 10)

    a_c         = a_lpc[:, 1:]                          # (nfrm, 10)
    denom_real  = 1.0 + a_c.dot(z_real.T)               # (nfrm, 70)
    denom_imag  = a_c.dot(z_imag.T)                     # (nfrm, 70)
    denom_sq    = np.maximum(denom_real**2 + denom_imag**2, 1e-30)
    mag         = (G_vec[:, None] ** 2) / denom_sq      # (nfrm, 70)

    features = 10.0 * _lpc_log10f_fast(np.maximum(mag, 1e-30))
    return features.astype(np.float32)


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


class GenericAudioDataset(Dataset):
    """
    Generic folder-based audio classification dataset.

    Expected layout:
        <dataset_dir>/
            <class_a>/
            <class_b>/
            ...

    Returns:
        X_raw[index], X[index], Y[index]

    X_raw:
        Raw processed waveform tensor, shape [1, n_audio]

    X:
        Feature tensor.
        MFCC -> [1, time_frames, n_mfcc]
        LPC  -> [1, time_frames, nlpc]
        FB   -> [1, n_audio, 1]             (n_audio contains int   values)
        RAW  -> [1, 1, n_audio]             (n_audio contains float values)
    """

    _logger_name = "root.GenericAudioDataset"

    def __init__(self, subset=None, dataset_dir=None, **kwargs):
        super().__init__()

        self.logger = getLogger(self._logger_name)

        self.subset = subset or "training"
        self._path = dataset_dir

        self.classes = []
        self.label_map = {}
        self.inverse_label_map = {}

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

        self.frame_size = int(getattr(self, "frame_size", 512))
        self.feature_size_per_frame = int(getattr(self, "feature_size_per_frame", 32))
        self.num_frame_concat = int(getattr(self, "num_frame_concat", 1))
        self.min_bin = int(getattr(self, "min_bin", 1))
        self.q15_scale_factor = int(getattr(self, "q15_scale_factor", 8))
        self.bin_size = self.frame_size // 2 // self.feature_size_per_frame

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
        all_paths.extend(glob(os.path.join(self._path, "*", "*.flac")))
        all_paths.extend(glob(os.path.join(self._path, "*", "*.FLAC")))

        all_paths = sorted(all_paths)

        if not all_paths:
            raise FileNotFoundError(
                f"No wav files found under {self._path}/<class>/*.wav"
            )

        return all_paths

    def _get_classes(self):
        classes = sorted(
            set([Path(datafile).parent.name for datafile in self._walker])
        )

        if not classes:
            raise FileNotFoundError(f"No classes found under: {self._path}")

        return classes

    def _load_audio(self, file_path):
        # soundfile (libsndfile) instead of torchaudio.load: torchaudio >=2.9 routes
        # load() through torchcodec, which needs FFmpeg shared libs matching one of a
        # handful of exact ABI versions -- an extra native dependency this plain WAV
        # read doesn't need.
        if self.audio_feature == "FB" or self.audio_feature == "FFT":
            data, source_sampling_rate = sf.read(file_path, dtype="int16", always_2d=True)
        else:
            data, source_sampling_rate = sf.read(file_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)

        if self.mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if source_sampling_rate != self.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=source_sampling_rate,
                new_freq=self.sampling_rate,
            )

        n_samples = waveform.shape[-1]
        if n_samples < self.n_audio:
            waveform = F.pad(waveform, (0, self.n_audio - n_samples))
        else:
            waveform = waveform[..., :self.n_audio]

        if self.normalize_audio and self.audio_feature != "FB":
            max_value = torch.max(torch.abs(waveform)).item()
            if max_value > 0:
                waveform = waveform / max_value

        return waveform

    def _transform_to_q15_fast(self, wave_frame):
        scaled = np.round(np.asarray(wave_frame, dtype=np.float32) * 32768.0)
        result_wave = np.clip(scaled, -32768, 32767).astype(np.int16)
        return wave_frame, result_wave

    def _transform_fft_q15(self, wave_frame, frame_size):
        inst = dsp.arm_rfft_instance_q15()
        dsp.arm_rfft_init_q15(inst, frame_size, 0, 1)
        result_wave = dsp.arm_rfft_q15(inst, wave_frame)
        return wave_frame, result_wave

    def _transform_q15_scale(self, wave_frame, q15_scale_factor):
        result_wave = dsp.arm_shift_q15(wave_frame, q15_scale_factor)
        return wave_frame, result_wave

    def _transform_q15_cmplx_mag(self, wave_frame, frame_size):
        result_wave = dsp.arm_cmplx_mag_q15(wave_frame[:frame_size + 2])
        return wave_frame, result_wave

    def _transform_binning_q15_fast(self, wave_frame, feature_size, fft_bin_size, min_bin):
        start_idx = min_bin
        end_idx = start_idx + (feature_size * fft_bin_size)
        sub_matrix = wave_frame[start_idx:end_idx].reshape(feature_size, fft_bin_size)
        sums = np.sum(sub_matrix, axis=1)
        avg = sums // fft_bin_size
        result_wave = np.clip(avg, -32768, 32767).astype(np.int16)
        return wave_frame, result_wave

    def _transform_fft_concatenation(self, features_of_frame, index_frame, num_frame_concat):
        result_wave = np.array(features_of_frame[index_frame - num_frame_concat + 1: index_frame + 1]).flatten()
        return features_of_frame, result_wave

    def _extract_features(self, raw_audio):
        if self.audio_feature == "MFCC":
            return self._extract_mfcc(raw_audio)

        if self.audio_feature == "LPC":
            return self._extract_lpc(raw_audio)

        if self.audio_feature == "FFT":
            return self._extract_fft(raw_audio)

        if self.audio_feature == "FB":
            return self._extract_filterbank(raw_audio)

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
        audio_np = raw_audio.squeeze(0).detach().cpu().numpy().astype(np.float64)

        # Scale to int16 range — matches firmware ISR (raw >> 10, clamp)
        if np.abs(audio_np).max() <= 1.0:
            audio_np = np.round(audio_np * 32767.0)
        audio_np = np.clip(audio_np, -32768, 32767)

        # Pre-emphasis: y[n] = x[n] - x[n-1]  (matches lfilter([1,-1],1,y) with zero IC)
        audio_pe = np.empty_like(audio_np)
        audio_pe[0]  = audio_np[0]
        audio_pe[1:] = audio_np[1:] - audio_np[:-1]

        window_len = int(self.sampling_rate * self.frame_length_ms / 1000)
        frame_len  = int(self.sampling_rate * self.frame_step_ms  / 1000)

        frames   = _lpc_preprocess(audio_pe, window_len, frame_len)
        features = _lpc_compute_features(frames, float(self.sampling_rate), self.lpc_order)

        if features.shape[-1] != self.nlpc:
            raise ValueError(
                f"LPC feature extraction produced {features.shape[-1]} features "
                f"(len(_LPC_FRQVEC)=70), but nlpc={self.nlpc}. Set nlpc=70 to match the firmware."
            )

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _extract_filterbank(self, raw_audio):
        data_int16 = raw_audio.to(torch.int16)
        shift = 16 - self.input_bit_depth
        return (data_int16 >> shift).unsqueeze(-1)

    def _extract_fft(self, raw_audio):
        audio_np = raw_audio.squeeze(0).detach().cpu().numpy().astype(np.float32)

        length = len(audio_np)
        num_frame = length // self.frame_size
        audio_np = audio_np[:num_frame * self.frame_size].reshape(num_frame, self.frame_size)

        features_of_frame = []

        for frame_idx in range(num_frame):
            wave_frame = audio_np[frame_idx, :].astype(np.float32)

            if 'TO_Q15' in self.feat_ext_transform:
                _, wave_frame = self._transform_to_q15_fast(wave_frame)
            if 'FFT_Q15' in self.feat_ext_transform:
                _, wave_frame = self._transform_fft_q15(wave_frame, self.frame_size)
            if 'Q15_SCALE' in self.feat_ext_transform:
                _, wave_frame = self._transform_q15_scale(wave_frame, self.q15_scale_factor)
            if 'Q15_MAG' in self.feat_ext_transform:
                _, wave_frame = self._transform_q15_cmplx_mag(wave_frame, self.frame_size)
            if 'BINNING' in self.feat_ext_transform:
                _, wave_frame = self._transform_binning_q15_fast(wave_frame, self.feature_size_per_frame, self.bin_size, self.min_bin)

            features_of_frame.append(wave_frame)

        features_array = np.array(features_of_frame)
        return torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)

    def _prepare_audio_variables(self):
        self.preprocessing_flags = []

        self.audio_preprocessing_params["FE_NN_OUT_SIZE"] = len(self.classes)

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

        elif self.audio_feature == "FFT":
            self.audio_preprocessing_params["FE_VARIABLES"] = int(getattr(self, "variables", 1))
            self.audio_preprocessing_params["FE_FRAME_SIZE"] = self.frame_size
            if 'FFT_Q15' in self.feat_ext_transform:
                self.preprocessing_flags.append("FE_RFFT")
            if 'Q15_SCALE' in self.feat_ext_transform:
                self.audio_preprocessing_params["FE_COMPLEX_MAG_SCALE_FACTOR"] = self.q15_scale_factor
                self.preprocessing_flags.append("FE_COMPLEX_MAG_SCALE")
            if 'Q15_MAG' in self.feat_ext_transform:
                self.preprocessing_flags.append("FE_MAG")
            if 'BINNING' in self.feat_ext_transform:
                self.preprocessing_flags.append("FE_BIN")
                self.audio_preprocessing_params["FE_FEATURE_SIZE_PER_FRAME"] = self.feature_size_per_frame
                self.audio_preprocessing_params["FE_BIN_SIZE"] = self.bin_size
                self.audio_preprocessing_params['FE_BIN_OFFSET'] = self.min_bin
                self.audio_preprocessing_params['FE_BIN_NORMALIZE'] = 0
            if 'CONCAT' in self.feat_ext_transform:
                self.wl = self.audio_preprocessing_params['FE_FEATURE_SIZE_PER_FRAME'] * self.num_frame_concat
                self.audio_preprocessing_params['FE_STACKING_FRAME_WIDTH'] = self.wl

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

        self.logger.info(f"Found {len(self._walker)} wav files")

        # Configure Cache Directory specifically for FFT feature processing
        self.use_fft_cache = getattr(self, "use_fft_cache", True)
        self.cache_dir = os.path.join(os.path.dirname(self._path), "fft_cache_dir")
        if self.audio_feature == "FFT" and self.use_fft_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.logger.info(f"FFT On-Disk Cache activated at: {self.cache_dir}")

        # 1. Handle Lazy Loading & Caching for FFT feature explicitly
        if self.audio_feature == "FFT":
            self.logger.info("FFT detected: Enabling memory-optimized Lazy Loading with Caching.")
            
            for file_path in self._walker:
                label_name = Path(file_path).parent.name
                self.Y.append(self.label_map[label_name])
                self.file_paths.append(file_path)
                self.file_names.append(file_path)

            if not len(self.file_paths):
                raise Exception("Aborting run as the audio dataset loaded is empty.")

            self.Y = torch.tensor(self.Y, dtype=torch.long)
            self.file_names = np.array(self.file_names)

            # Assign the caching lazy virtual arrays
            self.X_raw = LazyTensorArray(self, is_feature=False)
            self.X = LazyTensorArray(self, is_feature=True)

        # 2. Keep the original In-Memory behavior for MFCC, LPC, RAW, FB
        else:
            self.logger.info(f"{self.audio_feature} detected: Loading entire dataset into RAM.")
            self.X = []
            self.X_raw = []
            
            for file_path in tqdm(self._walker, desc=f"Loading {self.subset} audio", unit="wav"):
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

        # Run setup configuration variables
        self._prepare_audio_variables()
        self._prepare_feature_extraction_variables()

        self.logger.info(f"Prepared GenericAudioDataset with {len(self.Y)} samples")
        self.logger.info(f"X_raw shape: {self.X_raw.shape}")
        self.logger.info(f"X shape: {self.X.shape}")
        return self

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # Keeps processing memory footprint contained dynamically
        if self.audio_feature == "FFT":
            return self.X_raw[index], self.X[index], self.Y[index]
        else:
            return self.X_raw[index], self.X[index], self.Y[index]


# ── Smart Caching Virtual Array Wrapper ───────────────────────────────────────

class LazyTensorArray:
    """Virtual array wrapper providing memory safety, indexing, and on-disk caching."""
    def __init__(self, dataset, is_feature=False):
        self._dataset = dataset
        self._is_feature = is_feature
        
        # Calculate dimension signatures on a single mock run
        sample_raw = dataset._load_audio(dataset.file_paths[0])
        if not self._is_feature:
            self.shape = (len(dataset.file_paths), *sample_raw.shape)
        else:
            sample_feat = dataset._extract_features(sample_raw)
            self.shape = (len(dataset.file_paths), *sample_feat.shape)
        
    def __getitem__(self, index):
        file_path = self._dataset.file_paths[index]
        
        # Raw files are loaded instantly from disk natively 
        if not self._is_feature:
            return self._dataset._load_audio(file_path)
            
        # Feature Processing with Persistent Disk Caching
        if self._dataset.use_fft_cache:
            # Create a unique filename based on the source path hash or path structure
            safe_filename = f"feat_{Path(file_path).parent.name}_{Path(file_path).stem}.pt"
            cache_path = os.path.join(self._dataset.cache_dir, safe_filename)
            
            try:
                if os.path.exists(cache_path):
                    return torch.load(cache_path, weights_only=True)  # Instantly read cache
            except RuntimeError:
                pass

            # Cache miss: Compute features, save to disk, then return
            raw_audio = self._dataset._load_audio(file_path)
            feature_tensor = self._dataset._extract_features(raw_audio)
            torch.save(feature_tensor, cache_path)
            return feature_tensor
        else:
            # Cache explicitly turned off: Calculate dynamically
            raw_audio = self._dataset._load_audio(file_path)
            return self._dataset._extract_features(raw_audio)
