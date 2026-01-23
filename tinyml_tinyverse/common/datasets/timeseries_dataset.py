#################################################################################
# Copyright (c) 2023-2024, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#################################################################################

import os
from glob import glob
from logging import getLogger
from os.path import basename as opb
from os.path import dirname as opd
from os.path import splitext as ops

from ast import literal_eval
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import yaml
import cmsisdsp as dsp
from scipy.stats import kurtosis
from scipy.stats import entropy

from ..augmenters import AddNoise, Convolve, Crop, Drift, Dropout, Pool, Quantize, Resize, Reverse, TimeWarp
from ..transforms import basic_transforms
from ..transforms.haar import haar_forward
from ..transforms.hadamard import hadamard_forward_vectorized
from ..utils.misc_utils import str2int_or_float, str2bool, str2bool_or_none
from ..utils.mdcl_utils import create_dir


# Define a function to create augmenters
def create_augmenter(name, args):
    logger = getLogger("root.timeseries_dataset.create_augmenter")
    augmenters = dict(AddNoise=AddNoise, Convolve=Convolve, Crop=Crop, Drift=Drift, Dropout=Dropout,
                      Pool=Pool, Quantize=Quantize, Resize=Resize, Reverse=Reverse, TimeWarp=TimeWarp)
    if name not in augmenters:
        logger.warning(f"Skipping unknown augmenter: {name}")
    return augmenters[name](**args)


def apply_augmenters(data, augmenter_pipeline):
    augmented_data = data
    for augmenter in augmenter_pipeline:
        augmented_data = augmenter.augment(augmented_data)
    return augmented_data


class BaseGenericTSDataset(Dataset):
    """
    Base class for all timeseries datasets. Contains common functionality for:
    - Initialization and configuration loading
    - File loading (supports .npy, .pkl, .csv, .txt)
    - Data transforms (FFT, windowing, binning, etc.)
    - Feature extraction pipeline
    - Augmentation pipeline

    Subclasses should override:
    - _load_datafile(): For task-specific data loading
    - _process_targets(): For task-specific target processing
    - __getitem__(): For task-specific item retrieval
    - _get_classes(): For task-specific class extraction
    """

    # Class attribute to identify the logger name for each subclass
    _logger_name = "root.BaseGenericTSDataset"

    def __init__(self, subset: str = None, dataset_dir: str = None, **kwargs):
        self.logger = getLogger(self._logger_name)
        self._path = dataset_dir
        self.classes = list()
        self.label_map = dict()
        self.feature_extraction_params = dict()
        self.preprocessing_flags = []
        self.X = []
        self.Y = []
        self.file_names = []  # Stores file name of each sample

        # Store all the kwargs in self
        for key, value in kwargs.items():
            value = str2int_or_float(value)
            setattr(self, key, value)

        # Parse variables parameter to support multiple formats
        if hasattr(self, 'variables'):
            self._parse_variables_parameter()

        # Setup augmentation pipeline
        self.augment_pipeline = []
        if hasattr(self, 'augment_config'):
            if str2int_or_float(self.augment_config) and os.path.exists(self.augment_config):
                self.logger.info(f"Parsing {self.augment_config} to form Augmentation Pipeline")
                try:
                    with open(self.augment_config) as fp:
                        augment_config = yaml.load(fp, Loader=yaml.CLoader)
                except yaml.YAMLError as exc:
                    self.logger.critical(f"{exc} error parsing {self.augment_config}")
                # Create a pipeline of augmenters
                self.augment_pipeline = [create_augmenter(name, params) for name, params in augment_config.items()]

        # Flatten transforms list
        self.transforms = np.array(self.transforms).flatten().tolist()

        # Load file list based on subset
        self._walker = self._load_file_list(subset, kwargs)

        # Get classes from the dataset
        self.classes = self._get_classes()

    def _parse_variables_parameter(self):
        """
        Parse variables parameter to support multiple formats.
        Sets:
        - self._variables_format: 'integer', 'list_of_indices', or 'list_of_names'
        - self.selected_column_indices: List of indices (or None for column names)
        - self.selected_column_names: List of names (or None for integer/indices)
        """
        # Handle string representation of list
        if isinstance(self.variables, str) and self.variables.startswith('['):
            try:
                self.variables = literal_eval(self.variables)
            except (ValueError, SyntaxError) as e:
                self.logger.error(f"Failed to parse variables: {self.variables}")
                raise ValueError(f"Invalid variables format: {self.variables}")

        # Integer format (current behavior)
        if isinstance(self.variables, int):
            self._variables_format = 'integer'
            self.selected_column_indices = list(range(self.variables))
            self.selected_column_names = None
            return

        # List format
        if isinstance(self.variables, list):
            if len(self.variables) == 0:
                raise ValueError("variables parameter cannot be an empty list")

            is_int = lambda x: isinstance(x, int) or (isinstance(x, str) and x.isdigit())

            # List of integer indices
            if all(is_int(x) for x in self.variables):
                self._variables_format = 'list_of_indices'
                self.selected_column_indices = [int(x) for x in self.variables]
                self.selected_column_names = None
                self.variables = len(self.selected_column_indices)
            # List of column names
            elif all(isinstance(x, str) and not x.isdigit() for x in self.variables):
                self._variables_format = 'list_of_names'
                self.selected_column_names = list(self.variables)
                self.selected_column_indices = None  # Resolved during file load
                self.variables = len(self.selected_column_names)
            else:
                raise ValueError(f"variables list must be all integers or all names, not mixed: {self.variables}")
            return

        raise TypeError(f"variables must be int or list, got {type(self.variables)}")

    def _resolve_column_selection(self, data, datafile, has_header, is_npy=False):
        """
        Resolve column names to indices and validate column selections.

        Returns: List of valid column indices (0-indexed after time removal)
        Raises: ValueError if no valid columns found
        """
        # List of column names
        if self._variables_format == 'list_of_names':
            if is_npy:
                raise ValueError(
                    f"Column names not supported for .npy files. "
                    f"Use integer indices. File: {datafile}"
                )

            if not has_header:
                raise ValueError(
                    f"Column names specified but file has no header. "
                    f"File: {datafile}, Requested: {self.selected_column_names}"
                )

            # Get header row (data has been processed to remove time columns)
            header_list = list(data.columns) if isinstance(data, pd.DataFrame) else data.iloc[0].tolist()
            resolved_indices = []
            invalid_columns = []

            for col_name in self.selected_column_names:
                try:
                    idx = header_list.index(col_name)
                    resolved_indices.append(idx)
                except ValueError:
                    invalid_columns.append(col_name)
                    self.logger.warning(
                        f"Column '{col_name}' not found in {datafile}. "
                        f"Available: {header_list}"
                    )

            if len(resolved_indices) == 0:
                raise ValueError(
                    f"None of the specified columns found in {datafile}. "
                    f"Requested: {self.selected_column_names}, Available: {header_list}"
                )

            if invalid_columns:
                self.logger.warning(
                    f"Continuing with {len(resolved_indices)}/{len(self.selected_column_names)} "
                    f"valid columns. Skipped: {invalid_columns}"
                )

            # Update variables count to actual extracted columns
            self.variables = len(resolved_indices)
            return resolved_indices

        # List of integer indices
        elif self._variables_format == 'list_of_indices':
            max_cols = data.shape[1]
            valid_indices = []
            invalid_indices = []

            for idx in self.selected_column_indices:
                if idx < max_cols:
                    valid_indices.append(idx)
                else:
                    invalid_indices.append(idx)
                    self.logger.warning(
                        f"Column index {idx} exceeds available columns ({max_cols}) "
                        f"in {datafile}. Skipping."
                    )

            if len(valid_indices) == 0:
                raise ValueError(
                    f"None of the specified column indices are valid in {datafile}. "
                    f"Requested: {self.selected_column_indices}, Available: 0-{max_cols-1}"
                )

            if invalid_indices:
                self.logger.warning(
                    f"Continuing with {len(valid_indices)}/{len(self.selected_column_indices)} "
                    f"valid columns"
                )

            # Update variables count to actual extracted columns
            self.variables = len(valid_indices)
            return valid_indices

        # Integer format (current behavior)
        else:
            return list(range(self.variables))

    def _load_file_list(self, subset, kwargs):
        """Load the list of data files based on subset (train/test/val)."""
        def load_list(kwargs_list, file_name):
            joined_path = os.path.join(os.path.dirname(self._path), 'annotations', file_name)
            list_to_load = glob(joined_path)[0]
            if kwargs.get(kwargs_list):
                list_to_load = kwargs.get(kwargs_list)
            loadList = []
            with open(list_to_load) as fileobj:
                for line in fileobj:
                    line = line.strip()
                    if line:
                        loadList.append(os.path.join(self._path, line))
            return loadList

        if subset in ["training", "train"]:
            kwargs_list_key = 'training_list'
            list_filename = '*train*_list.txt'
        elif subset in ["testing", "test"]:
            kwargs_list_key = 'testing_list'
            try:
                list_filename = '*test*_list.txt'
                return load_list(kwargs_list_key, list_filename)
            except IndexError:
                list_filename = '*file*_list.txt'
                return load_list(kwargs_list_key, list_filename)
        elif subset in ["validation", "val"]:
            kwargs_list_key = 'validation_list'
            list_filename = '*val*_list.txt'

        return load_list(kwargs_list_key, list_filename)

    def _get_classes(self):
        """Extract classes from the dataset. Override in subclasses if needed."""
        return sorted(set([opb(opd(datafile)) for datafile in self._walker]))

    # ==================== Data Processing Transforms ====================

    def _transform_downsample(self, x_temp):
        """Downsample the dataset by a factor (sampling_rate/new_sr)."""
        return basic_transforms.Downsample(x_temp, self.sampling_rate, self.new_sr)

    def _transform_simple_window(self, x_temp):
        """Form windows of size frame_size from the dataset."""
        stride_window = int(self.frame_size * self.stride_size)
        if stride_window == 0:
            self.logger.warning("Stride Window size calculated is 0. Defaulting the value to Sample Window i.e. no overlap.")
            stride_window = self.frame_size
        if not hasattr(self, 'keep_short_tails'):
            self.keep_short_tails = False
        x_temp = np.array(basic_transforms.SimpleWindow(x_temp, self.frame_size, stride_window, self.keep_short_tails))
        if len(self.feat_ext_transform) > 0:
            x_temp = x_temp.reshape(-1, self.variables)
        return x_temp

    def _transform_gain(self, array, label):
        """Apply a gain factor to the array."""
        try:
            gain_variations = literal_eval(self.gain_variations)
            if label in gain_variations.keys():
                gain = random.uniform(gain_variations[label][0], gain_variations[label][1])
                array = array * gain
            else:
                self.logger.warning(f"Dataset class:{label} not found in input gain variations: {gain_variations.keys()}")
        except ValueError as e:
            self.logger.warning(f"Gain Variations couldn't be applied: {e}")
        return array

    # ==================== Feature Extraction Transforms ====================

    def _transform_raw(self, wave_frame):
        result_wave = wave_frame
        mean_of_wave = np.sum(result_wave) // len(result_wave)
        if self.dc_remove:
            result_wave = result_wave - mean_of_wave
        return wave_frame, result_wave

    def _transform_windowing(self, wave_frame):
        result_wave = wave_frame * np.hanning(self.frame_size)
        return wave_frame, result_wave

    def _transform_fft(self, wave_frame):
        result_wave = np.fft.fft(wave_frame)
        return wave_frame, result_wave

    def _transform_binning(self, wave_frame):
        result_wave = np.empty((self.feature_size_per_frame))
        bin_size = self.bin_size
        bin_offset = self.min_bin if self.min_bin else 0

        for idx in range(self.feature_size_per_frame):
            idx1 = bin_offset + (idx) * bin_size
            sum_of_bin = np.sum(wave_frame[idx1:idx1 + bin_size])
            result_wave[idx] = sum_of_bin
            result_wave[idx] /= bin_size if self.normalize_bin else 1
        return wave_frame, result_wave

    def _transform_concatenation(self, features_of_frame_per_ax, index_frame):
        result_wave = np.array(features_of_frame_per_ax[index_frame - self.num_frame_concat + 1: index_frame + 1]).flatten()
        return features_of_frame_per_ax, result_wave

    def _transform_pos_half_fft(self, wave_frame):
        """Takes the DC + min_bin samples of the fft."""
        idx = self.frame_size // 2
        idx += self.min_bin if self.min_bin else 1
        result_wave = wave_frame[:idx]
        return wave_frame, result_wave

    def _transform_absolute(self, wave_frame):
        """Converts the complex value to real values."""
        result_wave = abs(wave_frame)
        return wave_frame, result_wave

    def _transform_haar(self, wave_frame):
        result_wave = haar_forward(wave_frame)
        return wave_frame, result_wave

    def _transform_hadamard(self, wave_frame):
        result_wave = hadamard_forward_vectorized(wave_frame)
        return wave_frame, result_wave

    def _transform_normalize(self, wave_frame):
        result_wave = wave_frame / self.frame_size
        return wave_frame, result_wave
    
    def _transform_roundoff(self, wave_frame):
        result_wave = np.round((np.array(wave_frame)))
        return wave_frame, result_wave

    def _transform_log(self, wave_frame):
        if self.log_base == 'e':
            result_wave = self.log_mul * np.log(self.log_threshold + wave_frame)
        else:
            try:
                base = float(self.log_base)
                result_wave = self.log_mul * np.log(self.log_threshold + wave_frame) / np.log(base)
            except (ValueError, TypeError):
                self.logger.warning("log_base can't be converted to float, default to base 10")
                result_wave = self.log_mul * np.log10(self.log_threshold + wave_frame)
        return wave_frame, result_wave

    def _transform_to_q15(self, wave_frame):
        def q15sat(x):
            if x > 0x7FFF:
                return np.int16(0x7FFF)
            elif x < -0x8000:
                return np.int16(-0x8000)
            else:
                return np.int16(x)
        q15satV = np.vectorize(q15sat)
        def toQ15(x):
            return q15satV(np.round(np.array(x) * (1 << 15)))
        result_wave = toQ15(wave_frame)
        return wave_frame, result_wave

    def _transform_binning_q15(self, wave_frame):
        result_wave = np.empty((self.feature_size_per_frame))
        bin_size = self.bin_size
        bin_offset = self.min_bin if self.min_bin else 0
        for idx in range(self.feature_size_per_frame):
            idx1 = bin_offset + (idx) * bin_size
            sum_of_bin = np.sum(wave_frame[idx1:idx1 + bin_size])
            avg = sum_of_bin // (bin_size if self.normalize_bin else 1)
            result_wave[idx] = np.clip(avg, -32768, 32767).astype(np.int16)
        return wave_frame, result_wave

    def _transform_fft_q15(self, wave_frame):
        inst = dsp.arm_rfft_instance_q15()
        dsp.arm_rfft_init_q15(inst, self.frame_size, 0, 1)
        result_wave = dsp.arm_rfft_q15(inst, wave_frame)
        return wave_frame, result_wave

    def _transform_q15_scale(self, wave_frame):
        result_wave = dsp.arm_shift_q15(wave_frame, self.q15_scale_factor)
        return wave_frame, result_wave

    def _transform_q15_cmplx_mag(self, wave_frame):
        result_wave = dsp.arm_cmplx_mag_q15(wave_frame[:self.frame_size + 2])
        return wave_frame, result_wave

    def _transform_kurtosis(self, wave_frame):
        wave_frame = np.array(wave_frame, dtype=np.float32)
        padded_frame = np.pad(wave_frame, (4, 3), mode='constant', constant_values=0)
        result_wave = []
        stride_window = int(self.frame_size * self.stride_size)
        window_length = len(padded_frame) - (self.window_count * stride_window)
        for i in range(self.window_count):
            start = i * stride_window
            end = start + window_length
            if end > len(padded_frame):
                break
            window = padded_frame[start:end]
            window_kurtosis = []
            chunk_count = window_length // self.chunk_size
            for j in range(chunk_count):
                chunk_start = j * self.chunk_size
                chunk_end = chunk_start + self.chunk_size
                chunk = window[chunk_start:chunk_end]
                eps = 1e-12
                if np.nanstd(chunk) < eps:
                    chunk_kurtosis = 0.0
                else:
                    chunk_kurtosis = kurtosis(chunk, fisher=True, bias=False)
                window_kurtosis.append(chunk_kurtosis)
            result_wave.append(window_kurtosis)
        result_wave = np.array(result_wave)
        return wave_frame, result_wave

    def _transform_slope_changes(self, wave_frame):
        first_derivative = np.diff(wave_frame)
        result_wave = np.sum(np.diff(np.sign(first_derivative)) != 0) / len(first_derivative)
        return wave_frame, result_wave

    def _transform_slope_changes_fixed_point(self, wave_frame):
        first_derivative = np.diff(wave_frame)
        total_slope = 0
        currSlope = np.sign(first_derivative[0])
        threshold = 1 
        for ele in first_derivative[1:]:
            if abs(ele) > threshold:
                if np.sign(ele) != currSlope:
                    currSlope = np.sign(ele)
                    total_slope += 1
        result_wave = total_slope * 1000    # Scaling values by 1000
        return wave_frame, result_wave

    def _transform_zero_crossing_rate_fixed_point(self, wave_frame):
        result_wave = np.sum(np.diff(np.sign(wave_frame)) != 0)
        result_wave = result_wave * 1000     # Scaling values by 1000   
        return wave_frame, result_wave

    def _transform_zero_crossing_rate(self, wave_frame):
        result_wave = np.sum(np.diff(np.sign(wave_frame)) != 0) / len(wave_frame)
        return wave_frame, result_wave

    def _transform_dominant_frequency_fixed_point(self, wave_frame):
        window = wave_frame * 128
        # Symmetrically mirror the window to 128 samples
        _, mirrored_window = self._transform_symmetric_mirror(window)
        num_bins = self.fft_size// 2 + 1 
        # Perform FFT and take absolute values               
        inst = dsp.arm_rfft_instance_q15()
        dsp.arm_rfft_init_q15(inst, self.fft_size,0,1)         # forward FFT
        fft_res = dsp.arm_rfft_q15(inst, mirrored_window)
        mag_res = dsp.arm_cmplx_mag_q15(fft_res)[:num_bins]
        mag_res = mag_res * 128
        mag_res[0] = 0        # DC  Remove
        freqIndex1 = 0
        freqIndex2 = 0
        for i in range (1,num_bins):
            if (mag_res[freqIndex1] < mag_res[i]):
                freqIndex2 = freqIndex1
                freqIndex1 = i
            elif (mag_res[freqIndex2] < mag_res[i]):
                freqIndex2 = i
        maxFreq1 = (((freqIndex1) * float(self.sampling_rate) // (2*num_bins))) * 1000       # Scaling values by 1000
        maxFreq2 = (((freqIndex2) * float(self.sampling_rate) // (2*num_bins))) * 1000       # Scaling values by 1000
        result_wave = [maxFreq1,maxFreq2]
        return wave_frame, result_wave 

    def _transform_dominant_frequency(self, wave_frame):
        num_bins = self.fft_size // 2 + 1
        fft_result = np.abs(np.fft.fft(wave_frame, n=self.fft_size))[:num_bins]
        freq_bins = np.linspace(0, float(self.sampling_rate) / 2.0, int(num_bins))
        sorted_indices = np.argsort(fft_result[1:])[-2:] + 1
        result_wave = freq_bins[sorted_indices]
        return wave_frame, result_wave

    def _transform_spectral_entropy(self, wave_frame):
        num_bins = self.fft_size // 2 + 1
        fft_result = np.abs(np.fft.fft(wave_frame, n=self.fft_size))[:num_bins]
        power_spectrum = fft_result ** 2
        psd_norm = power_spectrum / np.sum(power_spectrum)
        result_wave = entropy(psd_norm)
        return wave_frame, result_wave

    def _transform_symmetric_mirror(self, wave_frame):
        if len(wave_frame) == 0:
            result_wave = np.zeros(self.fft_size, dtype=np.float32)
        else:
            result_wave = np.tile(wave_frame, self.fft_size // len(wave_frame) + 1)
            result_wave = result_wave[:self.fft_size]
        return wave_frame, result_wave

    def _transform_pir_feature_extract(self, wave_frame):
        padded_frame = np.pad(wave_frame, (4, 3), mode='constant', constant_values=0)
        fft_features = []
        zcr_features = []
        slope_features = []
        dom_freq_features = []
        spectral_entropy_features = []
        stride_window = int(self.frame_size * self.stride_size)
        window_size = len(padded_frame) - (self.window_count * stride_window)
        for i in range(self.window_count + 1):
            start = i * stride_window
            end = start + window_size
            if end > len(padded_frame):
                break
            window = padded_frame[start:end]
            _, mirrored_window = self._transform_symmetric_mirror(window)
            num_bins = self.fft_size // 2 + 1
            fft_result = np.abs(np.fft.fft(mirrored_window, n=self.fft_size))[:num_bins]
            fft_features.append(fft_result)
        fft_features = np.array(fft_features)
        fft_features = fft_features[:-1, 1:]
        bin_count_pool = 16
        pool_size = self.fft_size // (2 * bin_count_pool)
        fft_features = fft_features.reshape(self.window_count, bin_count_pool, pool_size).mean(axis=-1)
        _, kurt_features = self._transform_kurtosis(wave_frame)
        for i in range(self.window_count):
            start = i * stride_window
            end = start + window_size
            if end > len(padded_frame):
                break
            window = padded_frame[start:end]
            _, zcr = self._transform_zero_crossing_rate(window)
            zcr_features.append(zcr)
            _, slope = self._transform_slope_changes(window)
            slope_features.append(slope)
            _, dom_freqs = self._transform_dominant_frequency(window)
            dom_freq_features.append(dom_freqs)
            _, spectral_entropy_val = self._transform_spectral_entropy(window)
            spectral_entropy_features.append(spectral_entropy_val)
        zcr_features = np.array(zcr_features).reshape(-1, 1)
        slope_features = np.array(slope_features).reshape(-1, 1)
        dom_freq_features = np.array(dom_freq_features)
        spectral_entropy_features = np.array(spectral_entropy_features).reshape(-1, 1)
        result_wave = np.hstack([fft_features, kurt_features, zcr_features, slope_features, dom_freq_features, spectral_entropy_features])
        return wave_frame, result_wave

    def _transform_pir_feature_extract_fixed_point(self, wave_frame):
        wave_frame = wave_frame - 128    # Remove DC Offset to centre signal around zero.
        padded_frame = np.pad(wave_frame, (4, 3), mode='constant', constant_values=0)

        fft_features = []
        zcr_features = []
        slope_features = []
        dom_freq_features = []        
        stride_window = int(self.frame_size * self.stride_size)
        window_size = len(padded_frame) - (self.window_count*stride_window) 
        for i in range(self.window_count + 1):            
            start = i * stride_window
            end = start + window_size
            if end > len(padded_frame):  # Ensure window does not exceed length
                break
            window = padded_frame[start:end]  # Extract window
            window = window * 128
            # Symmetrically mirror the window to 128 samples
            _, mirrored_window = self._transform_symmetric_mirror(window)
            num_bins = self.fft_size// 2 + 1 
            # Perform FFT and take absolute values            
            inst = dsp.arm_rfft_instance_q15()
            dsp.arm_rfft_init_q15(inst, self.fft_size,0,1)         # forward FFT
            fft_res = dsp.arm_rfft_q15(inst, mirrored_window)
            mag_res = dsp.arm_cmplx_mag_q15(fft_res)[:num_bins]
            mag_res = mag_res * 128       # Scale the magnitude by 128
            fft_features.append(mag_res)  # Store FFT output            

        fft_features = np.array(fft_features)        
        fft_features = fft_features[:-1,1:]  # Remove the last one to maintain window_count
        bin_count_pool = 16
        pool_size      = self.fft_size//(2*bin_count_pool)
        fft_features = np.clip(
            np.round(fft_features.reshape(self.window_count, bin_count_pool, pool_size).mean(axis=-1)), 
            np.iinfo(np.int16).min, 
            np.iinfo(np.int16).max
            ).astype(np.int16)  # Average Pooling                
        for i in range(self.window_count):
            start = i * stride_window
            end = start + window_size
            if end > len(padded_frame):  # Ensure window does not exceed length
                break
            window = padded_frame[start:end]        
            _,zcr = self._transform_zero_crossing_rate_fixed_point(window)
            zcr_features.append(zcr)
            _,slope = self._transform_slope_changes_fixed_point(window)
            slope_features.append(slope)
            _,dom_freqs = self._transform_dominant_frequency_fixed_point(window)
            dom_freq_features.append(dom_freqs)
        zcr_features = np.array(zcr_features)
        slope_features = np.array(slope_features)
        dom_freq_features = np.array(dom_freq_features)
        zcr_features = zcr_features.reshape(-1, 1)
        slope_features = slope_features.reshape(-1, 1)        
        # Feature Extraction for PIR is still in development phase. Support for fixed point kurtosis and spectral entropy will be added later.
        kurt_features = np.zeros((25, 4), dtype=int)
        spectral_entropy_features = np.zeros((25, 1), dtype=int)
        result_wave = np.hstack([fft_features, kurt_features, zcr_features, slope_features, dom_freq_features, spectral_entropy_features])    
        return wave_frame, result_wave

    def _transform_shape(self, concatenated_features, raw_frames):
        """Rearrange the shape from (C,N,W) to (N,C,W,H)."""
        N = concatenated_features.shape[1]
        if concatenated_features.ndim == 3:
            x_temp = concatenated_features.transpose(1, 0, 2).reshape(N, self.ch, self.wl, self.hl)
        else:
            x_temp = concatenated_features.transpose(1, 0, 2, 3)  # PIR Detection gives concatenated_features as (N,C,W,H)
        x_temp_raw_out = raw_frames.transpose(1, 0, 2)
        return x_temp, x_temp_raw_out

    # ==================== File Loading ====================

    def _load_datafile(self, datafile):
        """
        Load data from a file. Override in subclasses for task-specific loading.
        Returns: (x_temp, label_or_y, x_temp_raw_out)
        """
        file_extension = ops(datafile)[-1]

        if file_extension == ".npy":
            x_temp = np.load(datafile)
            if len(x_temp.shape) > 1:
                if self._variables_format == 'integer':
                    x_temp = x_temp[:, 1:self.variables + 1]
                else:
                    # Resolve and validate indices
                    valid_indices = self._resolve_column_selection(
                        x_temp, datafile, has_header=False, is_npy=True
                    )
                    # Adjust for time column (add 1 to skip first column)
                    adjusted_indices = [idx + 1 for idx in valid_indices]
                    x_temp = x_temp[:, adjusted_indices]

        elif file_extension in [".pkl", ".csv", ".txt"]:
            if file_extension == ".pkl":
                x_temp = pd.read_pickle(datafile)
                if not isinstance(x_temp, pd.DataFrame):
                    raise TypeError("Data loaded from .pkl file is not a pandas DataFrame. Please check the file format.")
                if x_temp.shape[0] == 0:
                    raise ValueError("File has no rows.")
                x_temp = x_temp[[col for col in x_temp.columns if 'time' not in str(col).lower()]]
            else:
                x_temp = pd.read_csv(datafile, header=None, dtype=str)
                if x_temp.shape[0] == 0:
                    raise ValueError("File has no rows.")
                x_temp = x_temp[[col_index for col_index, value in x_temp.iloc[0].items() if 'time' not in str(value).lower()]]

            # Detect if header exists
            has_header = False
            header_row = None
            try:
                float(x_temp.iloc[0, 0])
                has_header = False
            except (ValueError, TypeError):
                has_header = True
                header_row = x_temp.iloc[0]  # Save header before removal
                x_temp = x_temp[1:]

            # Resolve column selection before extracting values
            if self._variables_format != 'integer':
                # For column names, we need the header
                if self._variables_format == 'list_of_names':
                    if not has_header:
                        raise ValueError(
                            f"Column names specified but no header in {datafile}. "
                            f"Requested: {self.selected_column_names}"
                        )
                    # Create temporary DataFrame with header for resolution
                    temp_df = pd.DataFrame(x_temp.values, columns=header_row.values)
                    valid_indices = self._resolve_column_selection(
                        temp_df, datafile, has_header=True, is_npy=False
                    )
                    self.selected_column_indices = valid_indices
                else:
                    # List of indices - validate against data shape
                    valid_indices = self._resolve_column_selection(
                        x_temp, datafile, has_header=has_header, is_npy=False
                    )
                    self.selected_column_indices = valid_indices

            # Extract data
            x_temp = x_temp.values.astype(float)
            if self._variables_format == 'integer':
                x_temp = x_temp[:, :self.variables]
            else:
                x_temp = x_temp[:, self.selected_column_indices]
        else:
            raise Exception("Supports only .npy, .pkl, .txt and .csv file formats for now")

        label = opb(opd(datafile))

        # Perform the data processing transformations
        if 'Downsample' in self.transforms:
            x_temp = self._transform_downsample(x_temp)
        if 'SimpleWindow' in self.transforms:
            x_temp = self._transform_simple_window(x_temp)
        if self.scale:
            x_temp = x_temp / float(self.scale)
        # Apply augmenters
        x_temp = apply_augmenters(x_temp, self.augment_pipeline)

        x_temp_raw_out = x_temp.copy()
        return x_temp, label, x_temp_raw_out

    def _store_feat_ext(self, datafile, x_temp, x_temp_raw_out, label):
        """Store the (x_temp, x_temp_raw_out, y_temp) after applying all the feat-ext-transform."""
        y_temp = np.array([label for i in range(x_temp.shape[0])])
        if hasattr(self, 'store_feat_ext_data'):
            if hasattr(self, 'feat_ext_store_dir'):
                x_raw_out_file_path = os.path.join(
                    self.feat_ext_store_dir,
                    os.path.splitext(os.path.basename(datafile))[0] + '__' + 'raw' + '_X.npy')
                np.save(x_raw_out_file_path, x_temp_raw_out)
                self.logger.debug(f"Stored raw data in {x_raw_out_file_path}")

                transforms_chosen = '_'.join(self.transforms)
                out_file_name = os.path.splitext(os.path.basename(datafile))[0] + '__' + transforms_chosen

                x_out_file_path = os.path.join(self.feat_ext_store_dir, out_file_name + '_X.npy')
                np.save(x_out_file_path, x_temp)
                self.logger.debug(f"Stored intermediate data in {x_out_file_path}")

                y_out_file_path = os.path.join(self.feat_ext_store_dir, out_file_name + '_Y.npy')
                np.save(y_out_file_path, y_temp)
                self.logger.debug(f"Stored intermediate targets in {y_out_file_path}")
            else:
                self.logger.warning("'store_feat_ext_data' chosen but 'feat_ext_store_dir' not provided. Skipping storage")

    def _rearrange_dims(self, datafile, x_temp, label, x_temp_raw_out, **kwargs):
        """Rearrange the dimensions of x_temp to generalize for different channels."""
        y_temp = np.array([label for i in range(x_temp.shape[0])])
        try:
            if x_temp.ndim == 2:
                self.logger.warning("Not enough dimensions present. Extract more features")
                raise Exception("Not enough dimensions present. Extract more features")
            if x_temp.ndim == 3:
                x_temp = x_temp.transpose(0, 2, 1)
                x_temp_raw_out = x_temp_raw_out.transpose(0, 2, 1)
                x_temp = np.expand_dims(x_temp, axis=3)
                x_temp_raw_out = np.expand_dims(x_temp_raw_out, axis=3)
            if len(self.X) == 0:
                self.X = x_temp
                self.X_raw = x_temp_raw_out
            else:
                self.X = np.concatenate((self.X, x_temp))
                self.X_raw = np.concatenate((self.X_raw, x_temp_raw_out))
            self.Y = np.concatenate((self.Y, y_temp))
        except ValueError as e:
            self.logger.warning('Skipping {} as Error encountered: {}'.format(datafile, e))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Override in subclasses for task-specific item retrieval."""
        return torch.from_numpy(self.X_raw[index]), torch.from_numpy(self.X[index]), self.Y[index]

    def _process_targets(self):
        """Process targets after all data is loaded. Override in subclasses."""
        self.label_map = {k: v for v, k in enumerate(self.classes)}
        self.inverse_label_map = {k: v for v, k in self.label_map.items()}
        self.Y = [self.label_map[i] for i in self.Y]

        if not len(self.Y):
            self.logger.error("No data could be loaded. Either file paths were erroneous or dimensions mismatched. Check prior logger messages/ data processing configurations")
            raise Exception("No data could be loaded. Either file paths were erroneous or dimensions mismatched. Check prior logger messages/ data processing configurations")

    # ==================== Feature Extraction Variables ====================

    def _prepare_feature_extraction_variables(self):
        """
        Calculate the feature extraction parameters, preprocessing flags and model information.
        The parameters are stored in self.feature_extraction_params and self.preprocessing_flags
        and then saved in user_input_config.h file to be used in application code.
        """
        # Dataset definition (1, self.ch, self.wl, self.hl)
        self.ch = self.variables
        self.hl = 1
        self.wl = self.frame_size

        # This defines the input of the feature extraction library (run_feature_extraction)
        self.feature_extraction_params['FE_VARIABLES'] = self.variables
        self.feature_extraction_params['FE_FRAME_SIZE'] = self.frame_size
        self.feature_extraction_params['FE_HL'] = self.hl
        self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.frame_size

        # This defines the input of the model (tvmgen_default_run)
        self.feature_extraction_params['FE_STACKING_CHANNELS'] = self.ch
        self.feature_extraction_params['FE_STACKING_FRAME_WIDTH'] = self.wl

        # This defines the output of the model (tvmgen_default_run)
        self.feature_extraction_params['FE_NN_OUT_SIZE'] = len(self.classes)

        if len(self.feat_ext_transform) > 0:
            # Recalculating wl and ch when its 1D stacking
            if self.stacking == '1D':
                self.wl *= self.variables
                self.ch = 1
                self.feature_extraction_params['FE_STACKING_CHANNELS'] = self.ch
                self.feature_extraction_params['FE_STACKING_FRAME_WIDTH'] = self.wl

            # Store other feature extraction parameters
            self.feature_extraction_params['FE_OFFSET'] = self.offset
            self.feature_extraction_params['FE_SCALE'] = self.scale

            # Store the preprocessing flags specifically related to AI Library
            if 'FFT_Q15' in self.transforms:
                self.preprocessing_flags.append('FE_RFFT')
            if 'Q15_SCALE' in self.transforms:
                self.feature_extraction_params['FE_COMPLEX_MAG_SCALE_FACTOR'] = self.q15_scale_factor
                self.preprocessing_flags.append('FE_COMPLEX_MAG_SCALE')
            if 'Q15_MAG' in self.transforms:
                self.preprocessing_flags.append('FE_MAG')
            if 'WINDOWING' in self.transforms:
                self.preprocessing_flags.append('FE_WIN')
            if 'RAW_FE' in self.transforms:
                self.preprocessing_flags.append('FE_RAW')
            if 'PIR_FE' in self.transforms:
                self.preprocessing_flags.append('FE_PIR')
            if 'KURT_FE' in self.transforms:
                self.preprocessing_flags.append('FE_KURT')
            if 'ENT_FE' in self.transforms:
                self.preprocessing_flags.append('FE_ENT')
            if 'ZCR_FE' in self.transforms:
                self.preprocessing_flags.append('FE_ZCR')
            if 'DOM_FE' in self.transforms:
                self.preprocessing_flags.append('FE_DOM')
            if 'SLOPE_FE' in self.transforms:
                self.preprocessing_flags.append('FE_SLOPE')
            if 'FFT_FE' in self.transforms:
                self.preprocessing_flags.append('FE_FFT')
                self.feature_extraction_params['FE_FFT_STAGES'] = int(np.log2(self.frame_size))
                self.feature_extraction_params['FE_MIN_FFT_BIN'] = self.min_bin
                self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.frame_size // 2 + (self.min_bin if self.min_bin else 1)
            if 'DC_REMOVE' in self.transforms:
                self.preprocessing_flags.append('FE_DC_REM')
                self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] - 1
            if 'NORMALIZE' in self.transforms:
                self.preprocessing_flags.append('FE_NORMALIZE')
            if 'BINNING' in self.transforms:
                self.preprocessing_flags.append('FE_BIN')
                if self.feature_size_per_frame is None:
                    raise ValueError("Error: 'feature_size_per_frame' must be specified when using BINNING transform")
                self.bin_size = self.frame_size // 2 // self.feature_size_per_frame // self.analysis_bandwidth
                self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.feature_size_per_frame
                self.feature_extraction_params['FE_FFT_BIN_SIZE'] = self.bin_size
                self.feature_extraction_params['FE_MIN_FFT_BIN'] = self.min_bin
                self.feature_extraction_params['FE_BIN_NORMALIZE'] = 1 if self.normalize_bin else 0
            if 'BIN_Q15' in self.transforms:
                self.preprocessing_flags.append('FE_BIN')
                if self.feature_size_per_frame is None:
                    raise ValueError("Error: 'feature_size_per_frame' must be specified when using BINNING transform")
                self.bin_size = self.frame_size // 2 // self.feature_size_per_frame // self.analysis_bandwidth
                self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.feature_size_per_frame
                self.feature_extraction_params['FE_BIN_SIZE'] = self.bin_size
                self.feature_extraction_params['FE_BIN_OFFSET'] = self.min_bin
                self.feature_extraction_params['FE_BIN_NORMALIZE'] = 1 if self.normalize_bin else 0
            if 'LOG_DB' in self.transforms:
                self.preprocessing_flags.append('FE_LOG')
                if not self.log_mul:
                    self.log_mul = 20
                    self.logger.warning(f"Defaulting log multiplier to: {self.log_mul}.")
                self.feature_extraction_params['FE_LOG_MUL'] = self.log_mul
                if not self.log_base:
                    self.log_base = 10
                    self.logger.warning(f"Defaulting log base to: {self.log_base}.")
                if self.log_base == 'e':
                    self.feature_extraction_params['FE_LOG_BASE'] = 2.71828183
                else:
                    self.feature_extraction_params['FE_LOG_BASE'] = self.log_base

                try:
                    self.log_threshold = eval(self.log_threshold)
                except Exception as e:
                    self.log_threshold = 1e-100
                    self.logger.warning(f"Defaulting log threshold to: {self.log_threshold}. Because of exception: {e}")
                self.feature_extraction_params['FE_LOG_TOL'] = self.log_threshold
            if 'CONCAT' in self.transforms:
                self.preprocessing_flags.append('FE_CONCAT')
                self.feature_extraction_params['FE_NUM_FRAME_CONCAT'] = self.num_frame_concat
                self.wl = self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] * self.num_frame_concat
                if self.stacking == '1D':
                    self.wl *= self.variables
                self.feature_extraction_params['FE_STACKING_FRAME_WIDTH'] = self.wl

    # ==================== Feature Extraction ====================

    def _feature_extraction(self, x_temp, datafile, apply_gain_variations=False):
        """Main feature extraction loop."""
        number_of_frames = x_temp.shape[1] // self.frame_size
        if number_of_frames < self.num_frame_concat:
            raise ValueError(f"number_of_frames formed with the file ({number_of_frames}) < num_frame_concat ({self.num_frame_concat}) provided by the user's config. Either increase rows in the file or choose lesser frames to concatenate.")

        concatenated_features = []
        concatenated_raw_frames = []

        # Iterate the number of variables in dataset
        for ax in range(self.variables):
            x_temp_per_ax = x_temp[ax]

            number_of_steps = 1
            raw_frame_per_ax = []
            features_of_frame_per_ax = []
            concatenated_raw_frames_per_ax = []
            concatenated_features_per_ax = []

            if self.offset:
                number_of_steps = x_temp_per_ax.shape[0] // self.offset
                last_n = (x_temp_per_ax.shape[0] - (number_of_steps - 1) * self.offset) // self.frame_size
                if last_n == 0:
                    number_of_steps -= 1

            for n in range(number_of_steps):
                concatenated_features_per_step = []
                concatenated_raw_frames_per_step = []
                number_of_frames = (x_temp_per_ax.shape[0]) // self.frame_size
                if self.offset:
                    number_of_frames = (x_temp_per_ax.shape[0] - (n * self.offset)) // self.frame_size

                # Apply transformations on the frames
                for index_frame in range(number_of_frames):
                    start_idx = index_frame * self.frame_size
                    if self.offset:
                        start_idx = index_frame * self.frame_size + n * self.offset
                    end_idx = start_idx + self.frame_size
                    wave_frame = x_temp_per_ax[start_idx: end_idx]
                    if apply_gain_variations:
                        wave_frame = self._transform_gain(wave_frame, opb(opd(datafile)))
                    raw_wave = wave_frame
                    raw_frame_per_ax.append(raw_wave)

                    # Apply feature extraction transforms
                    wave_frame = self._apply_feature_transforms(wave_frame)
                    features_of_frame_per_ax.append(wave_frame)

                for index_frame in range(self.num_frame_concat - 1, number_of_frames):
                    if (index_frame % self.frame_skip == 0):
                        if 'CONCAT' in self.transforms:
                            _, result_wave = self._transform_concatenation(features_of_frame_per_ax, index_frame)
                            concatenated_features_per_step.append(result_wave)
                            concatenated_raw_frames_per_step = raw_frame_per_ax[self.num_frame_concat - 1: number_of_frames]

                concatenated_features_per_ax = concatenated_features_per_step
                concatenated_raw_frames_per_ax = concatenated_raw_frames_per_step

                if self.num_frame_concat < 2:
                    concatenated_features_per_ax = features_of_frame_per_ax
                    concatenated_raw_frames_per_ax = raw_frame_per_ax

            concatenated_features.append(concatenated_features_per_ax)
            concatenated_raw_frames.append(concatenated_raw_frames_per_ax)

        concatenated_features = np.array(concatenated_features)
        concatenated_raw_frames = np.array(concatenated_raw_frames)

        if hasattr(self, 'dont_train_just_feat_ext') and str2bool(self.dont_train_just_feat_ext):
            x_raw_out_file_path = os.path.join(self.feat_ext_store_dir, os.path.splitext(os.path.basename(datafile))[0] + '_features.npy')
            np.save(x_raw_out_file_path, concatenated_features.transpose(1, 0, 2))

        x_temp, x_temp_raw_out = self._transform_shape(concatenated_features, concatenated_raw_frames)

        return x_temp, x_temp_raw_out

    def _apply_feature_transforms(self, wave_frame):
        """Apply all feature extraction transforms to a wave frame."""
        if 'WINDOWING' in self.transforms:
            _, wave_frame = self._transform_windowing(wave_frame)
        if 'RAW_FE' in self.transforms:
            _, wave_frame = self._transform_raw(wave_frame)
        if 'TO_Q15' in self.transforms:
            _, wave_frame = self._transform_to_q15(wave_frame)
        if 'FFT_Q15' in self.transforms:
            _, wave_frame = self._transform_fft_q15(wave_frame)
        if 'Q15_SCALE' in self.transforms:
            _, wave_frame = self._transform_q15_scale(wave_frame)
        if 'Q15_MAG' in self.transforms:
            _, wave_frame = self._transform_q15_cmplx_mag(wave_frame)
        if 'FFT_FE' in self.transforms:
            _, wave_frame = self._transform_fft(wave_frame)
        if 'NORMALIZE' in self.transforms:
            _, wave_frame = self._transform_normalize(wave_frame)
        if 'ROUND_OFF' in self.transforms:
            _, wave_frame = self._transform_roundoff(wave_frame)
        if 'FFT_POS_HALF' in self.transforms:
            _, wave_frame = self._transform_pos_half_fft(wave_frame)
        if 'ABS' in self.transforms:
            _, wave_frame = self._transform_absolute(wave_frame)
        if 'KURT_FE' in self.transforms:
            _, wave_frame = self._transform_kurtosis(wave_frame)
        if 'ENT_FE' in self.transforms:
            _, wave_frame = self._transform_spectral_entropy(wave_frame)
        if 'ZCR_FE' in self.transforms:
            _, wave_frame = self._transform_zero_crossing_rate(wave_frame)
        if 'DOM_FE' in self.transforms:
            _, wave_frame = self._transform_dominant_frequency(wave_frame)
        if 'SLOPE_FE' in self.transforms:
            _, wave_frame = self._transform_slope_changes(wave_frame)
        if 'PIR_FE' in self.transforms:
            _, wave_frame = self._transform_pir_feature_extract(wave_frame)
        if 'PIR_FE_Q15' in self.transforms:
            _, wave_frame = self._transform_pir_feature_extract_fixed_point(wave_frame)
        if 'DC_REMOVE' in self.transforms:
            wave_frame = wave_frame[1:]
        if 'BIN_Q15' in self.transforms:
            _, wave_frame = self._transform_binning_q15(wave_frame)
        if 'BINNING' in self.transforms:
            _, wave_frame = self._transform_binning(wave_frame)
        if 'LOG_DB' in self.transforms:
            _, wave_frame = self._transform_log(wave_frame)
        return wave_frame

    # ==================== Main Preparation ====================

    def prepare(self, **kwargs):
        """Main preparation method. Loads all data and applies transforms."""
        self._prepare_feature_extraction_variables()
        if hasattr(self, 'store_feat_ext_data'):
            if not str2bool_or_none(self.feat_ext_store_dir):
                self.feat_ext_store_dir = os.path.join(self.output_dir, 'feat_ext_data')
                self.logger.info(f"feat_ext_store_dir has been defaulted to: {self.feat_ext_store_dir}")
            create_dir(self.feat_ext_store_dir)

        for datafile in tqdm(self._walker):
            try:
                self._process_datafile(datafile)
            except ValueError as v:
                self.logger.warning(f"File will be skipped due to an error: {datafile} : {v}")
            except IndexError as i:
                self.logger.error(f"Unexpected dataset dimensions. Check input options or dataset content. \nFile: {datafile}. Error message: {i}")
                exit()
            except KeyboardInterrupt:
                exit()

        try:
            if not self.X.shape[0]:
                self.logger.error("Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
                raise Exception("Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
        except Exception:
            self.logger.error("Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
            raise Exception("Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")

        if self.X.ndim == 3:
            self.X = np.expand_dims(self.X, axis=-1)
        self._process_targets()
        return self

    def _process_datafile(self, datafile):
        """Process a single datafile. Override in subclasses for task-specific processing."""
        x_temp, label, x_temp_raw_out = self._load_datafile(datafile)

        if hasattr(self, 'dont_train_just_feat_ext') and self.dont_train_just_feat_ext:
            x_raw_out_file_path = os.path.join(self.feat_ext_store_dir, os.path.splitext(os.path.basename(datafile))[0] + '_raw.npy')
            np.save(x_raw_out_file_path, x_temp.flatten())

        if len(self.feat_ext_transform) > 0:
            if hasattr(self, 'gain_variations') and self.gain_variations:
                gain_variations = literal_eval(self.gain_variations)
                if label in gain_variations.keys():
                    x_temp1, x_temp_raw_out1 = self._feature_extraction(x_temp.T, datafile, apply_gain_variations=True)
                    x_temp2, x_temp_raw_out2 = self._feature_extraction(x_temp.T, datafile)
                    x_temp = np.concatenate((x_temp1, x_temp2))
                    x_temp_raw_out = np.concatenate((x_temp_raw_out1, x_temp_raw_out2))
                else:
                    x_temp, x_temp_raw_out = self._feature_extraction(x_temp.T, datafile)
            else:
                x_temp, x_temp_raw_out = self._feature_extraction(x_temp.T, datafile)

            if hasattr(self, 'store_feat_ext_data') and self.store_feat_ext_data:
                self._store_feat_ext(datafile, x_temp, x_temp_raw_out, label)

        file_names = [datafile for i in range(x_temp.shape[0])]
        self.file_names.extend(file_names)
        self._rearrange_dims(datafile, x_temp, label, x_temp_raw_out)


class GenericTSDataset(BaseGenericTSDataset):
    """
    Dataset class for timeseries classification tasks.
    Inherits all functionality from BaseGenericTSDataset.
    """
    _logger_name = "root.GenericTSDataset"


class GenericTSDatasetReg(BaseGenericTSDataset):
    """
    Dataset class for timeseries regression tasks.
    Target values are continuous (from the last column of data).
    """
    _logger_name = "root.GenericTSDatasetReg"

    def _load_datafile(self, datafile):
        """Load data file with continuous target values from the last column."""
        file_extension = ops(datafile)[-1]

        if file_extension == ".npy":
            data = np.load(datafile)
            if len(data.shape) > 1:
                if self._variables_format == 'integer':
                    x_temp = data[:, 1:self.variables + 1]
                else:
                    # Resolve and validate indices
                    valid_indices = self._resolve_column_selection(
                        data, datafile, has_header=False, is_npy=True
                    )
                    # Adjust for time column (add 1 to skip first column)
                    adjusted_indices = [idx + 1 for idx in valid_indices]
                    x_temp = data[:, adjusted_indices]
                y_temp = data[:, -1].reshape(-1, 1)
        elif file_extension in [".pkl", ".csv", ".txt"]:
            if file_extension == ".pkl":
                data = pd.read_pickle(datafile)
                if not isinstance(data, pd.DataFrame):
                    raise TypeError("Data loaded from .pkl file is not a pandas DataFrame. Please check the file format.")
                if data.shape[0] == 0:
                    raise ValueError("File has no rows.")
                data = data[[col for col in data.columns if 'time' not in str(col).lower()]]
            else:
                data = pd.read_csv(datafile, header=None, dtype=str)
                if data.shape[0] == 0:
                    raise ValueError("File has no rows.")
                data = data[[col_index for col_index, value in data.iloc[0].items() if 'time' not in str(value).lower()]]

            # Detect if header exists
            has_header = False
            header_row = None
            try:
                float(data.iloc[0, 0])
                has_header = False
            except (ValueError, TypeError):
                has_header = True
                header_row = data.iloc[0]  # Save header before removal
                data = data[1:]

            # Resolve column selection before extracting values
            if self._variables_format != 'integer':
                # For column names, we need the header
                if self._variables_format == 'list_of_names':
                    if not has_header:
                        raise ValueError(
                            f"Column names specified but no header in {datafile}. "
                            f"Requested: {self.selected_column_names}"
                        )
                    # Create temporary DataFrame with header for resolution
                    temp_df = pd.DataFrame(data.values, columns=header_row.values)
                    valid_indices = self._resolve_column_selection(
                        temp_df, datafile, has_header=True, is_npy=False
                    )
                    self.selected_column_indices = valid_indices
                else:
                    # List of indices - validate against data shape
                    valid_indices = self._resolve_column_selection(
                        data, datafile, has_header=has_header, is_npy=False
                    )
                    self.selected_column_indices = valid_indices

            # Extract data
            data = data.values.astype(float)
            if self._variables_format == 'integer':
                x_temp = data[:, :self.variables]
            else:
                x_temp = data[:, self.selected_column_indices]
            y_temp = data[:, -1].reshape(-1, 1)
        else:
            raise Exception("Supports only .npy, .pkl, .txt and .csv file formats for now")

        # Perform the data processing transformations
        if 'Downsample' in self.transforms:
            x_temp = self._transform_downsample(x_temp)
            y_temp = basic_transforms.Downsample(y_temp, self.sampling_rate, self.new_sr)
        if 'SimpleWindow' in self.transforms:
            x_temp = self._transform_simple_window(x_temp)
            stride_window = int(self.frame_size * self.stride_size)
            if stride_window == 0:
                stride_window = self.frame_size
            if not hasattr(self, 'keep_short_tails'):
                self.keep_short_tails = False
            y_temp = np.array(basic_transforms.SimpleWindow(y_temp, self.frame_size, stride_window, self.keep_short_tails))
        if self.scale:
            x_temp = x_temp / float(self.scale)

        x_temp = apply_augmenters(x_temp, self.augment_pipeline)
        x_temp_raw_out = x_temp.copy()

        if y_temp.ndim == 2:
            y_temp = np.array(basic_transforms.SimpleWindow(y_temp, self.frame_size, 1, keep_short_tails=False))

        return x_temp, y_temp.mean(axis=1), x_temp_raw_out

    def _rearrange_dims(self, datafile, x_temp, y_temp, x_temp_raw_out, **kwargs):
        """Rearrange dimensions for regression (y_temp is continuous values, not labels)."""
        try:
            if x_temp.ndim == 2:
                self.logger.warning("Not enough dimensions present. Extract more features")
                raise Exception("Not enough dimensions present. Extract more features")
            if x_temp.ndim == 3:
                x_temp = x_temp.transpose(0, 2, 1)
                x_temp_raw_out = x_temp_raw_out.transpose(0, 2, 1)
                x_temp = np.expand_dims(x_temp, axis=3)
                x_temp_raw_out = np.expand_dims(x_temp_raw_out, axis=3)
            if len(self.X) == 0:
                self.X = x_temp
                self.X_raw = x_temp_raw_out
            else:
                self.X = np.concatenate((self.X, x_temp))
                self.X_raw = np.concatenate((self.X_raw, x_temp_raw_out))
            self.Y = np.concatenate((self.Y, y_temp)) if len(self.Y) else y_temp
        except ValueError as e:
            self.logger.warning('Skipping {} as Error encountered: {}'.format(datafile, e))

    def __getitem__(self, index):
        return torch.from_numpy(self.X_raw[index]), torch.from_numpy(self.X[index]), torch.from_numpy(self.Y[index])

    def _process_targets(self):
        """Process targets for regression (reshape Y)."""
        self.Y = self.Y.reshape(-1, 1)

    def _process_datafile(self, datafile):
        """Process a single datafile for regression."""
        x_temp, y_temp, x_temp_raw_out = self._load_datafile(datafile)

        if hasattr(self, 'dont_train_just_feat_ext') and self.dont_train_just_feat_ext:
            x_raw_out_file_path = os.path.join(self.feat_ext_store_dir, os.path.splitext(os.path.basename(datafile))[0] + '_raw.npy')
            np.save(x_raw_out_file_path, x_temp.flatten())

        if len(self.feat_ext_transform) > 0:
            x_temp, x_temp_raw_out = self._feature_extraction(x_temp.T, datafile)
            if hasattr(self, 'store_feat_ext_data') and self.store_feat_ext_data:
                self._store_feat_ext(datafile, x_temp, x_temp_raw_out, y_temp)

        self._rearrange_dims(datafile, x_temp, y_temp, x_temp_raw_out)

    def _transform_shape(self, concatenated_features, raw_frames):
        """Rearrange the shape from (C,N,W) to (N,C,W,H) for regression."""
        x_temp = concatenated_features.transpose(1, 0, 2)
        x_temp = np.expand_dims(x_temp, -1)
        x_temp_raw_out = raw_frames.transpose(1, 0, 2)
        return x_temp, x_temp_raw_out


class GenericTSDatasetAD(BaseGenericTSDataset):
    """
    Dataset class for timeseries anomaly detection tasks.
    Uses simplified feature extraction compared to classification.
    """
    _logger_name = "root.GenericTSDatasetAD"

    def _get_classes(self):
        """Get classes from directory listing for anomaly detection."""
        return sorted(set([path for path in os.listdir(self._path)]))

    def __getitem__(self, index):
        return torch.from_numpy(self.X_raw[index]), torch.from_numpy(self.X[index]), torch.from_numpy(np.array([self.Y[index]]))

    def _prepare_feature_extraction_variables(self):
        """Prepare feature extraction variables for anomaly detection (simplified)."""
        self.ch = self.variables
        self.hl = 1
        self.wl = self.frame_size

        self.feature_extraction_params['FE_VARIABLES'] = self.variables
        self.feature_extraction_params['FE_FRAME_SIZE'] = self.frame_size
        self.feature_extraction_params['FE_HL'] = self.hl
        self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.frame_size

        # Store the preprocessing flags used for AI Library
        if 'WINDOWING' in self.transforms:
            self.preprocessing_flags.append('FE_WIN')
        if 'RAW_FE' in self.transforms:
            self.preprocessing_flags.append('FE_RAW')
        if 'FFT_FE' in self.transforms:
            self.preprocessing_flags.append('FE_FFT')
        if 'NORMALIZE' in self.transforms:
            self.preprocessing_flags.append('FE_NORMALIZE')
        if 'FFT_POS_HALF' in self.transforms:
            self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] // 2 + (self.min_bin if self.min_bin else 1)
        if 'DC_REMOVE' in self.transforms:
            self.preprocessing_flags.append('FE_DC_REM')
            self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] - 1
        if 'BINNING' in self.transforms:
            self.preprocessing_flags.append('FE_BIN')
            if self.feature_size_per_frame is None:
                raise ValueError("Error: 'feature_size_per_frame' must be specified when using BINNING transform")
            self.feature_extraction_params['FE_BIN_NORMALIZE'] = 1 if self.normalize_bin else 0
            self.bin_size = self.frame_size // 2 // self.feature_size_per_frame // self.analysis_bandwidth
            self.feature_extraction_params['FE_FFT_BIN_SIZE'] = self.bin_size
            self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.feature_size_per_frame
        if 'LOG_DB' in self.transforms:
            self.preprocessing_flags.append('FE_LOG')
            if not self.log_mul:
                self.log_mul = 20
                self.logger.warning(f"Defaulting log multiplier to: {self.log_mul}.")
            self.feature_extraction_params['FE_LOG_MUL'] = self.log_mul
            if not self.log_base:
                self.log_base = 10
                self.logger.warning(f"Defaulting log base to: {self.log_base}.")
            if self.log_base == 'e':
                self.feature_extraction_params['FE_LOG_BASE'] = 2.71828183
            else:
                self.feature_extraction_params['FE_LOG_BASE'] = self.log_base
            try:
                self.log_threshold = eval(self.log_threshold)
            except Exception as e:
                self.log_threshold = 1e-100
                self.logger.warning(f"Defaulting log threshold to: {self.log_threshold}. Because of exception: {e}")
            self.feature_extraction_params['FE_LOG_TOL'] = self.log_threshold
        if 'CONCAT' in self.transforms:
            self.preprocessing_flags.append('FE_CONCAT')

        self.wl = self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] * self.num_frame_concat

        if self.stacking == '1D':
            self.wl *= self.variables
            self.ch = 1

        self.feature_extraction_params['FE_STACKING_CHANNELS'] = self.ch
        self.feature_extraction_params['FE_STACKING_FRAME_WIDTH'] = self.wl
        self.feature_extraction_params['FE_OFFSET'] = self.offset
        self.feature_extraction_params['FE_SCALE'] = self.scale
        self.feature_extraction_params['FE_MIN_FFT_BIN'] = self.min_bin if self.min_bin is not None else 1
        self.feature_extraction_params['FE_NUM_FRAME_CONCAT'] = self.num_frame_concat if self.num_frame_concat else 1
        self.feature_extraction_params['FE_NN_OUT_SIZE'] = self.wl * self.ch
        self.feature_extraction_params['FE_FEATURE_SIZE'] = self.wl
        self.feature_extraction_params['FE_FRAME_SKIP'] = self.frame_skip
        self.feature_extraction_params['FE_FFT_STAGES'] = int(np.log2(self.frame_size))

    def _feature_extraction(self, x_temp, datafile, apply_gain_variations=False):
        """Simplified feature extraction for anomaly detection."""
        number_of_frames = x_temp.shape[1] // self.frame_size

        concatenated_features = []
        concatenated_raw_frames = []

        for ax in range(self.variables):
            x_temp_per_ax = x_temp[ax]

            number_of_steps = 1
            raw_frame_per_ax = []
            features_of_frame_per_ax = []
            concatenated_raw_frames_per_ax = []
            concatenated_features_per_ax = []

            if self.offset:
                number_of_steps = x_temp_per_ax.shape[0] // self.offset
                last_n = (x_temp_per_ax.shape[0] - (number_of_steps - 1) * self.offset) // self.frame_size
                if last_n == 0:
                    number_of_steps -= 1

            for n in range(number_of_steps):
                concatenated_features_per_step = []
                concatenated_raw_frames_per_step = []
                number_of_frames = (x_temp_per_ax.shape[0]) // self.frame_size
                if self.offset:
                    number_of_frames = (x_temp_per_ax.shape[0] - (n * self.offset)) // self.frame_size

                for index_frame in range(number_of_frames):
                    start_idx = index_frame * self.frame_size
                    if self.offset:
                        start_idx = index_frame * self.frame_size + n * self.offset
                    end_idx = start_idx + self.frame_size
                    wave_frame = x_temp_per_ax[start_idx: end_idx]
                    raw_wave = wave_frame
                    raw_frame_per_ax.append(raw_wave)

                    # Simplified transforms for anomaly detection
                    if 'WINDOWING' in self.transforms:
                        _, wave_frame = self._transform_windowing(wave_frame)
                    if 'RAW_FE' in self.transforms:
                        _, wave_frame = self._transform_raw(wave_frame)
                    if 'FFT_FE' in self.transforms:
                        _, wave_frame = self._transform_fft(wave_frame)
                    if 'NORMALIZE' in self.transforms:
                        _, wave_frame = self._transform_normalize(wave_frame)
                    if 'FFT_POS_HALF' in self.transforms:
                        _, wave_frame = self._transform_pos_half_fft(wave_frame)
                    if 'ABS' in self.transforms:
                        _, wave_frame = self._transform_absolute(wave_frame)
                    if 'DC_REMOVE' in self.transforms:
                        wave_frame = wave_frame[1:]
                    if 'BINNING' in self.transforms:
                        _, wave_frame = self._transform_binning(wave_frame)
                    if 'LOG_DB' in self.transforms:
                        _, wave_frame = self._transform_log(wave_frame)

                    features_of_frame_per_ax.append(wave_frame)

                if number_of_frames < self.num_frame_concat:
                    self.logger.warning(f"Only {number_of_frames} frames available in {datafile}, but {self.num_frame_concat} required for concatenation")

                for index_frame in range(self.num_frame_concat - 1, number_of_frames):
                    if (index_frame % self.frame_skip == 0):
                        if 'CONCAT' in self.transforms:
                            _, result_wave = self._transform_concatenation(features_of_frame_per_ax, index_frame)
                            concatenated_features_per_step.append(result_wave)
                            concatenated_raw_frames_per_step = raw_frame_per_ax[self.num_frame_concat - 1: number_of_frames]

                concatenated_features_per_ax = concatenated_features_per_step
                concatenated_raw_frames_per_ax = concatenated_raw_frames_per_step

                if self.num_frame_concat < 2:
                    concatenated_features_per_ax = features_of_frame_per_ax
                    concatenated_raw_frames_per_ax = raw_frame_per_ax

            concatenated_features.append(concatenated_features_per_ax)
            concatenated_raw_frames.append(concatenated_raw_frames_per_ax)

        concatenated_features = np.array(concatenated_features)
        concatenated_raw_frames = np.array(concatenated_raw_frames)

        if hasattr(self, 'dont_train_just_feat_ext') and str2bool(self.dont_train_just_feat_ext):
            x_raw_out_file_path = os.path.join(self.feat_ext_store_dir, os.path.splitext(os.path.basename(datafile))[0] + '_features.npy')
            np.save(x_raw_out_file_path, concatenated_features.transpose(1, 0, 2))

        x_temp, x_temp_raw_out = self._transform_shape(concatenated_features, concatenated_raw_frames)

        return x_temp, x_temp_raw_out

    def _transform_shape(self, concatenated_features, raw_frames):
        """Shape transform for anomaly detection."""
        N = concatenated_features.shape[1]
        x_temp = concatenated_features.transpose(1, 0, 2).reshape(N, self.ch, self.wl, self.hl)
        x_temp_raw_out = raw_frames.transpose(1, 0, 2)
        return x_temp, x_temp_raw_out


class GenericTSDatasetForecasting(BaseGenericTSDataset):
    """
    Dataset class for timeseries forecasting tasks.
    Predicts future values based on historical data.
    """
    _logger_name = "root.GenericTSDatasetForecasting"

    def __init__(self, subset: str = None, dataset_dir: str = None, **kwargs):
        # Don't call parent __init__ directly as we need different initialization
        self.logger = getLogger(self._logger_name)
        self._path = dataset_dir
        self.label_map = dict()
        self.feature_extraction_params = dict()
        self.preprocessing_flags = []
        self.X = []
        self.Y = []

        # Store all the kwargs in self
        for key, value in kwargs.items():
            value = str2int_or_float(value)
            setattr(self, key, value)

        # Parse variables parameter to support multiple formats
        if hasattr(self, 'variables'):
            self._parse_variables_parameter()

        # Flatten transforms list
        self.transforms = np.array(self.transforms).flatten().tolist()

        # Load file list based on subset
        self._walker = self._load_file_list(subset, kwargs)

    def _transform_simple_window(self, x_temp, y_temp):
        """Form windows with forecast horizon for forecasting."""
        stride_window = int(self.frame_size * self.stride_size)
        if stride_window == 0:
            self.logger.warning("Stride Window size calculated is 0. Defaulting the value to Sample Window i.e. no overlap.")
            stride_window = self.frame_size

        if not hasattr(self, 'keep_short_tails'):
            self.keep_short_tails = False

        x_windows = []
        y_windows = []

        total_length = len(x_temp)
        max_index = total_length - (self.frame_size + self.forecast_horizon) + 1

        for i in range(0, max_index, stride_window):
            x_window = x_temp[i:i + self.frame_size]
            y_window = y_temp[i + self.frame_size:i + self.frame_size + self.forecast_horizon, [col_num for col_name in self.header_row for col_num in col_name.values()]]
            x_windows.append(x_window)
            y_windows.append(y_window)

        x_windows = np.array(x_windows)
        y_windows = np.array(y_windows)

        return x_windows, y_windows

    def _load_datafile(self, datafile):
        """Load data file for forecasting."""
        file_extension = ops(datafile)[-1]

        if file_extension == ".npy":
            data = np.load(datafile)
            if len(data.shape) > 1:
                if self._variables_format == 'integer':
                    x_temp = data[:, 1:self.variables + 1]
                else:
                    # Resolve and validate indices
                    valid_indices = self._resolve_column_selection(
                        data, datafile, has_header=False, is_npy=True
                    )
                    # Adjust for time column (add 1 to skip first column)
                    adjusted_indices = [idx + 1 for idx in valid_indices]
                    x_temp = data[:, adjusted_indices]
                y_temp = data[:, 1:]
        elif file_extension in [".pkl", ".csv", ".txt"]:
            if file_extension == ".pkl":
                data = pd.read_pickle(datafile)
                if not isinstance(data, pd.DataFrame):
                    raise TypeError("Data loaded from .pkl file is not a pandas DataFrame. Please check the file format.")
                if data.shape[0] == 0:
                    raise ValueError("File has no rows.")
                data = data[[col for col in data.columns if 'time' not in str(col).lower()]]
            else:
                data = pd.read_csv(datafile, header=None, dtype=str)
                if data.shape[0] == 0:
                    raise ValueError("File has no rows.")
                data = data[[col_index for col_index, value in data.iloc[0].items() if 'time' not in str(value).lower()]]

            # Detect if header exists and set up target_variables header_row
            has_header = False
            header_row_temp = None
            try:
                float(data.iloc[0, 0])
                has_header = False
                if self.target_variables == []:
                    self.header_row = [{'{}'.format(idx): idx} for idx in range(len(data.iloc[0].tolist()))]
                else:
                    self.header_row = [{str(idx): int(idx)} for idx in self.target_variables]
            except (ValueError, TypeError):
                has_header = True
                header_row_temp = data.iloc[0]  # Save header before removal
                is_int = lambda x: isinstance(x, int) or (isinstance(x, str) and x.isdigit())
                if self.target_variables == []:
                    self.header_row = [{value: idx} for idx, value in enumerate(data.iloc[0].tolist())]
                elif is_int(self.target_variables[0]):
                    self.header_row = [{data.iloc[0].tolist()[int(idx)]: int(idx)} for idx in self.target_variables]
                elif isinstance(self.target_variables[0], str):
                    self.header_row = [{value: data.iloc[0].tolist().index(value)} for value in self.target_variables]
                data = data[1:]

            # Resolve column selection before extracting values
            if self._variables_format != 'integer':
                # For column names, we need the header
                if self._variables_format == 'list_of_names':
                    if not has_header:
                        raise ValueError(
                            f"Column names specified but no header in {datafile}. "
                            f"Requested: {self.selected_column_names}"
                        )
                    # Create temporary DataFrame with header for resolution
                    temp_df = pd.DataFrame(data.values, columns=header_row_temp.values)
                    valid_indices = self._resolve_column_selection(
                        temp_df, datafile, has_header=True, is_npy=False
                    )
                    self.selected_column_indices = valid_indices
                else:
                    # List of indices - validate against data shape
                    valid_indices = self._resolve_column_selection(
                        data, datafile, has_header=has_header, is_npy=False
                    )
                    self.selected_column_indices = valid_indices

            # Extract data
            data = data.values.astype(float)

            #from sklearn.preprocessing import MinMaxScaler
            #scaler = MinMaxScaler()
            #data = scaler.fit_transform(data)

            if self._variables_format == 'integer':
                x_temp = data[:, :self.variables]
            else:
                x_temp = data[:, self.selected_column_indices]
            y_temp = data
        else:
            raise Exception("Supports only .npy, .pkl, .txt and .csv file formats for now")

        if 'Downsample' in self.transforms:
            x_temp = self._transform_downsample(x_temp)
            y_temp = basic_transforms.Downsample(y_temp, self.sampling_rate, self.new_sr)
        if 'SimpleWindow' in self.transforms and len(self.feat_ext_transform) == 0:
            x_temp, y_temp = self._transform_simple_window(x_temp, y_temp)
        if self.scale:
            x_temp = x_temp / float(self.scale)

        x_temp_raw_out = x_temp.copy()
        return x_temp, y_temp, x_temp_raw_out

    def _rearrange_dims(self, datafile, x_temp, y_temp, x_temp_raw_out, **kwargs):
        """Rearrange dimensions for forecasting."""
        try:
            if x_temp.ndim == 2:
                self.logger.warning("Not enough dimensions present. Extract more features")
                raise Exception("Not enough dimensions present. Extract more features")
            if x_temp.ndim == 3:
                x_temp = x_temp.transpose(0, 2, 1)
                x_temp_raw_out = x_temp_raw_out.transpose(0, 2, 1)
                x_temp = np.expand_dims(x_temp, axis=3)
                x_temp_raw_out = np.expand_dims(x_temp_raw_out, axis=3)
            if len(self.X) == 0:
                self.X = x_temp
                self.X_raw = x_temp_raw_out
            else:
                self.X = np.concatenate((self.X, x_temp))
                self.X_raw = np.concatenate((self.X_raw, x_temp_raw_out))
            self.Y = np.concatenate((self.Y, y_temp)) if len(self.Y) else y_temp
        except ValueError as e:
            self.logger.warning('Skipping {} as Error encountered: {}'.format(datafile, e))

    def __getitem__(self, index):
        return torch.from_numpy(self.X_raw[index]), torch.from_numpy(self.X[index]), torch.from_numpy(self.Y[index])

    def _prepare_feature_extraction_variables(self):
        """Prepare feature extraction variables for forecasting."""
        import ast
        if isinstance(self.target_variables, (int, float)):
            self.target_variables = [self.target_variables]
        elif isinstance(self.target_variables, str):
            try:
                self.target_variables = ast.literal_eval(self.target_variables)
                if isinstance(self.target_variables, (int, float)):
                    self.target_variables = [self.target_variables]
            except (ValueError, SyntaxError):
                # Handle plain column names or comma-separated column names
                # e.g., "indoorTemperature" or "col1,col2"
                if ',' in self.target_variables:
                    self.target_variables = [v.strip() for v in self.target_variables.split(',')]
                else:
                    self.target_variables = [self.target_variables]

        self.ch = self.variables
        self.hl = 1
        self.wl = self.frame_size

        self.feature_extraction_params['FE_STACKING_CHANNELS'] = self.ch
        self.feature_extraction_params['FE_VARIABLES'] = self.variables
        self.feature_extraction_params['FE_STACKING_FRAME_WIDTH'] = self.wl
        self.feature_extraction_params['FE_HL'] = self.hl
        self.feature_extraction_params['FE_FRAME_SIZE'] = self.frame_size
        self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.frame_size
        self.feature_extraction_params['FE_NN_OUT_SIZE'] = self.forecast_horizon * len(self.target_variables)

    def _process_targets(self):
        """No target processing needed for forecasting."""
        pass

    def _process_datafile(self, datafile):
        """Process a single datafile for forecasting."""
        x_temp, y_temp, x_temp_raw_out = self._load_datafile(datafile)

        if hasattr(self, 'dont_train_just_feat_ext') and self.dont_train_just_feat_ext:
            x_raw_out_file_path = os.path.join(self.feat_ext_store_dir, os.path.splitext(os.path.basename(datafile))[0] + '_raw.npy')
            np.save(x_raw_out_file_path, x_temp.flatten())

        # Note: Feature extraction transforms are not yet enabled for forecasting
        self._rearrange_dims(datafile, x_temp, y_temp, x_temp_raw_out)

    def prepare(self, **kwargs):
        """Main preparation method for forecasting."""
        self._prepare_feature_extraction_variables()
        if hasattr(self, 'store_feat_ext_data'):
            if not str2bool_or_none(self.feat_ext_store_dir):
                self.feat_ext_store_dir = os.path.join(self.output_dir, 'feat_ext_data')
                self.logger.info(f"feat_ext_store_dir has been defaulted to: {self.feat_ext_store_dir}")
            create_dir(self.feat_ext_store_dir)

        for datafile in tqdm(self._walker):
            try:
                self._process_datafile(datafile)
            except ValueError as v:
                self.logger.warning(f"File will be skipped due to an error: {datafile} : {v}")
            except IndexError as i:
                self.logger.error(f"Unexpected dataset dimensions. Check input options or dataset content. \nFile: {datafile}. Error message: {i}")
                exit()
            except KeyboardInterrupt:
                exit()

        if not self.X.shape[0]:
            self.logger.error("Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
            raise Exception("Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")

        if self.X.ndim == 3:
            self.X = np.expand_dims(self.X, axis=-1)
        self._process_targets()
        return self