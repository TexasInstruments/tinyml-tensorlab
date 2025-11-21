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


class GenericTSDataset(Dataset):
    def __init__(self, subset: str = None, dataset_dir: str = None, **kwargs):

        self.logger = getLogger("root.GenericTSDataset") 
        self._path = dataset_dir
        self.classes = list()
        self.label_map = dict()
        self.feature_extraction_params = dict()
        self.preprocessing_flags = []
        self.X = []
        self.Y = []
        self.file_names=[] # Stores file name of each sample

        # store all the kwargs in self
        for key, value in kwargs.items():
            value = str2int_or_float(value)
            setattr(self, key, value)

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

        # The self.transforms will contain str only
        # This is just an efficient to flatten the multidimensional list
        # TODO: Change the below line
        self.transforms = np.array(self.transforms).flatten().tolist()
        
        # Stores the path of data_files to be used
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
        
        # Files storing the datafile names
        if subset in ["training", "train"]:
            kwargs_list_key = 'training_list'
            list_filename = '*train*_list.txt'
        elif subset in ["testing", "test"]:
            kwargs_list_key = 'testing_list'
            try:
                list_filename = '*test*_list.txt'
                self._walker = load_list(kwargs_list_key, list_filename)
            except IndexError:
                list_filename = '*file*_list.txt'
                self._walker = load_list(kwargs_list_key, list_filename)
        elif subset in ["validation", "val"]:
            kwargs_list_key = 'validation_list'
            list_filename = '*val*_list.txt'
            
        self._walker = load_list(kwargs_list_key, list_filename)
        self.classes = sorted(set([opb(opd(datafile)) for datafile in self._walker]))

    # Downsample the dataset by a factor (sampling_rate/new_sr)
    def __transform_downsample(self, x_temp):
        x_temp = basic_transforms.Downsample(x_temp, self.sampling_rate, self.new_sr)
        return x_temp
    
    # Form windows of size frame_size from the dataset
    def __transform_simple_window(self, x_temp):
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

    # Applies a gain factor to the array
    def __transform_gain(self, array, label):
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

    # Stores the data in x_temp from the datafile
    def _load_datafile(self, datafile):
        file_extension = ops(datafile)[-1]

        # For .npy files, we assume it is regular numpy array and not structured array with named columns.
        # Also we assume first column is timestamp, hence excluding first column.
        if file_extension == ".npy":
            x_temp = np.load(datafile)
            if len(x_temp.shape) > 1:
                x_temp = x_temp[:, 1:self.variables+1]

        elif file_extension in [".pkl", ".csv", ".txt"]:
            # Note: When saving a DataFrame to .pkl, if headers are provided, they are saved as metadata. If not pandas assigns default headers (0,1,2...).
            # In both these cases headers are not part of .values array.
            if file_extension == ".pkl":
                x_temp = pd.read_pickle(datafile)
                if not isinstance(x_temp,pd.DataFrame):
                    raise TypeError("Data loaded from .pkl file is not a pandas DataFrame. Please check the file format.")
                if x_temp.shape[0]==0:
                    raise ValueError("File has no rows.")
                x_temp = x_temp[[col for col in x_temp.columns if 'time' not in str(col).lower()]] 
            else:
                # For .csv files, first row is considered as value and not header, hence specifying header=None. 
                x_temp = pd.read_csv(datafile, header=None, dtype=str) # Read as string to avoid dtype issues.
                if x_temp.shape[0]==0:
                    raise ValueError("File has no rows.")
                x_temp=x_temp[[col_index for col_index,value in x_temp.iloc[0].items() if 'time' not in str(value).lower()]]
                try:
                    float(x_temp.iloc[0, 0])
                except (ValueError, TypeError):
                    x_temp = x_temp[1:]

            x_temp = x_temp.values.astype(float) # Converting values from str back to float.
            x_temp = x_temp[:, :self.variables]
        else:
            raise Exception("Supports only .npy, .pkl, .txt and .csv file formats for now")
        label = opb(opd(datafile))

        # Perform the data processing transformations
        if 'Downsample' in self.transforms:
            x_temp = self.__transform_downsample(x_temp)
        if 'SimpleWindow' in self.transforms:
            x_temp = self.__transform_simple_window(x_temp)
        if self.scale:
            x_temp = x_temp / float(self.scale)
        # Apply augmenters
        x_temp = apply_augmenters(x_temp, self.augment_pipeline)  # https://tsaug.readthedocs.io/en/stable/quickstart.html
                
        x_temp_raw_out = x_temp.copy()
        return x_temp, label, x_temp_raw_out
    
    # Stores the (x_temp, x_temp_raw_out, y_temp) after applying all the feat-ext-transform
    def _store_feat_ext(self, datafile, x_temp, x_temp_raw_out, label):
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
        return

    # Rearranges the dimensions of x_temp to generalize for different channels
    def _rearrange_dims(self, datafile, x_temp, label, x_temp_raw_out, **kwargs):
        y_temp = np.array([label for i in range(x_temp.shape[0])])
        try:
            if x_temp.ndim == 2:
                self.logger.warning("Not enough dimensions present. Extract more features")
                raise Exception("Not enough dimensions present. Extract more features")
            if x_temp.ndim == 3:
                x_temp = x_temp.transpose(0, 2, 1)
                x_temp = np.expand_dims(x_temp, axis=3)
            if len(self.X) == 0:
                self.X = x_temp
                self.X_raw = x_temp_raw_out
            else:
                self.X = np.concatenate((self.X, x_temp))
                self.X_raw = np.concatenate((self.X_raw, x_temp_raw_out))
            self.Y = np.concatenate((self.Y, y_temp))
        except ValueError as e:
            self.logger.warning('Skipping {} as Error encountered: {}'.format(datafile, e))
        return

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X_raw[index]), torch.from_numpy(self.X[index]), self.Y[index]

    def _process_targets(self):
        self.label_map = {k: v for v, k in enumerate(self.classes)}         # {'class_0': 0, 'class_1': 1}
        self.inverse_label_map = {k: v for v, k in self.label_map.items()}  # {0 :'class_0', 1: 'class_1'}
        self.Y = [self.label_map[i] for i in self.Y]

        if not len(self.Y):
            self.logger.error("No data could be loaded. Either file paths were erroneous or dimensions mismatched. Check prior logger messages/ data processing configurations")
            raise Exception("No data could be loaded. Either file paths were erroneous or dimensions mismatched. Check prior logger messages/ data processing configurations")
        return
    
    def __transform_raw(self, wave_frame):
        # self.logger.debug(f"Transform: RAW shape: {wave_frame.shape}")
        result_wave = wave_frame
        mean_of_wave = np.sum(result_wave) // len(result_wave)
        if self.dc_remove:
            result_wave = result_wave - mean_of_wave
        return wave_frame, result_wave
    
    def __transform_windowing(self, wave_frame):
        # self.logger.debug(f"Transform: Windowing shape: {wave_frame.shape}")
        result_wave = wave_frame * np.hanning(self.frame_size)
        return wave_frame, result_wave

    def __transform_fft(self, wave_frame):
        # self.logger.debug(f"Transform: FFT shape: {wave_frame.shape}")
        result_wave = np.fft.fft(wave_frame)
        return wave_frame, result_wave

    def __transform_binning(self, wave_frame):
        # self.logger.debug(f"Transform: Binning input shape: {wave_frame.shape}")
        result_wave = np.empty((self.feature_size_per_frame))
        bin_size = self.bin_size
        bin_offset = self.min_bin

        for idx in range(self.feature_size_per_frame):
            idx1 = bin_offset + (idx)*bin_size
            sum_of_bin = np.sum(wave_frame[idx1:idx1+bin_size])
            result_wave[idx] = sum_of_bin 
            result_wave[idx] /= bin_size if self.normalize_bin else 1
        # self.logger.debug(f"Transform: Binning output shape: {result_wave.shape}")
        return wave_frame, result_wave

    def __transform_concatenation(self, features_of_frame_per_ax, index_frame):
        # self.logger.debug(f"Transform: Concatenation input shape: {np.array(features_of_frame_per_ax).shape}")
        result_wave = np.array(features_of_frame_per_ax[index_frame - self.num_frame_concat + 1: index_frame + 1]).flatten()
        # self.logger.debug(f"Transform: Concatenation output shape: {result_wave.shape}")
        return features_of_frame_per_ax, result_wave
    
    def __transform_pos_half_fft(self, wave_frame):
        # self.logger.debug(f"Transform: Pos Half FFT input shape: {wave_frame.shape}")
        # takes the DC + min_bin samples of the fft
        idx = self.frame_size//2
        idx += self.min_bin if self.min_bin else 1
        result_wave = wave_frame[:idx]
        # self.logger.debug(f"Transform: Pos Half FFT output shape: {result_wave.shape}")
        return wave_frame, result_wave

    def __transform_absolute(self, wave_frame):
        # self.logger.debug(f"Transform: Absolute shape: {wave_frame.shape}")
        # Converts the complex value to real values
        result_wave = abs(wave_frame)
        return wave_frame, result_wave
    
    def __transform_haar(self, wave_frame):
        # self.logger.debug(f"Transform: HAAR shape: {wave_frame.shape}")
        result_wave = haar_forward(wave_frame)
        return wave_frame, result_wave
    
    def __transform_hadamard(self, wave_frame):
        # self.logger.debug(f"Transform: Hadamard output shape: {wave_frame.shape}")
        result_wave = hadamard_forward_vectorized(wave_frame)
        # self.logger.debug(f"Transform: Hadamard output shape: {result_wave.shape}")
        return wave_frame, result_wave

    def __transform_normalize(self, wave_frame):
        result_wave = wave_frame / self.frame_size
        return wave_frame, result_wave
    
    def __transform_log(self, wave_frame):
        if self.log_base == 'e':
            result_wave = self.log_mul * np.log(self.log_threshold + wave_frame)
        else:
            try:
                base = float(self.log_base)
                result_wave = self.log_mul * np.log(self.log_threshold + wave_frame) / np.log(base)
            except (ValueError, TypeError):
                # If log_base can't be converted to float, default to base 10
                self.logger.warning("log_base can't be converted to float, default to base 10")
                result_wave = self.log_mul * np.log10(self.log_threshold + wave_frame)
        return wave_frame, result_wave
    
    def __transform_to_q15(self, wave_frame):
        def q15sat(x):
            if x > 0x7FFF:
                return(np.int16(0x7FFF))
            elif x < -0x8000:
                return(np.int16(-0x8000))
            else:
                return(np.int16(x))
        q15satV=np.vectorize(q15sat)
        def toQ15(x):
            return(q15satV(np.round(np.array(x) * (1<<15))))
        result_wave = toQ15(wave_frame)
        return wave_frame,result_wave

    def __transform_binning_q15(self, wave_frame):
        result_wave = np.empty((self.feature_size_per_frame))
        bin_size = self.bin_size
        bin_offset = self.min_bin if self.min_bin else 0
        for idx in range(self.feature_size_per_frame): 
            idx1 = bin_offset + (idx)*bin_size 
            sum_of_bin = np.sum(wave_frame[idx1:idx1+bin_size])
            avg = sum_of_bin // (bin_size if self.normalize_bin else 1)
            result_wave[idx]=np.clip(avg, -32768, 32767).astype(np.int16)  
        return wave_frame,result_wave

    def __transform_fft_q15(self, wave_frame):
        inst = dsp.arm_rfft_instance_q15()
        dsp.arm_rfft_init_q15(inst, self.frame_size,0,1)         # forward FFT
        result_wave = dsp.arm_rfft_q15(inst, wave_frame)
        return wave_frame, result_wave
 
    def __transform_q15_scale(self,wave_frame):
         result_wave = dsp.arm_shift_q15(wave_frame, self.q15_scale_factor)
         return wave_frame, result_wave
    
    def __transform_q15_cmplx_mag(self,wave_frame):  
        # self.logger.debug(f"Transform: q15 mag: Shape output shape: {wave_frame.shape}")
        result_wave = dsp.arm_cmplx_mag_q15(wave_frame[:self.frame_size+2]) 
        # self.logger.debug(f"Transform: q15 mag: Shape output shape: {result_wave.shape}")
        return wave_frame, result_wave


    def __transform_kurtosis(self, wave_frame):
        wave_frame = np.array(wave_frame, dtype=np.float32)  # Convert row to NumPy array
        # Pad the row (4 zeros at the beginning, 3 zeros at the end)
        padded_frame = np.pad(wave_frame, (4, 3), mode='constant', constant_values=0)
        kurtosis_features = []  # List to store kurtosis features
        result_wave = []
        stride_window = int(self.frame_size * self.stride_size)
        window_length = len(padded_frame) -(self.window_count*stride_window)
        for i in range(self.window_count):            
            start = i * stride_window 
            end = start + window_length
            if end > len(padded_frame):  # Ensure window does not exceed length
                break
            window = padded_frame[start:end]  # Extract 32-sample window

            window_kurtosis = []  # List to store kurtosis for this window
            # Compute kurtosis for each 8-sample chunk within the 32-sample window
            chunk_count = window_length//self.chunk_size # DEFAULT: 4
            for j in range(chunk_count): 
                chunk_start = j * self.chunk_size
                chunk_end = chunk_start + self.chunk_size
                chunk = window[chunk_start:chunk_end]
                # Compute kurtosis of the chunk
                eps = 1e-12
                if np.nanstd(chunk)<eps:
                    chunk_kurtosis = 0.0
                else:
                    chunk_kurtosis = kurtosis(chunk, fisher=True, bias=False)  # Fisher=True for normal zero kurtosis
                window_kurtosis.append(chunk_kurtosis)
            result_wave.append(window_kurtosis)
        #result_wave = np.array(result_wave).flatten()
        result_wave = np.array(result_wave)
        return wave_frame, result_wave
    
    def __transform_slope_changes(self, wave_frame):
        first_derivative = np.diff(wave_frame)
        result_wave = np.sum(np.diff(np.sign(first_derivative)) != 0) / len(first_derivative)
        return wave_frame, result_wave

    def __transform_zero_crossing_rate(self, wave_frame):
        result_wave = np.sum(np.diff(np.sign(wave_frame)) != 0) / len(wave_frame)
        return wave_frame, result_wave

    def __transform_dominant_frequency(self, wave_frame):
        # Function to calculate top two dominant frequencies in a signal
        num_bins   = self.fft_size // 2 + 1  # Number of frequency bins
        fft_result = np.abs(np.fft.fft(wave_frame, n=self.fft_size))[:num_bins]  # Compute FFT magnitude
        freq_bins = np.linspace(0, float(self.sampling_rate)/2.0, int(num_bins))  # Compute frequency bins
        sorted_indices = np.argsort(fft_result[1:])[-2:] + 1  # Get indices of top 2 frequencies (excluding DC)
        result_wave = freq_bins[sorted_indices] # Return the corresponding frequency values
        return wave_frame, result_wave 
    
    def __transform_spectral_entropy(self, wave_frame):
        num_bins   = self.fft_size // 2 + 1  # Number of frequency bins
        fft_result = np.abs(np.fft.fft(wave_frame, n=self.fft_size))[:num_bins]  # Compute FFT magnitude
        power_spectrum = fft_result ** 2  # Compute power spectrum
        psd_norm = power_spectrum / np.sum(power_spectrum)  # Normalize
        result_wave = entropy(psd_norm)  # Compute entropy
        return wave_frame, result_wave 

    def __transform_symmetric_mirror(self, wave_frame):
        if len(wave_frame) ==0:
            result_wave = np.zeros(self.fft_size,dtype=np.float32)
        else:
            result_wave = np.tile(wave_frame, self.fft_size // len(wave_frame) + 1)
            result_wave = result_wave[:self.fft_size]
        return wave_frame, result_wave 


    def __transform_pir_feature_extract(self, wave_frame):
        padded_frame = np.pad(wave_frame, (4, 3), mode='constant', constant_values=0)
        fft_features = []
        zcr_features = []
        slope_features = []
        dom_freq_features = []
        spectral_entropy_features = []
        stride_window = int(self.frame_size * self.stride_size)
        window_size = len(padded_frame) - (self.window_count*stride_window) 
        for i in range(self.window_count + 1):            
            start = i * stride_window
            end = start + window_size
            if end > len(padded_frame):  # Ensure window does not exceed length
                break
            window = padded_frame[start:end]  # Extract window
            # Symmetrically mirror the window to 128 samples
            _, mirrored_window = self.__transform_symmetric_mirror(window)
            num_bins = self.fft_size// 2 + 1 
            # Perform FFT and take absolute values
            fft_result = np.abs(np.fft.fft(mirrored_window, n=self.fft_size))[:num_bins]
            fft_features.append(fft_result)  # Store FFT output
        fft_features = np.array(fft_features)
        fft_features = fft_features[:-1,1:]  # Remove the last one to maintain window_count
        bin_count_pool = 16
        pool_size      = self.fft_size//(2*bin_count_pool)
        fft_features = fft_features.reshape(self.window_count, bin_count_pool, pool_size).mean(axis=-1)  # Average Pooling
        _,kurt_features = self.__transform_kurtosis(wave_frame)
        for i in range(self.window_count):
            start = i * stride_window
            end = start + window_size
            if end > len(padded_frame):  # Ensure window does not exceed length
                break
            window = padded_frame[start:end]
            _,zcr = self.__transform_zero_crossing_rate(window)
            zcr_features.append(zcr)
            _,slope = self.__transform_slope_changes(window)
            slope_features.append(slope)
            _,dom_freqs = self.__transform_dominant_frequency(window)
            dom_freq_features.append(dom_freqs)
            _,spectral_entropy =self. __transform_spectral_entropy(window)
            spectral_entropy_features.append(spectral_entropy)
        zcr_features = np.array(zcr_features)
        slope_features = np.array(slope_features)
        dom_freq_features = np.array(dom_freq_features)
        spectral_entropy_features = np.array(spectral_entropy_features)
        zcr_features = zcr_features.reshape(-1, 1)
        slope_features = slope_features.reshape(-1, 1)
        spectral_entropy_features = spectral_entropy_features.reshape(-1, 1)
        result_wave = np.hstack([fft_features, kurt_features, zcr_features, slope_features, dom_freq_features, spectral_entropy_features])
        return wave_frame, result_wave


    # Rearrange the shape from (C,N,W) to (N,C,W,H)
    def __transform_shape(self, concatenated_features, raw_frames):
        # self.logger.debug(f"Transform: Shape input shape: {concatenated_features.shape}")
        # print(f"Transform: Shape input shape: {concatenated_features.shape}")
        N = concatenated_features.shape[1]
        if concatenated_features.ndim == 3:  # Most cases have (C,N,W) --> (N,C,W,H)
            x_temp = concatenated_features.transpose(1, 0, 2).reshape(N, self.ch, self.wl, self.hl)
        else:
            x_temp = concatenated_features.transpose(1, 0, 2, 3)  # PIR Detection gives concatenated_features as (N,C,W,H)
        x_temp_raw_out = raw_frames.transpose(1, 0, 2)
        # self.logger.debug(f"Transform: Shape output shape: {x_temp.shape}")
        # print(f"Transform: Shape output shape: {x_temp.shape}")
        return x_temp, x_temp_raw_out

    def __prepare_feature_extraction_variables(self):
        '''
        This function calculates the feature extraction parameters, preprocessing flags and model information.
        The parameters are stored in self.feature_extraction_params and self.preprocessing_flags
        and then saved in user_input_config.h file to be using in application code.
        '''

        # Dataset definition (1, self.ch, self.wl, self.hl)
        self.ch = self.variables
        self.hl = 1
        self.wl = self.frame_size
        # Below are the parameters required even when feature extraction transforms are not applied. 

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
        # self.feature_extraction_params['FE_COMPLEX_MAG_SCALE_FACTOR'] = self.q15_scale_factor
                # Whether to perform batch normalization in FE Library or in the model. based on (skip_normalize=true)
        self.preprocessing_flags.append('SKIP_NORMALIZE')
        # Below are the feature extraction transform specific parameters.
        if len(self.feat_ext_transform) > 0:
            # Recalculating wl and ch when its 1D stacking
            if self.stacking == '1D':
                self.wl *= self.variables
                self.ch = 1
                self.feature_extraction_params['FE_STACKING_CHANNELS'] = self.ch
                self.feature_extraction_params['FE_STACKING_FRAME_WIDTH'] = self.wl
            # Store other feature extraction parameters
            # Offset params
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
                # In feature extraction library this is equivalent to FFT + POS_HALF + ABS
                self.preprocessing_flags.append('FE_FFT')
                self.feature_extraction_params['FE_FFT_STAGES'] = int(np.log2(self.frame_size))
                self.feature_extraction_params['FE_MIN_FFT_BIN'] = self.min_bin
                self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.frame_size//2 + (self.min_bin if self.min_bin else 1)
            if 'DC_REMOVE' in self.transforms:
                self.preprocessing_flags.append('FE_DC_REM')
                self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] - 1
            if 'NORMALIZE' in self.transforms:
                # Normalize the FFT output by frame size
                self.preprocessing_flags.append('FE_NORMALIZE')
            if 'BINNING' in self.transforms:
                self.preprocessing_flags.append('FE_BIN')
                if self.feature_size_per_frame is None:
                    raise ValueError("Error: 'feature_size_per_frame' must be specified when using BINNING transform")
                self.bin_size = self.frame_size // 2 // self.feature_size_per_frame // self.analysis_bandwidth
                self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.feature_size_per_frame
                self.feature_extraction_params['FE_FFT_BIN_SIZE'] = self.bin_size
                self.feature_extraction_params['FE_MIN_FFT_BIN'] = self.min_bin
                # Normalize the binned output by bin size
                self.feature_extraction_params['FE_BIN_NORMALIZE'] = 1 if self.normalize_bin else 0
            if 'BIN_Q15' in self.transforms:
                self.preprocessing_flags.append('FE_BIN')
                if self.feature_size_per_frame is None:
                    raise ValueError("Error: 'feature_size_per_frame' must be specified when using BINNING transform")
                self.bin_size = self.frame_size // 2 // self.feature_size_per_frame // self.analysis_bandwidth
                self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.feature_size_per_frame
                self.feature_extraction_params['FE_BIN_SIZE'] = self.bin_size
                self.feature_extraction_params['FE_BIN_OFFSET']= self.min_bin
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
                self.feature_extraction_params['FE_LOG_TOL']=self.log_threshold
            if 'CONCAT' in self.transforms:
                self.preprocessing_flags.append('FE_CONCAT')
                self.feature_extraction_params['FE_NUM_FRAME_CONCAT'] = self.num_frame_concat
                self.wl = self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] * self.num_frame_concat
                if self.stacking == '1D':
                    self.wl *= self.variables
                self.feature_extraction_params['FE_STACKING_FRAME_WIDTH'] = self.wl
                # No need to pass frame_skip since modelmaker only uses frame_skip
                #self.feature_extraction_params['FE_FRAME_SKIP'] = self.frame_skip 
        return

    def __feature_extraction(self, x_temp, datafile, apply_gain_variations=False):

        number_of_frames = x_temp.shape[1] // self.frame_size
        if number_of_frames < self.num_frame_concat:
            raise ValueError(f"number_of_frames formed with the file ({number_of_frames}) < num_frame_concat ({self.num_frame_concat}) provided by the user's config. Either increase rows in the file or choose lesser frames to concatenate.")

        concatenated_features = []
        concatenated_raw_frames = []
        
        # Iterate the number of variables in dataset
        for ax in range(self.variables):
            x_temp_per_ax = x_temp[ax]
            
            number_of_steps = 1
            # Stores the features and raw frames for all steps of single variable
            raw_frame_per_ax = []
            features_of_frame_per_ax = []
            concatenated_raw_frames_per_ax = []
            concatenated_features_per_ax = []

            if self.offset:
                number_of_steps = x_temp_per_ax.shape[0] // self.offset
                last_n = (x_temp_per_ax.shape[0] - (number_of_steps-1)*self.offset) // self.frame_size      # elements left // self.frame_size
                if last_n==0:
                    number_of_steps -= 1    # don't include the leftover elements
            
            # If no offset: number_of_steps = 1, number_of_frames formed with frame_size
            # else: calculate the number_of_frames formed by adding 1/n offsets for each frame
            for n in range(number_of_steps):
                # Store the features and raw frames of single step of single variable
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
                        wave_frame = self.__transform_gain(wave_frame, opb(opd(datafile)))  # gain variation augmentation
                    raw_wave = wave_frame
                    # Contains the frame on which transforms will be applied
                    raw_frame_per_ax.append(raw_wave)

                    if 'WINDOWING' in self.transforms:
                        raw_wave, wave_frame = self.__transform_windowing(wave_frame)
                    if 'RAW_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_raw(wave_frame)
                    if 'TO_Q15' in self.transforms:
                        raw_wave, wave_frame = self.__transform_to_q15(wave_frame)
                    if 'FFT_Q15' in self.transforms:
                        raw_wave, wave_frame = self.__transform_fft_q15(wave_frame)
                    if 'Q15_SCALE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_q15_scale(wave_frame)
                    if 'Q15_MAG' in self.transforms:
                        raw_wave, wave_frame = self.__transform_q15_cmplx_mag(wave_frame)
                    if 'FFT_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_fft(wave_frame)
                    if 'NORMALIZE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_normalize(wave_frame)
                    if 'FFT_POS_HALF' in self.transforms:
                        raw_wave, wave_frame = self.__transform_pos_half_fft(wave_frame)
                    if 'ABS' in self.transforms:
                        raw_wave, wave_frame = self.__transform_absolute(wave_frame)
                    if 'KURT_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_kurtosis(wave_frame)
                    if 'ENT_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_spectral_entropy(wave_frame)
                    if 'ZCR_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_zero_crossing_rate(wave_frame)
                    if 'DOM_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_dominant_frequency(wave_frame)
                    if 'SLOPE_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_slope_changes(wave_frame)
                    if 'PIR_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_pir_feature_extract(wave_frame)
                    if 'DC_REMOVE' in self.transforms:
                        wave_frame = wave_frame[1:]
                    # if 'HAAR' in self.transforms:
                    #     raw_wave, wave_frame = self.__transform_haar(wave_frame)
                    # if 'HADAMARD' in self.transforms:
                    #     raw_wave, wave_frame = self.__transform_hadamard(wave_frame)
                    # if 'BIN' in self.transforms:
                    #     raw_wave, wave_frame = self.__transform_basic_binning(wave_frame)
                    if 'BIN_Q15' in self.transforms:
                        raw_wave, wave_frame = self.__transform_binning_q15(wave_frame)
                    if 'BINNING' in self.transforms:
                        raw_wave, wave_frame = self.__transform_binning(wave_frame)
                    if 'LOG_DB' in self.transforms:
                        raw_wave, wave_frame = self.__transform_log(wave_frame)
                    
                    # Contains the frame after all transformations are applied except concatenation
                    features_of_frame_per_ax.append(wave_frame)

                for index_frame in range(self.num_frame_concat-1, number_of_frames):
                    if (index_frame % self.frame_skip == 0):
                        if 'CONCAT' in self.transforms:
                            _, result_wave = self.__transform_concatenation(features_of_frame_per_ax, index_frame)
                            concatenated_features_per_step.append(result_wave)
                            concatenated_raw_frames_per_step = raw_frame_per_ax[self.num_frame_concat - 1: number_of_frames]
                
                # Store the features and raw frames with concatenation
                concatenated_features_per_ax = concatenated_features_per_step      
                concatenated_raw_frames_per_ax = concatenated_raw_frames_per_step  

                # Store the features and raw frames w/o concatenating
                if self.num_frame_concat < 2:
                    concatenated_features_per_ax = features_of_frame_per_ax
                    concatenated_raw_frames_per_ax = raw_frame_per_ax

            concatenated_features.append(concatenated_features_per_ax)
            concatenated_raw_frames.append(concatenated_raw_frames_per_ax)
        
        concatenated_features = np.array(concatenated_features)
        concatenated_raw_frames = np.array(concatenated_raw_frames)
        
        if hasattr(self, 'dont_train_just_feat_ext') and  str2bool(self.dont_train_just_feat_ext):
            x_raw_out_file_path = os.path.join(self.feat_ext_store_dir, os.path.splitext(os.path.basename(datafile))[0] + '_features.npy')
            np.save(x_raw_out_file_path, concatenated_features.transpose(1,0,2))

        x_temp, x_temp_raw_out = self.__transform_shape(concatenated_features, concatenated_raw_frames)

        return x_temp, x_temp_raw_out
    
    def prepare(self, **kwargs):
        self.__prepare_feature_extraction_variables()
        if hasattr(self, 'store_feat_ext_data'):
            if not str2bool_or_none(self.feat_ext_store_dir):
                self.feat_ext_store_dir = os.path.join(self.output_dir, 'feat_ext_data')
                self.logger.info(f"feat_ext_store_dir has been defaulted to: {self.feat_ext_store_dir}")
            create_dir(self.feat_ext_store_dir)
        for datafile in tqdm(self._walker):
            try:
                x_temp, label, x_temp_raw_out = self._load_datafile(datafile)   # Loads the dataset and applied data processing transforms
                if hasattr(self, 'dont_train_just_feat_ext') and self.dont_train_just_feat_ext:
                    x_raw_out_file_path = os.path.join(self.feat_ext_store_dir, os.path.splitext(os.path.basename(datafile))[0] + '_raw.npy')
                    np.save(x_raw_out_file_path, x_temp.flatten())
                if len(self.feat_ext_transform) > 0:
                    if self.gain_variations:
                        gain_variations = literal_eval(self.gain_variations)
                        if label in gain_variations.keys():
                            x_temp1, x_temp_raw_out1 = self.__feature_extraction(x_temp.T, datafile, apply_gain_variations=True)   # Applies feature extraction transforms
                            x_temp2, x_temp_raw_out2 = self.__feature_extraction(x_temp.T, datafile)  # Applies feature extraction transforms
                            x_temp = np.concatenate((x_temp1, x_temp2))
                            x_temp_raw_out = np.concatenate((x_temp_raw_out1, x_temp_raw_out2))
                        else:
                            x_temp, x_temp_raw_out = self.__feature_extraction(x_temp.T, datafile)  # Applies feature extraction transforms
                    else:
                        x_temp, x_temp_raw_out = self.__feature_extraction(x_temp.T, datafile)   # Applies feature extraction transforms
                    if hasattr(self, 'store_feat_ext_data') and self.store_feat_ext_data:
                        self._store_feat_ext(datafile, x_temp, x_temp_raw_out, label)
                file_names= [datafile for i in range(x_temp.shape[0])]
                self.file_names.extend(file_names)
                self._rearrange_dims(datafile, x_temp, label, x_temp_raw_out)
            except ValueError as v:
                self.logger.warning(f"File will be skipped due to an error: {datafile} : {v}")
            except IndexError as i:
                self.logger.error(f"Unexpected dataset dimensions. Check input options or dataset content. \nFile: {datafile}. Error message: {i}")
                exit()
            except KeyboardInterrupt as ki:
                exit()
        try:
            if not self.X.shape[0]:
                self.logger.error("Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
                raise Exception("Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
        except Exception as e:
            self.logger.error("Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
            raise Exception("Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
        if self.X.ndim == 3:
            self.X = np.expand_dims(self.X, axis=-1)
        self._process_targets()
        return self


class GenericTSDatasetReg(Dataset):
    def __init__(self, subset: str = None, dataset_dir: str = None, **kwargs):

        self.logger = getLogger("root.GenericTSDataset")
        self._path = dataset_dir
        self.classes = list()
        self.label_map = dict()
        self.feature_extraction_params = dict()
        self.preprocessing_flags = []
        self.X = []
        self.Y = []

        # store all the kwargs in self
        for key, value in kwargs.items():
            value = str2int_or_float(value)
            setattr(self, key, value)

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

        # The self.transforms will contain str only
        # This is just an efficient to flatten the multidimensional list
        # TODO: Change the below line
        self.transforms = np.array(self.transforms).flatten().tolist()

        # Stores the path of data_files to be used
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

        # Files storing the datafile names
        if subset in ["training", "train"]:
            kwargs_list_key = 'training_list'
            list_filename = '*train*_list.txt'
        elif subset in ["testing", "test"]:
            kwargs_list_key = 'testing_list'
            try:
                list_filename = '*test*_list.txt'
                self._walker = load_list(kwargs_list_key, list_filename)
            except IndexError:
                list_filename = '*file*_list.txt'
                self._walker = load_list(kwargs_list_key, list_filename)
        elif subset in ["validation", "val"]:
            kwargs_list_key = 'validation_list'
            list_filename = '*val*_list.txt'

        self._walker = load_list(kwargs_list_key, list_filename)
        self.classes = sorted(set([opb(opd(datafile)) for datafile in self._walker]))

    # Downsample the dataset by a factor (sampling_rate/new_sr)
    def __transform_downsample(self, x_temp, y_temp):
        x_temp = basic_transforms.Downsample(x_temp, self.sampling_rate, self.new_sr)
        y_temp = basic_transforms.Downsample(y_temp, self.sampling_rate, self.new_sr)
        return x_temp, y_temp

    # Form windows of size frame_size from the dataset
    def __transform_simple_window(self, x_temp, y_temp):
        stride_window = int(self.frame_size * self.stride_size)
        if stride_window == 0:
            self.logger.warning(
                "Stride Window size calculated is 0. Defaulting the value to Sample Window i.e. no overlap.")
            stride_window = self.frame_size
        if not hasattr(self, 'keep_short_tails'):
            self.keep_short_tails = False
        x_temp = np.array(basic_transforms.SimpleWindow(x_temp, self.frame_size, stride_window, self.keep_short_tails))
        y_temp = np.array(basic_transforms.SimpleWindow(y_temp, self.frame_size, stride_window, self.keep_short_tails))
        if len(self.feat_ext_transform) > 0:
            x_temp = x_temp.reshape(-1, self.variables)
        return x_temp, y_temp

    # Stores the data in x_temp from the datafile
    def _load_datafile(self, datafile):
        file_extension = ops(datafile)[-1]
        if file_extension == ".npy":
            data = np.load(datafile)
            if len(data.shape) > 1:
                x_temp = data[:, 1:self.variables+1]
                y_temp = data[:, -1].reshape(-1, 1)
        elif file_extension in [".pkl", ".csv", ".txt"]:
            # Note: When saving a DataFrame to .pkl, if headers are provided, they are saved as metadata. If not pandas assigns default headers (0,1,2...).
            # In both these cases headers are not part of .values array.
            if file_extension == ".pkl":
                data = pd.read_pickle(datafile)
                if not isinstance(data,pd.DataFrame):
                    raise TypeError("Data loaded from .pkl file is not a pandas DataFrame. Please check the file format.")
                if data.shape[0]==0:
                    raise ValueError("File has no rows.")
                data = data[[col for col in data.columns if 'time' not in str(col).lower()]]
            else:
                # For .csv files, first row is considered as value and not header, hence specifying header=None. 
                data = pd.read_csv(datafile, header=None, dtype=str) # Read as string to avoid dtype issues.
                if data.shape[0]==0:
                    raise ValueError("File has no rows.")
                data=data[[col_index for col_index,value in data.iloc[0].items() if 'time' not in str(value).lower()]]
                try:
                    float(data.iloc[0, 0])
                except (ValueError, TypeError):
                    data = data[1:]
            data = data.values.astype(float) 
            x_temp = data[:, :self.variables]
            y_temp = data[:, -1].reshape(-1, 1)
        else:
            raise Exception("Supports only .npy, .pkl, .txt and .csv file formats for now")

        # Perform the data processing transformations
        if 'Downsample' in self.transforms:
            x_temp, y_temp = self.__transform_downsample(x_temp, y_temp)
        if 'SimpleWindow' in self.transforms:
            x_temp, y_temp = self.__transform_simple_window(x_temp, y_temp)
        if self.scale:
            x_temp = x_temp / float(self.scale)
        # Apply augmenters
        x_temp = apply_augmenters(x_temp, self.augment_pipeline)  # https://tsaug.readthedocs.io/en/stable/quickstart.html

        x_temp_raw_out = x_temp.copy()
        if y_temp.ndim == 2:
            y_temp = np.array(basic_transforms.SimpleWindow(y_temp, self.frame_size, 1, keep_short_tails=False))
        # y_temp = y_temp.transpose(0, 2, 1) # (10, 1024, 2) -> (10, 2, 1024)
        # y_temp = y_temp.mean(axis=2)       # (10, 2, 1024) -> (10, 2)
        return x_temp, y_temp.mean(axis=1), x_temp_raw_out

    # Stores the (x_temp, x_temp_raw_out, y_temp) after applying all the feat-ext-transform
    def _store_feat_ext(self, datafile, x_temp, x_temp_raw_out, label):
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
                self.logger.warning(
                    "'store_feat_ext_data' chosen but 'feat_ext_store_dir' not provided. Skipping storage")
        return

    # Rearranges the dimensions of x_temp to generalize for different channels
    def _rearrange_dims(self, datafile, x_temp, y_temp, x_temp_raw_out, **kwargs):
        # y_temp = np.array([label for i in range(x_temp.shape[0])])
        try:
            if x_temp.ndim == 2:
                self.logger.warning("Not enough dimensions present. Extract more features")
                raise Exception("Not enough dimensions present. Extract more features")
            if x_temp.ndim == 3:
                x_temp = x_temp.transpose(0, 2, 1)
                x_temp = np.expand_dims(x_temp, axis=3)
            if len(self.X) == 0:
                self.X = x_temp
                self.X_raw = x_temp_raw_out
            else:
                self.X = np.concatenate((self.X, x_temp))
                self.X_raw = np.concatenate((self.X_raw, x_temp_raw_out))
            self.Y = np.concatenate((self.Y, y_temp)) if len(self.Y) else y_temp
        except ValueError as e:
            self.logger.warning('Skipping {} as Error encountered: {}'.format(datafile, e))
        return

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X_raw[index]), torch.from_numpy(self.X[index]), torch.from_numpy(self.Y[index])

    def _process_targets(self):
        self.Y = self.Y.reshape(-1, 1)
        # self.label_map = {k: v for v, k in enumerate(self.classes)}  # {'class_0': 0, 'class_1': 1}
        # self.inverse_label_map = {k: v for v, k in self.label_map.items()}  # {0 :'class_0', 1: 'class_1'}
        # self.Y = [self.label_map[i] for i in self.Y]
        #
        # if not len(self.Y):
        #     self.logger.error(
        #         "No data could be loaded. Either file paths were erroneous or dimensions mismatched. Check prior logger messages/ data processing configurations")
        #     raise Exception(
        #         "No data could be loaded. Either file paths were erroneous or dimensions mismatched. Check prior logger messages/ data processing configurations")
        return

    def __transform_raw(self, wave_frame):
        # self.logger.debug(f"Transform: RAW shape: {wave_frame.shape}")
        result_wave = wave_frame
        mean_of_wave = np.sum(result_wave) // len(result_wave)
        if self.dc_remove:
            result_wave = result_wave - mean_of_wave
        return wave_frame, result_wave

    def __transform_windowing(self, wave_frame):
        # self.logger.debug(f"Transform: Windowing shape: {wave_frame.shape}")
        result_wave = wave_frame * np.hanning(self.frame_size)
        return wave_frame, result_wave

    def __transform_fft(self, wave_frame):
        # self.logger.debug(f"Transform: FFT shape: {wave_frame.shape}")
        result_wave = np.fft.fft(wave_frame)
        return wave_frame, result_wave

    def __transform_binning(self, wave_frame):
        # self.logger.debug(f"Transform: Binning input shape: {wave_frame.shape}")
        result_wave = np.empty((self.feature_size_per_frame))
        bin_size = self.bin_size
        bin_offset = self.min_bin if self.min_bin else 0

        for idx in range(self.feature_size_per_frame):
            idx1 = bin_offset + (idx) * bin_size
            sum_of_bin = np.sum(wave_frame[idx1:idx1 + bin_size])
            result_wave[idx] = sum_of_bin
            result_wave[idx] /= bin_size if self.normalize_bin else 1
        # self.logger.debug(f"Transform: Binning output shape: {result_wave.shape}")
        return wave_frame, result_wave

    def __transform_concatenation(self, features_of_frame_per_ax, index_frame):
        # self.logger.debug(f"Transform: Concatenation input shape: {np.array(features_of_frame_per_ax).shape}")
        result_wave = np.array(
            features_of_frame_per_ax[index_frame - self.num_frame_concat + 1: index_frame + 1]).flatten()
        # self.logger.debug(f"Transform: Concatenation output shape: {result_wave.shape}")
        return features_of_frame_per_ax, result_wave

    def __transform_pos_half_fft(self, wave_frame):
        # self.logger.debug(f"Transform: Pos Half FFT input shape: {wave_frame.shape}")
        # takes the DC + min_bin samples of the fft
        idx = self.frame_size // 2
        idx += self.min_bin if self.min_bin else 1
        result_wave = wave_frame[:idx]
        # self.logger.debug(f"Transform: Pos Half FFT output shape: {result_wave.shape}")
        return wave_frame, result_wave

    def __transform_absolute(self, wave_frame):
        # self.logger.debug(f"Transform: Absolute shape: {wave_frame.shape}")
        # Converts the complex value to real values
        result_wave = abs(wave_frame)
        return wave_frame, result_wave

    def __transform_haar(self, wave_frame):
        # self.logger.debug(f"Transform: HAAR shape: {wave_frame.shape}")
        result_wave = haar_forward(wave_frame)
        return wave_frame, result_wave

    def __transform_hadamard(self, wave_frame):
        # self.logger.debug(f"Transform: Hadamard output shape: {wave_frame.shape}")
        result_wave = hadamard_forward_vectorized(wave_frame)
        # self.logger.debug(f"Transform: Hadamard output shape: {result_wave.shape}")
        return wave_frame, result_wave

    def __transform_normalize(self, wave_frame):
        result_wave = wave_frame / self.frame_size
        return wave_frame, result_wave

    def __transform_log(self, wave_frame):
        result_wave = wave_frame
        if self.log_base == 'e':
            result_wave = self.log_mul * np.log(self.log_threshold + wave_frame)
        elif self.log_base == 10:
            result_wave = self.log_mul * np.log10(self.log_threshold + wave_frame)
        else:
            self.logger.warning("log_base value not defined, defaulting base of log to np.log10")
            result_wave = self.log_mul * np.log10(self.log_threshold + wave_frame)
        return wave_frame, result_wave

    # Rearrange the shape from (C,N,W) to (N,C,W,H)
    def __transform_shape(self, concatenated_features, raw_frames):
        # self.logger.debug(f"Transform: Shape input shape: {concatenated_features.shape}")
        x_temp = concatenated_features.transpose(1, 0, 2)
        x_temp = np.expand_dims(x_temp, -1)
        x_temp_raw_out = raw_frames.transpose(1, 0, 2)
        # self.logger.debug(f"Transform: Shape output shape: {x_temp.shape}")
        return x_temp, x_temp_raw_out

    def __prepare_feature_extraction_variables(self):
        '''
        This function calculates the feature extraction parameters, preprocessing flags and model information.
        The parameters are stored in self.feature_extraction_params and self.preprocessing_flags
        and then saved in user_input_config.h file to be using in application code.
        '''
        # Calculation of feature extraction parameters
        self.ch = self.variables
        self.hl = 1
        self.wl = self.frame_size

        # Store the model information parameters
        self.feature_extraction_params['FE_STACKING_CHANNELS'] = self.ch
        self.feature_extraction_params['FE_STACKING_FRAME_WIDTH'] = self.wl
        self.feature_extraction_params['FE_HL'] = self.hl
        self.feature_extraction_params['FE_NN_OUT_SIZE'] = len(self.classes)
        
        if len(self.feat_ext_transform) > 0:
            self.wl = self.feature_size_per_frame * self.num_frame_concat

            if self.stacking == '1D':
                self.wl *= self.variables
                self.ch = 1
            self.feature_extraction_params['FE_STACKING_CHANNELS'] = self.ch
            self.feature_extraction_params['FE_STACKING_FRAME_WIDTH'] = self.wl
            self.feature_size = self.feature_size_per_frame * self.num_frame_concat
            self.bin_size = self.frame_size // 2 // self.feature_size_per_frame // self.analysis_bandwidth
            # Store the feature extraction parameters
            self.feature_extraction_params['FE_FRAME_SIZE'] = self.frame_size
            self.feature_extraction_params['FE_OFFSET'] = self.offset
            self.feature_extraction_params['FE_SCALE'] = self.scale
            self.feature_extraction_params['FE_MIN_FFT_BIN'] = self.min_bin if self.min_bin != None else self.min_bin if self.min_bin else 1
            self.feature_extraction_params['FE_FFT_BIN_SIZE'] = self.bin_size
            self.feature_extraction_params['FE_NUM_FRAME_CONCAT'] = self.num_frame_concat if self.num_frame_concat else 1
            self.feature_extraction_params['FE_FFT_STAGES'] = int(np.log2(self.frame_size))
            self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] = self.feature_size_per_frame

        # Store the feature extraction parameters specifically related to FFT used for AI Library
            if 'RAW_FE' not in self.transforms:
                self.feature_extraction_params['FE_FEATURE_SIZE'] = self.feature_size
                self.feature_extraction_params['FE_FRAME_SKIP'] = self.frame_skip
                if not self.log_mul:
                    self.log_mul = 20
                    self.logger.warning(f"Defaulting log multiplier to: {self.log_mul}.")
                self.feature_extraction_params['FE_LOG_MUL'] = self.log_mul
                if not self.log_base:
                    self.log_base = 10
                    self.logger.warning(f"Defaulting log base to: {self.log_base}.")
                self.feature_extraction_params['FE_LOG_BASE'] = self.log_base
                try:
                    self.log_threshold = eval(self.log_threshold)
                except Exception as e:
                    self.log_threshold = 1e-100
                    self.logger.warning(f"Defaulting log threshold to: {self.log_threshold}. Because of exception: {e}")
            
        # Store the preprocessing flags used for AI Library
            if 'WINDOWING' in self.transforms:
                self.preprocessing_flags.append('FE_WIN')
            if 'FFT_FE' in self.transforms:
                self.preprocessing_flags.append('FE_FFT')
            if 'BINNING' in self.transforms:
                self.preprocessing_flags.append('FE_BIN')
            if 'RAW_FE' in self.transforms:
                self.preprocessing_flags.append('FE_RAW')
        
        return

    def __feature_extraction(self, x_temp, datafile):

        number_of_frames = x_temp.shape[1] // self.frame_size

        concatenated_features = []
        concatenated_raw_frames = []

        # Iterate the number of variables in dataset
        for ax in range(self.variables):
            x_temp_per_ax = x_temp[ax]

            number_of_steps = 1
            # Stores the features and raw frames for all steps of single variable
            raw_frame_per_ax = []
            features_of_frame_per_ax = []
            concatenated_raw_frames_per_ax = []
            concatenated_features_per_ax = []

            if self.offset:
                number_of_steps = x_temp_per_ax.shape[0] // self.offset
                last_n = (x_temp_per_ax.shape[0] - (
                            number_of_steps - 1) * self.offset) // self.frame_size  # elements left // self.frame_size
                if last_n == 0:
                    number_of_steps -= 1  # don't include the leftover elements

            # If no offset: number_of_steps = 1, number_of_frames formed with frame_size
            # else: calculate the number_of_frames formed by adding 1/n offsets for each frame
            for n in range(number_of_steps):
                # Store the features and raw frames of single step of single variable
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
                    raw_wave = wave_frame
                    # Contains the frame on which transforms will be applied
                    raw_frame_per_ax.append(raw_wave)

                    if 'WINDOWING' in self.transforms:
                        raw_wave, wave_frame = self.__transform_windowing(wave_frame)
                    if 'RAW_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_raw(wave_frame)
                    if 'FFT_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_fft(wave_frame)
                    if 'NORMALIZE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_normalize(wave_frame)
                    if 'FFT_POS_HALF' in self.transforms:
                        raw_wave, wave_frame = self.__transform_pos_half_fft(wave_frame)
                    if 'ABS' in self.transforms:
                        raw_wave, wave_frame = self.__transform_absolute(wave_frame)
                    if 'DC_REMOVE' in self.transforms:
                        wave_frame = wave_frame[1:]
                    # if 'HAAR' in self.transforms:
                    #     raw_wave, wave_frame = self.__transform_haar(wave_frame)
                    # if 'HADAMARD' in self.transforms:
                    #     raw_wave, wave_frame = self.__transform_hadamard(wave_frame)
                    # if 'BIN' in self.transforms:
                    #     raw_wave, wave_frame = self.__transform_basic_binning(wave_frame)
                    if 'BINNING' in self.transforms:
                        raw_wave, wave_frame = self.__transform_binning(wave_frame)
                    if 'LOG_DB' in self.transforms:
                        raw_wave, wave_frame = self.__transform_log(wave_frame)

                    # Contains the frame after all transformations are applied except concatenation
                    features_of_frame_per_ax.append(wave_frame)

                for index_frame in range(self.num_frame_concat - 1, number_of_frames):
                    if (index_frame % self.frame_skip == 0):
                        if 'CONCAT' in self.transforms:
                            _, result_wave = self.__transform_concatenation(features_of_frame_per_ax, index_frame)
                            concatenated_features_per_step.append(result_wave)
                            concatenated_raw_frames_per_step = raw_frame_per_ax[self.num_frame_concat - 1: number_of_frames]

                # Store the features and raw frames with concatenation
                concatenated_features_per_ax = concatenated_features_per_step
                concatenated_raw_frames_per_ax = concatenated_raw_frames_per_step

                # Store the features and raw frames w/o concatenating
                if self.num_frame_concat < 2:
                    concatenated_features_per_ax = features_of_frame_per_ax
                    concatenated_raw_frames_per_ax = raw_frame_per_ax

            concatenated_features.append(concatenated_features_per_ax)
            concatenated_raw_frames.append(concatenated_raw_frames_per_ax)

        concatenated_features = np.array(concatenated_features)
        concatenated_raw_frames = np.array(concatenated_raw_frames)

        if hasattr(self, 'dont_train_just_feat_ext') and str2bool(self.dont_train_just_feat_ext):
            x_raw_out_file_path = os.path.join(self.feat_ext_store_dir,
                                               os.path.splitext(os.path.basename(datafile))[0] + '_features.npy')
            np.save(x_raw_out_file_path, concatenated_features.transpose(1, 0, 2))

        x_temp, x_temp_raw_out = self.__transform_shape(concatenated_features, concatenated_raw_frames)

        return x_temp, x_temp_raw_out

    def prepare(self, **kwargs):
        self.__prepare_feature_extraction_variables()
        if hasattr(self, 'store_feat_ext_data'):
            if not str2bool_or_none(self.feat_ext_store_dir):
                self.feat_ext_store_dir = os.path.join(self.output_dir, 'feat_ext_data')
                self.logger.info(f"feat_ext_store_dir has been defaulted to: {self.feat_ext_store_dir}")
            create_dir(self.feat_ext_store_dir)
        for datafile in tqdm(self._walker):
            try:
                x_temp, y_temp, x_temp_raw_out = self._load_datafile(
                    datafile)  # Loads the dataset and applied data processing transforms
                if hasattr(self, 'dont_train_just_feat_ext') and self.dont_train_just_feat_ext:
                    x_raw_out_file_path = os.path.join(self.feat_ext_store_dir, os.path.splitext(os.path.basename(datafile))[0] + '_raw.npy')
                    np.save(x_raw_out_file_path, x_temp.flatten())
                if len(self.feat_ext_transform) > 0:
                    x_temp, x_temp_raw_out = self.__feature_extraction(x_temp.T, datafile)  # Applies feature extraction transforms
                    if hasattr(self, 'store_feat_ext_data') and self.store_feat_ext_data:
                        self._store_feat_ext(datafile, x_temp, x_temp_raw_out, y_temp)
                self._rearrange_dims(datafile, x_temp, y_temp, x_temp_raw_out)
            except ValueError as v:
                self.logger.warning(f"File will be skipped due to an error: {datafile} : {v}")
            except IndexError as i:
                self.logger.error(
                    f"Unexpected dataset dimensions. Check input options or dataset content. \nFile: {datafile}. Error message: {i}")
                exit()
            except KeyboardInterrupt as ki:
                exit()

        if not self.X.shape[0]:
            self.logger.error(
                "Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
            raise Exception(
                "Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
        if self.X.ndim == 3:
            self.X = np.expand_dims(self.X, axis=-1)
        self._process_targets()
        return self


class GenericTSDatasetAD(Dataset):
    def __init__(self, subset: str = None, dataset_dir: str = None, **kwargs):

        self.logger = getLogger("root.GenericTSDatasetAD")
        self._path = dataset_dir
        self.classes = list()
        self.label_map = dict()
        self.feature_extraction_params = dict()
        self.preprocessing_flags = []
        self.X = []
        self.Y = []

        # store all the kwargs in self
        for key, value in kwargs.items():
            value = str2int_or_float(value)
            setattr(self, key, value)

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

        # The self.transforms will contain str only
        # This is just an efficient to flatten the multidimensional list
        # TODO: Change the below line
        self.transforms = np.array(self.transforms).flatten().tolist()

        # Stores the path of data_files to be used
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

        # Files storing the datafile names
        if subset in ["training", "train"]:
            kwargs_list_key = 'training_list'
            list_filename = '*train*_list.txt'
        elif subset in ["testing", "test"]:
            kwargs_list_key = 'testing_list'
            try:
                list_filename = '*test*_list.txt'
                self._walker = load_list(kwargs_list_key, list_filename)
            except IndexError:
                list_filename = '*file*_list.txt'
                self._walker = load_list(kwargs_list_key, list_filename)
        elif subset in ["validation", "val"]:
            kwargs_list_key = 'validation_list'
            list_filename = '*val*_list.txt'

        self._walker = load_list(kwargs_list_key, list_filename)
        self.classes = sorted(set([path for path in os.listdir(dataset_dir) ]))
        return None

    # Downsample the dataset by a factor (sampling_rate/new_sr)
    def __transform_downsample(self, x_temp):
        x_temp = basic_transforms.Downsample(x_temp, self.sampling_rate, self.new_sr)
        return x_temp

    # Form windows of size frame_size from the dataset
    def __transform_simple_window(self, x_temp):
        stride_window = int(self.frame_size * self.stride_size)
        if stride_window == 0:
            self.logger.warning(
                "Stride Window size calculated is 0. Defaulting the value to Sample Window i.e. no overlap.")
            stride_window = self.frame_size
        if not hasattr(self, 'keep_short_tails'):
            self.keep_short_tails = False
        x_temp = np.array(
            basic_transforms.SimpleWindow(x_temp, self.frame_size, stride_window, self.keep_short_tails))
        if len(self.feat_ext_transform) > 0:
            x_temp = x_temp.reshape(-1, self.variables)
        return x_temp

    # Stores the data in x_temp from the datafile
    def _load_datafile(self, datafile):
        file_extension = ops(datafile)[-1]

        # For .npy files, we assume it is regular numpy array and not structured array with named columns.
        # Also we assume first column is timestamp, hence excluding first column.
        if file_extension == ".npy":
            x_temp = np.load(datafile)
            if len(x_temp.shape) > 1:
                x_temp = x_temp[:, 1:self.variables+1]

        elif file_extension in [".pkl", ".csv", ".txt"]:
            # Note: When saving a DataFrame to .pkl, if headers are provided, they are saved as metadata. If not pandas assigns default headers (0,1,2...).
            # In both these cases headers are not part of .values array.
            if file_extension == ".pkl":
                x_temp = pd.read_pickle(datafile)
                if not isinstance(x_temp,pd.DataFrame):
                    raise TypeError("Data loaded from .pkl file is not a pandas DataFrame. Please check the file format.")
                if x_temp.shape[0]==0:
                    raise ValueError("File has no rows.")
                x_temp = x_temp[[col for col in x_temp.columns if 'time' not in str(col).lower()]] 
            else:
                # For .csv files, first row is considered as value and not header, hence specifying header=None. 
                x_temp = pd.read_csv(datafile, header=None, dtype=str) # Read as string to avoid dtype issues.
                if x_temp.shape[0]==0:
                    raise ValueError("File has no rows.")
                x_temp=x_temp[[col_index for col_index,value in x_temp.iloc[0].items() if 'time' not in str(value).lower()]]
                try:
                    float(x_temp.iloc[0, 0])
                except (ValueError, TypeError):
                    x_temp = x_temp[1:]

            x_temp = x_temp.values.astype(float) # Converting values from str back to float.
            x_temp = x_temp[:, :self.variables]
        else:
            raise Exception("Supports only .npy, .pkl, .txt and .csv file formats for now")
        label = opb(opd(datafile))

        # Perform the data processing transformations
        if 'Downsample' in self.transforms:
            x_temp = self.__transform_downsample(x_temp)
        if 'SimpleWindow' in self.transforms:
            x_temp = self.__transform_simple_window(x_temp)
        if self.scale:
            x_temp = x_temp / float(self.scale)
        # Apply augmenters
        x_temp = apply_augmenters(x_temp,
                                  self.augment_pipeline)  # https://tsaug.readthedocs.io/en/stable/quickstart.html

        x_temp_raw_out = x_temp.copy()
        return x_temp, label, x_temp_raw_out

    # Stores the (x_temp, x_temp_raw_out, y_temp) after applying all the feat-ext-transform
    def _store_feat_ext(self, datafile, x_temp, x_temp_raw_out, label):
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
                self.logger.warning(
                    "'store_feat_ext_data' chosen but 'feat_ext_store_dir' not provided. Skipping storage")
        return

    # Rearranges the dimensions of x_temp to generalize for different channels
    def _rearrange_dims(self, datafile, x_temp, label, x_temp_raw_out, **kwargs):
        y_temp = np.array([label for i in range(x_temp.shape[0])])
        try:
            if x_temp.ndim == 2:
                self.logger.warning("Not enough dimensions present. Extract more features")
                raise Exception("Not enough dimensions present. Extract more features")
            if x_temp.ndim == 3:
                x_temp = x_temp.transpose(0, 2, 1)
                x_temp = np.expand_dims(x_temp, axis=3)
            if len(self.X) == 0:
                self.X = x_temp
                self.X_raw = x_temp_raw_out
            else:
                self.X = np.concatenate((self.X, x_temp))
                self.X_raw = np.concatenate((self.X_raw, x_temp_raw_out))
            self.Y = np.concatenate((self.Y, y_temp))
        except ValueError as e:
            self.logger.warning('Skipping {} as Error encountered: {}'.format(datafile, e))
        return

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X_raw[index]),torch.from_numpy(self.X[index]), torch.from_numpy(np.array([self.Y[index]]))

    def _process_targets(self):
        self.label_map = {k: v for v, k in enumerate(self.classes)}  # {'class_0': 0, 'class_1': 1}
        self.inverse_label_map = {k: v for v, k in self.label_map.items()}  # {0 :'class_0', 1: 'class_1'}
        self.Y = [self.label_map[i] for i in self.Y]

        if not len(self.Y):
            self.logger.error(
                "No data could be loaded. Either file paths were erroneous or dimensions mismatched. Check prior logger messages/ data processing configurations")
            raise Exception(
                "No data could be loaded. Either file paths were erroneous or dimensions mismatched. Check prior logger messages/ data processing configurations")
        return

    def __transform_raw(self, wave_frame):
        # self.logger.debug(f"Transform: RAW shape: {wave_frame.shape}")
        result_wave = wave_frame
        mean_of_wave = np.sum(result_wave) // len(result_wave)
        if self.dc_remove:
            result_wave = result_wave - mean_of_wave
        return wave_frame, result_wave

    def __transform_windowing(self, wave_frame):
        # self.logger.debug(f"Transform: Windowing shape: {wave_frame.shape}")
        result_wave = wave_frame * np.hanning(self.frame_size)
        return wave_frame, result_wave

    def __transform_fft(self, wave_frame):
        # self.logger.debug(f"Transform: FFT shape: {wave_frame.shape}")
        result_wave = np.fft.fft(wave_frame)
        return wave_frame, result_wave

    def __transform_binning(self, wave_frame):
        # self.logger.debug(f"Transform: Binning input shape: {wave_frame.shape}")
        result_wave = np.empty((self.feature_size_per_frame))
        bin_size = self.bin_size
        bin_offset = self.min_bin if self.min_bin else 0

        for idx in range(self.feature_size_per_frame):
            idx1 = bin_offset + (idx) * bin_size
            sum_of_bin = np.sum(wave_frame[idx1:idx1 + bin_size])
            result_wave[idx] = sum_of_bin
            result_wave[idx] /= bin_size if self.normalize_bin else 1
        # self.logger.debug(f"Transform: Binning output shape: {result_wave.shape}")
        return wave_frame, result_wave

    def __transform_concatenation(self, features_of_frame_per_ax, index_frame):
        # self.logger.debug(f"Transform: Concatenation input shape: {np.array(features_of_frame_per_ax).shape}")
        result_wave = np.array(
            features_of_frame_per_ax[index_frame - self.num_frame_concat + 1: index_frame + 1]).flatten()
        # self.logger.debug(f"Transform: Concatenation output shape: {result_wave.shape}")
        return features_of_frame_per_ax, result_wave

    def __transform_pos_half_fft(self, wave_frame):
        # self.logger.debug(f"Transform: Pos Half FFT input shape: {wave_frame.shape}")
        # takes the DC + min_bin samples of the fft
        idx = self.frame_size // 2
        idx += self.min_bin if self.min_bin else 1
        result_wave = wave_frame[:idx]
        # self.logger.debug(f"Transform: Pos Half FFT output shape: {result_wave.shape}")
        return wave_frame, result_wave

    def __transform_absolute(self, wave_frame):
        # self.logger.debug(f"Transform: Absolute shape: {wave_frame.shape}")
        # Converts the complex value to real values
        result_wave = abs(wave_frame)
        return wave_frame, result_wave

    def __transform_haar(self, wave_frame):
        # self.logger.debug(f"Transform: HAAR shape: {wave_frame.shape}")
        result_wave = haar_forward(wave_frame)
        return wave_frame, result_wave

    def __transform_hadamard(self, wave_frame):
        # self.logger.debug(f"Transform: Hadamard output shape: {wave_frame.shape}")
        result_wave = hadamard_forward_vectorized(wave_frame)
        # self.logger.debug(f"Transform: Hadamard output shape: {result_wave.shape}")
        return wave_frame, result_wave

    def __transform_normalize(self, wave_frame):
        result_wave = wave_frame / self.frame_size
        return wave_frame, result_wave

    def __transform_log(self, wave_frame):
        result_wave = wave_frame
        if self.log_base == 'e':
            result_wave = self.log_mul * np.log(self.log_threshold + wave_frame)
        elif self.log_base == 10:
            result_wave = self.log_mul * np.log10(self.log_threshold + wave_frame)
        else:
            self.logger.warning("log_base value not defined, defaulting base of log to np.log10")
            result_wave = self.log_mul * np.log10(self.log_threshold + wave_frame)
        return wave_frame, result_wave

    # Rearrange the shape from (C,N,W) to (N,C,W,H)
    def __transform_shape(self, concatenated_features, raw_frames):
        # self.logger.debug(f"Transform: Shape input shape: {concatenated_features.shape}")
        N = concatenated_features.shape[1]
        x_temp = concatenated_features.transpose(1, 0, 2).reshape(N, self.ch, self.wl, self.hl)
        x_temp_raw_out = raw_frames.transpose(1, 0, 2)
        # self.logger.debug(f"Transform: Shape output shape: {x_temp.shape}")
        return x_temp, x_temp_raw_out

    def __prepare_feature_extraction_variables(self):
        '''
        This function calculates the feature extraction parameters, preprocessing flags and model information.
        The parameters are stored in self.feature_extraction_params and self.preprocessing_flags
        and then saved in user_input_config.h file to be using in application code.
        '''
        # Calculation of feature extraction parameters
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
            self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] =  self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME']//2 + (self.min_bin if self.min_bin else 1)
        if 'DC_REMOVE' in self.transforms:
            self.preprocessing_flags.append('FE_DC_REM')    
            self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] =  self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] - 1
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
                self.feature_extraction_params['FE_LOG_TOL']=self.log_threshold
        if 'CONCAT' in self.transforms:
            self.preprocessing_flags.append('FE_CONCAT')
            self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME'] *= self.num_frame_concat
        
        self.wl = self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME']
        
        if self.stacking == '1D':
            self.wl *= self.variables
            self.ch = 1
            
        self.feature_extraction_params['FE_STACKING_CHANNELS'] = self.ch
        self.feature_extraction_params['FE_STACKING_FRAME_WIDTH'] = self.wl
        self.feature_extraction_params['FE_OFFSET'] = self.offset
        self.feature_extraction_params['FE_SCALE'] = self.scale
        self.feature_extraction_params['FE_MIN_FFT_BIN'] = self.min_bin if self.min_bin != None else self.min_bin if self.min_bin else 1
        self.feature_extraction_params['FE_NUM_FRAME_CONCAT'] = self.num_frame_concat if self.num_frame_concat else 1
        self.feature_extraction_params['FE_NN_OUT_SIZE'] = self.wl*self.ch
        self.feature_extraction_params['FE_FEATURE_SIZE'] = self.wl
        self.feature_extraction_params['FE_FRAME_SKIP'] = self.frame_skip
        self.feature_extraction_params['FE_FFT_STAGES'] = int(np.log2(self.frame_size))
        return

    def __feature_extraction(self, x_temp, datafile):

        number_of_frames = x_temp.shape[1] // self.frame_size

        concatenated_features = []
        concatenated_raw_frames = []

        # Iterate the number of variables in dataset
        for ax in range(self.variables):
            x_temp_per_ax = x_temp[ax]

            number_of_steps = 1
            # Stores the features and raw frames for all steps of single variable
            raw_frame_per_ax = []
            features_of_frame_per_ax = []
            concatenated_raw_frames_per_ax = []
            concatenated_features_per_ax = []

            if self.offset:
                number_of_steps = x_temp_per_ax.shape[0] // self.offset
                last_n = (x_temp_per_ax.shape[0] - (
                            number_of_steps - 1) * self.offset) // self.frame_size  # elements left // self.frame_size
                if last_n == 0:
                    number_of_steps -= 1  # don't include the leftover elements

            # If no offset: number_of_steps = 1, number_of_frames formed with frame_size
            # else: calculate the number_of_frames formed by adding 1/n offsets for each frame
            for n in range(number_of_steps):
                # Store the features and raw frames of single step of single variable
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
                    raw_wave = wave_frame
                    # Contains the frame on which transforms will be applied
                    raw_frame_per_ax.append(raw_wave)

                    if 'WINDOWING' in self.transforms:
                        raw_wave, wave_frame = self.__transform_windowing(wave_frame)
                    if 'RAW_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_raw(wave_frame)
                    if 'FFT_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_fft(wave_frame)
                    if 'NORMALIZE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_normalize(wave_frame)
                    if 'FFT_POS_HALF' in self.transforms:
                        raw_wave, wave_frame = self.__transform_pos_half_fft(wave_frame)
                    if 'ABS' in self.transforms:
                        raw_wave, wave_frame = self.__transform_absolute(wave_frame)
                    if 'DC_REMOVE' in self.transforms:
                        wave_frame = wave_frame[1:]
                    # if 'HAAR' in self.transforms:
                    #     raw_wave, wave_frame = self.__transform_haar(wave_frame)
                    # if 'HADAMARD' in self.transforms:
                    #     raw_wave, wave_frame = self.__transform_hadamard(wave_frame)
                    # if 'BIN' in self.transforms:
                    #     raw_wave, wave_frame = self.__transform_basic_binning(wave_frame)
                    if 'BINNING' in self.transforms:
                        raw_wave, wave_frame = self.__transform_binning(wave_frame)
                    if 'LOG_DB' in self.transforms:
                        raw_wave, wave_frame = self.__transform_log(wave_frame)

                    # Contains the frame after all transformations are applied except concatenation
                    features_of_frame_per_ax.append(wave_frame)
                if number_of_frames < self.num_frame_concat:
                    self.logger.warning(f"Only {number_of_frames} frames available in {datafile}, but {self.num_frame_concat} required for concatenation")
                for index_frame in range(self.num_frame_concat - 1, number_of_frames):
                    if (index_frame % self.frame_skip == 0):
                        if 'CONCAT' in self.transforms:
                            _, result_wave = self.__transform_concatenation(features_of_frame_per_ax, index_frame)
                            concatenated_features_per_step.append(result_wave)
                            concatenated_raw_frames_per_step = raw_frame_per_ax[
                                                               self.num_frame_concat - 1: number_of_frames]

                # Store the features and raw frames with concatenation
                concatenated_features_per_ax = concatenated_features_per_step
                concatenated_raw_frames_per_ax = concatenated_raw_frames_per_step

                # Store the features and raw frames w/o concatenating
                if self.num_frame_concat < 2:
                    concatenated_features_per_ax = features_of_frame_per_ax
                    concatenated_raw_frames_per_ax = raw_frame_per_ax

            concatenated_features.append(concatenated_features_per_ax)
            concatenated_raw_frames.append(concatenated_raw_frames_per_ax)

        concatenated_features = np.array(concatenated_features)
        concatenated_raw_frames = np.array(concatenated_raw_frames)

        if hasattr(self, 'dont_train_just_feat_ext') and str2bool(self.dont_train_just_feat_ext):
            x_raw_out_file_path = os.path.join(self.feat_ext_store_dir,
                                               os.path.splitext(os.path.basename(datafile))[0] + '_features.npy')
            np.save(x_raw_out_file_path, concatenated_features.transpose(1, 0, 2))

        x_temp, x_temp_raw_out = self.__transform_shape(concatenated_features, concatenated_raw_frames)

        return x_temp, x_temp_raw_out

    def prepare(self, **kwargs):
        self.__prepare_feature_extraction_variables()
        if hasattr(self, 'store_feat_ext_data'):
            if not str2bool_or_none(self.feat_ext_store_dir):
                self.feat_ext_store_dir = os.path.join(self.output_dir, 'feat_ext_data')
                self.logger.info(f"feat_ext_store_dir has been defaulted to: {self.feat_ext_store_dir}")
            create_dir(self.feat_ext_store_dir)
        for datafile in tqdm(self._walker):
            try:
                x_temp, label, x_temp_raw_out = self._load_datafile(
                    datafile)  # Loads the dataset and applied data processing transforms
                if hasattr(self, 'dont_train_just_feat_ext') and self.dont_train_just_feat_ext:
                    x_raw_out_file_path = os.path.join(self.feat_ext_store_dir,
                                                       os.path.splitext(os.path.basename(datafile))[0] + '_raw.npy')
                    np.save(x_raw_out_file_path, x_temp.flatten())
                if len(self.feat_ext_transform) > 0:
                    x_temp, x_temp_raw_out = self.__feature_extraction(x_temp.T,
                                                                       datafile)  # Applies feature extraction transforms
                    if hasattr(self, 'store_feat_ext_data') and self.store_feat_ext_data:
                        self._store_feat_ext(datafile, x_temp, x_temp_raw_out, label)
                self._rearrange_dims(datafile, x_temp, label, x_temp_raw_out)
            except ValueError as v:
                self.logger.warning(f"File will be skipped due to an error: {datafile} : {v}")
            except IndexError as i:
                self.logger.error(
                    f"Unexpected dataset dimensions. Check input options or dataset content. \nFile: {datafile}. Error message: {i}")
                exit()
            except KeyboardInterrupt as ki:
                exit()

        if not self.X.shape[0]:
            self.logger.error(
                "Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
            raise Exception(
                "Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
        if self.X.ndim == 3:
            self.X = np.expand_dims(self.X, axis=-1)
        self._process_targets()
        return self
    
class GenericTSDatasetForecasting(Dataset):
    def __init__(self, subset: str = None, dataset_dir: str = None, **kwargs):
        self.logger = getLogger("root.GenericTSDatasetForecasting")
        self._path = dataset_dir
        self.label_map = dict()
        self.feature_extraction_params = dict()
        self.preprocessing_flags = []
        self.X = []
        self.Y = []

        # store all the kwargs in self
        for key, value in kwargs.items():
            value = str2int_or_float(value)
            setattr(self, key, value)

        # TODO: Augmentation 
        '''
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
        '''

        # The self.transforms will contain str only
        # This is just an efficient way to flatten the multidimensional list
        # TODO: Change the below line
        self.transforms = np.array(self.transforms).flatten().tolist()

        # Stores the path of data_files to be used
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

        # Files storing the datafile names
        if subset in ["training", "train"]:
            kwargs_list_key = 'training_list'
            list_filename = '*train*_list.txt'
        elif subset in ["testing", "test"]:
            kwargs_list_key = 'testing_list'
            try:
                list_filename = '*test*_list.txt'
                self._walker = load_list(kwargs_list_key, list_filename)
            except IndexError:
                list_filename = '*file*_list.txt'
                self._walker = load_list(kwargs_list_key, list_filename)
        elif subset in ["validation", "val"]:
            kwargs_list_key = 'validation_list'
            list_filename = '*val*_list.txt'

        self._walker = load_list(kwargs_list_key, list_filename)

    # Downsample the dataset by a factor (sampling_rate/new_sr)
    def __transform_downsample(self, x_temp, y_temp):
        x_temp = basic_transforms.Downsample(x_temp, self.sampling_rate, self.new_sr)
        y_temp = basic_transforms.Downsample(y_temp, self.sampling_rate, self.new_sr)
        return x_temp,y_temp

    # Form windows of size frame_size from the dataset
    def __transform_simple_window(self, x_temp, y_temp):
        stride_window = int(self.frame_size * self.stride_size)
        if stride_window == 0:
            self.logger.warning(
                "Stride Window size calculated is 0. Defaulting the value to Sample Window i.e. no overlap."
            )
            stride_window = self.frame_size

        if not hasattr(self, 'keep_short_tails'):
            self.keep_short_tails = False

        x_windows = []
        y_windows = []

        total_length = len(x_temp)
        max_index = total_length - (self.frame_size + self.forecast_horizon) + 1

        for i in range(0, max_index, stride_window):
            x_window = x_temp[i:i + self.frame_size] # shape: (frame_size, variables)
            y_window = y_temp[i + self.frame_size:i + self.frame_size + self.forecast_horizon,[col_num for col_name in self.header_row for col_num in col_name.values()]] # shape: (forecast_horizon, num_columns)

            x_windows.append(x_window)
            y_windows.append(y_window)
        
        x_windows = np.array(x_windows)  # shape: (num_windows, frame_size, variables)
        y_windows = np.array(y_windows)  # shape: (num_windows, forecast_horizon, num_columns)
        
        return x_windows, y_windows

    def _load_datafile(self, datafile):
        file_extension = ops(datafile)[-1]
        if file_extension == ".npy":
            data = np.load(datafile)
            if len(data.shape) > 1:
                x_temp = data[:, 1:self.variables+1]
                y_temp= data[:,1:]
        elif file_extension in [".pkl", ".csv", ".txt"]:
            # Note: When saving a DataFrame to .pkl, if headers are provided, they are saved as metadata. If not pandas assigns default headers (0,1,2...).
            # In both these cases headers are not part of .values array.
            if file_extension == ".pkl":
                data = pd.read_pickle(datafile)
                if not isinstance(data,pd.DataFrame):
                    raise TypeError("Data loaded from .pkl file is not a pandas DataFrame. Please check the file format.")
                if data.shape[0]==0:
                    raise ValueError("File has no rows.")
                data = data[[col for col in data.columns if 'time' not in str(col).lower()]]
            else:
                # For .csv files, first row is considered as value and not header, hence specifying header=None. 
                data = pd.read_csv(datafile, header=None, dtype=str) # Read as string to avoid dtype issues
                if data.shape[0]==0:
                    raise ValueError("File has no rows.")
                data=data[[col_index for col_index,value in data.iloc[0].items() if 'time' not in str(value).lower()]]
                try:
                    float(data.iloc[0, 0])
                    if self.target_variables == []:
                        self.header_row = [{'{}'.format(idx): idx} for idx in range(len(data.iloc[0].tolist()))]
                    else:
                        self.header_row= [{str(idx): int(idx)} for idx in self.target_variables]
                except (ValueError, TypeError):
                    is_int=lambda x: isinstance(x, int) or (isinstance(x, str) and x.isdigit())
                    if self.target_variables==[]:
                        self.header_row = [{value: idx} for idx, value in enumerate(data.iloc[0].tolist())]
                    elif is_int(self.target_variables[0]):
                        self.header_row = [{data.iloc[0].tolist()[int(idx)]: int(idx)} for idx in self.target_variables]
                    elif isinstance(self.target_variables[0], str):
                        self.header_row = [{value: data.iloc[0].tolist().index(value)} for value in self.target_variables]
                    data = data[1:]
            data = data.values.astype(float) 
            
            from sklearn.preprocessing import MinMaxScaler
            scaler=MinMaxScaler()
            data = scaler.fit_transform(data)  # Scale the data to [0, 1] range
            

            x_temp = data[:, :self.variables]
            y_temp=data
        else:
            raise Exception("Supports only .npy, .pkl, .txt and .csv file formats for now")
        if 'Downsample' in self.transforms:
            x_temp, y_temp = self.__transform_downsample(x_temp, y_temp)
        if 'SimpleWindow' in self.transforms and len(self.feat_ext_transform) == 0:
            x_temp, y_temp = self.__transform_simple_window(x_temp, y_temp)
        if self.scale:
            x_temp = x_temp / float(self.scale)

        #x_temp = apply_augmenters(x_temp, self.augment_pipeline)

        x_temp_raw_out = x_temp.copy()
        return x_temp, y_temp, x_temp_raw_out

    '''
    # Stores the (x_temp, x_temp_raw_out, y_temp) after applying all the feat-ext-transform
    def _store_feat_ext(self, datafile, x_temp, x_temp_raw_out, label):
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
                self.logger.warning(
                    "'store_feat_ext_data' chosen but 'feat_ext_store_dir' not provided. Skipping storage")
        return
    '''

    # Rearranges the dimensions of x_temp to generalize for different channels
    def _rearrange_dims(self, datafile, x_temp, y_temp, x_temp_raw_out, **kwargs):
        # y_temp = np.array([label for i in range(x_temp.shape[0])])
        try:
            if x_temp.ndim == 2:
                self.logger.warning("Not enough dimensions present. Extract more features")
                raise Exception("Not enough dimensions present. Extract more features")
            if x_temp.ndim == 3:
                x_temp = x_temp.transpose(0, 2, 1)
                x_temp_raw_out = x_temp_raw_out.transpose(0, 2, 1)
                x_temp = np.expand_dims(x_temp, axis=3)
            if len(self.X) == 0:
                self.X = x_temp
                self.X_raw = x_temp_raw_out
            else:
                self.X = np.concatenate((self.X, x_temp))
                self.X_raw = np.concatenate((self.X_raw, x_temp_raw_out))
            self.Y = np.concatenate((self.Y, y_temp)) if len(self.Y) else y_temp
        except ValueError as e:
            self.logger.warning('Skipping {} as Error encountered: {}'.format(datafile, e))
        return

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X_raw[index]), torch.from_numpy(self.X[index]), torch.from_numpy(self.Y[index])

    #TODO: Enable Feature Extraction transforms for forecasting
    '''
    def __transform_raw(self, wave_frame):
        # self.logger.debug(f"Transform: RAW shape: {wave_frame.shape}")
        result_wave = wave_frame
        mean_of_wave = np.sum(result_wave) // len(result_wave)
        if self.dc_remove:
            result_wave = result_wave - mean_of_wave
        return wave_frame, result_wave

    def __transform_windowing(self, wave_frame):
        # self.logger.debug(f"Transform: Windowing shape: {wave_frame.shape}")
        result_wave = wave_frame * np.hanning(self.frame_size)
        return wave_frame, result_wave

    def __transform_fft(self, wave_frame):
        # self.logger.debug(f"Transform: FFT shape: {wave_frame.shape}")
        result_wave = np.fft.fft(wave_frame)
        return wave_frame, result_wave

    def __transform_binning(self, wave_frame):
        # self.logger.debug(f"Transform: Binning input shape: {wave_frame.shape}")
        result_wave = np.empty((self.feature_size_per_frame))
        bin_size = self.bin_size
        bin_offset = self.min_bin if self.min_bin else 0

        for idx in range(self.feature_size_per_frame):
            idx1 = bin_offset + (idx) * bin_size
            sum_of_bin = np.sum(wave_frame[idx1:idx1 + bin_size])
            result_wave[idx] = sum_of_bin
            result_wave[idx] /= bin_size if self.normalize_bin else 1
        # self.logger.debug(f"Transform: Binning output shape: {result_wave.shape}")
        return wave_frame, result_wave

    def __transform_concatenation(self, features_of_frame_per_ax, index_frame):
        # self.logger.debug(f"Transform: Concatenation input shape: {np.array(features_of_frame_per_ax).shape}")
        result_wave = np.array(
            features_of_frame_per_ax[index_frame - self.num_frame_concat + 1: index_frame + 1]).flatten()
        # self.logger.debug(f"Transform: Concatenation output shape: {result_wave.shape}")
        return features_of_frame_per_ax, result_wave

    def __transform_pos_half_fft(self, wave_frame):
        # self.logger.debug(f"Transform: Pos Half FFT input shape: {wave_frame.shape}")
        # takes the DC + min_bin samples of the fft
        idx = self.frame_size // 2
        idx += self.min_bin if self.min_bin else 1
        result_wave = wave_frame[:idx]
        # self.logger.debug(f"Transform: Pos Half FFT output shape: {result_wave.shape}")
        return wave_frame, result_wave

    def __transform_absolute(self, wave_frame):
        # self.logger.debug(f"Transform: Absolute shape: {wave_frame.shape}")
        # Converts the complex value to real values
        result_wave = abs(wave_frame)
        return wave_frame, result_wave

    def __transform_haar(self, wave_frame):
        # self.logger.debug(f"Transform: HAAR shape: {wave_frame.shape}")
        result_wave = haar_forward(wave_frame)
        return wave_frame, result_wave

    def __transform_hadamard(self, wave_frame):
        # self.logger.debug(f"Transform: Hadamard output shape: {wave_frame.shape}")
        result_wave = hadamard_forward_vectorized(wave_frame)
        # self.logger.debug(f"Transform: Hadamard output shape: {result_wave.shape}")
        return wave_frame, result_wave

    def __transform_normalize(self, wave_frame):
        result_wave = wave_frame / self.frame_size
        return wave_frame, result_wave

    def __transform_log(self, wave_frame):
        result_wave = wave_frame
        if self.log_base == 'e':
            result_wave = self.log_mul * np.log(self.log_threshold + wave_frame)
        elif self.log_base == 10:
            result_wave = self.log_mul * np.log10(self.log_threshold + wave_frame)
        else:
            self.logger.warning("log_base value not defined, defaulting base of log to np.log10")
            result_wave = self.log_mul * np.log10(self.log_threshold + wave_frame)
        return wave_frame, result_wave

    # Rearrange the shape from (C,N,W) to (N,C,W,H)
    def __transform_shape(self, concatenated_features, raw_frames):
        # self.logger.debug(f"Transform: Shape input shape: {concatenated_features.shape}")
        x_temp = concatenated_features.transpose(1, 0, 2)
        x_temp = np.expand_dims(x_temp, -1)
        x_temp_raw_out = raw_frames.transpose(1, 0, 2)
        # self.logger.debug(f"Transform: Shape output shape: {x_temp.shape}")
        return x_temp, x_temp_raw_out
    '''
    def __prepare_feature_extraction_variables(self):
        #print("All self attributes:")
        #for key, value in vars(self).items():
        #    print(f"{key}: {value}")
        import ast
        self.target_variables = ast.literal_eval(self.target_variables)
        '''
        This function calculates the feature extraction parameters, preprocessing flags and model information.
        The parameters are stored in self.feature_extraction_params and self.preprocessing_flags
        and then saved in user_input_config.h file to be using in application code.
        '''

        # Calculation of feature extraction parameters
        self.ch = self.variables
        self.hl = 1
        self.wl = self.frame_size

        # Below are the parameters required even when feature extraction transforms are not applied.
        self.feature_extraction_params['FE_STACKING_CHANNELS'] = self.ch
        self.feature_extraction_params['FE_VARIABLES'] = self.variables
        self.feature_extraction_params['FE_STACKING_FRAME_WIDTH'] = self.wl
        self.feature_extraction_params['FE_HL'] = self.hl
        self.feature_extraction_params['FE_FRAME_SIZE'] = self.frame_size
        # feature_size per frame is required even if feature extraction transforms are not there because it is required for batchnorm in feature extraction library
        self.feature_extraction_params['FE_FEATURE_SIZE_PER_FRAME']=self.frame_size
        self.feature_extraction_params['FE_NN_OUT_SIZE']=self.forecast_horizon*len(self.target_variables)
        return
    '''
    def __feature_extraction(self, x_temp, datafile):
        number_of_frames = x_temp.shape[1] // self.frame_size
        concatenated_features = []
        concatenated_raw_frames = []

        # Iterate the number of variables in dataset
        for ax in range(self.variables):
            x_temp_per_ax = x_temp[ax]

            number_of_steps = 1
            # Stores the features and raw frames for all steps of single variable
            raw_frame_per_ax = []
            features_of_frame_per_ax = []
            concatenated_raw_frames_per_ax = []
            concatenated_features_per_ax = []

            if self.offset:
                number_of_steps = x_temp_per_ax.shape[0] // self.offset
                last_n = (x_temp_per_ax.shape[0] - (
                            number_of_steps - 1) * self.offset) // self.frame_size  # elements left // self.frame_size
                if last_n == 0:
                    number_of_steps -= 1  # don't include the leftover elements

            # If no offset: number_of_steps = 1, number_of_frames formed with frame_size
            # else: calculate the number_of_frames formed by adding 1/n offsets for each frame
            for n in range(number_of_steps):
                # Store the features and raw frames of single step of single variable
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
                    raw_wave = wave_frame
                    # Contains the frame on which transforms will be applied
                    raw_frame_per_ax.append(raw_wave)

                    if 'WINDOWING' in self.transforms:
                        raw_wave, wave_frame = self.__transform_windowing(wave_frame)
                    if 'RAW_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_raw(wave_frame)
                    if 'FFT_FE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_fft(wave_frame)
                    if 'NORMALIZE' in self.transforms:
                        raw_wave, wave_frame = self.__transform_normalize(wave_frame)
                    if 'FFT_POS_HALF' in self.transforms:
                        raw_wave, wave_frame = self.__transform_pos_half_fft(wave_frame)
                    if 'ABS' in self.transforms:
                        raw_wave, wave_frame = self.__transform_absolute(wave_frame)
                    if 'DC_REMOVE' in self.transforms:
                        wave_frame = wave_frame[1:]
                    # if 'HAAR' in self.transforms:
                    #     raw_wave, wave_frame = self.__transform_haar(wave_frame)
                    # if 'HADAMARD' in self.transforms:
                    #     raw_wave, wave_frame = self.__transform_hadamard(wave_frame)
                    # if 'BIN' in self.transforms:
                    #     raw_wave, wave_frame = self.__transform_basic_binning(wave_frame)
                    if 'BINNING' in self.transforms:
                        raw_wave, wave_frame = self.__transform_binning(wave_frame)
                    if 'LOG_DB' in self.transforms:
                        raw_wave, wave_frame = self.__transform_log(wave_frame)

                    # Contains the frame after all transformations are applied except concatenation
                    features_of_frame_per_ax.append(wave_frame)

                for index_frame in range(self.num_frame_concat - 1, number_of_frames):
                    if (index_frame % self.frame_skip == 0):
                        if 'CONCAT' in self.transforms:
                            _, result_wave = self.__transform_concatenation(features_of_frame_per_ax, index_frame)
                            concatenated_features_per_step.append(result_wave)
                            concatenated_raw_frames_per_step = raw_frame_per_ax[self.num_frame_concat - 1: number_of_frames]

                # Store the features and raw frames with concatenation
                concatenated_features_per_ax = concatenated_features_per_step
                concatenated_raw_frames_per_ax = concatenated_raw_frames_per_step

                # Store the features and raw frames w/o concatenating
                if self.num_frame_concat < 2:
                    concatenated_features_per_ax = features_of_frame_per_ax
                    concatenated_raw_frames_per_ax = raw_frame_per_ax

            concatenated_features.append(concatenated_features_per_ax)
            concatenated_raw_frames.append(concatenated_raw_frames_per_ax)

        concatenated_features = np.array(concatenated_features)
        concatenated_raw_frames = np.array(concatenated_raw_frames)

        if hasattr(self, 'dont_train_just_feat_ext') and str2bool(self.dont_train_just_feat_ext):
            x_raw_out_file_path = os.path.join(self.feat_ext_store_dir,
                                               os.path.splitext(os.path.basename(datafile))[0] + '_features.npy')
            np.save(x_raw_out_file_path, concatenated_features.transpose(1, 0, 2))
        x_temp, x_temp_raw_out = self.__transform_shape(concatenated_features, concatenated_raw_frames)
        return x_temp, x_temp_raw_out
    '''
    def prepare(self, **kwargs):
        
        self.__prepare_feature_extraction_variables()
        '''
        if hasattr(self, 'store_feat_ext_data'):
            if not str2bool_or_none(self.feat_ext_store_dir):
                self.feat_ext_store_dir = os.path.join(self.output_dir, 'feat_ext_data')
                self.logger.info(f"feat_ext_store_dir has been defaulted to: {self.feat_ext_store_dir}")
            create_dir(self.feat_ext_store_dir)
        '''
        for datafile in tqdm(self._walker):
            try:
                x_temp, y_temp, x_temp_raw_out = self._load_datafile(datafile)  # Loads the dataset and applied data processing transforms
                '''
                if hasattr(self, 'dont_train_just_feat_ext') and self.dont_train_just_feat_ext:
                    x_raw_out_file_path = os.path.join(self.feat_ext_store_dir, os.path.splitext(os.path.basename(datafile))[0] + '_raw.npy')
                    np.save(x_raw_out_file_path, x_temp.flatten())
                '''
                '''
                if len(self.feat_ext_transform) > 0:
                    x_temp, x_temp_raw_out = self.__feature_extraction(x_temp.T, datafile)  # Applies feature extraction transforms
                    if hasattr(self, 'store_feat_ext_data') and self.store_feat_ext_data:
                        self._store_feat_ext(datafile, x_temp, x_temp_raw_out, y_temp)
                '''
                self._rearrange_dims(datafile, x_temp, y_temp, x_temp_raw_out)
            except ValueError as v:
                self.logger.warning(f"File will be skipped due to an error: {datafile} : {v}")
            except IndexError as i:
                self.logger.error(
                    f"Unexpected dataset dimensions. Check input options or dataset content. \nFile: {datafile}. Error message: {i}")
                exit()
            except KeyboardInterrupt as ki:
                exit()
        if not self.X.shape[0]:
            self.logger.error(
                "Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
            raise Exception(
                "Aborting run as the dataset loaded is empty. Check either input options or data or compatibility between the two.")
        if self.X.ndim == 3:
            self.X = np.expand_dims(self.X, axis=-1)
        return self