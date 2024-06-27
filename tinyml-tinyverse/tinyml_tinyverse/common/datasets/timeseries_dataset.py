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

from logging import getLogger
import numpy as np
import os
from os.path import join as opj, basename as opb, splitext as ops, dirname as opd
from glob import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from ..transforms import basic_transforms
from tqdm import tqdm


class SimpleTSDataset(Dataset):
    """Univariate & Multivariate Time Series Dataset."""

    def __init__(self, subset: str = None, dataset_dir: str = None, transforms: list = [], org_sr=313000,
                 sequence_window=0.25, variables=1, **kwargs):
        """
        Parameters
        ----------
        subset
        dataset_dir
        transforms
        org_sr
        sequence_window
        variables
        kwargs: For MFCC/STFT
        """
        self._path = dataset_dir
        self.transforms = transforms
        if len(transforms) and isinstance(transforms[0], list):  # When input is received from modelmaker, it becomes [['DownSample', 'SimpleWindow']]
            self.transforms = []
            [self.transforms.extend(x.split('_')) for x in transforms[0]]  # Accommodates both trannsforms like 'DownSample', 'SimpleWindow' as well as 'MotorFault_256IN_16FFTBIN_8FR_3CH_rmDC_1x384x1'
        self.org_sr = org_sr
        self.sequence_window = sequence_window
        self.classes = list()
        self.label_map = dict()
        self.variables = variables
        self.resampling_factor = int(kwargs.get('resampling_factor', 1))
        self.logger = getLogger("root.SimpleTSDataset")
        self.logger.info("Data is being picked up from: {}".format(dataset_dir))
        self.logger.info("Inputs by user: ")
        self.logger.info("Number of Time Series Components/variables/channels: {}".format(self.variables))
        self.logger.info("Original Sample rate: {}Hz".format(self.org_sr))
        self.feature_extraction_params = dict()  # This is just useful for header file as a part of golden vectors

        # reorganize the list of data
        def load_list(filename):
            loadList = []
            # filepath = os.path.join(self._path, filename)
            with open(filename) as fileobj:
                for line in fileobj:  # line denotes each line from the list of data set aside for valiation or testing (e.g., 'right/a69b9b3e_nohash_0.wav' in 'validation_list.txt')
                    line = line.strip()
                    if line:
                        line = os.path.join(self._path, line)
                        loadList.append(line)
            return loadList

        if subset in ["validation", "val"]:
            val = kwargs.get('validation_list') if kwargs.get('validation_list') else \
            glob(os.path.join(self._path, '../annotations', "*val*_list.txt"))[0]
            self.logger.info("Loading validation data from {}".format(val))
            self._walker = load_list(val)
        elif subset in ["testing", "test"]:
            try:
                test_list = glob(os.path.join(self._path, '../annotations', "*test*_list.txt"))[0]
            except IndexError:
                test_list = glob(os.path.join(self._path, '../annotations', "*file*_list.txt"))[0]
            test = kwargs.get('testing_list', test_list)

            self.logger.info("Loading testing data from {}".format(test))
            self._walker = load_list(test)
        elif subset in ["training", "train"]:
            train = kwargs.get('training_list') if kwargs.get('training_list') else \
            glob(os.path.join(self._path, '../annotations', "*train*_list.txt"))[0]
            self.logger.info("Loading training data from {}".format(train))
            self._walker = load_list(train)

        """
        Sanity checks:
        org_sr : Original Sample Rate
        sequence_window: Each sequence duration
        """
        # print(type(self.resampling_factor))
        self.new_sr = self.org_sr / self.resampling_factor  # Resampling_factor=1 by default
        self.stride_window = self.sequence_window

        """
        Transforms:

        """
        if 'Downsample' in self.transforms:
            if 'new_sr' in kwargs.keys() and kwargs.get('new_sr') not in [None, 'None']:
                self.logger.info("resampling_factor ignored as New Sampling rate (new_sr) has been provided")
                self.new_sr = int(kwargs.get("new_sr"))
            self.logger.info("Data to be down/re-sampled to: {}Hz".format(self.new_sr))
            # else:
            #     raise AttributeError("Downasampled Sample rate: 'new_sr' required for 'DownSample' transform")

        if ('SimpleWindow' in self.transforms):
            if ('stride_window' in kwargs.keys()):
                self.stride_window = kwargs.get('stride_window')
            else:
                self.logger.warning("Stride window not provided. It will be same as sequence window, i.e. ZERO OVERLAP")
                # raise AttributeError("Stride(ms): 'stride_window' required for 'SimpleWindow' transform")


    def _prepare_empty_variables(self, **kwargs):
        self.samples_in_sequence = int(self.new_sr * self.sequence_window)
        self.samples_in_stride = int(self.new_sr * self.stride_window)

        self.logger.info("Data samples in one frame: {}".format(self.sequence_window, self.samples_in_sequence))
        self.logger.info("Stride of {} samples implies window is moved by {} samples".format(self.stride_window,
                                                                                             self.samples_in_stride))

        self.X = np.array([]).reshape(0, self.variables, self.samples_in_sequence, )
        self.X_raw = np.array([]).reshape(0, self.variables, self.samples_in_sequence, )

        self.Y = []  # np.array([], dtype=np.uint8).reshape(0, num_classes)

        ''' Process Data '''
        self.classes = sorted(set([opb(opd(datafile)) for datafile in self._walker]))

    def _load_datafile(self, datafile, **kwargs):
        file_extension = ops(datafile)[-1]
        if file_extension == ".npy":
            x_temp = np.load(datafile)
            if len(x_temp.shape) > 1:
                x_temp = x_temp[:, 1:]  # Remove the timestamp column if it exists # [..., np.newaxis]
        elif file_extension == ".pkl":
            x_temp = pd.read_pickle(datafile)
            non_time_columns = [col for col in x_temp.columns if 'time' not in col.lower()]
            x_temp = x_temp[non_time_columns]
            if len(x_temp.columns) > 1:
                x_temp = x_temp.iloc[:, 1:].values  # Remove the timestamp column if it exists
        elif file_extension == ".csv":
            x_temp = pd.read_csv(datafile)
            non_time_columns = [col for col in x_temp.columns if 'time' not in col.lower()]
            x_temp = x_temp[non_time_columns]
            if len(x_temp.columns) - self.variables:  # if len(x_temp.columns) > 1:
                x_temp = x_temp.iloc[:, 1:].values  # Remove the first auto numbered column by pandas
        elif file_extension == ".txt":
            x_temp = np.loadtxt(datafile)
            if len(x_temp.shape) > 1:
                x_temp = x_temp[:, 1:]  # Remove the first auto numbered column by pandas
        else:
            raise Exception("Supports only .npy, .pkl, .txt and .csv file formats for now")
        label = opb(opd(datafile))
        # self.classes.add(label)
        if 'Downsample' in self.transforms:
            # Anyway the new_sr will be org_sr in case downsample is not given as a transform
            x_temp = basic_transforms.Downsample(x_temp, self.org_sr, self.new_sr)
        if 'SimpleWindow' in self.transforms:
            # Sequences are just split as cuts (no windowing) in case stride window isn't given
            x_temp = np.array(basic_transforms.SimpleWindow(x_temp, window_size=self.samples_in_sequence,
                                                            stride=self.samples_in_stride,
                                                            keep_short_tails=kwargs.get('keep_short_tails')))
        x_temp_raw_out = x_temp.copy()  # x_temp_raw_out is used to compare with stored feature extracted data
        return x_temp, label, x_temp_raw_out

    def _rearrange_dims(self, datafile, x_temp, label, x_temp_raw_out, **kwargs):
        y_temp = np.array([label for i in range(x_temp.shape[0])])
        try:
            if x_temp.ndim == 2:
                x_temp = np.expand_dims(x_temp, axis=1)
            elif x_temp.ndim == 3:
                x_temp = x_temp.transpose(0, 2, 1)  # Interchange N,H,C to N,C,H

            if self.X.size == 0:
                self.X = x_temp
                self.X_raw = x_temp_raw_out
            else:
                self.X = np.concatenate((self.X, x_temp))  # n, 1, 752
                self.X_raw = np.concatenate((self.X_raw, x_temp_raw_out))  # n, 1, 752
            # self.Y.extend(y_temp)
            self.Y = np.concatenate((self.Y, y_temp))  # n
            if kwargs.get('store_feat_ext_data'):
                if kwargs.get('feat_ext_store_dir'):
                    x_raw_out_file_path = os.path.join(
                        kwargs.get('feat_ext_store_dir'),
                        os.path.splitext(os.path.basename(datafile))[0] + '__' + 'raw' + '_X.npy')
                    np.save(x_raw_out_file_path, x_temp_raw_out)
                    self.logger.debug(f"Stored raw data in {x_raw_out_file_path}")

                    transforms_chosen = '_'.join(self.transforms)
                    out_file_name = os.path.splitext(os.path.basename(datafile))[0] + '__' + transforms_chosen

                    x_out_file_path = os.path.join(kwargs.get('feat_ext_store_dir'), out_file_name + '_X.npy')
                    np.save(x_out_file_path, x_temp)
                    self.logger.debug(f"Stored intermediate data in {x_out_file_path}")

                    y_out_file_path = os.path.join(kwargs.get('feat_ext_store_dir'), out_file_name + '_Y.npy')
                    np.save(y_out_file_path, y_temp)
                    self.logger.debug(f"Stored intermediate targets in {y_out_file_path}")
                else:
                    self.logger.warning(
                        "'store_feat_ext_data' chosen but 'feat_ext_store_dir' not provided. Skipping storage")

        except ValueError as e:
            self.logger.warning('Skipping {} as Error encountered: {}'.format(datafile, e))

    def _process_targets(self):
        self.label_map = {k: v for v, k in enumerate(self.classes)}  # E.g: {'arc': 0, 'non_arc': 1}
        self.inverse_label_map = {k: v for v, k in self.label_map.items()}  # E.g: {'arc': 0, 'non_arc': 1}
        if self.X.ndim == 3:
            # This was added because all the models during APL require N,C, H(features), W(1)
            self.X = np.expand_dims(self.X, axis=-1)

        ''' Process Targets '''
        # targets are processed separately as we need to develop indices for the same
        # if 'FFT' in self.transforms:
        #     # In FFT case, this is already done
        #     pass
        # else:
        #     self.Y = [self.label_map[i] for i in self.Y]
        self.Y = [self.label_map[i] for i in self.Y]

        # if self.variables == 1:
        #     self.X = np.float32(self.X[:, None, :])  # Adding a dummy dimension
        self.logger.info("Dataset Label Map: {}".format(self.label_map))
        self.logger.info("Data Dimensions (N,C,H,(W?)): {}".format(self.X.shape))
        self.logger.info("Target Length: {}".format(len(self.Y)))
        if not len(self.Y):
            self.logger.error("No data could be loaded. Either file paths were erroneous or dimensions mismatched."
                              " Check prior logger messages/ data processing configurations")

        # self.X = torch.from_numpy(self.X)
        # self.Y = torch.from_numpy(self.Y)

    def prepare(self, **kwargs):
        # Space for Dataset specific initialisations
        self._prepare_empty_variables(**kwargs)
        for datafile in tqdm(self._walker):
            try:
                x_temp, label, x_temp_raw_out = self._load_datafile(datafile, **kwargs)
                # Space for Dataset specific initialisations
                self._rearrange_dims(datafile, x_temp, label, x_temp_raw_out, **kwargs)
            except ValueError as v:
                self.logger.warning(f"File will be skipped due to an error: {datafile} : {v}")
        self._process_targets()
        return self

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # if 'MFCC' in self.transforms:
        #     mfcc_layer = MFCC(sample_rate=sample_rate, n_mfcc=self.mfcc_bins, log_mels=self.log_mels)
        #     # 1, 16000
        #     mel_spectogram = mfcc_layer(waveform)  # 1, 64, 81
        #     mel_spectogram = padding(mel_spectogram, seq_len=2*self.mfcc_bins)  # 1, 64, 128
        #     waveform = mel_spectogram# .unsqueeze(0)
        #
        # if 'STFT' in self.transforms:
        #     stft_layer = STFT(self.filter_length, self.hop_length)
        #     real, imag = stft_layer(waveform)
        #     spectrogram = torch.sqrt(real ** 2 + imag ** 2)  # 1,129,63
        #     waveform = spectrogram#.squeeze(1)
        return torch.from_numpy(self.X[index]), self.Y[index]

class ArcFaultDataset(SimpleTSDataset):
    def __init__(self, subset: str = None, dataset_dir: str = None, transforms: list = [], org_sr=313000,
                 sequence_window=0.25, variables=1, **kwargs):
        super().__init__(subset=subset, dataset_dir=dataset_dir, transforms=transforms, org_sr=org_sr,
                         sequence_window=sequence_window, variables=variables, **kwargs)

        self.frame_size = kwargs.get('frame_size', 1024)

        self.feature_size_per_frame = kwargs.get('feature_size_per_frame', 256)
        self.num_frame_concat = kwargs.get('num_frame_concat', 1)  # feature concatenation
        # self.feature_size_per_frame = round(self.feature_size / self.num_frame_concat)
        self.feature_size = self.feature_size_per_frame * self.num_frame_concat

        self.frame_skip = kwargs.get('frame_skip', 1)  # skipped frames during feature concatenation
        self.min_fft_bin = kwargs.get('min_fft_bin', 1)  # 1: remove dc 0: keep dc
        self.fft_bin_size = kwargs.get('fft_bin_size', 2)

        self.sequence_window = self.feature_size

        self.feature_extraction_params['frame_size'] = self.frame_size
        self.feature_extraction_params['feature_size_per_frame'] = self.feature_size_per_frame
        self.feature_extraction_params['num_frame_concat'] = self.num_frame_concat
        self.feature_extraction_params['feature_size'] = self.feature_size
        self.feature_extraction_params['frame_skip'] = self.frame_skip
        self.feature_extraction_params['min_fft_bin'] = self.min_fft_bin
        self.feature_extraction_params['fft_bin_size'] = self.fft_bin_size

    def __feature_extraction(self, x_temp):
        num_frame = len(x_temp) // self.frame_size
        feature_framecase = []
        raw_framecase = []
        feature_frame = []
        raw_frame = []
        for index_frame in range(0, num_frame):
            # Get one frame and run FFT
            wave_frame = x_temp.squeeze()[index_frame * self.frame_size:(index_frame + 1) * self.frame_size]  # Take one batch of (self.frame_size,) -> (1024,)
            raw_frame.append(wave_frame)
            wave_frame_win = wave_frame * np.hanning(self.frame_size)
            fft_out = np.fft.fft(wave_frame_win) / self.frame_size  # compute fft and normalize.  (1024,)
            mag = abs(fft_out[0:(int(self.frame_size / 2) + self.min_fft_bin)])  # (513,)
            fft_bin_out = np.zeros((self.feature_size_per_frame,))
            for index_feature in range(0, self.feature_size_per_frame):
                for index_bin in range(0, self.fft_bin_size):
                    fft_bin_out[index_feature] = fft_bin_out[index_feature] + mag[
                        self.min_fft_bin + index_feature * self.fft_bin_size + index_bin]
            magdB = 10 * np.log(1e-12 + fft_bin_out)
            # feature_frame contains all feature in this file
            feature_frame.append(
                magdB)  # [[ ]] -> 'num_frames' number of lists of 'self.feature_size_per_frame' elements each
            # concatenate multiple previous frames if needed
            # store feature2 with proper concatenation into feature_framecase
            if (index_frame >= self.num_frame_concat - 1) and (index_frame % self.frame_skip == 0):
                temp = np.array(
                    [x for item in feature_frame[index_frame - self.num_frame_concat + 1:index_frame + 1] for x in
                     item])  # self.num_frame_concat*(self.feature_size_per_frame,)
                feature_framecase.append(temp)

                # temp = np.array([x for item in raw_frame[index_frame - self.num_frame_concat + 1:index_frame + 1] for x in item])
                temp = np.array(np.array(raw_frame[index_frame - self.num_frame_concat + 1:index_frame + 1][-1]))
                raw_framecase.append(temp)
        x_temp = np.array(feature_framecase)
        x_temp_raw_out = np.array(raw_framecase)
        return x_temp, x_temp_raw_out

    def prepare(self, **kwargs):
        # Space for Dataset specific initialisations
        self._prepare_empty_variables(**kwargs)
        for datafile in tqdm(self._walker):
            try:
                x_temp, label, x_temp_raw_out = self._load_datafile(datafile, **kwargs)
                # Space for Dataset specific feature_extractions
                x_temp, x_temp_raw_out = self.__feature_extraction(x_temp)
                self._rearrange_dims(datafile, x_temp, label, x_temp_raw_out, **kwargs)
            except ValueError as v:
                self.logger.warning(f"File will be skipped due to an error: {datafile} : {v}")

        # ArcFault_base1 requires an additional dimension N,C, H(features), W(1)
        self.X = np.expand_dims(self.X, axis=-1)
        self._process_targets()
        return self


class MotorFaultDataset(SimpleTSDataset):
    def __init__(self, subset: str = None, dataset_dir: str = None, transforms: list = [], org_sr=313000,
                 sequence_window=0.25, variables=1, **kwargs):
        super().__init__(subset=subset, dataset_dir=dataset_dir, transforms=transforms, org_sr=org_sr,
                         sequence_window=sequence_window, variables=variables, **kwargs)

        self.frame_size = kwargs.get('frame_size', 256)
        self.feature_size_per_frame = kwargs.get('feature_size_per_frame', 16)
        self.num_frame_concat = kwargs.get('num_frame_concat', 8)
        # self.num_channel = kwargs.get('num_channel', 3)
        self.dc_remove = kwargs.get('dc_remove', True)
        self.stacking = kwargs.get('stacking', '1D')

        if self.stacking == '1D':
            self.ch = 1
            self.wl = self.feature_size_per_frame * self.variables * self.num_frame_concat
            self.hl = 1
        elif self.stacking == '2D1':
            self.ch = self.variables
            self.wl = self.feature_size_per_frame * self.num_frame_concat
            self.hl = 1
        self.offset = kwargs.get('offset', 0)
        self.scale = kwargs.get('scale', 1)
        self.fs = kwargs.get('fs', 1)  # not used

        self.feature_extraction_params['frame_size'] = self.frame_size
        self.feature_extraction_params['feature_size_per_frame'] = self.feature_size_per_frame
        self.feature_extraction_params['num_frame_concat'] = self.num_frame_concat
        self.feature_extraction_params['dc_remove'] = int(self.dc_remove)
        self.feature_extraction_params['ch'] = self.ch
        self.feature_extraction_params['wl'] = self.wl
        self.feature_extraction_params['hl'] = self.hl
        self.feature_extraction_params['offset'] = self.offset
        self.feature_extraction_params['scale'] = self.scale
        self.feature_extraction_params['fs'] = self.fs

        ## Computes one side FFT ##
    def __fft_oneside(self, y, fs):
        n = len(y)  # Length of signal
        k = np.arange(n)
        T = n / fs
        frq = k / T  # two sides frequency range, center at Fs/2
        # Y = np.fft.fft(y)/n  # compute fft and normalize.
        Y = np.fft.fft(y)  # compute fft; no normalize; this is to match C2000 RFFT_f32_mag_TMU0() implementation
        frq = frq[0:(int(n / 2) + 1)]  # returns DC + n/2 samples; for visualization only
        Y = abs(Y[0:(int(n / 2) + 1)])
        return frq, Y

    ## Computes FFT and binning (mimic c implementation) ##
    def __fft_bin(self, s, fs, nbins, no_dc):
        f, fft1 = self.__fft_oneside(s, fs)
        n_fft = len(fft1)
        if (no_dc == True):
            fft1 = fft1[1:]  # skip the first DC bin
            f = f[1:]
            nSamples = (n_fft - 1) // nbins
        else:
            nSamples = n_fft // nbins
        fbins = []
        mbins = []
        for i in range(0, nbins):
            tmp = 0
            i1 = int(i * nSamples)
            i2 = i1 + nSamples
            for m in range(i1, i2):
                tmp = tmp + fft1[m]
            mbins.append(tmp / nSamples)
            fbins.append(f[i2 - 1])  # for visualization only; not needed in training
        return fbins, mbins

    def __mf_getLabel(self, fname):
        p = fname.split('/')[-1].split('_')
        q = [int(a.split('label')[1]) for a in p if 'label' in a]
        return q[0]

    def __feature_extraction(self, x_temp):
        # Calculate features per axis
        vax = [[] for a in range(self.variables)]
        vax_copy = [[] for a in range(self.variables)]
        for ax in range(self.variables):
            # The below code is for each axis.
            vsel = x_temp.iloc[:, ax] / self.scale
            vsel = np.floor(vsel).astype(int)  # convert data to fixpoint

            # Calculate number of frames with or w/o sample overlap
            if (self.offset != 0):
                Nos_steps = len(vsel) // self.offset  # Number of segments with overlap
                Nlast = len(vsel[int((
                                                 Nos_steps - 1) * self.offset):]) // self.frame_size  # Make sure the last segment has at least Ns samples
                if (Nlast == 0):
                    Nos_steps -= 1  # skip last one
            else:
                Nos_steps = 1  # non-overlap mode; start with sample at 0 index

            # Prepare features per frame
            vl = []
            vl_copy = []  # To store adc values
            vl_mf = []
            vl_mf_copy = []  # To store adc values
            for n in range(0, Nos_steps):
                Nsteps = len(vsel[int(n * self.offset):]) // self.frame_size
                for m in range(0, Nsteps):
                    m1 = int(m * self.frame_size + n * self.offset)
                    m2 = m1 + int(self.frame_size)
                    vs = vsel[m1:m2]  # segmented data
                    vs_copy = vs.copy()
                    # Select pre-processing methods
                    if ('RAW' in self.transforms):
                        if (self.dc_remove == True):
                            vs = vs - np.sum(vs) // len(vs)  # -np.mean(vs)
                        vs_m = list(vs)
                    elif ('FFTBIN' in self.transforms):
                        vs_f, vs_m = self.__fft_bin(vs, self.fs, self.feature_size_per_frame, self.dc_remove)
                        vs_m = 20 * np.log10(vs_m)  # dB
                        vs_m = list(vs_m)
                    else:
                        raise 'Unsupported transform for MotorFault!!'
                    # end method
                    vl.append(vs_m)  # save result
                    vl_copy.append(
                        vs_copy.to_list())  # In case of RAW, it will have the same dimensions as vs_m. In case of FFTBIN, it will not have the same dimensions as vs_m

                if (self.num_frame_concat > 1):  # concatenate multiple frames
                    vl_mf = [sum(vl[o:o + self.num_frame_concat], []) for o in
                             range(Nsteps - self.num_frame_concat + 1)]
                    # vl_mf_copy = [sum(vl_copy[o:o + self.num_frame_concat], []) for o in range(Nsteps - self.num_frame_concat + 1)]
                    # Use the above line instead of the below line if all frames are required instead of just the last frame
                    vl_mf_copy = [vl_copy[o + self.num_frame_concat - 1] for o in
                                  range(Nsteps - self.num_frame_concat + 1)]

            vax[ax] = vl_mf if (
                        self.num_frame_concat > 1) else vl  # save result per axis; select from single and multi-frame
            vax_copy[ax] = vl_mf_copy if (
                        self.num_frame_concat > 1) else vl_copy  # save result per axis; select from single and multi-frame
            # vax_copy[ax] = vl_copy  # TODO: Enable above line if more than first frame raw data is required
        # Complete feature pre-processing

        # Pack data in 1d format: samples x features; features; for example, xaxis[0:128]-yaxis[128:256]-zaxis[256:384]
        # This entire below loop just converts an array of length (n,3) to (3n,1) or (n,3) based on self.variables
        # TODO: Can replace the confusing loop with a single line using numpy arrays instead of lists
        dd = []
        for a in range(len(vax[0])):  # loop through each measurement samples
            cc = []
            for c in range(self.variables):  # concatenate x+y+z data
                cc = cc + vax[c][a]
            dd.append(cc)
        # vs_data[icond] = vs_data[icond] + dd
        x_temp = np.array(dd)  # format: samples x features
        # Reshape data to ch,wl,hl per measurement; final format: N,C,W,H; For example, 1500,3,128,1; dataloader random pick from N
        N = x_temp.shape[0]
        data_packed = np.zeros(
            (N, self.ch, self.wl, self.hl))  # sample x channel x feature_length x height_length
        for n in range(0, N):
            for i in range(0, self.ch):
                nc = x_temp[n].shape[0] // self.ch  # length per channel
                o1 = i * nc
                o2 = o1 + nc
                tmp2 = x_temp[n][o1:o2]
                nh = nc // self.hl  # length per height (break sequential data into multi-segments; 2d)
                for m in range(0, self.hl):
                    s1 = m * nh
                    s2 = s1 + nh
                    data_packed[n, i, :, m] = tmp2[s1:s2]
        x_temp = data_packed
        x_temp_raw_out = np.array(vax_copy).transpose(1, 0, 2)  # Convert 3,n,256 to n,3,256 (256 is frame size)

        return x_temp, x_temp_raw_out

    def prepare(self, **kwargs):
        # Space for Dataset specific initialisations
        self._prepare_empty_variables(**kwargs)
        for datafile in tqdm(self._walker):
            try:
                x_temp, label, x_temp_raw_out = self._load_datafile(datafile, **kwargs)
                # Space for Dataset specific feature_extractions
                x_temp, x_temp_raw_out = self.__feature_extraction(x_temp)
                self._rearrange_dims(datafile, x_temp, label, x_temp_raw_out, **kwargs)
            except ValueError as v:
                self.logger.warning(f"File will be skipped due to an error: {datafile} : {v}")
        self._process_targets()
        return self
