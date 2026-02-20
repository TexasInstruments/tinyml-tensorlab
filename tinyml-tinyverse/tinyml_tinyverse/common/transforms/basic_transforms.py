#################################################################################
# Copyright (c) 2023-2026, Texas Instruments
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
import numpy as np


def SimpleWindow(x, window_size=100, stride=50, keep_short_tails=False):
    # if keep_short_tails is set to True, the slices shorter than window_size at the end of the result will be kept
    # length = x.size(0)
    length, channels = x.shape
    range_end = length
    if not keep_short_tails:
        range_end = length - window_size + 1
    splits = np.zeros((channels, (range_end+stride-1)//stride, window_size))
    x = x.T
    
    for ax in range(channels):
        idx = 0
        for slice_start in range(0, range_end, stride):
            slice_end = min(length, slice_start + window_size)
            length_of_window = slice_end-slice_start
            splits[ax][idx][:length_of_window] = x[ax][slice_start:slice_end]
            idx += 1
    x = splits.transpose(1,2,0)
    return x


def Downsample(x, sampling_rate, new_sr):
    '''
    if x.ndim == 1:
        return x[::int(sampling_rate // new_sr)] # Downsample the 1d array before windowing
    else:
        return x[:, ::int(sampling_rate // new_sr)]  # Downsample along the time dimension after windowing
    '''
    channels = x.shape[1]
    x = x.T
    x_scratch = []
    for ax in range(channels):
        x_scratch.append(x[ax][::int(sampling_rate//new_sr)])
    x = np.array(x_scratch).T
    return x # DownSampling is being done only before windowing so for now it doesn't matter

class Jittering(object):
    def __init__(self, sigma):
        assert isinstance(sigma, (float, tuple))
        self.sigma = sigma

    def __call__(self, sample):
        #print(sample)
        data, label = sample['data'], sample['label']
        if isinstance(self.sigma, float):
            myNoise = np.random.normal(loc=0, scale=self.sigma, size=data.shape)
            data = data+myNoise
        return {'data': data, 'label': label}