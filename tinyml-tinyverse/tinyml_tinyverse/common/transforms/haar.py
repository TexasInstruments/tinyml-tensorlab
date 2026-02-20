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

import copy
import numpy as np


def _haar_forward_stage(x):
    length = len(x)
    length_half = length//2
    l = (x[0::2] + x[1::2])/2
    h = (x[0::2] - x[1::2])/2
    x[:length_half] = l
    x[length_half:] = h


def haar_forward(x, full_decomposition=False):
    x = np.array(x, dtype=np.float32)
    length = len(x)
    log2_len = np.log2(length)
    assert log2_len == int(log2_len), f"input window size can only be power of 2, given size was {length}"

    if length <= 1:
        return

    stages = int(np.log2(length))
    transform_length = length
    num_tr_in_stage = 1

    for stage in range(stages):
        for step in range(num_tr_in_stage):
            start_offset = (step * transform_length)
            _haar_forward_stage(x[start_offset:start_offset+transform_length])

        transform_length = transform_length//2
        if full_decomposition:
            num_tr_in_stage = num_tr_in_stage*2
    return x


def _haar_inverse_stage(x):
    length = len(x)
    length_half = length//2
    l = x[0:length_half]
    h = x[length_half:]
    x_first = (l + h)
    x_second = (l - h)
    x[0::2] = x_first
    x[1::2] = x_second


def haar_inverse(x, full_decomposition=False):
    length = len(x)
    log2_len = np.log2(length)
    assert log2_len == int(log2_len), "input window size can only be power of 2"

    if length <= 1:
        return

    stages = int(np.log2(length))
    transform_length = 2
    num_tr_in_stage = length//2

    for stage in range(stages):
        for step in range(num_tr_in_stage):
            start_offset = (step * transform_length)
            _haar_inverse_stage(x[start_offset:start_offset+transform_length])

        transform_length = transform_length*2
        if full_decomposition:
            num_tr_in_stage = num_tr_in_stage//2

    return x


if __name__ == "__main__":
    x = np.array(np.arange(8), dtype=np.float32)
    y = haar_forward(copy.deepcopy(x), full_decomposition=True)
    x_hat = haar_inverse(copy.deepcopy(y), full_decomposition=True)
    diff = np.max(np.abs(x_hat - x))
    print(f"Input array: {x}")
    print(f"Haar Transformed array: {y}")
    print(f"Reconstructed array: {x_hat}")
    print(f"Max reconstruction error is: {diff}")