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
import copy


def hadamard_forward(x):
    result = copy.deepcopy(x)
    N = len(x)
    step = 1
    while step < N:  # Scale
        i = 0
        while i < N:  # Entwine
            for j in range(i, i + step):  # Box
                a = (result[j] + result[j + step])
                b = (result[j] - result[j + step])
                result[j] = a
                result[j + step] = b
            i += step * 2
        step *= 2
    result = [value / N for value in result]
    return result


def hadamard_forward_vectorized(x):
    result = np.array(x, dtype=np.float32)
    N = len(result)
    step = 1
    while step < N:
        for i in range(0, N, step * 2):
            result[i:i + step], result[i + step:i + 2 * step] = (result[i:i + step] + result[i + step:i + 2 * step]), (
                        result[i:i + step] - result[i + step:i + 2 * step])
        step *= 2
    result /= N
    return result


def hadamard_inverse_vectorized(result):
    x = np.array(result, dtype=np.float32)
    N = len(x)
    step = 1
    while step < N:
        for i in range(0, N, step * 2):
            x[i:i + step], x[i + step:i + 2 * step] = (x[i:i + step] + x[i + step:i + 2 * step]), (x[i:i + step] - x[i + step:i + 2 * step])
        step *= 2
    return x


def hadamard_inverse(result):
    x = copy.deepcopy(result)
    N = len(x)
    step = 1
    while step < N:  # Scale
        i = 0
        while i < N:  # Entwine
            lower_box_bound = i
            upper_box_bound = i + step
            for j in range(lower_box_bound, upper_box_bound):  # Box
                a = (x[j] + x[j + step])
                b = (x[j] - x[j + step])
                x[j] = a
                x[j + step] = b
            i += step * 2
        step *= 2
    return x


if __name__ == "__main__":
    input_array = np.array(np.arange(8), dtype=np.float32)
    from time import time

    # print(f"Input array: {input_array}")

    t1 = time()
    y = hadamard_forward(copy.deepcopy(input_array))
    x_hat = hadamard_inverse(copy.deepcopy(y))
    diff = np.max(np.abs(x_hat - input_array))
    normal = time() - t1
    print(normal)
    print(f"Hadamard Transformed array: {y}")
    print(f"Reconstructed array: {x_hat}")
    print(f"Max reconstruction error is: {diff}")

    t1 = time()
    y = hadamard_forward_vectorized(copy.deepcopy(input_array))
    x_hat = hadamard_inverse_vectorized(copy.deepcopy(y))
    diff = np.max(np.abs(x_hat - input_array))
    vectorized = time() - t1
    print(vectorized)
    print(f"Hadamard Transformed array: {y}")
    print(f"Reconstructed array: {x_hat}")
    print(f"Max reconstruction error is: {diff}")

    print(f"Speedup: {normal / vectorized}")
