#################################################################################
# Copyright (c) 2018-2025, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

import torch


def _ceil2_func(x):
    y = torch.pow(2,torch.ceil(torch.log2(x)))
    return y

def _ceil2_tensor(x):
    y = x
    if x.data.abs().sum() != 0:
        x2 = _ceil2_func(torch.abs(x))
        y = torch.sign(x) * x2
    return y

def _round_tensor(x):
    y = torch.round(x)
    return y

def _propagate_quant_ste(x, y):
    # straight-through estimation (STE) - this is the preferred mode for propagating outputs for quantization
    # the backward gradients uses x (i.e. the input itself) and not y (the output)
    # because the effect of y is completely detached during forward
    # this works functionally as STE, but exports an onnx graph containing
    # all the operators used to compute y as well
    # out = x + (y - x).detach()
    #
    # this is another way of doing STE. in this case the operators used to generate y are skipped from onnx graph
    out = x.clone()
    out.data = y.data
    return out

def ceil2_tensor(x):
    y = _ceil2_tensor(x)
    return _propagate_quant_ste(x, y)

def round_tensor(x):
    y = _round_tensor(x)
    return _propagate_quant_ste(x, y)

def _adjust_qparams_power2_scale(symmetric, min_val, max_val, quant_min, quant_max, scale, zero_point, eps):
    r"""Calculates the quantization parameters."""
    # make scale a power of 2 value
    scale = ceil2_tensor(scale)
    scale = torch.max(scale, eps)
    if not symmetric:
        # adjust the zero_point based on new scale
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
    return scale, zero_point