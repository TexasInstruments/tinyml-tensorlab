#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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

# Imports Torch
import torch
import torch.ao.quantization


#################################################################################
def _ceil2_func(x):
    y = torch.pow(2,torch.ceil(torch.log2(x)))
    return y

def _ceil2_tensor(x):
    y = x
    if x.data.abs().sum() != 0:
        x2 = _ceil2_func(torch.abs(x))
        y = torch.sign(x) * x2
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

def _adjust_qparams_power2_scale(min_val, max_val, quant_min, quant_max, scale, zero_point, eps):
    r"""Calculates the quantization parameters."""
    # make scale a power of 2 value
    scale = ceil2_tensor(scale)
    scale = torch.max(scale, eps)
    if len(torch.unique(zero_point))>1 or torch.unique(zero_point) not in (0,127):
        # adjust the zero_point based on new scale
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
    return scale, zero_point


class SimplePerChannelWeightObserver(torch.ao.quantization.PerChannelMinMaxObserver):
    def __init__(self, *args, quant_min=-128, quant_max=+127, qscheme=torch.per_channel_symmetric, power2_scale=False, range_max=None, fixed_range=False, **kwargs):
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, qscheme=qscheme, **kwargs)
        self.power2_scale = power2_scale
        self.range_max = range_max
        self.fixed_range = fixed_range

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        # weights qparams are always symmetric and this is ensured inside the super class, no need to handle it here.
        scale, zero_point = super()._calculate_qparams(min_val, max_val)
        if self.power2_scale:
            scale, zero_point = _adjust_qparams_power2_scale(
                min_val, max_val, self.quant_min, self.quant_max, scale, zero_point, self.eps)
        return scale, zero_point

    def forward(self, x_orig):
        x_orig = super().forward(x_orig)
        if self.range_max is not None:
            signed_range = torch.min(self.min_val.detach()).item() < 0.0
            min_val = (-self.range_max) if signed_range else 0.0
            max_val = (+self.range_max) if signed_range else (+self.range_max)
            if self.fixed_range:
                self.min_val.fill_(min_val)
                self.max_val.fill_(max_val)
            else:
                self.min_val = torch.clamp(self.min_val, min=min_val, max=0.0)
                self.max_val = torch.clamp(self.max_val, min=0.0, max=max_val)
            #
        #
        return x_orig


class SimpleActivationObserver(torch.ao.quantization.MovingAverageMinMaxObserver):
    def __init__(self, *args, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, power2_scale=False, range_max=None, fixed_range=False, **kwargs):
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, qscheme=qscheme, **kwargs)
		# activation quantization cannot use torch.per_channel_symmetric, it has to be torch.per_tensor_symmetric
        self.symmetric = (qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric))
        self.power2_scale = power2_scale
        self.range_max = range_max
        self.fixed_range = fixed_range
        self.freeze_observer = False

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        if self.symmetric:
            signed_range = torch.min(min_val.detach()).item() < 0.0
            max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
            min_val = -max_abs if signed_range else max_abs * 0.0
            max_val = max_abs

        scale, zero_point = super()._calculate_qparams(min_val, max_val)

        if self.power2_scale:
            scale, zero_point = _adjust_qparams_power2_scale(
                min_val, max_val, self.quant_min, self.quant_max, scale, zero_point, self.eps)
        return scale, zero_point

    def forward(self, x_orig):
        x_orig = super().forward(x_orig)
        if self.range_max is not None:
            signed_range = torch.min(self.min_val.detach()).item() < 0.0
            min_val = (-self.range_max) if signed_range else 0.0
            max_val = (+self.range_max) if signed_range else (+self.range_max)
            if self.fixed_range:
                self.min_val.fill_(min_val)
                self.max_val.fill_(max_val)
            else:
                self.min_val = torch.clamp(self.min_val, min=min_val, max=0.0)
                self.max_val = torch.clamp(self.max_val, min=0.0, max=max_val)
            #
        #
        return x_orig

