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

import warnings
import torch
import torch.ao.quantization

from . import functional_utils
from .observer_utils import MovingAverageRangeShrinkFastHistogramObserver, RangeShrinkFastHistogramObserver


def get_weight_observer_type(base_class=None):
    if base_class is None:
        warnings.warn("please pass appropriate base class for the weight observer")
        base_class = torch.ao.quantization.PerChannelMinMaxObserver
    #
    class SimplePerChannelWeightObserver(base_class):
        def __init__(self, *args, quant_min=-128, quant_max=+127, qscheme=torch.per_channel_symmetric, power2_scale=False, range_max=None, fixed_range=False, **kwargs):
            super().__init__(*args, quant_min=quant_min, quant_max=quant_max, qscheme=qscheme, **kwargs)
            self.power2_scale = power2_scale
            self.range_max = range_max
            self.fixed_range = fixed_range
            self.symmetric = (qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric))

        @torch.jit.export
        def _calculate_qparams(self, min_val, max_val):
            r"""Calculates the quantization parameters."""
            # weights qparams are always symmetric and this is ensured inside the super class, no need to handle it here.
            scale, zero_point = super()._calculate_qparams(min_val, max_val)
            if self.power2_scale:
                scale, zero_point = functional_utils._adjust_qparams_power2_scale(self.symmetric,
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
    return SimplePerChannelWeightObserver


def get_activation_observer_type(base_class):
    if base_class is None:
        warnings.warn("please pass appropriate base class for the activation observer")
        base_class = torch.ao.quantization.MovingAverageMinMaxObserver
    #
    class SimpleActivationObserver(base_class):
        def __init__(self, *args, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, power2_scale=False,
                range_max=None, fixed_range=False, bias_calibration_factor=0.0, **kwargs):
            super().__init__(*args, quant_min=quant_min, quant_max=quant_max, qscheme=qscheme, **kwargs)
            # activation quantization cannot use torch.per_channel_symmetric, it has to be torch.per_tensor_symmetric
            self.symmetric = (qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric))
            self.power2_scale = power2_scale
            self.range_max = range_max
            self.fixed_range = fixed_range
            self.freeze_observer = False
            self.bias_calibration_factor = bias_calibration_factor

        @torch.jit.export
        def _calculate_qparams(self, min_val, max_val):
            r"""Calculates the quantization parameters."""
            if self.symmetric:
                unsigned_range = torch.min(min_val.detach()).item() >= 0.0
                max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
                min_val = (max_abs * 0.0) if unsigned_range else (-max_abs)
                max_val = max_abs
                if unsigned_range:
                    # in unsigned case, we can use a better scale than what pytorch uses (use the full range)
                    # backup qscheme and set it to torch.per_tensor_affine,
                    # so that the whole unsigned range will be used for scale computation
                    # this is a hack to reuse super()._calculate_qparams() for this case
                    qscheme_backup = self.qscheme
                    self.qscheme = torch.per_tensor_affine
                #
                scale, zero_point = super()._calculate_qparams(min_val, max_val)
                if unsigned_range:
                    # restore qscheme
                    self.qscheme = qscheme_backup
                #
            else:
                scale, zero_point = super()._calculate_qparams(min_val, max_val)

            if self.power2_scale:
                scale, zero_point = functional_utils._adjust_qparams_power2_scale(self.symmetric,
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

    return SimpleActivationObserver