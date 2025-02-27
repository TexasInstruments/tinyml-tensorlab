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
import torch
from logging import getLogger


class TinyMLQuantizationVersion():
    NO_QUANTIZATION = 0
    QUANTIZATION_GENERIC = 1
    QUANTIZATION_TINPU = 2

    @classmethod
    def get_dict(cls):
        return {k:v for k,v in cls.__dict__.items() if not k.startswith("__")}

    @classmethod
    def get_choices(cls):
        return {v:k for k,v in cls.__dict__.items() if not k.startswith("__")}


class TinyMLModelQConfigFormat:
    FLOAT_MODEL = "FLOAT_MODEL"    # original float model format
    FAKEQ_MODEL = "FAKEQ_MODEL"    # trained FakeQ model before conversion
    QDQ_MODEL = "QDQ_MODEL"        # converted QDQ model
    INT_MODEL = "INT_MODEL"        # integer model
    TINPU_INT_MODEL = "TINPU_INT_MODEL"
    _NUM_FORMATS_ = 5

    @classmethod
    def choices(cls):
        return [value for value in dir(cls) if not value.startswith('__') and value != 'choices']


class TinyMLQuantizationMethod():
    PTQ = 'PTQ'
    QAT = 'QAT'

    @classmethod
    def get_dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith("__")}

    @classmethod
    def get_choices(cls):
        return {v: k for k, v in cls.__dict__.items() if not k.startswith("__")}


class TinyMLQConfigType:
    def __init__(self, weight_bitwidth: int=8, activation_bitwidth: int=8):
        self.logger = getLogger("root.main.TinyMLQConfigType")
        self.logger.info(f"Quantization Bitwidths: Weight-{weight_bitwidth} Activation- {activation_bitwidth}")
        self.qconfig_type = None
        if weight_bitwidth is None or activation_bitwidth is None:
            '''
            # 8bit weight / activation is default - no need to specify inside.
            qconfig_type = {
                'weight': {
                    'bitwidth': 8,
                    'qscheme': torch.per_channel_symmetric,
                    'power2_scale': True,
                    'range_max': None,
                    'fixed_range': False
                },
                'activation': {
                    'bitwidth': 8,
                    'qscheme': torch.per_tensor_symmetric,
                    'power2_scale': True,
                    'range_max': None,
                    'fixed_range': False
                }
            }
            '''
            self.qconfig_type = None
        elif weight_bitwidth == 8:
            self.qconfig_type = {
                'weight': {
                    'bitwidth': weight_bitwidth,
                    'qscheme': torch.per_channel_symmetric,
                    'power2_scale': True,
                    'range_max': None,
                    'fixed_range': False
                },
                'activation': {
                    'bitwidth': activation_bitwidth,
                    'qscheme': torch.per_tensor_symmetric,
                    'power2_scale': True,
                    'range_max': None,
                    'fixed_range': False
                }
            }
        elif weight_bitwidth == 4:
            self.qconfig_type = {
                'weight': {
                    'bitwidth': weight_bitwidth,
                    'qscheme': torch.per_channel_symmetric,
                    'power2_scale': False,
                    'range_max': None,
                    'fixed_range': False
                },
                'activation': {
                    'bitwidth': activation_bitwidth,
                    'qscheme': torch.per_tensor_symmetric,
                    'power2_scale': False,
                    'range_max': None,
                    'fixed_range': False
                }
            }
        elif weight_bitwidth == 2:
            self.qconfig_type = {
                'weight': {
                    'bitwidth': weight_bitwidth,
                    'qscheme': torch.per_channel_symmetric,
                    'power2_scale': False,
                    'range_max': None,
                    'fixed_range': False,
                    'quant_min': -1,
                    'quant_max': 1,
                },
                'activation': {
                    'bitwidth': activation_bitwidth,
                    'qscheme': torch.per_tensor_symmetric,
                    'power2_scale': False,
                    'range_max': None,
                    'fixed_range': False
                }
            }
        else:
            raise RuntimeError("unsupported quantization parameters")

