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
    def __init__(self, weight_bitwidth: int=8, activation_bitwidth: int=8, auto_quantization: bool=True,
                 inputs=None, targets=None, criterion=None,
                 calibration_dataloader=None, eval_dataloader=None,
                 task_type: str=None, float_metric: float=None, example_inputs=None,
                 weight_mixed_precision=None, activation_mixed_precision=None, **kwargs):

        self.logger = getLogger("root.main.TinyMLQConfigType")

        if weight_bitwidth is None:
            weight_bitwidth = 8
            self.logger.info("Default the weight bitwidth to 8")
        if activation_bitwidth is None:
            activation_bitwidth = 8
            self.logger.info("Default the activation bitwidth to 8")

        if weight_bitwidth not in [2, 4, 8] or activation_bitwidth not in [2, 4, 8]:
            raise ValueError("Weight Bitwidth supported {2, 4, 8} and Activation Bitwidth supported {2, 4, 8}")

        if auto_quantization:
            self.logger.info("Quantization Bitwidths: Auto quantization")
        else:
            self.logger.info(f"Quantization Bitwidths: Weight-{weight_bitwidth} Activation-{activation_bitwidth}")
            if weight_mixed_precision:
                self.logger.info(f"Mixed Precision for Weight enabled {weight_mixed_precision}")
            if activation_mixed_precision:
                self.logger.info(f"Mixed Precision for Activation enabled {activation_mixed_precision}")

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

        self.qconfig_type = {
            'weight': {
                'bitwidth': weight_bitwidth,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': True if weight_bitwidth == 8 else False,
                'mixed_precision': weight_mixed_precision,
                'range_max': None,
                'fixed_range': False,
                'soft_quant': 'soft_sigmoid' if weight_bitwidth == 4 else 'dbq' if weight_bitwidth == 2 else 'default'
            },
            'activation': {
                'bitwidth': activation_bitwidth,
                'qscheme': torch.per_tensor_symmetric,
                'power2_scale': True if weight_bitwidth == 8 else False,
                'mixed_precision': activation_mixed_precision,
                'range_max': None,
                'fixed_range': False,
                'soft_quant': 'soft_sigmoid' if activation_bitwidth == 4 else 'dbq' if activation_bitwidth == 2 else 'default'
            },
            'auto_quantization' : auto_quantization,
        }
        if self.qconfig_type is not None and auto_quantization:
            required_params = {
                'inputs': inputs,
                'targets': targets,
                'criterion': criterion,
                'calibration_dataloader': calibration_dataloader,
                'eval_dataloader': eval_dataloader,
                'task_type': task_type,
                'float_metric': float_metric,
                'example_inputs': example_inputs,
            }

            # missing_params = [name for name, value in required_params.items() if value is None]
            # if missing_params:
            #     raise ValueError(
            #         f"Auto Quantization enabled but missing required parameters: {', '.join(missing_params)}. "
            #         f"Either set auto_quantization to False or provide all required parameters."
            #     )

            for name, value in required_params.items():
                self.qconfig_type[name] = value

            for key in (
                'autoquant_tolerance_classification',
                'autoquant_tolerance_regression',
                'autoquant_tolerance_forecasting',
                'autoquant_tolerance_anomaly',
            ):
                if kwargs.get(key) is not None:
                    self.qconfig_type[key] = kwargs[key]

