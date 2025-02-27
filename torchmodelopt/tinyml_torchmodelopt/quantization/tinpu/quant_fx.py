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

import warnings
import torch
import torch.ao.quantization
from torch.fx import GraphModule

import platform
from typing import List, Tuple


from ..common import *
from ..base.fx import TinyMLQuantFxBaseModule

from .quant_utils import TINPUQuantizedReplacementUtils
from ... import surgery


class TINPUTinyMLQuantFxModule(TinyMLQuantFxBaseModule):
    def __init__(self, *args, qconfig_type=None, **kwargs) -> None:
        '''
        The QAT wrapper module does the preparation like in:
        qat_model = quantize_fx.prepare_qat_fx(nn_model, qconfig_mapping, example_input)
        It also uses an appropriate qconfig that imposes the constraints of the hardware.

        The api being called doesn't actually pass qconfig_type - so it will be defined inside.
        But if you need to pass, it can be defined this way.
        # qconfig_type supported for TINPU in F28 devices
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
        self.weight_bw = qconfig_type['weight']['bitwidth'] if qconfig_type else 8
        self.activation_bw = qconfig_type['activation']['bitwidth'] if qconfig_type else 8
        self.power2_scale = qconfig_type['weight']['power2_scale'] if qconfig_type else True

        if self.weight_bw >= 8:
            assert self.power2_scale is True, 'for 8bit quantization, power2_scale must be set to True'
        else:
            if self.power2_scale:
                warnings.warn('for bit_widths < 8, it power2_scale=False is supported and can be used for better accuracy.')
            #
        #

        backend = 'fbgemm' if platform.system() in ['Windows'] else 'qnnpack'
        super().__init__(*args, qconfig_type=qconfig_type, backend=backend, **kwargs)

    def convert(self, *args, model_qconfig_format=TinyMLModelQConfigFormat.TINPU_INT_MODEL, output_dequantize=False, **kwargs):
        # first convert the model to int
        super().convert(*args, model_qconfig_format=model_qconfig_format, **kwargs)
        _convert_replacement_func = lambda module, pattern, *largs, **lkwargs: self._convert_replacement(module, pattern, *largs, output_dequantize=output_dequantize, **lkwargs)
        # then apply the transformation to required output format
        if model_qconfig_format == TinyMLModelQConfigFormat.TINPU_INT_MODEL:
            self.module = surgery.replace_unsupported_layers(self.module, replacement_dict={'tinyml_modelopt_quant_replace_types': {'quant_replace_types': _convert_replacement_func}})
        return self

    def export(self, *args, model_qconfig_format=TinyMLModelQConfigFormat.TINPU_INT_MODEL, simplify=True, skipped_optimizers=None, **kwargs):
        skipped_optimizers = skipped_optimizers or ['fuse_add_bias_into_conv', 'eliminate_nop_with_unit']
        super().export(*args, model_qconfig_format=model_qconfig_format, simplify=simplify, skipped_optimizers=skipped_optimizers, **kwargs)

    def measure_stats(self, float_output, quant_output):
        diff_output = (float_output - quant_output)
        diff_output_abs = diff_output.abs()
        diff_output_sqr = diff_output**2
        float_output_sqr = float_output**2
        quant_error_min = diff_output_abs.min().item()
        quant_error_max = diff_output_abs.max().item()
        quant_error_mean = diff_output_abs.mean().item()
        quant_snr_db = (10 * torch.log10(float_output_sqr.mean() / diff_output_sqr.mean())).item()
        quant_psnr_db = (10 * torch.log10(float_output_sqr.max() / diff_output_sqr.mean())).item()
        quant_absmu_by_sigma = (float_output.abs().mean() / diff_output.std()).item()
        diff_output_stats = dict(snr_db=quant_snr_db, psnr_db=quant_psnr_db, absmu_by_sigma=quant_absmu_by_sigma,
                                 mean=quant_error_mean, min=quant_error_min, max=quant_error_max)
        return diff_output_stats

    def is_batch_normalized(self, module: GraphModule) -> bool:
        named_modules = dict(module.named_modules())
        batch_norm_modules = (torch.ao.nn.quantized.modules.batchnorm.BatchNorm2d, torch.nn.BatchNorm2d)
        for name_entry, module_entry in named_modules.items():
            if len(list(module_entry.parameters(recurse=False))) > 0 and isinstance(module_entry, batch_norm_modules):
                return True
        return False

    def replacement_rules(self, replacement_utils: TINPUQuantizedReplacementUtils, is_batch_normalized: bool, output_dequantize: bool) -> List[Tuple]:
        # List to store the pattern and corresponding replacement function
        replacement_rules = [
            ([torch.ao.nn.quantized.modules.batchnorm.BatchNorm2d], replacement_utils.from_qbn),
            # Pooling Modules
            ([torch.nn.AvgPool2d], replacement_utils.from_avg_pool2d),                              # OSS required
            ([torch.nn.AdaptiveAvgPool2d], replacement_utils.from_adaptive_avg_pool2d),             # OSS required
            ([torch.nn.MaxPool2d], replacement_utils.from_max_pool2d),                              # OSS not required
            # Flatten Modules
            (['dequantize', torch.nn.Flatten], replacement_utils.from_dq_flatten),                  # Removes dequantization
            ([torch.quantize_per_tensor, torch.nn.Flatten], replacement_utils.from_q_module),        # Removes quantization
            # Torch Functions
            ([torch.ops.quantized.add_relu], replacement_utils.from_add_relu),
            ([torch.ops.quantized.add], replacement_utils.from_add),
            # ConvRelu2D Module
            ([torch.ao.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d], replacement_utils.from_qconv_relu),
            ([torch.ao.nn.quantized.modules.conv.Conv2d], replacement_utils.from_qconv),
            # LinearRelu Module
            ([torch.ao.nn.intrinsic.quantized.modules.linear_relu.LinearReLU], replacement_utils.from_qlinear_relu),
            ([torch.ao.nn.quantized.modules.linear.Linear], replacement_utils.from_qlinear),
            # Leftover Modules
            ([torch.quantize_per_tensor], replacement_utils.from_q),
        ]
        # Dequantization Module
        if output_dequantize:
            # Replaces dequantization layer with OSS
            replacement_rules += [(['dequantize'], replacement_utils.from_dq_with_dq)]
        else:
            # Replaces dequantization layer with Identity
            replacement_rules += [(['dequantize'], replacement_utils.from_dq)]

        return replacement_rules

    def _convert_replacement(self, module: GraphModule, pattern, *args, output_dequantize: bool = False, **kwargs) -> GraphModule:
        # Check if the model has batch normalization
        is_batch_normalized = self.is_batch_normalized(module)
        # Convert the module using symbolic trace
        module = torch.fx.symbolic_trace(module) if not isinstance(module, torch.fx.GraphModule) else module
        # Get the replacement rules to change the pattern
        replacement_utils = TINPUQuantizedReplacementUtils(module, self.weight_bw, self.activation_bw, self.power2_scale)
        replacement_rules = self.replacement_rules(replacement_utils, is_batch_normalized, output_dequantize)
        # Replace the patterns using the replacement function
        for replacement_pattern, replacement_function in replacement_rules:
            matches = replacement_utils.search_pattern(replacement_pattern)
            for (start, end) in matches:
                replacement_function(start, end)
        replacement_utils.update_module(module)
        return module


class TINPUTinyMLQATFxModule(TINPUTinyMLQuantFxModule):
    '''
    The QAT base class.
    Any additional enhancements that we do specifically only QAT later can be added in this class.
    '''
    pass


class TINPUTinyMLPTQFxModule(TINPUTinyMLQuantFxModule):
    '''
    The PTQ base class.
    Any additional enhancements that we do specifically only PTQ later can be added in this class.
    '''

    def __init__(self, *args, is_qat=False, **kwargs):
        super().__init__(*args, is_qat=is_qat, **kwargs)
