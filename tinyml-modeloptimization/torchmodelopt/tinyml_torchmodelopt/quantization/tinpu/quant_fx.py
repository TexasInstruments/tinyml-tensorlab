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

from ..common import *
from ..base.fx import TinyMLQuantFxBaseModule

from torch.fx import GraphModule
from typing import List, Tuple, Optional

from ...surgery.quant_helper_func import remove_identity, assign_same_observers_for_residual_inputs, assign_same_observers_for_flatten
from .quant_utils import TINPUQuantizedReplacementUtils


class TINPUTinyMLQuantFxModule(TinyMLQuantFxBaseModule):
    def __init__(self, *args, qconfig_type: TinyMLQConfigType, output_int: bool = True, **kwargs) -> None:
        '''
        The QAT wrapper module does the preparation like in:
        qat_model = quantize_fx.prepare_qat_fx(nn_model, qconfig_mapping, example_input)
        It also uses an appropriate qconfig that imposes the constraints of the hardware.

        The API being called doesn't actually pass qconfig_type - so it will be defined inside.
        But if you need to pass, it can be defined the way in Args.

        Args:
            qconfig_type: Similar representation of QConfig dict that defines the \
                quantization configuration for the model.
            output_int: The ONNX model output format. \
                If True, the output of model will be quantized int8, if False, the output will be dequantized float
        '''
        self.output_int = output_int
        super().__init__(*args, qconfig_type=qconfig_type, **kwargs)
        assign_same_observers_for_residual_inputs(self.module)
        assign_same_observers_for_flatten(self.module)

    def convert(self, *args, **kwargs):
        '''
        The convert function is used to convert the model to TINPU supported ONNX model. 
        '''
        # first convert the model to int
        super().convert(*args, **kwargs)
        # then apply the transformation to required output format
        self.module = self._convert_replacement(self.module, self.output_int)
        return self

    def export(self, *args, simplify: bool = True, **kwargs):
        skipped_optimizers = ['fuse_add_bias_into_conv', 'eliminate_nop_with_unit']
        super().export(*args, simplify=simplify, skipped_optimizers=skipped_optimizers, **kwargs)

    def measure_stats(self, float_output, quant_output):
        diff_output = (float_output - quant_output)
        diff_output_abs = diff_output.abs()
        diff_output_sqr = diff_output ** 2
        float_output_sqr = float_output ** 2
        quant_error_min = diff_output_abs.min().item()
        quant_error_max = diff_output_abs.max().item()
        quant_error_mean = diff_output_abs.mean().item()
        quant_snr_db = (10 * torch.log10(float_output_sqr.mean() / diff_output_sqr.mean())).item()
        quant_psnr_db = (10 * torch.log10(float_output_sqr.max() / diff_output_sqr.mean())).item()
        quant_absmu_by_sigma = (float_output.abs().mean() / diff_output.std()).item()
        diff_output_stats = dict(
            snr_db=quant_snr_db, 
            psnr_db=quant_psnr_db, 
            absmu_by_sigma=quant_absmu_by_sigma,
            mean=quant_error_mean, 
            min=quant_error_min, 
            max=quant_error_max)
        return diff_output_stats

    def replacement_rules(self, replacement_utils: TINPUQuantizedReplacementUtils, output_int: bool) -> List[Tuple]:
        # List to store the pattern and corresponding replacement function
        replacement_rules = [
            ([torch.nn.BatchNorm2d, torch.quantize_per_tensor], replacement_utils.from_bnq),
            ([torch.ao.nn.intrinsic.modules.fused.ConvReLU2d], replacement_utils.from_conv_bn_relu),
            (['dequantize', torch.nn.ConvTranspose2d, torch.nn.ReLU], replacement_utils.from_dq_t_conv_bn_relu),
            ([torch.nn.ConvTranspose2d, torch.nn.ReLU], replacement_utils.from_t_conv_bn_relu),
            ([torch.ao.nn.quantized.modules.conv.ConvTranspose2d], replacement_utils.from_t_conv),
            ([torch.ao.nn.intrinsic.modules.fused.ConvBn2d], replacement_utils.from_conv_bn),
            ([torch.ao.nn.intrinsic.modules.fused.LinearReLU], replacement_utils.from_linear_relu),
            ([torch.ao.nn.quantized.modules.batchnorm.BatchNorm2d], replacement_utils.from_qbn),
            # Pooling Modules
            ([torch.nn.AvgPool2d], replacement_utils.from_avg_pool2d),                              # OSS required
            ([torch.nn.AdaptiveAvgPool2d], replacement_utils.from_adaptive_avg_pool2d),             # OSS required
            ([torch.nn.MaxPool2d], replacement_utils.from_max_pool2d),                              # OSS not required
            # Flatten Modules
            (['dequantize', torch.nn.Flatten], replacement_utils.from_dq_flatten),                  # Removes dequantization
            ([torch.quantize_per_tensor, torch.nn.Flatten], replacement_utils.from_q_module),        # Removes quantization
            ([torch.quantize_per_tensor, torch.ops.quantized.matmul, torch.ops.quantized.add], replacement_utils.from_matmul),
            ([torch.quantize_per_tensor, 'permute'], replacement_utils.from_permute),
            ([torch.quantize_per_tensor, 'transpose'], replacement_utils.from_transpose),
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
        if not output_int:
            # Replaces dequantization layer with OSS (dequantize to float)
            replacement_rules += [(['dequantize'], replacement_utils.from_dq_with_dq)]
        else:
            # Replaces dequantization layer with Identity (keep as int8)
            replacement_rules += [(['dequantize'], replacement_utils.from_dq)]

        return replacement_rules

    def _convert_replacement(self, module: GraphModule, output_int: bool = True) -> GraphModule:
        module = remove_identity(module)
        module.delete_all_unused_submodules()
        # Convert the module using symbolic trace
        module = torch.fx.symbolic_trace(module) if not isinstance(module, torch.fx.GraphModule) else module
        # Get the replacement rules to change the pattern
        replacement_utils = TINPUQuantizedReplacementUtils(module)
        replacement_rules = self.replacement_rules(replacement_utils, output_int)
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

    def __init__(self, *args, is_qat=True, model_output_format=TinyMLModelQConfigFormat.TINPU_INT_MODEL, **kwargs):
        super().__init__(*args, is_qat=is_qat, model_output_format=model_output_format, **kwargs)


class TINPUTinyMLPTQFxModule(TINPUTinyMLQuantFxModule):
    '''
    The PTQ base class.
    Any additional enhancements that we do specifically only PTQ later can be added in this class.
    '''

    def __init__(self, *args, is_qat=False, model_output_format=TinyMLModelQConfigFormat.TINPU_INT_MODEL, **kwargs):
        super().__init__(*args, is_qat=is_qat, model_output_format=model_output_format, **kwargs)