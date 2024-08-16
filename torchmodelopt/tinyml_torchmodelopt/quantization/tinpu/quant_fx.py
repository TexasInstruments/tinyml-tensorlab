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

import platform
import torch
import edgeai_torchmodelopt
from ..common import TinyMLQuantizationVersion, TinyMLModelQuantFormat, GenericTinyMLQATFxModuleBase
from . import quant_utils


class TINPUTinyMLQATFxModule(GenericTinyMLQATFxModuleBase):
    def __init__(self, *args, qconfig_type=None, **kwargs):
        if qconfig_type is None:
            # there are multiple ways to specify qconfig_type - one is to use a dictionary like this.
            # qconfig_type = qconfig_type or dict(weight=dict(bitwidth=8, qscheme=torch.per_channel_symmetric, power2_scale=True),
            #   activation=dict(bitwidth=8, qscheme=torch.per_tensor_symmetric, power2_scale=True, range_max=None, fixed_range=False))
            # another way is to use one of the predefined presets
            qconfig_type = edgeai_torchmodelopt.xmodelopt.quantization.v2.qconfig_types.QConfigType.WC8SYMP2_AT8SYMP2
        #
        super().__init__(*args, qconfig_type=qconfig_type, backend='fbgemm' if platform.system() in ['Windows'] else 'qnnpack', **kwargs)

    def convert(self, *args, model_quant_format=TinyMLModelQuantFormat.TINPU_INT_MODEL, output_dequantize=False, **kwargs):
        # first convert the model to int
        super().convert(*args, **kwargs)
        _convert_replacement_func = lambda module, pattern, *largs, **lkwargs: \
            self._convert_replacement(module, pattern, *largs, output_dequantize=output_dequantize, **lkwargs)

        # then apply the transformation to required output format
        if model_quant_format == TinyMLModelQuantFormat.TINPU_INT_MODEL:
            self.module = edgeai_torchmodelopt.xmodelopt.surgery.v2.convert_to_lite_fx(self.module,
                                    replacement_dict={'replace_types1': _convert_replacement_func})
        #
        return self

    def export(self, *args, model_quant_format=TinyMLModelQuantFormat.TINPU_INT_MODEL, simplify=True, skipped_optimizers=None, **kwargs):
        skipped_optimizers = skipped_optimizers or ['fuse_add_bias_into_conv', 'eliminate_nop_with_unit']
        super().export(*args, model_quant_format=model_quant_format, simplify=simplify,
                       skipped_optimizers=skipped_optimizers, **kwargs)

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

    def _convert_replacement(self, module, pattern, *args, output_dequantize=False, **kwargs):
        named_modules = dict(module.named_modules())
        modules_list = list(named_modules.values())
        first_module_with_params = None
        for name_entry, module_entry in named_modules.items():
            if len(list(module_entry.parameters(recurse=False))) > 0:
                first_module_with_params = module_entry
                break
            #
        #
        with_input_batchnorm = isinstance(first_module_with_params,
                    (torch.ao.nn.quantized.modules.batchnorm.BatchNorm2d, torch.nn.BatchNorm2d))

        module = torch.fx.symbolic_trace(module) if not isinstance(module, torch.fx.GraphModule) else module

        # for qdq model
        # replacement_entries_qdq = [
        #    ([torch.ao.nn.intrinsic.modules.fused.ConvReLU2d,edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize], model_quant_utils.TINPUQuantizedReplacement.from_conv_relu_fq),
        #    ([edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize, torch.nn.BatchNorm2d, edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize], model_quant_utils.TINPUQuantizedReplacement.from_fq_bn_fq),
        #    ([edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize], model_quant_utils.TINPUQuantizedReplacement.from_fq),
        # }

        # for converted model
        if with_input_batchnorm:
            first_entry = [
                ([torch.quantize_per_tensor, torch.ao.nn.quantized.modules.batchnorm.BatchNorm2d], quant_utils.TINPUQuantizedReplacement.from_q_qbn)
            ]
        else:
            first_entry = [
                ([torch.quantize_per_tensor, torch.nn.Identity], quant_utils.TINPUQuantizedReplacement.from_q_id),
                ([torch.quantize_per_tensor], quant_utils.TINPUQuantizedReplacement.from_q)
            ]

        # for converted model
        replacement_entries_converted = first_entry + [
            ([torch.nn.Sequential], quant_utils.TINPUQuantizedReplacement.from_child_module),
            ([torch.nn.Module], quant_utils.TINPUQuantizedReplacement.from_child_module),
            ([torch.nn.MaxPool2d], quant_utils.TINPUQuantizedReplacement.from_passthrough_module),
            ([torch.nn.AdaptiveAvgPool2d, 'dequantize'], quant_utils.TINPUQuantizedReplacement.from_module_with_dq),
            ([torch.nn.AdaptiveAvgPool2d], quant_utils.TINPUQuantizedReplacement.from_passthrough_module),
            ([torch.nn.Flatten], quant_utils.TINPUQuantizedReplacement.from_passthrough_module),
            ([torch.ao.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d], quant_utils.TINPUQuantizedReplacement.from_qconv_relu),
            ([torch.ao.nn.intrinsic.quantized.modules.linear_relu.LinearReLU], quant_utils.TINPUQuantizedReplacement.from_qlinear_relu),
            ([torch.nn.Flatten, 'quantize_per_tensor'], quant_utils.TINPUQuantizedReplacement.from_module_with_q),
            ([torch.ao.nn.quantized.modules.linear.Linear], quant_utils.TINPUQuantizedReplacement.from_qlinear),
            (['dequantize'], quant_utils.TINPUQuantizedReplacement.from_dq_with_dq if output_dequantize else quant_utils.TINPUQuantizedReplacement.from_dq),
        ]

        for replacement_pattern, replacement_function in replacement_entries_converted:
            matches = edgeai_torchmodelopt.xmodelopt.surgery.v2.replacer.straight_type_chain_searcher(
                module, replacement_pattern)
            for no_of_module_replaced, (start, end) in enumerate(matches):
                new_fq_module = replacement_function(module, start, end)
                edgeai_torchmodelopt.xmodelopt.surgery.v2.replacer._replace_pattern(
                    module, start, end, new_fq_module, no_of_module_replaced)
            #
        #
        return module
