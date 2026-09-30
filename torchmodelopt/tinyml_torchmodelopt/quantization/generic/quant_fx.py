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

from ..common import *
from ..base.fx import TinyMLQuantFxBaseModule

from torch.fx import GraphModule
from typing import List, Tuple, Optional

from .quant_utils import GENERICQuantizedReplacementUtils
from ...surgery import remove_identity

class GenericTinyMLQuantFxModule(TinyMLQuantFxBaseModule):
    def __init__(self, model, *args, qconfig_type: TinyMLQConfigType, output_int: bool = True, **kwargs):
        '''
        The QAT wrapper module does the preparation like in:
        qat_model = quantize_fx.prepare_qat_fx(nn_model, qconfig_mapping, example_input)
        This can also export a full INT8 model.

        The api being called doesn't actually pass qconfig_type - so it will be defined inside.
        But if you need to pass, it can be defined this way.
        # qconfig_type supported for TINPU in F28 devices
        '''
        # qconfig_type = None is equivalent to WC8AT8 (or DEFAULT) which uses per_tensor_affine
        # Note: activation qscheme=torch.per_tensor_affine can be converted onnx model with QOperator using onnxruntime optimization
        # but activation qscheme=torch.per_tensor_symmetric stays as QDQ even when using onnxruntime optimization
        self.output_int = output_int
        super().__init__(model, *args, qconfig_type=qconfig_type, **kwargs)
    
    def convert(self, *args, **kwargs):
        '''
        The convert function is used to convert the model to TINPU supported ONNX model. 
        '''
        # first convert the model to int
        super().convert(*args, **kwargs)
        # then apply the transformation to required output format
        self.module = self._convert_replacement(self.module, self.output_int)
        return self

    def export(self, *args, simplify=True, **kwargs):
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


    def replacement_rules(self, replacement_utils: GENERICQuantizedReplacementUtils) -> List[Tuple]:
        # List to store the pattern and corresponding replacement function
        replacement_rules = [
            (['permute', 'unsqueeze'], replacement_utils.from_permute),
        ]
        return replacement_rules

    def _convert_replacement(self, module: GraphModule, output_int: bool = False) -> GraphModule:
        module = remove_identity(module)
        module.delete_all_unused_submodules()
        # Convert the module using symbolic trace
        module = torch.fx.symbolic_trace(module) if not isinstance(module, torch.fx.GraphModule) else module
        # Get the replacement rules to change the pattern
        replacement_utils = GENERICQuantizedReplacementUtils(module)
        replacement_rules = self.replacement_rules(replacement_utils)
        # Replace the patterns using the replacement function
        for replacement_pattern, replacement_function in replacement_rules:
            matches = replacement_utils.search_pattern(replacement_pattern)
            for (start, end) in matches:
                replacement_function(start, end)
        replacement_utils.update_module(module)
        return module

class GenericTinyMLQATFxModule(GenericTinyMLQuantFxModule):
    '''
    The QAT base class.
    Any additional enhancements that we do specifically only QAT later can be added in this class.
    '''

    def __init__(self, *args, is_qat=True, model_output_format=TinyMLModelQConfigFormat.INT_MODEL ,**kwargs):
        super().__init__(*args, is_qat=is_qat, model_output_format=model_output_format, **kwargs)


class GenericTinyMLPTQFxModule(GenericTinyMLQuantFxModule):
    '''
    The PTQ base class.
    Any additional enhancements that we do specifically only PTQ later can be added in this class.
    '''

    def __init__(self, *args, is_qat=False, model_output_format=TinyMLModelQConfigFormat.INT_MODEL , **kwargs):
        super().__init__(*args, is_qat=is_qat, model_output_format=model_output_format, **kwargs)
