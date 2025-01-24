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

from ..common import *
from ..base.fx import TinyMLQuantFxBaseModule


class GenericTinyMLQuantFxModule(TinyMLQuantFxBaseModule):
    def __init__(self, model, *args, qconfig_type=None,  **kwargs):
        '''
        The QAT wrapper module does the preparation like in:
        qat_model = quantize_fx.prepare_qat_fx(nn_model, qconfig_mapping, example_input)
        This can also export a full INT8 model.

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

        # qconfig_type = None is equivalent to WC8AT8 (or DEFAULT) which uses per_tensor_affine
        # Note: activation qscheme=torch.per_tensor_affine can be converted onnx model with QOperator using onnxruntime optimization
        # but activation qscheme=torch.per_tensor_symmetric stays as QDQ even when using onnxruntime optimization
        super().__init__(model, *args, qconfig_type=qconfig_type, backend='fbgemm' if platform.system() in ['Windows'] else 'qnnpack', **kwargs)

    def convert(self, *args, model_qconfig_format=TinyMLQConfigFormat.INT_MODEL, **kwargs):
        return super().convert(*args, model_qconfig_format=model_qconfig_format, **kwargs)

    def export(self, *args, model_qconfig_format=TinyMLQConfigFormat.INT_MODEL, **kwargs):
        super().export(*args, model_qconfig_format=model_qconfig_format, **kwargs)


class GenericTinyMLQATFxModule(GenericTinyMLQuantFxModule):
    '''
    The QAT base class.
    Any additional enhancements that we do specifically only QAT later can be added in this class.
    '''
    pass


class GenericTinyMLPTQFxModule(GenericTinyMLQuantFxModule):
    '''
    The PTQ base class.
    Any additional enhancements that we do specifically only PTQ later can be added in this class.
    '''

    def __init__(self, *args, is_qat=False, **kwargs):
        super().__init__(*args, is_qat=is_qat, **kwargs)
