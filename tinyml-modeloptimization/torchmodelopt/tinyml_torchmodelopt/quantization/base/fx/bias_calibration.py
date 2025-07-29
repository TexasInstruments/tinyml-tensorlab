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


from functools import partial

import torch.ao.quantization


def _bias_calibration_hook(m, x, y, bias_calibration_factor, bias_module):
    bias_error = 0
    if isinstance(x, tuple):
        x = x[0]
    if len(x.shape) == 3:
        float_mean = x.mean(dim=(0,1))
        quant_mean = y.mean(dim=(0,1))
        bias_error = float_mean - quant_mean
    elif len(x.shape) == 4:
        if x.shape[1] == bias_module.bias.shape[0]:
            float_mean = x.mean(dim=(0,2,3))
            quant_mean = y.mean(dim=(0,2,3))
            bias_error = float_mean - quant_mean
        elif x.shape[3] == bias_module.bias.shape[0]:
            float_mean = x.mean(dim=(0,1,2))
            quant_mean = y.mean(dim=(0,1,2))
            bias_error = (float_mean - quant_mean)
        #
    #
    bias_module.bias.data += (bias_error * bias_calibration_factor)
    return y


def _add_bias_calibration_hook(model, total_epochs, num_epochs_tracked):
    epoch_fraction = num_epochs_tracked / total_epochs
    bias_calibration_fraction = (0.1 if (epoch_fraction <= 0.25 or epoch_fraction >= 0.75) else 1.0)
    all_modules = dict(model.named_modules())
    all_hooks = []
    for node in model.graph.nodes:
        if (node.prev.target in all_modules) and (node.target in all_modules):
            cur_module = all_modules[node.target]
            if isinstance(cur_module, torch.ao.quantization.FakeQuantizeBase):
                bias_module = all_modules[node.prev.target]
                if getattr(bias_module, 'bias', None) is None and hasattr(bias_module, 'bn'):
                    bias_module = bias_module.bn
                if getattr(bias_module, 'bias', None) is not None:
                    fake_quantize_module = all_modules[node.target]
                    observer_module = fake_quantize_module.activation_post_process
                    bias_calibration_factor = observer_module.bias_calibration_factor
                    _bias_calibration_hook_bind = partial(_bias_calibration_hook, bias_calibration_factor=bias_calibration_factor*bias_calibration_fraction, bias_module=bias_module)
                    this_hook = fake_quantize_module.register_forward_hook(_bias_calibration_hook_bind)
                    all_hooks.append(this_hook)
                #
            #
        #
    #
    return all_hooks


def insert_bias_calibration_hooks(model, total_epochs, num_epochs_tracked):
    bias_hooks = []
    bias_hooks += _add_bias_calibration_hook(model, total_epochs, num_epochs_tracked)
    return bias_hooks


def remove_hooks(model, hooks):
    for hook_handle in hooks:
        hook_handle.remove()
    #
    hooks = []
    return hooks