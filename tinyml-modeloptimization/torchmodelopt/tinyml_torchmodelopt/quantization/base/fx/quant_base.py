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

import os
import warnings
import copy
import torch
from torch.ao.quantization import quantize_fx
from torch.ao.quantization import QConfigMapping


from ... import common
from . import qconfig_types
from . import quant_utils
from . import bias_calibration


class TinyMLQuantFxBaseModule(torch.nn.Module):
    def __init__(self, model, qconfig_type=None, example_inputs=None, is_qat=True, backend="qnnpack",
                 total_epochs=0, num_batch_norm_update_epochs=None, num_observer_update_epochs=False, 
                 prepare_qdq=True, bias_calibration_factor=0.0, verbose=True):
        '''
        Parameters:
            model: input model to be quantized
            qconfig_type: a dictionary, QConfigType or QConfigMode
            example_inputs: example input tensor
            is_qat: indicates QAT or PTQ
            total_epochs: number of epochs of training
            num_batch_norm_update_epochs:
                False: do not freeze batch norm
                None: freeze batch norm at half the epochs
                Otherwise (a number): freeze batch norm at the specified number of epochs
            num_observer_update_epochs:
                False: do not freeze observers
                None: freeze observers at half the epochs
                Otherwise (a number): freeze observers at the specified number of epochs

        The QAT wrapper module does the preparation like in:
        qat_model = quantize_fx.prepare_qat_fx(nn_model, qconfig_mapping, example_input)

        The api being called doesn't actually pass qconfig_type - so it will be defined inside.
        But if you need to pass, it can be defined this way.
        # qconfig_type supported for TINPU in F28 devices
        qconfig_type = {
            'weight': {
                'bitwidth': 8,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': True (TINPU) / False(Generic),
                'range_max': None,
                'fixed_range': False
            },
            'activation': {
                'bitwidth': 8,
                'qscheme': torch.per_tensor_symmetric,
                'power2_scale': True (TINPU) / False(Generic),
                'range_max': None,
                'fixed_range': False
            }
        }
        '''

        super().__init__()
        if not total_epochs:
            raise RuntimeError("total_epochs must be provided")
        #

        # split if qconfig is a comma separated list of segments
        # (qconfig will change after some epochs if this has comma separated values)
        if isinstance(qconfig_type, dict) or qconfig_type is None:
            qconfig_mapping = qconfig_types.get_default_qconfig_mapping(qconfig_type)
        elif isinstance(qconfig_type, torch.ao.quantization.QConfig):
            qconfig_mapping = QConfigMapping().set_global(qconfig_type)
        elif isinstance(qconfig_type, torch.ao.quantization.QConfigMapping):
            qconfig_mapping = qconfig_type
        else:
            raise RuntimeError(f"invalid value for qconfig_type: {qconfig_type}")
        #

        if prepare_qdq:
            model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs)
        else:
            model = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)
        #

        self.module = model

        # other parameters
        self.is_qat = is_qat
        self.backend = backend
        self.qconfig_type = qconfig_type
        self.num_batch_norm_update_epochs = num_batch_norm_update_epochs
        self.num_observer_update_epochs = num_observer_update_epochs
        self.num_epochs_tracked = 0
        self.total_epochs = total_epochs
        self.bias_calibration_hooks = []
        # set the quantization backend - qnnpack, fbgemm, x86, onednn etc.
        self.set_quant_backend(backend)

        # related to adaptive quantization
        self.bias_calibration_factor = bias_calibration_factor

        if not self.is_qat:
            self.disable_backward_for_ptq()
            # find the bias calibration factor from observer - to be used as a flag.
            for m in self.module.modules():
                if isinstance(m, torch.ao.quantization.ObserverBase):
                    if hasattr(m, 'bias_calibration_factor'):
                        self.bias_calibration_factor = max(self.bias_calibration_factor, m.bias_calibration_factor)
                        m.bias_calibration_factor = self.bias_calibration_factor
                    #
                #
            #
        #
        if not verbose:
            quant_utils.print_once_dict = {'Freezing BN for subsequent epochs': None,
                                           'Freezing ranges for subsequent epochs': None}

    def set_quant_backend(self, backend=None):
        if backend not in torch.backends.quantized.supported_engines:
            raise RuntimeError("Quantized backend not supported: " + str(backend))
        torch.backends.quantized.engine = backend

    def load_weights(self, pretrained, *args, strict=True, state_dict_name=None, **kwargs):
        data_dict = torch.load(self, pretrained, *args, **kwargs)
        if state_dict_name:
            state_dict_names = state_dict_name if isinstance(state_dict_name, (list,tuple)) else [state_dict_name]
            for s_name in state_dict_names:
                data_dict = data_dict[s_name] if ((data_dict is not None) and s_name in data_dict) else data_dict
            #
        #
        self.load_state_dict(data_dict, strict=strict)

    def train(self, mode: bool = True):
        # put the model in expected mode
        super().train(mode=mode)
        # also freeze the params if required
        if mode is True:
            # set the default epoch at which freeze occurs during training (if missing)
            enable_freeze_bn = (self.num_batch_norm_update_epochs is not False)
            enable_freeze_observer = (self.num_observer_update_epochs is not False)
            num_batch_norm_update_epochs = self.num_batch_norm_update_epochs or ((self.total_epochs//2)-1)
            num_observer_update_epochs = self.num_observer_update_epochs or ((self.total_epochs//2)+1)
            freeze_bn = enable_freeze_bn and ((not self.is_qat) or (self.num_epochs_tracked >= num_batch_norm_update_epochs))
            freeze_observers = enable_freeze_observer and ((self.num_epochs_tracked >= num_observer_update_epochs))
            self.freeze(freeze_bn=freeze_bn, freeze_observers=freeze_observers)
            self.num_epochs_tracked += 1
            if (not self.is_qat) and self.bias_calibration_factor:
                self.bias_calibration_hooks = bias_calibration.insert_bias_calibration_hooks(self.module, self.total_epochs, self.num_epochs_tracked)
            #
        else:
            self.freeze()
            if (not self.is_qat) and self.bias_calibration_factor:
                self.bias_calibration_hooks = bias_calibration.remove_hooks(self.module, self.bias_calibration_hooks)
            #
        #
        return self

    def freeze(self, freeze_bn=True, freeze_observers=True):
        if freeze_observers is True:
            self.apply(torch.ao.quantization.disable_observer)
            quant_utils.print_once('Freezing ranges for subsequent epochs')
        elif freeze_observers is False:
            self.apply(torch.ao.quantization.enable_observer)
        #
        if freeze_bn is True:
            self.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            quant_utils.print_once('Freezing BN for subsequent epochs')
        elif freeze_bn is False:
            self.apply(torch.nn.intrinsic.qat.update_bn_stats)
        #
        return self

    def unfreeze(self, freeze_bn=False, freeze_observers=False):
        self.freeze(freeze_bn, freeze_observers)
        return self

    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)

    def convert(self, model_qconfig_format=None, inplace=False, device='cpu'):
        self.freeze()
        # make a copy in order not to alter the original
        model = self.module if inplace else copy.deepcopy(self.module)
        # convert requires cpu model
        model = model.to(torch.device(device))
        # now do the actual conversion
        self.module = quantize_fx.convert_fx(model)
        return self

    def _is_observed_module(self) -> bool:
        # from: torch/ao/quantization/fx/graph_module.py
        return hasattr(self.module, "meta") and "_observed_graph_module_attrs" in self.module.meta

    def export(self, example_inputs, filename='model.onnx', opset_version=17, model_qconfig_format=None,
               preserve_qdq_model=True, simplify=True, skipped_optimizers=None, device='cpu', make_copy=True,
               is_converted=True, verbose=False, **export_kwargs):
        if self._is_observed_module():
            model = self.convert(self, device=device, model_qconfig_format=model_qconfig_format, make_copy=make_copy)
        elif not is_converted:
            model = self.convert(self, device=device, model_qconfig_format=model_qconfig_format, make_copy=make_copy)
        else:
            model = self.module
            if verbose:
                warnings.warn("model has already been converted before calling export. make sure it is done correctly.")

        if model_qconfig_format == common.TinyMLModelQConfigFormat.INT_MODEL:
            # # Convert QDQ format to Int8 format
            import onnxruntime as ort
            qdq_filename = os.path.splitext(filename)[0] + '_qdq.onnx'
            torch.onnx.export(model, example_inputs.to(device=device), qdq_filename, opset_version=opset_version, **export_kwargs)
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            so.optimized_model_filepath = filename
            # logger.info("Inplace conversion of QDQ model to INT8 model at: {}".format(onnx_file))
            ort.InferenceSession(qdq_filename, so)
            if not preserve_qdq_model:
                os.remove(qdq_filename)
            #
        else:
            torch.onnx.export(model, example_inputs.to(device=device), filename, opset_version=opset_version,
                              **export_kwargs)
        #
        if simplify:
            try:
                import onnx
                from onnxsim import simplify
                onnx_model = onnx.load(filename)
                onnx_model, check = simplify(onnx_model, skipped_optimizers=skipped_optimizers)
                onnx.save(onnx_model, filename)
            except:
                print("Something went wrong in simplification - maybe due to multi processes, skipping this step")
        #

    def disable_backward_for_ptq(self):
        '''
        a utility method that can be called to disable backward - useful for PTQ
        '''
        def backward_hook_with_error(m, g_in, g_out):
            raise RuntimeError("backward need not be called for PTQ - aborting")
        #
        self.register_full_backward_hook(backward_hook_with_error)
