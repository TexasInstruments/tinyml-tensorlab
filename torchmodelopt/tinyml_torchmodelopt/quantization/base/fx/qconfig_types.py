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

# Imports Torch
import torch
from torch.ao.quantization import QConfig, QConfigMapping
import torch.ao.quantization

from . import observer_types
from . import fake_quant_types
from . import auto_quantization


def _get_fake_quant_from_name(fake_quant_name):
    if fake_quant_name == 'soft_tanh':
        return fake_quant_types.SoftTanhFakeQuantize
    elif fake_quant_name == 'soft_sigmoid':
        return fake_quant_types.SoftSigmoidFakeQuantize
    elif fake_quant_name == 'default':
        return torch.ao.quantization.FakeQuantize
    else:
        raise ValueError(
            "Invalid soft quantization type. "
            "Soft Quantization types could be 'soft_tanh', 'soft_sigmoid' and 'default'"
        )

def _get_observer_class_from_name(observer_name, qscheme, is_weight=True):
    '''
    Helper function to select observer class based on observer name and qscheme.

    Args:
        observer_name (str): Name of the observer ('histogram_range', 'lsq_observer', 'channel_adaptive', or None)
        qscheme: Quantization scheme (e.g., torch.per_channel_symmetric)
        is_weight (bool): Whether this is for weight (True) or activation (False)

    Returns:
        Observer class
    '''
    if observer_name == 'lsq_observer':
        if is_weight:
            if qscheme == torch.per_channel_symmetric:
                return observer_types.LSQPerChannelObserver
            else:
                return observer_types.LSQObserver
        else:
            if qscheme == torch.per_tensor_symmetric or qscheme == torch.per_tensor_affine:
                return observer_types.LSQObserver
            else:
                return observer_types.LSQPerChannelObserver
    elif observer_name == 'histogram_range':
        # Choose histogram observer based on quantization scheme
        if qscheme == torch.per_channel_symmetric:
            # Use per-channel histogram observer for per_channel quantization
            return observer_types.RangeShrinkPerChannelHistogramObserver
        else:
            # Use standard histogram observer for per_tensor quantization
            return observer_types.RangeShrinkFastHistogramObserver
    elif observer_name == 'kl_divergence':
        # Choose histogram observer based on quantization scheme
        if qscheme == torch.per_channel_symmetric:
            # Use per-channel histogram observer for per_channel quantization
            return observer_types.KLDivergencePerChannelObserver
        else:
            # Use standard histogram observer for per_tensor quantization
            return observer_types.KLDivergenceObserver
    elif observer_name == 'entropy_based_cutoff':
        # Choose histogram observer based on quantization scheme
        if qscheme == torch.per_channel_symmetric:
            # Use per-channel histogram observer for per_channel quantization
            return observer_types.EntropyBasedCutoffPerChannelObserver
        else:
            # Use standard histogram observer for per_tensor quantization
            return observer_types.EntropyBasedCutoffObserver
    elif observer_name is None or observer_name == 'default':
        # Use default observers based on qscheme if no specific observer requested
        if is_weight:
            if qscheme == torch.per_channel_symmetric:
                # we don't have a histogram observer that can do per_channel_symmetric - so use MinMax
                return torch.ao.quantization.PerChannelMinMaxObserver
            else:
                return torch.ao.quantization.MinMaxObserver
        else:
            return torch.ao.quantization.MovingAverageMinMaxObserver
    else:
        raise ValueError(
            f"Invalid observer name: {observer_name}. "
            "Supported observer names are: 'histogram_range', 'lsq_observer', or None"
        )


def get_default_qconfig(qconfig_dict=None):
    '''
    This default qconfig uses symmetric, power2 quantization.
    It can be changed by passing appropriate qconfig_dict.
    '''
    qconfig_dict = qconfig_dict or dict()
    weight_qconfig = qconfig_dict.get('weight', dict())
    weight_dtype = weight_qconfig.get('dtype', torch.qint8)
    weight_bitwidth = weight_qconfig.get('bitwidth', 8)
    weight_quant_min = weight_qconfig.get('quant_min', -((2 ** (weight_bitwidth - 1)) - 1))
    weight_quant_max = weight_qconfig.get('quant_max', ((2 ** (weight_bitwidth - 1)) - 1))
    weight_qscheme = weight_qconfig.get('qscheme', torch.per_channel_symmetric)
    weight_power2_scale = weight_qconfig.get('power2_scale', True)
    weight_range_max = weight_qconfig.get('range_max', None)
    weight_fixed_range = weight_qconfig.get('fixed_range', False)
    weight_observer = weight_qconfig.get('observer', None)
    weight_soft_quant = weight_qconfig.get('soft_quant', 'default')

    activation_qconfig = qconfig_dict.get('activation', dict())
    activation_dtype = activation_qconfig.get('dtype', torch.quint8)
    activation_bitwidth = activation_qconfig.get('bitwidth', 8)
    activation_quant_min = activation_qconfig.get('quant_min', 0)
    activation_quant_max = activation_qconfig.get('quant_max', (2 ** activation_bitwidth) - 1)
    activation_qscheme = activation_qconfig.get('qscheme', torch.per_tensor_symmetric)
    activation_power2_scale = activation_qconfig.get('power2_scale', True)
    activation_range_max = activation_qconfig.get('range_max', None)
    activation_fixed_range = activation_qconfig.get('fixed_range', False)
    activation_observer = activation_qconfig.get('observer', None)
    bias_calibration_factor = activation_qconfig.get('bias_calibration_factor', 0.0)
    activation_soft_quant = activation_qconfig.get('soft_quant', 'default')

    # Select weight observer based on observer parameter or default behavior
    weight_observer_base_class = _get_observer_class_from_name(weight_observer, weight_qscheme, is_weight=True)        
    weight_fake_quant_type = _get_fake_quant_from_name(weight_soft_quant)

    weight_fake_quant = weight_fake_quant_type.with_args(
        observer=observer_types.get_weight_observer_type(base_class=weight_observer_base_class),
        quant_min=weight_quant_min, quant_max=weight_quant_max,
        qscheme=weight_qscheme, dtype=weight_dtype, power2_scale=weight_power2_scale,
        range_max=weight_range_max, fixed_range=weight_fixed_range)

    # Select activation observer based on observer parameter or default behavior
    activation_observer_base_class = _get_observer_class_from_name(activation_observer, activation_qscheme, is_weight=False)
    activation_fake_quant_type = _get_fake_quant_from_name(activation_soft_quant)

    activation_fake_quant = activation_fake_quant_type.with_args(
        observer=observer_types.get_activation_observer_type(base_class=activation_observer_base_class),
        quant_min=activation_quant_min, quant_max=activation_quant_max,
        qscheme=activation_qscheme, dtype=activation_dtype, power2_scale=activation_power2_scale,
        range_max=activation_range_max, fixed_range=activation_fixed_range,
        bias_calibration_factor=bias_calibration_factor)

    qconfig = QConfig(weight=weight_fake_quant, activation=activation_fake_quant)
    return qconfig


def apply_mixed_precision(qconfig_mapping, qconfig_dict, mixed_precision):
    for bit_width in mixed_precision:
        layers = mixed_precision[bit_width]
        if bit_width == 32:
            for layer in layers:
                qconfig_mapping.set_module_name(layer, None)
        else:
            qconfig_dict['weight']['bitwidth'] = bit_width
            qconfig_dict['activation']['bitwidth'] = bit_width
            qconfig = get_default_qconfig(qconfig_dict=qconfig_dict)
            for layer in layers:
                qconfig_mapping.set_module_name(layer, qconfig)
    return qconfig_mapping

def get_default_qconfig_mapping(model, qconfig_type=None):
    qconfig_dict = qconfig_type
    if isinstance(qconfig_dict, dict) or qconfig_dict is None:
        qconfig_type = get_default_qconfig(qconfig_dict=qconfig_dict)

    if not isinstance(qconfig_type, QConfig):
        raise RuntimeError("Unrecognized type of qconfig_type")

    qconfig_mapping = QConfigMapping().set_global(qconfig_type)
    if qconfig_dict is None:
        return qconfig_mapping

    weight_mixed_precision = qconfig_dict.get('weight', {}).get('mixed_precision', {})
    if weight_mixed_precision:
        qconfig_mapping = apply_mixed_precision(qconfig_mapping, qconfig_dict, weight_mixed_precision)
    auto_quantization_enabled = qconfig_dict.get('auto_quantization')
    if auto_quantization_enabled:
        qconfig_mapping = auto_quantization.run_auto_quantization(
            model, qconfig_dict, qconfig_mapping, get_default_qconfig, apply_mixed_precision
        )

    return qconfig_mapping