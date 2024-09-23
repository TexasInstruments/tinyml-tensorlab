
import edgeai_torchmodelopt
import model_quant_utils
import numpy as np
import torch
from torch.ao.quantization import QuantType
from torch.ao.quantization.quantize_fx import ConvertCustomConfig


class CustomQuantizedBatchNorm2d(torch.nn.Module):
    def __init__(self, add_value, mul_value, shift_value):
        super().__init__()
        self.add_value = add_value
        self.mul_vaue = mul_value
        self.shift_value = shift_value
        self.shift_to_mul = np.power(2, self.shift_value)

    def forward(self, x):
        return ((x + self.add_value) * self.mul_vaue) * self.shift_to_mul

    @staticmethod
    def from_observed(observed):
        if hasattr(observed, 'with_convert_custom_config') and observed.with_convert_custom_config:
            sigma = torch.sqrt(observed.running_var + observed.eps)
            fused_add_value = observed.running_mean / sigma + observed.bias
            fused_mul_value = observed.weight / sigma
            shift, scale = model_quant_utils.compute_shift_scale(fused_mul_value)
            return CustomQuantizedBatchNorm2d(fused_add_value.detach().numpy(), shift.detach().numpy(), scale.detach().numpy())
        else:
            return observed


def get_convert_custom_config(replace_input_norm=False, replace_activation_fake_quant=True):
    convert_custom_config = ConvertCustomConfig()
    if replace_input_norm:
        convert_custom_config.set_observed_to_quantized_mapping(torch.nn.BatchNorm2d, CustomQuantizedBatchNorm2d, QuantType.STATIC)
    #
    if replace_activation_fake_quant:
        convert_custom_config.set_observed_to_quantized_mapping(edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize, AdaptiveActivationFQ2dOffsetShiftScale, QuantType.STATIC)
    #
    return convert_custom_config
