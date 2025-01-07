# Imports Torch
import torch
import torch.ao.quantization
from torch.ao.quantization import QConfig, QConfigMapping

def _ceil2_func(x):
    y = torch.pow(2,torch.ceil(torch.log2(x)))
    return y

def _ceil2_tensor(x):
    y = x
    if x.data.abs().sum() != 0:
        x2 = _ceil2_func(torch.abs(x))
        y = torch.sign(x) * x2
    return y

def _propagate_quant_ste(x, y):
    # straight-through estimation (STE) - this is the preferred mode for propagating outputs for quantization
    # the backward gradients uses x (i.e. the input itself) and not y (the output)
    # because the effect of y is completely detached during forward
    # this works functionally as STE, but exports an onnx graph containing
    # all the operators used to compute y as well
    # out = x + (y - x).detach()
    #
    # this is another way of doing STE. in this case the operators used to generate y are skipped from onnx graph
    out = x.clone()
    out.data = y.data
    return out

def ceil2_tensor(x):
    y = _ceil2_tensor(x)
    return _propagate_quant_ste(x, y)

def _adjust_qparams_power2_scale(min_val, max_val, quant_min, quant_max, scale, zero_point, eps):
    r"""Calculates the quantization parameters."""
    # make scale a power of 2 value
    scale = ceil2_tensor(scale)
    scale = torch.max(scale, eps)
    if len(torch.unique(zero_point))>1 or torch.unique(zero_point) not in (0,127):
        # adjust the zero_point based on new scale
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
    return scale, zero_point


class SimplePerChannelWeightObserver(torch.ao.quantization.PerChannelMinMaxObserver):
    def __init__(self, *args, quant_min=-128, quant_max=+127, qscheme=torch.per_channel_symmetric, power2_scale=True, **kwargs):
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, qscheme=qscheme, **kwargs)
        self.power2_scale = power2_scale

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        # weights qparams are always symmetric and this is ensured inside the super class, no need to handle it here.
        scale, zero_point = super()._calculate_qparams(min_val, max_val)
        if self.power2_scale:
            scale, zero_point = _adjust_qparams_power2_scale(
                min_val, max_val, self.quant_min, self.quant_max, scale, zero_point, self.eps)
        return scale, zero_point

    def forward(self, x_orig):
        x_orig = super().forward(x_orig)
        return x_orig


class SimpleActivationObserver(torch.ao.quantization.MovingAverageMinMaxObserver):
    def __init__(self, *args, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, power2_scale=True, **kwargs):
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, qscheme=qscheme, **kwargs)
		# activation quantization cannot use torch.per_channel_symmetric, it has to be torch.per_tensor_symmetric
        self.symmetric = (qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric))
        self.power2_scale = power2_scale

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        if self.symmetric:
            signed_range = torch.min(min_val.detach()).item() < 0.0
            max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
            min_val = -max_abs if signed_range else max_abs * 0.0
            max_val = max_abs

        scale, zero_point = super()._calculate_qparams(min_val, max_val)

        if self.power2_scale:
            scale, zero_point = _adjust_qparams_power2_scale(
                min_val, max_val, self.quant_min, self.quant_max, scale, zero_point, self.eps)
        return scale, zero_point

    def forward(self, x_orig):
        x_orig = super().forward(x_orig)
        return x_orig


def get_default_qconfig_mapping(qconfig):
    
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    
    return qconfig_mapping

def get_default_qconfig():
    
    weight_fake_quant = torch.ao.quantization.fake_quantize.FakeQuantize.with_args(observer=SimplePerChannelWeightObserver, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, dtype=torch.qint8)
    activation_fake_quant = torch.ao.quantization.fake_quantize.FakeQuantize.with_args(observer=SimpleActivationObserver, quant_min=0, quant_max=255, qscheme=torch.per_tensor_symmetric, dtype=torch.quint8)
    
    qconfig = QConfig(weight=weight_fake_quant, activation=activation_fake_quant)
    
    return qconfig