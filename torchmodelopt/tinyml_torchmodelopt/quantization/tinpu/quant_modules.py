import torch
from .quant_helper_func import *

class ReduceSum(torch.nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=(2, 3))

class RoundModule(torch.nn.Module):
    def forward(self, x):
        return torch.round(x)
            
class MultiplyModule(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def forward(self, x):
        return torch.mul(x, self.value)

class AdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self, *args, activation_bw=8, num_bits_scale=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.round = RoundModule()
        self.reduce_sum = ReduceSum()
        self.num_bits_scale = num_bits_scale
        self.activation_bw = activation_bw

    def forward(self, x):
        shape = x.shape
        area = shape[2] * shape[3]

        offset, mult, shift_mult = compute_offset_scale_shift(1, 1 / area, num_bits_scale=self.num_bits_scale)
        oss = TINPUOffsetScaleShift(offset, mult, shift_mult, -(2**(self.activation_bw - 1)), 2**(self.activation_bw - 1) - 1, ndim=2, dim=1)
        
        x = self.reduce_sum(x) 
        x = self.round(x)         
        x = oss(x)     
        return x
    
class AddReLUBlock(torch.nn.Module):
    def __init__(self, min_relu_clip, max_relu_clip, scale, zero_point, with_relu, num_bits_scale=1):
        super().__init__()
        self.with_relu = with_relu
        if with_relu:
            quant_min, quant_max = -max_relu_clip, max_relu_clip
        else:
            quant_min, quant_max = -128, 127
        #
        self.num_bits_scale = num_bits_scale

        offset, mult, shift_mult = compute_offset_scale_shift(zero_point, scale, num_bits_scale=num_bits_scale)
        self.oss = TINPUOffsetScaleShift(offset, mult, shift_mult, quant_min, quant_max)
        self.relu = torch.nn.ReLU()
        self.clip = torch.nn.Hardtanh(min_relu_clip, max_relu_clip)

    def forward(self, x, y):
        out = x + y
        y = self.oss(out)
        if self.with_relu:
            y = self.relu(y)
            y = self.clip(y)
        return y

class TINPUOffsetScaleShift(torch.nn.Module):
    def __init__(self, offset, mult, shift_mult, quant_min, quant_max, quantize_per_channel=False, use_floor=True, ndim=4, dim=1):
        super().__init__()
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.quantize_per_channel = quantize_per_channel
        self.use_floor = use_floor
        if ndim == 4 and dim == 1:
            self.register_buffer('offset', offset.reshape(1, -1, 1, 1))
            self.register_buffer('mult', mult.reshape(1, -1, 1, 1))
            self.register_buffer('shift_mult', shift_mult.reshape(1, -1, 1, 1))
        elif ndim == 2 and dim == 1:
            self.register_buffer('offset', offset.reshape(1, -1))
            self.register_buffer('mult', mult.reshape(1, -1))
            self.register_buffer('shift_mult', shift_mult.reshape(1, -1))
        elif ndim == 1:
            self.register_buffer('offset', offset.reshape(-1))
            self.register_buffer('mult', mult.reshape(-1))
            self.register_buffer('shift_mult', shift_mult.reshape(-1))
        else:
            raise RuntimeError('Invalid dimensions')

    def extra_repr(self):
        return f'offset={self.offset}, mult={self.mult}, shift={self.shift_mult}, quant_min={self.quant_min}, quant_max={self.quant_max}'

    def forward(self, x):
        y = (x + self.offset) * self.mult
        y = y * self.shift_mult
        if self.use_floor:
            # The floor operation mimics the actual shift and bit select in hardware
            y = torch.floor(y).clamp(min=self.quant_min, max=self.quant_max)
        else:
            y = torch.round(y).clamp(min=self.quant_min, max=self.quant_max)
        return y
