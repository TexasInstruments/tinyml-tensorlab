import torch
from .quant_helper_func import *

class ReduceSum(torch.nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=(2, 3))

class RoundModule(torch.nn.Module):
    def forward(self, x):
        return torch.round(x)

class FloorClip(torch.nn.Module):
    def __init__(self, min_val=None, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    def forward(self, x):
        x = torch.floor(x)
        if self.min_val:
            x = torch.clamp(x, self.min_val, self.max_val)
        return x

class AddModule(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def forward(self, x):
        return x + self.value

class MultiplyModule(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def forward(self, x):
        return torch.mul(x, self.value)

class QDQModule(torch.nn.Module):
    def __init__(self, scale_1, scale_2):
        super().__init__()
        self.q = MultiplyModule(scale_1)
        self.round = RoundModule()
        self.dq = MultiplyModule(scale_2)
    def forward(self, x):
        x = self.q(x)
        x = self.round(x)
        x = self.dq(x)
        return x   

class PermuteModule(torch.nn.Module):
    def __init__(self, perm):
        super().__init__()
        self.perm = perm

    def forward(self, x):
        return x.permute(self.perm)

class AdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self, scale, zero_point, activation_bw=8, num_bits_scale=1):
        super().__init__()
        self.reduce_sum = ReduceSum()
        self.round = RoundModule()
        self.scale = scale
        self.zero_point = zero_point
        self.activation_bw = activation_bw
        self.num_bits_scale = num_bits_scale
        if zero_point == 0:
            self.quant_min = 0
            self.quant_max = 2**(activation_bw) - 1
        else:
            self.quant_min = -2**(activation_bw - 1)
            self.quant_max = 2**(activation_bw - 1) - 1

    def forward(self, x):
        shape = x.shape
        area = shape[2] * shape[3]

        offset, mult, shift_mult = compute_offset_scale_shift(self.zero_point, 1 / area, num_bits_scale=self.num_bits_scale)
        oss = TINPUOffsetScaleShift(offset, mult, shift_mult, self.quant_min, self.quant_max, ndim=2, dim=1)
        
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
            if zero_point:
                quant_min, quant_max = 0, max_relu_clip
            else:
                quant_min, quant_max = -max_relu_clip//2 - 1, max_relu_clip//2
        #
        self.num_bits_scale = num_bits_scale

        offset, mult, shift_mult = compute_offset_scale_shift(zero_point, scale, num_bits_scale=num_bits_scale)
        self.oss = TINPUOffsetScaleShift(offset, mult, shift_mult, quant_min, quant_max, ndim=2, dim=1)
        self.relu = torch.nn.ReLU()
        self.clip = torch.nn.Hardtanh(min_relu_clip, max_relu_clip)

    def forward(self, x, y):
        out = x + y
        y = self.oss(out)
        if self.with_relu:
            y = self.relu(y)
            y = self.clip(y)
        return y

class DQAddReLUBlock(torch.nn.Module):
    def __init__(self, activation_bw, add_scale, is1, is2, zp, zp1, zp2, with_relu, num_bits_scale=1):
        super().__init__()
        self.with_relu = with_relu
        self.zp = zp
        self.zp1 = zp1
        self.zp2 = zp2
        self.num_bits_scale = num_bits_scale
        self.dq1 = MultiplyModule(is1)
        self.dq2 = MultiplyModule(is2)
        self.qdq = QDQModule(1/add_scale, 1)
        self.relu = torch.nn.ReLU()
        min_relu_clip = -2**(activation_bw - 1) if zp else 0
        max_relu_clip = 2**(activation_bw - 1) - 1 if zp else 2**activation_bw - 1
        self.clip = torch.nn.Hardtanh(min_relu_clip, max_relu_clip)

    def forward(self, x, y):
        x = self.dq1(x - self.zp1)
        y = self.dq2(y - self.zp2)
        out = x + y
        out = self.qdq(out)
        y = out + self.zp
        if self.with_relu:
            y = self.relu(y)
            y = self.clip(y)
        return y
    
class AddReLUWithBias(torch.nn.Module):
    def __init__(self, bias, min_relu_clip: int, max_relu_clip: int, scale: torch.Tensor, zero_point: torch.Tensor, with_relu: bool = False, num_bits_scale: int = 1):
        super().__init__()
        self.with_relu = with_relu
        self.bias = bias
        if with_relu:
            quant_min, quant_max = -max_relu_clip, max_relu_clip
        else:
            if zero_point:
                quant_min, quant_max = 0, max_relu_clip
            else:
                quant_min, quant_max = -max_relu_clip//2 - 1, max_relu_clip//2
        #
        self.num_bits_scale = num_bits_scale

        offset, mult, shift_mult = compute_offset_scale_shift(zero_point, scale, num_bits_scale=num_bits_scale)
        self.oss = TINPUOffsetScaleShift(offset, mult, shift_mult, quant_min, quant_max, ndim=2, dim=1)
        self.relu = torch.nn.ReLU()
        self.clip = torch.nn.Hardtanh(min_relu_clip, max_relu_clip)

    def forward(self, y):
        y = self.oss(y)
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