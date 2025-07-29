import torch

class GENERICOffsetScaleShift(torch.nn.Module):
    def __init__(self, offset, mult, shift_mult, scale, zp, quant_min, quant_max, quantize_per_channel=False, use_floor=True, ndim=4, dim=1):
        super().__init__()
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.quantize_per_channel = quantize_per_channel
        self.use_floor = use_floor
        self.scale = scale
        self.zp = zp
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
        y = torch.quantize_per_tensor(y, self.scale, self.zp, torch.quint8)
        return y

class PermBlock(torch.nn.Module):
    def __init__(self, scale, zp):
        super().__init__()
        self.scale = scale
        self.zp = zp
    def forward(self, x):
        x = x.dequantize()
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(-1)
        x = torch.quantize_per_tensor(x, self.scale, self.zp, torch.quint8)
        return x