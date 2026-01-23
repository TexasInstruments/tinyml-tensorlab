"""Quantization modules for TinyML TINPU quantization framework.

This module provides custom PyTorch modules for handling quantization-aware training (QAT),
including offset-scale-shift operations, adaptive pooling, and ReLU blocks with quantization.

Classes:
    ReduceSum: Reduces tensor dimensions 2 and 3
    RoundModule: Applies rounding operation
    FloorClip: Floor operation with optional clipping
    AddModule: Adds a constant value
    MultiplyModule: Multiplies by a constant value
    QDQModule: Quantize-Dequantize module
    PermuteModule: Permutes tensor dimensions
    TransposeModule: Transposes tensor dimensions
    AdaptiveAvgPool2d: Adaptive average pooling with quantization
    AddReLUBlock: Addition with ReLU and quantization
    DQAddReLUBlock: Dequantize-Add-ReLU block
    AddReLUWithBias: Addition with bias and ReLU quantization
    TINPUOffsetScaleShift: Core quantization module for offset-scale-shift operation
"""

import torch
from typing import Optional, Tuple
from ...surgery.quant_helper_func import compute_offset_scale_shift


class ReduceSum(torch.nn.Module):
    """Reduces tensor along spatial dimensions (2, 3).
    
    Used for spatial reduction operations in pooling layers.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce sum along dimensions 2 and 3.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Reduced tensor of shape (N, C)
        """
        return torch.sum(x, dim=(2, 3))

class RoundModule(torch.nn.Module):
    """Applies rounding operation to tensor values."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Round tensor values to nearest integer.
        
        Args:
            x: Input tensor
            
        Returns:
            Rounded tensor
        """
        return torch.round(x)

class FloorClip(torch.nn.Module):
    """Applies floor operation followed by optional clipping.
    
    Args:
        min_val: Minimum clipping value (optional)
        max_val: Maximum clipping value (optional)
        
    Raises:
        ValueError: If min_val is set but max_val is None
    """

    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        """Initialize FloorClip module.
        
        Args:
            min_val: Minimum clipping value
            max_val: Maximum clipping value
            
        Raises:
            ValueError: If min_val and max_val are inconsistent
        """
        super().__init__()
        if (min_val is not None) != (max_val is not None):
            raise ValueError("Both min_val and max_val must be provided together or both None")
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply floor and optional clipping.
        
        Args:
            x: Input tensor
            
        Returns:
            Floored tensor, optionally clipped to [min_val, max_val]
        """
        x = torch.floor(x)
        if self.min_val is not None:
            x = torch.clamp(x, self.min_val, self.max_val)
        return x

class AddModule(torch.nn.Module):
    """Adds a constant value to tensor.
    
    Args:
        value: Constant value to add
    """

    def __init__(self, value: float):
        """Initialize AddModule.
        
        Args:
            value: Constant value to add to input tensor
        """
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add constant value to tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with value added
        """
        return x + self.value

class MultiplyModule(torch.nn.Module):
    """Multiplies tensor by a constant value.
    
    Args:
        value: Constant multiplier value
    """

    def __init__(self, value: float):
        """Initialize MultiplyModule.
        
        Args:
            value: Constant value to multiply input tensor by
        """
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply tensor by constant value.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor multiplied by value
        """
        return torch.mul(x, self.value)

class QDQModule(torch.nn.Module):
    """Quantize-Dequantize (QDQ) module.
    
    Performs quantization followed by rounding and dequantization.
    
    Args:
        scale_1: Quantization scale factor
        scale_2: Dequantization scale factor
    """

    def __init__(self, scale_1: float, scale_2: float):
        """Initialize QDQ module.
        
        Args:
            scale_1: Scale factor for quantization step
            scale_2: Scale factor for dequantization step
        """
        super().__init__()
        self.quantize = MultiplyModule(scale_1)
        self.round = RoundModule()
        self.dequantize = MultiplyModule(scale_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantize-round-dequantize sequence.
        
        Args:
            x: Input tensor
            
        Returns:
            QDQ processed tensor
        """
        x = self.quantize(x)
        x = self.round(x)
        x = self.dequantize(x)
        return x   

class PermuteModule(torch.nn.Module):
    """Permutes tensor dimensions according to a permutation.
    
    Args:
        perm: Permutation order for dimensions
    """

    def __init__(self, perm: Tuple[int, ...]):
        """Initialize PermuteModule.
        
        Args:
            perm: Tuple specifying the new dimension order
        """
        super().__init__()
        self.perm = perm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Permute tensor dimensions.
        
        Args:
            x: Input tensor
            
        Returns:
            Permuted tensor
        """
        return x.permute(self.perm)
        
class TransposeModule(torch.nn.Module):
    """Transposes two specified tensor dimensions.
    
    Args:
        dims: Tuple of two dimension indices to transpose
    """

    def __init__(self, dims: Tuple[int, int]):
        """Initialize TransposeModule.
        
        Args:
            dims: Tuple of two dimension indices to transpose
        """
        super().__init__()
        self.dims = dims
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor dimensions.
        
        Args:
            x: Input tensor
            
        Returns:
            Transposed tensor with dims[0] and dims[1] swapped
        """
        return x.transpose(*self.dims)

class AdaptiveAvgPool2d(torch.nn.Module):
    """Adaptive average pooling with quantization support.
    
    Performs adaptive average pooling with quantization-aware handling for
    different output size configurations.
    
    Args:
        pool_module: Base pooling module with output_size attribute
        scale: Quantization scale factor
        zero_point: Zero point for quantization
        activation_bw: Bit-width for activation quantization (default: 8)
        num_bits_scale: Number of bits for scale representation (default: 1)
    """

    def __init__(self, pool_module: torch.nn.Module, scale: float, zero_point: float, 
                 activation_bw: int = 8, num_bits_scale: int = 1):
        """Initialize AdaptiveAvgPool2d.
        
        Args:
            pool_module: Base pooling module with output_size attribute
            scale: Quantization scale factor
            zero_point: Zero point offset
            activation_bw: Bit-width for activation (default: 8)
            num_bits_scale: Scale bit-width (default: 1)
        """
        super().__init__()
        self.reduce_sum = ReduceSum()
        self.round = RoundModule()
        self.pool_module = pool_module
        self.output_size = pool_module.output_size
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive average pooling with quantization.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Pooled tensor with quantization applied
        """
        shape = x.shape
        area = shape[2] * shape[3]
        if(self.output_size[0]*self.output_size[1] == 1):
            offset, mult, shift_mult = compute_offset_scale_shift(self.zero_point, 1 / area, num_bits_scale=self.num_bits_scale)
            oss = TINPUOffsetScaleShift(offset, mult, shift_mult, self.quant_min, self.quant_max, ndim=2, dim=1)
            
            x = self.reduce_sum(x) 
            x = self.round(x)         
            x = oss(x) 
        elif ((shape[2] % self.output_size[0] == 0) and (shape[3] % self.output_size[1] == 0)):
            stride_size = (shape[2] // self.output_size[0], shape[3] // self.output_size[1])
            kernel_size = (shape[2] - (self.output_size[0] - 1) * stride_size[0],
                           shape[3] - (self.output_size[1] - 1) * stride_size[1])
            avg_pool_2d = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride_size)
            total_kernel_area = kernel_size[0] * kernel_size[1]
            if total_kernel_area <= 0:
                raise ValueError(f"Invalid kernel area: {total_kernel_area}. Must be positive.")
            oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(0, 1 / total_kernel_area, num_bits_scale=self.num_bits_scale)
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, 0, 2**self.activation_bw - 1, ndim=2, dim=1)
            # Multiply Module
            mult_module = MultiplyModule(total_kernel_area)
            # Round Module
            round_module = RoundModule()
            # Sequential Module comprising of AvgPool2D, Multiply, Round, OSS
            output_module = torch.nn.Sequential(avg_pool_2d, mult_module, round_module, oss_module)
            x = output_module(x)
        else:
            x = self.pool_module(x)
        return x
    
class AddReLUBlock(torch.nn.Module):
    """Addition with optional ReLU and quantization.
    
    Performs element-wise addition followed by quantization and optional ReLU activation.
    
    Args:
        min_relu_clip: Minimum clipping value for ReLU
        max_relu_clip: Maximum clipping value for ReLU
        scale: Quantization scale factor
        zero_point: Zero point for quantization
        with_relu: Whether to apply ReLU activation (default: False)
        num_bits_scale: Number of bits for scale representation (default: 1)
    """

    def __init__(self, min_relu_clip: float, max_relu_clip: float, scale: float, 
                 zero_point: float, with_relu: bool, num_bits_scale: int = 1):
        """Initialize AddReLUBlock.
        
        Args:
            min_relu_clip: Minimum clipping bound
            max_relu_clip: Maximum clipping bound
            scale: Quantization scale
            zero_point: Quantization zero point
            with_relu: Whether to apply ReLU
            num_bits_scale: Scale bit-width (default: 1)
        """
        super().__init__()
        self.with_relu = with_relu
        if with_relu:
            quant_min, quant_max = -max_relu_clip, max_relu_clip
        else:
            if zero_point:
                quant_min, quant_max = 0, max_relu_clip
            else:
                quant_min, quant_max = -max_relu_clip//2 - 1, max_relu_clip//2
        
        self.num_bits_scale = num_bits_scale
        offset, mult, shift_mult = compute_offset_scale_shift(zero_point, scale, num_bits_scale=num_bits_scale)
        self.oss = TINPUOffsetScaleShift(offset, mult, shift_mult, quant_min, quant_max, ndim=2, dim=1)
        self.relu = torch.nn.ReLU()
        self.clip = torch.nn.Hardtanh(min_relu_clip, max_relu_clip)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add two tensors, quantize, and optionally apply ReLU.
        
        Args:
            x: First input tensor
            y: Second input tensor
            
        Returns:
            Output tensor after addition, quantization, and optional ReLU
        """
        out = x + y
        y = self.oss(out)
        if self.with_relu:
            y = self.relu(y)
            y = self.clip(y)
        return y

class DQAddReLUBlock(torch.nn.Module):
    """Dequantize-Add-ReLU block with optional quantization.
    
    Performs dequantization, addition, quantization, and optional ReLU activation.
    
    Args:
        activation_bw: Bit-width for activation quantization
        add_scale: Scale factor for addition quantization
        is1: Inverse scale for first input dequantization
        is2: Inverse scale for second input dequantization
        zp: Zero point for output quantization
        zp1: Zero point for first input
        zp2: Zero point for second input
        with_relu: Whether to apply ReLU activation
        num_bits_scale: Number of bits for scale representation (default: 1)
    """

    def __init__(self, activation_bw: int, add_scale: float, is1: float, is2: float, 
                 zp: float, zp1: float, zp2: float, with_relu: bool, num_bits_scale: int = 1):
        """Initialize DQAddReLUBlock.
        
        Args:
            activation_bw: Bit-width for activation
            add_scale: Scale for addition operation
            is1: Inverse scale 1
            is2: Inverse scale 2
            zp: Zero point
            zp1: Zero point 1
            zp2: Zero point 2
            with_relu: Apply ReLU
            num_bits_scale: Scale bit-width (default: 1)
        """
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Dequantize, add, quantize, and optionally apply ReLU.
        
        Args:
            x: First input tensor
            y: Second input tensor
            
        Returns:
            Output tensor after dequantization, addition, quantization, and optional ReLU
        """
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
    """Addition with bias, ReLU, and quantization.
    
    Performs addition with bias term, quantization, and optional ReLU activation.
    
    Args:
        bias: Bias term to add
        min_relu_clip: Minimum clipping value
        max_relu_clip: Maximum clipping value
        scale: Quantization scale factor
        zero_point: Zero point for quantization
        with_relu: Whether to apply ReLU activation (default: False)
        num_bits_scale: Number of bits for scale representation (default: 1)
    """

    def __init__(self, bias: float, min_relu_clip: int, max_relu_clip: int, 
                 scale: torch.Tensor, zero_point: torch.Tensor, with_relu: bool = False, 
                 num_bits_scale: int = 1):
        """Initialize AddReLUWithBias.
        
        Args:
            bias: Bias value to add
            min_relu_clip: Minimum clip value
            max_relu_clip: Maximum clip value
            scale: Quantization scale
            zero_point: Zero point for quantization
            with_relu: Apply ReLU (default: False)
            num_bits_scale: Scale bit-width (default: 1)
        """
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
        
        self.num_bits_scale = num_bits_scale
        offset, mult, shift_mult = compute_offset_scale_shift(zero_point, scale, num_bits_scale=num_bits_scale)
        self.oss = TINPUOffsetScaleShift(offset, mult, shift_mult, quant_min, quant_max, ndim=2, dim=1)
        self.relu = torch.nn.ReLU()
        self.clip = torch.nn.Hardtanh(min_relu_clip, max_relu_clip)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Add bias, quantize, and optionally apply ReLU.
        
        Args:
            y: Input tensor
            
        Returns:
            Output tensor after bias addition, quantization, and optional ReLU
        """
        y = self.oss(y)
        if self.with_relu:
            y = self.relu(y)
            y = self.clip(y)
        return y
    
class TINPUOffsetScaleShift(torch.nn.Module):
    """Core quantization module implementing offset-scale-shift operation.
    
    Performs quantization by applying offset, scale, and shift operations with
    optional clipping to quantization bounds. Supports per-channel quantization.
    
    Args:
        offset: Offset tensor for quantization
        mult: Multiplication scale factor
        shift_mult: Shift multiplication factor
        quant_min: Minimum quantization value
        quant_max: Maximum quantization value
        quantize_per_channel: Whether to apply per-channel quantization (default: False)
        use_floor: Whether to use floor operation (default: True)
        ndim: Number of dimensions for buffer reshaping (1, 2, or 4)
        dim: Dimension for reshaping (typically 1)
        
    Raises:
        RuntimeError: If invalid ndim value is provided
    """

    def __init__(self, offset: torch.Tensor, mult: torch.Tensor, shift_mult: torch.Tensor, 
                 quant_min: float, quant_max: float, quantize_per_channel: bool = False, 
                 use_floor: bool = True, ndim: int = 4, dim: int = 1):
        """Initialize TINPUOffsetScaleShift.
        
        Args:
            offset: Offset values for quantization
            mult: Scale multiplication values
            shift_mult: Shift multiplication values
            quant_min: Minimum quantized value
            quant_max: Maximum quantized value
            quantize_per_channel: Enable per-channel quantization
            use_floor: Use floor vs round operation
            ndim: Number of dimensions for buffer shape
            dim: Dimension index for reshaping
            
        Raises:
            RuntimeError: If ndim not in (1, 2, 4) or incompatible with dim
        """
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
            raise RuntimeError(f'Invalid ndim {ndim}: must be 1, 2, or 4')

    def extra_repr(self) -> str:
        """Return extra representation string for module.
        
        Returns:
            String containing module configuration details
        """
        return (f'offset={self.offset}, mult={self.mult}, shift={self.shift_mult}, '
                f'quant_min={self.quant_min}, quant_max={self.quant_max}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply offset-scale-shift quantization.
        
        Performs: y = clamp(floor/round((x + offset) * mult * shift_mult), min, max)
        
        Args:
            x: Input tensor to quantize
            
        Returns:
            Quantized tensor clamped to [quant_min, quant_max]
        """
        y = (x + self.offset) * self.mult
        y = y * self.shift_mult
        if self.use_floor:
            # Floor operation mimics actual shift and bit select in hardware
            y = torch.floor(y).clamp(min=self.quant_min, max=self.quant_max)
        else:
            y = torch.round(y).clamp(min=self.quant_min, max=self.quant_max)
        return y