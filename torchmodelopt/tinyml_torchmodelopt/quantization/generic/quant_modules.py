"""Quantization modules for generic quantization framework.

This module provides PyTorch modules for generic quantization operations including
offset-scale-shift quantization and permutation with quantization support.

Classes:
    GENERICOffsetScaleShift: Generic offset-scale-shift quantization module
    PermBlock: Permutation with quantization module
"""

import torch


class GENERICOffsetScaleShift(torch.nn.Module):
    """Generic offset-scale-shift quantization module.
    
    Performs quantization, offset-scale-shift transformation, and dequantization
    with support for different tensor dimension configurations.
    
    Args:
        q_scale: Quantization scale factor
        q_zp: Quantization zero point
        offset: Offset tensor for transformation
        mult: Multiplication scale factor
        shift_mult: Shift multiplication factor
        scale: Output quantization scale
        zp: Output zero point
        quant_min: Minimum quantization value
        quant_max: Maximum quantization value
        quantize_per_channel: Enable per-channel quantization (default: False)
        use_floor: Use floor vs round operation (default: True)
        ndim: Number of dimensions for buffer reshaping (1, 2, or 4)
        dim: Dimension index for reshaping (default: 1)
        
    Raises:
        RuntimeError: If ndim not in (1, 2, 4)
    """

    def __init__(self, q_scale: float, q_zp: float, offset: torch.Tensor, 
                 mult: torch.Tensor, shift_mult: torch.Tensor, scale: float, 
                 zp: float, quant_min: float, quant_max: float, 
                 quantize_per_channel: bool = False, use_floor: bool = True, 
                 ndim: int = 4, dim: int = 1):
        """Initialize GENERICOffsetScaleShift.
        
        Args:
            q_scale: Input quantization scale
            q_zp: Input quantization zero point
            offset: Offset values for transformation
            mult: Multiplication scale values
            shift_mult: Shift multiplication values
            scale: Output quantization scale
            zp: Output quantization zero point
            quant_min: Minimum quantized value
            quant_max: Maximum quantized value
            quantize_per_channel: Enable per-channel mode
            use_floor: Use floor operation
            ndim: Number of dimensions (1, 2, or 4)
            dim: Dimension index for reshaping
            
        Raises:
            RuntimeError: If ndim not in supported values
        """
        super().__init__()
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.quantize_per_channel = quantize_per_channel
        self.use_floor = use_floor
        self.scale = scale
        self.zp = zp
        self.q_scale = q_scale
        self.q_zp = q_zp
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
        """Apply quantization with offset-scale-shift transformation.
        
        Process:
        1. Quantize input tensor
        2. Dequantize to floating point
        3. Apply offset-scale-shift transformation
        4. Optionally apply floor or round operation
        5. Clamp to quantization bounds
        6. Quantize output
        
        Args:
            x: Input tensor
            
        Returns:
            Quantized tensor after offset-scale-shift transformation
        """
        x = torch.quantize_per_tensor(x, self.q_scale, self.q_zp, torch.quint8)
        x = x.dequantize()
        y = (x + self.offset) * self.mult
        y = y * self.shift_mult
        if self.use_floor:
            # Floor operation mimics actual shift and bit select in hardware
            y = torch.floor(y).clamp(min=self.quant_min, max=self.quant_max)
        else:
            y = torch.round(y).clamp(min=self.quant_min, max=self.quant_max)
        y = torch.quantize_per_tensor(y, self.scale, self.zp, torch.quint8)
        return y

class PermBlock(torch.nn.Module):
    """Permutation with dequantization and re-quantization.
    
    Performs tensor dequantization, dimension permutation, and re-quantization
    with optional dimension expansion.
    
    Args:
        scale: Quantization scale factor
        zp: Quantization zero point
    """

    def __init__(self, scale: float, zp: float):
        """Initialize PermBlock.
        
        Args:
            scale: Quantization scale value
            zp: Quantization zero point value
        """
        super().__init__()
        self.scale = scale
        self.zp = zp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply permutation with quantization.
        
        Process:
        1. Dequantize input tensor
        2. Permute dimensions (0, 2, 1)
        3. Expand dimension at position -1
        4. Re-quantize tensor
        
        Args:
            x: Quantized input tensor
            
        Returns:
            Permuted and re-quantized tensor
        """
        x = x.dequantize()
        x = x.permute(0, 2, 1)  # Swap middle two dimensions
        x = x.unsqueeze(-1)      # Add dimension at end
        x = torch.quantize_per_tensor(x, self.scale, self.zp, torch.quint8)
        return x