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

import torch
import torch.ao.quantization
from typing import Type

import warnings
from torch.jit._trace import TracerWarning

warnings.filterwarnings("ignore", category=TracerWarning)

class SoftSigmoidFakeQuantize(torch.ao.quantization.FakeQuantize):
    """Sigmoid-based fake quantization for differentiable QAT.

    This class implements temperature-controlled sigmoid rounding for smooth
    quantization during quantization-aware training (QAT). The temperature
    parameter controls the smoothness of the rounding approximation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = 6.0

    def update_temperature(self, temperature):
        """Update the temperature parameter for sigmoid approximation."""
        self.temperature = temperature

    def soft_round(self, x):
        """Compute smooth rounding using temperature-controlled sigmoid.

        Args:
            x: Input tensor to round.

        Returns:
            Rounded tensor with smooth gradients for backpropagation.
        """
        floor_x = self.floor_ste(x)
        delta = x - floor_x
        soft_delta = torch.sigmoid(self.temperature * (delta - 0.5))
        return floor_x + soft_delta

    def floor_ste(self, x):
        """Compute floor with straight-through estimator (STE).

        Uses STE to enable gradients through the non-differentiable floor operation
        by replacing it with identity in the backward pass.

        Args:
            x: Input tensor.

        Returns:
            Floor of x with STE for gradient computation.
        """
        return x + (torch.floor(x) - x).detach()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass with soft quantization during training.

        Args:
            X: Input tensor to quantize.

        Returns:
            Quantized tensor (soft quantization during training, hard during eval).
        """
        Y = super().forward(X)

        if self.training:
            # Prepare scale and zero_point with correct dimensions
            if self.is_per_channel:
                scale = self.scale.clone().unsqueeze(-1)
                zero_point = self.zero_point.clone().unsqueeze(-1)
            else:
                scale = self.scale.clone()
                zero_point = self.zero_point.clone()

            # Simulate QDQ (Quantize-Dequantize) with soft rounding
            data_flatten = X.view(X.shape[0], -1)
            data_quantized = (data_flatten / scale) + zero_point
            data_soft_quant = self.soft_round(data_quantized)
            data_soft_clamped = torch.clamp(data_soft_quant, min=self.quant_min, max=self.quant_max)
            data_soft_dequantized = (data_soft_clamped - zero_point) * scale
            return data_soft_dequantized.view(X.shape)
        else:
            return Y
    

class SoftTanhFakeQuantize(torch.ao.quantization.FakeQuantize):
    """Tanh-based fake quantization for differentiable QAT.

    This class implements temperature-controlled tanh rounding for smooth
    quantization during quantization-aware training (QAT). Tanh provides
    better numerical properties than sigmoid for rounding approximation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = 1.0

    def update_temperature(self, temperature):
        """Update the temperature parameter for tanh approximation.

        Args:
            temperature: New temperature value. Higher values make rounding sharper.
        """
        self.temperature = temperature

    def soft_round(self, x, temperature, eps=1e-3):
        """Differentiable approximation to rounding using temperature-scaled tanh.

        Uses tanh to approximate rounding with controllable smoothness. For very low
        temperatures, behaves like identity to avoid numerical issues.

        Args:
            x: Input tensor to round.
            temperature: Controls smoothness of approximation. Higher = sharper rounding.
            eps: Threshold below which soft_round returns identity.

        Returns:
            Rounded tensor with smooth gradients for backpropagation.
        """
        # Ensure temperature is at least eps to avoid numerical issues and NaNs
        temperature_bounded = torch.maximum(
            torch.tensor(temperature, dtype=x.dtype, device=x.device),
            torch.tensor(eps, dtype=x.dtype, device=x.device)
        )

        # Compute midpoint and distance from it
        midpoint = self.floor_ste(x) + 0.5
        distance = x - midpoint

        # Tanh-based rounding: smoothly interpolate between floor and ceil
        # using tanh scaled by temperature
        scaling_factor = torch.tanh(temperature_bounded / 2.0) * 2.0
        y = midpoint + torch.tanh(temperature_bounded * distance) / scaling_factor

        # For very low temperatures, return identity to avoid numerical issues
        if isinstance(temperature, torch.Tensor):
            mask = (temperature < eps)
            return torch.where(mask, x, y)
        else:
            return x if temperature < eps else y

    def floor_ste(self, x):
        """Compute floor with straight-through estimator (STE).

        Uses STE to enable gradients through the non-differentiable floor operation
        by replacing it with identity in the backward pass.

        Args:
            x: Input tensor.

        Returns:
            Floor of x with STE for gradient computation.
        """
        return x + (torch.floor(x) - x).detach()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass with soft quantization during training.

        Args:
            X: Input tensor to quantize.

        Returns:
            Quantized tensor (soft quantization during training, hard during eval).
        """
        Y = super().forward(X)

        if self.training:
            # Prepare scale and zero_point with correct dimensions
            if self.is_per_channel:
                scale = self.scale.clone().unsqueeze(-1)
                zero_point = self.zero_point.clone().unsqueeze(-1)
            else:
                scale = self.scale.clone()
                zero_point = self.zero_point.clone()

            # Simulate QDQ (Quantize-Dequantize) with temperature-controlled tanh rounding
            data_flatten = X.view(X.size(0), -1)
            data_quantized = (data_flatten / scale) + zero_point
            data_round = self.soft_round(data_quantized, self.temperature)
            data_clamped = torch.clamp(data_round, self.quant_min, self.quant_max)
            data_dequantized = (data_clamped - zero_point) * scale
            return data_dequantized.view(X.size())
        else:
            return Y      

