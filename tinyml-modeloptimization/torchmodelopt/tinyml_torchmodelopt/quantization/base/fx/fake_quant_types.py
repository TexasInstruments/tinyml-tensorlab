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

class DBQFakeQuantize(torch.ao.quantization.FakeQuantize):
    """DBQ (Differentiable Binary/Ternary Quantization) with explicit QDQ.

    Implements learnable threshold quantization with visible quantize-dequantize
    nodes in the FX graph, enabling pattern fusion during convert_fx.

    Key changes from standard DBQ:
    - Explicit quantize() and dequantize() in forward pass
    - DBQ threshold logic applied in quantization space
    - Weights not pre-quantized in module (like SoftTanh)
    - Maintains identical mathematical behavior and training dynamics

    During training:
      sigmoid(T * (x_norm - τ)) for smooth gradient flow

    During inference:
      step(x_norm - τ) for discrete ternary quantization

    Graph structure (visible to FX):
      get_attr(weight) → call_module(DBQFakeQuantizeExplicit)
                      → [Explicit Quantize → DBQ Logic → Dequantize]
                      → call_module(Conv2D)
    """
    dbq_scales: torch.Tensor
    dbq_offset: torch.Tensor
    dbq_thresholds: torch.Tensor

    def __init__(self, observer: Type, quant_min: int, quant_max: int, dtype: torch.dtype, qscheme, **kwargs):
        super().__init__(observer=observer, quant_min=quant_min, quant_max=quant_max,
                         dtype=dtype, qscheme=qscheme)

        # DBQ-specific quantization setup
        self.num_levels = kwargs.get('num_levels', 3)
        self.T = kwargs.get('T_init', 1.0)
        self.T_scale = kwargs.get('T_scale', 2)

        # Setup quantization levels and scales
        # For ternary (num_levels=3): levels = [-1, 0, 1]
        levels = torch.arange(self.num_levels, dtype=torch.float32) - (self.num_levels - 1) / 2.0
        scales = torch.diff(levels)  # [1, 1] for ternary
        self.register_buffer('dbq_scales', scales)
        self.register_buffer('dbq_offset', torch.sum(scales) / 2.0)  # 1.0 for ternary

        # Learnable thresholds (initialized symmetrically around 0)
        # For ternary: divide [-1, 1] into 3 regions at ±1/3
        self.register_buffer('dbq_thresholds', torch.tensor([-1.0/3.0, 1.0/3.0], dtype=torch.float32))

        # Function to use for threshold crossing (sigmoid or step)
        self._step_fn = torch.sigmoid

    def set_temperature(self, T):
        """Set temperature for sigmoid approximation"""
        self.T = T

    def update_temperature(self):
        """Update temperature according to schedule"""
        self.T = self.T * self.T_scale
        self.T = min(self.T, 2**20)

    def _sigmoid_2_step(self, x):
        """Step function approximation of sigmoid for inference"""
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    
    def train(self, mode=True):
        """Switch between differentiable (sigmoid) and discrete (step) functions"""
        super().train(mode)
        if mode:
            # Training: smooth sigmoid for gradients
            self._step_fn = torch.sigmoid
        else:
            # Inference: discrete step function
            self._step_fn = self._sigmoid_2_step
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass with EXPLICIT quantize-dequantize.

        Makes the quantization pattern visible in FX graph by:
        1. Calling observer.forward() for calibration
        2. Explicitly quantizing with (X / scale) + zero_point
        3. Applying DBQ thresholding logic in quantization space
        4. Explicitly dequantizing with (X_clamped - zero_point) * scale

        This contrasts with earlier DBQ which pre-quantized weights in module.
        """

        data_dequantized = self._explicit_quantize_dequantize_dbq(X)
        data_dequantized = super().forward(data_dequantized)
            
        return data_dequantized


    def _explicit_quantize_dequantize_dbq(self, X: torch.Tensor) -> torch.Tensor:
        """Apply DBQ with explicit quantization steps visible in FX graph.

        Returns:
            Dequantized tensor with same shape as X, with DBQ thresholding applied
            in the quantization space.
        """

        original_shape = X.shape
        data_flatten = X.view(X.size(0), -1)

        # Normalize data to [-1, 1] range using per-sample max
        # Handles variable amplitude signals (motor fault case)
        max_abs_val = data_flatten.abs().max(dim=1, keepdim=True)[0]
        max_abs_val = torch.clamp(max_abs_val, min=1e-8)
        normalized_data = data_flatten / max_abs_val  # Now in [-1, 1]

        # Expand for threshold comparison
        # Shape: (num_thresholds, batch_size, features)
        normalized_expanded = normalized_data.unsqueeze(0).expand(
            self.num_levels - 1, -1, -1
        )

        # Compute threshold distances: each threshold vs normalized data
        # Shape: (num_thresholds, batch_size, features)
        thresholds_expanded = self.dbq_thresholds.view(-1, 1, 1)
        threshold_diffs = normalized_expanded - thresholds_expanded

        # Apply soft/hard assignment based on threshold crossing
        # Training: sigmoid(T * diff) gives smooth [0, 1] assignments
        # Inference: step(T * diff) gives discrete {0, 1} assignments
        # Shape: (num_thresholds, batch_size, features)
        soft_assignments = self._step_fn(threshold_diffs * self.T)

        # Weight by quantization scales and sum across thresholds
        # For ternary: scales = [1, 1]
        scales_expanded = self.dbq_scales.view(-1, 1, 1)
        weighted_sum = torch.sum(soft_assignments * scales_expanded, dim=0)

        # Subtract offset to center around 0
        # For ternary: offset = 1.0, so weighted_sum ∈ {-1, 0, 1}
        quantized = weighted_sum - self.dbq_offset

        # Denormalize back to quantized data range
        data_rounded = max_abs_val * quantized / ((self.num_levels - 1) / 2.0)

        # This dequantizes back to original value range
        data_dequantized = data_rounded.view(original_shape)

        return data_dequantized
