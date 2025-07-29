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


class SoftSigmoidFakeQuantize(torch.ao.quantization.fake_quantize.FakeQuantize):
    """
    A custom quantization class that extends FakeQuantize.
    This class can be used to define specific quantization behavior.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Temperature parameter for sigmoid approximation
        self.temperature = 6.0

    def update_temperature(self, t):
        self.temperature = t

    def soft_round(self, x):
        '''
        Smooth quantization using temperature-controlled approximation of round
        '''
        delta = x - self.floor_ste(x)
        soft_delta = torch.sigmoid(self.temperature * (delta - 0.5))
        y = self.floor_ste(x) + soft_delta
        return y

    def hard_round(self, x):
        y = torch.round(x)
        return y

    # def propagate_quant_ste(self, x, y):
    #     # this works functionally as STE, but exports an onnx graph containing
    #     # all the operators used to compute y as well
    #     # out = x + (y - x).detach()
    #     #
    #     # this is another way of doing STE. in this case the operators used to generate y are skipped from onnx graph
    #     out = x.clone()
    #     out.data = y.data
    #     return out
    #
    # def floor_ste(self, x):
    #     return self.propagate_quant_ste(x, torch.floor(x))

    def floor_ste(self, x):
        # Smart way to do a STE, using detach we can remove non-differentiable function from computation graph
        # Above is a more generic approach to do so for any kind of non-differentiable function
        return x + (torch.floor(x)-x).detach()

    def forward(self, X: torch.Tensor):
        Y = super().forward(X)

        if self.training:
            # Apply smooth quantization
            if self.is_per_channel:
                scale = self.scale.clone().unsqueeze(-1)
                zero_point = self.zero_point.clone().unsqueeze(-1)
            else:
                scale = self.scale.clone()
                zero_point = self.zero_point.clone()
            # Simulate QDQ using soft round
            data_flatten = X.reshape(X.shape[0], -1)
            data_quantized = (data_flatten / scale) + zero_point
            data_soft_quant = self.soft_round(data_quantized)
            data_soft_clamped = torch.clamp(data_soft_quant, min=self.quant_min, max=self.quant_max)
            data_soft_dequantized = (data_soft_clamped - zero_point) * scale
            quantized_data_soft = data_soft_dequantized.reshape(X.shape)
            output = quantized_data_soft
        else:
            output = Y

        return output
    
class SoftTanhFakeQuantize(torch.ao.quantization.FakeQuantize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = 1.0

    def floor_ste(self, x):
        return x + (torch.floor(x) - x).detach()
    
    def update_temperature(self, temperature):
        self.temperature = temperature

    def soft_round(self, x, temperature, eps=1e-3):
        """
        Differentiable approximation to `round` (PyTorch version).

        Args:
            x: torch.Tensor. Inputs to the rounding function.
            alpha: float or torch.Tensor. Controls smoothness of the approximation.
            eps: float. Threshold below which `soft_round` will return identity.

        Returns:
            torch.Tensor
        """
        # Ensure alpha is at least eps to avoid numerical issues and NaNs.
        # This is important for the gradient of torch.where below.
        import warnings
        from torch.jit import TracerWarning
        warnings.filterwarnings("ignore", category=TracerWarning)
        temperature_bounded = torch.maximum(
            torch.tensor(temperature, dtype=x.dtype, device=x.device),  # Convert alpha to tensor if needed
            torch.tensor(eps, dtype=x.dtype, device=x.device)     # Minimum threshold
        )

        # Compute the midpoint between two integers (e.g., 2.5 for x in [2,3))
        m = self.floor_ste(x) + 0.5
        # Compute the residual distance from x to the midpoint
        r = x - m
        # Compute a scaling factor for the tanh output, which depends on alpha
        z = torch.tanh(temperature_bounded / 2.0) * 2.0
        # The core soft-rounding formula: smoothly interpolate between floor and ceil
        y = m + torch.tanh(temperature_bounded * r) / z
        # For very low alphas, soft_round behaves like identity (no rounding)
        if isinstance(temperature, torch.Tensor):
            # If alpha is a tensor, create a mask where alpha < eps and use x there, otherwise use y
            mask = (temperature < eps)
            return torch.where(mask, x, y)
        else:
            # If alpha is a scalar, just return x if alpha < eps, else y
            return x if temperature < eps else y

    def forward(self, X):
        Y = super().forward(X)

        if self.training:
            #
            if self.is_per_channel:
                scale = self.scale.clone().unsqueeze(-1)
                zero_point = self.zero_point.clone().unsqueeze(-1)
            else:
                scale = self.scale.clone()
                zero_point = self.zero_point.clone()
            # Simulate QDQ using soft round and temperature
            data_flatten = X.view(X.size(0), -1)
            data_quantized = (data_flatten / scale) + zero_point
            data_round = self.soft_round(data_quantized, self.temperature)
            data_quantized = torch.clamp(data_round, self.quant_min, self.quant_max)
            data_dequantized = (data_quantized - zero_point) * scale
            data_dequantized = data_dequantized.view(X.size())
            return data_dequantized
        else:
            return Y