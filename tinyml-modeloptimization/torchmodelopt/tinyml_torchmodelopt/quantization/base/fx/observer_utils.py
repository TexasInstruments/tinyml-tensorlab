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


# Imports Torch
import torch
import torch.ao.quantization


class MovingAverageRangeShrinkFastHistogramObserver(torch.ao.quantization.MinMaxObserver):
    RANGE_SHRINK_PERCENTILE_DEFAULT = 0.01
    USE_TORCH_QUANTILE_FOR_RANGE = True
    
    # histogram observer may improve accuracy.
    # default histogram observer in torch.ao.quantization is too slow - so using a custom one
    def __init__(
        self,
        averaging_constant=0.01,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        range_shrink_percentile=RANGE_SHRINK_PERCENTILE_DEFAULT,
        moving_average=True,
        **kwargs
    ) -> None:
        self.averaging_constant = averaging_constant
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            **kwargs
        )
        self.range_shrink_percentile = range_shrink_percentile
        self.moving_average = moving_average
        self.freeze_observer = False

    def forward(self, x_orig):
        if x_orig.numel() == 0 or self.freeze_observer:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if (not self.moving_average) or (min_val == float("inf") and max_val == float("-inf")):
            min_val, max_val = self.histogram_range(x)
        else:
            min_val_cur, max_val_cur = self.histogram_range(x)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    def histogram_range(self, x_orig):
        quantile_l = self.range_shrink_percentile/100.0
        quantile_h = 1.0 - quantile_l
        r_min = torch.quantile(x_orig, quantile_l)
        r_max = torch.quantile(x_orig, quantile_h)
        min_val, max_val = r_min, r_max
        if torch.isnan(min_val) or torch.isnan(max_val):
            return torch.min(x_orig), torch.max(x_orig)
        return min_val, max_val


class RangeShrinkFastHistogramObserver(MovingAverageRangeShrinkFastHistogramObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, moving_average=False, **kwargs)


class MovingAverageRangeShrinkPerChannelHistogramObserver(torch.ao.quantization.PerChannelMinMaxObserver):
    """Per-channel histogram observer with range shrinking for quantization.

    Applies quantile-based range shrinking per-channel to handle outliers and
    improve quantization accuracy while maintaining per-channel granularity.
    """
    RANGE_SHRINK_PERCENTILE_DEFAULT = 0.01

    def __init__(
        self,
        averaging_constant=0.01,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        range_shrink_percentile=RANGE_SHRINK_PERCENTILE_DEFAULT,
        moving_average=True,
        ch_axis=0,
        **kwargs
    ) -> None:
        self.averaging_constant = averaging_constant
        self.range_shrink_percentile = range_shrink_percentile
        self.moving_average = moving_average
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            ch_axis=ch_axis,
            **kwargs
        )
        self.freeze_observer = False

    def forward(self, x_orig):
        if x_orig.numel() == 0 or self.freeze_observer:
            return x_orig

        x = x_orig.detach().to(self.min_val.dtype)
        x_flat = x.view(x.shape[self.ch_axis], -1)

        min_val_cur, max_val_cur = self.histogram_range(x_flat)

        # Initialize buffers if needed (must be done before using old values)
        if self.min_val.numel() == 0 or self.min_val.shape != min_val_cur.shape:
            self.min_val.resize_(min_val_cur.shape)
            self.max_val.resize_(max_val_cur.shape)
            self.min_val.fill_(float('inf'))
            self.max_val.fill_(float('-inf'))

        min_val = self.min_val
        max_val = self.max_val

        if (not self.moving_average) or (torch.all(torch.isinf(min_val)) and torch.all(torch.isinf(max_val))):
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)

        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    def histogram_range(self, x_flat):
        """Compute per-channel histogram-based range using quantiles.

        Args:
            x_flat: Tensor of shape (num_channels, -1) with flattened channel data

        Returns:
            Tuple of (min_val, max_val) per-channel tensors
        """
        quantile_l = self.range_shrink_percentile / 100.0
        quantile_h = 1.0 - quantile_l

        # Compute quantiles per-channel
        r_min = torch.quantile(x_flat, quantile_l, dim=1)
        r_max = torch.quantile(x_flat, quantile_h, dim=1)

        # Handle NaN values - fallback to min/max
        r_min = torch.where(torch.isnan(r_min), torch.min(x_flat, dim=1)[0], r_min)
        r_max = torch.where(torch.isnan(r_max), torch.max(x_flat, dim=1)[0], r_max)

        return r_min, r_max


class RangeShrinkPerChannelHistogramObserver(MovingAverageRangeShrinkPerChannelHistogramObserver):
    """Per-channel histogram observer without moving average."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, moving_average=False, **kwargs)


class EntropyBasedCutoffObserver(torch.ao.quantization.MinMaxObserver):
    """Entropy-based observer for per-tensor quantization.

    Finds optimal quantization range by minimizing information loss (entropy).
    Much faster than KL divergence while maintaining high accuracy through
    efficient entropy-based search.
    """
    NUM_BINS_DEFAULT = 256

    def __init__(
        self,
        averaging_constant=0.01,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        num_bins=NUM_BINS_DEFAULT,
        moving_average=True,
        **kwargs
    ) -> None:
        self.averaging_constant = averaging_constant
        self.num_bins = num_bins
        self.moving_average = moving_average
        super().__init__(
            dtype=dtype, qscheme=qscheme, reduce_range=reduce_range,
            quant_min=quant_min, quant_max=quant_max, **kwargs
        )
        self.freeze_observer = False

    def forward(self, x_orig):
        if x_orig.numel() == 0 or self.freeze_observer:
            return x_orig

        x = x_orig.detach().to(self.min_val.dtype)
        is_first_call = torch.all(torch.isinf(self.min_val)) and torch.all(torch.isinf(self.max_val))

        min_val_cur, max_val_cur = self._compute_entropy_optimal_range(x)

        if is_first_call:
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            if self.moving_average:
                min_val = self.min_val + self.averaging_constant * (min_val_cur - self.min_val)
                max_val = self.max_val + self.averaging_constant * (max_val_cur - self.max_val)
            else:
                min_val = min_val_cur
                max_val = max_val_cur

        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    def _compute_entropy_optimal_range(self, x):
        """Compute optimal range by minimizing entropy loss.

        Uses fast entropy search: scans key percentile boundaries to find
        the range that minimizes information loss when clipping.
        """
        min_x = torch.min(x)
        max_x = torch.max(x)

        if min_x == max_x:
            return min_x, max_x

        # Create histogram
        hist, bin_edges = torch.histogram(x, bins=self.num_bins, range=(min_x.item(), max_x.item()))
        hist = hist / (hist.sum() + 1e-10)

        # Compute cumulative entropy from edges
        # Entropy loss = probability of clipped values × bits needed
        best_entropy_loss = float('inf')
        best_range = (min_x, max_x)

        # Quick scan: check key percentile boundaries
        percentiles = [1, 2, 5, 10, 25]  # Fewer percentiles for speed

        for left_p in percentiles:
            left_idx = max(0, int(self.num_bins * left_p / 100.0))

            for right_p in percentiles:
                right_idx = min(self.num_bins - 1, self.num_bins - int(self.num_bins * right_p / 100.0))

                if left_idx >= right_idx:
                    continue

                range_min = bin_edges[left_idx]
                range_max = bin_edges[right_idx]

                # Compute entropy loss for this range
                entropy_loss = self._compute_entropy_loss(hist, bin_edges, left_idx, right_idx)

                if entropy_loss < best_entropy_loss:
                    best_entropy_loss = entropy_loss
                    best_range = (range_min, range_max)

        return best_range[0], best_range[1]

    def _compute_entropy_loss(self, hist, bin_edges, left_idx, right_idx):
        """Compute entropy loss for a given clipping range.

        Entropy loss = probability of clipped values (information discarded)
        """
        # Probability of values outside the range (being clipped)
        prob_left = hist[:left_idx].sum()
        prob_right = hist[right_idx + 1:].sum()
        entropy_loss = prob_left + prob_right

        return entropy_loss.item()


class EntropyBasedCutoffPerChannelObserver(torch.ao.quantization.PerChannelMinMaxObserver):
    """Entropy-based observer for per-channel quantization.

    Applies entropy-based optimization independently per-channel for
    better granular control and faster computation than KL divergence.
    """
    NUM_BINS_DEFAULT = 256

    def __init__(
        self,
        averaging_constant=0.01,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        num_bins=NUM_BINS_DEFAULT,
        moving_average=True,
        ch_axis=0,
        **kwargs
    ) -> None:
        self.averaging_constant = averaging_constant
        self.num_bins = num_bins
        self.moving_average = moving_average
        super().__init__(
            dtype=dtype, qscheme=qscheme, reduce_range=reduce_range,
            quant_min=quant_min, quant_max=quant_max, ch_axis=ch_axis, **kwargs
        )
        self.freeze_observer = False

    def forward(self, x_orig):
        if x_orig.numel() == 0 or self.freeze_observer:
            return x_orig

        x = x_orig.detach().to(self.min_val.dtype)
        x_flat = x.view(x.shape[self.ch_axis], -1)

        # Compute optimal ranges per-channel
        min_val_cur = torch.zeros(x.shape[self.ch_axis], dtype=x.dtype, device=x.device)
        max_val_cur = torch.zeros(x.shape[self.ch_axis], dtype=x.dtype, device=x.device)

        for ch in range(x.shape[self.ch_axis]):
            ch_data = x_flat[ch]
            min_ch, max_ch = self._compute_entropy_optimal_range(ch_data)
            min_val_cur[ch] = min_ch
            max_val_cur[ch] = max_ch

        # Initialize buffers if needed
        if self.min_val.numel() == 0 or self.min_val.shape != min_val_cur.shape:
            self.min_val.resize_(min_val_cur.shape)
            self.max_val.resize_(max_val_cur.shape)
            self.min_val.fill_(float('inf'))
            self.max_val.fill_(float('-inf'))

        is_first_call = torch.all(torch.isinf(self.min_val)) and torch.all(torch.isinf(self.max_val))

        if is_first_call:
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            if self.moving_average:
                min_val = self.min_val + self.averaging_constant * (min_val_cur - self.min_val)
                max_val = self.max_val + self.averaging_constant * (max_val_cur - self.max_val)
            else:
                min_val = min_val_cur
                max_val = max_val_cur

        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    def _compute_entropy_optimal_range(self, x):
        """Compute optimal range by minimizing entropy loss for single channel."""
        min_x = torch.min(x)
        max_x = torch.max(x)

        if min_x == max_x or x.numel() == 0:
            return min_x, max_x

        hist, bin_edges = torch.histogram(x, bins=self.num_bins, range=(min_x.item(), max_x.item()))
        hist = hist / (hist.sum() + 1e-10)

        best_entropy_loss = float('inf')
        best_range = (min_x, max_x)

        percentiles = [1, 2, 5, 10, 25]

        for left_p in percentiles:
            left_idx = max(0, int(self.num_bins * left_p / 100.0))

            for right_p in percentiles:
                right_idx = min(self.num_bins - 1, self.num_bins - int(self.num_bins * right_p / 100.0))

                if left_idx >= right_idx:
                    continue

                range_min = bin_edges[left_idx]
                range_max = bin_edges[right_idx]

                entropy_loss = self._compute_entropy_loss(hist, left_idx, right_idx)

                if entropy_loss < best_entropy_loss:
                    best_entropy_loss = entropy_loss
                    best_range = (range_min, range_max)

        return best_range[0], best_range[1]

    def _compute_entropy_loss(self, hist, left_idx, right_idx):
        """Compute entropy loss (probability of clipped values)."""
        prob_left = hist[:left_idx].sum()
        prob_right = hist[right_idx + 1:].sum()
        entropy_loss = prob_left + prob_right

        return entropy_loss.item()

def sigmoid_2_step(x):
    """Step function approximation of sigmoid for inference"""
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

class DBQObserver(torch.nn.Module):
	"""Observer that computes quantization statistics from data.

	Supports lazy initialization: statistics computed on first forward pass.
	"""
      
	alpha: torch.Tensor
      
	def __init__(self, quant_min, quant_max, qscheme, num_levels=3, **kwargs):
		super(DBQObserver, self).__init__()
		# Store quantization bounds for compatibility with standard PyTorch observers
		self.quant_min = quant_min
		self.quant_max = quant_max
		self.qscheme = qscheme
		self.num_levels = num_levels

		# Standard observer attributes expected by PyTorch quantization framework
		self.is_dynamic = False  # DBQObserver is static (not adaptive during inference)
		self.eps = 1e-8  # Small epsilon for numerical stability
		# Extract dtype from kwargs or use default (qint8 for quantized int8)
		self.dtype = kwargs.get('dtype', torch.qint8)
		self.reduce_range = kwargs.get('reduce_range', False)  # Standard observer attribute
		self.has_customized_qrange = False  # Flag for customized quantization range
		self.ch_axis = 0  # Channel axis for per-channel quantization (0 = output channels)

		self._initialized = False

	def forward(self, x):
		"""Compute and cache statistics on first call"""
		self._initialize_from_data(x)
		return x

	def _initialize_from_data(self, data):
		"""Initialize beta and alpha statistics from data (per-channel for weights)"""
		if data.numel() == 0:
			return

		# Determine if this is a weight tensor (has batch dim added) or activation tensor
		# Weight tensors are added batch dim in _pre_quantize_dbq_weights before calling observer
		# For weights: expected shapes are [1, out_channels, ...] where ... can be 2D (linear) or higher (conv)
		# For activations: expected shapes are [batch_size, channels, height, width] or similar

		if data.dim() >= 3 and data.size(0) == 1:
			# This looks like a weight tensor with batch dim [1, out_channels, ...]
			# Compute max per output channel (dim 1)
			out_channels = data.size(1)
			if data.dim() == 3:
				# Linear weight: [1, out_channels, in_channels]
				data_reshaped = data.view(1, out_channels, -1)
				max_vals = data_reshaped.abs().amax(dim=[0, 2])  # shape: [out_channels]
			else:
				# Conv weight: [1, out_channels, in_channels, ...]
				data_reshaped = data.view(1, out_channels, -1)
				max_vals = data_reshaped.abs().amax(dim=[0, 2])  # shape: [out_channels]
		else:
			# Activation-like tensor: per-sample max
			data_reshaped = data.view(data.size(0), -1)
			max_vals = data_reshaped.abs().max(dim=1)[0]  # shape: [batch_size]

		beta = 1.0 / (max_vals + 1e-8)
		# For DBQ ternary quantization, alpha represents the scale factor
		# The formula ensures alpha = max_vals / expected_range
		# For DBQ: expected_range = (num_levels - 1) / 2 = 1 for ternary
		alpha = max_vals / ((self.num_levels - 1) / 2 + 1e-8)

		self.register_buffer('beta', beta)
		self.register_buffer('alpha', alpha)
		self._initialized = True

	def reset_min_max_vals(self):
		"""Reset initialized state for re-calibration"""
		self._initialized = False
		# Delete buffers so they can be re-registered in _initialize_from_data
		if hasattr(self, '_buffers') and 'beta' in self._buffers:
			delattr(self, 'beta')
		if hasattr(self, '_buffers') and 'alpha' in self._buffers:
			delattr(self, 'alpha')

	def calculate_qparams(self):
		"""Return quantization scale and zero_point for PyTorch compatibility"""
		if self._initialized and self.alpha is not None:
			# Return per-channel scales as 1D tensor
			scale = self.alpha if self.alpha.dim() > 0 else self.alpha.unsqueeze(0)
			if scale.dim() > 1:
				scale = scale.flatten()
			# Clamp minimum to avoid numerical issues
			scale = torch.clamp(scale, min=1e-5)
			# zero_point must be 1D with same length as scale
			zero_point = torch.zeros(scale.shape[0], dtype=torch.int64)
			return scale, zero_point
		# Default fallback
		return torch.tensor(1.0), torch.tensor(0, dtype=torch.int64)
