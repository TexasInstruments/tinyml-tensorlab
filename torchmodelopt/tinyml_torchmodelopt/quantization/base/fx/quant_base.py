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

import os
import warnings
import copy
import numpy as np
import torch
from torch.ao.quantization import quantize_fx
from torch.ao.quantization import QConfigMapping

from ... import common
from . import qconfig_types
from . import quant_utils
from . import bias_calibration
from . import fake_quant_types


class TinyMLQuantFxBaseModule(torch.nn.Module):
    """Quantization-aware training (QAT) and post-training quantization (PTQ) wrapper for PyTorch models.

    This module handles the preparation, training, conversion, and export of quantized models
    using PyTorch's FX graph mode quantization framework.

    """

    # ========================================================================
    # Initialization
    # ========================================================================

    def __init__(self, model, total_epochs, qconfig_type=None, example_inputs=None, is_qat=True, backend="qnnpack",
                 num_batch_norm_update_epochs=None, num_observer_update_epochs=None,
                 prepare_qdq=True, bias_calibration_factor=0.0, verbose=True, float_ops=[]):
        """Initialize the TinyML quantization wrapper.

        Args:
            model: Input model to be quantized (torch.nn.Module)
            total_epochs: Number of epochs of training
            qconfig_type: Quantization configuration type. Can be:
                - dict: Custom quantization configuration
                - None: Use default configuration
                - torch.ao.quantization.QConfig: Single QConfig
                - torch.ao.quantization.QConfigMapping: QConfigMapping instance
            example_inputs: Example input tensor for model tracing
            is_qat: If True, use QAT (Quantization-Aware Training); if False, use PTQ
            backend: Quantization backend ('qnnpack', 'fbgemm', 'x86', 'onednn', etc.)
            num_batch_norm_update_epochs:
                - False: Do not freeze batch norm
                - None: Freeze batch norm at half the epochs (default)
                - int: Freeze batch norm at the specified number of epochs
            num_observer_update_epochs:
                - False: Do not freeze observers
                - None: Freeze observers at half the epochs (default)
                - int: Freeze observers at the specified number of epochs
            prepare_qdq: If True, use prepare_qat_fx; if False, use prepare_fx
            bias_calibration_factor: Factor for bias calibration (0.0 = disabled)
            verbose: If False, suppress verbose quantization messages
            float_ops: List of operations to keep in float (not quantized)

        Example qconfig_type for TINPU in F28 devices:
            qconfig_type = {
                'weight': {
                    'bitwidth': 8,
                    'qscheme': torch.per_channel_symmetric,
                    'power2_scale': True,  # True for TINPU, False for Generic
                    'range_max': None,
                    'fixed_range': False
                },
                'activation': {
                    'bitwidth': 8,
                    'qscheme': torch.per_tensor_symmetric,
                    'power2_scale': True,  # True for TINPU, False for Generic
                    'range_max': None,
                    'fixed_range': False
                }
            }
        """
        super().__init__()

        # Validate inputs
        if not total_epochs:
            raise ValueError("total_epochs must be a positive number, got: {}".format(total_epochs))
        if model and not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module instance, got: {}".format(type(model)))

        # Core model and configuration parameters
        self.module = model
        self.is_qat = is_qat
        self.backend = backend
        self.qconfig_type = qconfig_type
        self.example_inputs = example_inputs

        # Epoch and training tracking parameters
        self.total_epochs = total_epochs
        self.num_epochs_tracked = 0
        self.num_batch_norm_update_epochs = num_batch_norm_update_epochs
        self.num_observer_update_epochs = num_observer_update_epochs

        # Quantization-specific parameters
        self.bias_calibration_factor = bias_calibration_factor
        self.bias_calibration_hooks = []
        self.temperature_log_space = np.exp(np.linspace(np.log(1), np.log(500), self.total_epochs))

        # Prepare quantization configuration
        self._prepare_quantization_config(qconfig_type, model, example_inputs, prepare_qdq)

        # Set quantization backend
        self.set_quant_backend(self.backend)

        # Configure PTQ-specific settings
        if not self.is_qat:
            self.disable_backward_for_ptq()
            self._configure_ptq_bias_calibration()

        # Configure verbosity
        if not verbose:
            self._suppress_verbose_messages()

    def _prepare_quantization_config(self, qconfig_type, model, example_inputs, prepare_qdq):
        """Prepare and apply quantization configuration to the model.

        Args:
            qconfig_type: Quantization configuration type
            model: Input model
            example_inputs: Example inputs for model tracing
            prepare_qdq: Whether to use QAT (True) or PTQ (False) preparation
        """
        # Build QConfigMapping from various input formats
        if isinstance(qconfig_type, dict) or qconfig_type is None:
            qconfig_mapping = qconfig_types.get_default_qconfig_mapping(model, qconfig_type)
        elif isinstance(qconfig_type, torch.ao.quantization.QConfig):
            qconfig_mapping = QConfigMapping().set_global(qconfig_type)
        elif isinstance(qconfig_type, torch.ao.quantization.QConfigMapping):
            qconfig_mapping = qconfig_type
        else:
            raise TypeError("qconfig_type must be dict, QConfig, QConfigMapping, or None. "
                          "Got: {}".format(type(qconfig_type)))

        # Prepare model for quantization
        if prepare_qdq:
            self.module = quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs)
        else:
            self.module = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)

    def _configure_ptq_bias_calibration(self):
        """Configure bias calibration for PTQ mode."""
        bias_factor = self.bias_calibration_factor
        for observer in self.module.modules():
            if isinstance(observer, torch.ao.quantization.ObserverBase):
                if hasattr(observer, 'bias_calibration_factor'):
                    bias_factor = max(bias_factor, observer.bias_calibration_factor)
                    observer.bias_calibration_factor = bias_factor
        self.bias_calibration_factor = bias_factor

    def _suppress_verbose_messages(self):
        """Suppress verbose quantization messages."""
        quant_utils.print_once.set_messages({
            'Freezing BN for subsequent epochs': None,
            'Freezing ranges for subsequent epochs': None
        })

    # ========================================================================
    # Configuration Management
    # ========================================================================

    def set_quant_backend(self, backend=None):
        """Set the quantization backend engine.

        Args:
            backend: Backend name ('qnnpack', 'fbgemm', 'x86', 'onednn', etc.)

        Raises:
            RuntimeError: If backend is not supported
        """
        if backend is None:
            backend = self.backend

        if backend not in torch.backends.quantized.supported_engines:
            raise RuntimeError(f"Quantized backend not supported: {backend}. "
                             f"Supported backends: {torch.backends.quantized.supported_engines}")
        torch.backends.quantized.engine = backend

    def load_weights(self, pretrained, *args, strict=True, state_dict_name=None, **kwargs):
        """Load pretrained weights into the model.

        Args:
            pretrained: Path to pretrained weights file
            strict: If False, allow missing or extra keys in state dict
            state_dict_name: Name(s) of state dict key(s) to extract
            *args, **kwargs: Additional arguments passed to torch.load()
        """
        if not isinstance(pretrained, str):
            raise TypeError("pretrained path must be a string, got: {}".format(type(pretrained)))

        data_dict = torch.load(pretrained, *args, **kwargs)

        # Extract state dict from nested structure if needed
        if state_dict_name:
            state_dict_names = state_dict_name if isinstance(state_dict_name, (list, tuple)) else [state_dict_name]
            for s_name in state_dict_names:
                data_dict = data_dict.get(s_name, data_dict)

        self.load_state_dict(data_dict, strict=strict)

    # ========================================================================
    # Core Operations: Forward, Training, Freezing
    # ========================================================================

    def forward(self, *input, **kwargs):
        """Forward pass through the quantized model."""
        return self.module(*input, **kwargs)

    def train(self, mode: bool = True):
        """Set training mode and manage epoch-wise freezing.

        Args:
            mode: If True, set to training mode; if False, set to eval mode

        Returns:
            self: For method chaining
        """
        super().train(mode=mode)

        if mode is True:
            self._train_mode_setup()
        else:
            self._eval_mode_setup()

        return self

    def _train_mode_setup(self):
        """Setup freezing, temperature updates, and bias calibration for training epoch."""
        # Determine freeze settings based on epoch tracking
        enable_freeze_bn = (self.num_batch_norm_update_epochs is not False)
        enable_freeze_observer = (self.num_observer_update_epochs is not False)

        num_bn_freeze_epoch = self.num_batch_norm_update_epochs or ((self.total_epochs // 2) - 1)
        num_observer_freeze_epoch = self.num_observer_update_epochs or ((self.total_epochs // 2) + 1)

        freeze_bn = enable_freeze_bn and ((not self.is_qat) or (self.num_epochs_tracked >= num_bn_freeze_epoch))
        freeze_observers = enable_freeze_observer and (self.num_epochs_tracked >= num_observer_freeze_epoch)

        # Apply freezing
        self.freeze(freeze_bn=freeze_bn, freeze_observers=freeze_observers)

        # Setup bias calibration hooks for PTQ
        if (not self.is_qat) and self.bias_calibration_factor:
            self.bias_calibration_hooks = bias_calibration.insert_bias_calibration_hooks(
                self.module, self.total_epochs, self.num_epochs_tracked)

        # Update temperature for temperature-aware quantizers
        if not freeze_observers:
            self._update_fake_quant_temperatures()

        self.num_epochs_tracked += 1

    def _eval_mode_setup(self):
        """Setup for evaluation mode - finalize freezing and remove calibration hooks."""
        # Final freeze at end of training
        if self.num_epochs_tracked == self.total_epochs:
            self.freeze()

        # Remove bias calibration hooks
        if (not self.is_qat) and self.bias_calibration_factor:
            self.bias_calibration_hooks = bias_calibration.remove_hooks(
                self.module, self.bias_calibration_hooks)

    def _update_fake_quant_temperatures(self):
        """Update temperature for temperature-aware fake quantization modules."""
        for module in self.modules():
            if isinstance(module, (fake_quant_types.SoftTanhFakeQuantize,
                                 fake_quant_types.SoftSigmoidFakeQuantize)):
                temperature = self.temperature_log_space[self.num_epochs_tracked]
                module.update_temperature(temperature)
            if isinstance(module, (fake_quant_types.DBQFakeQuantize)):
                module.update_temperature()

    def freeze(self, freeze_bn=True, freeze_observers=True):
        """Freeze batch norm and/or observer statistics.

        Args:
            freeze_bn: If True, freeze batch norm statistics
            freeze_observers: If True, freeze observer ranges

        Returns:
            self: For method chaining
        """
        if freeze_observers is True:
            self.apply(torch.ao.quantization.disable_observer)
            quant_utils.print_once('Freezing ranges for subsequent epochs')
        elif freeze_observers is False:
            self.apply(torch.ao.quantization.enable_observer)

        if freeze_bn is True:
            self.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            quant_utils.print_once('Freezing BN for subsequent epochs')
        elif freeze_bn is False:
            self.apply(torch.nn.intrinsic.qat.update_bn_stats)

        return self

    def unfreeze(self, freeze_bn=False, freeze_observers=False):
        """Unfreeze batch norm and/or observer statistics.

        Args:
            freeze_bn: If True, freeze batch norm; if False, unfreeze
            freeze_observers: If True, freeze observers; if False, unfreeze

        Returns:
            self: For method chaining
        """
        self.freeze(freeze_bn, freeze_observers)
        return self

    # ========================================================================
    # Model Conversion and Export
    # ========================================================================

    def _apply_dbq_quantization_to_weights(self, model):
        """Apply DBQ quantization logic to weights being quantized by DBQFakeQuantize modules.

        For each DBQFakeQuantize module, finds the parent module whose weight it quantizes
        and applies DBQ thresholding before convert_fx, preserving DBQ semantics.
        """
        for name, dbq_module in model.named_modules():
            if isinstance(dbq_module, fake_quant_types.DBQFakeQuantize):
                # Extract parent module name from DBQFakeQuantize name
                # e.g., "layers.1.weight_fake_quant" → "layers.1"
                if 'weight_fake_quant' in name:
                    parent_name = name.replace('.weight_fake_quant', '')
                    parent_module = dict(model.named_modules()).get(parent_name)

                    if parent_module is not None and hasattr(parent_module, 'weight'):
                        weight = parent_module.weight
                        if weight is not None:
                            # Apply DBQ quantization to the parent module's weight
                            with torch.no_grad():
                                quantized_weight = dbq_module._explicit_quantize_dequantize_dbq(weight)
                                parent_module.weight.data = quantized_weight

    def _is_observed_module(self) -> bool:
        """Check if model is still in observed state (before conversion).

        Returns:
            bool: True if model has not been converted yet
        """
        # Reference: torch/ao/quantization/fx/graph_module.py
        return hasattr(self.module, "meta") and "_observed_graph_module_attrs" in self.module.meta

    def convert(self, model_qconfig_format=None, inplace=False, device='cpu'):
        """Convert quantized model to use integer quantization.

        Converts FX quantized model from observed state to converted state,
        replacing FakeQuantize modules with actual quantization operations.

        Args:
            model_qconfig_format: Output format specification
            inplace: If True, modify in-place; if False, work on a copy
            device: Device to use for conversion ('cpu', 'cuda', etc.)

        Returns:
            self: For method chaining
        """
        self.freeze()

        # Make a copy if not converting in-place
        model = self.module if inplace else copy.deepcopy(self.module)

        # Convert requires CPU model
        model = model.to(torch.device(device))

        # Apply DBQ quantization to weights before convert_fx
        self._apply_dbq_quantization_to_weights(model)

        # Convert model using PyTorch's convert_fx
        self.module = quantize_fx.convert_fx(model)
        return self

    def export(self, example_inputs, filename='model.onnx', opset_version=17, model_qconfig_format=None,
               preserve_qdq_model=True, simplify=True, skipped_optimizers=None, device='cpu', make_copy=True,
               is_converted=True, verbose=False, **export_kwargs):
        """Export quantized model to ONNX format.

        Args:
            example_inputs: Example input tensor(s) for model tracing
            filename: Output ONNX filename
            opset_version: ONNX opset version
            model_qconfig_format: Output quantization format
            preserve_qdq_model: If True, keep QDQ intermediate file (for INT_MODEL format)
            simplify: If True, simplify the exported ONNX model
            skipped_optimizers: Optimizers to skip during simplification
            device: Device to use for export
            make_copy: (Deprecated) unused parameter
            is_converted: If False, convert model before export
            verbose: If True, show warnings about model state
            **export_kwargs: Additional arguments passed to torch.onnx.export()
        """
        # Ensure model is converted before export
        if self._is_observed_module():
            self.convert(device=device, model_qconfig_format=model_qconfig_format)
            model = self.module
        elif not is_converted:
            self.convert(device=device, model_qconfig_format=model_qconfig_format)
            model = self.module
        else:
            model = self.module
            if verbose:
                warnings.warn("Model has already been converted before export. "
                            "Please verify the model was converted correctly.")

        # Export based on output format
        if model_qconfig_format == common.TinyMLModelQConfigFormat.INT_MODEL:
            self._export_int_model(model, example_inputs, filename, opset_version,
                                 preserve_qdq_model, device, export_kwargs)
        else:
            torch.onnx.export(model, example_inputs.to(device=device), filename,
                            opset_version=opset_version, **export_kwargs)

        # Optionally simplify the exported model
        if simplify:
            self._simplify_onnx_model(filename, skipped_optimizers)

    def _export_int_model(self, model, example_inputs, filename, opset_version,
                         preserve_qdq_model, device, export_kwargs):
        """Export model in INT format with QDQ nodes.

        Args:
            model: Model to export
            example_inputs: Example inputs
            filename: Output filename
            opset_version: ONNX opset version
            preserve_qdq_model: Whether to keep QDQ file
            device: Target device
            export_kwargs: Additional ONNX export arguments
        """
        import onnxruntime as ort

        qdq_filename = os.path.splitext(filename)[0] + '_qdq.onnx'

        # Export with QDQ nodes
        torch.onnx.export(model, example_inputs.to(device=device), qdq_filename,
                        opset_version=opset_version, **export_kwargs)

        # Use ONNX Runtime to optimize and convert to INT format
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        session_options.optimized_model_filepath = filename
        ort.InferenceSession(qdq_filename, session_options)

        # Clean up QDQ file if not needed
        if not preserve_qdq_model:
            os.remove(qdq_filename)

    def _simplify_onnx_model(self, filename, skipped_optimizers):
        """Simplify exported ONNX model.

        Args:
            filename: ONNX model filename
            skipped_optimizers: Optimizers to skip
        """
        try:
            import onnx
            from onnxsim import simplify

            onnx_model = onnx.load(filename)
            onnx_model, check = simplify(onnx_model, skipped_optimizers=skipped_optimizers)
            onnx.save(onnx_model, filename)
        except Exception as e:
            print(f"Warning: ONNX model simplification failed "
                  f"(possibly due to multi-process issues): {e}")

    # ========================================================================
    # PTQ-Specific Utilities
    # ========================================================================

    def disable_backward_for_ptq(self):
        """Disable backward pass for PTQ mode.

        Useful for post-training quantization where gradients are not needed.
        Raises an error if backward is accidentally called.
        """
        def backward_hook_with_error(module, grad_input, grad_output):
            raise RuntimeError("backward() should not be called for PTQ - aborting to prevent accidental training")

        self.register_full_backward_hook(backward_hook_with_error)
