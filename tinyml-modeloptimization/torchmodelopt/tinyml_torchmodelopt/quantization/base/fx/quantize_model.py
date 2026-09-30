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
#   contributors may be used to endorse or promote pr
# oducts derived from
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
from torch.ao.quantization import quantize_fx, QConfig


def apply_quantization_to_supported_layers(qconfig_mapping, model):
    """Remove quantization from layers that are not Conv, BatchNorm, Linear, or Pooling.

    This function keeps the global qconfig but disables it for unsupported layer types,
    restricting quantization to commonly quantizable layers. Only leaf (non-container)
    modules are checked - container modules are allowed to propagate quantization to
    their children.

    For ConvTranspose modules, applies ch_axis=1 since their weight layout differs from
    standard Conv modules (output channels are in dimension 1 instead of 0).

    Args:
        qconfig_mapping: QConfigMapping instance to be configured
        model: Model to iterate over

    Returns:
        Modified QConfigMapping with unsupported layers set to None
    """
    supported_types = (
        torch.nn.Identity, torch.nn.Dropout,
        torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
        torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
        torch.nn.Linear,
        torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d,
        torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d,
        torch.nn.AdaptiveAvgPool1d, torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveAvgPool3d,
        torch.nn.AdaptiveMaxPool1d, torch.nn.AdaptiveMaxPool2d, torch.nn.AdaptiveMaxPool3d,
    )

    convtranspose_types = (
        torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
    )

    # Get the global qconfig
    global_qconfig = qconfig_mapping.global_qconfig

    # Recursively check modules and set unsupported leaf modules to None
    for name, module in model.named_modules():
        if name == '':
            continue

        # Only check leaf modules (modules with no children)
        if list(module.children()):
            continue

        # Set qconfig to None for unsupported leaf modules
        if not isinstance(module, supported_types):
            qconfig_mapping.set_module_name(name, None)
        # For ConvTranspose modules, apply qconfig with ch_axis=1
        elif isinstance(module, convtranspose_types) and global_qconfig is not None:
            qconfig_with_ch_axis_1 = _get_qconfig_with_ch_axis(global_qconfig, ch_axis=1)
            qconfig_mapping.set_module_name(name, qconfig_with_ch_axis_1)

    return qconfig_mapping


def _get_qconfig_with_ch_axis(qconfig, ch_axis):
    """Create a new QConfig with modified ch_axis for weight observer.

    Args:
        qconfig: Original QConfig
        ch_axis: Channel axis value to set

    Returns:
        New QConfig with weight observer modified to use the specified ch_axis
    """
    if qconfig is None:
        return None

    # Extract weight and activation fake quantize objects
    weight_fake_quant = qconfig.weight
    activation_fake_quant = qconfig.activation

    # Create new weight fake quantize with ch_axis parameter
    if weight_fake_quant is not None:
        weight_fake_quant = weight_fake_quant.with_args(ch_axis=ch_axis)

    # Create new QConfig with modified weight fake quantize
    return QConfig(weight=weight_fake_quant, activation=activation_fake_quant)


def _has_batch_norm_after_observer(model, observer_node):
    """Check if batch norm exists in the data flow after observer node.

    Traverses the graph from observer node to find batch norm layers,
    accounting for QuantizeLinear/DequantizeLinear operations.

    Args:
        model: The model (GraphModule)
        observer_node: The observer node to check from

    Returns:
        bool: True if batch norm is found in the data flow
    """
    visited = set()
    to_visit = list(observer_node.users.keys())
    named_modules = dict(model.named_modules())

    while to_visit:
        node = to_visit.pop(0)
        if node in visited:
            continue
        visited.add(node)

        if node.op == 'call_module':
            module = named_modules.get(node.target)
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.Identity)):
                return True
            # Continue traversing through non-BN modules
            to_visit.extend(node.users.keys())
        elif node.op == 'call_function':
            # Skip through function calls (e.g., quantize/dequantize operations)
            to_visit.extend(node.users.keys())

    return False


def remove_input_observer_before_bn(model):
    """Remove observer attached to input placeholder node.

    After FX model preparation, an observer is attached to the input placeholder
    (activation_post_process_0). This removes that observer and rewires the graph
    to pass input directly to the first layer, avoiding QuantizeLinear/DequantizeLinear
    on raw inputs.

    Only removes the observer if there is a batch normalization after it.
    The graph is recompiled to maintain consistency after rewiring.

    Args:
        model: The prepared GraphModule
    """
    if not hasattr(model, 'graph'):
        return

    # Find the first placeholder node (model input)
    placeholder_node = None
    for node in model.graph.nodes:
        if node.op == 'placeholder':
            placeholder_node = node
            break

    if placeholder_node is None:
        return

    # Find observer call immediately after placeholder
    observer_node = None
    for user in placeholder_node.users:
        if user.op == 'call_module' and 'activation_post_process' in user.target:
            observer_node = user
            break

    if observer_node is None:
        return

    # Check if observer is followed by batch normalization (possibly through QDQ operations)
    has_bn_after = _has_batch_norm_after_observer(model, observer_node)

    # Only remove observer if batch norm follows it
    if not has_bn_after:
        return

    # Collect users first to avoid modification during iteration
    observer_users = list(observer_node.users.keys())

    # Rewire users of observer to use placeholder directly
    for observer_user in observer_users:
        observer_user.replace_input_with(observer_node, placeholder_node)

    # Remove the observer node from graph
    model.graph.erase_node(observer_node)

    # Remove observer module from model
    if hasattr(model, observer_node.target):
        delattr(model, observer_node.target)

    # Recompile to ensure consistency
    model.graph.lint()
    model.recompile()


def _make_convtranspose_forward_with_fq(orig_forward):
    """Create a forward function that calls weight_fake_quant before conv_transpose.

    This wrapper ensures weight quantization is applied during QAT training.
    """
    def forward_with_fq(self, input, output_size=None):
        weight = self.weight_fake_quant(self.weight) if hasattr(self, 'weight_fake_quant') else self.weight
        import torch.nn.functional as F
        if self.padding_mode != 'zeros':
            raise ValueError(f"Only 'zeros' padding mode supported, got '{self.padding_mode}'")
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation
        )
        return F.conv_transpose2d(
            input, weight, self.bias,
            self.stride, self.padding, output_padding, self.groups, self.dilation
        )
    return forward_with_fq


def remove_activation_observer_after_convtranspose(model):
    """Remove activation observers that are directly followed by ReLU after ConvTranspose2d.

    This removes activation_post_process nodes only when they're between ConvTranspose2d
    and ReLU. Observers without ReLU after them are preserved.

    Pattern to remove: ConvTranspose2d → activation_post_process → ReLU
    Pattern to keep:   ConvTranspose2d → activation_post_process (no ReLU after)

    Args:
        model: The prepared GraphModule
    """
    if not hasattr(model, 'graph'):
        return

    named_modules = dict(model.named_modules())
    nodes_to_remove = []

    def is_convtranspose(node):
        """Check if node is a ConvTranspose module."""
        if node.op != 'call_module':
            return False
        module = named_modules.get(node.target)
        return isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d))

    def is_identity(node):
        """Check if node is Identity."""
        if node.op != 'call_module':
            return False
        module = named_modules.get(node.target)
        return isinstance(module, torch.nn.Identity)

    def is_relu(node):
        """Check if node is ReLU."""
        if node.op != 'call_module':
            return False
        module = named_modules.get(node.target)
        return isinstance(module, torch.nn.ReLU)

    def trace_back_to_convtranspose(node, max_depth=5):
        """Check if this node comes from ConvTranspose2d."""
        if max_depth <= 0:
            return False

        if is_convtranspose(node):
            return True

        if len(node.args) > 0 and isinstance(node.args[0], torch.fx.Node):
            input_node = node.args[0]
            # Continue tracing through Identity and other modules
            if is_identity(input_node) or (input_node.op == 'call_module'):
                module = named_modules.get(input_node.target)
                # Skip BatchNorm nodes
                if module and isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    return False
                return trace_back_to_convtranspose(input_node, max_depth - 1)

        return False

    def has_relu_downstream(node, max_depth=10, visited=None):
        """Check if there's a ReLU node anywhere downstream from this node (before another ConvTranspose)."""
        if visited is None:
            visited = set()

        if max_depth <= 0 or id(node) in visited:
            return False

        visited.add(id(node))

        # Check if this node itself is ReLU
        if is_relu(node):
            return True

        # If we hit another ConvTranspose, stop tracing (don't look beyond)
        if is_convtranspose(node):
            return False

        # Recursively check all users
        if node.users:
            for user in node.users:
                if has_relu_downstream(user, max_depth - 1, visited):
                    return True

        return False

    for node in model.graph.nodes:
        # Find activation_post_process nodes
        if node.op == 'call_module' and 'activation_post_process' in node.target:
            if len(node.args) > 0 and isinstance(node.args[0], torch.fx.Node):
                from_convtranspose = trace_back_to_convtranspose(node.args[0])
                has_relu_after = has_relu_downstream(node)
                # Only remove if: comes from ConvTranspose AND has ReLU somewhere downstream
                if from_convtranspose and has_relu_after:
                    nodes_to_remove.append(node)

    # Rewire and remove the marked observer nodes
    for obs_node in nodes_to_remove:
        # Collect users before removing
        obs_users = list(obs_node.users.keys())
        input_node = obs_node.args[0]

        # Rewire: users of observer now use the previous node output directly
        for user in obs_users:
            user.replace_input_with(obs_node, input_node)

        # Remove the observer node
        model.graph.erase_node(obs_node)

        # Remove the corresponding module if it exists
        if hasattr(model, obs_node.target):
            delattr(model, obs_node.target)

    # Recompile if we made changes
    if nodes_to_remove:
        model.graph.lint()
        model.recompile()


def attach_weight_fake_quant_to_convtranspose(model):
    """After prepare_qat_fx, attach weight_fake_quant to ConvTranspose modules.

    PyTorch's prepare_qat_fx doesn't recognize ConvTranspose2d/3d as QAT-able modules,
    so weight_fake_quant is never created. This function:
    1. Creates and attaches weight_fake_quant using the qconfig
    2. Patches the forward method to call weight_fake_quant on weights

    Args:
        model: The prepared GraphModule (after prepare_qat_fx)
    """
    for name, module in model.named_modules():
        # Only process ConvTranspose2d for now (Conv1d/3d handling similar)
        if not isinstance(module, torch.nn.ConvTranspose2d):
            continue

        # Skip if weight_fake_quant already exists
        if hasattr(module, 'weight_fake_quant'):
            continue

        # Get qconfig (should have been set by apply_quantization_to_supported_layers)
        if not hasattr(module, 'qconfig') or module.qconfig is None:
            continue

        qconfig = module.qconfig

        # Create and attach weight_fake_quant
        if qconfig.weight is not None:
            weight_fake_quant = qconfig.weight()
            module.weight_fake_quant = weight_fake_quant
            # Patch the forward method to call weight_fake_quant
            module.forward = _make_convtranspose_forward_with_fq(module.forward).__get__(module, type(module))


def fold_bn_into_convtranspose_in_model(model):
    """Fold BatchNorm into ConvTranspose2d in the original model (before prepare).

    Finds ConvTranspose2d → BatchNorm2d patterns and:
    1. Folds BN parameters into ConvTranspose2d weights/bias
    2. Removes the BN module (replaces with Identity so indexing doesn't break)

    This prevents prepare_qat_fx from fusing BN+ReLU, allowing the quantization pattern:
    ConvTranspose2d (weight QDQ) → Identity (removed in FX) → ReLU → activation QDQ

    Args:
        model: Original model (not yet prepared)
    """
    def find_convtranspose_bn_patterns():
        """Find all ConvTranspose2d → BatchNorm2d patterns."""
        patterns = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Sequential):
                children = list(module.children())
                for i in range(len(children) - 1):
                    if (isinstance(children[i], torch.nn.ConvTranspose2d) and
                        isinstance(children[i + 1], torch.nn.BatchNorm2d)):
                        patterns.append((module, i, name))
        return patterns

    patterns = find_convtranspose_bn_patterns()

    for seq_module, idx, seq_name in patterns:
        conv = seq_module[idx]
        bn = seq_module[idx + 1]

        # Fold BN parameters into ConvTranspose2d
        bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bn_bias_folded = bn.bias - bn_scale * bn.running_mean

        # Conv weight shape: (in_channels, out_channels, kH, kW)
        # Reshape scale to (1, out_channels, 1, 1) for broadcasting on dim 1
        conv.weight.data = conv.weight.data * bn_scale.view(1, -1, 1, 1)
        if conv.bias is None:
            conv.bias = torch.nn.Parameter(torch.zeros(conv.out_channels, device=conv.weight.device))
        conv.bias.data = conv.bias.data * bn_scale + bn_bias_folded

        # Replace BN with Identity (keeps indexing intact for Sequential)
        seq_module[idx + 1] = torch.nn.Identity()

def get_prepare_custom_config(model):
    """Create prepare_custom_config for models with pre-quantized inputs.

    Treats the model input as already in the quantized domain (scale=1, zero_point=0), so
    no observer/quantize node is inserted on the raw sensor input. The first layer's weight
    and output activation are still quantized normally via its own qconfig.

    """
    prepare_custom_config = {}
    FilterBank_layer_names = ['model.0']

    for name, module in model.named_modules():
        if name in FilterBank_layer_names and type(module).__name__ == 'FilterBank':
            # sets the model input as already quantized
            prepare_custom_config["input_quantized_idxs"] = [0]

    return prepare_custom_config

def prepare_quantized_model(model, qconfig_mapping, example_inputs, is_qat):
    """Prepare model for quantization with FX graph mode.

    Applies quantization configuration to supported layers, prepares the model
    for either QAT or PTQ, and removes input observers.

    Args:
        model: PyTorch model to quantize
        qconfig_mapping: QConfigMapping with quantization configuration
        example_inputs: Example input tensor for model tracing
        is_qat: If True, use prepare_qat_fx; if False, use prepare_fx

    Returns:
        Prepared GraphModule ready for quantization
    """
    # Fold BN into ConvTranspose2d BEFORE prepare to prevent BNReLU fusion
    if is_qat:
        fold_bn_into_convtranspose_in_model(model)

    # Apply quantization only to supported layers
    qconfig_mapping = apply_quantization_to_supported_layers(qconfig_mapping, model)

    # No quantization for the inputs, if it is a FilterBank model
    prepare_custom_config = get_prepare_custom_config(model)

    # Prepare model for quantization
    if is_qat:
        prepared_model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs,
                                                     prepare_custom_config=prepare_custom_config)
        # PyTorch doesn't support ConvTranspose QAT natively, so manually attach weight_fake_quant
        attach_weight_fake_quant_to_convtranspose(prepared_model)
    else:
        prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs,
                                                 prepare_custom_config=prepare_custom_config)

    # Remove input observer to avoid quantization on raw inputs
    remove_input_observer_before_bn(prepared_model)

    # Remove activation observers from ConvTranspose2d AFTER remove_input_observer_before_bn
    # to ensure graph structure is stable
    if is_qat:
        remove_activation_observer_after_convtranspose(prepared_model)

    return prepared_model
