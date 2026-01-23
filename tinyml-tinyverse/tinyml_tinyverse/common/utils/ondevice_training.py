#################################################################################
# Copyright (c) 2025, Texas Instruments
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
#################################################################################
#
# Few lines are from: https://github.com/pytorch/vision
#
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
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
#################################################################################

"""
On-Device Training Export Utilities
Handles ONNX model splitting, weight flattening, and config generation
for on-device training on TI chips.
"""


import os
import logging
import datetime
import numpy as np
import onnx
import onnx_graphsurgeon as gs

# ============================================================================
# EXCEPTIONS
# ============================================================================

class UnsupportedLayerError(Exception):
    #Exception raised when an unsupported layer is encountered.
    pass

# ============================================================================
# LAYER SUPPORT CONFIGURATION
# ============================================================================

# Main layers that count towards k (have trainable parameters)
MAIN_SUPPORTED_LAYERS = ["Gemm"]

# Other layers that don't count towards k (activation, reshape, etc.)
OTHER_SUPPORTED_LAYERS = ["Reshape", "Relu", "Flatten"]

# ============================================================================
# LAYER SUPPORT CHECK FUNCTIONS
# ============================================================================

def is_main_supported_layer(node):
    """
    Check if ONNX node is a main supported layer (counts towards k).
    
    Args:
        node: ONNX node
        
    Returns:
        bool: True if supported main layer, False otherwise
    """
    return node.op in MAIN_SUPPORTED_LAYERS


def is_other_supported_layer(node):
    """
    Check if ONNX node is an other supported layer (doesn't count towards k).
    
    Args:
        node: ONNX node
        
    Returns:
        bool: True if supported other layer, False otherwise
    """
    return node.op in OTHER_SUPPORTED_LAYERS



# ============================================================================
# ONNX GRAPH SPLITTING
# ============================================================================
def find_split_point(graph, k):
    """
    Traverse ONNX graph backwards from output to find k-th trainable layer.
    
    Args:
        graph: ONNX GraphSurgeon graph
        k: Number of trainable layers from end
        
    Returns:
        split_tensor: The tensor where split should happen
                     (output of frozen part of model, input to trainable part of model)
                     
    Raises:
        ValueError: If cannot find k trainable layers
        UnsupportedLayerError: If unsupported layer is encountered
    """
    logger = logging.getLogger("root.find_split_point")
    
    # Start from output
    output_tensor = graph.outputs[0]
    current_tensor = output_tensor
    trainable_count = 0
    visited_nodes = []
    
    logger.info(f" Searching for split point (k={k} trainable layers from end)")
    
    # Traverse backwards
    while trainable_count < k:
        # Check if we can go further back
        if len(current_tensor.inputs) == 0:
            logger.error(f"Reached graph input, only found {trainable_count} trainable layers")
            logger.error(f"Visited nodes: {[n.name for n in visited_nodes]}")
            raise ValueError(f"Could not find {k} trainable layers. Graph only has {trainable_count}.")
        
        # Get producer node (node that creates this tensor)
        producer_node = current_tensor.inputs[0]
        visited_nodes.append(producer_node)
        
        # Check if supported
        if not is_main_supported_layer(producer_node) and not is_other_supported_layer(producer_node):
            error_msg = f"{producer_node.name} (op: {producer_node.op}) is not supported for on-device training"
            logger.error(error_msg)
            raise UnsupportedLayerError(error_msg)
        
        # Count if main layer
        if is_main_supported_layer(producer_node):
            trainable_count += 1
            if trainable_count == k:
                # Found the k-th trainable layer!
                # Split point is the INPUT to this layer
                split_tensor = producer_node.inputs[0]
                logger.info(f" Split point found: {split_tensor.name}")
                return split_tensor
        
        # Move to previous tensor (input of current producer)
        current_tensor = producer_node.inputs[0]
        
        
        
def extract_frozen_subgraph(graph, split_tensor):
    """
    Create frozen subgraph (from input to split point).
    
    Args:
        graph: Original ONNX GraphSurgeon graph
        split_tensor: Tensor where split happens (becomes output of frozen part)
        
    Returns:
        frozen_graph: ONNX GraphSurgeon graph for frozen part
    """
    logger = logging.getLogger("root.extract_frozen_subgraph")
    logger.info(f" Creating frozen subgraph (input → {split_tensor.name})")
    
    # Copy the graph
    frozen_graph = graph.copy()
    
    # Get the split tensor in the new graph
    tensor_map = frozen_graph.tensors()
    new_split_tensor = tensor_map[split_tensor.name]
    
    # Set split tensor as the output
    frozen_graph.outputs = [new_split_tensor]
    
    # Cleanup removes all nodes/tensors not needed for this output
    frozen_graph.toposort()
    frozen_graph.cleanup()
    
    logger.info(f" Frozen model created:")
    logger.info(f" Frozen model Inputs: {[inp.name for inp in frozen_graph.inputs]}")
    logger.info(f" Frozen model Outputs: {[out.name for out in frozen_graph.outputs]}")
    
    return frozen_graph
        
        
def extract_trainable_layers(graph, split_tensor):
    """
    Extract metadata for all layers after split point.
    Traverses from split_tensor to output.
    
    Args:
        graph: Original ONNX graph
        split_tensor: Starting point (input to trainable part)
        
    Returns:
        layers: List of layer metadata dicts
        
    Raises:
        UnsupportedLayerError: If encounters unsupported layer
    """
    logger = logging.getLogger("root.extract_trainable_layers")
    logger.info(f" Extracting trainable layers (from {split_tensor.name} → output)")
    
    layers = []
    current_tensor = split_tensor
    layer_index = 0
    
    # Traverse forward from split point to output
    while current_tensor:
        # Check if this is the output (end of graph)
        if current_tensor in graph.outputs:
            break
        
        consumer_node = current_tensor.outputs[0]
        
        # Check if supported
        if not is_main_supported_layer(consumer_node) and not is_other_supported_layer(consumer_node):
            error_msg = f"Layer {consumer_node.name} (op: {consumer_node.op}) is not supported for on-device training"
            logger.error(error_msg)
            raise UnsupportedLayerError(error_msg)
        
        # Parse this node
        layer_info = parse_layer_node(consumer_node)
        
        if layer_info:
            layer_info['index'] = layer_index
            layers.append(layer_info)
            logger.info(f" Layer {layer_index}: {layer_info['type']} ({layer_info['input_size']}→{layer_info['output_size']})")
            layer_index += 1
        
        current_tensor = consumer_node.outputs[0]
    
    logger.info(f" Extracted total {len(layers)} layers including both main and intermediate layers")
    return layers


# ============================================================================
# LAYER PARSING
# ============================================================================

def parse_layer_node(node):
    """
    Parse ONNX node and extract layer information.
    Dispatcher function that calls specific parser based on node type.
    
    Args:
        node: ONNX node
        graph: ONNX graph (needed to access initializers)
        
    Returns:
        dict: Layer metadata, or None if node should be skipped
        
    Raises:
        UnsupportedLayerError: If layer type not supported
    """
    logger = logging.getLogger("root.parse_layer_node")
    
    if node.op == 'Gemm':
        return parse_linear_layer(node)
    elif node.op in ['Reshape', 'Flatten']:
        # These don't need parsing (no parameters, just shape change)
        # Skip them - they're handled implicitly by tensor shapes
        return None
    elif node.op == 'Relu':
        return parse_relu_layer(node)
    # TODO: Add support for more layers here
    # elif node.op == 'Conv':
    #     return parse_conv_layer(node, graph)
    # elif node.op == 'BatchNormalization':
    #     return parse_batchnorm_layer(node, graph)
    else:
        # Unknown/unsupported layer
        error_msg = f"Layer type '{node.op}' is not supported for on-device training"
        logger.error(error_msg)
        raise UnsupportedLayerError(error_msg)


def parse_linear_layer(node):
    """
    Extract Linear/Gemm layer metadata.
    
    Args:
        node: ONNX Gemm node
        graph: ONNX graph
        
    Returns:
        dict: Layer metadata with weights, bias, sizes
    """
    logger = logging.getLogger("root.parse_linear_layer")
    logger.info(f"  Parsing Linear layer: {node.name}")
    
    if len(node.inputs) < 3:
        raise ValueError(f"Linear layer {node.name} has insufficient inputs")
    
    # node.inputs[0] is the input to linear layer
    # node.inputs[1] is weights
    # node.inputs[2] is bias
    weights = node.inputs[1].values
    bias = node.inputs[2].values
    
    # Check transB attribute
    trans_b = node.attrs.get('transB', 0)
    
    if trans_b == 0:
        # Weights stored as [input_size, output_size]
        # Need to transpose for C format [output_size, input_size]
        input_size, output_size = weights.shape
        weights = weights.T
    else:
        # transB = 1
        # Weights stored as [output_size, input_size]
        # Already in correct format for C
        output_size, input_size = weights.shape
    
    layer_info = {
        'type': 'Linear',
        'name': node.name,
        'input_size': input_size,
        'output_size': output_size,
        'weights': weights,
        'bias': bias,
    }
    
    return layer_info


def parse_relu_layer(node):
    """
    Extract ReLU layer metadata.
    ReLU has no parameters, just need input/output size.
    
    Args:
        node: ONNX Relu node
        graph: ONNX graph
        
    Returns:
        dict: Layer metadata (no weights/bias)
    """
    logger = logging.getLogger("root.parse_relu_layer")
    
    input_tensor = node.inputs[0]
    
    # Get shape from input tensor
    # Shape is typically [batch, features] or [batch, channels, height, width]
    # We need to get total feature size (exclude batch dimension)
    shape = input_tensor.shape
    
    # Exclude batch dimension (first dimension)
    # Calculate total size: product of remaining dimensions
    size = np.prod(shape[1:])
    
    logger.info(f" Parsing ReLu Layer: {node.name}, input size(flattened): {size}")
    
    layer_info = {
        'type': 'ReLU',
        'name': node.name,
        'input_size': size,
        'output_size': size,
        'weights': None,
        'bias': None,
    }
    
    return layer_info


# ============================================================================
# WEIGHT AND BUFFER MANAGEMENT
# ============================================================================

def flatten_weights(layers):
    """
    This function takes layer metadata (which includes weights/biases as numpy arrays)
    and flattens everything into a single contiguous array. This is needed for:
    - Efficient memory layout in C (single array vs multiple allocations)
    - Easy save/load of all weights (single memcpy)
    
    Args:
        layers: List of layer metadata dicts
        
    Returns:
        tuple: (all_weights, offsets, layer_weight_map)
            - all_weights: 1D numpy array containing all weights/biases concatenated
            - offsets: List of integer offsets marking boundaries
            - layer_weight_map: List of tuples (weight_offset_idx, bias_offset_idx)
   
    
    Example:
        layers = [
            {'type': 'Linear', 'weights': np.array([48, 24]), 'bias': np.array([48])},
            {'type': 'ReLU', 'weights': None, 'bias': None},
            {'type': 'Linear', 'weights': np.array([10, 48]), 'bias': np.array([10])}
        ]
        
        all_weights, offsets, weight_map = flatten_weights(layers)
        
        all_weights = [w0_flat..., b0_flat..., w1_flat..., b1_flat...]
        offsets = [0, 1152, 1200, 1680, 1690]
        weight_map = [(0, 1), (-1, -1), (2, 3)]
    """
    
    
    logger = logging.getLogger("root.flatten_weights")
    logger.info(f" Flattening weights for {len(layers)} layers")
    
    all_weights = []
    offsets = [0] 
    layer_weight_map = []
    
    # Process each layer
    for layer_idx, layer in enumerate(layers):
        weight_offset_idx = -1
        bias_offset_idx = -1
        
        # Process weights
        if layer.get('weights') is not None:
            weights = layer['weights']
            weights_flat = weights.flatten()
            weight_offset_idx = len(offsets) - 1
            all_weights.extend(weights_flat)
            offsets.append(len(all_weights))
        
        # Process bias
        if layer.get('bias') is not None:
            bias = layer['bias']
            bias_flat = bias.flatten()
            bias_offset_idx = len(offsets) - 1
            all_weights.extend(bias_flat)
            offsets.append(len(all_weights))
        
        # Record mapping for this layer
        layer_weight_map.append((weight_offset_idx, bias_offset_idx))
  
    all_weights_array = np.array(all_weights, dtype=np.float32)
    
    logger.info(f" Flattening complete:")
    logger.info(f" Total parameters: {len(all_weights_array)}")
    logger.info(f" Offsets: {offsets}")
    logger.info(f" Layer weight map: {layer_weight_map}")
    
    return all_weights_array, offsets, layer_weight_map


def compute_buffer_offsets(layers, frozen_output_size):
    """
    Compute memory offsets for intermediate activation and gradient buffers.
    
    During forward pass, we need to store the output of each layer (activations).
    During backward pass, we need to store gradients for each layer.
    
    Instead of allocating separate arrays for each layer, we use a single flat
    pool with offsets (same strategy as weight flattening).
    
    Buffer layout:
        buffer[0]: Output from frozen model (input to first trainable layer)
        buffer[1]: Output from layer 0
        buffer[2]: Output from layer 1
        ...
        buffer[n]: Output from last layer (final output)
    
    Args:
        layers: List of layer metadata dicts, each containing 'output_size'
        frozen_output_size: Size of the frozen model's output (input to trainable part)
    
    Returns:
        tuple: (buffer_offsets, total_size)
            - buffer_offsets: List of integer offsets [0, size0, size0+size1, ...]
            - total_size: Total buffer size needed (sum of all sizes)
    
    Example:
        layers = [
            {'type': 'Linear', 'output_size': 32},
            {'type': 'ReLU', 'output_size': 32},
            {'type': 'Linear', 'output_size': 10}
        ]
        frozen_output_size = 24
        
        offsets, total = compute_buffer_offsets(layers, 24)
        
        # offsets = [0, 24, 56, 88, 98]
        #            ^   ^   ^   ^   ^
        #            |   |   |   |   └─ End (total size)
        #            |   |   |   └───── After layer 2 (32+32+10)
        #            |   |   └───────── After layer 1 (32+32)
        #            |   └───────────── After layer 0 (32)
        #            └───────────────── Start (frozen output)
        
        # total = 98 floats needed
    """
    logger = logging.getLogger("root.compute_buffer_offsets")
    logger.info(f" Computing buffer offsets for {len(layers)} layers")
    logger.info(f" Frozen output size: {frozen_output_size}")
    
    # Collect all buffer sizes
    # First buffer is frozen model output
    sizes = [frozen_output_size]  
    
    for layer_idx, layer in enumerate(layers):
        output_size = layer.get('output_size')
        sizes.append(output_size)
    
    # Compute cumulative offsets
    offsets = [0]
    cumulative = 0
    for size in sizes:
        cumulative += size
        offsets.append(cumulative)
    
    total_size = offsets[-1]
    
    logger.info(f" Buffer computation completed. Number of buffers: {len(sizes)}, Offsets: {offsets}")
    return offsets, total_size



# ============================================================================
# CONFIG FILE GENERATION
# ============================================================================

def generate_source_file(all_weights, output_dir):
    """
    Generate trainable_model_config.c
    
    Contains:
        Weight data definitions (ALL_WEIGHTS with actual values)
        Buffer arrays (INTERMEDIATE_BUFFERS, GRADIENT_BUFFERS)
        Conditional weight gradients
    
    Args:
        all_weights: Flatenned weights of all layers
        output_dir : Output dir to save the source file
    """
    logger = logging.getLogger("root.generate_source_file")
    
    # Start building content
    content = []
    
    # ========================================================================
    # SECTION 1: Include Header
    # ========================================================================
    
    content.append("""// ============================================================================
// trainable_model_config.c
// AUTO-GENERATED by TI Tiny ML ModelMaker
// Weight and buffer data definitions
// ============================================================================

#include "trainable_model_config.h"

""")
    
    # ========================================================================
    # SECTION 2: Weight Data
    # ========================================================================
    
    content.append("""// ============================================================================
// WEIGHT STORAGE - Definitions
// ============================================================================

// Initial weights from trained model
#pragma DATA_SECTION(ALL_WEIGHTS, "trainable_parameters")
float ALL_WEIGHTS[TOTAL_PARAMS] = {
""")
    
    # Format weight array
    indent = 4
    values_per_line = 8
    lines = []
    indent_str = ' ' * indent
    
    for i in range(0, len(all_weights), values_per_line):
        chunk = all_weights[i:i + values_per_line]
        formatted_values = [f"{val:.8f}f" for val in chunk]
        line = indent_str + ', '.join(formatted_values)
        # Add comma if not last line
        if i + values_per_line < len(all_weights):
            line += ','
        
        lines.append(line)
    
    all_weights_str = '\n'.join(lines)
    content.append(all_weights_str)
    
    content.append("""
};

// Best weights storage 
#pragma DATA_SECTION(ALL_BEST_WEIGHTS, "trainable_best_weights")
float ALL_BEST_WEIGHTS[TOTAL_PARAMS];

""")
    
    # ========================================================================
    # SECTION 3: Buffer Data
    # ========================================================================
    
    content.append("""// ============================================================================
// BUFFER STORAGE - Definitions
// ============================================================================

// Intermediate activation buffers (for forward pass)
#pragma DATA_SECTION(INTERMEDIATE_BUFFERS, "intermediate_buffers")
float INTERMEDIATE_BUFFERS[TOTAL_INTERMEDIATE_BUFFER_SIZE];

// Gradient buffers (for backward pass)
#pragma DATA_SECTION(GRADIENT_BUFFERS, "gradient_buffers")
float GRADIENT_BUFFERS[TOTAL_GRADIENT_BUFFER_SIZE];

// Weight gradient accumulators (conditional - only for batch training)
#if USE_GRADIENT_ACCUMULATION
#pragma DATA_SECTION(ALL_WEIGHT_GRADS, "trainable_weight_grads")
float ALL_WEIGHT_GRADS[TOTAL_PARAMS];
#endif

""")
    
    # ========================================================================
    # Write to file
    # ========================================================================
    
    full_content = ''.join(content)
    output_path = os.path.join(output_dir, 'trainable_model_config.c')
    
    with open(output_path, 'w') as f:
        f.write(full_content)
    
    return output_path


def generate_header_file(layers, all_weights, weight_offsets, layer_weight_map,
                        buffer_offsets, total_buffer_size, split_tensor, 
                        task_config, model_name, output_dir):
    """
    Generate trainable_model_config.h
    
    Contains:
        Header guard and includes
        All #defines (sizes, enums, hyperparameters)
        typedef for LayerParams_t
        Const config arrays (LAYER_PARAMS_INIT, BUFFER_OFFSETS)
        extern declarations for data arrays
    
    Args:
        layers: List of layer metadata dicts
        all_weights: Flattened weight array (for size calculation)
        weight_offsets: Weight offset list
        layer_weight_map: Layer to weight mapping
        buffer_offsets: Buffer offset list
        total_buffer_size: Total buffer size
        split_tensor: Output of frozen model and input to trainable model
        task_config: Dict with task_type, loss_function
        args: Training arguments (lr, batch_size, etc.)
        output_dir: Where to write file
    """
    
    # Extract key values
    num_layers = len(layers)
    total_params = len(all_weights)
    frozen_output_size = np.prod(split_tensor.shape[1:])
    final_output_size = layers[-1]['output_size']
    #Configurable from header file in CCS project example
    train_batch_size = 1
    val_batch_size = 1
    lr = 0.0001
    
    # Build layer sequence string
    layer_seq = []
    for layer in layers:
        if layer['type'] == 'Linear':
            layer_seq.append(f"Linear({layer['input_size']}→{layer['output_size']})")
        elif layer['type'] == 'ReLU':
            layer_seq.append(f"ReLU({layer['input_size']})")
    layer_seq_str = " → ".join(layer_seq)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building content
    content = []
    
    # ========================================================================
    # SECTION 1: Header Comment & Guard
    # ========================================================================
    
    content.append(f"""// ============================================================================
// trainable_model_config.h
// AUTO-GENERATED by TI TinyML ModelMaker
//
// Model: {model_name}
// Generated: {timestamp}
// Trainable layers: {num_layers}
// Trainable model Architecture: {layer_seq_str}
// ============================================================================

#ifndef TRAINABLE_MODEL_CONFIG_H
#define TRAINABLE_MODEL_CONFIG_H

#include <stdint.h>

""")
    
    # ========================================================================
    # SECTION 2: Defines - Sizes
    # ========================================================================
    
    content.append(f"""// ============================================================================
// MODEL ARCHITECTURE - Sizes
// ============================================================================

#define NUM_TRAINABLE_LAYERS {num_layers}
#define FROZEN_OUTPUT_SIZE {frozen_output_size}
#define FINAL_OUTPUT_SIZE {final_output_size}
#define TOTAL_PARAMS {total_params}
#define TOTAL_INTERMEDIATE_BUFFER_SIZE {total_buffer_size}
#define TOTAL_GRADIENT_BUFFER_SIZE {total_buffer_size}

""")
    
    # ========================================================================
    # SECTION 3: Enumerations
    # ========================================================================
    
    content.append("""// ============================================================================
// ENUMERATIONS
// ============================================================================

// Layer types
typedef enum {
    LAYER_TYPE_LINEAR,
    LAYER_TYPE_RELU,
} LayerType_t;

// Task types
typedef enum {
    TASK_TYPE_CLASSIFICATION,
    TASK_TYPE_ANOMALY_DETECTION,
} TaskType_t;

// Loss functions
typedef enum {
    LOSS_FUNCTION_MSE,
    LOSS_FUNCTION_CROSSENTROPY,
} LossFunction_t;

""")
    
    # ========================================================================
    # SECTION 4: Task Configuration
    # ========================================================================
    
    task_type_map = {
        'classification': 'TASK_TYPE_CLASSIFICATION',
        'anomalydetection': 'TASK_TYPE_ANOMALY_DETECTION',
    }
    task_type_enum = task_type_map.get(task_config['task_type'].lower())
    
    loss_map = {
        'MSE': 'LOSS_FUNCTION_MSE',
        'CrossEntropy': 'LOSS_FUNCTION_CROSSENTROPY',
    }
    loss_enum = loss_map.get(task_config['loss_function'])
    
    content.append(f"""// ============================================================================
// TASK CONFIGURATION
// ============================================================================

#define TASK_TYPE {task_type_enum}
#define LOSS_FUNCTION {loss_enum}

""")
    
    # ========================================================================
    # SECTION 5: Training Hyperparameters
    # ========================================================================
    
    content.append(f"""// ============================================================================
// TRAINING HYPERPARAMETERS 
// ============================================================================

#define TRAIN_BATCH_SIZE {train_batch_size}
#define VAL_BATCH_SIZE {val_batch_size}

// Gradient accumulation (conditional based on batch size)
#if TRAIN_BATCH_SIZE > 1
    #define USE_GRADIENT_ACCUMULATION 1
#else
    #define USE_GRADIENT_ACCUMULATION 0
#endif

""")
    
    # ========================================================================
    # SECTION 6: LayerParams_t Typedef
    # ========================================================================
    
    content.append("""// ============================================================================
// LAYER PARAMETER STRUCTURE
// ============================================================================

typedef struct {
    LayerType_t type;          // Layer type
    uint16_t input_size;       // Input size to this layer
    uint16_t output_size;      // Output size from this layer
    
    // Weight/bias information
    uint16_t weight_offset;    // Offset in ALL_WEIGHTS array
    uint16_t weight_count;     // Number of weight parameters
    uint16_t bias_offset;      // Offset in ALL_WEIGHTS array for bias
    uint16_t bias_count;       // Number of bias parameters
    
    // Layer-specific shape information (for indexing)
    union {
        struct {
            uint16_t rows;     // Output size
            uint16_t cols;     // Input size
        } linear;
        
        struct {
            uint16_t out_channels;
            uint16_t in_channels;
            uint16_t kernel_size;
            uint16_t stride;
            uint16_t padding;
        } conv1d;
    } shape;
    
    // Runtime pointers 
    float* weights;            // Points to location in ALL_WEIGHTS
    float* bias;               // Points to location in ALL_WEIGHTS
    
#if USE_GRADIENT_ACCUMULATION
    float* weight_grad;        // Points to location in ALL_WEIGHT_GRADS
    float* bias_grad;          // Points to location in ALL_WEIGHT_GRADS
#endif
    
} LayerParams_t;

""")
    
    # ========================================================================
    # SECTION 7: LAYER_PARAMS_INIT Array
    # ========================================================================
    
    content.append("""// ============================================================================
// LAYER CONFIGURATIONS
// ============================================================================

static const LayerParams_t LAYER_PARAMS_INIT[NUM_TRAINABLE_LAYERS] = {
""")
    
    # Generate each layer config
    layer_inits = []
    for i, layer in enumerate(layers):
        layer_type = layer['type']
        
        # Get weight/bias offsets
        weight_idx, bias_idx = layer_weight_map[i]
        
        if weight_idx >= 0:
            weight_offset = weight_offsets[weight_idx]
            weight_count = weight_offsets[weight_idx + 1] - weight_offsets[weight_idx]
        else:
            weight_offset = 0
            weight_count = 0
        
        if bias_idx >= 0:
            bias_offset = weight_offsets[bias_idx]
            bias_count = weight_offsets[bias_idx + 1] - weight_offsets[bias_idx]
        else:
            bias_offset = 0
            bias_count = 0
        
        # Generate layer-specific init
        if layer_type == 'Linear':
            layer_init = f"""    // Layer {i}: Linear({layer['input_size']} → {layer['output_size']})
    {{
        .type = LAYER_TYPE_LINEAR,
        .input_size = {layer['input_size']},
        .output_size = {layer['output_size']},
        .weight_offset = {weight_offset},
        .weight_count = {weight_count},
        .bias_offset = {bias_offset},
        .bias_count = {bias_count},
        .shape.linear = {{.rows = {layer['output_size']}, .cols = {layer['input_size']}}}
    }}"""
        
        elif layer_type == 'ReLU':
            layer_init = f"""    // Layer {i}: ReLU({layer['input_size']})
    {{
        .type = LAYER_TYPE_RELU,
        .input_size = {layer['input_size']},
        .output_size = {layer['output_size']},
        .weight_offset = 0,
        .weight_count = 0,
        .bias_offset = 0,
        .bias_count = 0
    }}"""
        
        layer_inits.append(layer_init)
    
    # Join all layer inits
    content.append(',\n\n'.join(layer_inits))
    content.append("\n};\n\n")
    
    # ========================================================================
    # SECTION 8: Buffer Offsets Array
    # ========================================================================
    
    offsets_str = ', '.join(map(str, buffer_offsets))
    num_offsets = len(buffer_offsets)
    
    content.append(f"""// ============================================================================
// BUFFER OFFSETS
// ============================================================================

// Buffer offsets for intermediate and gradient buffers
// Format: [frozen_output, layer0_output, layer1_output, ..., final_output]
static const uint16_t BUFFER_OFFSETS[{num_offsets}] = {{
    {offsets_str}
}};

""")
    
    # ========================================================================
    # SECTION 9: Extern Declarations
    # ========================================================================
    
    content.append("""// ============================================================================
// EXTERNAL DECLARATIONS (defined in trainable_model_config.c)
// ============================================================================

// Weight arrays
extern float ALL_WEIGHTS[TOTAL_PARAMS];
extern float ALL_BEST_WEIGHTS[TOTAL_PARAMS];

// Buffer arrays
extern float INTERMEDIATE_BUFFERS[TOTAL_INTERMEDIATE_BUFFER_SIZE];
extern float GRADIENT_BUFFERS[TOTAL_GRADIENT_BUFFER_SIZE];

// Weight gradient accumulators (conditional)
#if USE_GRADIENT_ACCUMULATION
extern float ALL_WEIGHT_GRADS[TOTAL_PARAMS];
#endif

""")
    # ========================================================================
    # SECTION 10: Footer
    # ========================================================================
    
    content.append("#endif // TRAINABLE_MODEL_CONFIG_H\n")
    
    # ========================================================================
    # Write to file
    # ========================================================================
    
    full_content = ''.join(content)
    output_path = os.path.join(output_dir, 'trainable_model_config.h')
    
    with open(output_path, 'w') as f:
        f.write(full_content)
        
    return output_path


# ============================================================================
# MAIN EXPORT FUNCTION
# ============================================================================

def export_for_ondevice_training(model_onnx_path, args):
    """
    Export model for on-device training.
    
    Main orchestrator function that:
    1. Loads ONNX model
    2. Splits into frozen + trainable parts
    3. Saves frozen model for TVM
    4. Generates trainable model configs (.h and .c)
    
    Args:
        model_onnx_path: Path to trained model.onnx
        args: Training arguments object 
    
    Generates:
        output_dir/
        ├── frozen_model/
        │   └── model.onnx          (for TVM compilation)
        |── trainable_model/
        |   └── trainable_model_config.h
        |   └── trainable_model_config.c
    """
    logger = logging.getLogger("root.export_for_ondevice_training")
    
    k = args.trainable_layers_from_last
    output_dir = args.output_dir
    logger.info(f" Trainable layers from end (k): {k}")
    
    # Load ONNX Model
    onnx_model = onnx.load(model_onnx_path)
    graph = gs.import_onnx(onnx_model)
    
    # Split ONNX Graph
    split_tensor = find_split_point(graph, k)
    
    # Extract Frozen Subgraph
    frozen_graph = extract_frozen_subgraph(graph, split_tensor)
    
    # Save Frozen Model
    frozen_model_dir = os.path.join(output_dir, 'frozen_model')
    os.makedirs(frozen_model_dir, exist_ok=True)
    frozen_model_path = os.path.join(frozen_model_dir, 'model.onnx')
    
    # Convert the frozen model fron graph to onnx format
    frozen_onnx = gs.export_onnx(frozen_graph)
    onnx.save(frozen_onnx, frozen_model_path)
    logger.info(f" Frozen model saved: {frozen_model_path}")
    
    # Extract Trainable Layers
    trainable_layers = extract_trainable_layers(graph, split_tensor)
    
    # Flatten Weights
    all_weights, weight_offsets, layer_weight_map = flatten_weights(trainable_layers)
    
    # Compute Buffer Offsets
    frozen_output_size = np.prod(split_tensor.shape[1:])
    buffer_offsets, total_buffer_size = compute_buffer_offsets(trainable_layers, frozen_output_size)
    
    # Determine Task Configuration
    loader_type = getattr(args, 'loader_type')
    
    if loader_type == 'classification':
        task_type = 'classification'
        loss_function = 'CrossEntropy'
    elif loader_type in ['regression', 'anomalydetection']:
        task_type = 'anomalydetection'
        loss_function = 'MSE'
    
    task_config = {
        'task_type': task_type,
        'loss_function': loss_function,
    }
    
    trainable_model_dir =  os.path.join(output_dir, 'trainable_model')
    os.makedirs(trainable_model_dir, exist_ok=True)
    
    # Generate Header File
    header_file_path = generate_header_file(
        trainable_layers,
        all_weights,
        weight_offsets,
        layer_weight_map,
        buffer_offsets,
        total_buffer_size,
        split_tensor,
        task_config,
        args.model,
        trainable_model_dir
    )
    logger.info(f"Generated trainable model config header file at {header_file_path}")

    
    # Generate Source File
    source_file_path = generate_source_file(
        all_weights,
        trainable_model_dir
    )
    logger.info(f"Generated trainable model config source file at {source_file_path}")