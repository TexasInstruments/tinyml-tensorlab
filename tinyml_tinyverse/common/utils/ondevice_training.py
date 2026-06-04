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
from ast import literal_eval

# ============================================================================
#  Export some sample data for on device training
# ============================================================================

def parse_export_samples_argument(samples_str):
    """
    Parse export_samples_per_class string argument.
    
    Args:
        samples_str: String like '[10,5,5]' or '(10,5,5)'
    
    Returns:
        tuple: (train_samples, val_samples, test_samples)
    
    Raises:
        ValueError: If format is invalid
    """
    
    try:
        samples_list = literal_eval(samples_str)
        if not isinstance(samples_list, (list, tuple)) or len(samples_list) != 3:
            raise ValueError("Must be a list/tuple of 3 integers")
        
        train_samples, val_samples, test_samples = samples_list
        
        # Validate all should be positive integers
        if not all(isinstance(x, int) and x > 0 for x in [train_samples, val_samples, test_samples]):
            raise ValueError("All values must be positive integers")
        
        return train_samples, val_samples, test_samples
        
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid export_samples_per_class format: '{samples_str}'. " f"Expected format: 'ex: [10,5,5]'. Error: {e}")

def sample_training_data(dataset, samples_per_class, num_frame_concat, subset_name='train'):
    """
    Sample raw and preprocessed data for on-device training export.
    
    Handles frame concatenation by reconstructing multi-frame raw data from 
    single-frame X_raw storage. Ensures all frames come from the same source file
    to maintain data correctness.
    
    Args:
        dataset: Dataset object (must have X, X_raw, Y, file_names attributes)
        samples_per_class: Number of samples to export per class
        subset_name: 'train', 'val', or 'test'
    
    Returns:
        tuple: (X_raw_reconstructed, Y)
            - X_raw_reconstructed: numpy array of shape (N, C, W_raw) where W_raw = frame_size * num_frame_concat
            - Y: numpy array of labels, shape (N,)
    """
    logger = logging.getLogger("root.sample_training_data")
    
    if len(dataset.file_names) != len(dataset.X_raw):
        error_msg = f"Array size mismatch: file_names has {len(dataset.file_names)} entries but X has {len(dataset.X_raw)} samples"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    min_valid_index = num_frame_concat - 1
    valid_indices_all = []
    
    for idx in range(len(dataset.X_raw)):
        # Check 1: Minimum index (need enough history)
        if idx < min_valid_index:
            continue
        
        # Check 2: All required frames must be from the SAME file
        current_file = dataset.file_names[idx]
        all_same_file = True
        
        for offset in range(1, num_frame_concat):
            prev_idx = idx - offset
            if dataset.file_names[prev_idx] != current_file:
                all_same_file = False
                break
        
        if all_same_file:
            valid_indices_all.append(idx)
    
    if len(valid_indices_all) == 0:
        error_msg = (
            f"No valid samples available after filtering. "
            f"This may occur if num_frame_concat ({num_frame_concat}) is too large "
            f"or dataset files are too small."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    sampled_indices = []
    sampled_Y = []
    
    num_classes = len(dataset.classes)
    
    for class_idx in range(num_classes):
        # Get all valid indices for this class
        valid_class_indices = [
            idx for idx in valid_indices_all 
            if dataset.Y[idx] == class_idx
        ]
        
        if len(valid_class_indices) == 0:
            logger.warning(
                f"Class {class_idx} ({dataset.inverse_label_map.get(class_idx, 'Unknown')}): "
                f"No valid samples after filtering"
            )
            continue
        
        # Sample from valid indices
        num_available = len(valid_class_indices)
        num_to_sample = min(samples_per_class, num_available)
        
        if num_available < samples_per_class:
            logger.warning(
                f"Class {class_idx} ({dataset.inverse_label_map.get(class_idx, 'Unknown')}): "
                f"Only {num_available} valid samples available, requested {samples_per_class}"
            )
        
        selected_indices = np.random.choice(
            valid_class_indices, size=num_to_sample, replace=False
        )
        
        sampled_indices.extend(selected_indices.tolist())
        sampled_Y.extend([class_idx] * num_to_sample)
    
    # Reconstruct Raw Frames
    sampled_X_raw_reconstructed = []
    
    for sample_num, idx in enumerate(sampled_indices):
        # Reconstruct raw frames by looking back
        reconstructed_frames = []
        for offset in range(num_frame_concat - 1, -1, -1):
            frame_idx = idx - offset
            raw_frame = dataset.X_raw[frame_idx]
            reconstructed_frames.append(raw_frame)
        
        # Shape after stacking: (num_frame_concat, C, W, H)
        reconstructed_frames = np.array(reconstructed_frames)  
        
        # Reshape: (num_frame_concat, C, W) → (C, num_frame_concat, W) → (C, num_frame_concat*W)
        num_channels = reconstructed_frames.shape[1]
        reconstructed_frames = reconstructed_frames.transpose(1, 0, 2, 3) if len(reconstructed_frames.shape) == 4 else reconstructed_frames.transpose(1, 0, 2)
        reconstructed_frames = reconstructed_frames.reshape(num_channels, -1)
        
        sampled_X_raw_reconstructed.append(reconstructed_frames)
    
    sampled_X_raw_reconstructed = np.array(sampled_X_raw_reconstructed)
    sampled_Y = np.array(sampled_Y)
    
    return sampled_X_raw_reconstructed, sampled_Y

def get_pragma_for_device(target_device, variable_name, section_name):
    """
    Get the appropriate pragma/attribute directive for memory section based on device.
    
    Args:
        target_device: Device name (e.g., 'F28P55', 'F29H85')
        variable_name: Variable name (e.g., 'TRAIN_INPUTS')
        section_name: Memory section name (e.g., 'training_data')
    
    Returns:
        str: Pragma directive string
    """
    # C28 devices (F28xxx series)
    if 'F28' in target_device:
        return f'#pragma DATA_SECTION({variable_name}, "{section_name}")'
    # C29 devices (F29xxx series) 
    elif 'F29' in target_device or 'AM13' in target_device :
        return f'__attribute__((section("{section_name}")))'
    else:
        return ''

def generate_data_array(prefix, X_data, Y_data, input_size, target_device):
    """
    Generate C code for one dataset (TRAIN/VALIDATION/TEST).
    
    Args:
        prefix: 'TRAIN', 'VALIDATION', or 'TEST'
        X_data: Input array (N, C, W)
        Y_data: Label array (N,)
        input_size: Flattened input size
        target_device: Target device name
    
    Returns:
        str: C code string
    """
    
    content = []
    
    # Section header
    content.append(f"""// ============================================================================
// {prefix} DATA
// ============================================================================

""")
    
    # Generate INPUTS array
    input_var = f"{prefix}_INPUTS"
    input_section = f"{prefix.lower()}_input_data"
    
    pragma = get_pragma_for_device(target_device, input_var, input_section)
    content.append(f"{pragma}\n")
    content.append(f"const float {input_var}[NUM_{prefix}_SAMPLES][RAW_INPUT_SIZE] = {{\n")
    
    # Write input data 
    for idx, (x, y) in enumerate(zip(X_data, Y_data)):
        x_flat = x.flatten()
        content.append(f"    // Sample {idx} - Class {y}\n")
        content.append("    {")
        
        # Format values (8 per line)
        for i in range(0, len(x_flat), 8):
            chunk = x_flat[i:i+8]
            values = ', '.join([f"{val:.8f}f" for val in chunk])
            if i + 8 < len(x_flat):
                content.append(f"{values}, ")
            else:
                content.append(values)
        
        if idx < len(X_data) - 1:
            content.append("},\n")
        else:
            content.append("}\n")
    
    content.append("};\n\n")
    
    # Generate LABELS array
    label_var = f"{prefix}_LABELS"
    label_section = f"{prefix.lower()}_label_data"
    
    pragma = get_pragma_for_device(target_device, label_var, label_section)
    content.append(f"{pragma}\n")
    content.append(f"const int16_t {label_var}[NUM_{prefix}_SAMPLES] = {{\n")
    
    # Write labels (10 per line)
    content.append("    ")
    for idx, label in enumerate(Y_data):
        content.append(f"{int(label)}")
        if idx < len(Y_data) - 1:
            content.append(", ")
            if (idx + 1) % 10 == 0:
                content.append("\n    ")
    
    content.append("\n};\n\n")
    
    return ''.join(content)

def generate_training_data_header(train_data, val_data, test_data, input_size, num_classes, 
                                   output_dir, target_device):
    """
    Generate ondevice_training_data.h header file.
    
    Args:
        train_data: (X_train, Y_train) tuple
        val_data: (X_val, Y_val) tuple  
        test_data: (X_test, Y_test) tuple
        input_size: Flattened input size
        num_classes: Number of classes
        output_dir: Output directory (trainable_model/)
        target_device: Target device name for conditional pragmas
    
    Returns:
        str: Path to generated header file
    """
    logger = logging.getLogger("root.generate_training_data_header")
    
    X_train, Y_train = train_data
    X_val, Y_val = val_data
    X_test, Y_test = test_data
    
    num_train = len(X_train)
    num_val = len(X_val)
    num_test = len(X_test)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""// ============================================================================
// ondevice_training_data.h
// AUTO-GENERATED by TI TinyML ModelMaker
// 
// Generated: {timestamp}
// Target Device: {target_device}
// Train: {num_train} samples, Val: {num_val} samples, Test: {num_test} samples
// ============================================================================

#ifndef ONDEVICE_TRAINING_DATA_H
#define ONDEVICE_TRAINING_DATA_H

#include <stdint.h>

// ============================================================================
// DATA DIMENSIONS
// ============================================================================

#define NUM_TRAIN_SAMPLES {num_train}
#define NUM_VALIDATION_SAMPLES {num_val}
#define NUM_TEST_SAMPLES {num_test}
#define RAW_INPUT_SIZE {input_size}
#define NUM_CLASSES {num_classes}

// ============================================================================
// EXTERNAL DECLARATIONS
// ============================================================================

extern const float TRAIN_INPUTS[NUM_TRAIN_SAMPLES][RAW_INPUT_SIZE];
extern const int16_t TRAIN_LABELS[NUM_TRAIN_SAMPLES];

extern const float VALIDATION_INPUTS[NUM_VALIDATION_SAMPLES][RAW_INPUT_SIZE];
extern const int16_t VALIDATION_LABELS[NUM_VALIDATION_SAMPLES];

extern const float TEST_INPUTS[NUM_TEST_SAMPLES][RAW_INPUT_SIZE];
extern const int16_t TEST_LABELS[NUM_TEST_SAMPLES];

#endif // ONDEVICE_TRAINING_DATA_H
"""
    
    # Write to file
    header_path = os.path.join(output_dir, 'ondevice_training_data.h')
    with open(header_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Generated ondevice_training_data.h header file: {header_path}")
    return header_path

def generate_training_data_source(train_data, val_data, test_data, input_size, 
                                   output_dir, target_device):
    """
    Generate ondevice_training_data.c source file with actual data.
    
    Args:
        train_data: (X_train, Y_train) tuple
        val_data: (X_val, Y_val) tuple  
        test_data: (X_test, Y_test) tuple
        input_size: Flattened input size
        output_dir: Output directory 
        target_device: Target device name (e.g., 'F28P55', 'F29H85')
    
    Returns:
        str: Path to generated source file
    """
    
    logger = logging.getLogger("root.generate_training_data_source")
    
    X_train, Y_train = train_data
    X_val, Y_val = val_data
    X_test, Y_test = test_data
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = []
    
    content.append(f"""// ============================================================================
// ondevice_training_data.c
// AUTO-GENERATED by TI TinyML ModelMaker
// 
// Generated: {timestamp}
// Target Device: {target_device}
// ============================================================================

#include "ondevice_training_data.h"

""")
    
    # Generate TRAIN data
    content.append(generate_data_array('TRAIN', X_train, Y_train, input_size, target_device))
    
    # Generate VAL data
    content.append(generate_data_array('VALIDATION', X_val, Y_val, input_size, target_device))
    
    # Generate TEST data
    content.append(generate_data_array('TEST', X_test, Y_test, input_size, target_device))
    
    # Write to file
    source_path = os.path.join(output_dir, 'ondevice_training_data.c')
    with open(source_path, 'w') as f:
        f.write(''.join(content))
    
    logger.info(f"Generated ondevice_training_data.c source file: {source_path}")
    return source_path

def export_training_data(dataset_train, dataset_val, dataset_test, args):
    """
    Export training data for on-device training.
    
    Args:
        dataset_train: Training dataset
        dataset_val: Validation dataset  
        dataset_test: Test dataset
        args: Training arguments
        
    Generates:
        output_dir/trainable_model/
        ├── ondevice_training_data.h
        └── ondevice_training_data.c
    """
    logger = logging.getLogger("root.export_training_data")
    
    output_dir = args.output_dir
    target_device = args.target_device
    num_frame_concat = int(args.num_frame_concat)
    
    # Parse samples_per_class
    train_samples, val_samples, test_samples = parse_export_samples_argument(args.export_samples_per_class)
    logger.info(f"Export configuration: Train={train_samples}, Val={val_samples}, Test={test_samples} samples per class") 
    
    X_train, Y_train = sample_training_data(dataset_train, train_samples, num_frame_concat, 'train')
    X_val, Y_val = sample_training_data(dataset_val, val_samples, num_frame_concat,'val')
    X_test, Y_test = sample_training_data(dataset_test,  test_samples, num_frame_concat, 'test')
    
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    Y_train = Y_train[shuffle_idx]
    
    shuffle_idx = np.random.permutation(len(X_val))
    X_val = X_val[shuffle_idx]
    Y_val = Y_val[shuffle_idx]

    shuffle_idx = np.random.permutation(len(X_test))
    X_test = X_test[shuffle_idx]
    Y_test = Y_test[shuffle_idx]
    
    # Calculate sizes
    num_classes = len(dataset_train.classes)
    input_size = np.prod(X_train[0].shape)  
    total_samples = len(X_train) + len(X_val) + len(X_test)
    
    # Calculate memory required (float32 for inputs, int16 for labels)
    input_bytes = total_samples * input_size * 4  
    label_bytes = total_samples * 2
    total_bytes = input_bytes + label_bytes
    total_kb = total_bytes / 1024
    
    logger.info(f"Required flash memory: {total_kb:.2f} KB ({total_bytes} bytes)")
    
    # Check against device flash
    flash_kb = args.target_device_flash_kb
    
    if flash_kb is None or flash_kb == 'None':
        logger.error("target_device_flash_kb not specified")
        raise ValueError("target_device_flash_kb is required when we need to export data")
    else:
        flash_kb = int(flash_kb)
        
        if total_kb > flash_kb:
            error_msg = (
                f"MEMORY OVERFLOW: Training data requires {total_kb:.2f} KB but device only has {flash_kb} KB flash.\n"
                f"Suggestions:\n"
                f"  1. Reduce samples_per_class\n"
                f"  2. Reduce num_frame_concat \n"
                f"  3. Use a device with more flash memory\n"
                f"  4. Reduce frame_size in feature extraction"
            )
            logger.error(error_msg)
            raise MemoryError(error_msg)
    
    trainable_model_dir = os.path.join(output_dir, 'trainable_model')
    os.makedirs(trainable_model_dir, exist_ok=True)
    
    header_path = generate_training_data_header(
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test),
        input_size, num_classes, trainable_model_dir, target_device
    )
    
    source_path = generate_training_data_source(
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test),
        input_size, trainable_model_dir, target_device
    )
    
    return header_path, source_path


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
MAIN_SUPPORTED_LAYERS = ["Gemm", "Conv"]

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
        split_tensor: The tensor where split should happen (output of frozen part of model, input to trainable part of model)
                     
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
    elif node.op == 'Conv':
        return parse_conv2d_layer(node)
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


def parse_conv2d_layer(node):
    """
    Extract Conv2D layer metadata from ONNX Conv node.
    
    Supports standard Conv2D with:
        - 4D weights [out_channels, in_channels, kernel_h, kernel_w]
        - Supports padding, stride
        - Required bias
        - dilation = 1 only
        - groups = 1 only
    
    Args:
        node: ONNX Conv node (from onnx_graphsurgeon)
        
    Returns:
        dict: Layer metadata with weights, bias, sizes, conv parameters
        
    Raises:
        ValueError: If not Conv2D (must have 4D weights)
        ValueError: If bias is missing
        ValueError: If dilation != [1, 1]
        ValueError: If groups != 1
        ValueError: If asymmetric padding
    """
    logger = logging.getLogger("root.parse_conv2d_layer")
    logger.info(f"  Parsing Conv2D layer: {node.name}")
    
    if len(node.inputs) < 3 or node.inputs[2].values is None:
        raise ValueError(f"Conv2D layer {node.name} must have bias. Bias is required for on-device training.")
    
    weights = node.inputs[1].values
    if weights.ndim != 4:
        raise ValueError( f"Conv layer {node.name} has {weights.ndim}D weights. Only Conv2D (4D weights) is supported. Got shape: {weights.shape}")
    
    out_channels, in_channels, kernel_h, kernel_w = weights.shape
    bias = node.inputs[2].values
    
    # Extract and validate conv attributes
    strides = node.attrs.get('strides', [1, 1])
    stride_h, stride_w = strides[0], strides[1]
    
    # Pads (default [0, 0, 0, 0] = [top, left, bottom, right])
    pads = node.attrs.get('pads', [0, 0, 0, 0])
    
    # Validate symmetric padding
    if pads[0] != pads[2] or pads[1] != pads[3]:
        raise ValueError(f"Conv2D layer {node.name} has asymmetric padding {pads}. Only symmetric padding is supported (top==bottom, left==right).")
    
    padding_h, padding_w = pads[0], pads[1]
    
    # Dilations (must be [1, 1])
    dilations = node.attrs.get('dilations', [1, 1])
    if dilations != [1, 1]:
        raise ValueError(f"Conv2D layer {node.name} has dilation {dilations}. Only dilation=[1,1] is supported for on-device training.")
    
    # Groups (must be 1)
    group = node.attrs.get('group', 1)
    if group != 1:
        raise ValueError(f"Conv2D layer {node.name} has group={group}. Only group=1 is supported.")
    
    # Get input dimensions
    input_tensor = node.inputs[0]
    input_shape = input_tensor.shape  # [batch, in_channels, height, width]
    
    # input_shape[0] is batch, input_shape[1] is channels
    input_height = input_shape[2]
    input_width = input_shape[3]
    
    output_tensor = node.outputs[0]
    output_shape = output_tensor.shape  # [batch, out_channels, height, width]
    
    output_height = output_shape[2]
    output_width = output_shape[3]
    
    input_size = in_channels * input_height * input_width
    output_size = out_channels * output_height * output_width
    
    logger.info(f"    Input:  {in_channels} x {input_height} x {input_width} (flat: {input_size})")
    logger.info(f"    Output: {out_channels} x {output_height} x {output_width} (flat: {output_size})")
    logger.info(f"    Kernel: {kernel_h} x {kernel_w}")
    logger.info(f"    Stride: {stride_h} x {stride_w}")
    logger.info(f"    Padding: {padding_h} x {padding_w}")
    logger.info(f"    Weights: {weights.shape} ({weights.size} params)")
    logger.info(f"    Bias: {bias.shape} ({bias.size} params)")
    
    layer_info = {
        'type': 'Conv2D',
        'name': node.name,
        'input_size': input_size,
        'output_size': output_size,
        'weights': weights,  # Shape: [out_channels, in_channels, kernel_h, kernel_w]
        'bias': bias,        # Shape: [out_channels]
        'out_channels': out_channels,
        'in_channels': in_channels,
        'kernel_h': kernel_h,
        'kernel_w': kernel_w,
        'stride_h': stride_h,
        'stride_w': stride_w,
        'padding_h': padding_h,
        'padding_w': padding_w,
        'input_height': input_height,
        'input_width': input_width,
        'output_height': output_height,
        'output_width': output_width,
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

def generate_source_file(all_weights, output_dir, target_device):
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

// Initial weights from trained model\n""")

    pragma = get_pragma_for_device(target_device, 'ALL_WEIGHTS', 'odt_trainable_parameters')
    content.append(f"{pragma}\nfloat ALL_WEIGHTS[TOTAL_PARAMS] = {{\n")
    
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
    
    content.append("\n};\n\n// Best weights storage\n")
    
    pragma = get_pragma_for_device(target_device, 'ALL_BEST_WEIGHTS', 'odt_best_parameters')
    content.append(f"{pragma}\nfloat ALL_BEST_WEIGHTS[TOTAL_PARAMS];\n\n")

    # ========================================================================
    # SECTION 3: Buffer Data
    # ========================================================================
    
    content.append("""// ============================================================================
// BUFFER STORAGE - Definitions
// ============================================================================

// Intermediate activation buffers (for forward pass)
""")

    pragma = get_pragma_for_device(target_device, 'INTERMEDIATE_BUFFERS', 'odt_intermediate_buffers')
    content.append(f"{pragma}\nfloat INTERMEDIATE_BUFFERS[TOTAL_INTERMEDIATE_BUFFER_SIZE];\n\n")

    content.append("// Gradient buffers (for backward pass)\n")
    
    pragma = get_pragma_for_device(target_device, 'GRADIENT_BUFFERS', 'odt_gradient_buffers')
    content.append(f"{pragma}\nfloat GRADIENT_BUFFERS[TOTAL_GRADIENT_BUFFER_SIZE];\n\n")

    content.append("// Weight gradient accumulators (conditional - only for batch training)\n")
    pragma = get_pragma_for_device(target_device, 'ALL_WEIGHT_GRADS', 'odt_gradient_parameters')
    content.append(f"#if USE_GRADIENT_ACCUMULATION\n{pragma}\nfloat ALL_WEIGHT_GRADS[TOTAL_PARAMS];\n#endif\n\n")
    
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
        elif layer['type'] == 'Conv2D':
            layer_seq.append(f"Conv2D({layer['in_channels']}x{layer['input_height']}x{layer['input_width']}→{layer['out_channels']}x{layer['output_height']}x{layer['output_width']})")
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
    LAYER_TYPE_CONV2D,
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
            uint16_t kernel_h;
            uint16_t kernel_w;
            uint16_t stride_h;
            uint16_t stride_w;
            uint16_t padding_h;
            uint16_t padding_w;
            uint16_t input_height;
            uint16_t input_width;
            uint16_t output_height;
            uint16_t output_width;
        } conv2d;
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
        
        elif layer_type == 'Conv2D':
            layer_init = f"""    // Layer {i}: Conv2D(in={layer['in_channels']}x{layer['input_height']}x{layer['input_width']}, out={layer['out_channels']}x{layer['output_height']}x{layer['output_width']}, k={layer['kernel_h']}x{layer['kernel_w']})
    {{
        .type = LAYER_TYPE_CONV2D,
        .input_size = {layer['input_size']},
        .output_size = {layer['output_size']},
        .weight_offset = {weight_offset},
        .weight_count = {weight_count},
        .bias_offset = {bias_offset},
        .bias_count = {bias_count},
        .shape.conv2d = {{
            .out_channels = {layer['out_channels']},
            .in_channels = {layer['in_channels']},
            .kernel_h = {layer['kernel_h']},
            .kernel_w = {layer['kernel_w']},
            .stride_h = {layer['stride_h']},
            .stride_w = {layer['stride_w']},
            .padding_h = {layer['padding_h']},
            .padding_w = {layer['padding_w']},
            .input_height = {layer['input_height']},
            .input_width = {layer['input_width']},
            .output_height = {layer['output_height']},
            .output_width = {layer['output_width']}
        }}
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
    
    # Run shape inference to populate intermediate tensor shapes .This is needed because PyTorch ONNX export doesn't always include shape metadata for intermediate tensors
    logger.info(" Running ONNX shape inference...")
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model, data_prop=True)
    except Exception as e:
        logger.warning(f" Shape inference failed: {e}. Continuing without full shape info.")
    
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
        trainable_model_dir,
        args.target_device
    )
    logger.info(f"Generated trainable model config source file at {source_file_path}")