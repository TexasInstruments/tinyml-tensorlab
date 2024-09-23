import numpy as np
import onnx


def parse_tensor_shape(tensor_proto):
    shape = []
    for dim in tensor_proto.dims:
        shape.append(dim)
    #
    return shape


def parse_dtype(tensor_proto):
    if tensor_proto.data_type == tensor_proto.INT:
        return np.int32
    elif tensor_proto.data_type == tensor_proto.INT64:
        return np.int64
    elif tensor_proto.data_type == tensor_proto.FLOAT:
        return np.float32
    elif tensor_proto.data_type == tensor_proto.DOUBLE:
        return np.float64
    elif tensor_proto.data_type == tensor_proto.FLOAT16:
        return np.float16
    elif tensor_proto.data_type == tensor_proto.INT8:
        return np.int8
    elif tensor_proto.data_type == tensor_proto.UINT8:
        return np.uint8
    elif tensor_proto.data_type == tensor_proto.BOOL:
        return np.bool_
    elif tensor_proto.data_type == tensor_proto.STRING:
        return np.string_
    else:
        return None


def parse_tensor_to_array(tensor_proto):
    shape = parse_tensor_shape(tensor_proto)
    dtype = parse_dtype(tensor_proto)


def parse_attribute(onnx_model, attribute):
    if attribute.type == attribute.INT:
        return attribute.i
    elif attribute.type == attribute.FLOAT:
        return attribute.f
    elif attribute.type == attribute.STRING:
        return attribute.s
    elif attribute.type == attribute.INTS:
        values = []
        for ints in attribute.ints:
            values.append(ints)
        #
        return values
    elif attribute.type == attribute.FLOATS:
        values = []
        for floats in attribute.floats:
            values.append(floats)
        #
        return values
    elif attribute.type == attribute.STRINGS:
        values = []
        for strings in attribute.strings:
            values.append(strings)
        #
        return values
    elif attribute.type == attribute.TENSOR:
        value = parse_tensor_to_array(attribute.t)
        return value
    else:
        return None


def parse_attributes(onnx_model, node):
    node_attributes = {}
    for attribute in node.attribute:
        value = parse_attribute(onnx_model, attribute)
        node_attributes.update({attribute.name:value})
    #
    return node_attributes


def make_conv_node(onnx_model, old_node):
    node_attributes = parse_attributes(onnx_model, old_node)
    new_node = onnx.helper.make_node(op_type='Conv', inputs=old_node.input, outputs=old_node.output, **node_attributes)
    return new_node


def replace_node(onnx_model, old_node, new_node):
    # print(old_node)
    onnx_model.graph.node.append(new_node)
    onnx_model.graph.node.remove(old_node)

