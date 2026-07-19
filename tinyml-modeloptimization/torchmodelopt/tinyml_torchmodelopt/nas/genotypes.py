from collections import namedtuple

# Namedtuple describing a CNN cell architecture
Genotype_CNN = namedtuple('Genotype_CNN', 'normal normal_concat reduce reduce_concat')

# Primitive operations available in the CNN search space (all Nx1 for 1D time-series)
PRIMITIVES_CNN = [
    'none',                # No operation (used for pruning)
    'avg_pool_3x1',        # 3x1 average pooling
    'max_pool_3x1',        # 3x1 max pooling
    'skip_connect',        # Identity (skip connection)
    'conv_bn_relu_3x1',    # 3x1 convolution + batch norm + ReLU
    'conv_bn_relu_5x1',    # 5x1 convolution + batch norm + ReLU
    'conv_bn_relu_7x1',    # 7x1 convolution + batch norm + ReLU
]

# Example genotype for motor fault classification
MOTOR_FAULT_6 = Genotype_CNN(
    normal=[('conv_bn_relu_3x1', 0), ('conv_bn_relu_5x1', 1), ('skip_connect', 0), ('conv_bn_relu_7x1', 1)],
    normal_concat=range(1, 5),
    reduce=[('max_pool_3x1', 0), ('conv_bn_relu_7x1', 1), ('max_pool_3x1', 2), ('conv_bn_relu_5x1', 0)],
    reduce_concat=range(1, 5)
)
