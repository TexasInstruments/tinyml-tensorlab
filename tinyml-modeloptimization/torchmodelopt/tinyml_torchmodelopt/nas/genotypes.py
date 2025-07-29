from collections import namedtuple  # Import namedtuple for creating simple classes

# Define a namedtuple for CNN genotype, which describes the architecture
Genotype_CNN = namedtuple('Genotype_CNN', 'normal normal_concat reduce reduce_concat')
# Define a namedtuple for RNN genotype, which describes the architecture
Genotype_RNN = namedtuple('Genotype_RNN', 'recurrent concat')

# List of primitive operations for CNN search space
PRIMITIVES_CNN = [
    'none',                # No operation (used for pruning)
    'avg_pool_3x1',        # 3x1 average pooling
    'max_pool_3x1',        # 3x1 max pooling
    'skip_connect',        # Identity (skip connection)
    'conv_bn_relu_3x1',    # 3x1 convolution + batch norm + ReLU
    'conv_bn_relu_5x1',    # 5x1 convolution + batch norm + ReLU
    'conv_bn_relu_7x1',    # 7x1 convolution + batch norm + ReLU
    # 'sep_conv_3x1',      # 3x1 separable convolution (commented out)
    # 'sep_conv_5x1',      # 5x1 separable convolution (commented out)
    # 'sep_conv_7x1',      # 7x1 separable convolution (commented out)
    # 'dil_conv_3x1',      # 3x1 dilated convolution (commented out)
    # 'dil_conv_5x1',      # 5x1 dilated convolution (commented out)
]

# List of primitive operations for RNN search space
PRIMITIVES_RNN = [
    'none',      # No operation (used for pruning)
    'tanh',      # Tanh activation
    'relu',      # ReLU activation
    'sigmoid',   # Sigmoid activation
    'identity',  # Identity (skip connection)
]

# Example genotype for demonstration (CNN)
# - normal: list of (operation, input node) tuples for normal cell
# - normal_concat: indices of nodes to concatenate for normal cell output
# - reduce: list of (operation, input node) tuples for reduction cell
# - reduce_concat: indices of nodes to concatenate for reduction cell output
DEMO = Genotype_CNN(
    normal=[('conv_bn_relu_7x1', 0), ('sep_conv_3x1', 0), ('max_pool_3x1', 0), ('skip_connect', 1)],
    normal_concat=range(1, 5),
    reduce=[('dil_conv_3x1', 0), ('dil_conv_3x1', 1), ('skip_connect', 2), ('conv_bn_relu_7x1', 0)],
    reduce_concat=range(1, 5)
)

# Example genotype for a specific use case (MOTOR_FAULT_6)
MOTOR_FAULT_6 = Genotype_CNN(
    normal=[('conv_bn_relu_3x1', 0), ('conv_bn_relu_5x1', 1), ('skip_connect', 0), ('conv_bn_relu_7x1', 1)],
    normal_concat=range(1, 5),
    reduce=[('max_pool_3x1', 0), ('conv_bn_relu_7x1', 1), ('max_pool_3x1', 2), ('conv_bn_relu_5x1', 0)],
    reduce_concat=range(1, 5)
)