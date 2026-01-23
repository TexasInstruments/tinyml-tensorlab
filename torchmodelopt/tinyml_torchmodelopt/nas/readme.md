# Neural Architecture Search (NAS) for Tiny ML with PyTorch

This module provides a flexible and extensible framework for Neural Architecture Search (NAS) targeting CNN architectures, with a focus on Tiny ML and resource-constrained environments. It supports differentiable architecture search, resource-aware optimization (memory and compute), and is designed for easy integration into PyTorch projects.

---

## Features

- **Differentiable NAS**: Uses gradient-based optimization to search for optimal neural architectures.
- **Resource-Aware Search**: Incorporate memory and compute (MACs) constraints into the search process.
- **Unrolled Optimization**: Optionally use unrolled optimization for more accurate architecture gradient estimation.
- **Extensible Search Space**: Easily add or modify primitive operations for CNNs.
- **PyTorch Native**: Fully compatible with PyTorch models, optimizers, and training loops.

---

## Directory Structure

```
nas/
    ├── architect.py         # Core NAS optimization logic (Architect class)
    ├── genotypes.py         # Genotype definitions and search space primitives
    ├── model.py             # Final model definition (for evaluation)
    ├── model_search_cnn.py  # Search-phase model definition (with architecture parameters)
    ├── operations.py        # Primitive operations for the search space
    ├── train_cnn_search.py  # Training loop for NAS (search phase)
    ├── utils.py             # Utility functions (metrics, parameter counting, etc.)
```

---

## Key Components

### 1. `architect.py`
- Implements the `Architect` class, which manages the optimization of architecture parameters.
- Supports both standard and unrolled optimization.
- Allows for resource-aware penalties (memory, compute) during search.

### 2. `genotypes.py`
- Defines the search space via `PRIMITIVES_CNN`.
- Contains namedtuples for genotypes and example genotypes for demonstration.

### 3. `model_search_cnn.py`
- Defines the search-phase network, which includes architecture parameters (`alphas`).
- Supports parsing of learned architecture into a discrete genotype.

### 4. `model.py`
- Defines the final evaluation model, which is instantiated with a fixed genotype.

### 5. `operations.py`
- Implements all primitive operations (convolutions, pooling, skip, zero, etc.) used in the search space.

### 6. `train_cnn_search.py`
- Contains the main NAS training loop.
- Handles architecture and weight updates, validation, and genotype extraction.

### 7. `utils.py`
- Utility functions for metrics, parameter counting, checkpointing, etc.

---

## How to Use

### 1. Define Your Search Space

You can define or modify the search space by editing `genotypes.py` and `operations.py`.

#### Example: Adding/Modifying CNN Primitives

In `genotypes.py`, the list `PRIMITIVES_CNN` defines the available operations for the search space:

```python
# filepath: nas/genotypes.py
PRIMITIVES_CNN = [
    'none',
    'avg_pool_3x1',
    'max_pool_3x1',
    'skip_connect',
    'conv_bn_relu_3x1',
    'conv_bn_relu_5x1',
    'conv_bn_relu_7x1',
    # 'sep_conv_3x1',
    # 'sep_conv_5x1',
    # 'sep_conv_7x1',
    # 'dil_conv_3x1',
    # 'dil_conv_5x1',
]
```

To add a new operation, simply add its name to this list and define its implementation in `operations.py`.

#### Example: Adding a Custom Operation

Suppose you want to add a 1x1 convolution as a primitive:

1. **Add to `PRIMITIVES_CNN`:**

```python
PRIMITIVES_CNN = [
    # ...existing code...
    'conv_bn_relu_1x1',
]
```

2. **Define in `operations.py`:**

```python
# filepath: nas/operations.py
OPS = {
    # ...existing code...
    'conv_bn_relu_1x1': lambda C, stride, affine: ConvBNReLU(C, C, (1, 1), (stride, 1), (0, 0), affine=affine),
}
```

#### Example: Custom Genotype

You can also define a custom genotype for evaluation or demonstration:

```python
# filepath: nas/genotypes.py
from collections import namedtuple

Genotype_CNN = namedtuple('Genotype_CNN', 'normal normal_concat reduce reduce_concat')

MY_CUSTOM_GENOTYPE = Genotype_CNN(
    normal=[('conv_bn_relu_3x1', 0), ('skip_connect', 1), ('max_pool_3x1', 0), ('conv_bn_relu_5x1', 1)],
    normal_concat=range(1, 5),
    reduce=[('max_pool_3x1', 0), ('conv_bn_relu_7x1', 1), ('skip_connect', 2), ('conv_bn_relu_5x1', 0)],
    reduce_concat=range(1, 5)
)
```

You can then use this genotype to instantiate the final model for evaluation.

**How the architecture is formed from the genotype:**

The genotype specifies the structure of each cell in the final network. Each cell (normal or reduction) is built by stacking nodes, where each node receives inputs from previous nodes or the cell's input, and applies the specified operations. For each entry in the `normal` or `reduce` list, the tuple `('op_name', input_idx)` means that the operation `op_name` is applied to the output of node `input_idx`. The outputs listed in `normal_concat` or `reduce_concat` are concatenated to form the cell's output. This mapping from genotype to architecture is handled in the model implementation (see `model.py`), ensuring the final network matches the searched architecture.

**How to instantiate and run a model from a custom genotype:**

To evaluate or fine-tune a model with your custom genotype, use the `NetworkCnn` class from `model.py`:

```python
# filepath: nas/model.py
from nas.model import NetworkCnn
from nas.genotypes import MY_CUSTOM_GENOTYPE

# Example arguments (adjust as needed)
model = NetworkCnn(
    C=16,                    # initial channels
    num_classes=10,          # number of output classes
    layers=8,                # number of layers/cells
    genotype=MY_CUSTOM_GENOTYPE,
    in_channels=1,           # input channels (e.g., 1 for grayscale)
    nodes_per_layer=4,
    multiplier=4,
    stem_multiplier=3
)

# Move to device, load weights, and run inference or training as usual
model = model.cuda()
# model.load_state_dict(torch.load('your_weights.pth'))
# outputs = model(inputs)
```

Replace the arguments as appropriate for your use case. You can now use this model for evaluation, further training, or deployment.

### 2. Prepare Your Dataset

Prepare PyTorch `DataLoader` objects for training and validation data. These should be assigned to `args.train_loader` and `args.valid_loader`.

### 3. Set Up Arguments

Create an `argparse.Namespace` or similar object with all required hyperparameters, e.g.:

```python
import argparse

args = argparse.Namespace(
    gpu=0,
    nas_init_channels=16,
    num_classes=10,
    nas_layers=8,
    in_channels=1,
    nas_nodes_per_layer=4,
    nas_multiplier=4,
    nas_stem_multiplier=3,
    lr=0.025,
    momentum=0.9,
    weight_decay=3e-4,
    nas_budget=50,
    nas_optimization_mode='Memory',  # or 'Compute'
    grad_clip=5,
    unrolled=False,
    train_loader=your_train_loader,
    valid_loader=your_valid_loader,
    mode='cnn',
    arch_learning_rate=3e-4,
    arch_weight_decay=1e-3,
    # ... add other required arguments ...
)
```

#### Argument Descriptions

| Argument                  | Type      | Description |
|---------------------------|-----------|-------------|
| **gpu**                   | int       | GPU device index to use for training and search. |
| **nas_init_channels**     | int       | Initial number of channels for the first cell/layer. Controls the network width. |
| **num_classes**           | int       | Number of output classes for classification. |
| **nas_layers**            | int       | Number of layers (cells) in the network during search. |
| **in_channels**           | int       | Number of input channels (e.g., 1 for grayscale, 3 for RGB). |
| **nas_nodes_per_layer**   | int       | Number of intermediate nodes per cell (controls search space complexity). |
| **nas_multiplier**        | int       | Number of outputs to concatenate per cell (affects output width). |
| **nas_stem_multiplier**   | int       | Multiplier for the number of channels in the initial stem convolution. |
| **lr**                    | float     | Initial learning rate for model weights (SGD). |
| **momentum**              | float     | Momentum for SGD optimizer. |
| **weight_decay**          | float     | Weight decay (L2 regularization) for model weights. |
| **nas_budget**            | int       | Number of epochs (or search iterations) for NAS. |
| **nas_optimization_mode** | str       | Resource-aware penalty mode: `'Memory'` (parameter count) or `'Compute'` (MACs). |
| **grad_clip**             | float     | Maximum gradient norm for gradient clipping. |
| **unrolled**              | bool      | Whether to use unrolled optimization for architecture gradients. |
| **train_loader**          | DataLoader| PyTorch DataLoader for training data. |
| **valid_loader**          | DataLoader| PyTorch DataLoader for validation data. |
| **mode**                  | str       | Search mode: `'cnn'` for convolutional. |
| **arch_learning_rate**    | float     | Learning rate for architecture parameters (alphas). |
| **arch_weight_decay**     | float     | Weight decay for architecture parameters. |

**Notes:**
- You may need to add additional arguments depending on your dataset, training regime, or custom modifications (e.g., batch size, logging directory, etc.).
- `nas_optimization_mode` allows you to bias the search towards architectures that are more memory- or compute-efficient.
- `unrolled=True` enables unrolled optimization, which can improve search quality at the cost of speed and memory.

### 4. Run NAS Search

```python
from .train_cnn_search import search_and_get_model

# Run the search and get the best model (with fixed genotype)
final_model = search_and_get_model(args)
```

### 5. Evaluate or Fine-tune the Final Model

You can now use `final_model` for further training, evaluation, or deployment.

---

## Example: Full NAS Workflow

```python
from torch.utils.data import DataLoader
from my_dataset import MyDataset
from .train_cnn_search import search_and_get_model

# Prepare your datasets and loaders
train_dataset = MyDataset(split='train')
valid_dataset = MyDataset(split='valid')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

# Set up arguments (see above for details)
args = ...  # Fill in all required fields

# Assign loaders
args.train_loader = train_loader
args.valid_loader = valid_loader

# Run NAS
final_model = search_and_get_model(args)

# Save or evaluate the final model
torch.save(final_model.state_dict(), 'final_model.pth')
```

---

## Tips & Notes

- **Resource-Aware Search**: Set `nas_optimization_mode` to `'Memory'` or `'Compute'` to penalize large models or high MACs.
- **Unrolled Optimization**: Set `unrolled=True` for more accurate architecture gradients (slower, but sometimes better results).
- **Custom Operations**: Add new operations to `operations.py` and include them in `PRIMITIVES_CNN`.
- **Genotype Extraction**: After search, the best genotype is extracted and used to instantiate the final model.

---

## References

- [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)
- [β-DARTS: Beta-Decay Regularization for Differentiable Architecture Search](https://arxiv.org/abs/2203.01665)
- [Balanced One-shot Neural Architecture Optimization](https://arxiv.org/abs/1909.10815)

