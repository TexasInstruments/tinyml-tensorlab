# NPU Configuration Guidelines for TI Neural Network Compiler

This document provides comprehensive guidelines for designing neural network models that are optimally configured for the TI NPU (Neural Processing Unit) accelerator found in devices like F28P55 and F28P65.

## Table of Contents

1. [Overview](#overview)
2. [Supported Layer Types](#supported-layer-types)
3. [Terminology and Notation](#terminology-and-notation)
4. [Layer Configuration Constraints](#layer-configuration-constraints)
5. [Optimal Design Patterns](#optimal-design-patterns)
6. [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
7. [Model Design Checklist](#model-design-checklist)

---

## Overview

The TI NPU accelerator provides hardware acceleration for common neural network operations. However, to fully leverage the NPU, models must conform to specific layer configurations. Layers that don't meet these constraints will fall back to software execution, significantly reducing performance.

**Key Principle**: Design models with NPU constraints in mind from the start, rather than trying to adapt existing models.

---

## Supported Layer Types

| Layer Type | NPU Name | Description |
|------------|----------|-------------|
| First Convolution | **FCONV** | Convolution with input channel = 1 |
| Generic Convolution | **GCONV** | Standard convolution with input channels as multiple of 4 |
| Depth-Wise Convolution | **DWCONV** | Convolution where groups = input channels |
| Point-Wise Convolution | **PWCONV** | 1x1 convolution for channel mixing |
| Point-Wise Conv + Residual | **PWCONVRES** | 1x1 convolution with residual addition |
| Transposed Convolution | **TCONV** | Upsampling convolution |
| Fully-Connected | **FC** | Dense/Linear layer |
| Average Pooling | **AVGPOOL** | Global and non-global average pooling |
| Max Pooling | **MAXPOOL** | Maximum pooling |

---

## Terminology and Notation

### Dimension Notation

| Symbol | Meaning | Example |
|--------|---------|---------|
| **iB** | Input bit-width | 8 (8-bit quantized) |
| **oB** | Output bit-width | 8 |
| **kB** | Kernel/weight bit-width | 2, 4, or 8 |
| **iH** | Input height | Sequence length for 1D |
| **iW** | Input width | 1 for 1D time series |
| **iC** | Input channels | Number of input features |
| **oH** | Output height | After convolution/pooling |
| **oW** | Output width | After convolution/pooling |
| **oC** | Output channels | Number of output features |
| **kH** | Kernel height | Convolution kernel size |
| **kW** | Kernel width | Convolution kernel size |
| **sH** | Stride height | Vertical stride |
| **sW** | Stride width | Horizontal stride |

### Value Notation

| Notation | Meaning | Examples |
|----------|---------|----------|
| **any** | Any positive integer | 1, 2, 3, ... |
| **m4** | Multiples of 4 | 4, 8, 12, 16, 20, ... |
| **m8** | Multiples of 8 | 8, 16, 24, 32, ... |
| **m1b2e7** | Range from 2 to 7 | 2, 3, 4, 5, 6, 7 |
| **m1b8** | Minimum 8, any value | 8, 9, 10, 11, ... |
| **m1b16** | Minimum 16, any value | 16, 17, 18, ... |

---

## Layer Configuration Constraints

### FCONV (First Convolution Layer)

Use when the input has **exactly 1 channel** (e.g., single-variable time series).

| Parameter | Constraint | Notes |
|-----------|------------|-------|
| Input Channels (iC) | **1** | Fixed - this defines FCONV |
| Output Channels (oC) | **m4** | Must be 4, 8, 12, 16, ... |
| Kernel Height (kH) | any | Flexible |
| Kernel Width (kW) | **1-8** | Maximum 8 for 1D convolutions |
| Kernel Bit-width (kB) | 2, 4, or 8 | 8-bit most common |

**Example (PyTorch)**:
```python
# Good: FCONV with iC=1, oC=8
Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 1))

# Bad: oC=6 is not m4
Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 1))
```

### GCONV (Generic Convolution Layer)

Use for intermediate convolution layers where input channels > 1.

| Parameter | Constraint | Notes |
|-----------|------------|-------|
| Input Channels (iC) | **m4** | Will be padded to m4 if not |
| Output Channels (oC) | **m4** | Must be 4, 8, 12, 16, ... |
| Kernel Height (kH) | **1-7** | Maximum 7! |
| Kernel Width (kW) | any | Flexible |
| Kernel Bit-width (kB) | 2, 4, or 8 | 8-bit most common |

**Critical**: Kernel height is limited to **7 or less**!

**Example (PyTorch)**:
```python
# Good: kH=5 within limit
Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 1))

# Bad: kH=9 exceeds limit of 7
Conv2d(in_channels=16, out_channels=32, kernel_size=(9, 1))
```

### DWCONV (Depth-Wise Convolution Layer)

Use for efficient spatial filtering with `groups=in_channels`.

| Parameter | Constraint | Notes |
|-----------|------------|-------|
| Input Channels (iC) | **m4** | Must be 4, 8, 12, 16, ... |
| Output Channels (oC) | **m4** | Equal to iC for true depthwise |
| Kernel Height (kH) | any | Flexible |
| Kernel Width (kW) | **1-7** | Maximum 7! |
| Groups | **iC** | Must equal input channels |

**Example (PyTorch)**:
```python
# Good: Depthwise with kW=1, groups=16
Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 1), groups=16)

# Bad: kW=9 exceeds limit
Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 9), groups=16)
```

### PWCONV (Point-Wise Convolution Layer)

Use for channel mixing after depthwise convolution (1x1 convolution).

| Parameter | Constraint | Notes |
|-----------|------------|-------|
| Input Channels (iC) | **m4** | Must be 4, 8, 12, 16, ... |
| Output Channels (oC) | **m4** | Must be 4, 8, 12, 16, ... |
| Kernel Size | **(1, 1)** | Fixed for pointwise |
| Stride | **(1, 1)** | Fixed |

**Example (PyTorch)**:
```python
# Good: 1x1 conv with m4 channels
Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1))
```

### FC (Fully-Connected Layer)

| Parameter | Constraint (8-bit) | Constraint (4-bit) |
|-----------|-------------------|-------------------|
| Input Features | **>= 16** | **>= 8** |
| Output Features | any | any |

**Critical**: Ensure sufficient input features before FC layer!

**Example (PyTorch)**:
```python
# Good: input features = 64 (from 16 channels * 4 spatial)
AdaptiveAvgPool2d((4, 1))  # With 16 channels -> 64 features
Linear(in_features=64, out_features=num_classes)

# Bad: input features = 4 (below minimum)
AdaptiveAvgPool2d((1, 1))  # With 4 channels -> 4 features
Linear(in_features=4, out_features=num_classes)
```

### MAXPOOL (Max Pooling Layer)

| Parameter | Constraint | Notes |
|-----------|------------|-------|
| Input Channels (iC) | **m4** | Must be 4, 8, 12, 16, ... |
| Output Channels (oC) | **m4** | Same as input |
| Kernel Height (kH) | **1-4** | Maximum 4! |
| Kernel Width (kW) | **1-4** | Maximum 4! |

**Example (PyTorch)**:
```python
# Good: kernel within limits
MaxPool2d(kernel_size=(3, 1), stride=(2, 1))

# Bad: kH=8 exceeds limit
MaxPool2d(kernel_size=(8, 1), stride=(4, 1))
```

### AVGPOOL (Average Pooling Layer)

**Global Average Pooling**:
| Parameter | Constraint | Notes |
|-----------|------------|-------|
| Input Channels (iC) | **m4** | Must be 4, 8, 12, 16, ... |
| Output Size | **(1, 1)** | Global pooling |
| Condition | **(iH * iW) > 2** | Must have spatial dimensions |

**Non-Global Average Pooling**: Converted to DWCONV internally. Follow DWCONV constraints.

---

## Optimal Design Patterns

### Pattern 1: MobileNet-Style Depthwise Separable Convolution

The most efficient pattern for NPU acceleration:

```python
# Depthwise convolution (spatial filtering)
Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 1), groups=16)
BatchNorm2d(16)
ReLU()

# Pointwise convolution (channel mixing)
Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1))
BatchNorm2d(32)
ReLU()
```

### Pattern 2: Channel Doubling Progression

Efficient channel progression that maintains m4 constraint:

```python
# Start: 1 -> 8 (FCONV)
# Then: 8 -> 16 -> 32 -> 64 (GCONV)
channels = [1, 8, 16, 32, 64]
```

### Pattern 3: Replace Large Kernels with Multiple Small Kernels

Instead of one large kernel, use multiple smaller ones:

```python
# Bad: Single large kernel (kH=9 exceeds limit)
Conv2d(in_channels=16, out_channels=32, kernel_size=(9, 1))

# Good: Two smaller kernels (both kH<=7)
Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 1))
Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 1))
```

### Pattern 4: Ensure Sufficient FC Input

Design pooling to ensure minimum FC input features:

```python
# Good: Ensure >= 16 features for FC
# Option A: More channels
AdaptiveAvgPool2d((1, 1))  # With 16+ channels

# Option B: Larger spatial output
AdaptiveAvgPool2d((4, 1))  # With 4+ channels -> 16+ features
```

---

## Common Pitfalls to Avoid

### 1. Large Kernel Heights (kH > 7)

**Problem**: GCONV only supports kH from 2-7.

```python
# WRONG
Conv2d(in_channels=16, out_channels=32, kernel_size=(9, 1))
Conv2d(in_channels=16, out_channels=32, kernel_size=(12, 1))
Conv2d(in_channels=16, out_channels=32, kernel_size=(16, 1))
```

**Solution**: Use multiple smaller kernels or strided convolutions.

### 2. Non-m4 Channel Counts

**Problem**: Channels not divisible by 4.

```python
# WRONG
Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 1))  # 12 is OK
Conv2d(in_channels=12, out_channels=18, kernel_size=(3, 1))  # 18 is NOT m4!
```

**Solution**: Always use channels: 4, 8, 12, 16, 20, 24, 32, 48, 64...

### 3. Insufficient FC Input Features

**Problem**: FC layer receives fewer than 16 features (8-bit) or 8 features (4-bit).

```python
# WRONG: Only 4 features to FC
Conv2d(1, 4, kernel_size=(3, 1))
AdaptiveAvgPool2d((1, 1))  # 4 channels * 1 * 1 = 4 features
Linear(4, num_classes)  # FAILS on NPU!
```

**Solution**: Increase channels or spatial size before FC.

### 4. Large MaxPool Kernels

**Problem**: MaxPool kernel > 4.

```python
# WRONG
MaxPool2d(kernel_size=(8, 1))  # kH=8 exceeds limit
MaxPool2d(kernel_size=(5, 1))  # kH=5 exceeds limit
```

**Solution**: Use multiple smaller pooling operations or strided convolutions.

---

## Model Design Checklist

Use this checklist when designing or reviewing models for NPU compatibility:

### Channel Dimensions
- [ ] First layer input channels = 1 (for FCONV) OR multiple of 4
- [ ] All intermediate layer channels are multiples of 4
- [ ] Output channels of all convolutions are multiples of 4

### Kernel Sizes
- [ ] All GCONV kernel heights (kH) are <= 7
- [ ] All DWCONV kernel widths (kW) are <= 7
- [ ] All FCONV kernel widths (kW) are <= 8
- [ ] All MaxPool kernels are <= 4 in each dimension

### Fully-Connected Layers
- [ ] FC input features >= 16 (for 8-bit weights)
- [ ] FC input features >= 8 (for 4-bit weights)

### Pooling
- [ ] Global AvgPool has (iH * iW) > 2
- [ ] Non-global AvgPool follows DWCONV constraints
- [ ] MaxPool kernels <= 4

### Depthwise Convolutions
- [ ] Groups parameter equals input channels
- [ ] Both input and output channels are m4

### General
- [ ] No padding requirements exceed NPU support
- [ ] Stride values are supported for each layer type

---

## Quick Reference Card

| Layer Type | Max kH | Max kW | iC Constraint | oC Constraint |
|------------|--------|--------|---------------|---------------|
| FCONV | any | 8 | 1 | m4 |
| GCONV | **7** | any | m4 (padded) | m4 |
| DWCONV | any | **7** | m4 | m4 |
| PWCONV | 1 | 1 | m4 | m4 |
| MAXPOOL | **4** | **4** | m4 | m4 |
| FC | N/A | N/A | >=16 (8-bit) | any |

---

## References

- TI Neural Network Compiler for MCUs User's Guide v2.1.0
- Section 5: Layer Configurations Supported on the NPU
- [TI Software Download](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/)

---

*Document Version: 1.0*
*Last Updated: January 2025*
