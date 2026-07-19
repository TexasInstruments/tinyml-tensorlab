# Trainable Model Configuration Files

## Table of Contents

- [1. Overview](#1-overview)
- [2. trainable_model_config.h](#2-trainable_model_configh)
  - [2.1 File Header](#21-file-header)
  - [2.2 Architecture Defines](#22-architecture-defines)
  - [2.3 Enumerations](#23-enumerations)
  - [2.4 Task and Training Configuration](#24-task-and-training-configuration)
  - [2.5 LayerParams_t Structure](#25-layerparams_t-structure)
  - [2.6 LAYER_PARAMS_INIT Table](#26-layer_params_init-table)
  - [2.7 BUFFER_OFFSETS Array](#27-buffer_offsets-array)
  - [2.8 Extern Declarations](#28-extern-declarations)
- [3. trainable_model_config.c](#3-trainable_model_configc)
  - [3.1 Weight Storage](#31-weight-storage)
  - [3.2 Buffer Storage](#32-buffer-storage)
  - [3.3 Memory Section Placement](#33-memory-section-placement)
- [4. How trainable_layers_from_last Maps to These Files](#4-how-trainable_layers_from_last-maps-to-these-files)
- [5. Further Reading](#5-further-reading)

---

## 1. Overview

The `trainable_model_config.h` and `trainable_model_config.c` files define the complete trainable portion of the neural network for on-device training. They are **auto-generated** by the ModelZoo Python pipeline and should not be manually edited except for the hyperparameters (unless you understand the internal layout).

Together, these files tell the on-device training library:
- How many layers exist and what type each is
- Where each layer's weights and biases live in a flat array
- How large the intermediate and gradient buffers need to be
- What task type and loss function to use

For context on how these files are generated, see [Running ModelZoo for On-Device Training](running_modelzoo_for_odt.md). For how the on-device training library uses these files, see [On-Device Training Library](ondevice_training_lib.md).

---

## 2. trainable_model_config.h

### 2.1 File Header

The header contains a comment block identifying the model, generation timestamp, number of trainable layers, and the full architecture as a human-readable string:

```c
// Model: AD_3_LAYER_DEEP_ONDEVICE_TRAINABLE_MODEL_TS
// Generated: 2026-02-17 15:08:56
// Trainable layers: 11
// Trainable model Architecture: Linear(48→24) → ReLU(24) → Linear(24→12) → ...
```

### 2.2 Architecture Defines

```c
#define NUM_TRAINABLE_LAYERS 11
#define FROZEN_OUTPUT_SIZE 48
#define FINAL_OUTPUT_SIZE 48
#define TOTAL_PARAMS 3150
#define TOTAL_INTERMEDIATE_BUFFER_SIZE 252
#define TOTAL_GRADIENT_BUFFER_SIZE 252
```

| Define | Meaning |
|--------|---------|
| `NUM_TRAINABLE_LAYERS` | Total number of layers in the trainable portion (includes both layers with parameters and other intermediate layers like ReLu) |
| `FROZEN_OUTPUT_SIZE` | Output size of the frozen model = input size to the first trainable layer |
| `FINAL_OUTPUT_SIZE` | Output size of the last trainable layer = final model output size |
| `TOTAL_PARAMS` | Total number of float parameters (all weights + all biases across all layers) |
| `TOTAL_INTERMEDIATE_BUFFER_SIZE` | Total floats needed for forward pass activation storage |
| `TOTAL_GRADIENT_BUFFER_SIZE` | Total floats needed for backward pass gradient storage|

### 2.3 Enumerations

```c
typedef enum {
    LAYER_TYPE_LINEAR,
    LAYER_TYPE_RELU,
} LayerType_t;

typedef enum {
    TASK_TYPE_CLASSIFICATION,
    TASK_TYPE_ANOMALY_DETECTION,
} TaskType_t;

typedef enum {
    LOSS_FUNCTION_MSE,
    LOSS_FUNCTION_CROSSENTROPY,
} LossFunction_t;
```

These enumerations are defined here so both the generated config and the training library share the same type definitions. Currently supported: `LAYER_TYPE_LINEAR` and `LAYER_TYPE_RELU` for layers, `TASK_TYPE_ANOMALY_DETECTION` with `LOSS_FUNCTION_MSE` for the task.

### 2.4 Task and Training Configuration

```c
#define TASK_TYPE TASK_TYPE_ANOMALY_DETECTION
#define LOSS_FUNCTION LOSS_FUNCTION_MSE

#define TRAIN_BATCH_SIZE 1
#define VAL_BATCH_SIZE 1

#if TRAIN_BATCH_SIZE > 1
    #define USE_GRADIENT_ACCUMULATION 1
#else
    #define USE_GRADIENT_ACCUMULATION 0
#endif
```

| Define | Meaning |
|--------|---------|
| `TASK_TYPE` | Determines which task library to use |
| `LOSS_FUNCTION` | Determines which loss function the training library applies |
| `TRAIN_BATCH_SIZE` | Number of samples per training weight update |
| `VAL_BATCH_SIZE` | Number of samples per validation evaluation |
| `USE_GRADIENT_ACCUMULATION` | **Compile-time flag** — when batch_size=1, this is 0 and gradient accumulation code is compiled out entirely. When batch_size>1, this is 1 and gradient accumulation buffers are allocated. This is a key memory optimization: batch_size=1 avoids allocating `TOTAL_PARAMS` extra floats for gradient storage. |

### 2.5 LayerParams_t Structure

This structure describes one layer in the trainable model:

```c
typedef struct {
    LayerType_t type;          // LAYER_TYPE_LINEAR or LAYER_TYPE_RELU
    uint16_t input_size;       // Number of input features
    uint16_t output_size;      // Number of output features

    // Offset and count into the flat ALL_WEIGHTS[] array
    uint16_t weight_offset;    // Starting index of weights in ALL_WEIGHTS
    uint16_t weight_count;     // Number of weight floats
    uint16_t bias_offset;      // Starting index of biases in ALL_WEIGHTS
    uint16_t bias_count;       // Number of bias floats

    // Layer-specific shape information
    union {
        struct {
            uint16_t rows;     // Output size (for weight matrix indexing)
            uint16_t cols;     // Input size (for weight matrix indexing)
        } linear;

        struct {
            uint16_t out_channels;
            uint16_t in_channels;
            uint16_t kernel_size;
            uint16_t stride;
            uint16_t padding;
        } conv1d;              // Reserved for future Conv1D support
    } shape;

    // Runtime pointers (assigned during ODT_Init)
    float* weights;            // Points into ALL_WEIGHTS at weight_offset
    float* bias;               // Points into ALL_WEIGHTS at bias_offset

#if USE_GRADIENT_ACCUMULATION
    float* weight_grad;        // Points into ALL_WEIGHT_GRADS at weight_offset
    float* bias_grad;          // Points into ALL_WEIGHT_GRADS at bias_offset
#endif
} LayerParams_t;
```

**Key design decisions:**

- **Offsets into a flat array**: Rather than separate allocations per layer, all weights are stored in a single `ALL_WEIGHTS[]` array. Each layer knows its offset and count. This enables single-`memcpy` save/load of all weights.
- **Runtime pointers**: The `weights`, `bias`, `weight_grad`, and `bias_grad` pointers are `NULL` in the static initializer. They are assigned during `ODT_Init()` to point into the actual arrays.
- **Conditional gradient fields**: When `USE_GRADIENT_ACCUMULATION` is 0, the gradient pointer fields don't exist, saving memory per layer.

### 2.6 LAYER_PARAMS_INIT Table

The static initializer table describes every layer in the trainable model. The on-device training library copies this table at initialization time and assigns runtime pointers.

**Example — walking through two entries from the fan blade model:**

```c
static const LayerParams_t LAYER_PARAMS_INIT[NUM_TRAINABLE_LAYERS] = {
    // Layer 0: Linear(48 → 24)
    {
        .type = LAYER_TYPE_LINEAR,
        .input_size = 48,
        .output_size = 24,
        .weight_offset = 0,        // Weights start at ALL_WEIGHTS[0]
        .weight_count = 1152,      // 48 × 24 = 1152 weight floats
        .bias_offset = 1152,       // Biases start at ALL_WEIGHTS[1152]
        .bias_count = 24,          // 24 bias floats
        .shape.linear = {.rows = 24, .cols = 48}
    },

    // Layer 1: ReLU(24)
    {
        .type = LAYER_TYPE_RELU,
        .input_size = 24,
        .output_size = 24,
        .weight_offset = 0,        // No weights for ReLU
        .weight_count = 0,
        .bias_offset = 0,          // No bias for ReLU
        .bias_count = 0
    },

    // ... remaining layers ...
};
```

**How to read this:**
- **Linear layers** have `weight_count = input_size × output_size` and `bias_count = output_size`. The weight matrix is stored in row-major order `[output_size × input_size]`.
- **ReLU layers** have zero weights and biases — they are purely computational (element-wise max(0, x)).
- **Weight offsets are cumulative**: Layer 0's weights occupy `ALL_WEIGHTS[0..1151]`, Layer 0's biases occupy `ALL_WEIGHTS[1152..1175]`, Layer 2's weights start at `ALL_WEIGHTS[1176]`, and so on.

### 2.7 BUFFER_OFFSETS Array

```c
static const uint16_t BUFFER_OFFSETS[13] = {
    0, 48, 72, 96, 108, 120, 126, 132, 144, 156, 180, 204, 252
};
```

This array maps each layer's input/output to a position in the flat `INTERMEDIATE_BUFFERS[]` and `GRADIENT_BUFFERS[]` arrays:

```
BUFFER_OFFSETS[0]  = 0     → frozen model output / input to layer 0  (48 floats)
BUFFER_OFFSETS[1]  = 48    → output of layer 0 / input to layer 1    (24 floats)
BUFFER_OFFSETS[2]  = 72    → output of layer 1 / input to layer 2    (24 floats)
...
BUFFER_OFFSETS[12] = 252   → end of buffer (total size)
```

The size of each buffer slot equals the output size of the corresponding layer. The array has `NUM_TRAINABLE_LAYERS + 2` entries (one for the frozen output, one per layer output, and one for the end marker).

During `ODT_Init()`, the library sets up pointers:
```c
ctx->intermediate_buffers[i] = &INTERMEDIATE_BUFFERS[BUFFER_OFFSETS[i]];
ctx->gradient_buffers[i]     = &GRADIENT_BUFFERS[BUFFER_OFFSETS[i]];
```

### 2.8 Extern Declarations

```c
extern float ALL_WEIGHTS[TOTAL_PARAMS];
extern float ALL_BEST_WEIGHTS[TOTAL_PARAMS];
extern float INTERMEDIATE_BUFFERS[TOTAL_INTERMEDIATE_BUFFER_SIZE];
extern float GRADIENT_BUFFERS[TOTAL_GRADIENT_BUFFER_SIZE];

#if USE_GRADIENT_ACCUMULATION
extern float ALL_WEIGHT_GRADS[TOTAL_PARAMS];
#endif
```

These arrays are defined in `trainable_model_config.c` and referenced by the training library.

---

## 3. trainable_model_config.c

### 3.1 Weight Storage

```c
#pragma DATA_SECTION(ALL_WEIGHTS, "trainable_parameters")
float ALL_WEIGHTS[TOTAL_PARAMS] = {
    // Layer 0 weights (1152 floats): W[24][48] row-major
    0.01234567f, -0.02345678f, ...
    // Layer 0 biases (24 floats)
    0.00012345f, ...
    // Layer 2 weights (288 floats): W[12][24] row-major
    ...
    // ... all layers concatenated ...
};

#pragma DATA_SECTION(ALL_BEST_WEIGHTS, "trainable_best_weights")
float ALL_BEST_WEIGHTS[TOTAL_PARAMS];

#if USE_GRADIENT_ACCUMULATION
#pragma DATA_SECTION(ALL_WEIGHT_GRADS, "trainable_weight_grads")
float ALL_WEIGHT_GRADS[TOTAL_PARAMS];
#endif
```

**`ALL_WEIGHTS`**: Contains the initial values for all trainable parameters, exported from the PC-trained model. This is the single source of truth for model parameters — all layers reference into this array via offsets.

**`ALL_BEST_WEIGHTS`**: Uninitialized storage of the same size. During training, whenever a new best validation loss is achieved, the current weights are copied here via `ODT_SaveBestWeights()`. When training ends, the best weights are restored via `ODT_LoadBestWeights()`.

**`ALL_WEIGHT_GRADS`**: Only allocated when `TRAIN_BATCH_SIZE > 1`. Stores accumulated weight and bias gradients across multiple samples in a batch. When batch_size=1, gradients are applied immediately during the backward pass and this buffer is not needed.

### 3.2 Buffer Storage

```c
#pragma DATA_SECTION(INTERMEDIATE_BUFFERS, "intermediate_buffers")
float INTERMEDIATE_BUFFERS[TOTAL_INTERMEDIATE_BUFFER_SIZE];

#pragma DATA_SECTION(GRADIENT_BUFFERS, "gradient_buffers")
float GRADIENT_BUFFERS[TOTAL_GRADIENT_BUFFER_SIZE];
```

**`INTERMEDIATE_BUFFERS`**: Stores forward pass activations. Each layer's output is written here during the forward pass and read during the backward pass (needed for gradient computation).

**`GRADIENT_BUFFERS`**: Stores backward pass gradients. Each layer's input gradient is written here during the backward pass.

### 3.3 Memory Section Placement

All arrays use `#pragma DATA_SECTION` to place them in named memory sections:

| Section Name | Array | Typical Placement | Rationale |
|-------------|-------|-------------------|-----------|
| `trainable_parameters` | `ALL_WEIGHTS` | RAM | Must be writable (updated during training) |
| `trainable_best_weights` | `ALL_BEST_WEIGHTS` | RAM | Must be writable (checkpoint storage) |
| `intermediate_buffers` | `INTERMEDIATE_BUFFERS` | RAM | Scratch space, frequently read/written |
| `gradient_buffers` | `GRADIENT_BUFFERS` | RAM | Scratch space, frequently read/written |
| `trainable_weight_grads` | `ALL_WEIGHT_GRADS` | RAM | Gradient accumulation (batch > 1 only) |

These section names are referenced in the project's linker command files (`ram_lnk.cmd` / `flash_lnk.cmd`) to control physical memory placement. This allows embedded engineers to optimize memory layout — for example, placing weight arrays in fast SRAM while keeping training data in slower flash.

---

## 4. How `trainable_layers_from_last` Maps to These Files

The `trainable_layers_from_last` (k) parameter controls the split point, but the relationship between k and the number of layers in these files is often not 1:1:

- **k counts only main layers** — layers with trainable parameters (e.g., Linear/Gemm)
- **All layers between the split point and output are included** — including activation layers (ReLU) that have no trainable parameters

### How the split works

The pipeline walks backward from the model output, counting only main layers (those with weights). When it reaches the k-th main layer, it cuts the graph *before* that layer. Everything from the cut point to the output becomes the trainable portion — including any activation layers in between.

### Example

Consider a model with 5 layers:

```
Full model:  Linear_A → ReLU → Linear_B → ReLU → Linear_C
               (1)               (2)               (3)       ← main layer count
```

**With `trainable_layers_from_last: 1`** — count 1 main layer from the end:

```
Frozen:     Linear_A → ReLU → Linear_B → ReLU
Trainable:  Linear_C

NUM_TRAINABLE_LAYERS = 1   (1 Linear)
```

**With `trainable_layers_from_last: 2`** — count 2 main layers from the end:

```
Frozen:     Linear_A → ReLU
Trainable:  Linear_B → ReLU → Linear_C

NUM_TRAINABLE_LAYERS = 3   (2 Linear + 1 ReLU between them)
```

**With `trainable_layers_from_last: 3`** — count 3 main layers (all of them):

```
Frozen:     (empty, or just Flatten/Reshape if present)
Trainable:  Linear_A → ReLU → Linear_B → ReLU → Linear_C

NUM_TRAINABLE_LAYERS = 5   (3 Linear + 2 ReLU between them)
```

Notice that `NUM_TRAINABLE_LAYERS` in the generated config is always ≥ k, because activation layers between the main layers are included automatically. The value of k determines *where to cut* the graph, not *how many layers end up in the trainable portion*.

---

## 5. Further Reading

- **How the library uses these files at runtime** → [On-Device Training Library](ondevice_training_lib.md)
- **Understanding the embedded training data** → [On-Device Training Data](ondevice_training_data.md)
- **How these files are generated** → [Running ModelZoo for On-Device Training](running_modelzoo_for_odt.md)