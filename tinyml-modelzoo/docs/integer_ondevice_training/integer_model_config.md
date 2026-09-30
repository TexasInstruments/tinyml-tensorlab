# Integer Model Configuration: Deep Dive

## Table of Contents

- [1. Overview](#1-overview)
- [2. integer_model_config.h](#2-integer_model_configh)
- [3. IntLayerDesc_t: Layer Descriptor Structure](#3-intlayerdesc_t-layer-descriptor-structure)
  - [3.1 Common Fields](#31-common-fields)
  - [3.2 PREQUANT Sub-struct](#32-prequant-sub-struct)
  - [3.3 Conv Sub-struct](#33-conv-sub-struct)
  - [3.4 Linear Sub-struct](#34-linear-sub-struct)
- [4. IntModelCtx_t: Model Context](#4-intmodelctx_t-model-context)
- [5. integer_model_config.c: Buffer Layout and Sizing](#5-integer_model_configc-buffer-layout-and-sizing)
  - [5.1 Activation Buffers](#51-activation-buffers)
  - [5.2 ReLU Mask Buffers](#52-relu-mask-buffers)
  - [5.3 Grad Buffers](#53-grad-buffers)
  - [5.4 Weight and Parameter Arrays](#54-weight-and-parameter-arrays)

---

## 1. Overview

`integer_model_config.h` and `integer_model_config.c` are the two auto-generated files that fully describe your model to the integer ODT C library. Together they contain:
- All INT8 weights and INT32 offsets extracted from the QAT ONNX
- Static buffers for activations, ReLU masks, and gradient ping-pong
- Layer descriptors connecting every buffer and array pointer
- The `IntModelCtx_t` context struct that the three public API functions operate on

After these files are compiled into your project, you never need to initialize or configure the model. `INT_MODEL` is statically initialized and ready to use immediately.

---

## 2. integer_model_config.h

```c
#ifndef INTEGER_MODEL_CONFIG_H
#define INTEGER_MODEL_CONFIG_H

#include "integer_odl.h"
```

The header always includes `integer_odl.h` so that `IntLayerDesc_t`, `IntModelCtx_t`, and `IntLayerType_t` are in scope everywhere `integer_model_config.h` is included.

---

### IS_FULL_MODEL_RETRAIN

```c
#define IS_FULL_MODEL_RETRAIN  1   // 1 = full model, 0 = partial model
```

Tells `main.c` how the model is split:
- `1` (full model): TVM outputs INT8 to the C library. In `main.c`, pass raw float training inputs directly to TVM; TVM does the quantization.
- `0` (partial model): TVM outputs UINT8 to the C library. In `main.c`, similarly pass raw inputs to TVM, but TVM runs the frozen compute layers too before passing UINT8 to the C library.

Your `main.c` can use this define to branch behavior, or always pass floats to TVM and let TVM handle the type conversion. The difference is how much computation TVM does before the C library takes over.

---

### INPUT_SIZE

```c
#define INPUT_SIZE  128
```

The flat size of one raw input sample **before** feature extraction.

---

### FROZEN_OUTPUT_SIZE

```c
#define FROZEN_OUTPUT_SIZE  128
```

The size of TVM's output, the input to `INT_LAYERS[0]`. For full model ODT where TVM only does PREQUANT, this equals the PREQUANT output size, which is the same as INPUT_SIZE. For partial model ODT, this is the spatial size of the last frozen layer's output.

---

### N_CLASSES

```c
#define N_CLASSES  2
```

Number of output classes, equal to `INT_LAYERS[N_INT_LAYERS-1].out_size`. 


---

### BATCH_SIZE

```c
#define BATCH_SIZE  8
```

How many samples are processed together in one forward+backward call. All `act_buf`, `relu_mask`, and `grad_buf` static arrays are sized as `BATCH_SIZE * element_count`.

---

### TARGET_MAG_CORRECT and TARGET_MAG_WRONG

```c
static const int8_t TARGET_MAG_CORRECT[N_CLASSES] = {  24,  20 };
static const int8_t TARGET_MAG_WRONG[N_CLASSES]   = { -31, -28 };
```

These are the per-class loss targets derived from the model's actual output distribution. They answer: "what does the model typically output for class c when the input belongs to class c (CORRECT), and when it belongs to some other class (WRONG)?"


**How to interpret them:**
- `TARGET_MAG_CORRECT[0] = 24`: when a class-0 sample is correctly classified, logit[0] is typically ~24
- `TARGET_MAG_WRONG[0] = -31`: when a class-1 sample is misclassified, logit[0] is typically ~-31

The loss function computes `gradient = logit - target` and pushes each logit toward its target. 

---

### MU_WEIGHT_DEFAULT and MU_OFFSET_DEFAULT

```c
#define MU_WEIGHT_DEFAULT  1
#define MU_OFFSET_DEFAULT  1
```

Initial NITI learning rate parameters. Written into `INT_MODEL.mu_weight` and `INT_MODEL.mu_offset` at initialization.

You can change these at runtime in `main.c` after initialization:

See [Integer ODL Library: NITI Math](integer_odl_lib.md#niti-math) for the effect of different mu values.

---

### N_TRAINABLE_LAYERS

```c
#define N_TRAINABLE_LAYERS  3
```

`N_TRAINABLE_LAYERS`: Number of trainable layers

---

## 3. IntLayerDesc_t: Layer Descriptor Structure

Each entry in `INT_LAYERS[]` is an `IntLayerDesc_t`. Here is the complete structure from `integer_odl.h`:

### 3.1 Common Fields

```c
typedef struct {
    IntLayerType_t type;    // Layer type enum
    uint16_t       in_size; // Flat input element count (per sample, excluding batch)
    uint16_t       out_size;// Flat output element count (per sample, excluding batch)
    int32_t        clip_lo; // Lower bound for output saturation (from ONNX Clip node)
    int32_t        clip_hi; // Upper bound for output saturation (from ONNX Clip node)
    void    *act_buf;       // [BATCH_SIZE * out_size] activation buffer
    uint8_t *relu_mask;     // [BATCH_SIZE * ceil(out_size/8)] bitmask; NULL if no ReLU
    union { ... };          // Layer-type-specific parameters
} IntLayerDesc_t;
```

**`type`**, one of:
```c
INT_LAYER_PREQUANT    = 0  // float→int8 quantization
INT_LAYER_CONV2D      = 1  // Conv2D with OSS, no ReLU
INT_LAYER_CONV2D_RELU = 2  // Conv2D with OSS + ReLU
INT_LAYER_LINEAR      = 3  // Linear/MatMul with OSS, no ReLU
INT_LAYER_LINEAR_RELU = 4  // Linear/MatMul with OSS + ReLU
```

### 3.2 PREQUANT Sub-struct

```c
struct {
    const float *offset;  // Per-channel offset [n_ch]
    const float *scale;   // Per-channel scale [n_ch] (combined mul1*mul2)
    uint16_t     n_ch;    // Number of channels (1 for global scale/offset)
} prequant;
```

Not present in `INT_LAYERS[]` for ModelMaker-generated artifacts, PREQUANT lives in TVM. Present for any future use

### 3.3 Conv Sub-struct

```c
struct {
    int8_t       *weights;   // [n_filters × in_ch × kH × kW] flat row-major
    int32_t      *offset;    // [n_filters] per-filter bias — mutable
    const int8_t *mult;      // [n_filters] OSS multiply factor — const, all 1 currently
    const int8_t *k;         // [n_filters] OSS right-shift amount — const
    uint16_t      n_filters;
    uint16_t      conv_out_h;  // output spatial height
    uint16_t      conv_out_w;  // output spatial width
    uint8_t       kH;          // kernel height
    uint8_t       kW;          // kernel width
} conv;
```

**`weights` layout:** `[filter_idx, in_ch_idx, krow, kcol]` stored contiguously:
```
weights[filter * (in_ch * kH * kW) + in_ch * (kH * kW) + krow * kW + kcol]
```
This is standard ONNX/PyTorch Conv2D weight layout.

**`offset`**: INT32 per-filter bias.

**`k`**:  per-filter right-shift. Derived from the ONNX OSS chain as `k = round(-log2(|shift_mult|))`. For example, if the ONNX Mul constant is `0.03125 = 2^-5`, then `k = 5`. The C forward pass computes `(acc + offset[f]) >> k[f]`. See [Section 6](#6-the-k-array--what-it-is-and-where-it-comes-from).

### 3.4 Linear Sub-struct

```c
struct {
    int8_t       *weights;  // [in_size × out_size] row-major — mutable
    int32_t      *offset;   // [out_size] per-output bias — mutable
    const int8_t *mult;     // [out_size] — const, all 1
    const int8_t *k;        // [out_size] — const
} linear;
```

**`weights` layout:** `weights[in_idx * out_size + out_idx]` 

---

## 4. IntModelCtx_t: Model Context

```c
typedef struct {
    uint16_t        n_layers;
    IntLayerDesc_t *layers;
    int8_t         *grad_buf[2];  // ping-pong delta buffers
    int8_t          mu_weight;    // NITI learning rate for weights
    int8_t          mu_offset;    // NITI learning rate for offsets
    IntInputType_t  input_type;   // INT_INPUT_INT8 or INT_INPUT_UINT8
} IntModelCtx_t;
```

**`grad_buf[2]`**: two int8 buffers for ping-pong use during backward pass. Sized to `BATCH_SIZE * max(max_out_size_across_all_layers, layers[0].in_size)`.

The two-buffer ping-pong works as follows:
1. Before `INTODT_Backward`: user writes `dy` (loss gradient) into `grad_buf[0]`
2. Backward processes the last layer: reads from `grad_buf[0]`, writes propagated delta to `grad_buf[1]`
3. Backward processes second-to-last layer: reads from `grad_buf[1]`, writes to `grad_buf[0]`
4. Alternates until the first trainable layer is processed

During `INTODT_Infer`, `grad_buf[0/1]` are used as temporary layer activation storage (since `act_buf` is not written during inference). The ping-pong buffers are reused across forward and inference, safe because they are never used simultaneously.

---

## 5. integer_model_config.c: Buffer Layout and Sizing

### 5.1 Activation Buffers

Each layer has its own activation buffer, sized at compile time:

```c
// RELU layer → uint8_t (output range [0, 255])
static uint8_t  layer0_act_buf[BATCH_SIZE * output_size];  

// Non-RELU layer → int8_t (output range [-128, 127])
static int8_t   layer2_act_buf[BATCH_SIZE * output_size];
```

During `INTODT_Forward`, each sample's output is written to `layer->act_buf + sample_idx * out_size`.

During `INTODT_Backward`, each layer reads its input from `layers[layer_idx-1].act_buf`

### 5.2 ReLU Mask Buffers

RELU layers need a bitmask per output element per sample:

```c
static uint8_t layer0_relu_mask[BATCH_SIZE * ceil(out_size / 8)];
```

### 5.3 Grad Buffers

Two ping-pong grad buffers shared across all layers:

```c
static int8_t grad_buf_0[BATCH_SIZE * MAX_BUF];
static int8_t grad_buf_1[BATCH_SIZE * MAX_BUF];
```

Where `MAX_BUF = max(max_out_size_across_all_layers, layers[0].in_size)`:

### 5.4 Weight and Parameter Arrays

Weight array naming convention: `layer{i}_weights`, `layer{i}_offset`, `layer{i}_mult`, `layer{i}_k`:

```c
//layer weights — int8_t, mutable (updated by backward)
static int8_t layer0_weights[] = {
    -12,   8,  23, -41,  17, -33,  52, -28,  11, -19,  44, -37,  29, -8,  15, -22,
     33, -14,  -7,  41, -26,  18,  -9,  37, -45,  22,  13, -31,   8, 42, -17,  25,
    ...
};

//layer offsets — int32_t, mutable
static int32_t layer0_offset[] = { 142, -88, 213 };

// OSS multiply
static const int8_t layer0_mult[] = { 1, 1, 1 };

// OSS right-shift amounts 
static const int8_t layer0_k[] = { 9, 8, 9 };
```

**What gets modified during ODT:** Only `weights` and `offset` arrays. The `mult` and `k` arrays are `const` and remain unchanged for the lifetime of the model.

---
