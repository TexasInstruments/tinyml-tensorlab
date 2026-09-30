# Generated Artifacts Overview

## Table of Contents

- [1. Overview](#1-overview)
- [2. Artifact Summary](#2-artifact-summary)
- [3. Standard Artifacts](#3-standard-artifacts)
- [4. Integer ODT-Specific Artifacts](#4-integer-odt-specific-artifacts)
  - [4.1 integer_model_config.h](#41-integer_model_configh)
  - [4.2 integer_model_config.c](#42-integer_model_configc)
  - [4.3 train_data.h](#43-train_datah)
  - [4.4 train_data.c](#44-train_datac)
- [5. How Artifacts Connect at Runtime](#5-how-artifacts-connect-at-runtime)
- [6. Differences from Float ODT Artifacts](#6-differences-from-float-odt-artifacts)
- [7. Next Steps](#7-next-steps)

---

## 1. Overview

When ModelMaker runs with `ondevice_training: true` and `quantization: 2`, it generates a set of files that together define everything needed to run integer on-device training on a microcontroller. This page provides a complete map of all generated files, what each contains and where it fits in the system.

---

## 2. Artifact Summary

| File | Integer ODT-Specific? | Purpose |
|------|-----------------------|---------|
| `mod.a` | No (same as inference) | Compiled frozen model (TVM output, includes PREQUANT for full model) |
| `tvmgen_default.h` | No | API header for the frozen model |
| `user_input_config.h` | No | Feature extraction config |
| `test_vector.c` | No | Golden vectors for device validation |
| `integer_model_config.h` | **Yes** | Defines: INPUT_SIZE, N_CLASSES, BATCH_SIZE, TARGET_MAG arrays, extern layer decls |
| `integer_model_config.c` | **Yes** | INT8 weights, INT32 offsets, k/mult arrays, layer descriptors, model context |
| `train_data.h` | **Yes** | Dimension defines, extern declarations for training/val/test arrays |
| `train_data.c` | **Yes** | Float training samples (TRAIN_INPUTS) and INT8 labels (TRAIN_LABELS) |

Note on `mod.a` for integer ODT: the frozen model compiled for integer ODT is different from the standard inference `mod.a`. For full model ODT, this `mod.a` contains only the PREQUANT block (float→INT8), not the full inference model. For partial model ODT, it contains PREQUANT plus the frozen compute layers. Never mix a standard inference `mod.a` with integer ODT artifacts.

---

## 3. Standard Artifacts

### mod.a: Compiled Frozen Model

Compiled by TVM from `frozen_model/model.onnx`. For integer ODT this model is different from standard inference:

**Full model ODT:** `mod.a` implements only the PREQUANT operation, float32 input → INT8 output. The entire model computation happens in the integer ODT C library. The TVM model is essentially a per-channel float-to-int8 quantizer.

**Partial model ODT:** `mod.a` implements PREQUANT + frozen compute layers. Its output is UINT8 (because the last frozen layer will have ReLU and its output is in [0,255]).

Calling pattern in application code:
```c
struct tvmgen_default_inputs  inputs  = { .input  = (void *)raw_float_input };
struct tvmgen_default_outputs outputs = { .output = (void *)frozen_output_buf };
tvmgen_default_run(&inputs, &outputs);
// frozen_output_buf now has INT8 (full model) or UINT8 (partial model) data
// ready to pass to INTODT_Forward
```

### tvmgen_default.h: Frozen Model API

Same header as standard inference. Declares the input/output structs and the `tvmgen_default_run()` function. 

### user_input_config.h: Feature Extraction Config

Same format as standard inference. Defines feature extraction flags (FFT, binning, logging) and parameters (frame size, feature size, channel count). Used by the `feature_extract` library before calling TVM.

### test_vector.c: Golden Vectors

Contains a known input-output pair computed on the PC. Used for device-side validation after flashing.

---

## 4. Integer ODT-Specific Artifacts

### 4.1 integer_model_config.h

This header defines all compile-time constants and declares the model descriptor arrays. It is the first file to look at when understanding a generated model.

**Full annotated example:**
```c
#ifndef INTEGER_MODEL_CONFIG_H
#define INTEGER_MODEL_CONFIG_H

#include "integer_odl.h"   // IntLayerDesc_t, IntModelCtx_t, IntLayerType_t

// ── Model mode ──────────────────────────────────────────────────────────────
// 1 = full model ODT (PREQUANT in TVM, INT8 input to C library)
// 0 = partial model ODT (RELU output of frozen layers, UINT8 input to C library)
#define IS_FULL_MODEL_RETRAIN  1

// ── Model dimensions ────────────────────────────────────────────────────────
// INPUT_SIZE: flat size of raw float input to the model (before feature extraction)
//   For arc fault: 128 time samples × 1 channel = 128
#define INPUT_SIZE          128

// FROZEN_OUTPUT_SIZE: the size of the TVM model's output — i.e., the input to INT_LAYERS[0]
//   For full model: size of PREQUANT output = same as INPUT_SIZE
//   For partial model: size of last frozen layer's output
#define FROZEN_OUTPUT_SIZE  128

// N_CLASSES: number of output logits (last layer's out_size)
#define N_CLASSES           2

// BATCH_SIZE: number of samples processed together per forward+backward call
//   Must match the actual batch_size used in your training loop
#define BATCH_SIZE          8

// ── Loss function targets ────────────────────────────────────────────────────
// TARGET_MAG_CORRECT[c]: target INT8 logit for class c when the sample IS class c
//   Derived from the model's actual output distribution on training samples.
//   ODT pushes logit[c] toward this value when predicting class c correctly.
static const int8_t TARGET_MAG_CORRECT[N_CLASSES] = {  24,  20 };

// TARGET_MAG_WRONG[c]: target INT8 logit for class c when the sample is NOT class c
//   Typically negative — ODT pushes logit[c] away from the correct class.
static const int8_t TARGET_MAG_WRONG[N_CLASSES]   = { -31, -28 };

// ── Learning rate controls ───────────────────────────────────────────────────
// mu_weight: NITI learning rate for weight updates. Default 1.
//   Higher = more aggressive weight updates. Range: 0-4 in practice.
#define MU_WEIGHT_DEFAULT   1

// mu_offset: NITI learning rate for bias/offset updates. Default 1.
#define MU_OFFSET_DEFAULT   1

// ── Layer counts ─────────────────────────────────────────────────────────────
// N_INT_LAYERS: total entries in INT_LAYERS[] (all trainable compute layers)
//   For full model with Conv+Linear+Linear: 3
//   Does NOT include PREQUANT — that lives in TVM mod.a
#define N_INT_LAYERS        3

// N_TRAINABLE_LAYERS: subset of N_INT_LAYERS that are actually updated during ODT
//   Equals N_INT_LAYERS for full model. May be less for partial model.
#define N_TRAINABLE_LAYERS  3

// ── Extern declarations ──────────────────────────────────────────────────────
// INT_LAYERS[]: array of layer descriptors — defined in integer_model_config.c
extern IntLayerDesc_t INT_LAYERS[N_INT_LAYERS];

// INT_MODEL: model context — defined in integer_model_config.c
// Initialized with n_layers, layers pointer, grad_buf pointers, mu values, input_type
extern IntModelCtx_t  INT_MODEL;

#endif // INTEGER_MODEL_CONFIG_H
```

**Key points for users:**
- `TARGET_MAG_CORRECT` and `TARGET_MAG_WRONG` are model-specific, they are derived from your actual model's output distribution and will differ between runs/models.
- `BATCH_SIZE=8` is hardcoded. To change it, update it before compilation. The static arrays in `integer_model_config.c` (act_buf, grad_buf, relu_mask) are sized at compile time using this define.
- `IS_FULL_MODEL_RETRAIN` tells `main.c` whether to pass float or int8 data to TVM. For full model ODT, raw feature-extracted floats go to TVM; for partial, they also go to TVM but the output is uint8.

### 4.2 integer_model_config.c

This file defines all layer parameters and static buffers. It is auto-generated and contains:

**Layer topology comment:**
```c
// Model topology
//   layer0: CONV2DRELU    [ 128 ->  147]  (3 filters, 16x2 kernel)
//   layer1: LINEARRELU    [ 147 ->  160]
//   layer2: LINEAR        [ 160 ->    2]
```

**Static buffer declarations** (all sized at compile time):
```c
static int8_t        layer0_weights[];      // Conv weights [n_filters × in_ch × kH × kW]
static int32_t       layer0_offset[];       // Conv biases [n_filters], mutable
static const int8_t  layer0_mult[];         // OSS mult [n_filters], const
static const int8_t  layer0_k[];            // OSS right-shift [n_filters], const
static uint8_t       layer0_act_buf[BATCH_SIZE * 147];     // RELU → uint8
static uint8_t       layer0_relu_mask[BATCH_SIZE * 19];    
// ... layer1, layer2 similarly ...
static int8_t        grad_buf_0[BATCH_SIZE * 160];         // ping-pong delta buffers
static int8_t        grad_buf_1[BATCH_SIZE * 160];
```

**INT_LAYERS[] initializer**, one struct per layer:
```c
IntLayerDesc_t INT_LAYERS[N_INT_LAYERS] = {
    {   // Layer 0: CONV2DRELU
        .type      = INT_LAYER_CONV2D_RELU,
        .in_size   = 128,
        .out_size  = 147,
        .clip_lo   = 0,
        .clip_hi   = 255,
        .act_buf   = layer0_act_buf,
        .relu_mask = layer0_relu_mask,
        .conv = {
            .weights    = layer0_weights,
            .offset     = layer0_offset,
            .mult       = layer0_mult,
            .k          = layer0_k,
            .n_filters  = 3,
            .kH         = 16,
            .kW         = 2,
            .conv_out_h = 113,
            .conv_out_w = 1,
        },
    },
    {   // Layer 1: LINEARRELU
        .type      = INT_LAYER_LINEAR_RELU,
        .in_size   = 147,    
        .out_size  = 160,
        .clip_lo   = 0,
        .clip_hi   = 255,
        .act_buf   = layer1_act_buf,
        .relu_mask = layer1_relu_mask,
        .linear = {
            .weights = layer1_weights,
            .offset  = layer1_offset,
            .mult    = layer1_mult,
            .k       = layer1_k,
        },
    },
    {   // Layer 2: LINEAR (output, no ReLU)
        .type      = INT_LAYER_LINEAR,
        .in_size   = 160,
        .out_size  = 2,
        .clip_lo   = -128,
        .clip_hi   = 127,
        .act_buf   = layer2_act_buf,
        .relu_mask = NULL,    // ← no relu mask for non-RELU layer
        .linear = {
            .weights = layer2_weights,
            .offset  = layer2_offset,
            .mult    = layer2_mult,
            .k       = layer2_k,
        },
    },
};
```

**INT_MODEL context:**
```c
IntModelCtx_t INT_MODEL = {
    .n_layers   = N_INT_LAYERS,
    .layers     = INT_LAYERS,
    .grad_buf   = { grad_buf_0, grad_buf_1 },
    .mu_weight  = MU_WEIGHT_DEFAULT,     // 1
    .mu_offset  = MU_OFFSET_DEFAULT,     // 1
    .input_type = INT_INPUT_INT8,        // INT_INPUT_UINT8 for partial model
};
```

**Weight arrays** (at the end of the file):
```c
// layer0: CONV2DRELU
static int8_t layer0_weights[] = {
    // 3 × 1 × 16 × 2 = 96 values, 16 per row
     -12,   8,  23, -41, ...
    ...
};
static int32_t layer0_offset[] = {
    // 3 values (one per filter)
      142,  -88,  213
};
static const int8_t layer0_mult[] = {
    1, 1, 1    // all 1 — OSS multiply is effectively no-op for current models
};
static const int8_t layer0_k[] = {
    9, 8, 9    // right-shift amounts derived from ONNX: k = round(-log2(|shift|))
};
```

`layer0_weights`, `layer0_offset`, `layer1_weights`, etc. are declared as `static int8_t`, these are the values that `INTODT_Backward` modifies in-place. The `mult` and `k` arrays are `const` and never modified.

### 4.3 train_data.h

Declares the training, validation, and test data arrays:

```c
#ifndef TRAIN_DATA_H
#define TRAIN_DATA_H

#include <stdint.h>
#include "integer_model_config.h"

// Sample counts (total across all classes)
// With export_samples_per_class='[10,5,5]' and 2 classes:
#define N_TRAIN_SAMPLES    20   // 10 per class × 2 classes
#define N_VAL_SAMPLES      10   //  5 per class × 2 classes
#define N_TEST_SAMPLES     10   //  5 per class × 2 classes

// Training inputs: float32, raw feature-extracted data
// For full model ODT: same data that goes through TVM PREQUANT during training
extern const float  TRAIN_INPUTS [N_TRAIN_SAMPLES][INPUT_SIZE];
extern const int8_t TRAIN_LABELS [N_TRAIN_SAMPLES];

extern const float  VAL_INPUTS   [N_VAL_SAMPLES  ][INPUT_SIZE];
extern const int8_t VAL_LABELS   [N_VAL_SAMPLES  ];

extern const float  TEST_INPUTS  [N_TEST_SAMPLES ][INPUT_SIZE];
extern const int8_t TEST_LABELS  [N_TEST_SAMPLES ];

#endif // TRAIN_DATA_H
```

The training inputs are stored as float32 even though the C library works in INT8. Your `main.c` passes them through the TVM frozen model first (which does the float→INT8 conversion), then passes the INT8 result to `INTODT_Forward`. The float values are the same data that TVM was compiled to process.

### 4.4 train_data.c

Contains the actual training data. Each row is annotated with its index and label for debugging:

```c
// 40 total samples (train=20, val=10, test=10)
#include "train_data.h"

const float TRAIN_INPUTS[N_TRAIN_SAMPLES][INPUT_SIZE] = {
    /* [   0] label= 0 */ { 0.01234567f, -0.87654321f,  0.45678901f, ... },
    /* [   1] label= 0 */ { 0.23456789f,  0.12345678f, -0.34567890f, ... },
    ...
    /* [  19] label= 1 */ { -0.98765432f, 0.76543210f,  0.65432198f, ... },
};

const int8_t TRAIN_LABELS[N_TRAIN_SAMPLES] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   // 10 samples of class 0
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1    // 10 samples of class 1
};

const float VAL_INPUTS[N_VAL_SAMPLES][INPUT_SIZE] = { ... };
const int8_t VAL_LABELS[N_VAL_SAMPLES] = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };

const float TEST_INPUTS[N_TEST_SAMPLES][INPUT_SIZE] = { ... };
const int8_t TEST_LABELS[N_TEST_SAMPLES] = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };
```

Training data is shuffled: the TRAIN_INPUTS array is randomly shuffled by ModelMaker before writing. Class 0 and class 1 samples are interleaved, not grouped, so each training batch contains a mix of classes.

---

## 5. How Artifacts Connect at Runtime

```
┌─────────────────────┐
│  TRAIN_INPUTS[][]   │  float[N_TRAIN][INPUT_SIZE]  (from train_data.c)
└──────────┬──────────┘
           │  float* batch pointer
           ▼
┌─────────────────────────────────────────────┐
│         TVM Frozen Model (mod.a)            │
│                                             │
│   tvmgen_default_run(&inputs, &outputs)     │
│   Full model:   PREQUANT → INT8 output      │
│   Partial model: PREQUANT + layers → UINT8  │
└──────────┬──────────────────────────────────┘
           │  INT8* or UINT8* 
           ▼
┌─────────────────────────────────────────────┐
│      INTODT_Forward(&INT_MODEL, batch, sz)  │
│                                             │
│  Layer 0: conv_forward or linear_forward    │
│    reads INT8/UINT8 input                   │
│    writes INT8/UINT8 to layer0_act_buf      │
│    sets layer0_relu_mask bits               │
│                                             │
│  Layer 1: linear_forward                   │
│    reads layer0_act_buf                     │
│    writes to layer1_act_buf                 │
│    sets layer1_relu_mask bits               │
│                                             │
│  Layer 2: linear_forward (output)           │
│    reads layer1_act_buf                     │
│    writes to layer2_act_buf (logits)        │
└──────────┬──────────────────────────────────┘
           │  logits in layer2_act_buf
           ▼
┌─────────────────────────────────────────────┐
│  compute_loss(ctx, TRAIN_LABELS+start, sz)  │  (user code in main.c)
│                                             │
│  Reads: layer2_act_buf[b][c] — INT8 logits  │
│  Reads: TARGET_MAG_CORRECT[], TARGET_MAG_WRONG[] (from integer_model_config.h)
│  Writes: ctx->grad_buf[0][b][c] = clamp(logit - target, -128, 127)
└──────────┬──────────────────────────────────┘
           │  grad_buf[0] filled with dy
           ▼
┌─────────────────────────────────────────────┐
│  INTODT_Backward(&INT_MODEL, batch, sz)     │
│                                             │
│  Layer 2 backward:                          │
│    Propagate delta from grad_buf[0]         │
│    Update layer2_weights, layer2_offset     │
│    Write delta_out to grad_buf[1]           │
│                                             │
│  Apply layer1 relu_mask to grad_buf[1]      │
│                                             │
│  Layer 1 backward:                          │
│    Propagate delta from grad_buf[1]         │
│    Update layer1_weights, layer1_offset     │
│    Write delta_out to grad_buf[0]           │
│                                             │
│  Apply layer0 relu_mask to grad_buf[0]      │
│                                             │
│  Layer 0 backward:                          │
│    Update layer0_weights, layer0_offset     │
│    (No delta propagation for first layer)   │
└─────────────────────────────────────────────┘
```

---

## 6. Next Steps

- **Deep dive into integer_model_config.h/.c** → [Integer Model Configuration](integer_model_config.md)
- **Deep dive into train_data.h/.c** → [Training Data](training_data.md)
- **Understand the C library that uses these artifacts** → [Integer ODL Library](integer_odl_lib.md)
