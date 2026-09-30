# Integer On-Device Training: Overview

## Table of Contents

- [1. What is Integer ODT?](#1-what-is-integer-odt)
- [2. Why Integer Arithmetic?](#2-why-integer-arithmetic)
- [3. The NITI Algorithm](#3-the-niti-algorithm)
- [4. The Frozen + Trainable Split for Integer Models](#4-the-frozen--trainable-split-for-integer-models)
- [5. Full Model vs Partial Model](#5-full-model-vs-partial-model)
- [6. End-to-End Workflow](#6-end-to-end-workflow)
- [7. Supported Configurations](#7-supported-configurations)
- [8. Memory Considerations](#8-memory-considerations)
- [9. Limitations](#9-limitations)
- [10. Further Reading](#10-further-reading)

---

## 1. What is Integer ODT?

Integer on-device training is the ability for an INT8-quantized neural network to **continue training directly on a microcontroller** using integer-only arithmetic. After deployment, the device can adapt its model weights to the local operating environment without any floating point operations during the training pass.

**Standard inference-only flow:**
```
Train on PC (float) → Quantize to INT8 → Compile → Deploy → Inference only (frozen INT8)
```

**Integer ODT flow:**
```
Train on PC (QAT) → Export INT8 model → Compile frozen part → Deploy → Inference + Training on MCU (all INT8)
```

---

## 2. Why Integer Arithmetic?

### Speed

Cortex-M33 and similar MCUs execute 8-bit and 16-bit integer operations faster than 32-bit float operations. The bottleneck in a training pass is the accumulator math (multiply-accumulate across weight matrices). Integer MACs run faster than float MACs on these cores.

### Memory

INT8 weights take **4× less RAM** than float32 weights. For a model with 50,000 parameters:
- Float weights: 50,000 × 4 bytes = 200 KB
- INT8 weights:  50,000 × 1 byte  =  50 KB

This matters enormously on MCUs where total RAM may be 32KB KB to 512 KB.

### NPU Compatibility

We can futher take the training Int8 weights and pack the weights as needed by NPU, we can then do the inference on NPU as well. 

### Stability

A correctly calibrated QAT model has weights whose integer representation is already meaningful: the quantization error is small and the model is trained to operate in INT8 space. Integer ODT fine-tunes these weights in the same space they were trained in, which produces stable and predictable gradient updates.

---

## 3. The NITI Algorithm

NITI is the gradient quantization scheme used by the integer ODT library. It solves the core problem of integer training: **gradients are large, weights are INT8, how do you update weights without overflow and without discarding all gradient information?**

### The Problem

A naive approach would use a fixed right-shift to scale gradients into INT8 range before adding to weights. But gradient magnitudes vary enormously during training. A shift that works for large gradients early in training causes precision loss for small gradients later.

### The Solution: Dynamic Shifts

NITI computes a fresh shift each batch based on the actual gradient magnitudes seen in that batch:

```
Given: accumulated gradient tensor G for one weight matrix

Step 1: Find maximum absolute value
    max_val = max(|G|)

Step 2: Compute minimum shift to fit in target_bits
    bits_needed = ceil(log2(max_val + 1))
    shift = max(0, bits_needed - target_bits)

Step 3: Shift and round
    G_quantized = round(G >> shift)   // round-half-up

Step 4: Apply learning rate scaling
    weight_shift = max(0, shift - mu_weight)
    weight_delta = round(G_quantized >> weight_shift)

Step 5: Update weights
    weight_new = sat_int8(weight - weight_delta)
```

The `mu_weight` parameter controls how aggressively the shift is reduced before updating. 

For delta propagation (passing gradient to lower layers), `target_bits=7` so the propagated delta fits in INT8 without accumulation overflow.

### Why This Works

The shift adapts automatically across:
- **Different batch sizes:** larger batches accumulate larger gradients; NITI scales appropriately
- **Different training stages:** large gradients early in training, small gradients near convergence
### Worked Example
---

## 4. The Frozen + Trainable Split for Integer Models

The model is split into two parts at export time:

```
Raw Float Input
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TVM Frozen Model (mod.a)                         │
│                                                                     │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐  │
│  │  PREQUANT    │────►│  Frozen Compute Layers (partial model)   │  │
│  │  (float→INT8)│     │  or nothing (full model)                 │  │
│  └──────────────┘     └──────────────────────────────────────────┘  │
│                                                                     │
│  Input: float32                      Output: INT8 (full model)      │
│                                              UINT8 (partial model)  │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼  INT8 or UINT8
┌─────────────────────────────────────────────────────────────────────┐
│                Integer ODT Library (integer_odl.c)                  │
│                                                                     │
│   INT_LAYERS[] — array of IntLayerDesc_t                            │
│                                                                     │
│   [Layer 0: Conv2D/Linear]                                          │
│   [Layer 1: Conv2D_RELU/Linear_RELU]                                │
│   ...                                                               │
│   [Last Layer: Linear → logits]                                     │
│                                                                     │
│  Weights: INT8 (mutable, updated by backward pass)                  │
│  Offsets: INT32 (mutable, updated by backward pass)                 │
│  k array: INT8 (OSS right-shift per filter/output)                  │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼  INT8 logits
┌──────────────┐
│  main.c      │  compute_loss()  →  writes dy to ctx->grad_buf[0]
│  User code   │  INTODT_Backward() ← reads dy from grad_buf[0]
└──────────────┘
```

### What the TVM Frozen Model Contains

**Full model ODT** (`trainable_layers_from_last == total compute layers`):
The frozen model contains only the **PREQUANT block**: the float-to-INT8 quantization operation. It receives float input from the feature extractor and outputs INT8 to the integer ODT library.

**Partial model ODT** (`trainable_layers_from_last < total compute layers`):
The frozen model contains the PREQUANT block plus the frozen compute layers (Conv2D, Linear, ReLU up to the split point). Its output is UINT8 because the last frozen layer has ReLU and its output range is [0, 255].

### What the C Library Contains

`INT_LAYERS[]` is an array of `IntLayerDesc_t` structs, one per trainable layer. Each descriptor holds:
- The layer type (`INT_LAYER_CONV2D`, `INT_LAYER_LINEAR_RELU`, etc.)
- Pointers to mutable weight and offset arrays
- Pointers to const k (shift) and mult arrays extracted from the ONNX OSS chain
- Pointers to activation buffer and ReLU mask buffer
- Shape metadata (in_size, out_size, n_filters, kH, kW, etc.)

The PREQUANT layer is never in `INT_LAYERS[]`, even when running full model ODT. It lives in the TVM frozen model.

---

## 5. Full Model vs Partial Model

### Full Model ODT

All compute layers (Conv, Linear) are trainable. `trainable_layers_from_last` equals the total number of compute layers.

```
                TVM frozen model               │  C library (integer_odl.c)
─────────────────────────────────────────────────────────────────────────────
float input → [PREQUANT] → INT8 output    →    [Conv RELU] → [Linear RELU] → [Linear] → logits
                                               ╔═══════════════════════════════════════╗
                                               ║     All layers trainable              ║
                                               ╚═══════════════════════════════════════╝
```

The TVM model performs float→INT8 quantization. All actual computation happens in the C library.
Use this when: the full model is small enough to retrain, you want maximum adaptation capability.

### Partial Model ODT

Only the last k layers are trainable. Earlier compute layers are frozen in TVM.

```
         TVM frozen model                             │  C library
─────────────────────────────────────────────────────────────────────────────────
float input → [PREQUANT] → [Conv RELU] → UINT8 out → [Linear RELU] → [Linear] → logits
              ╔══════════════════════════════════╗    ╔═════════════════════════╗
              ║       Frozen — runs on NPU       ║    ║ Trainable — updates CPU ║
              ╚══════════════════════════════════╝    ╚═════════════════════════╝
```

The NPU handles the frozen computation. The CPU only trains the last few layers.
Use this when: the full model is large and you only need the task-specific layers to adapt.


Currently in partial model training, training data is still stored as float32 raw inputs even for partial model ODT. Your `main.c` must call TVM for every training sample to get the activations before passing to `INTODT_Forward`. See [Limitations](#9-limitations).

### Controlling the Split

The `trainable_layers_from_last` YAML field controls how many **compute layers** from the end are trainable. Compute layers are Conv and Linear/MatMul nodes; activation layers (ReLU) and reshape nodes are not counted.

| `trainable_layers_from_last` | Effect on arc fault model (Conv→Linear→Linear) |
|------------------------------|------------------------------------------------|
| `1` | Only last Linear trainable (Conv and first Linear frozen) |
| `2` | Last 2 Linears trainable (Conv frozen) |
| `3` | All layers trainable, full model ODT |

---

## 6. End-to-End Workflow

```
         PC Side (ModelMaker)                            Device Side (MCU)
┌──────────────────────────────────────────┐    ┌────────────────────────────────────────┐
│                                          │    │                                        │
│  1. QAT training                         │    │  7. Import CCS project                 │
│     (quantization: 2 in YAML)            │    │     (tinyml-sdk arc_fault_odl)         │
│              ↓                           │    │              ↓                         │
│  2. Export QAT ONNX                      │    │  8. Copy artifacts to project          │
│     (INT8 weights, OSS chain)            │    │     (mod.a, integer_model_config, etc) │
│              ↓                           │    │              ↓                         │
│  3. Analyze output distribution          │    │  9. Build and flash                    │
│     → TARGET_MAG_CORRECT/WRONG           │    │              ↓                         │
│              ↓                           │    │  10. Initialization                    │
│  4. Parse ONNX graph                     │    │      (INT_MODEL, INT_LAYERS verified)  │
│     → layer types, weights, k arrays     │    │              ↓                         │
│     → full vs partial detection          │    │  11. Pre-training accuracy             │
│              ↓                           │    │      (baseline evaluation)             │
│  5. Generate integer_model_config.h/.c   │    │              ↓                         │
│     + train_data.h/.c                    │    │  12. Training loop                     │
│              ↓                           │    │      INTODT_Forward + compute_loss +   │
│  6. Generate frozen_model/model.onnx     │    │      INTODT_Backward per batch         │
│     → TVM compiles → mod.a               │    │              ↓                         │
│                                          │    │  13. Post-training accuracy            │
└──────────────────────────────────────────┘    │      (evaluation with updated weights) │
                                                └────────────────────────────────────────┘
```

### PC-Side Steps (What ModelMaker Does Automatically)

**Step 1: QAT training.** The model trains using Quantization-Aware Training (`quantization: 2`). During QAT, fake quantization is applied at every layer, forcing the model to learn INT8-compatible weights. The result is a model whose weights, when rounded to INT8, still produce accurate outputs.

**Step 2: ONNX export.** The trained model is exported as ONNX. Each layer appears as a compute node (Conv/MatMul/Gemm) followed by an OSS (Output Shift and Scale) chain: `Add(offset) → Mul(scale1) → Mul(scale2) → Floor → Clip`. This encodes the per-channel quantization shifts that the C library uses for the k arrays.

**Step 3: Output distribution analysis.** ModelMaker runs the QAT ONNX on the exported training samples using onnxruntime and measures the distribution of output logits for each class. From this it derives:
- `TARGET_MAG_CORRECT[c]` = mean logit for class c when the sample truly belongs to class c
- `TARGET_MAG_WRONG[c]` = mean logit for class c when the sample belongs to a different class

These values are auto-calibrated to the specific model and dataset. They are written as INT8 constants in `integer_model_config.h`.

**Step 4: ONNX graph parsing.** The ONNX graph is segmented into blocks and classified. Each block type maps to a layer type in the C enum. Weights, offsets, and k arrays are extracted.

**Step 5: C artifact generation.** `integer_model_config.h` and `integer_model_config.c` are written with all layer descriptors, weight arrays, and static buffers. `train_data.h` and `train_data.c` are written with the training samples.

**Step 6: Frozen model generation.** The ONNX graph is split at the trainable boundary. The frozen subgraph is saved and compiled by TVM into `mod.a`.

### Device-Side Steps (What Your Code Does)

**Step 10: Initialization.** `INT_MODEL` is initialized by the generated `integer_model_config.c`. No runtime initialization call is needed; the `INT_MODEL` struct is statically initialized with all pointers set.

**Step 11: Pre-training accuracy.** Call `INTODT_Infer` on test samples to establish a baseline accuracy before training begins.

**Step 12: Training loop.** For each epoch and each batch:
1. `INTODT_Forward(&INT_MODEL, batch, size)`: runs forward pass, fills `act_buf` and `relu_mask`
2. Your `compute_loss()`: reads logits from last layer's `act_buf`, writes `dy` to `grad_buf[0]`
3. `INTODT_Backward(&INT_MODEL, batch, size)`: reads `dy` from `grad_buf[0]`, updates weights

**Step 13: Post-training accuracy.** Call `INTODT_Infer` again to measure improvement.

---

## 7. Supported Configurations

| Category | Supported | Notes |
|----------|-----------|-------|
| **Quantization** | INT8 weights (QAT) | Requires `quantization: 2` in YAML |
| **Layer types** | Conv2D, Conv2D+ReLU, Linear, Linear+ReLU | |
| **Grouped conv** | Not supported | Raises ValueError at export time |
| **Dilated conv** | Not supported | Raises ValueError at export time |
| **Tasks** | Classification | N-class output with per-class logits |
| **Loss function** | User-written MSE with TARGET_MAG | See [Integer ODL Library](integer_odl_lib.md) |
| **Optimizer** | NITI (integer SGD) | Adaptive shifts via effective_bits |
| **Training modes** | Partial (last k layers) or full (all layers) | Controlled by `trainable_layers_from_last` |
| **Input to C library** | INT8 (full model) or UINT8 (partial model) | Determined automatically by first layer type |
| **Example devices** | AM13E230X (M33 core) | C library works on any MCU |

---

## 8. Memory Considerations

### Weight Storage

INT8 weights use 1 byte per parameter. For a model with CONV(3 filters, 16×2 kernel) + Linear(147→160) + Linear(160→2):

| Component | Parameters | INT8 bytes |
|-----------|-----------|-----------|
| Conv weights (3×1×16×2) | 96 | 96 B |
| Conv offsets (3) | 3 | 12 B (int32) |
| Linear1 weights (147×160) | 23,520 | 23,520 B |
| Linear1 offsets (160) | 160 | 640 B (int32) |
| Linear2 weights (160×2) | 320 | 320 B |
| Linear2 offsets (2) | 2 | 8 B (int32) |
| **Total weights** | | **~24 KB** |

### Activation Buffers

Each compute layer needs `BATCH_SIZE * out_size` bytes for `act_buf`:
- Conv out: BATCH_SIZE × (3 × 49) = 8 × 147 = 1,176 bytes
- Linear1 out: BATCH_SIZE × 160 = 8 × 160 = 1,280 bytes
- Linear2 out: BATCH_SIZE × 2 = 8 × 2 = 16 bytes

### ReLU Mask Buffers

RELU layers need `BATCH_SIZE * ceil(out_size / 8)` bytes:
- Conv ReLU mask: 8 × ceil(147/8) = 8 × 19 = 152 bytes
- Linear1 ReLU mask: 8 × ceil(160/8) = 8 × 20 = 160 bytes

### Grad Buffers

Two ping-pong buffers sized to `BATCH_SIZE * max(max_out_size, first_layer_in_size)`:
- Max out size = max(147, 160, 2) = 160
- First layer in_size = 128 (raw input after PREQUANT)
- grad_buf size = 2 × (8 × 160) = 1,280 bytes

### Training Data

`TRAIN_INPUTS[N_TRAIN][INPUT_SIZE]` as float32:
- 20 samples × 128 floats × 4 bytes = 10,240 bytes = 10 KB

Total approximate RAM for the arc fault example: **~30 KB** (weights + buffers).

---

## 9. Limitations

### Linear Model Graph Only

The model must have a linear (sequential) graph with no skip connections, residual connections, or branches. The ONNX parser expects a strictly sequential topology: each block connects to exactly one next block.

### Only Classification Task

The integer ODT framework currently supports classification (N-class output with per-class logits and MSE loss against TARGET_MAG targets). Regression and anomaly detection etc are not currently supported however the underlying training framework is general and can be extended to these tasks types. 

### Trained Weights in RAM: Not Persisted Across Power Cycles

After on-device training completes, the updated weights exist only in RAM. `INT_LAYERS[i].weights` and `INT_LAYERS[i].offset` are mutable arrays in `.data` section; they are lost on power cycle. On reset, weights reload from the initial values baked into `integer_model_config.c` in flash.

**To persist trained weights:** Use your device's flash write API to save the weight arrays to non-volatile storage, and restore them at boot before calling any `INTODT_*` function. The arrays to save are all `layer{i}_weights` and `layer{i}_offset` arrays declared in `integer_model_config.c`. Sizes are deterministic and compile-time-known: snapshot them after training and restore them at startup. The exact flash write API is device-specific (refer to your MCU SDK).

---

## 10. Further Reading

- [Running ModelMaker for Integer ODT](running_modelmaker_for_integer_odt.md): how to generate artifacts
- [Integer ODL Library](integer_odl_lib.md): NITI math, C API, custom training loop
- [Float ODT Overview](../ondevice_training/overview.md): conceptual background on frozen+trainable split
