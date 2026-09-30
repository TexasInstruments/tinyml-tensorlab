# Integer On-Device Training Documentation

## Table of Contents

- [Introduction](#introduction)
- [How This Differs from Float ODT](#how-this-differs-from-float-odt)
- [What You Need](#what-you-need)
- [Key Concepts](#key-concepts)
- [Quickstart](#quickstart)
  - [Step 1: Configure YAML](#step-1-configure-yaml)
  - [Step 2: Run ModelMaker](#step-2-run-modelmaker)
  - [Step 3: Copy Artifacts to CCS Project](#step-3-copy-artifacts-to-ccs-project)
  - [Step 4: Build, Flash, and Run](#step-4-build-flash-and-run)
  - [What to Expect](#what-to-expect)
- [Documentation Map](#documentation-map)
- [Reading Paths](#reading-paths)

---

## Introduction

Integer on-device training (integer ODT) enables deep learning models to **continue training directly on TI microcontrollers using pure integer arithmetic**. Unlike the float ODT framework which uses 32-bit floating point throughout the training pass, integer ODT works entirely in INT8 weights, INT32 accumulators, and integer-only gradient updates.

```
Float ODT:    Train on PC (float) → Deploy → Train on device with float32 math
Integer ODT:  Train on PC (QAT)   → Deploy → Train on device with INT8/INT32 math
```

**Why integer?** The Cortex-M33 and similar MCUs can execute integer MAC operations significantly faster than float. INT8 weights take one quarter the RAM of float weights, freeing memory for larger models or larger training batches. And if needed, we can later extend the inference to happen on NPU if available.

The current example targets the **arc fault classification** application on the **AM13E230X** LaunchPad (M33 core), but the C library is device-agnostic and works on any MCU. 

---

## How This Differs from Float ODT

If you are already familiar with the float ODT framework, here is what changes:

| Aspect | Float ODT | Integer ODT |
|--------|-----------|-------------|
| **YAML `quantization`** | `0` (float) | `2` (QAT, required) |
| **Weight representation** | float32 | INT8 |
| **Gradient arithmetic** | float32 | INT32 accumulate, INT8 store |
| **Learning rate method** | Fixed SGD step | NITI dynamic shifts (`mu_weight`, `mu_offset`) |
| **Activation buffers** | float32 | 8 bit integer activations |
| **Training C library** | `ondevice_training_lib.h/.c` | `integer_odl.h/.c` |
| **Config file name** | `trainable_model_config.h/.c` | `integer_model_config.h/.c` |
| **Training data file** | `ondevice_training_data.h/.c` | `train_data.h/.c` |
| **Supported layers** | Linear, Conv2d, ReLU | Conv2D, Linear, ReLU |

For background on what on-device training is and why the frozen+trainable split architecture is used, see the float ODT [Overview](../ondevice_training/overview.md). Those concepts apply equally here.

---

## What You Need

| Requirement | Details |
|-------------|---------|
| **Hardware** | AM13E230X LaunchPad |
| **SDK** | AM13E230X SDK with TI ARM Clang toolchain |
| **IDE** | Code Composer Studio (CCS) |
| **ModelMaker** | tinyml-tensorlab ModelMaker environment set up and working |
| **Baseline working** | The standard (non-ODT) arc fault classification example should build and run before attempting integer ODT. This verifies your SDK, CCS, and hardware setup are correct. |

---

## Key Concepts

### Integer Arithmetic

Standard backpropagation in float computes gradients like:
```
weight_new = weight - lr * gradient
```

Integer ODT does the same computation in integer form using the **NITI** algorithm. The key insight is that gradient magnitudes vary widely during training, rather than using a fixed shift to fit gradients into INT8 range, NITI **measures the actual gradient magnitude each batch** and computes the minimum shift needed:

```
shift = max(0, ceil(log2(max_gradient)) - target_bits)
quantized_gradient = round(gradient >> shift)
weight_new = sat_int8(weight - quantized_gradient)
```

This adaptive scaling keeps gradients in INT8 range without overflow while preserving as much precision as possible. The effective learning rate is controlled by two parameters in `IntModelCtx_t`:
- `mu_weight`: learning rate for weight updates (default: 1)
- `mu_offset`: learning rate for bias/offset updates (default: 1)

Higher mu → smaller effective shift → larger learning rate. See [Integer ODL Library](integer_odl_lib.md) for the full math.

### No PREQUANT Layer in the C Library

The float-to-INT8 quantization step (the PREQUANT block in the ONNX graph) **does not appear in `INT_LAYERS[]`**. Instead it is compiled into the TVM frozen model and handled by the NPU. The C library receives INT8 data directly from TVM's output, with no float-to-int conversion needed at runtime.

For **full model ODT** (all compute layers trainable): TVM compiles only the PREQUANT block. Its output is INT8 and feeds directly into `INTODT_Forward`.

For **partial model ODT** (last k layers trainable): TVM compiles the PREQUANT block plus the frozen compute layers. Its output is UINT8 (the RELU output of the last frozen layer) and feeds directly into `INTODT_Forward`.

### Activation Buffer Types

Because the model computes in integer arithmetic, intermediate activations have different types depending on whether the layer has ReLU:

- **RELU layers** (`INT_LAYER_CONV2D_RELU`, `INT_LAYER_LINEAR_RELU`): outputs are in `[0, 255]`, stored as `uint8_t`
- **Non-RELU layers** (`INT_LAYER_CONV2D`, `INT_LAYER_LINEAR`): outputs are in `[-128, 127]`, stored as `int8_t`

The C library handles this automatically. You do not need to cast or convert between the two.


### Supported Layer Types

| Type | Training support |
|------|-----------------|
| Conv2D (no ReLU) | Yes | 
| Conv2D + ReLU | Yes | 
| Linear / MatMul (no ReLU) | Yes |
| Linear / MatMul + ReLU |  Yes |

---

## Quickstart

This walkthrough uses the **arc fault classification** example. By the end, you will have a model that was fine-tuned on-device using integer arithmetic.

### Step 1: Configure YAML

Add these fields to your arc fault classification YAML, or use the example YAML directly:

```yaml
training:
    quantization: 2                       # Required: QAT (integer weights)
    ondevice_training: true               # Enable integer ODT export pipeline
    trainable_layers_from_last: 3         # 3 compute layers 
    export_samples_per_class: '[10,5,5]'  # [train, val, test] samples per class
```

**Critical:** `quantization: 2` is mandatory. The integer ODT pipeline requires QAT, the weights must already be quantized to INT8 before export. Running with `quantization: 0` (float) will not generate integer ODT artifacts.

For full YAML details see [Running ModelMaker for Integer ODT](running_modelmaker_for_integer_odt.md).

### Step 2: Run ModelMaker

```bash
./run_tinyml_modelzoo.sh examples/dc_arc_fault/config_dsi_ondevice_learning.yaml
```

This runs QAT training, then additionally:
- Parses the QAT ONNX graph (identifies Conv, Linear, PREQUANT blocks)
- Analyzes output distribution → derives `TARGET_MAG_CORRECT/WRONG` per class
- Exports `integer_model_config.h/.c` with INT8 weights and layer descriptors
- Exports `train_data.h/.c` with float training samples and INT8 labels
- Saves `frozen_model/model.onnx` at the correct split point for TVM
- TVM compiles `frozen_model/model.onnx` → `mod.a` + `tvmgen_default.h`

### Step 3: Copy Artifacts to CCS Project

Import the `arc_fault_odl` example from the SDK into CCS:
1. **File → Import → CCS Projects**
2. Browse to `tinyml-sdk/am13/ai/examples/arc_fault_odl/`
3. Import

Then copy these files from the ModelMaker output into `artifacts/` in the imported project:

| File | ModelMaker output path |
|------|----------------------|
| `mod.a` | `<run_dir>/compilation/artifacts/mod.a` |
| `tvmgen_default.h` | `<run_dir>/compilation/artifacts/tvmgen_default.h` |
| `integer_model_config.h` | `<run_dir>/training/quantization/integer_model_config/integer_model_config.h` |
| `integer_model_config.c` | `<run_dir>/training/quantization/integer_model_config/integer_model_config.c` |
| `train_data.h` | `<run_dir>/training/quantization/integer_model_config/train_data.h` |
| `train_data.c` | `<run_dir>/training/quantization/integer_model_config/train_data.c` |
| `test_vector.c` | `<run_dir>/training/quantization/golden_vectors/test_vector.c` |
| `user_input_config.h` | `<run_dir>/training/quantization/golden_vectors/user_input_config.h` |

For exact file-by-file instructions see [Running ModelMaker: Copying Artifacts](running_modelmaker_for_integer_odt.md#6-copying-artifacts-to-the-ccs-project).

### Step 4: Build, Flash, and Run

1. Select the **Flash** build configuration in CCS
2. Build the project
3. Connect your AM13E230X LaunchPad and flash
4. Open the CCS console, you should see something like below output:

```
============================================================
  INPUT_SIZE=128  N_CLASSES=2  BATCH_SIZE=8  EPOCHS=30  N_SAMPLES=20
============================================================

Accuracy BEFORE ODL: 7 / 20  (35.0%)
```

The application evaluates accuracy with the freshly-deployed (barely trained) model, then runs the training loop, then re-evaluates:

```
Epoch [1/30]   Loss: 183204.0000
Epoch [2/30]   Loss: 112847.0000
...
Epoch [30/30]  Loss: 1243.5000

Accuracy AFTER  ODL: 20 / 20  (100.0%)
Done.
```

---

## Documentation Map

### 1. [Overview](overview.md)
What integer ODT is, why integer arithmetic, how NITI works, the frozen+trainable split for integer models, full vs partial model modes, memory layout, supported configurations, and current limitations. Start here if you want to understand the architecture before using it.

### 2. [Running ModelMaker for Integer ODT](running_modelmaker_for_integer_odt.md)
Step-by-step guide to generating artifacts. Covers every ODT-specific YAML field with explanation, a complete working YAML, the exact command to run, console log walkthrough, output directory structure, and file-by-file copy guide for the CCS project.

### 3. [Generated Artifacts Overview](generated_artifacts_overview.md)
A map of all generated files, what each contains, and how they connect at runtime. Read this if you want to understand what ModelMaker produces before diving into individual files.

### 4. [Integer Model Configuration](integer_model_config.md)
Deep dive into `integer_model_config.h` and `integer_model_config.c`. Covers every `#define`, the `IntLayerDesc_t` structure field by field, `IntModelCtx_t`, weight/k/offset arrays, activation buffer layout, ReLU mask memory, and grad_buf sizing.

### 5. [Training Data](training_data.md)
Deep dive into `train_data.h` and `train_data.c`. Covers data layout, label encoding, why inputs are float (and the partial model limitation), and how the training loop iterates through these arrays.

### 6. [Integer ODL Library: API Reference](integer_odl_lib.md)
Complete API reference for `integer_odl.h/.c`. Covers all three public functions (`INTODT_Infer`, `INTODT_Forward`, `INTODT_Backward`), the NITI shift calculation with worked examples, conv/linear forward and backward math, the user loss function pattern, and a complete custom training loop.

---

## Reading Paths

**New to on-device training entirely?**
→ Read the float ODT [Overview](../ondevice_training/overview.md) first for concepts, then come back here.

**Familiar with float ODT, switching to integer?**
→ [How This Differs from Float ODT](#how-this-differs-from-float-odt) → [Overview](overview.md) → [Running ModelMaker](running_modelmaker_for_integer_odt.md)

**Just want to get it working quickly?**
→ [Quickstart](#quickstart) → [Running ModelMaker](running_modelmaker_for_integer_odt.md)

**Understanding the generated files?**
→ [Generated Artifacts Overview](generated_artifacts_overview.md) → [Integer Model Configuration](integer_model_config.md) → [Training Data](training_data.md)

**Writing your own training loop / adapting main.c?**
→ [Integer ODL Library](integer_odl_lib.md) (Section: Writing a Complete Training Loop)
