# On-Device Training — Overview

## Table of Contents

- [1. What is On-Device Training?](#1-what-is-on-device-training)
- [2. Why On-Device Training?](#2-why-on-device-training)
- [3. The Frozen + Trainable Split Architecture](#3-the-frozen--trainable-split-architecture)
- [4. End-to-End Workflow](#4-end-to-end-workflow)
- [5. Features](#5-features)
- [6. Supported Configurations](#6-supported-configurations)
- [7. Supported Task Libraries](#7-supported-task-libraries)
- [8. Memory Considerations](#8-memory-considerations)
- [9. Limitations](#9-limitations)
- [10. FAQ & Troubleshooting](#10-faq--troubleshooting)
- [11. Further Reading](#11-further-reading)

---

## 1. What is On-Device Training?

On-device training (ODT) is the ability for a deep learning model to **continue training directly on a microcontroller** after it has been deployed. This goes beyond the traditional embedded ML workflow where models are trained on a PC, compiled, and deployed as inference-only artifacts.

**Traditional workflow:**
```
Train on PC → Compile model → Deploy to MCU → Inference only
```

**On-device training workflow:**
```
Train on PC → Split model → Compile frozen part → Export trainable part → Deploy to MCU → Continue training on MCU → Inference
```

With on-device training, the model shipped to the device is trained using the training dataset. The device itself retrains using data collected in its actual operating environment when the need arises. 

---

## 2. Why On-Device Training?

Deploying a frozen works well when the deployment environment closely matches the training environment. In practice, this assumption often breaks down:

- **Data drift**: A fan blade anomaly detector trained in a lab may encounter different vibration characteristics when installed in a factory floor with different mounting, temperature, or airflow conditions. The model needs to adapt to the new "normal." An arc fault classifier trained may perform poorly when deployed in real world if the deployed environment is different from the environment in which data is collected for training.

- **Privacy and data sensitivity**: In some applications, raw sensor data cannot leave the device due to regulatory or security constraints. On-device training allows the model to learn from local data without transmitting it.

- **Personalization**: Each installation may have unique characteristics. A motor vibration model trained on one motor model may need to adapt to a different motor. A keyword detection model may need to adapt to a specific speaker's accent. On-device training enables per-installation customization.

- **Reduced re-deployment cost**: Without on-device training, adapting to a new environment requires collecting data, shipping it to a PC, re-training, re-compiling, and re-flashing. On-device training eliminates this round-trip entirely.

- **Zero-shot deployment**: In an extreme case, the trainable portion can be deployed with **zero epochs** of PC-side training, and the device trains the model entirely from scratch using locally collected data.

---

## 3. The Frozen + Trainable Split Architecture

On-device training works by **splitting the model into two parts**: a frozen part that runs as optimized compiled code, and a trainable part that can be updated on-device.

### Partial model training (large models)

For larger models, only the last few layers are made trainable while the earlier layers are frozen. This reduces the memory and compute required for on-device training while preserving the learned feature representations from PC training.

```
┌─────────────────────────────────────────────────────────────────┐
│                          Full Model                             │
│                                                                 │
│   ┌───────────────────────┐    ┌──────────────────────────┐     │
│   │     Frozen Part       │    │     Trainable Part       │     │
│   │                       │───►│                          │     │
│   │  Compiled via TVM     │    │  Parameters and layers   │     │
│   │  Stored as mod.a      │    │  are custom exported     │     │
│   │  Runs as fixed        │    │  Trains on device        │     │
│   │  inference only       │    │  Adapts to new data      │     │
│   └───────────────────────┘    └──────────────────────────┘     │
│                                                                 │
│   ◄──── Earlier layers ─────►  ◄──── Last k layers ────────►    │
└─────────────────────────────────────────────────────────────────┘
```

- **The frozen part** consists of the earlier layers of the model. These layers learn general-purpose feature representations during PC training that tend to generalize well across environments. This part is compiled into an optimized static library (`mod.a`) using the [TI Neural Network Compiler (TVM-based)](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/index.html). 

- **The trainable part** consists of the last k layers of the model. These layers perform the task-specific computation that is most affected by environmental changes. This part is stored as **float C arrays** with forward and backward pass implementations, allowing it to be updated on-device.

### Full model training (small models)

For small models where memory and compute are not a concern, the **entire model** can be made trainable on-device. The device trains all layers from scratch or fine-tunes all of them.

```
┌─────────────────────────────────────────────────────────────────┐
│                          Full Model                             │
│                                                                 │
│   ┌──────────────┐    ┌─────────────────────────────────────┐   │
│   │ Frozen Part  │    │          Trainable Part             │   │
│   │              │───►│                                     │   │
│   │ (Flatten /   │    │  All functional layers              │   │
│   │  Reshape     │    │  Parameters stored in RAM           │   │
│   │  only)       │    │  Full model trains on device        │   │
│   └──────────────┘    └─────────────────────────────────────┘   │
│                                                                 │ 
│   ◄─ non-functional ─►  ◄──── All model layers ────────────►    │
└─────────────────────────────────────────────────────────────────┘
```

In this mode, the frozen part contains only non-functional layers (such as Flatten or Reshape) that exist in the ONNX graph before the first trainable layer. The `mod.a` compiled model effectively passes the input through unchanged, and all actual computation happens in the trainable portion.

> **Current limitation:** The split mechanism requires at least one non-functional layer (Flatten, Reshape) to exist before the first main layer in the ONNX graph. This layer becomes the split point, and the frozen `mod.a` contains just this pass-through operation. Model architectures intended for full on-device training should include such a layer.

### The `trainable_layers_from_last` parameter

The split point is controlled by the `trainable_layers_from_last` parameter in the YAML configuration. This parameter specifies how many **main layers** (layers with trainable parameters, such as Linear/Gemm) from the end of the model should be trainable. Activation layers (ReLU), reshape, and flatten operations between the split point and the output are automatically included in the trainable portion.

| Value of k | Behavior |
|------------|----------|
| Small (e.g., 1-2) | Only last few layers are trainable. Earlier layers frozen. Best for large models where early layers generalize well. |
| Equal to total main layers | Entire model is trainable on-device. Frozen part contains only non-functional layers. Best for small models. |

For example, with `trainable_layers_from_last: 1` on a model with 6 Linear layers, the split happens at the last Linear layer. However, if the model architecture places multiple Linear + ReLU pairs after the split point, all of those layers become part of the trainable portion.

---

## 4. End-to-End Workflow

```
         PC Side (ModelZoo/ModelMaker)                    Device Side (MCU)
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│                                      │     │                                      │
│  1. Train model on PC                │     │  6. Validate golden vectors          │
│     (or train for 0 epochs)          │     │     (verify artifacts loaded)        │
│                 ↓                    │     │                 ↓                    │
│  2. Export ONNX model                │     │  7. Pre-training evaluation          │
│                 ↓                    │     │     (baseline performance)           │
│  3. Split ONNX at layer k            │     │                 ↓                    │
│          ↓              ↓            │     │  8. On-device training               │
│  4a. Frozen part   4b. Trainable     │     │     (train/validate/early stop)      │
│      → TVM compile      → C export   │     │                 ↓                    │
│      → mod.a                         │     │  9. Task-specific post-training      │
│                 ↓                    │     │     (e.g., threshold calc)           │
│  5. Export training data (.h/.c)     │     │                 ↓                    │
│                                      │     │  10. Post-training inference         │
│                                      │     │      (improved performance)          │
│                                      │     │                                      │
└──────────────────────────────────────┘     └──────────────────────────────────────┘
```

### PC-side steps

1. **Train**: The model is trained on PC using the standard ModelZoo pipeline. The number of PC-side training epochs can range from full convergence to zero (if training will happen entirely on-device).

2. **Export**: The trained model is exported to ONNX format.

3. **Split**: The ONNX graph is traversed backward from the output to find the k-th trainable layer. The graph is split at this point into a frozen subgraph and a trainable layer sequence.

4. **Compile and export**:
   - The frozen subgraph is saved as a separate ONNX model and compiled via TVM into `mod.a` (a static library optimized for the target device).
   - The trainable layers are parsed, and their weights are flattened into contiguous C arrays. Architecture metadata is exported as `trainable_model_config.h` and `trainable_model_config.c`.

5. **Export training data**: A configurable number of samples from the training, validation, and test datasets are exported as C arrays in `ondevice_training_data.h` and `ondevice_training_data.c`.

### Device-side steps

6. **Golden vector validation**: A known test input is processed through feature extraction, the frozen model, and the trainable model. The output is compared against a golden reference to verify all artifacts are loaded correctly.

7. **Pre-training evaluation**: Test samples are run through the full pipeline to establish a baseline performance before on-device training.

8. **On-device training**: The training loop runs on-device using embedded training data. The core ODT library handles forward pass, backward pass, and weight updates. Task-specific libraries manage the training orchestration (epoch structure, validation, early stopping).

9. **Task-specific post-training steps**: After training completes, task-specific calibration may be needed. For example, the anomaly detection task library recalculates the detection threshold on validation data. Other task types may have different post-training requirements.

10. **Post-training inference**: Test samples are re-evaluated using the updated model, demonstrating improved performance.

---

## 5. Features

### Core ODT framework features

These features are provided by the `ondevice_training_lib` and are available to all task types:

| Feature | Description |
|---------|-------------|
| **SGD optimizer** | Stochastic Gradient Descent with configurable learning rate |
| **Batch size = 1 optimization** | When batch_size=1, weight updates happen immediately during backward pass — no gradient accumulation buffer needed, reducing memory usage |
| **Gradient accumulation** | For batch_size > 1, gradients are accumulated across samples and averaged before weight update. Controlled via compile-time `USE_GRADIENT_ACCUMULATION` flag |
| **Best weight checkpointing** | Model weights are saved/loaded via `ODT_SaveBestWeights()` / `ODT_LoadBestWeights()` for checkpoint-based training |
| **Configurable logging** | `DEBUG` builds enable printf logging; `USE_INTERRUPT_SAFE_LOGGING` wraps prints in DINT/EINT for ISR safety; release builds compile logging to zero overhead |
| **Memory section placement** | Weights, buffers, and gradients are placed in named memory sections via `#pragma DATA_SECTION` for linker-controlled RAM/Flash placement |
| **Partial or full model training** | Supports training only the last k layers (partial) or the entire model (full), controlled by `trainable_layers_from_last` |

### Anomaly detection task library features

These features are provided by the `anomaly_detection_odt` task library:

| Feature | Description |
|---------|-------------|
| **EMA-smoothed validation loss** | Exponential Moving Average smoothing on validation loss reduces noise from small validation sets, providing more stable early stopping decisions |
| **Early stopping** | Training automatically stops when EMA validation loss shows no improvement (exceeding `min_improvement_delta`) for `patience` consecutive epochs |
| **Percentile-based threshold** | Anomaly detection threshold computed as the k-th percentile (e.g., 99th) of reconstruction errors on normal validation data |
| **Gaussian threshold** | Alternative threshold method: mean + k × standard deviation of reconstruction errors |
| **Majority voting inference** | Sliding window of recent reconstruction errors with majority vote — anomaly declared only if >50% of window exceeds threshold, reducing false positives from transient spikes |
| **Rolling inference statistics** | Tracks anomaly/normal counts and percentages over a configurable window for runtime monitoring |

---

## 6. Supported Configurations

| Category | Currently Supported |
|----------|-------------------|
| **Layer types** | Linear (fully connected), ReLU |
| **Tasks** | Anomaly detection (autoencoder) |
| **Loss functions** | MSE (Mean Squared Error) |
| **Optimizers** | SGD |
| **Devices** | F29H85x (C29x family) |
| **Training modes** | Partial model (last k layers), Full model (all layers) |
| **Data format** | Float32 |

---

## 7. Supported Task Libraries

The ODT framework is task-agnostic — the core `ondevice_training_lib` provides neural network primitives (forward, backward, weight update, loss computation) that can be used by any task. Task-specific behavior is implemented in dedicated task libraries built on top of the core.

| Task Library | Status | Description |
|-------------|--------|-------------|
| **`anomaly_detection_odt`** | Available | Complete anomaly detection workflow for autoencoder models. Provides a state machine with three modes (training, threshold calculation, inference), EMA-smoothed early stopping, percentile/Gaussian threshold calculation, and majority voting inference. See [Anomaly Detection ODT Library](anomaly_detection_odt.md). |
| **`classification_odt`** | Planned | On-device training for classification models with cross-entropy loss. |

---

## 8. Memory Considerations

On-device training requires more memory than inference alone. The following buffers are allocated:

| Buffer | Size | Purpose |
|--------|------|---------|
| `ALL_WEIGHTS` | `TOTAL_PARAMS × 4` bytes | Active model weights |
| `ALL_BEST_WEIGHTS` | `TOTAL_PARAMS × 4` bytes | Checkpoint of best weights |
| `INTERMEDIATE_BUFFERS` | `TOTAL_INTERMEDIATE_BUFFER_SIZE × 4` bytes | Forward pass activations (needed for backward pass) |
| `GRADIENT_BUFFERS` | `TOTAL_GRADIENT_BUFFER_SIZE × 4` bytes | Backward pass gradient storage |
| `ALL_WEIGHT_GRADS` | `TOTAL_PARAMS × 4` bytes | Gradient accumulators (only when batch_size > 1) |


**Example** (fan blade anomaly detection model with 3,150 trainable parameters):
- Weights + best weights + batch gradeints(batch size > 1): ~40 KB
- Intermediate + gradient buffers: ~2 KB
- Threshold buffer: ~4 KB
- **Total ODT overhead: ~46KB** (on top of frozen model)
---

## 9. Limitations

- **Float-only trainable path**: The trainable portion operates in float32. Quantization cannot be applied. Set `quantization: 0` in the YAML configuration.

- **Pure SGD**: Only basic SGD is supported. There is no momentum, Adam, or other adaptive optimizers. 

- **Training data embedded at compile time**: In the current examples, training data is exported in C arrays. For real deployments, data would come from live sensors — the library APIs support this, but the example uses embedded data for reproducibility.

- **Full model training requires a non-functional split layer**: When training the entire model on-device, the ONNX graph must contain a non-functional layer (Flatten, Reshape) before the first main layer. This layer serves as the split point — the frozen `mod.a` contains only this pass-through operation. Model architectures intended for full on-device training should include such a layer.

- **No dynamic memory allocation**: All buffers are statically allocated. This is by design for MCU reliability but means buffer sizes must be known at compile time.

---

## 10. FAQ & Troubleshooting
For common issues and their solutions, see the [FAQ & Troubleshooting](faq.md) page. It covers:

- **Training issues** — loss not decreasing, loss exploding, early stopping too early/late, validation loss oscillating
- **Deployment & build issues** — golden vector mismatch, undefined symbols, RAM overflow
- **Threshold & inference issues** — all samples classified as anomaly/normal, false positives
- **Configuration issues** — YAML mistakes, `trainable_layers_from_last` too high
- **General questions** — learning rate tuning, live sensor data, retraining, persistence, sample count guidance

---

## 11. Further Reading
| Document | Description |
|----------|-------------|
| [Running ModelZoo for On-Device Training](running_modelzoo_for_odt.md) | YAML configuration, running the pipeline, output paths |
| [Generated Artifacts Overview](generated_artifacts_overview.md) | All generated files at a glance |
| [Trainable Model Configuration](trainable_model_config.md) | Deep dive into trainable_model_config.h and .c |
| [On-Device Training Data](ondevice_training_data.md) | Deep dive into embedded training datasets |
| [On-Device Training Library](ondevice_training_lib.md) | Low-level NN training primitives (forward, backward, SGD) |
| [Anomaly Detection ODT Library](anomaly_detection_odt.md) | Task library: state machine, training loop, threshold, inference |
| [Application Example — Fan Blade](application_example.md) | Complete device-side walkthrough with console output |
| [FAQ & Troubleshooting](faq.md) | Common issues, solutions, and general questions |