# On-Device Training Documentation

## Table of Contents

- [Introduction](#introduction)
- [What You Need](#what-you-need)
- [Key Concepts](#key-concepts)
- [Quickstart](#quickstart)
  - [Step 1: Configure YAML](#step-1-configure-yaml)
  - [Step 2: Run ModelZoo](#step-2-run-modelzoo)
  - [Step 3: Copy Artifacts to CCS Project](#step-3-copy-artifacts-to-ccs-project)
  - [Step 4: Build, Flash, and Run](#step-4-build-flash-and-run)
  - [Step 5: Trigger Training](#step-5-trigger-training)
  - [What to Expect](#what-to-expect)
- [Documentation Map](#documentation-map)
- [Reading Paths](#reading-paths)

---

## Introduction

On-device training enables deep learning models to **continue training directly on TI microcontrollers** after deployment. Instead of the traditional "train on PC → deploy frozen model" workflow, models can adapt to new environments, personalize to specific installations, and maintain accuracy as conditions change — all without sending data back to a PC.

The on-device training framework supports both **partial model training** (freeze early layers, train last k layers) and **full model training** (train all layers on-device). The core ODT libraries are **device-agnostic** and designed to work across TI MCU families. It currently supports the **anomaly detection** task with autoencoder models.

```
Traditional:   Train on PC → Compile → Deploy → Inference only (fixed model)

On-Device:     Train on PC → Split model → Deploy frozen + trainable parts → Continue training on MCU → Adapt to real environment → Inference
```

---

## What You Need

| Requirement | Details |
|------------|--------|
| **Hardware** | A supported TI MCU LaunchPad. Currently, examples are available for the **F29H85x** (C29x family). |
| **SDK** | The corresponding device SDK and Code generation tools (e.g., F29H85x SDK for C29x) |
| **IDE** | Code Composer Studio (CCS)|
| **ModelZoo** | tinyml-tensorlab ModelZoo environment set up ([setup instructions](https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo)) |
| **Baseline working** | The standard (non-ODT) anomaly detection example for your device should build and run successfully before attempting on-device training. This verifies your SDK, CCS, and hardware setup are correct. See the [fan blade example readme](../../examples/fan_blade_fault_classification/readme_anomaly_detection.md). |
---

## Key Concepts

This section summarizes the core architectural ideas. For full details, see [Overview](overview.md).

### Frozen + Trainable Split

The model is split into two parts before deployment:

```
Raw Input → [Feature Extraction] → [Frozen Model] → [Trainable Model] → Output
                                     ─────┬─────      ──────┬──────
                                          │                 │
                                  Compiled via TVM     Exported as C arrays
                                  Stored as mod.a      Weights updated on-device
                                  Cannot be changed    
```

- The **frozen part** (earlier layers) is compiled into an optimized static library (`mod.a`) via TVM. It extracts general-purpose features (e.g., frequency patterns from FFT) that transfer well across environments. It runs fast but cannot be updated on-device.

- The **trainable part** (last k layers) is exported as float C arrays with full forward and backward pass support. These layers perform task-specific computation that is most affected by environmental changes. They are retrained on-device to adapt to the deployment environment.

- The split point is controlled by `trainable_layers_from_last` in the YAML config. Setting it to `1` trains only the last main layer. Setting it equal to the total number of main layers trains the entire model on-device.

### PC-Side Pipeline → Device-Side Training

The workflow has two phases:

**PC side** (run once):
1. Train the model using ModelZoo (or export with 0 epochs for fully on-device training)
2. Split the ONNX model → compile frozen part (`mod.a`) + export trainable part (`trainable_model_config.h/.c`)
3. Export a small number of training/validation/test samples as C arrays (`ondevice_training_data.h/.c`)
4. Copy all artifacts to the CCS project

**Device side** (runs on MCU):
1. Initialize the model (frozen + trainable parts)
2. Train the trainable layers using the embedded data (or live sensor data)
3. Perform task-specific post-training calibration (e.g., threshold recalculation for anomaly detection)
4. Run inference with the updated model

### On-Device Training Loop

The device-side training is a standard neural network training loop adapted for MCU constraints:

- **Forward pass**: Input flows through frozen model → trainable model → output
- **Loss computation**: Loss is computed between model output and target
- **Backward pass**: Gradients propagate through trainable layers only; weights are updated via SGD
- **Validation**: After each epoch, validation loss is computed
- **Early stopping**: Training stops automatically when validation loss stops improving (EMA-smoothed, with configurable patience)
- **Best weight checkpointing**: The best-performing weights are saved during training and restored at the end

For batch_size=1 (the default), weight updates happen immediately during the backward pass — no gradient accumulation buffer needed, saving both memory and compute.

### Task-Specific Libraries

The core ODT library (`ondevice_training_lib`) is **task-agnostic** — it provides forward pass, backward pass, weight updates, and loss computation. Task-specific behavior is implemented in separate libraries built on top:

| Task Library | Status | Key Features |
|-------------|--------|------|
| `anomaly_detection_odt` | **Available** | State machine (train/threshold/inference modes), EMA-smoothed early stopping, percentile and Gaussian threshold calculation, majority voting inference |
| `classification_odt` | Planned | Cross-entropy loss, accuracy tracking |

### Supported Configurations

| Category | Currently Supported |
|----------|--------------------|
| **Layer types** | Linear (fully connected), ReLU |
| **Tasks** | Anomaly detection (autoencoder) |
| **Loss functions** | MSE (Mean Squared Error) |
| **Optimizers** | SGD |
| **Data format** | Float32  |
| **Training modes** | Partial model (last k layers) or full model (all layers) |
| **Example devices** | Works across all devices like C28, C29 etc. |

---

## Quickstart

This walkthrough uses the **fan blade anomaly detection on device learning** example. By the end, you will have a model that got trained on device. 

### Step 1: Configure YAML

Start with the fan blade ODT YAML or add these ODT-specific fields to your existing anomaly detection YAML:

```yaml
training:
    model_name: 'Ondevice_Trainable_AD_Linear'   # Model that supports split training
    quantization: 0                                # Must be 0 (float for trainable path)
    ondevice_training: True                        # Enable ODT export pipeline
    trainable_layers_from_last: 1                  # Train last 1 main layer on-device
    export_samples_per_class: [10, 10, 10]         # [train, val, test] samples to embed
```

The complete YAML is at:

    tinyml-modelzoo/examples/fan_blade_fault_classification/fan_blade_anomaly_detection_ondevice_training.yaml

For full details on each field, see [Running ModelZoo for ODT](running_modelzoo_for_odt.md).

### Step 2: Run ModelZoo

```bash
./run_tinyml_modelzoo.sh examples/fan_blade_fault_classification/fan_blade_anomaly_detection_ondevice_training.yaml
```

This runs the standard training pipeline, then additionally:
- Splits the ONNX model at the configured layer
- Compiles the frozen part via TVM → `mod.a`
- Exports the trainable part → `trainable_model_config.h/.c`
- Exports embedded training data → `ondevice_training_data.h/.c`
- Generates golden vectors and feature extraction config

### Step 3: Import CCS Project and Copy Artifacts

First, import the on-device training example project from your device SDK into CCS:

1. In CCS: **File → Import → CCS Projects**
2. Browse to the on-device training example in your device SDK (for F29H85x: the `fan_blade_anomalydetection_ondevice_learning` example)
3. Import the project

Then copy these generated files from the ModelMaker output into the imported project's `artifacts/` directory:

| File | What it is |
|------|------------|
| `mod.a` | Compiled frozen model |
| `tvmgen_default.h` | Frozen model API |
| `trainable_model_config.h/.c` | Trainable layer architecture + weights |
| `ondevice_training_data.h/.c` | Embedded train/val/test datasets |
| `user_input_config.h` | Feature extraction config + initial threshold |
| `test_vector.c` | Golden vectors for validation |

These files replace the default artifacts that ship with the example project. For exact source paths in the ModelMaker output, see [Running ModelZoo — Copying Artifacts](running_modelzoo_for_odt.md#6-copying-artifacts-to-the-ccs-project).

### Step 4: Build, Flash, and Run

1. Select the **Flash** build configuration 
2. Build the project
3. Connect your LaunchPad and flash the device
4. Open the CCS console — you should see initialization logs:

```
========================================
ANOMALY DETECTION MODEL INITIALIZATION
========================================
Learning rate(multiplied by 10^5): 10
Batches per epoch: 10, total samples per epoch 10
...

========================================
ODT INITIALIZATION
========================================
  Layer 0: Linear(48→24)
  Layer 1: ReLU(24)
  ...
  Layer 10: Linear(24→48)
```

The application then runs golden vector validation and pre-training inference automatically, then waits for you to trigger training.

### Step 5: Trigger Training

In CCS, open the **Variables** view, add the variable `start_training`, and change its value from `0` to `1`. Training begins immediately.

The console shows per-epoch training and validation loss. Training runs until early stopping triggers . After training, the threshold is automatically recalculated and post-training inference runs.

### What to Expect

| Stage | What Happens |
|-------|--------------|
| Golden vector validation | All 48 outputs match → artifacts loaded correctly |
| Pre-training inference | Untrained model, produces random results, all predictions are random |
| Training | Loss drops from ~5500 to ~10 over ~120 epochs, early stopping triggers |
| Threshold recalculation | New threshold: ~25 (down from ~6430) |
| Post-training inference | 12/12 correct with normal errors ~10 and anomaly errors ~130–1000 — genuine separation |

For the complete walkthrough with real console output, see [Application Example — Fan Blade](application_example.md).

---

## Documentation Map

Read in order for a complete understanding, or jump to the section relevant to you.

### 1. [Overview](overview.md)
The starting point for understanding on-device training. Covers **what** ODT is and **why** it is needed (data drift, privacy, personalization). Explains the frozen+trainable split architecture in detail, what `trainable_layers_from_last` controls, partial vs full model training, the complete end-to-end workflow, feature tables for core framework and anomaly detection task library, supported configurations, memory considerations, and current limitations.

### 2. [Running ModelZoo for ODT](running_modelzoo_for_odt.md)
Step-by-step guide to generating all artifacts using the ModelZoo pipeline. Covers the ODT-specific YAML fields with detailed explanations, a complete YAML example, the exact command to run, real console log output at each stage, how the pipeline works under the hood (ONNX splitting, weight flattening, buffer offset computation), the output directory structure, and a file-by-file mapping of what to copy to the CCS project.

### 3. [Generated Artifacts Overview](generated_artifacts_overview.md)
A bird's-eye view of all generated files and how they connect. Shows which files are shared with the standard (non-ODT) pipeline and which are new for ODT. Explains the role of each artifact and how data flows between them at runtime on the device.

### 4. [Trainable Model Configuration](trainable_model_config.md)
Deep dive into `trainable_model_config.h` and `.c`. Covers every `#define`, every data structure (`LayerType_t`, `LayerParams_t`, `LAYER_PARAMS_INIT`), the flat weight array with offset-based access, intermediate and gradient buffer allocation, memory section attributes for MCU RAM placement, batch size configuration, and the `USE_GRADIENT_ACCUMULATION` compile-time optimization.

### 5. [On-Device Training Data](ondevice_training_data.md)
Deep dive into `ondevice_training_data.h` and `.c`. Explains how raw sensor data is exported as 2D C arrays, the data layout (samples x raw input size), label encoding, sample counts per split, `RAW_INPUT_SIZE` computation from the feature extraction config, and how the application iterates through these arrays during training.

### 6. [On-Device Training Library](ondevice_training_lib.md)
Complete API reference for the core training engine (`ondevice_training_lib.h/.c`). Covers `ModelContext_t` and its flat-array memory architecture with detailed diagrams, initialization and validation, forward and backward pass, MSE loss and gradient computation, weight checkpointing, the batch_size=1 optimization, Linear and ReLU layer implementations, the logging system, a complete custom training loop example, and how to add new layer types.

### 7. [Anomaly Detection ODT Library](anomaly_detection_odt.md)
Complete API reference for the anomaly detection task library (`anomaly_detection_odt.h/.c`). Covers the three-mode state machine (inference, training, threshold calculation), all configuration parameters, training orchestration with automatic epoch and phase management, EMA-smoothed early stopping, percentile and Gaussian threshold calculation, majority voting inference, rolling statistics, all API functions with detailed behavior, and usage examples showing the complete workflow.

### 8. [Application Example — Fan Blade](application_example.md)
A complete working example walkthrough. Walks through all six stages: initialization, golden vector validation, pre-training inference, on-device training with real epoch-by-epoch console output, threshold recalculation, and post-training inference. Includes before-vs-after comparison, key code patterns, hyperparameter reference, and instructions for adapting the example to your own dataset.

### 9. [FAQ & Troubleshooting](faq.md)
Common issues and their solutions. Covers training problems (loss not decreasing, exploding loss, early stopping too early/late), deployment issues (golden vector mismatch, build failures, RAM overflow), threshold and inference problems, YAML configuration mistakes, and general questions about learning rate tuning, live sensor data, retraining, and persistence.

---

## Reading Paths

**New to on-device training?**
→ [Overview](overview.md) → [Running ModelZoo](running_modelzoo_for_odt.md) → [Application Example](application_example.md)

**Developer integrating ODT into a project?**
→ [Application Example](application_example.md) → [Anomaly Detection ODT Library](anomaly_detection_odt.md) → [Generated Artifacts Overview](generated_artifacts_overview.md)

**Understanding the generated files?**
→ [Generated Artifacts Overview](generated_artifacts_overview.md) → [Trainable Model Configuration](trainable_model_config.md) → [On-Device Training Data](ondevice_training_data.md)

**Writing a custom training loop?**
→ [On-Device Training Library](ondevice_training_lib.md) (Section 12: Writing a Custom Training Loop)

**Extending the framework (new layers, new tasks)?**
→ [On-Device Training Library](ondevice_training_lib.md) (Section 13: How to Add a New Layer Type) → [Overview](overview.md) (Section 7: Supported Task Libraries)

**Something not working?**
→ [FAQ & Troubleshooting](faq.md)
