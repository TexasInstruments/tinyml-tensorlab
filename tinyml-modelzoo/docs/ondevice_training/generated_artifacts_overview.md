# Generated Artifacts Overview

- [1. Overview](#1-overview)
- [2. Artifact Summary](#2-artifact-summary)
- [3. Standard Artifacts](#3-standard-artifacts)
-  - [3.1 mod.a — Compiled Frozen Model](#31-moda--compiled-frozen-model)
-  - [3.2 tvmgen_default.h — Frozen Model API](#32-tvmgen_defaulth--frozen-model-api)
-  - [3.3 user_input_config.h — Feature Extraction & Threshold](#33-user_input_configh--feature-extraction--threshold)
-  - [3.4 test_vector.c — Golden Vectors](#34-test_vectorc--golden-vectors)
- [4. ODT-Specific Artifacts](#4-odt-specific-artifacts)
-  - [4.1 trainable_model_config.h  trainable_model_config.c](#41-trainable_model_configh--trainable_model_configc)
-  - [4.2 ondevice_training_data.h  ondevice_training_data.c](#42-ondevice_training_datah--ondevice_training_datac)
- [5. How Artifacts Connect at Runtime](#5-how-artifacts-connect-at-runtime)
- [6. Next Steps](#6-next-steps)

## 1. Overview

When the ModelZoo pipeline runs with `ondevice_training: True`, it generates a set of files that together define everything needed to run on-device training on a microcontroller. This page provides a map of all generated files — what each one contains and where it fits in the system.

For details on how to generate these files, see [Running ModelZoo for On-Device Training](running_modelzoo_for_odt.md).

---

## 2. Artifact Summary

The table below lists all artifacts. Files marked **ODT-specific** are unique to on-device training and do not exist in a standard inference-only deployment.

| File | ODT-Specific? | Purpose | Deep Dive |
|------|:------------:|---------|-----------|
| `mod.a` | No | Compiled frozen model (TVM output) | [Section 3.1](#31-moda--compiled-frozen-model) |
| `tvmgen_default.h` | No | API header for the frozen model | [Section 3.2](#32-tvmgen_defaulth--frozen-model-api) |
| `user_input_config.h` | No | Feature extraction configuration + initial threshold | [Section 3.3](#33-user_input_configh--feature-extraction--threshold) |
| `test_vector.c` | No | Golden vectors for device validation | [Section 3.4](#34-test_vectorc--golden-vectors) |
| `trainable_model_config.h` | **Yes** | Trainable model architecture, types, layer table, buffer offsets | [Trainable Model Configuration](trainable_model_config.md) |
| `trainable_model_config.c` | **Yes** | Trainable weight data, buffer definitions, memory sections | [Trainable Model Configuration](trainable_model_config.md) |
| `ondevice_training_data.h` | **Yes** | Dataset dimensions and extern declarations | [On-Device Training Data](ondevice_training_data.md) |
| `ondevice_training_data.c` | **Yes** | Embedded training, validation, and test data arrays | [On-Device Training Data](ondevice_training_data.md) |

---

## 3. Standard Artifacts

These files are also generated for non-ODT examples. In an on device training examples, their role is the same — they handle the frozen model and feature extraction.

### 3.1 `mod.a` — Compiled Frozen Model

The frozen portion of the model, compiled by the [TI Neural Network Compiler (TVM-based)](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/index.html) into a static library. This file contains optimized machine code for the target device and runs as a fixed-weight inference engine.

- **In partial model training:** Contains the earlier layers of the model (the feature extractor). Produces an intermediate representation that feeds into the trainable layers.
- **In full model training:** Contains only non-functional layers (Flatten/Reshape) and effectively passes the input through unchanged.

**Why it's needed:** The raw sensor data cannot be fed directly into the trainable layers. It must first pass through feature extraction (configured by `user_input_config.h`) and then through the frozen model to produce the intermediate representation that the trainable layers expect. Even during on-device training, every sample — whether from embedded training data or a live sensor — goes through this frozen model first. The frozen model's weights never change; only the trainable layers are updated.

**How it's used:** The application calls `tvmgen_default_run()` to execute this model. The function takes feature-extracted input, runs it through the compiled layers, and writes the output (of size `FROZEN_OUTPUT_SIZE`) to a buffer. This buffer then becomes the input to the trainable model's forward pass.

### 3.2 `tvmgen_default.h` — Frozen Model API

Header file exposing the API to interact with `mod.a`:

```c
// Input/output structures
struct tvmgen_default_inputs { void* input; };
struct tvmgen_default_outputs { void* output; };

// Run the frozen model
int tvmgen_default_run(struct tvmgen_default_inputs* inputs, struct tvmgen_default_outputs* outputs);
```

**How it's used:** The application creates input/output structs pointing to pre-allocated buffers, then calls `tvmgen_default_run()`. The input buffer contains the feature-extracted sensor data (shape and size determined by the feature extraction configuration). The output buffer receives the frozen model's intermediate representation of size `FROZEN_OUTPUT_SIZE`, which becomes the input to the trainable layers.

```c
// Example usage in application_main.c
struct tvmgen_default_inputs inputs = {(void *)&model_input[0]};
struct tvmgen_default_outputs outputs = {(void *)&frozen_output[0]};
tvmgen_default_run(&inputs, &outputs);
// frozen_output[] now contains FROZEN_OUTPUT_SIZE floats ready for the trainable model
```

### 3.3 `user_input_config.h` — Feature Extraction & Threshold

Defines the feature extraction pipeline configuration and the initial anomaly detection threshold. This file is auto-generated based on the `feature_extraction_name` preset selected in the YAML configuration.

```c
// Feature extraction flags — control which processing steps are applied
#define FE_FFT                        // Apply Fast Fourier Transform
#define FE_DC_REM                     // Remove DC component from FFT output
#define FE_BIN                        // Bin FFT magnitudes into fewer frequency bands
#define FE_LOG                        // Apply logarithmic scaling
#define FE_CONCAT                     // Concatenate features from multiple frames

// Feature extraction parameters
#define FE_FRAME_SIZE 256              // Raw samples per frame
#define FE_FEATURE_SIZE_PER_FRAME 16   // Features produced per frame after binning
#define FE_STACKING_CHANNELS 3         // Number of sensor channels (e.g., X, Y, Z axes)
#define FE_NUM_FRAME_CONCAT 1          // Number of frames concatenated for temporal context
#define FE_NN_OUT_SIZE 48              // Total model output size (channels × features)
// ... other FE parameters ...
```

**Why it's needed:** Raw sensor data (e.g., vibration readings) is not directly suitable as model input. The feature extraction pipeline transforms raw time-domain samples into compact frequency-domain features that the model was trained to process. The flags (like `FE_FFT`, `FE_BIN`, `FE_LOG`) control which signal processing steps are applied, and the parameters (like `FE_FRAME_SIZE`, `FE_FEATURE_SIZE_PER_FRAME`) control the dimensions at each stage. These must match exactly what was used during PC-side training — the model expects a specific input format.

**How it's used:** The feature extraction library reads these defines at compile time to configure its processing pipeline. When `FE_runFeatureExtract()` is called, it applies the enabled processing steps in sequence: raw input → FFT → DC removal → binning → log scaling → frame concatenation → model input. The output is a feature vector of size `FE_NN_OUT_SIZE` that feeds into the frozen model.

For a detailed explanation of the feature extraction library, its API, and the full processing pipeline, see the [Feature Extraction Library Reference](../../../tinyml-sdk/c29/ai/docs/feature_extract.md).

### 3.4 `test_vector.c` — Golden Vectors

Contains a known sample with pre-computed expected outputs at each stage of the pipeline. This is the first thing the application runs after initialization — a sanity check that everything is working.

```c
// Raw sensor data — input to feature extraction
float raw_input_test[RAW_SIZE] = { ... };

// Expected features after feature extraction — input to frozen model
float model_test_input[FE_NN_OUT_SIZE] = { ... };

// Expected final model output after frozen + trainable model
model_output_t golden_output[FE_NN_OUT_SIZE] = { ... };
```

**Why it's needed:** When deploying to a new device or updating artifacts, many things can go wrong silently — a mismatched `mod.a`, incorrect feature extraction parameters, corrupted weight data, or endianness issues. Golden vector validation catches these problems immediately by comparing the actual on-device output against a known-good reference computed on the PC.

**How it's used:** The application processes `raw_input_test` through the full pipeline:
1. Feature extraction → produces features, which can be compared against `model_test_input` to verify FE is correct
2. Frozen model inference → produces intermediate representation
3. Trainable model forward pass → produces final output
4. Compare final output against `golden_output[]` element-by-element (tolerance: 1e-3)

If all elements match within tolerance, the console prints `Golden vectors matched: 48 not matched: 0`. Any mismatch indicates a configuration or loading problem that must be resolved before proceeding to training or inference.

**Important:** Golden vectors validate the initial state of the model (with PC-trained weights). After on-device training modifies the weights, the golden vectors will no longer match — this is expected.

---

## 4. ODT-Specific Artifacts

These files are unique to on-device training and do not exist in a standard inference-only project.

### 4.1 `trainable_model_config.h` + `trainable_model_config.c`

Together these define the complete trainable portion of the neural network:

- **Header (`.h`)**: Architecture defines (`NUM_TRAINABLE_LAYERS`, `FROZEN_OUTPUT_SIZE`, `FINAL_OUTPUT_SIZE`, `TOTAL_PARAMS`), layer type enumerations, the `LayerParams_t` structure, the `LAYER_PARAMS_INIT` configuration table describing each layer's type/size/weight-offsets, buffer offset arrays, and batch size / gradient accumulation configuration.

- **Source (`.c`)**: The `ALL_WEIGHTS[]` array containing all trainable parameters as a single contiguous float array with initial values from PC training, `ALL_BEST_WEIGHTS[]` for checkpointing, `INTERMEDIATE_BUFFERS[]` and `GRADIENT_BUFFERS[]` for forward/backward pass storage, and conditional `ALL_WEIGHT_GRADS[]` for batch gradient accumulation. All arrays are placed in named memory sections via `#pragma DATA_SECTION`.

For a detailed walkthrough of every field and structure, see [Trainable Model Configuration](trainable_model_config.md).

### 4.2 `ondevice_training_data.h` + `ondevice_training_data.c`

Together these provide the embedded dataset for on-device training:

- **Header (`.h`)**: Dimension defines (`NUM_TRAIN_SAMPLES`, `NUM_VALIDATION_SAMPLES`, `NUM_TEST_SAMPLES`, `RAW_INPUT_SIZE`, `NUM_CLASSES`) and extern declarations for the data arrays.

- **Source (`.c`)**: The actual data arrays — `TRAIN_INPUTS[][]`, `TRAIN_LABELS[]`, `VALIDATION_INPUTS[][]`, `VALIDATION_LABELS[]`, `TEST_INPUTS[][]`, `TEST_LABELS[]`. Each sample contains raw sensor data (all channels and frames concatenated) as floats.

For details on the data format and how `export_samples_per_class` maps to these arrays, see [On-Device Training Data](ondevice_training_data.md).

---

## 5. How Artifacts Connect at Runtime

```
┌────────────────────┐     ┌───────────────┐     ┌──────────────────────────┐     ┌──────────────────────────┐
│   Input Data       │     │    Feature    │     │     Frozen model         │     │   Trainable model        |                                       
│                    │────►│   Extraction  │────►│     (mod.a)              │────►│   (trainable_model_      │────► Output   
│• Live sensor       │     │               │     │                          │     │    config.h/.c)          │
│  data              │     │  (user_input_ │     └──────────────────────────┘     └──────────────────────────┘  
│     OR             │     │   config.h)   │                  
│• Embedded data     │     └───────────────┘                  
│ (ondevice_         │                         
│  raining_data.c/.h)│                          
└────────────────────┘                          
```

**Input data → Feature extraction → Frozen model → Trainable model → Output**

---

## 6. Next Steps

- **Deep dive into trainable model files** → [Trainable Model Configuration](trainable_model_config.md)
- **Deep dive into training data files** → [On-Device Training Data](ondevice_training_data.md)
- **See these artifacts in action** → [Application Example — Fan Blade](application_example.md)