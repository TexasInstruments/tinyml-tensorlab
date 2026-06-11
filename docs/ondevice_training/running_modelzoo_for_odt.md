# Running ModelZoo for On-Device Training

## Table of Contents

- [1. Overview](#1-overview)
- [2. ODT-Specific YAML Configuration](#2-odt-specific-yaml-configuration)
- [3. Running the Command](#3-running-the-command)
- [4. How It Works Under the Hood](#4-how-it-works-under-the-hood)
- [5. Output Directory Structure](#5-output-directory-structure)
- [6. Copying Artifacts to the CCS Project](#6-copying-artifacts-to-the-ccs-project)
- [7. Next Steps](#7-next-steps)

---

## 1. Overview

This guide explains how to use TI's ModelZoo pipeline to generate all the artifacts needed for on-device training on microcontrollers. By the end of this page, you will have a set of files ready to copy into a CCS project for on-device training.

**Prerequisites:**
- ModelZoo environment set up ([setup instructions](https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo))
- A YAML configuration file with on-device training enabled

For conceptual background on what on-device training is and why it uses a frozen+trainable split, see [Overview](overview.md).

---

## 2. ODT-Specific YAML Configuration

On-device training builds on the standard YAML configuration used to run the training pipeline in ModelZoo. Most fields (dataset, feature extraction, compilation) are identical to a regular run. Only a few fields are specific to on-device training.

### ODT-specific fields

| Field | Example Value | Description |
|-------|---------------|-------------|
| `ondevice_training` | `True` | **Enables the on-device training export pipeline.** When set, ModelMaker will split the ONNX model, custom export the trainable portion, and generate training data files. |
| `trainable_layers_from_last` | `1` | **Number of main layers (Linear etc) from the end of the model to keep trainable.** The ONNX graph is traversed backward from the output; `k` counts only layers with trainable parameters. All layers (including activations like ReLU) between the split point and the output become part of the trainable portion. See [Overview — Split Architecture](overview.md#3-the-frozen--trainable-split-architecture) for details. |
| `export_samples_per_class` | `[10, 10, 10]` | **Number of samples to embed for on-device use, as [train, validation, test].** These samples are exported as C arrays in `ondevice_training_data.h/.c`. Each sample contains the complete raw input data (all channels and all frames if `num_frame_concat` is configured). For example, with 3 channels and `num_frame_concat: 1` and frame size 256, each sample is 3 × 1 × 256 = 768 floats. See [On-Device Training Data](ondevice_training_data.md) for details on the exported format. |
| `quantization` | `0` | **Must be 0 for on-device training.** The trainable portion operates in float32 — quantization is not compatible with on-device weight updates. The frozen portion could theoretically be quantized separately, but the current pipeline exports it as float. |
| `model_name` | `Ondevice_Trainable_AD_Linear` | **Model architecture that supports the frozen+trainable split.** The model must produce a **linear (sequential) ONNX graph** — no branches, skip connections, or cycles. Only supported layer types are allowed in the trainable portion: Gemm (Linear), ReLU, Reshape, and Flatten. Graph-structured models with skip connections etc are not supported for on-device training. |

### Fields that are unchanged from standard anomaly detection

All other fields work exactly as documented in the standard anomaly detection pipeline:

- **`dataset`** — Dataset source, split type, split ratios
- **`data_processing_feature_extraction`** — Feature extraction preset, variables, frame concatenation
- **`training.batch_size`**, **`training.training_epochs`** — PC-side training configuration. Note: `training_epochs` can be set to `0` if you want to train entirely on-device.
- **`training.learning_rate`** — PC-side learning rate (the on-device learning rate is configured separately in the C application code)
- **`compilation`** — TVM compilation settings for the frozen model

### Complete YAML example

Below is the fan blade on-device training YAML with ODT-specific fields highlighted:

```yaml
common:
    target_module: 'timeseries'
    task_type: 'generic_timeseries_anomalydetection'
    target_device: 'F28P55'
    run_name: '{date-time}/{model_name}'

dataset:
    enable: True
    dataset_name: fan_blade_ondevice
    input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/fan_blade_ad_dataset.zip'
    split_type: 'amongst_files'
    split_factor: [0.6, 0.1, 0.3]

data_processing_feature_extraction:
    data_proc_transforms: []
    feature_extraction_name: Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1
    variables: 3
    num_frame_concat: 1
    stride_size: 1

training:
    enable: True
    model_name: 'Ondevice_Trainable_AD_Linear'     # ← ODT: model architecture
    model_config: ''
    batch_size: 64
    training_epochs: 200
    num_gpus: 0
    learning_rate: 0.001
    quantization: 0                                  # ← ODT: must be 0
    output_int: False
    ondevice_training: True                          # ← ODT: enables split/export
    trainable_layers_from_last: 1                    # ← ODT: split point
    export_samples_per_class: [10, 10, 10]           # ← ODT: [train, val, test]

testing:
    enable: True

compilation:
    enable: True
    compile_preset_name: 'forced_soft_npu_preset'
```

---

## 3. Running the Command

Execute ModelZoo with the on-device training YAML:

```bash
./run_tinyml_modelzoo.sh examples/fan_blade_fault_classification/fan_blade_anomaly_detection_ondevice_training.yaml
```

### What to expect in the logs

The pipeline runs through the standard anomaly detection flow with additional ODT-specific steps. Here are the key log sections from an actual run (paths abbreviated):

1. **Dataset loading and feature extraction**
   ```
   Loading training data
   100%|████████████████████████████| 60/60 [00:08<00:00,  7.42it/s]
   Loading validation data
   100%|████████████████████████████| 10/10 [00:01<00:00,  8.07it/s]
   ```

2. **Model creation and summary**
   ```
   Creating model
   Variables: 3, Input_features: 16
   ===============================================================================================
   AD_3_LAYER_DEEP_ONDEVICE_TRAINABLE_MODEL_TS   [1, 3, 16, 1]             --
   ├─Sequential: 1-1                             [1, 24]                   --
   │    └─Flatten: 3-1                           [1, 48]                   --
   │    └─Linear: 3-2                            [1, 24]                   1,176
   │    └─ReLU: 3-3                              [1, 24]                   --
   │    ...                                      ...                       ...
   ├─Linear: 1-2                                 [1, 48]                   1,200
   ===============================================================================================
   Total params: 3,150
   ```

3. **Model training** (PC-side)
   ```
   Start training
   Best Epoch: None
   MSE inf
   Exporting model after training.
   ```
   > **Note:** With `training_epochs: 0`, no actual training occurs — the model exports with randomly initialized weights. `Best Epoch: None` and `MSE inf` confirm this. For non-zero epochs, you will see per-epoch loss values and a valid best epoch.

4. **On-device training export** (ODT-specific)
   ```
   Trainable layers from end (k): 6
   Searching for split point (k=6 trainable layers from end)
   Split point found: /frozen_model/encoder/encoder.0/Flatten_output_0
   Creating frozen subgraph (input -> /frozen_model/encoder/encoder.0/Flatten_output_0)
   Extracting trainable layers (from .../Flatten_output_0 -> output)
     Layer 0: Linear (48->24)
     Layer 1: ReLU (24->24)
     Layer 2: Linear (24->12)
     ...
     Layer 10: Linear (24->48)
   Extracted total 11 layers including both main and intermediate layers
   Flattening weights for 11 layers
     Total parameters: 3150
     Offsets: [0, 1152, 1176, 1464, ..., 3102, 3150]
   Computing buffer offsets for 11 layers
     Frozen output size: 48
   Generated trainable model config header file at .../trainable_model_config.h
   Generated trainable model config source file at .../trainable_model_config.c
   ```

5. **Golden vector and threshold generation**
   ```
   Creating Golden data for reference at .../golden_vectors
   Creating test_vector.c at: .../golden_vectors/test_vector.c
   Creating user_input_config.h at: .../golden_vectors/user_input_config.h
   ```

6. **Testing and threshold analysis**
   ```
   Reconstruction Error Statistics:
   Normal training data - Mean: 5705.585449, Std: 241.798920
   Anomaly test data - Mean: 8546.039062, Std: 1576.968628
   Threshold for K = 1.5 : 6068.283691
   Anomaly detection rate (recall): 99.52%
   Accuracy: 99.49%
   ```

7. **TVM compilation** (of frozen model)
   ```
   TI Model Library Memory Usage (mod.a)
   ============================================================
   Code:                        42 bytes (    0.04 KB)
   RO Data:                      0 bytes (    0.00 KB)
   RW Data:                      0 bytes (    0.00 KB)
   Total:                       42 bytes (    0.04 KB)
   ```
   > **Note:** The frozen model is only 42 bytes because in this full-model-training configuration (`trainable_layers_from_last: 6`), the frozen part contains only a Flatten operation. For partial training configurations, the frozen model will be significantly larger.

---

## 4. How It Works Under the Hood

When `ondevice_training: True` is set, the training script adds an extra step after the standard ONNX model export which performs the following:

1. **Load** the trained ONNX model
2. **Find split point** — traverse the ONNX graph backward from the output, counting main layers. Stop at the k-th one. The input tensor to this layer becomes the split point.
3. **Extract frozen subgraph** — copy the graph from input to split point, save as `frozen_model/model.onnx` for TVM compilation
4. **Extract trainable layers** — traverse forward from split point to output, parsing each layer 
5. **Flatten weights** — all trainable weights and biases are concatenated into a single contiguous float array with offset tracking
6. **Compute buffer offsets** — calculate memory layout for intermediate activations and gradient buffers
7. **Generate C files** — write `trainable_model_config.h` (architecture, types, layer table) and `trainable_model_config.c` (weight data, buffer definitions)

The frozen model ONNX is then compiled by TVM into `mod.a` through the standard compilation pipeline.

---

## 5. Output Directory Structure

After a successful run, the ModelMaker output directory contains:

```
tinyml-modelmaker/data/projects/<project_name>/run/<date-time>/<model_name>/
├── training/base/
│   ├── model.onnx                          # Full trained ONNX model
│   ├── checkpoint.pth                      # PyTorch checkpoint
│   ├── frozen_model/
│   │   └── model.onnx                      # Frozen subgraph (for TVM)
│   ├── trainable_model/
│   │   ├── trainable_model_config.h        # ★ Trainable architecture + types
│   │   ├── trainable_model_config.c        # ★ Trainable weights + buffers
|   |   ├── ondevice_training_data.h        # ★ Dataset dimensions + declarations
|   |   └── ondevice_training_data.c        # ★ Embedded train/val/test data
│   └── quantization/
│       └── golden_vectors/
│           ├── test_vector.c               # ★ Golden test vectors
│           └── user_input_config.h         # ★ Feature extraction config
│
├── compilation/
│   └── artifacts/
│       ├── mod.a                           # ★ Compiled frozen model
│       └── tvmgen_default.h                # ★ Frozen model API header


(★ = files to copy to CCS project)
```
---

## 6. Copying Artifacts to the CCS Project

The following files need to be copied from the ModelMaker output to the CCS project's `artifacts/` directory:

| File | Source (ModelMaker) | Destination (CCS Project) |
|------|-------------------|--------------------------|
| `mod.a` | `compilation/artifacts/mod.a` | `<example>/f29h85x/artifacts/mod.a` |
| `tvmgen_default.h` | `compilation/artifacts/tvmgen_default.h` | `<example>/f29h85x/artifacts/tvmgen_default.h` |
| `trainable_model_config.h` | `training/base/trainable_model/trainable_model_config.h` | `<example>/f29h85x/artifacts/trainable_model_config.h` |
| `trainable_model_config.c` | `training/base/trainable_model/trainable_model_config.c` | `<example>/f29h85x/artifacts/trainable_model_config.c` |
| `ondevice_training_data.h` | `training/base/trainable_model/ondevice_training_data.h` | `<example>/f29h85x/artifacts/ondevice_training_data.h` |
| `ondevice_training_data.c` | `training/base/trainable_model/ondevice_training_data.c` | `<example>/f29h85x/artifacts/ondevice_training_data.c` |
| `user_input_config.h` | `training/base/golden_vectors/user_input_config.h` | `<example>/f29h85x/artifacts/user_input_config.h` |
| `test_vector.c` | `training/base/golden_vectors/test_vector.c` | `<example>/f29h85x/artifacts/test_vector.c` |

**Compared to a standard (non-ODT) anomaly detection project**, the extra files are:
- `trainable_model_config.h` — model architecture for trainable layers
- `trainable_model_config.c` — trainable weight data and buffers
- `ondevice_training_data.h` — dataset dimensions
- `ondevice_training_data.c` — embedded training/validation/test data

---

## 7. Next Steps

- **Understand what was generated** → [Generated Artifacts Overview](generated_artifacts_overview.md)
- **Deploy and run on device** → [Application Example — Fan Blade](application_example.md)
- **Understand the on-device training concepts** → [Overview](overview.md)