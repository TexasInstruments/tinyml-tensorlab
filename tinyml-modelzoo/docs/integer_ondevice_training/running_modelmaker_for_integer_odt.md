# Running ModelMaker for Integer On-Device Training

## Table of Contents

- [1. Overview](#1-overview)
- [2. YAML Configuration](#2-yaml-configuration)
- [3. Running the Command](#3-running-the-command)
- [4. How It Works Under the Hood](#4-how-it-works-under-the-hood)
- [5. Output Directory Structure](#5-output-directory-structure)
- [6. Copying Artifacts to the CCS Project](#6-copying-artifacts-to-the-ccs-project)
- [7. Next Steps](#7-next-steps)

---

## 1. Overview

This page explains how to use the ModelMaker pipeline to generate all artifacts needed for integer on-device training. By the end, you will have a set of files ready to drop into a CCS project for on-device integer training.

**Prerequisites:**
- ModelMaker environment set up and working
- A working arc fault (or other classification) YAML for standard inference
- Basic familiarity with what integer ODT is (see [Overview](overview.md))

---

## 2. YAML Configuration

Integer ODT builds on the standard classification YAML. Most fields are unchanged. Only a handful of fields are specific to integer ODT.

### ODT-Specific Fields

| Field | Example | Description |
|-------|---------|-------------|
| `quantization` | `2` | **Required: must be 2 (QAT).** Float ODT uses 0; integer ODT needs QAT so weights are INT8 at export time. Setting to 0 or 1 will not generate integer ODT artifacts. |
| `ondevice_training` | `true` | **Enables the integer ODT export pipeline.** When set, ModelMaker runs the ONNX parser, derives TARGET_MAG values, and writes all integer_model_config and train_data files. |
| `trainable_layers_from_last` | `3` | **How many compute layers from the end are trainable.** Compute layers are Conv and Linear/MatMul, ReLU activations are not counted. Set equal to total compute layer count for full model ODT; set less for partial. |
| `export_samples_per_class` | `[10,5,5]` | **Samples to embed as C arrays, as [train, val, test] per class.** For a 2-class model with `[10,5,5]`, this embeds 20 training samples, 10 val, 10 test total. These become the TRAIN_INPUTS/TRAIN_LABELS arrays in train_data.c |

### Fields That Are Unchanged

All standard classification fields work exactly as documented

### Complete YAML Example

```yaml
common:
  task_type: arc_fault
  target_device: AM13E2
dataset:
  dataset_name: arc_fault_example_dsi_odl
  input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/arc_fault_classification_dsi.zip
data_processing_feature_extraction:
  data_proc_transforms:
  - Downsample
  - SimpleWindow
  gain_variations:
    arc:
    - 0.9
    - 1.1
    normal:
    - 0.8
    - 1.2
  sampling_rate: 313000
  new_sr: 3130
  frame_size: 128
  stride_size: 0.01
  variables: 1
  gof_test: false
training:
  model_name: CLS_24k_NPU
  model_config: ''
  batch_size: 2048
  training_epochs: 1
  auto_quantization: false
  num_gpus: 1
  quantization: 2                           # ← INT ODT: QAT required
  ondevice_training: true                   # ← INT ODT: enables integer export
  trainable_layers_from_last: 3             # ← INT ODT: 3  layers
  export_samples_per_class: '[10,5,5]'      # ← INT ODT: [train, val, test] per class
testing: {}
compilation: {}

```

### Choosing trainable_layers_from_last

Count the compute layers in your model (Conv + Linear nodes, not ReLU or Reshape):

### Choosing export_samples_per_class

This controls how many samples are embedded in `train_data.c`. More samples → larger flash usage but potentially better on-device training:

The training samples are used for weight updates. The val samples determine early stopping . The test samples are available for evaluation but are not used during training.

---

## 3. Running the Command

```bash
./run_tinyml_modelzoo.sh examples/dc_arc_fault/config_dsi_ondevice_learning.yaml
```

### Expected Console Log Walkthrough

Here is what the key stages look like in the console output:

**1. Standard training stages** (same as non-ODT run):
```
Loading training data
100%|████████████████████| 1200/1200 [00:12<00:00]
Loading validation data
...
Creating model
ArcFaultNet  [1, 1, 128, 1]
├─ Conv2d:   [1, 3, 113, 1]
├─ Linear:   [1, 160]
└─ Linear:   [1, 2]
Total params: 24,018

Start training
Epoch 001/100:  loss=1.423  val_loss=1.387
Epoch 002/100:  loss=1.198  val_loss=1.115
...
Training complete. Best epoch: 67
Exporting model after training.
```

**2. QAT export** (specific to quantization=2):
```
Running QAT post-processing
Exporting quantized ONNX: output_dir/model.onnx
```

**3. Integer ODT export** (specific to ondevice_training=true):
```
[integer_odt] parsing ONNX: output_dir/model.onnx
[integer_odt] Segments found:
  seg 0: [Add, Mul, Mul, Floor, Clip]               → PREQUANT
  seg 1: [Conv, Add, Mul, Mul, Floor, Clip, Relu, Clip] → CONV2DRELU
  seg 2: [Reshape]                                  → RESHAPE (skipped)
  seg 3: [MatMul, Add, Mul, Mul, Floor, Clip, Relu, Clip] → LINEARRELU
  seg 4: [MatMul, Add, Mul, Mul, Floor, Clip]       → LINEAR

[integer_odt] trainable_layers_from_last=3, total compute layers=3
[integer_odt] full model: emitting 3 compute layers. PREQUANT compiled into TVM frozen model.

[integer_odt] Output distribution analysis (ODL train samples)
class 0 (normal):
  correct-class logit: mean=24.3  std=8.1  min=8  max=47  TARGET_CORRECT=24
  wrong-class logit[c]: mean=-31.2  std=9.4  min=-55  max=-12  TARGET_WRONG=-31
class 1 (arc_fault):
  correct-class logit: mean=19.7  std=7.3  min=5  max=41  TARGET_CORRECT=20
  wrong-class logit[c]: mean=-27.8  std=11.2  min=-58  max=-7  TARGET_WRONG=-28
TARGET_CORRECT: [24, 20]  TARGET_WRONG: [-31, -28]

[integer_odt] integer_model_config.h written
[integer_odt] integer_model_config.c written
[integer_odt] train_data.h written (20 train, 10 val, 10 test samples)
[integer_odt] train_data.c written

[integer_odt] frozen model split: PREQUANT only (full model ODT)
[integer_odt] frozen model saved: output_dir/frozen_model/model.onnx
```

**4. TVM compilation** (frozen_model/model.onnx → mod.a):
```
Compiling frozen model...
TVM compilation complete
mod.a written to output_dir/compilation/artifacts/mod.a
tvmgen_default.h written to output_dir/compilation/artifacts/tvmgen_default.h
```

**5. Test vector generation:**
```
Generating golden vectors...
test_vector.c written
user_input_config.h written
```
---

## 4. How It Works Under the Hood

Understanding the internals helps when troubleshooting unexpected errors.

### ONNX Graph Parsing

The ONNX graph is processed in three steps:

**Step 1: Segmentation.** The graph nodes are split into segments. Everything before the first compute node (Conv/MatMul/Gemm) becomes the PREQUANT segment. Each compute node plus its following non-compute nodes (Add, Mul, Floor, Clip, ReLU) becomes one compute segment. Reshape nodes become their own single-node segment.

**Step 2: Classification.** Each segment's op sequence is matched against a table of known patterns. The supported patterns are:

```
PREQUANT:   [Add, Mul, Mul, Floor, Clip]
CONV2DRELU: [Conv, Add, Mul, Mul, Floor, Clip, Relu, Clip]
CONV2D:     [Conv, Add, Mul, Mul, Floor, Clip]
LINEARRELU: [MatMul, Add, Mul, Mul, Floor, Clip, Relu, Clip]
LINEAR:     [MatMul, Add, Mul, Mul, Floor, Clip]
GEMMRELU:   [Gemm, Mul, Mul, Floor, Clip, Relu, Clip]
GEMM:       [Gemm, Mul, Mul, Floor, Clip]
RESHAPE:    [Reshape]
```

If a segment does not match any known pattern, a `RuntimeError` is raised with the actual op sequence. This most commonly happens when a quantization toolkit produces a slightly different graph structure.

**Step 3: Extraction.** For each recognized block, values are extracted:
- Conv block: weights `[n_filters, in_ch, kH, kW]`, offset `[n_filters]`, `k[n_filters]` derived from `round(-log2(|shift_mult|))`
- Linear block: weights `[in_size, out_size]`, offset `[out_size]`, `k[out_size]`
- PREQUANT block: per-channel offset and scale (combined product of both Mul constants)

### OSS Chain: What the k Array Is

The OSS (Output Shift and Scale) chain is the sequence `Add → Mul → Mul → Floor → Clip` that appears after every compute node in the QAT ONNX graph. It implements the quantized output computation:

```
output = clip(floor((accumulator + offset) * mult * shift), clip_lo, clip_hi)
```

Where `mult` is typically 1.0 (no-op multiply) and `shift` is a power of 2. The C library converts this to an integer right-shift:

```
k = round(-log2(|shift|))
```

So `shift = 0.03125 = 2^-5` becomes `k = 5`, meaning the C code does `(acc + offset) >> 5`.

Every filter in a Conv layer or every output in a Linear layer has its own k value, because quantization shifts are per-channel.

### TARGET_MAG Derivation

ModelMaker runs the complete QAT ONNX on all training samples using onnxruntime. For each class c:
- Collect `logit[c]` for all samples where `label == c` → compute mean → `TARGET_MAG_CORRECT[c]`
- Collect `logit[c]` for all samples where `label != c` → compute mean → `TARGET_MAG_WRONG[c]`

Both values are rounded to integers and stored as `int8_t` in `integer_model_config.h`. These become the targets for the on-device loss function. When the model correctly classifies a sample, the logit for the correct class should ideally equal `TARGET_MAG_CORRECT`. The loss pushes misclassified outputs toward these distribution means.


## 5. Output Directory Structure

After a successful run, the output directory looks like:

```
<run_dir>/
├── training/
│   └── quantization/
│       ├── integer_model_config/
│       │   ├── integer_model_config.h    ← Copy to CCS artifacts/
│       │   ├── integer_model_config.c    ← Copy to CCS artifacts/
│       │   ├── train_data.h              ← Copy to CCS artifacts/
│       │   └── train_data.c              ← Copy to CCS artifacts/
│       ├── frozen_model/
│       │   └── model.onnx                ← Input to TVM (already compiled)
│       └── golden_vectors/
│           ├── test_vector.c             ← Copy to CCS artifacts/
│           └── user_input_config.h       ← Copy to CCS artifacts/
├── compilation/
│   └── artifacts/
│       ├── mod.a                         ← Copy to CCS artifacts/
│       └── tvmgen_default.h              ← Copy to CCS artifacts/
└── model.onnx                            ← Full QAT model (for reference)
```

---

## 6. Copying Artifacts to the CCS Project

### Step 1: Import the SDK Example

1. Open CCS
2. **File → Import → Code Composer Studio → CCS Projects**
3. Browse to: `tinyml-sdk/am13/ai/examples/arc_fault_odl/`
4. Select the project and click **Finish**

The project will import with placeholder artifacts. You will replace these with your ModelMaker-generated files.

### Step 2: Copy Files

Copy the following files to the `artifacts/` folder inside the imported project. If an existing file has the same name, replace it.

| Copy this file | From ModelMaker output path | To CCS project |
|---------------|----------------------------|----------------|
| `mod.a` | `<run_dir>/compilation/artifacts/mod.a` | `artifacts/mod.a` |
| `tvmgen_default.h` | `<run_dir>/compilation/artifacts/tvmgen_default.h` | `artifacts/tvmgen_default.h` |
| `integer_model_config.h` | `<run_dir>/training/quantization/integer_model_config/integer_model_config.h` | `artifacts/integer_model_config.h` |
| `integer_model_config.c` | `<run_dir>/training/quantization/integer_model_config/integer_model_config.c` | `artifacts/integer_model_config.c` |
| `train_data.h` | `<run_dir>/training/quantization/integer_model_config/train_data.h` | `artifacts/train_data.h` |
| `train_data.c` | `<run_dir>/training/quantization/integer_model_config/train_data.c` | `artifacts/train_data.c` |
| `test_vector.c` | `<run_dir>/training/quantization/golden_vectors/test_vector.c` | `artifacts/test_vector.c` |
| `user_input_config.h` | `<run_dir>/training/quantization/golden_vectors/user_input_config.h` | `artifacts/user_input_config.h` |

**Note:** `mod.a` keeps its original name. Do not rename it to `mod_m33_ti_arm_clang.a`.

### Step 3: Verify the Project

After copying, right-click the project in CCS → **Refresh**. All 8 files should appear under `artifacts/` in the project explorer.

### Step 4: Build

Select the project → **Project → Build Project**. The build should complete with no errors.

---

## 7. Next Steps

- **See the artifacts in detail** → [Generated Artifacts Overview](generated_artifacts_overview.md)
- **Understand integer_model_config.h/.c** → [Integer Model Configuration](integer_model_config.md)
- **Understand train_data.h/.c** → [Training Data](training_data.md)
- **Write your own training loop** → [Integer ODL Library](integer_odl_lib.md)
