# On-Device Training Data

## Table of Contents

- [1. Overview](#1-overview)
- [2. ondevice_training_data.h](#2-ondevice_training_datah)
  - [2.1 Dimension Defines](#21-dimension-defines)
  - [2.2 How RAW_INPUT_SIZE is Calculated](#22-how-raw_input_size-is-calculated)
  - [2.3 Extern Declarations](#23-extern-declarations)
- [3. ondevice_training_data.c](#3-ondevice_training_datac)
  - [3.1 Data Format](#31-data-format)
  - [3.2 Label Encoding](#32-label-encoding)
  - [3.3 Data Arrays](#33-data-arrays)
- [4. How export_samples_per_class Maps to These Arrays](#4-how-export_samples_per_class-maps-to-these-arrays)
- [5. How the Application Uses This Data](#5-how-the-application-uses-this-data)
  - [5.1 DatasetIterator_t Pattern](#51-datasetiterator_t-pattern)
  - [5.2 Circular Iteration](#52-circular-iteration)
- [6. Memory Impact](#6-memory-impact)
- [7. Further Reading](#7-further-reading)

---

## 1. Overview

The `ondevice_training_data.h` and `ondevice_training_data.c` files provide embedded datasets that the microcontroller uses for on-device training, validation, and testing. These files are **auto-generated** by Modelzoo based on the `export_samples_per_class` parameter in the YAML configuration.

In a real deployment, training data would come from live sensor readings. However, for development, testing, and reproducibility, the example applications use pre-collected data compiled directly into the firmware. The on-device training library APIs accept data from any source — the embedded arrays are simply a convenient way to provide known data during development.

For context on how these files are generated, see [Running ModelZoo for On-Device Training](running_modelzoo_for_odt.md).

---

## 2. ondevice_training_data.h

### 2.1 Dimension Defines

```c
#define NUM_TRAIN_SAMPLES 10
#define NUM_VALIDATION_SAMPLES 6
#define NUM_TEST_SAMPLES 12
#define RAW_INPUT_SIZE 768
#define NUM_CLASSES 2
```

| Define | Meaning |
|--------|---------|
| `NUM_TRAIN_SAMPLES` | Number of samples available for training. Used to cycle through training data across epochs. |
| `NUM_VALIDATION_SAMPLES` | Number of samples available for validation during training and for threshold calculation after training. |
| `NUM_TEST_SAMPLES` | Number of samples available for pre-training and post-training evaluation. Includes both normal and anomaly samples. |
| `RAW_INPUT_SIZE` | Total number of floats per sample. This is the raw sensor data size **before** feature extraction. |
| `NUM_CLASSES` | Number of classes in the dataset . |

### 2.2 How `RAW_INPUT_SIZE` is Calculated

The raw input size depends on the sensor configuration and feature extraction settings:

```
RAW_INPUT_SIZE = num_channels × num_frame_concat × frame_size
```

For the fan blade example:
```
RAW_INPUT_SIZE = 3 channels (X, Y, Z) × 1 frame_concat × 256 frame_size = 768
```

Each sample contains the complete raw sensor data needed to produce one model input. When `num_frame_concat > 1`, multiple consecutive frames are included in a single sample so that the on-device feature extraction can concatenate them for temporal context.

### 2.3 Extern Declarations

```c
extern const float TRAIN_INPUTS[NUM_TRAIN_SAMPLES][RAW_INPUT_SIZE];
extern const int16_t TRAIN_LABELS[NUM_TRAIN_SAMPLES];

extern const float VALIDATION_INPUTS[NUM_VALIDATION_SAMPLES][RAW_INPUT_SIZE];
extern const int16_t VALIDATION_LABELS[NUM_VALIDATION_SAMPLES];

extern const float TEST_INPUTS[NUM_TEST_SAMPLES][RAW_INPUT_SIZE];
extern const int16_t TEST_LABELS[NUM_TEST_SAMPLES];
```

The arrays are declared `const` because the training data itself is never modified — only the model weights change during training. This allows the linker to place the data in Flash memory if desired, saving RAM.

---

## 3. ondevice_training_data.c

### 3.1 Data Format

Each input sample is a flat array of `RAW_INPUT_SIZE` floats containing raw sensor readings. The data is organized as:

```
Sample layout (for 3 channels, 1 frame_concat, 256 frame_size):

[channel_0_frame_0 (256 floats)] [channel_1_frame_0 (256 floats)] [channel_2_frame_0 (256 floats)]
|<─────────────────────────────── RAW_INPUT_SIZE = 768 ──────────────────────────────────────────>|
```

When `num_frame_concat > 1` (e.g., 4 frames):

```
Sample layout (for 3 channels, 4 frame_concat, 256 frame_size):

[ch0_frame0..ch0_frame3 (1024 floats)] [ch1_frame0..ch1_frame3 (1024 floats)] [ch2_frame0..ch2_frame3 (1024 floats)]
|<────────────────────────────────────── RAW_INPUT_SIZE = 3072 ──────────────────────────────────────────────────>|
```

The data is stored channel-first: all frames for channel 0, then all frames for channel 1, and so on. When reading the frame by frame data for feature extraction, this structure is important as it tells how the frames are stored. 

### 3.2 Label Encoding

```c
const int16_t TRAIN_LABELS[NUM_TRAIN_SAMPLES] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
const int16_t TEST_LABELS[NUM_TEST_SAMPLES] = { 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1 };
```


**For anomaly detection:**
- **Training and validation samples** are all normal (label = 1). The autoencoder learns to reconstruct normal patterns — it never sees anomaly data during training.
- **Test samples** contain a mix of normal and anomaly samples to evaluate detection accuracy.

| Label Value | Meaning |
|:-----------:|---------|
| `0` | Anomaly |
| `1` | Normal |


### 3.3 Data Arrays

```c
const float TRAIN_INPUTS[NUM_TRAIN_SAMPLES][RAW_INPUT_SIZE] = {
    { /* sample 0: 768 floats of raw sensor data */ },
    { /* sample 1: 768 floats of raw sensor data */ },
    // ... 10 samples total
};

const float VALIDATION_INPUTS[NUM_VALIDATION_SAMPLES][RAW_INPUT_SIZE] = {
    { /* sample 0: 768 floats of raw sensor data */ },
    // ... 6 samples total
};

const float TEST_INPUTS[NUM_TEST_SAMPLES][RAW_INPUT_SIZE] = {
    { /* sample 0: 768 floats of raw sensor data */ },
    // ... 12 samples total
};
```

---

## 4. How `export_samples_per_class` Maps to These Arrays

The YAML field `export_samples_per_class: [10, 6, 6]` specifies `[train, validation, test]` counts. 
### For anomaly detection

| Split | YAML Count | What Gets Exported | Resulting Define |
|-------|-----------|-------------------|-----------------|
| Train | 10 | 10 normal samples | `NUM_TRAIN_SAMPLES = 10` |
| Validation | 6 | 6 normal samples  | `NUM_VALIDATION_SAMPLES = 6` |
| Test | 6 | 6 normal + 6 anomaly = 12 samples | `NUM_TEST_SAMPLES = 12` |

**Key points:**
- Training and validation samples are drawn from the **normal class only** (the autoencoder trains on normal data).
- Test samples include **both normal and anomaly** classes for evaluation.
- For test data, the YAML count applies **per class**, so `6` means 6 normal + 6 anomaly samples.
---

## 5. How the Application Uses This Data

### 5.1 DatasetIterator_t Pattern

The example application wraps these arrays in a simple iterator structure:

```c
typedef struct {
    const float (*inputs)[RAW_INPUT_SIZE];  // Pointer to 2D input array
    const int16_t* labels;                   // Pointer to labels array
    uint16_t num_samples;                    // Total samples in this split
    uint16_t current_index;                  // Current position (circular)
} DatasetIterator_t;
```

Three iterators are initialized — one each for training, validation, and test data:

```c
train_iter.inputs = TRAIN_INPUTS;
train_iter.labels = TRAIN_LABELS;
train_iter.num_samples = NUM_TRAIN_SAMPLES;
train_iter.current_index = 0;
```

### 5.2 Circular Iteration

The iterators wrap around when they reach the end of the dataset:

```c
iter->current_index++;
if (iter->current_index >= iter->num_samples) {
    iter->current_index = 0;  // Wrap around
}
```

This means:
- **During training:** The 10 training samples are reused across epochs. With `batches_per_epoch: 10` and `batch_size: 1`, each epoch uses all 10 samples exactly once.

---

## 6. Memory Impact

The embedded training data is typically the largest artifact in an on-device training project.

**Formula:**
```
Memory (bytes) = (NUM_TRAIN + NUM_VAL + NUM_TEST) × RAW_INPUT_SIZE × 4 bytes/float
```

**Fan blade example:**
```
(10 + 6 + 12) × 768 × 4 = 28 × 768 × 4 = 86,016 bytes ≈ 84 KB
```

### Guidance on choosing sample counts

| More samples | Fewer samples |
|-------------|--------------|
| Better training convergence | Less flash memory usage |
| More representative threshold calculation | Faster threshold calculation |
| Better test evaluation statistics | — |
| May exceed flash budget on small MCUs | May under-represent data distribution |
---

## 7. Further Reading

- **How these files are generated** → [Running ModelZoo for On-Device Training](running_modelzoo_for_odt.md)
- **All generated artifacts at a glance** → [Generated Artifacts Overview](generated_artifacts_overview.md)
- **How the trainable model uses this data** → [Anomaly Detection ODT Library](anomaly_detection_odt.md)
- **See the data in action** → [Application Example — Fan Blade](application_example.md)