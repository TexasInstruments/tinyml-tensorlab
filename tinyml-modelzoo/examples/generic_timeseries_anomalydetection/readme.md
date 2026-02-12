# Generic Time Series Anomaly Detection: Hello World Example
### - Jaswanth Jadda, Abhijeet Pal, Adithya Thonse, Tushar Sharma, Fasna Sharaf

<hr>

## Overview

### Anomaly Detection in Time Series

Anomaly detection identifies unusual patterns or deviations from expected behavior in time series data. It's widely used in industrial monitoring, predictive maintenance, and quality control—such as detecting motor faults, sensor malfunctions, or equipment degradation. In **TinyML ModelZoo**, we support time series anomaly detection using autoencoder-based models.

This example serves as a **"Hello World" introduction** to time series anomaly detection using the TinyML ModelZoo toolchain. It demonstrates how to use **any generic time series anomaly detection task** with our toolchain using a simple, mathematically defined pattern.

### The Synthetic Dataset

To demonstrate anomaly detection, we use a clean synthetic dataset based on a combined sinusoidal pattern.

**Normal Pattern:**
- Signal follows: **y = 1.2 sin(2πft) + 0.8 cos(2πft)**
- Base frequency: f = 1.0 Hz
- Base amplitude variation up to +/- 10% and base frequency variation up to +/- 5%
- Added slight noise to the signal with mean = 0 and std dev = 0.05

**Anomaly Types:**
- **Frequency Faster**: Signal frequency increases by 30-70%
- **Frequency Slower**: Signal frequency decreases by 20-50%
- **Amplitude Higher**: Signal amplitude increases by 60-120%
- **Amplitude Lower**: Signal amplitude decreases by 40-70%

This example will walk you through:
- How the dataset should be structured for anomaly detection
- How to configure the YAML file for anomaly detection tasks
- Running the complete pipeline from training to compilation

## About the Dataset

The dataset consists of synthetically generated time series signals with controlled variations:

### Signal Characteristics

| Parameter | Normal Range | Anomaly Ranges |
|-----------|--------------|----------------|
| **Frequency** | 0.95 - 1.05 Hz | Faster: 1.3 - 1.7 Hz<br>Slower: 0.5 - 0.8 Hz |
| **Amplitude (sine)** | 1.08 - 1.32 | Higher: 1.92 - 2.64<br>Lower: 0.36 - 0.72 |
| **Amplitude (cosine)** | 0.72 - 0.88 | Higher: 1.28 - 1.76<br>Lower: 0.24 - 0.48 |
| **Noise** | 0.03 - 0.07 std dev | Same for all types |

### Dataset Composition

**Dataset split:**
- Normal samples: 60 files
- Anomaly samples: 32 files (8 of each type)
- Total files: 92 files

Each file contains 5000 samples representing 50 seconds of data at 100 Hz sampling rate.

The dataset can be downloaded from here: [`generic_timeseries_anomalydetection.zip`](https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/generic_timeseries_anomalydetection.zip)

## Understanding Temporal Dependencies

### Why Frame Size Matters

Anomaly detection requires **temporal context**—the model must see multiple consecutive samples to understand the pattern.

**Example with frame_size=1 (FAILS):**
```
Point at t=5.0s: value = 0.83
Point at t=5.1s: value = 0.91
Point at t=5.2s: value = 0.76

Problem: All values look normal! Cannot tell if the signal is from 1Hz or 1.5Hz frequency.
```

**Example with frame_size > 100 (WORKS):**
```
100 consecutive samples reveal:
- Normal (1 Hz): Exactly 1 complete cycle
- Faster (1.5 Hz): 1.5 cycles compressed into same window
- Slower (0.7 Hz): Only 0.7 of a cycle

The autoencoder learns: "Normal = 1 cycle in 100 samples"
Anomaly detected when it sees 1.5 cycles or 0.7 cycles!
```

### Recommended Frame Sizes

For this dataset (100 Hz sampling, 1 Hz base frequency):

| Frame Size | Cycles Visible | Detection Quality |
|------------|----------------|-------------------|
| 1 - 10     | < 0.1 cycles   | Insufficient (no pattern visible) |
| 20 - 50    | 0.2 - 0.5      | Weak (partial cycle) |
| 100        | 1.0 cycle      | Good (full cycle visible) |
| >200       | 2.0 cycles     | Excellent (multiple cycles) |

**Rule of thumb:** Frame size should capture **at least 1 complete cycle** 

## Preparing the Dataset

For anomaly detection tasks, **ModelZoo** expects the dataset to be organized in a specific folder structure

```
dataset_name/
│
├── classes/
│   ├── Normal/
│   │   ├── normal_0000.csv
│   │   ├── normal_0001.csv
│   │   ├── ...
│   │   └── normal_0059.csv         (60 files)
│   │
│   └── Anomaly/
│       ├── freq_faster_0000.csv
│       ├── freq_faster_0001.csv
│       ├── ...
│       ├── freq_slower_0000.csv
│       ├── amp_higher_0000.csv
│       ├── amp_lower_0000.csv
│       └── ...                     (32 files total)
```

### Data File Format

Each CSV file contains a single column with the signal values:

**Example: normal_0000.csv**
```csv
signal
0.023
0.045
0.067
0.089
0.112
...
(5000 rows)
```

### Dataset Splitting Strategy

**Important:** For anomaly detection:
- **Training set**: 50% of Normal files only (autoencoder learns normal patterns)
- **Validation set**: 10% of Normal files only (monitors training)
- **Test set**: 40% of Normal files + **ALL Anomaly files** (evaluates detection)

This ensures the model learns **only from normal data** and is evaluated on its ability to detect unseen anomalies.

**Note:**
ModelZoo automatically handles the splitting when you run the example. The tool will:
1. Scan the `Normal/` folder
2. Split normal samples into train (50%), validation (10%), test (40%) as per the specified configuration
3. Add all files from `Anomaly/` folder to the test set
4. Create annotation files automatically

For this example, we have already prepared the dataset in the required format. You can find the zipped dataset at: [`generic_timeseries_anomalydetection.zip`](https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/generic_timeseries_anomalydetection.zip)

## Usage in TinyML ModelZoo

You can run this example directly in **TinyML ModelZoo** using the following command:

```bash
./run_tinyml_modelzoo.sh examples/generic_timeseries_anomalydetection/config.yaml
```

The model pipeline is configured using a YAML file, where you can enable or disable different stages such as dataset loading, data processing, feature extraction, training, testing, and compilation depending on your needs.

## Configuring the YAML File

### `common` Section

Set the task type to `generic_timeseries_anomalydetection` along with other basic parameters:

```yaml
common:
    target_module: 'timeseries'
    task_type: 'generic_timeseries_anomalydetection'
    target_device: 'F28P55'
    run_name: '{date-time}/{model_name}'
```

**Key parameters:**
- `task_type`: Must be `generic_timeseries_anomalydetection` for anomaly detection tasks
- `target_device`: Target MCU (F28P55X in this example)
---
### `dataset` Section

Defines dataset details and splitting strategy:

```yaml
dataset:
    enable: True
    dataset_name: generic_timeseries_anomalydetection
    input_data_path: '/path/to/generic_anomaly_detection.zip'
    split_factor: [0.5, 0.1, 0.4]
```

**Key parameters:**
- **`dataset_name`**: Name for your dataset (appears in logs)
- **`input_data_path`**: Path to the dataset zip file
- **`split_type`**: How to split data
  - `'amongst_files'`: Divide files into train/val/test sets 
  - `'within_files'`: Split each file internally
- **`split_factor`**: `[train, val, test]` ratios (must sum to ≤ 1.0)

---

### `data_processing_feature_extraction` Section

Configures how the raw signal is processed and what features are extracted:

```yaml
data_processing_feature_extraction:
   data_proc_transforms:
    - SimpleWindow
    - Downsample
    frame_size: 100
    sampling_rate: 100
    new_sr: 10
    variables: 1

```

**Key parameters:**

- **`data_proc_transforms`**: List of preprocessing operations
  - `SimpleWindow`: Extracts sliding windows from the signal
  - `Downsample`: Downsamples the input timeseries data form `sampling_rate` to `new_sr` specified above
 
- **`frame_size`**: Number of samples per window

- **`variables`**: Number of input channels
  - 1 for this dataset (single signal)
---

### `training` Section

Configures the autoencoder model and training parameters:

```yaml
training:
    enable: True
    model_name: 'AD_17k'
    model_config: ''
    batch_size: 64
    training_epochs: 50
    num_gpus: 1
    learning_rate: 0.001
    quantization: 1
    output_int: False
```

**Key parameters:**

- **`model_name`**: Autoencoder architecture
  - `'AD_17k'`: ~17,000 parameters (used in this example)

- **`training_epochs`**: Number of training iterations
  - 50 epochs is sufficient for this simple dataset
  - Model converged at epoch 46 in this run

- **`batch_size`**: Number of samples per training step

- **`num_gpus`**: Use GPU acceleration if available
  - 1 = use GPU (faster training)
  - 0 = use CPU only

- **`quantization`**: Model precision
  - `0` = Float model (no quantization)
  - `1` = Quantization for CPU (used in this example)
  - `2` = Quantization for TI NPU

- **`output_int`**: **Must be `False` for anomaly detection**
  - Reconstruction error must be in float for threshold comparison

**Note:** The autoencoder is trained **only on normal data**. It learns to reconstruct normal patterns with low error. When it sees an anomaly, reconstruction error is high → anomaly detected!

---
### `testing` and `compilation` Sections

Enable or disable testing and compilation as needed:

```yaml
testing:
    enable: True

compilation:
    enable: True 
    compile_preset_name: 'forced_soft_npu_preset'
```

**Key parameters:**
- **`testing.enable`**: Runs evaluation on test set (normal + anomaly samples)
- **`compilation.enable`**: Compiles model for F28P55X deployment
- **`compile_preset_name`**: `'forced_soft_npu_preset'` compiles for CPU

## Results

This section explains the training process, how anomaly detection works, and how to interpret the results from your actual run.

### How Autoencoder-Based Anomaly Detection Works

**Training Phase (Normal Data Only):**
```
1. Autoencoder learns to compress and reconstruct normal patterns
2. For normal samples: Reconstruction error is LOW
   Input:  100 frame size input signal
   Output: Reconstructed signal
   Error:  Mean Squared Error ≈ 0.008 (very small)
3. Model learns: "This is what normal looks like"
```

**Testing Phase (Normal + Anomaly Data):**
```
Normal sample:
   Input:  [0.8, 1.1, 1.3, 1.2, ...]
   Output: [0.79, 1.12, 1.31, 1.19, ...]
   Error:  MSE ≈ 0.008, Correctly identified as normal

Anomaly sample (higher frequency or amplitude):
   Input:  [1.6, 2.2, 2.6, 2.4, ...]
   Output: [0.81, 1.14, 1.29, 1.21, ...]  ← Tries to reconstruct as "normal"
   Error:  MSE ≈ 0.69, Flagged as anomaly! (88× higher than normal)
```

**Detection Logic:**
```
if reconstruction_error > threshold:
    → ANOMALY
else:
    → NORMAL
```

### Training Process

ModelZoo performs the following steps:

1. **Float Training** (Epochs 0-49)
   - Trains autoencoder on normal training data only
   - Monitors validation loss (reconstruction error on normal validation set)
   - Best epoch: 46 with validation MSE = 0.008

2. **Quantization Training** (Epochs 0-9)
   - Fine-tunes model with Quantization-Aware Training (QAT)
   - Prepares model for 8-bit integer deployment on MCU
   - Best epoch: 5 with validation MSE = 0.009

3. **Threshold Calculation**
   - Computes reconstruction errors on normal training data
   - Mean = 0.007815, Std = 0.002208
   - Generates thresholds for k = 0 to 4.5

4. **Testing**
   - Evaluates on test set
   - Calculates performance metrics for each threshold

### Training Logs from Your Run

#### Float Training - Best Epoch
```
INFO: root.main.FloatTrain.BestEpoch: Best Epoch: 46
INFO: root.main.FloatTrain.BestEpoch: MSE 0.008
```

**Interpretation:**
- Model converged at epoch 46
- Validation MSE: 0.008 (low reconstruction error on normal validation data)

#### Quantized Training - Best Epoch
```
INFO: root.main.QuantTrain.BestEpoch: Best Epoch: 5
INFO: root.main.QuantTrain.BestEpoch: MSE 0.009
```

**Interpretation:**
- Quantized model converged quickly at epoch 5
- Validation MSE: 0.009 (slightly higher than float, but still excellent)

### Reconstruction Error Statistics

```
INFO: root: Reconstruction Error Statistics:
INFO: root: Normal training data - Mean: 0.007815, Std: 0.002208
INFO: root: Anomaly test data - Mean: 0.689792, Std: 0.461646
INFO: root: Normal test data - Mean: 0.008051, Std: 0.002295
```

**Key observations:**
- **Normal training data**: Mean = 0.0078 (model learned normal patterns well)
- **Normal test data**: Mean = 0.0081 (excellent generalization! Almost identical to training)
- **Anomaly test data**: Mean = 0.6898 (**88× higher than normal!**)

**Excellent separation:** The massive difference between normal (≈0.008) and anomaly (≈0.69) reconstruction errors enables highly reliable detection with very low false alarm rates.

### Threshold Performance (k = 4.5)

```
INFO: root: Threshold for K = 4.5 : 0.017751
INFO: root: False positive rate: 0.07%
INFO: root: Anomaly detection rate (recall): 100.00%
INFO: root: Accuracy: 99.97%
INFO: root: Precision: 99.95%
INFO: root: F1 Score: 99.97%
```

**Interpretation:**
- **Threshold**: 0.0178 (mean + 4.5 × std)
- **Recall**: 100% (detected **all anomalies** across all 4 types)
- **Precision**: 99.95% (only 7 false alarms out of 22,456 samples)
- **False Positive Rate**: 0.07% (7 normal samples out of 9,624 flagged incorrectly)

### Confusion Matrix

```
+-----------------------+------------------------+-------------------------+
|                       |   Predicted as: Normal |   Predicted as: Anomaly |
+=======================+========================+=========================+
| Ground Truth: Normal  |                   9617 |                       7 |
+-----------------------+------------------------+-------------------------+
| Ground Truth: Anomaly |                      0 |                   12832 |
+-----------------------+------------------------+-------------------------+
```

**Breakdown:**
- **True Negatives (TN)**: 9,617 normal samples correctly identified (99.93%)
- **False Positives (FP)**: 7 normal samples incorrectly flagged (0.07%)
- **False Negatives (FN)**: 0 anomalies missed (**100% detection rate!**)
- **True Positives (TP)**: 12,832 anomalies correctly detected

Model successfully detects **all 4 anomaly types** (faster freq, slower freq, higher amp, lower amp) despite being trained only on normal data!

### Threshold Performance Table

Here's the complete threshold performance from your run:

| k_value | threshold | accuracy | precision | recall | f1_score | false_positive_rate | true_positives | true_negatives | false_positives | false_negatives |
|---------|-----------|----------|-----------|--------|----------|---------------------|----------------|----------------|-----------------|-----------------|
| 0.0     | 0.0078    | 78.73    | 72.88     | 100.0  | 84.31    | 49.63               | 12,832         | 4,848          | 4,776           | 0               |
| 0.5     | 0.0089    | 86.44    | 80.82     | 100.0  | 89.39    | 31.64               | 12,832         | 6,579          | 3,045           | 0               |
| 1.0     | 0.0100    | 92.24    | 88.04     | 100.0  | 93.64    | 18.11               | 12,832         | 7,881          | 1,743           | 0               |
| 1.5     | 0.0111    | 95.97    | 93.41     | 100.0  | 96.59    | 9.41                | 12,832         | 8,718          | 906             | 0               |
| 2.0     | 0.0122    | 97.78    | 96.26     | 100.0  | 98.09    | 5.18                | 12,832         | 9,125          | 499             | 0               |
| 2.5     | 0.0133    | 98.92    | 98.15     | 100.0  | 99.07    | 2.51                | 12,832         | 9,382          | 242             | 0               |
| 3.0     | 0.0144    | 99.49    | 99.11     | 100.0  | 99.55    | 1.19                | 12,832         | 9,509          | 115             | 0               |
| 3.5     | 0.0155    | 99.77    | 99.60     | 100.0  | 99.80    | 0.54                | 12,832         | 9,572          | 52              | 0               |
| 4.0     | 0.0166    | 99.92    | 99.85     | 100.0  | 99.93    | 0.20                | 12,832         | 9,605          | 19              | 0               |
| 4.5     | 0.0178    | 99.97    | 99.95     | 100.0  | 99.97    | 0.07                | 12,832         | 9,617          | 7               | 0               |


### Visualization of Results

ModelZoo generates histogram plots showing the distribution of reconstruction errors:

**Files generated:**
- `reconstruction_error_histogram.png` (linear scale)
- `reconstruction_error_histogram_log.png` (log scale - better for visualizing separation)

**Location:**
```
tinyml-modelmaker/data/projects/generic_anomaly_detection/run/{run-id}/AD_17k/training/quantization/post_training_analysis/
```

**What to look for:**
- **Two distinct peaks**: Normal (left, ~0.008) and Anomaly (right, ~0.69)
- **Huge gap**: Almost no overlap between distributions
- **Threshold line**: Shows decision boundary (at k=4.5, threshold=0.0178)

The massive separation (88× difference in means) makes threshold selection very robust.

## Viewing Detailed Results

After training completes, all results are stored in:
```
tinyml-modelmaker/data/projects/generic_anomaly_detection/run/{run-id}/AD_17k
```

### Key Output Directories

| Directory | Contents |
|-----------|----------|
| **`training/base/`** | Float model training results |
| **`training/base/golden_vectors/`** | Float model test vectors |
| **`training/quantization/`** | Quantized model training results |
| **`training/quantization/golden_vectors/`** | Quantized test vectors (for on-device validation) |
| **`training/quantization/post_training_analysis/`** | **Threshold performance analysis** |
| **`training/quantization/post_training_analysis/threshold_performance.csv`** | **Complete metrics table** |
| **`training/quantization/post_training_analysis/reconstruction_error_histogram.png`** | **Error distribution (linear)** |
| **`training/quantization/post_training_analysis/reconstruction_error_histogram_log.png`** | **Error distribution (log scale)** |
| **`compilation/artifacts/`** | Compiled model for F28P55X |

### Files for On-Device Deployment

To run the model on F28P55X, you need:

1. **Compiled model artifacts:**
   ```
   compilation/artifacts/
   ├── mod.a                    # Compiled model library
   └── tvmgen_default.h         # Model header file
   ```

2. **Golden vectors (for validation):**
   ```
   training/quantization/golden_vectors/
   ├── test_vector.c            # Input/output test vectors
   └── user_input_config.h      # Feature extraction configuration
   ```

## Running on Device

After successfully running ModelZoo, you will get the compiled model artifacts:

1. **Artifacts**:
   - `mod.a` and `tvmgen_default.h` are generated and stored in:
     ```
     tinyml-modelmaker/data/projects/generic_anomaly_detection/run/20260130-164026/AD_17k/compilation/artifacts
     ```

2. **Golden Vectors**:
   - `user_input_config.h` and `test_vector.c` are stored in:
     ```
     tinyml-modelmaker/data/projects/generic_anomaly_detection/run/20260130-164026/AD_17k/training/quantization/golden_vectors
     ```

Steps to run this example on-device can be found by following this guide: [Deploying Anomaly Detection Models from ModelZoo to Device](../../docs/deploying_anomaly_detection_models_from_modelzoo_to_device/readme.md)


For more complex real-world examples, see:
- [Fan Blade Anomaly Detection](../fan_blade_fault_classification/readme_anomaly_detection.md)
- [Motor Vibration Monitoring](../motor_bearing_fault/readme_anomaly_detection.md)


**Update history:**  
[2nd Feb 2026]: Compatible with v1.3 of TinyML ModelZoo