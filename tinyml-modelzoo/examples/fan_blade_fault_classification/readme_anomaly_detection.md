
# **Fan Blade Anomaly Detection using Vibration Analysis**
#### - Jaswanth Jadda, Adithya Thonse, Fasna S,Tushar Sharma, Abhijeeth Pal

## **Overview**

Fan blade faults such as imbalance, damage, and obstruction can lead to reduced efficiency, increased energy consumption, and potential system failure. Early detection of these anomalies through vibration analysis enables predictive maintenance, preventing costly downtime and extending equipment lifespan.

In this example, we demonstrate how to use **TinyML ModelMaker** to train an autoencoder-based anomaly detection model for fan blade fault detection. The model learns normal vibration patterns from a healthy fan and automatically detects deviations that indicate potential faults—including fault types it has never seen during training.

To learn more about anomaly detection, refer to the **[Anomaly Detection Model Architecture](../../docs/anomaly_detection/Anomaly_Model_Architecture.md)**.

To learn more about the overall strategy to prepare dataset, to train and evaluate the results, refer to the **[Dataset Preparation, Training & Evaluation Strategy for Anomaly Detection](../../docs/anomaly_detection/Dataset_Training_And_Evaluation_For_Anomaly_Detection.md)**.

### **What This Example Covers**

This guide walks through:
1. Dataset preparation (vibration data from 3-axis accelerometer)
2. Feature extraction configuration (FFT, binning, log scaling)
3. Model training with TinyML ModelMaker
4. Results interpretation


# **About the Dataset**

This dataset contains vibration measurements from a fan system collected using a **3-axis ADXL355 accelerometer**. The data captures both normal operation and various fault conditions.

---

## **Dataset Specifications**

| Parameter | Value |
|-----------|-------|
| **Sensor** | ADXL35|
| **Sampling Rate** | 4 kHz (4000 samples per second) |
| **Channels** | 3 (Vibration X, Y, Z axes) |
| **Samples per File** | ~20,000 samples (approximately 5 seconds of data) |
| **Normal Files** | 100 files |
| **Anomaly Files** | 187 files |
| **Total Files** | 287 files |

---

## **Data Format**

Each file is a CSV with the following structure:

**Columns:**
- `Time`: Timestamp (automatically dropped by ModelMaker)
- `Vibx`: Vibration measurement along X-axis
- `Viby`: Vibration measurement along Y-axis
- `Vibz`: Vibration measurement along Z-axis

**Example data:**
```
Time,Vibx,Viby,Vibz
0,1022,2056,62993
1,418,3870,64459
2,918,3721,63433
3,550,4300,63963
4,670,4994,63202
5,687,3545,63462
6,1093,4896,63318
7,1122,3090,63437
8,1663,3717,64174
9,1606,3710,63756
...
```

**Note:** The `Time` column is automatically excluded during data loading (any column with "time" in the header is dropped).

---

## **Fault Types**

The dataset includes the following conditions:

### **Normal Operation**
- Healthy fan blade
- Balanced rotation
- **100 files** representing various normal operating conditions

### **Anomalies**

The dataset contains **187 anomaly files** representing three types of faults:

| Fault Type | Description | 
|------------|-------------|
| **Imbalance** | Uneven mass distribution on fan blade | 
| **Damage** | Physical damage to blade  |
| **Obstruction** | Interfering with blade rotation |

 The model is trained **only on normal data**. All three anomaly types are used exclusively for testing and evaluation.

---

## **Dataset Organization**

The dataset follows the standard anomaly detection folder structure:

```
fan_blade_ad_dataset.zip/
│
├── classes/
│   ├── Normal/
│   │   ├── normal_001.csv
│   │   ├── normal_002.csv
│   │   ├── ...
│   │   └── normal_100.csv          (100 files)
│   │
│   └── Anomaly/
│       ├── imbalance_001.csv
│       ├── imbalance_002.csv
│       ├── damage_001.csv
│       ├── damage_002.csv
│       ├── obstruction_001.csv
│       ├── ...
│       └── obstruction_xxx.csv     (187 files total)
```

---

## **Download the Dataset**

The dataset is pre-packaged and available for download:

**URL:** 
```
https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/fan_blade_ad_dataset.zip
```

You don't need to manually download it—ModelMaker will automatically download and extract the dataset when you run the example.

---

## **Data Collection Setup**

**Hardware:**
- Fan system 
- ADXL355 3-axis accelerometer
- Data acquisition system sampling at 4 kHz

**Anomaly Induction:**
- **Imbalance:** Added weight to one blade
- **Damage:** Simulated blade damage (controlled defects)
- **Obstruction:** Placed obstruction in airflow path


To understand the detailed setup for data collection, please refer to the **[Fan Blade Fault Classification
](readme.md)**.

---

## **Dataset Preparation**

For this example, **the dataset is already prepared** in the required format. You don't need to reorganize files or create annotations—ModelMaker handles everything automatically.

**What ModelMaker does:**
1. Downloads and extracts the zip file
2. Scans `Normal/` and `Anomaly/` folders
3. Splits normal data into train/val/test (60%/10%/30%)
4. Adds all anomaly files to test set
5. Creates annotation files automatically

For detailed information on dataset format and splitting strategy, see: **[Dataset Preparation, Training & Evaluation Strategy for Anomaly Detection](../../docs/anomaly_detection/Dataset_Training_And_Evaluation_For_Anomaly_Detection.md)**.


## **Usage in TinyML ModelMaker**

You can run this example directly in **TinyML ModelMaker** using the following command:

```bash
./run_tinyml_modelmaker.sh examples/fan_blade_fault_classification/config_anomaly_detection.yaml
```

The model pipeline is configured using a YAML file, where you can enable or disable different stages such as dataset loading, data processing, feature extraction, training, testing, and compilation depending on your needs.

---

## **Configuring the YAML File**

### **`common` Section**

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

### **`dataset` Section**

Defines dataset details and splitting strategy:

```yaml
dataset:
    enable: True
    dataset_name: fan_blade_fault
    input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/fan_blade_ad_dataset.zip'
    split_type: 'amongst_files'
    split_factor: [0.6, 0.1, 0.3]
```

**Key parameters:**
- **`dataset_name`**: Descriptive name (appears in logs and output paths)
- **`input_data_path`**: URL or local path to dataset
- **`split_type`**: How to split data
  - `'amongst_files'`: Divide files into train/val/test sets
  - `'within_files'`: Split each file internally
- **`split_factor`**: `[train, val, test]` ratios (must sum to ≤ 1.0)

**Important:** For anomaly detection:
- **Training set**: 60% of Normal files only
- **Validation set**: 10% of Normal files only
- **Test set**: 30% of Normal files + **ALL Anomaly files**

This split strategy ensures the model learns only from healthy operation data.

---

### **`data_processing_feature_extraction` Section**

Configures feature extraction pipeline for vibration data:

```yaml
data_processing_feature_extraction:
    data_proc_transforms: []
    feature_extraction_name: Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1
    variables: 3
    num_frame_concat: 4
    stride_size: 0.5
```

**Key parameters:**
- **`variables`**: Number of input channels (3 for X, Y, Z axes)
- **`feature_extraction_name`**: Predefined feature extraction configuration

#### **Feature Extraction Pipeline Breakdown**

The feature extraction name `Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1` encodes the following pipeline:

**Step-by-step processing:**

1. **Frame Size: 256 samples**
   - Window of 256 samples from raw vibration data
   - At 4 kHz sampling rate: 256 samples = 64 ms window

2. **FFT_FE (Fast Fourier Transform)**
   - Converts time-domain signal to frequency-domain
   - Reveals frequency components in vibration

3. **FFT_POS_HALF (Positive Half)**
   - Keeps positive half of FFT 
   - Reduces data size (FFT is symmetric for real signals)

4. **DC_REMOVE**
   - Removes DC component (0 Hz)

5. **ABS (Absolute Value)**
   - Takes magnitude of complex FFT output
6. **BINNING**
   - Groups frequency bins into 16 features
   - Reduces dimensionality while preserving spectral shape

7. **LOG_DB (Logarithmic Scale)**
   - Converts magnitude to dB
8. **CONCAT (Concatenation)**
   - Concatenates 4 consecutive frames
   - Captures temporal patterns over 256 ms (4 × 64 ms)
   - Provides context for anomaly detection

**Result:**
- **Input to model**: 3 channels × 16 features × 4 frames = **192 features**

---

### **`training` Section**

Configures model architecture and training parameters:

```yaml
training:
    enable: True
    model_name: 'AD_17k'
    model_config: ''
    batch_size: 64
    training_epochs: 200
    num_gpus: 0
    learning_rate: 0.001
    quantization: 1
    output_int: False
```

**Key parameters:**
- **`model_name`**: `'AD_17k'` (autoencoder with ~17,000 parameters)
- **`training_epochs`**: 200 epochs (model converges around epoch 150-200)
- **`quantization`**:
  - `0` = No quantization (float model)
  - `1` = Quantization for CPU
  - `2` = Quantization for TI NPU (recommended for F28P55X)
- **`output_int`**: **Must be `False`** for anomaly detection
  
---

### **`testing` and `compilation` Sections**

Enable or disable testing and compilation as needed:

```yaml
testing:
    enable: True

compilation:
    enable: True
    compile_preset_name: forced_soft_npu_preset
```

**Key parameters:**
- **`testing.enable`**: Runs evaluation on test set (normal + anomaly)
- **`compilation.enable`**: Compiles model for F28P55X deployment
- **`compile_preset_name`**: `forced_soft_npu_preset` Compiles for TI CPU

---

## **Results**

This section explains the training process and how to interpret the results.

### **Training Process**

ModelMaker performs the following steps:

1. **Float Training**
   - Trains autoencoder on normal data only
   - Monitors validation loss (reconstruction error on normal validation set)
   - Selects best epoch based on lowest validation loss

2. **Quantization Training**
   - Fine-tunes model with Quantization-Aware Training (QAT)
   - Prepares model for 8-bit integer deployment on MCU

3. **Threshold Calculation**
   - Computes reconstruction errors on normal training data
   - Calculates statistics: `mean_train` and `std_train`
   - Generates thresholds for k = 0 to 4.5 using formula:
     ```
     threshold = mean_train + k × std_train
     ```

4. **Testing**
   - Evaluates model on test set (normal + anomaly samples)
   - Calculates performance metrics for each threshold
   - Generates confusion matrix, precision, recall, F1 score

---

### **Training Logs**

#### **Float Training - Best Epoch**

```
INFO: root.main.FloatTrain.BestEpoch: Printing statistics of best epoch:
INFO: root.main.FloatTrain.BestEpoch: Best Epoch: 198
INFO: root.main.FloatTrain.BestEpoch: MSE 13.111
```

**Interpretation:**
- Model converged at epoch 198
- Validation MSE: 13.111 (reconstruction error on normal validation data)

---

#### **Quantized Training - Best Epoch**

```
INFO: root.main.QuantTrain.BestEpoch: Printing statistics of best epoch:
INFO: root.main.QuantTrain.BestEpoch: Best Epoch: 26
INFO: root.main.QuantTrain.BestEpoch: MSE 13.815
```

**Interpretation:**
- Quantized model converged at epoch 26
- Validation MSE: 13.815 (slightly higher than float due to quantization)
- Still maintains good performance for anomaly detection

---

### **Reconstruction Error Statistics**

```
INFO: root: Reconstruction Error Statistics:
INFO: root: Normal training data - Mean: 16.130465, Std: 3.011868
INFO: root: Anomaly test data - Mean: 356.065460, Std: 368.977722
INFO: root: Normal test data - Mean: 16.260649, Std: 2.986460
```

**Key observations:**
- **Normal training data**: Mean error = 16.13 (model learned to reconstruct normal patterns)
- **Normal test data**: Mean error = 16.26 (excellent generalization, very similar to training)
- **Anomaly test data**: Mean error = 356.07 (22× higher than normal!)

**Excellent separation:** Clear distinction between normal and anomaly reconstruction errors.

### **Threshold Performance (k = 4)**

```
INFO: root: Threshold for K = 4.0 : 28.177937
INFO: root: False positive rate: 0.15%
INFO: root: Anomaly detection rate (recall): 100.00%
INFO: root: Accuracy: 99.98%
INFO: root: Precision: 99.98%
INFO: root: F1 Score: 99.99%
```

**Interpretation:**
- **Threshold**: 28.18 (mean + 4.0 × std)
- **Recall**: 100% (detected all anomalies)
- **Precision**: 99.98% (almost no false alarms)
- **False Positive Rate**: 0.15% (3 normal samples out of 1950 flagged incorrectly)

---
### **Confusion Matrix**


```
                     Predicted Normal    Predicted Anomaly
Actual Normal                    1947                     3
Actual Anomaly                      0                 13545
```

**Breakdown:**
- **True Negatives (TN)**: 1947 normal samples correctly identified
- **False Positives (FP)**: 3 normal samples incorrectly flagged (0.15% FPR)
- **False Negatives (FN)**: 0 anomalies missed (100% recall)
- **True Positives (TP)**: 13,545 anomalies correctly detected

---

### **Threshold Performance CSV**


ModelMaker generates a CSV file with metrics for different threshold values (k = 0 to 4.5):

**Location:** 
```
data/projects/fan_blade_fault/run/{date-time}/AD_17k/training/quantization/post_training_analysis/threshold_performance.csv
```

**Complete table:**

| k_value | threshold | accuracy | precision | recall | f1_score | false_positive_rate | true_positives | true_negatives | false_positives | false_negatives |
|---------|-----------|----------|-----------|--------|----------|---------------------|----------------|----------------|-----------------|-----------------|
| 0.0     | 16.13     | 94.93    | 94.52     | 100.0  | 97.18    | 40.26               | 13545          | 1165           | 785             | 0               |
| 0.5     | 17.64     | 97.28    | 96.98     | 100.0  | 98.47    | 21.64               | 13545          | 1528           | 422             | 0               |
| 1.0     | 19.14     | 98.27    | 98.06     | 100.0  | 99.02    | 13.74               | 13545          | 1682           | 268             | 0               |
| 1.5     | 20.65     | 98.72    | 98.56     | 100.0  | 99.27    | 10.15               | 13545          | 1752           | 198             | 0               |
| 2.0     | 22.15     | 99.14    | 99.03     | 100.0  | 99.51    | 6.82                | 13545          | 1817           | 133             | 0               |
| 2.5     | 23.66     | 99.65    | 99.60     | 100.0  | 99.80    | 2.77                | 13545          | 1896           | 54              | 0               |
| 3.0     | 25.17     | 99.89    | 99.87     | 100.0  | 99.94    | 0.87                | 13545          | 1933           | 17              | 0               |
| 3.5     | 26.67     | 99.97    | 99.97     | 100.0  | 99.99    | 0.21                | 13545          | 1946           | 4               | 0               |
| 4.0     | 28.18     | 99.98    | 99.98     | 100.0  | 99.99    | 0.15                | 13545          | 1947           | 3               | 0               |
| 4.5     | 29.68     | 99.98    | 99.98     | 100.0  | 99.99    | 0.15                | 13545          | 1947           | 3               | 0               |
---

### **Reconstruction Error Histogram**

ModelMaker generates histogram plots showing the distribution of reconstruction errors:

**Files generated:**
- `reconstruction_error_histogram.png` (linear scale)
- `reconstruction_error_histogram_log.png` (log scale - better for visualizing separation)

**Location:**
```
data/projects/fan_blade_fault/run/{date-time}/AD_17k/training/quantization/post_training_analysis/
```

![](assets/reconstruction_error_log_scale.png)
## **Viewing Detailed Results**

After training completes, all results are stored in:

```
data/projects/fan_blade_fault/run/{date-time}/AD_17k/
```

### **Key Output Directories**

| Directory | Contents |
|-----------|----------|
| **`training/base/`** | Float model training results |
| **`training/base/golden_vectors/`** | Float model test vectors (`test_vector.c`, `user_input_config.h`) |
| **`training/quantization/`** | Quantized model training results |
| **`training/quantization/golden_vectors/`** | Quantized model test vectors (for on-device validation) |
| **`training/quantization/post_training_analysis/`** | **Threshold performance analysis** |
| **`training/quantization/post_training_analysis/threshold_performance.csv`** | **Metrics for k=0 to 4.5** |
| **`training/quantization/post_training_analysis/reconstruction_error_histogram.png`** | **Error distribution plot (linear scale)** |
| **`training/quantization/post_training_analysis/reconstruction_error_histogram_log.png`** | **Error distribution plot (log scale)** |
| **`compilation/artifacts/`** | Compiled model for F28P55X (`mod.a`, `tvmgen_default.h`) |

---

### **Files for On-Device Deployment**

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
---
## **Performance Metrics**

Here are the key performance metrics for the quantized model running on F28P55X:

| Metric               | Value       |
|----------------------|-------------|
| **Cycles**           | 3156090     |
| **Inference Time**   | 21040.6 µs  |
| **Results Match**    | TRUE        |
| **Application Size** | 40628 bytes |
| **Flash Usage**      | 34783 bytes |
| **SRAM Usage**       | 5845 bytes  |