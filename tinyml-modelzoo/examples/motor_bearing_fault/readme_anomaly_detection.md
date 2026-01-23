# **Motor Bearing Fault Detection using Vibration Analysis**
#### - Jaswanth Jadda, Adithya Thonse, Fasna S,Tushar Sharma, Abhijeeth Pal

## **Overview**

Motor bearing faults such as lack of lubrication, erosion, localized defects, contamination, and flaking can lead to reduced efficiency, increased vibration, overheating, and catastrophic motor failure. Early detection of these bearing anomalies through vibration analysis enables predictive maintenance, preventing costly downtime and extending motor lifespan.

In this example, we demonstrate how to use **TinyML ModelMaker** to train an autoencoder-based anomaly detection model for motor bearing fault detection. The model learns normal vibration patterns from a healthy bearing and automatically detects deviations that indicate potential bearing faults—including fault types it has never seen during training.

To learn more about anomaly detection, refer to the **[Anomaly Detection Model Architecture](../../docs/anomaly_detection/Anomaly_Model_Architecture.md)**.

To learn more about the overall strategy to prepare dataset, to train and evaluate the results, refer to the **[Dataset Preparation, Training & Evaluation Strategy for Anomaly Detection](../../docs/anomaly_detection/Dataset_Training_And_Evaluation_For_Anomaly_Detection.md)**.

### **What This Example Covers**

This guide walks through:
1. Dataset preparation (vibration data from 3-axis accelerometer)
2. Feature extraction configuration (FFT, binning, log scaling)
3. Model training with TinyML ModelMaker
4. Results interpretation

# **About the Dataset**

This dataset contains vibration measurements from motor bearings collected using a **3-axis accelerometer**. The data captures both normal bearing operation and various bearing fault conditions commonly encountered in industrial motors.


## **Dataset Specifications**

| Parameter | Value |
|-----------|-------|
| **Sensor** | 3-axis accelerometer (X, Y, Z axes) |
| **Sampling Rates** | 10 Hz, 20 Hz, 30 Hz, 40 Hz (variable) |
| **Channels** | 3 (Vibration X, Y, Z axes) |
| **Samples per File** | Variable: 24,000, 72,000, or 144,000 samples |
| **Normal Files** | 36 files (healthy bearing operation) |
| **Anomaly Files** | 180 files (various bearing faults) |
| **Total Files** | 216 files |
| **Motor Configurations** | Multiple test motors (m0, m1, m2) |

## **Data Format**

Each file is a CSV (Excel format) with the following structure:

**Columns:**
- Column 1: Timestamp
- Column 2: Vibration measurement along X-axis
- Column 3: Vibration measurement along Y-axis  
- Column 4: Vibration measurement along Z-axis

**Note:** If a time column exists with "time" in its header (case-insensitive), it will be automatically dropped by ModelMaker.

**Example data (healthy bearing):**
```csv
Time	Vibx	Viby	Vibz
19393	-492	-470	64040
19394	-510	-491	64085
19395	-436	-585	64122
19396	-268	-565	64173
```

**Example data (bearing flaking fault):**
```csv
Time	Vibx	Viby	Vibz
1	   -1143	-351	64832
2	   403	    170	    63511
3	   -772	    -1402	64298
4	   -1958	-3539	64835
```

## **Bearing Fault Types**

The dataset includes the following conditions:

### **Normal Operation**
- Healthy bearing with proper lubrication
- Balanced operation
- No defects or contamination
- **36 files** representing normal bearing operation under various conditions

### **Bearing Faults**

The dataset contains **180 fault files** representing five types of bearing anomalies:

| Fault Type | Description |
|------------|-------------|
| **No Lubrication** | Bearing operating without adequate lubrication |
| **Erosion** | Surface wear due to abrasive particles or corrosion | 
| **Localized Fault** | Localized defects (pits, spalls, cracks) on races or rolling elements |
| **Contamination** | Foreign particles (dirt, metal debris) in bearing | 
| **Flaking** | Surface material flaking off due to fatigue |

**Important:** The model is trained **only on normal bearing data**. All five fault types are used exclusively for testing and evaluation.


## **Dataset Organization**

The dataset follows the standard anomaly detection folder structure:

```
motor_fault_anomaly_detection_dataset.zip/
│
├── classes/
│   ├── Normal/                              # Healthy bearing operation
│   │   ├── bearingNormal_..._1.csv
│   │   ├── bearingNormal_..._2.csv
│   │   ├── bearingNormal_..._1.csv
│   │   ├── ...
│   │   └── bearingNormal_..._N.csv     # 36 files total
│   │
│   └── Anomaly/                             # All bearing fault types
│       ├── bearingNoLubrication_..._1.csv
│       ├── bearingErosion_..._1.csv
│       ├── bearingLocalizedFault_..._1.csv
│       ├── bearingContaminated_..._1.csv
│       ├── bearingFlaking_..._1.csv
│       ├── ...
│       └── bearingFlaking_..._N.csv     # 180 files total
```

**Key characteristics:**
- **Normal folder:** 36 files containing only healthy bearing vibration
- **Anomaly folder:** 180 files with all 5 fault types combined
- **Variable sample counts:** Files contain 24k, 72k, or 144k samples 

## **Download the Dataset**

The dataset is pre-packaged and available for download:

**URL:** 
```
https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/motor_fault_anomaly_detection_dataset.zip
```

You don't need to manually download it—ModelMaker will automatically download and extract the dataset when you run the example.

## **Data Collection Setup**

**Hardware:**
- Electric motor with test bearings
- 3-axis accelerometer (mounted on bearing housing)
- Data acquisition system with variable sampling rates (10 Hz - 40 Hz)

**Normal Data Collection:**
- Healthy bearings with proper lubrication
- Various motor operating conditions

**Fault Data Collection:**

Simulate the following faults and collect the vibration data:
- No Lubrication
- Erosion 
- Localized Fault 
- Contamination 
- Flaking 

## **Dataset Preparation**

For this example, **the dataset is already prepared** in the required format. You don't need to reorganize files or create annotations—ModelMaker handles everything automatically.

**What ModelMaker does:**
1. Downloads and extracts the zip file
2. Scans `Normal/` and `Anomaly/` folders
3. Splits normal data into train/val/test (60%/30%/10%)
4. Adds all anomaly files to test set
5. Creates annotation files automatically

For detailed information on dataset format and splitting strategy, see: **[Dataset Preparation, Training & Evaluation Strategy for Anomaly Detection](../../docs/anomaly_detection/Dataset_Training_And_Evaluation_For_Anomaly_Detection.md)**.


## **Usage in TinyML ModelMaker**

You can run this example directly in **TinyML ModelMaker** using the following command:

```bash
./run_tinyml_modelmaker.sh examples/motor_fault_detection/config_anomaly_detection.yaml
```

The model pipeline is configured using a YAML file, where you can enable or disable different stages such as dataset loading, data processing, feature extraction, training, testing, and compilation depending on your needs.


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


### **`dataset` Section**

Defines dataset details and splitting strategy:

```yaml
dataset:
    enable: True 
    dataset_name: 'motor_fault_example_dsk_ad'
    input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/motor_fault_anomaly_detection_dataset.zip'
```

**Key parameters:**
- **`dataset_name`**: Descriptive name (appears in logs and output paths)
- **`input_data_path`**: URL or local path to dataset

### **`data_processing_feature_extraction` Section**

Configures feature extraction pipeline for vibration data:

```yaml
data_processing_feature_extraction:
    feature_extraction_name: Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1
    variables: 3
    feature_size_per_frame: 4
```

**Key parameters:**
- **`variables`**: Number of input channels (3 for X, Y, Z axes)
- **`feature_extraction_name`**: Predefined feature extraction configuration
- **`feature_size_per_frame`**: Number of features per frame after binning (4 in this case)


#### **Feature Extraction Pipeline Breakdown**

The feature extraction name `Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1` encodes the following pipeline:

**Step-by-step processing:**

1. **Frame Size: 256 samples**
   - Window of 256 samples from raw vibration data

2. **FFT_FE (Fast Fourier Transform)**
   - Converts time-domain vibration signal to frequency-domain
   - Reveals frequency components characteristic of bearing faults

3. **FFT_POS_HALF (Positive Half)**
   - Keeps positive half of FFT spectrum
   - Reduces data size (FFT is symmetric for real signals)

4. **DC_REMOVE**
   - Removes DC component (0 Hz)

5. **ABS (Absolute Value)**
   - Takes magnitude of complex FFT output

6. **BINNING (256 features → 4 features per frame)**
   - Groups 256 features into 4 features per frame
   - Reduces dimensionality while preserving key features

7. **LOG_DB (Logarithmic Scale)**
   - Converts magnitude to decibels (dB)

8. **CONCAT (8 frames)**
   - Concatenates 8 consecutive frames
   - Provides temporal context for anomaly detection
   - Captures evolution of vibration patterns over time

**Result:**
- **Input to model**: 3 channels × 4 features × 8 frames = **96 features**


### **`training` Section**

Configures model architecture and training parameters:

```yaml
training:
    enable: True
    model_name: 'AD_Linear'
    model_config: ''
    batch_size: 64
    learning_rate: 0.001
    training_epochs: 200
    num_gpus: 1
    quantization: 1
    output_int: False
```

**Key parameters:**
- **`model_name`**: `'AD_Linear'` (lightweight linear autoencoder)
- **`training_epochs`**: 200 epochs (model converged at epoch 182)
- **`batch_size`**: 64
- **`num_gpus`**: 1 (set to 0 for CPU-only training)
- **`quantization`**:
  - `0` = No quantization (float model)
  - `1` = Quantization for CPU (QAT - used in this example)
  - `2` = Quantization for TI NPU
- **`output_int`**: **Must be `False`** for anomaly detection
  - Ensures reconstruction error is computed in float precision




### **`testing` and `compilation` Sections**

Enable or disable testing and compilation as needed:

```yaml
testing:
    enable: True

compilation:
    enable: True 
    compile_preset_name: 'forced_soft_npu_preset'
```

**Key parameters:**
- **`testing.enable`**: Runs evaluation on test set (normal + anomaly)
- **`compilation.enable`**: Compiles model for F28P55X deployment
- **`compile_preset_name`**: `forced_soft_npu_preset` compiles for TI CPU


## **Results**

This section explains the training process and how to interpret the results.

### **Training Process**

ModelMaker performs the following steps:

1. **Float Training**
   - Trains autoencoder on normal bearing data only
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


### **Training Logs**

#### **Float Training - Best Epoch**

```
INFO: root.main.FloatTrain.BestEpoch: Printing statistics of best epoch:
INFO: root.main.FloatTrain.BestEpoch: Best Epoch: 182
INFO: root.main.FloatTrain.BestEpoch: MSE 1.602
```

**Interpretation:**
- Model converged at epoch 182
- Validation MSE: 1.602 (reconstruction error on normal validation data)

#### **Quantized Training - Best Epoch**

```
INFO: root.main.QuantTrain.BestEpoch: Printing statistics of best epoch:
INFO: root.main.QuantTrain.BestEpoch: Best Epoch: 39
INFO: root.main.QuantTrain.BestEpoch: MSE 1.773
```

**Interpretation:**
- Quantized model converged at epoch 39
- Validation MSE: 1.773 (10.7% increase from float due to 8-bit quantization but ~3-4x smaller model size than normal)
- Still maintains excellent performance for anomaly detection
---

### **Reconstruction Error Statistics**

```
INFO: root: Reconstruction Error Statistics:
INFO: root: Normal training data - Mean: 1.662490, Std: 1.968127
INFO: root: Anomaly test data - Mean: 141.985321, Std: 112.756683
INFO: root: Normal test data - Mean: 2.849831, Std: 1.343052
```

**Key observations:**

**Excellent separation (85× difference):**
- **Normal training data**: Mean error = 1.66 (model learned healthy bearing patterns)
- **Normal test data**: Mean error = 2.85 (good generalization)
- **Anomaly test data**: Mean error = 141.99 (**85× higher than normal!**)

**Motor bearing detection shows the best relative separation**, making it ideal for reliable fault detection with minimal false alarms.

---

### **Threshold Performance (k = 4.5)**

```
INFO: root: Threshold for K = 4.5 : 10.519060
INFO: root: False positive rate: 0.00%
INFO: root: Anomaly detection rate (recall): 100.00%
INFO: root: Accuracy: 100.00%
INFO: root: Precision: 100.00%
INFO: root: F1 Score: 100.00%
```

**Interpretation:**
- **Threshold**: 10.52 (mean + 4.5 × std)
- **Recall**: 100% (detected all 180 bearing fault samples across 5 fault types)
- **Precision**: 100% (zero false alarms on normal bearing test samples)
- **False Positive Rate**: 0.00% (perfect—no healthy bearings misclassified)

**Perfect detection achieved!** This is possible because:
- Large separation between healthy and fault error distributions (85× difference)
- Comprehensive healthy training data (36 files covering normal operation)
- Bearing fault signatures are highly distinct from healthy operation
- All 5 fault types produce vibration patterns clearly different from normal

---

### **Confusion Matrix**

```
Confusion Matrix:
                     Predicted Normal    Predicted Anomaly
Actual Normal                    899                     0
Actual Anomaly                     0                 54900
```

**Breakdown:**
- **True Negatives (TN)**: 899 normal bearing samples correctly identified
- **False Positives (FP)**: 0 normal samples incorrectly flagged (0% FPR)
- **False Negatives (FN)**: 0 faults missed (100% recall)
- **True Positives (TP)**: 54,900 bearing fault samples correctly detected


### **Threshold Performance CSV**

ModelMaker generates a CSV file with metrics for different threshold values (k = 0 to 4.5):

**Location:** 
```
data/projects/motor_fault_example_dsk_ad/run/{date-time}/AD_Linear/training/quantization/post_training_analysis/threshold_performance.csv
```


**Complete table:**

| k_value | threshold | accuracy | precision | recall | f1_score | false_positive_rate | true_positives | true_negatives | false_positives | false_negatives |
|---------|-----------|----------|-----------|--------|----------|---------------------|----------------|----------------|-----------------|-----------------|
| 0.0     | 1.662     | 98.65    | 98.65     | 100.0  | 99.32    | 83.54               | 54900          | 148            | 751             | 0               |
| 0.5     | 2.647     | 99.14    | 99.14     | 100.0  | 99.57    | 53.17               | 54900          | 421            | 478             | 0               |
| 1.0     | 3.631     | 99.71    | 99.70     | 100.0  | 99.85    | 18.13               | 54900          | 736            | 163             | 0               |
| 1.5     | 4.615     | 99.88    | 99.88     | 100.0  | 99.94    | 7.23                | 54900          | 834            | 65              | 0               |
| 2.0     | 5.599     | 99.92    | 99.92     | 100.0  | 99.96    | 4.78                | 54900          | 856            | 43              | 0               |
| 2.5     | 6.583     | 99.96    | 99.96     | 100.0  | 99.98    | 2.67                | 54900          | 875            | 24              | 0               |
| 3.0     | 7.567     | 99.98    | 99.98     | 100.0  | 99.99    | 1.11                | 54900          | 889            | 10              | 0               |
| 3.5     | 8.551     | 99.99    | 99.99     | 100.0  | 100.0    | 0.56                | 54900          | 894            | 5               | 0               |
| 4.0     | 9.535     | 100.0    | 100.0     | 100.0  | 100.0    | 0.22                | 54900          | 897            | 2               | 0               |
| 4.5     | 10.519    | 100.0    | 100.0     | 100.0  | 100.0    | 0.00                | 54900          | 899            | 0               | 0               |

### **Reconstruction Error Histogram**

ModelMaker generates histogram plots showing the distribution of reconstruction errors:

**Files generated:**
- `reconstruction_error_histogram.png` (linear scale)
- `reconstruction_error_histogram_log.png` (log scale - better for visualizing separation)

**Location:**
```
data/projects/motor_fault_example_dsk_ad/run/{date-time}/AD_Linear/training/quantization/post_training_analysis/
```

![Reconstruction Error Histogram - Log Scale](assets/reconstruction_error_log_scale.png)
*Figure 2: Reconstruction error distribution (log scale). Shows clear separation between normal and anomaly distributions.*

**Expected and Observed visualization characteristics:**
- **Normal samples (blue)**: Tight cluster at low error (mean ~2.85, std ~1.34)
- **Fault samples (red)**: Broad distribution at high error (mean ~142, std ~113)
- **Threshold line (green)**: Clear separation at k=4.5 (10.52)
- **Overlap**: None (perfect separation)

**Interpretation:**
- **No overlap** indicates the model can reliably distinguish all bearing fault types from healthy operation
- **Tight normal cluster** shows consistent reconstruction of healthy bearing patterns
- **Wide fault distribution** reflects diversity of the 5 bearing fault types (each has different vibration signature)

---

## **Viewing Detailed Results**

After training completes, all results are stored in:

```
data/projects/motor_fault_example_dsk_ad/run/{date-time}/AD_Linear/
```

### **Key Output Directories**

| Directory | Contents |
|-----------|----------|
| **`training/base/`** | Float model training results (checkpoints, logs) |
| **`training/base/golden_vectors/`** | Float model test vectors (`test_vector.c`, `user_input_config.h`) |
| **`training/quantization/`** | Quantized model training results |
| **`training/quantization/golden_vectors/`** | **Quantized model test vectors (for on-device validation)** |
| **`training/quantization/post_training_analysis/`** | **Threshold performance analysis** |
| **`training/quantization/post_training_analysis/threshold_performance.csv`** | **Metrics for k=0 to 4.5** |
| **`training/quantization/post_training_analysis/reconstruction_error_histogram.png`** | **Error distribution plot (linear scale)** |
| **`training/quantization/post_training_analysis/reconstruction_error_histogram_log.png`** | **Error distribution plot (log scale)** |
| **`compilation/artifacts/`** | **Compiled model for F28P55X (`mod.a`, `tvmgen_default.h`)** |

---

### **Files for On-Device Deployment**

To run the model on F28P55X for real-time motor bearing monitoring, you need:

1. **Compiled model artifacts:**
   ```
   compilation/artifacts/
   ├── mod.a                    # Compiled model library
   └── tvmgen_default.h         # Model header file (input/output dimensions)
   ```

2. **Golden vectors (for validation):**
   ```
   training/quantization/golden_vectors/
   ├── test_vector.c            # Input/output test vectors
   └── user_input_config.h      # Feature extraction configuration
   ```

3. **Threshold value:**
   - From `threshold_performance.csv`, select appropriate k value
   - Example: k=3.0 → threshold = 7.567
   - Hardcode in application code as per the usecase 

---

## **Performance Metrics**

Here are the key performance metrics for the quantized model:

| Metric               | Value       | Notes |
|----------------------|-------------|-------|
| **Training Time (Float)** | 1 min 40 sec | 200 epochs on GPU |
| **Training Time (Quantization)** | 58 sec | 40 epochs QAT  |
| **Cycles** | 102604 |102604 cycles to run AI model |
| **Model Size** | ~6k parameters | Lightweight linear autoencoder |
| **Inference Time** | 684 us | On F28P55X MCU |
| **Flash Usage** | 13KB | Model weights + code |
| **SRAM Usage** | ~400B | Runtime memory (activations, buffers) |
| **Results Match** | TRUE | Golden vectors validated |
