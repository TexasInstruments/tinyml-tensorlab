# **Dataset Preparation, Training & Evaluation Strategy for Anomaly Detection**

## Table of Contents
[1. Overview](#1-overview)<br>
[2. Dataset Format](#2-dataset-format)<br>
[3. Datafile Format](#3-datafile-format)<br>
[4. Dataset Splitting](#4-dataset-splitting-strategy)<br>
[5. Labelling Requirements](#5-labeling-requirements)<br>
[6. Training](#6-training-anomaly-detection-models-with-modelmaker)<br>
[7. Evaluation](#7-evaluation-metrics-for-anomaly-detection)<br>

## **1. Overview**

Anomaly detection is fundamentally different from classification and regression tasks. While classification requires labeled examples of all classes (including anomalies), **anomaly detection in ModelMaker uses a semi-supervised approach** where the model is trained **only on "normal" data**.

### **Key Differences from Classification**

| Aspect | Classification | Anomaly Detection |
|--------|----------------|-------------------|
| **Training Data** | Normal + Anomaly (all labeled) | **Only Normal data** |
| **Goal** | Predict which class an input belongs to | Detect if input deviates from "normal" |
| **Handles Unseen Types** |  Cannot classify unknown classes |  **Detects any deviation from normal** |
| **Labeling Effort** | High (need examples of all anomaly types) | Low (just collect normal samples and some possible anomalies for evaluation) |

### **Why This Approach?**

In real-world applications, anomalies are often:
- **Rare**: Hard to collect enough samples for training
- **Diverse**: New failure modes emerge over time
- **Unknown**: You may not know all possible anomaly types in advance

By training only on normal data, the model learns what "normal" behavior looks like. During inference, **anything that deviates from this learned pattern is flagged as an anomaly**—even if it's a type of failure the model has never seen before.

### **What This Document Covers**

This guide explains:
- How to structure your dataset (Normal vs Anomaly folders)
- Why training, validation, and test sets are split differently than in classification
- How to evaluate anomaly detection performance (precision, recall, F1 score)
- Best practices for collecting and labeling data

For details on **how the model works internally** (autoencoder architecture, reconstruction error, etc.), see the companion document: *"[Anomaly Detection Model Architecture](Anomaly_Model_Architecture.md)"*.



# **2. Dataset Format**

For anomaly detection tasks, **ModelMaker** expects the dataset to be organized with separate folders for **Normal** and **Anomaly** data. The structure is different from classification, where each class has its own folder.

## **2.1 Folder Structure**

The prepared dataset should have the following structure:

```
dataset_name/
│
├── classes/
│   ├── Normal/
│   │   ├── file1.csv
│   │   ├── file2.csv
│   │   └── ...
│   └── Anomaly/
│       ├── file1.csv
│       ├── file2.csv
│       └── ...
│
└── annotations/                          # Optional
    ├── file_list.txt                     # List of all files in the dataset
    ├── instances_train_list.txt          # List of files for training (only Normal)
    ├── instances_val_list.txt            # List of files for validation (only Normal)
    └── instances_test_list.txt           # List of files for testing (Normal + Anomaly)
```

### **Key Points:**

- **`classes`**: This is the directory containing your data files.

- **`Normal/` folder**: Contains all samples representing normal operating conditions. **All training and validation data comes from this folder.**

- **`Anomaly/` folder**: Contains samples representing anomalous behavior (faults, failures, defects, etc.). **These are used ONLY for testing**, never for training.

- **`annotations/` folder**: Optional. If not provided, ModelMaker will automatically generate the annotation files based on your `split_type` and `split_factor` settings.


## **2.2 Providing the Dataset to ModelMaker**

ModelMaker accepts the dataset path through the `input_data_path` argument in the `config.yaml` file.

### **Supported Formats:**

| Format | Example | Notes |
|--------|---------|-------|
| **URL** | `https://example.com/dataset.zip` | ModelMaker downloads and extracts automatically |
| **Local zip file** | `path_to_dataset/my_dataset.zip` | ModelMaker extracts to working directory |
| **Local directory** | `path_to_dataset/my_dataset/` | Should contain `classes` immediately inside |

### **Important:**

- If providing a **zip file**, the structure inside should be:
  ```
  dataset.zip/
  ├── classes/     ← Immediately inside the zip
  │   ├── Normal/
  │   └── Anomaly/
  ```
  **Don't** add an extra folder level like `dataset.zip/dataset_name/classes/`

- If providing a **directory**, specify the path up to `{dataset_name}/`. ModelMaker will look for the `classes` folder inside.


## **2.3 Configuration in YAML**

In your `config.yaml` file, specify:

```yaml
dataset:
    enable: True
    dataset_name: 'fan_blade_fault'
    input_data_path: 'path_to_dataset/fan_blade_ad_dataset.zip'
```

- **`dataset_name`**: A descriptive name (appears in output paths and logs)
- **`input_data_path`**: URL or local path to your dataset

**Note:** The data directory (containing Normal/ and Anomaly/ subfolders) is automatically named 'classes' for anomaly detection tasks.


## **2.4 Example Dataset Structure**

Here's a concrete example for a fan blade fault detection dataset:

```
fan_blade_ad_dataset.zip/
│
├── classes/
│   ├── Normal/
│   │   ├── normal_001.csv
│   │   ├── normal_002.csv
│   │   ├── ...
│   │   └── normal_100.csv              # 100 normal samples
│   │
│   └── Anomaly/
│       ├── imbalance_001.csv
│       ├── imbalance_002.csv
│       ├── damage_001.csv
│       ├── obstruction_001.csv
│       ├── ...
│       └── obstruction_005.csv         # 20 anomaly samples (mixed types)
│
└── annotations/                         # Optional - auto-generated if missing
    ├── file_list.txt
    ├── instances_train_list.txt
    ├── instances_val_list.txt
    └── instances_test_list.txt
```

**Note:** All anomaly types (imbalance, damage, obstruction) go in the same `Anomaly/` folder. The model treats all of them as "not normal" and doesn't distinguish between different anomaly types.


# **3. Datafile Format**

ModelMaker supports multiple file formats for time-series data. Each file can contain single or multiple variables (e.g., vibration measurements from X, Y, Z axes).


## **3.1 Supported File Formats**

| Format | Description |
|--------|-------------|
| **`.csv`** | Comma-separated values |
| **`.txt`** | Tab or comma-separated | 
| **`.npy`** | NumPy binary format | 
| **`.pkl`** | Pandas pickle format |


## **3.2 File Content Formats**

There are two accepted formats for how data can be organized within each file:

### **3.2.1 Headerless Format**

**No header row** and **no index column** in the file.

**Best for:** Single-variable measurements (e.g., single current sensor)

**Example:**
```
2078
2136
2117
2077
2029
1989
2056
```

### **3.2.2 Headered Format**

**Header row present** with column names, optionally with an index column (e.g., timestamp).

**Best for:** Multi-variable measurements (e.g., X, Y, Z axes from accelerometer)

**Example - Single variable with timestamp:**
```
Time(sec),Current(A)
-0.7969984,7.84
-0.7969952,7.76
-0.796992,7.76
-0.7969888,7.76
-0.7969856,7.76
-0.7969824,7.84
-0.7969792,7.84
```

**Example - Multiple variables with timestamp:**
```
Time,Vibration_X,Vibration_Y,Vibration_Z
0.0000,-2753,-558,64376
0.0001,-2551,-468,63910
0.0002,-424,-427,64032
0.0003,1429,-763,64132
0.0004,1236,-974,64065
0.0005,-903,-547,64242
0.0006,-1512,-467,63919
```

## **3.3 Time Column Handling**

**Important:** Any column with the text **`time`** in its header (case-insensitive) is **automatically dropped** by ModelMaker.

**Examples of dropped columns:**
- `Time`
- `TIME`
- `Timestamp`
- `time(ms)`
- `TIME (microsec)`

**Why?** Time values are typically not useful features for the model. The temporal relationships are captured through the sequential nature of the data and windowing operations.

**Warning:** If you have a column that is useful for your model but contains "time" in its name, **rename it** before using the dataset (e.g., `uptime_hours` → `operation_hours`).

## **3.4 Multi-Channel Data**

For sensors with multiple measurement axes (e.g., 3-axis accelerometer), organize data as separate columns:

```
Time,Vibration_X,Vibration_Y,Vibration_Z
0.0000,1234,5678,9012
0.0001,1235,5679,9013
...
```

In your `config.yaml`, specify the number of variables:

```yaml
data_processing_feature_extraction:
    variables: 3  # X, Y, Z axes
```

ModelMaker will automatically use the **first `variables` columns** (after dropping time columns) as input.

## **3.5 File Naming Conventions**

- **No strict naming requirements**, but descriptive names help with debugging
- **Avoid special characters** in filenames (use `_` instead of spaces)
- **Examples:**
  - `normal_run_001.csv`
  - `bearing_healthy_2024-01-15.csv`
  - `imbalance_fault_high_speed.csv`

## **3.6 Example Files**

### **Fan Blade Dataset (3-axis vibration):**

**Normal/normal_001.csv:**
```
Time,Vibration_X,Vibration_Y,Vibration_Z
0.00000,-2753,-558,64376
0.00025,-2551,-468,63910
0.00050,-424,-427,64032
0.00075,1429,-763,64132
...
```

**Anomaly/imbalance_001.csv:**
```
Time,Vibration_X,Vibration_Y,Vibration_Z
0.00000,-5234,-1203,68421
0.00025,-4982,-1156,67893
0.00050,-5123,-1189,68234
0.00075,-5001,-1145,68012
...
```

**Note:** The time column will be dropped, and only the 3 vibration columns will be used as input (since `variables: 3` in config).

# **4. Dataset Splitting Strategy**

In anomaly detection, the way data is split into training, validation, and test sets is **fundamentally different** from classification tasks. This section explains how ModelMaker handles dataset splitting for anomaly detection.

## **4.1 Split Types**

ModelMaker supports two methods for splitting your dataset:

### **4.1.1 `amongst_files` (Default)**

Files are divided into train, validation, and test sets.

**Example:** 
- You have 100 normal files (each with 1000 samples)
- `split_factor: [0.6, 0.1, 0.3]`
- **Result:**
  - Train: 60 files (each still has 1000 samples)
  - Val: 10 files (each still has 1000 samples)
  - Test: 30 files (each still has 1000 samples)


### **4.1.2 `within_files`**

Each file is split internally, and all files appear in all splits (with different portions).

**Example:**
- You have 100 normal files (each with 1000 samples)
- `split_factor: [0.6, 0.1, 0.3]`
- **Result:**
  - Train: 100 files (each has first 600 samples)
  - Val: 100 files (each has next 100 samples)
  - Test: 100 files (each has last 300 samples)


## **4.2 Split Factor Configuration**

The `split_factor` parameter defines how data is divided:

```yaml
dataset:
    split_type: 'amongst_files'      # or 'within_files'
    split_factor: [0.6, 0.1, 0.3]    # [train, val, test]
```

**Common configurations:**

| Configuration | Train | Val | Test | Use Case |
|---------------|-------|-----|------|----------|
| `[0.6, 0.1, 0.3]` | 60% | 10% | 30% | Balanced (default) |
| `[0.7, 0.15, 0.15]` | 70% | 15% | 15% | More training data |
| `[0.8, 0.1, 0.1]` | 80% | 10% | 10% | Large dataset, need more training samples |

**Note:** The values must sum to ≤ 1.0


## **4.3 What Goes Where: The Critical Difference**

This is where anomaly detection differs significantly from classification:

| Split | Normal Data | Anomaly Data | Purpose |
|-------|-------------|--------------|---------|
| **Training** |  60% of Normal files |  **None** | Learn what "normal" looks like |
| **Validation** |  10% of Normal files |  **None** | Monitor overfitting on normal patterns |
| **Test** |  30% of Normal files |  **All Anomaly files** | Evaluate detection performance |

### **Key Points:**

1. **Training uses ONLY Normal data**
   - The model learns to reconstruct normal patterns
   - Anomalies are deliberately excluded

2. **Validation uses ONLY Normal data**
   - Used to select the best epoch based on normal reconstruction error
   - Prevents overfitting to training data

3. **Test uses BOTH Normal and Anomaly data**
   - Normal test samples: Used to evaluate how well model detects normal samples. 
   - Anomaly samples: Used to evaluate how well the model detects anomalies
   - Metrics (precision, recall, F1) are calculated on the test set

## **4.4 Why This Split Strategy?**

### **Why train only on Normal data?**

The autoencoder learns to **compress and reconstruct normal patterns** with low error. When it encounters an anomaly (which it has never seen), it produces a **high reconstruction error** because it doesn't know how to reconstruct that pattern.

**Analogy:** 
- Train a JPEG encoder only on cat images
- It compresses/decompresses cats perfectly
- Show it a dog image → produces artifacts (high error)

### **Why validation uses only Normal data?**

We want to monitor how well the model generalizes to **unseen normal samples**, not how it handles anomalies. The validation set helps us:
- Detect overfitting (train error low, validation error high)
- Select the best epoch (lowest validation error on normal data)

### **Why test needs BOTH Normal and Anomaly?**

- **Normal test samples:** Ensure the model doesn't produce false alarms on normal data it hasn't seen
- **Anomaly samples:** Measure the model's ability to detect actual faults

This reflects real-world deployment: most of the time the system operates normally, with occasional anomalies.


## **4.5 Automatic Annotation Generation**

If you don't provide an `annotations/` folder, ModelMaker automatically creates it based on your split configuration.

**For `amongst_files`:**
1. Scans `Normal/` folder
2. Randomly splits normal files: 60% → train, 10% → val, 30% → test
3. Adds **all files from `Anomaly/`** to test list
4. Writes annotation files

**For `within_files`:**
1. Scans `Normal/` folder
2. For each normal file, splits content: first 60% → train, next 10% → val, last 30% → test
3. Creates new files for each split (e.g., `normal_001_train.csv`, `normal_001_val.csv`, `normal_001_test.csv`)
4. Adds **all files from `Anomaly/`** to test list
5. Writes annotation files

**Generated files:**
```
annotations/
├── file_list.txt                  # All Normal + Anomaly files
├── instances_train_list.txt       # Only Normal files (60%)
├── instances_val_list.txt         # Only Normal files (10%)
└── instances_test_list.txt        # Normal files (30%) + All Anomaly files
```





# **5. Labeling Requirements**

Anomaly detection has much simpler labeling requirements compared to classification tasks. You don't need to label individual samples or provide detailed annotations—just organize files into the correct folders.


## **5.1 Training Data: Minimal Labeling**

For training, you only need to identify which data is "normal":

 **What you need:**
- Collect samples representing normal operating conditions
- Place all normal files in the `Normal/` folder
- **No per-sample labels required**
- **No need to distinguish between different normal conditions** (e.g., different speeds, loads—all go in the same folder)

 **What you DON'T need:**
- Labels for individual time steps
- Annotations for specific features
- Subcategories within normal data

**Example:**
```
Normal/
├── normal_low_speed_001.csv
├── normal_high_speed_002.csv
├── normal_medium_load_003.csv
└── ...
```
All these go in the same `Normal/` folder, even though they represent different operating conditions.

## **5.2 Testing Data: Separate Normal and Anomaly**

For testing, you need to separate normal and anomaly samples so ModelMaker can evaluate detection performance:

 **What you need:**
- **Normal test samples:** Place in `Normal/` folder (will be automatically split into test set)
- **Anomaly samples:** Place in `Anomaly/` folder (all used for testing)

**Folder structure:**
```
classes/
├── Normal/
│   ├── normal_001.csv
│   ├── normal_002.csv
│   └── ...
└── Anomaly/
    ├── fault_001.csv
    ├── fault_002.csv
    └── ...
```

## **5.3 Multiple Anomaly Types**

If you have different types of anomalies (e.g., imbalance, damage, bearing wear), **all of them go in the same `Anomaly/` folder**.

**The model treats all anomalies as a single class: "not normal".**

**Example:**
```
Anomaly/
├── imbalance_001.csv
├── imbalance_002.csv
├── damage_001.csv
├── damage_002.csv
├── bearing_wear_001.csv
└── obstruction_001.csv
```
## **5.4 Labeling Effort Comparison**

| Task | Classification | Anomaly Detection |
|------|----------------|-------------------|
| **Training labels** | Need examples of all classes | Only need "normal" samples |
| **Anomaly samples** | Must collect and label all types | Optional for training, needed for testing |
| **New anomaly types** | Must retrain with new labeled samples | Automatically detected (no retraining needed) |
| **Effort** | High | **Low** |


# **6. Training Anomaly Detection Models with ModelMaker**

Once your dataset is prepared and organized, you can train an anomaly detection model using Tiny ML ModelMaker. This section provides a high-level overview of the training workflow. For detailed, step-by-step examples with specific datasets, see:
- **[Fan Blade Anomaly Detection Example](../examples/fan_blade_fault_classification/readme_anomaly_detection.md)**
- **[Motor Fault Anomaly Detection Example](../examples/motor_bearing_fault/readme_anomaly_detection.md)**



## **6.1 Training Workflow Overview**

The typical workflow for training an anomaly detection model consists of:

```
1. Prepare Dataset → 2. Configure YAML → 3. Run ModelMaker → 4. Review Results → 5. Deploy
```



## **6.2 Step 1: Prepare Your Dataset**

Organize your data following the structure described in Section 2:

```
{dataset_name}.zip/
├── classes/
│   ├── Normal/       
│   └── Anomaly/      
```

Ensure:
-  Normal folder contains diverse, high-quality samples
-  Files are in supported format (.csv, .npy, .pkl, .txt)
-  Data is properly formatted (see Section 3)



## **6.3 Step 2: Create Configuration File**

Create a `config.yaml` file specifying your training parameters:

### **Minimal Configuration Example:**

```yaml
common:
    target_module: 'timeseries'
    task_type: 'generic_timeseries_anomalydetection'
    target_device: 'F28P55'
    run_name: '{date-time}/{model_name}'

dataset:
    enable: True
    dataset_name: 'my_anomaly_dataset'
    input_data_path: './datasets/my_dataset.zip'
    split_type: 'amongst_files'
    split_factor: [0.6, 0.1, 0.3]

data_processing_feature_extraction:
    data_proc_transforms: []
    variables: 3  # Number of input channels (e.g., X, Y, Z)

training:
    enable: True
    model_name: 'TimeSeries_Generic_AD_17k_t'
    batch_size: 64
    training_epochs: 200
    num_gpus: 0
    quantization: 2
    output_int: False

testing:
    enable: True

compilation:
    enable: True
```

### **Key Parameters:**

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `task_type` | Must be `'generic_timeseries_anomalydetection'` | Fixed |
| `input_data_path` | Path to dataset (URL or local) | Your dataset location |
| `split_factor` | [train, val, test] ratios | `[0.6, 0.1, 0.3]` |
| `variables` | Number of input channels | 1, 3, etc. |
| `model_name` | Specify model name | Explore available models |
| `training_epochs` | Number of training iterations | 10-200 |
| `quantization` | 0=No Quantization, 1=Quantization for CPU, 2=Quantization for TI NPU | 0 - 2 |
| `output_int` | Must be `False` for anomaly detection | `False` |



## **6.4 Step 3: Run ModelMaker**

Execute the training pipeline:

```bash
./run_tinyml_modelmaker.sh path/to/your/config.yaml
```

### **What Happens During Training:**

1. **Dataset Loading**
   - Downloads/extracts dataset if needed
   - Creates train/val/test splits (if annotations not provided)
   - Loads only Normal data for training and validation

2. **Data Processing**
   - Applies windowing (SimpleWindow)
   - Optional: Feature extraction (FFT, binning, etc.)
   - Normalizes data

3. **Model Training**
   - Trains autoencoder on normal training data
   - Monitors validation loss (on normal validation data)
   - Selects best epoch (lowest validation loss)

4. **Quantization Training** (if `quantization:1 or 2`)
   - Fine-tunes with QAT or PTQ
   - Prepares model for MCU deployment

5. **Threshold Calculation**
   - Computes reconstruction errors on training data
   - Calculates `mean_train` and `std_train`
   - Generates thresholds for k = 0 to 4.5

6. **Testing**
   - Evaluates model on test set (normal + anomaly)
   - Calculates metrics for each threshold
   - Generates CSV with results

7. **Compilation**
   - Exports model for target device (F28P55)
   - Generates artifacts (mod.a, other headers)
   - Creates golden vectors for validation



## **6.5 Step 4: Review Training Results**

After training completes, ModelMaker generates outputs in:

```
data/projects/{dataset_name}/run/{date-time}/{model_name}/
```

### **Key Output Files:**

| File/Folder | Description |
|-------------|-------------|
| `training/base/` | Float model training results |
| `training/quantization/` | Quantized model results |
| `training/quantization/post_training_analysis/` | **Threshold performance analysis** |
| `training/quantization/post_training_analysis/threshold_performance.csv` | **Metrics for k=0 to 4.5** |
| `training/quantization/post_training_analysis/reconstruction_error_histogram.png` | **Error distribution plot** |
| `training/quantization/golden_vectors/` | Test vectors and config for on-device validation |
| `compilation/artifacts/` | Compiled model (mod.a, tvmgen_default.h) |


## **6.6 Understanding Training Logs**

### **Example Log Output:**

```
INFO: Best epoch: 39
INFO: MSE: 1.773

INFO: Reconstruction Error Statistics:
INFO: Normal training data - Mean: 1.662490, Std: 1.968127
INFO: Anomaly test data - Mean: 141.985321, Std: 112.756683
INFO: Normal test data - Mean: 2.849831, Std: 1.343052

INFO: Threshold for K = 4.5: 10.519060
INFO: False positive rate: 0.00%
INFO: Anomaly detection rate (recall): 100.00%
INFO: Accuracy: 100.00%
INFO: Precision: 100.00%
INFO: F1 Score: 100.00%

Confusion Matrix:
                     Predicted Normal    Predicted Anomaly
Actual Normal                    899                     0
Actual Anomaly                     0                 54900
```

### **What to Look For:**

 **Good training:**
- Validation loss decreases over epochs
- Training and validation loss converge

 **Good separation:**
- Good seperation between Normal and Anomaly samples in `reconstruction_error_histogram.png`
- Low false positive rate
- High recall (anomaly detection rate)

 **Warning signs:**
- Large gap between train and validation loss (overfitting)
- `mean_test_normal` much higher than `mean_train` (poor generalization)
- High false positive rate or low recall (need threshold tuning)


## **6.7 Threshold Performance CSV**

The most important output for deployment is the threshold performance CSV:

**Example: `threshold_performance.csv`**

```csv
k_value,threshold,accuracy,precision,recall,f1_score,false_positive_rate,true_positives,true_negatives,false_positives,false_negatives
0.0,1.662,98.65,98.65,100.0,99.32,83.54,54900,148,751,0
1.0,3.631,99.71,99.70,100.0,99.85,18.13,54900,736,163,0
2.0,5.599,99.92,99.92,100.0,99.96,4.78,54900,856,43,0
3.0,7.567,99.98,99.98,100.0,99.99,1.11,54900,889,10,0
4.0,9.535,100.0,100.0,100.0,100.0,0.22,54900,897,2,0
4.5,10.519,100.0,100.0,100.0,100.0,0.0,54900,899,0,0
```

**How to use:**
1. Identify your priority (recall, precision, or F1)
2. Find the row with acceptable performance
3. Note the threshold value
4. Use this threshold in your application code


## **6.8 Visualizing Results**

ModelMaker generates histogram plots showing reconstruction error distributions:

**Files generated:**
- `reconstruction_error_histogram.png` (linear scale)
- `reconstruction_error_histogram_log.png` (log scale)

**What to look for:**
- **Good separation:** Normal and anomaly distributions don't overlap
- **Clear threshold:** Green line (threshold) separates the two distributions
- **Low overlap:** Minimal anomalies below threshold, minimal normals above threshold

**Example interpretation:**
```
Normal samples:   Clustered at low error (mean ~2)
Anomaly samples:  High error (mean ~142)
Threshold:        10.5 (clearly separates them)
```

# **7. Evaluation Metrics for Anomaly Detection**

Evaluating anomaly detection models requires different metrics than classification tasks. This section explains which metrics matter, why, and how to interpret them.

## **7.1 Why Accuracy is Misleading**

Accuracy is the most common metric in classification, but it can be **highly misleading** for anomaly detection due to class imbalance.

### **Example Scenario:**

**Test set composition:**
- 950 normal samples (95%)
- 50 anomaly samples (5%)

**Naive model:** Always predicts "Normal" for every sample

**Results:**
```
Accuracy = (950 + 0) / 1000 = 95% ✓
```

**Looks great!** But the model has:
- **0% anomaly detection rate** (missed all 50 anomalies)
- **Complete failure** at its primary task

### **The Problem:**

In anomaly detection, the dataset is **intentionally imbalanced**—normal samples far outnumber anomalies (reflecting real-world conditions). A model can achieve high accuracy by simply predicting "normal" for everything, while being useless for detecting actual faults.

**Conclusion:** Accuracy alone is insufficient. We need metrics that specifically measure **how well the model detects anomalies**.



## **7.2 Key Metrics for Anomaly Detection**

### **Confusion Matrix**

First, understand the four possible outcomes:

```
                    Predicted Normal    Predicted Anomaly

Actual Normal            TN                    FP
                    (True Negative)      (False Positive)


Actual Anomaly           FN                    TP
                    (False Negative)     (True Positive)
```

**Definitions:**
- **True Positive (TP):** Anomaly correctly detected
- **True Negative (TN):** Normal correctly identified
- **False Positive (FP):** Normal incorrectly flagged as anomaly (false alarm)
- **False Negative (FN):** Anomaly missed (not detected)



### **Example Confusion Matrix:**

From motor fault detection results:

```
                    Predicted Normal    Predicted Anomaly
Actual Normal            899                   0
Actual Anomaly             0                54900
```

**Interpretation:**
- 899 normal samples correctly identified
- 54,900 anomalies correctly detected
- 0 false positives (no false alarms)
- 0 false negatives (no missed anomalies)
- **Perfect detection!**



## **7.3 Metric Definitions and Formulas**

### **7.3.1 Recall (Sensitivity, True Positive Rate, Detection Rate)**

```
Recall = TP / (TP + FN)
```

**What it measures:** What percentage of actual anomalies did we detect?

**Example:**
```
Recall = 54900 / (54900 + 0) = 100%
```

**Interpretation:** We detected all anomalies (no missed faults).

**When to prioritize:** Safety-critical applications where missing an anomaly is costly (e.g., aircraft engine monitoring, medical devices).



### **7.3.2 Precision (Positive Predictive Value)**

```
Precision = TP / (TP + FP)
```

**What it measures:** Of all samples we flagged as anomalies, what percentage were actually anomalous?

**Example:**
```
Precision = 54900 / (54900 + 0) = 100%
```

**Interpretation:** All our anomaly predictions were correct (no false alarms).

**When to prioritize:** Cost-sensitive applications where false alarms are expensive (e.g., unnecessary maintenance, production line shutdowns).



### **7.3.3 F1 Score**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**What it measures:** Harmonic mean of precision and recall (balanced metric).

**Example:**
```
F1 = 2 × (1.0 × 1.0) / (1.0 + 1.0) = 100%
```

**Interpretation:** Perfect balance between catching anomalies and avoiding false alarms.

**When to prioritize:** Most applications where you want a balance between detection and false alarm rate.



### **7.3.4 False Positive Rate (FPR)**

```
FPR = FP / (FP + TN)
```

**What it measures:** What percentage of normal samples were incorrectly flagged as anomalies?

**Example:**
```
FPR = 0 / (0 + 899) = 0%
```

**Interpretation:** No normal samples were incorrectly flagged.

**When to prioritize:** Production environments where false alarms disrupt operations or require manual investigation.



### **7.3.5 Accuracy**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**What it measures:** Overall correctness across all samples.

**Example:**
```
Accuracy = (54900 + 899) / (54900 + 899 + 0 + 0) = 100%
```

**When to use:** As a supplementary metric alongside recall/precision, **not as the primary metric**.



## **7.4 Metric Summary Table**

| Metric | Formula | What It Measures | Ideal Value | Priority For |
|--------|---------|------------------|-------------|--------------|
| **Recall** | TP/(TP+FN) | % of anomalies detected | **High** | Safety-critical (catch all faults) |
| **Precision** | TP/(TP+FP) | % of predictions that are correct | **High** | Cost-sensitive (avoid false alarms) |
| **F1 Score** | 2×(P×R)/(P+R) | Balance of precision & recall | **High** | Balanced applications |
| **False Positive Rate** | FP/(FP+TN) | % of normals incorrectly flagged | **Low** | Production deployment |
| **Accuracy** | (TP+TN)/Total | Overall correctness | High | Supplementary only |


## **7.5 Choosing Metrics Based on Application**

### **Safety-Critical Systems**
**Priority: Maximize Recall**

**Examples:** Aircraft engine monitoring, medical devices, nuclear plant sensors

**Goal:** Catch every anomaly, even if it means some false alarms

**Threshold strategy:** Use **lower threshold** (e.g., k=1 or k=2)
- High recall (e.g., 99-100%)
- Accept higher false positive rate (e.g., 5-10%)

**Trade-off:** More false alarms, but no missed critical failures


### **Cost-Sensitive Systems**
**Priority: Maximize Precision**

**Examples:** Predictive maintenance (avoid unnecessary servicing), manufacturing quality control (minimize production interruptions)

**Goal:** Only flag true anomalies, minimize false alarms

**Threshold strategy:** Use **higher threshold** (e.g., k=3 or k=4)
- High precision (e.g., 95-100%)
- Accept lower recall (e.g., 85-90%)

**Trade-off:** May miss some subtle anomalies, but avoid costly false alarms


### **Balanced Systems**
**Priority: Maximize F1 Score**

**Examples:** General industrial monitoring, IoT sensor networks, condition monitoring

**Goal:** Good detection with reasonable false alarm rate

**Threshold strategy:** Use **threshold with best F1 score** (ModelMaker recommends this)
- Balanced precision and recall (e.g., both ~90-95%)
- Moderate false positive rate (e.g., 2-5%)

**Trade-off:** Optimal balance for most applications

## **7.6 What If You Have No Anomaly Data?**

**Scenario:** You have collected normal data but no anomalies yet.

### **What You CAN Do:**

 **Train the model**
- Training only requires normal data
- Model learns to reconstruct normal patterns

 **Evaluate on normal test data**
- Calculate reconstruction error distribution
- Measure false positive rate (normal samples incorrectly flagged)

 **Set a default threshold**
- Use standard statistical threshold: `mean_train + 3 × std_train`
- This is a reasonable starting point (covers ~99.7% of normal data if Gaussian)

### **What You CANNOT Do:**

 **Measure anomaly detection performance**
- Cannot calculate recall (no anomalies to detect)
- Cannot calculate precision (no true positives)
- Cannot calculate F1 score

 **Validate that the model actually detects faults**
- You can only verify it doesn't produce false alarm

---