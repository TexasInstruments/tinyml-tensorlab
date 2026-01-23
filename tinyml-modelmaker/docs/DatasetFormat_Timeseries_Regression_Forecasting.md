# Dataset & Datafile format: Timeseries Regression and Forecasting

This document describes the dataset format requirements for **Timeseries Regression** and **Timeseries Forecasting** tasks in the TI Tiny ML ModelMaker ecosystem.

**Key Difference from Classification**: Unlike classification tasks that use a `classes/` folder structure, regression and forecasting tasks use a flat `files/` directory structure with targets stored within the data files themselves.

---

## Table of Contents
1. [Timeseries Regression](#1-timeseries-regression)
2. [Timeseries Forecasting](#2-timeseries-forecasting)
3. [Common Requirements](#3-common-requirements)
4. [Dataset Splitting](#4-dataset-splitting)
5. [Examples](#5-examples)

---

## 1. Timeseries Regression

### 1.1 Dataset Format

The prepared dataset should have the following structure:

<pre>
dataset_name/
     |
     |--files/                          # Data directory (ALWAYS named "files")
     |     |-- datafile1.csv            # Data file 1
     |     |-- datafile2.csv            # Data file 2
     |     |-- datafile3.npy            # Can mix formats (.csv, .npy, .pkl, .txt)
     |     |-- datafileN.csv            # As many files as needed
     |
     |--annotations/                    # Annotation directory
           |--file_list.txt             # List of all files (OPTIONAL - auto-generated)
           |--instances_train_list.txt  # Training set files (REQUIRED)
           |--instances_val_list.txt    # Validation set files (REQUIRED)
           |--instances_test_list.txt   # Test set files (OPTIONAL)
</pre>

**Key Points:**
- The data directory **MUST be named `files/`** (not `classes/` or any other name)
- All data files are placed in the flat `files/` directory (no subdirectories)
- The `annotations/` folder with train/val split files is **REQUIRED** for regression
- Files can have any naming convention (e.g., `class_0_train.csv`, `sensor_data_1.csv`)

### 1.2 Datafile Format

**Supported file types:** `.csv`, `.txt`, `.npy`, `.pkl`

#### Column Structure

Regression data files must follow this column structure:

<pre>
[Time Column] | [Feature 1] | [Feature 2] | ... | [Feature N] | [Target Value]
   (skipped)  |   (input)   |   (input)   | ... |  (input)    | (LAST COLUMN)
</pre>

**CRITICAL**: The **target value MUST be in the LAST COLUMN** of the file.

#### Example: Torque Measurement Dataset

<pre>
u_q,coolant,stator_winding,u_d,stator_tooth,motor_speed,i_d,i_q,pm,stator_yoke,Target
-0.4506815,-0.3500546,19.086669,18.80517,18.29321,0.00286556,0.00441913,0.00032810,24.554214,18.316547,0.1871007
-0.3257369,-0.3058030,19.092390,18.81857,18.29480,0.00025678,0.00060587,-0.0007853,24.538078,18.314954,0.2454174
-0.4408640,-0.3725026,19.089380,18.82876,18.29409,0.00235497,0.00128958,0.00038646,24.544692,18.326307,0.1766153
</pre>

- **Input Features**: Columns 1-10 (u_q, coolant, stator_winding, etc.)
- **Target**: Last column named `Target` (torque value to predict)
- **Time Column**: Any column with "time" in the name will be automatically dropped

### 1.3 Configuration Example

In your `config.yaml`:

```yaml
common:
  target_module: timeseries
  task_type: generic_timeseries_regression
  target_device: F28P55

dataset:
  dataset_name: torque_measurement
  input_data_path: /path/to/dataset.zip  # or directory path
  data_dir: files                         # Must be "files"
  annotation_dir: annotations
  enable: True

data_processing_feature_extraction:
  data_proc_transforms: [SimpleWindow]    # MANDATORY for regression
  variables: 10                           # Number of input feature columns
  frame_size: 128                         # Window size for time series
  stride_size: 0.25                       # Overlap between windows (0.0 - 1.0)

training:
  model_name: TimeSeries_Generic_1k_t     # Regression model from ModelZoo
  epochs: 100
  batch_size: 32
```

### 1.4 Key Requirements

1. **SimpleWindow Transform**: MANDATORY for regression tasks
2. **Target in Last Column**: The regression target value must be in the last column of each file
3. **Annotation Files**: REQUIRED (train/val split lists)
4. **Data Directory Name**: Must be `files/`
5. **Loss Function**: Mean Squared Error (MSE) is used automatically
6. **Evaluation Metrics**: MSE and R² Score

### 1.5 Target Processing

- Target values are continuous numerical values (e.g., torque, temperature, pressure)
- Target values are averaged across window frames during preprocessing
- Final output shape: `(num_samples, 1)`

---

## 2. Timeseries Forecasting

### 2.1 Dataset Format

The dataset structure is **identical to regression**:

<pre>
dataset_name/
     |
     |--files/                          # Data directory (ALWAYS named "files")
     |     |-- sequence1.csv            # Time series sequence 1
     |     |-- sequence2.csv            # Time series sequence 2
     |     |-- sequenceN.csv            # As many sequences as needed
     |
     |--annotations/                    # Annotation directory
           |--file_list.txt             # List of all files (OPTIONAL)
           |--instances_train_list.txt  # Training sequences (REQUIRED)
           |--instances_val_list.txt    # Validation sequences (REQUIRED)
           |--instances_test_list.txt   # Test sequences (OPTIONAL)
</pre>

**Key Difference from Regression**: Instead of predicting a single target value, forecasting predicts **future timesteps** of one or more variables.

### 2.2 Datafile Format

**Supported file types:** `.csv`, `.txt`, `.npy`, `.pkl`

#### Column Structure

Forecasting data files contain time series sequences with all variables as columns:

<pre>
[Time Column] | [Variable 1] | [Variable 2] | ... | [Variable N]
  (skipped)   |   (data)     |   (data)     | ... |   (data)
</pre>

**KEY CONCEPT**: Unlike regression, ALL variables can be both inputs and targets. You specify which variables to predict in the configuration.

#### Example 1: PMSM Rotor Temperature Forecasting

<pre>
ambient,coolant,u_d,u_q,i_a,pm
19.85062,18.81536,1.49915,0.03244,2.28195,22.93607
19.85062,18.79350,1.54285,-0.09223,2.28119,22.94180
19.85062,18.79041,1.45612,0.08181,2.28124,22.94419
19.85062,18.77453,1.47560,0.07353,2.28125,22.94615
</pre>

- **Input Features**: Columns 0 and 5 (`ambient` and `pm`)
- **Target Variable**: Column 5 (`pm` - permanent magnet temperature)
- **Goal**: Predict future `pm` temperature based on historical `ambient` and `pm` values

#### Example 2: HVAC Indoor Temperature Forecasting

<pre>
compressorFrequency,outdoorTemperature,indoorTemperature
44.22409,26.77617,27.09946
44.90539,26.43689,28.08932
43.40728,26.63597,27.96732
43.66071,27.09892,27.26332
</pre>

- **Input Features**: All 3 columns
- **Target Variable**: `indoorTemperature` (column 2)
- **Goal**: Predict future indoor temperature based on historical compressor frequency, outdoor temp, and indoor temp

### 2.3 Configuration Example

In your `config.yaml`:

```yaml
common:
  target_module: timeseries
  task_type: generic_timeseries_forecasting
  target_device: F28P55

dataset:
  dataset_name: pmsm_rotor_temp_prediction_dataset
  input_data_path: /path/to/dataset.zip
  data_dir: files
  annotation_dir: annotations
  enable: True

data_processing_feature_extraction:
  data_proc_transforms: [SimpleWindow]    # MANDATORY for forecasting
  frame_size: 3                           # Historical window (lookback period)
  stride_size: 0.4                        # Overlap between windows
  forecast_horizon: 1                     # How many timesteps ahead to predict

  # Option 1: Specify by column indices (after time column removal)
  variables: [0, 5]                       # Input feature columns
  target_variables: [5]                   # Columns to forecast

  # Option 2: Specify by column names (requires CSV header)
  # variables: ['ambient', 'pm']
  # target_variables: ['pm']

training:
  model_name: TimeSeries_Generic_Forecasting_LSTM8
  epochs: 100
  batch_size: 32
  output_int: False                       # MUST be False for forecasting
```

### 2.4 Key Requirements

1. **SimpleWindow Transform**: MANDATORY for forecasting
2. **Forecast Horizon**: Must specify how many timesteps ahead to predict
3. **Target Variables**: Must explicitly specify which variables to forecast
4. **Variables**: Specify which columns to use as input features
5. **Annotation Files**: REQUIRED (train/val split lists)
6. **Data Directory Name**: Must be `files/`
7. **Loss Function**: Huber Loss (robust to outliers)
8. **Evaluation Metrics**: SMAPE (Symmetric Mean Absolute Percentage Error) and R² Score
9. **Feature Extraction NOT Supported**: Forecasting works on raw time series only (no FFT, wavelets, etc.)
10. **output_int Must Be False**: Forecasting outputs are continuous values

### 2.5 Windowing Approach

Forecasting uses a sliding window approach:

- **Input Window**: Historical data of size `frame_size` timesteps
- **Output Window**: Future data of size `forecast_horizon` timesteps

**Example with frame_size=3, forecast_horizon=1:**
```
Input:  [t0, t1, t2]  →  Output: [t3]
Input:  [t1, t2, t3]  →  Output: [t4]
Input:  [t2, t3, t4]  →  Output: [t5]
```

### 2.6 Target Variable Specification Formats

You have three options to specify target variables:

**Option 1 - Integer Index:**
```yaml
target_variables: [5]    # Predict column at index 5 (after time removal)
```

**Option 2 - Column Name:**
```yaml
target_variables: ['pm']    # Predict column named 'pm'
```

**Option 3 - Multiple Targets:**
```yaml
target_variables: [5, 7, 11]    # Predict multiple columns simultaneously
# OR
target_variables: ['pm', 'coolant', 'ambient']
```

---

## 3. Common Requirements

### 3.1 Supported File Formats

Both regression and forecasting support:
- **CSV files** (`.csv`): Most common, human-readable
- **Text files** (`.txt`): Same format as CSV
- **NumPy arrays** (`.npy`): Binary format, faster loading
- **Pickle files** (`.pkl`): Python serialized pandas DataFrames

### 3.2 Header and Time Column Handling

**Two accepted formats:**

#### Format A: Headerless (No Column Names)
```
2078.5
2136.2
2117.8
2077.1
```
- No header row
- No index/time column
- Each row is a measurement
- Suitable for single-variable time series

#### Format B: With Header and Time Column
```
Time,Feature1,Feature2,Feature3,Target
0.001,2078.5,45.2,18.3,0.187
0.002,2136.2,45.3,18.2,0.245
0.003,2117.8,45.1,18.4,0.176
```
- Header row with column names
- Any column containing "time" (case-insensitive) is **automatically dropped**
- Examples of dropped columns: `Time`, `TIME`, `Timestamp`, `Time(sec)`, `TIME (microsec)`

### 3.3 Annotation Files Format

The annotation files contain simple lists of filenames (one per line):

**instances_train_list.txt:**
```
datafile1.csv
datafile3.csv
datafile5.csv
```

**instances_val_list.txt:**
```
datafile2.csv
datafile4.csv
```

**Notes:**
- File paths are relative to the `files/` directory
- No need to include the `files/` prefix
- One filename per line
- No empty lines

### 3.4 Dataset Path Specification

In `config.yaml`, you can specify the dataset path in multiple ways:

```yaml
dataset:
  dataset_name: my_dataset
  input_data_path: <path_or_url>
```

**Supported path types:**
1. **Local directory**: `/path/to/dataset_name/`
2. **Local zip file**: `/path/to/dataset.zip`
3. **Remote URL**: `https://example.com/dataset.zip`

**Important for zip files:**
- Must contain the `files/` and `annotations/` directories immediately inside
- Don't add extra hierarchy levels (e.g., `dataset_name/files/`)
- Zip file will be automatically extracted by ModelMaker

### 3.5 Data Quality Requirements

**For both tasks:**
- All files should have compatible dimensions (same number of variables)
- Number of columns must be consistent across files (except for varying sequence lengths)
- No missing values (NaN) - will cause errors
- Time columns are automatically removed (any column with "time" in the name)
- Data must be numeric (integers or floats)

**Minimum requirements:**
- At least 2 files recommended (1 train, 1 validation)
- Sufficient sequence length for windowing:
  - For regression: At least `frame_size` timesteps per file
  - For forecasting: At least `frame_size + forecast_horizon` timesteps per file

---

## 4. Dataset Splitting

### 4.1 Pre-split vs. Auto-split

**For Regression and Forecasting:**
- If `annotations/` folder with split files exists: ModelMaker uses your pre-defined splits
- If `annotations/` folder is missing: ModelMaker auto-generates splits

**Auto-split configuration:**
```yaml
dataset:
  split_type: amongst_files    # or 'within_files'
  split_factor: [0.6, 0.3, 0.1]    # train:val:test ratio (default)
```

### 4.2 Amongst Files (Default)

Files are split into different sets:

- Say you have **10 files** (each with 100 timesteps)
- Split into: **6 train files**, **3 val files**, **1 test file**
- Each file retains all 100 timesteps

**Use case**: When each file represents a distinct experiment/session

### 4.3 Within Files

Each file is split into train/val/test portions:

- Say you have **10 files** (each with 100 timesteps)
- All **10 files** appear in train, val, AND test sets
- Train files: first **60 timesteps** of each
- Val files: next **30 timesteps** of each
- Test files: last **10 timesteps** of each

**Use case**: When files contain long continuous sequences that can be safely split

---

## 5. Examples

### 5.1 Regression Example: Motor Torque Prediction

**Use Case**: Predict motor output torque from sensor readings

**Dataset Structure:**
```
torque_measurement/
├── files/
│   ├── class_0_train.csv    # 10 input features + 1 target
│   ├── class_1_test.csv
│   ├── class_2_val.csv
│   └── ... (10 files total)
└── annotations/
    ├── instances_train_list.txt
    └── instances_val_list.txt
```

**Data File Example (class_0_train.csv):**
```csv
u_q,coolant,stator_winding,u_d,stator_tooth,motor_speed,i_d,i_q,pm,stator_yoke,Target
-0.450,18.805,19.086,-0.350,18.293,0.002,0.004,0.0003,24.554,18.316,0.187
-0.325,18.818,19.092,-0.305,18.294,0.0002,0.0006,-0.0007,24.538,18.314,0.245
```

**Config Highlights:**
```yaml
task_type: generic_timeseries_regression
data_processing_feature_extraction:
  variables: 10              # 10 input features
  frame_size: 128            # Use 128 timesteps per prediction
  stride_size: 0.25
training:
  model_name: TimeSeries_Generic_1k_t
```

**Results:**
- R² Score: 0.994 (float32), 0.963 (quantized INT8)
- Model size: ~1K parameters
- Use case: Real-time motor control, predictive maintenance

---

### 5.2 Forecasting Example 1: PMSM Rotor Temperature

**Use Case**: Predict permanent magnet temperature in electric motor

**Dataset Structure:**
```
pmsm_rotor_temp_prediction_dataset/
├── files/
│   ├── profile_id_10.csv    # 6 variables: ambient, coolant, u_d, u_q, i_a, pm
│   ├── profile_id_11.csv
│   ├── profile_id_12.csv
│   └── ... (80 files total)
└── annotations/
    ├── instances_train_list.txt
    └── instances_val_list.txt
```

**Data File Example (profile_id_10.csv):**
```csv
ambient,coolant,u_d,u_q,i_a,pm
19.850,18.815,1.499,0.032,2.281,22.936
19.850,18.793,1.542,-0.092,2.281,22.941
19.850,18.790,1.456,0.081,2.281,22.944
```

**Config Highlights:**
```yaml
task_type: generic_timeseries_forecasting
data_processing_feature_extraction:
  variables: [0, 5]          # Use 'ambient' and 'pm' as inputs
  target_variables: [5]       # Predict 'pm' (permanent magnet temp)
  frame_size: 3               # Use 3 past timesteps
  forecast_horizon: 1         # Predict 1 timestep ahead
  stride_size: 0.4
training:
  model_name: TimeSeries_Generic_Forecasting_LSTM8
  output_int: False           # MUST be False for forecasting
```

**Results:**
- SMAPE: 0.13% (float32), 1.90% (quantized INT8)
- Lookback: 3 timesteps
- Prediction: 1 timestep ahead
- Use case: Thermal management, motor protection

---

### 5.3 Forecasting Example 2: HVAC Indoor Temperature

**Use Case**: Predict indoor temperature for HVAC control

**Dataset Structure:**
```
hvac_indoor_temp_forecast/
├── files/
│   ├── aug_file1_perm1.csv   # 3 variables: compressorFrequency, outdoorTemp, indoorTemp
│   ├── aug_file1_perm2.csv
│   ├── aug_file2_perm1.csv
│   └── ... (multiple files)
└── annotations/
    ├── instances_train_list.txt
    └── instances_val_list.txt
```

**Data File Example:**
```csv
compressorFrequency,outdoorTemperature,indoorTemperature
44.224,26.776,27.099
44.905,26.436,28.089
43.407,26.635,27.967
43.660,27.098,27.263
```

**Config Highlights:**
```yaml
task_type: generic_timeseries_forecasting
data_processing_feature_extraction:
  variables: [0, 1, 2]                    # All 3 variables as input
  target_variables: ['indoorTemperature']  # Predict indoor temp
  frame_size: 5                            # Use 5 past timesteps
  forecast_horizon: 3                      # Predict 3 timesteps ahead
training:
  model_name: TimeSeries_Generic_Forecasting_LSTM8
```

**Use case**: Energy-efficient HVAC control, building automation

---

## Key Differences Summary

| Aspect | Classification | Regression | Forecasting |
|--------|---------------|------------|-------------|
| **Folder Structure** | `classes/{class1}/`, `classes/{class2}/` | `files/{file1}`, `files/{file2}` | `files/{sequence1}`, `files/{sequence2}` |
| **Target Location** | Folder name | Last column of file | Specified columns + future timesteps |
| **Target Type** | Categorical labels | Single continuous value | Time series sequence |
| **Loss Function** | CrossEntropyLoss | MSELoss | HuberLoss |
| **Metrics** | Accuracy, F1-score | MSE, R² Score | SMAPE, R² Score |
| **Annotation Files** | Optional (auto-gen) | Required | Required |
| **Data Directory** | `data/` or `classes/` | `files/` | `files/` |
| **Feature Extraction** | Supported (FFT, wavelets) | Supported | NOT Supported (raw only) |
| **output_int** | Can be True | Can be True | MUST be False |
| **SimpleWindow** | Optional | MANDATORY | MANDATORY |
| **Forecast Horizon** | N/A | N/A | Required parameter |

---

## Reference Examples

### Available Example Datasets:

1. **Torque Measurement (Regression)**
   - Path: `tinyml-modelmaker/data/projects/torque_measurement/dataset/`
   - Example config: `tinyml-modelmaker/examples/torque_measurement_regression/config.yaml`
   - Variables: 10 input features, 1 target (torque)
   - Use case: Motor control

2. **PMSM Rotor Temperature (Forecasting)**
   - Path: `tinyml-modelmaker/data/projects/pmsm_rotor_temp_prediction_dataset/dataset/`
   - Variables: 6 features (ambient, coolant, voltages, current, PM temperature)
   - Forecast: 1 timestep ahead
   - Use case: Thermal management

3. **HVAC Indoor Temperature (Forecasting)**
   - Path: `tinyml-modelmaker/data/projects/hvac_indoor_temp_forecast/dataset/`
   - Variables: 3 features (compressor frequency, outdoor temp, indoor temp)
   - Forecast: Indoor temperature prediction
   - Use case: Building automation

### Code References:

- **Dataset Classes**: `tinyml-tinyverse/tinyml_tinyverse/common/datasets/timeseries_dataset.py`
  - `GenericTSDatasetReg` (Lines 1035-1189)
  - `GenericTSDatasetForecasting` (Lines 1376-1620)
- **Training Scripts**:
  - Regression: `tinyml-tinyverse/tinyml_tinyverse/references/timeseries_regression/train.py`
  - Forecasting: `tinyml-tinyverse/tinyml_tinyverse/references/timeseries_forecasting/train.py`

---

## Troubleshooting

### Common Issues:

1. **"Target column not found" error**
   - **Regression**: Ensure the target is in the LAST column of your CSV
   - **Forecasting**: Check that target_variables indices/names are correct

2. **"Insufficient sequence length" error**
   - Ensure files have at least `frame_size` (+ `forecast_horizon` for forecasting) timesteps

3. **"Annotation files missing" error**
   - For regression/forecasting, you must provide `instances_train_list.txt` and `instances_val_list.txt`
   - Or let ModelMaker auto-generate by not including the annotations folder

4. **"Data dimension mismatch" error**
   - Verify all CSV files have the same number of columns
   - Check that `variables` parameter matches your actual column count

5. **"Time column not found but data seems off" error**
   - Remember: ANY column with "time" (case-insensitive) is auto-dropped
   - Check if you accidentally named a feature column with "time" in it

---

## Best Practices

1. **File Organization**: Use descriptive filenames (e.g., `motor_test_1.csv`, `sensor_data_normal_1.csv`)
2. **Data Quality**: Remove outliers and NaN values before dataset preparation
3. **Annotation Files**: Manually create splits for reproducible experiments
4. **Column Names**: Use clear, descriptive header names in CSV files
5. **Avoid "time" in Feature Names**: Don't name features with "time" unless it's the time index
6. **Test Your Dataset**: Try training with a small subset first to validate format
7. **Consistent Units**: Ensure all measurements use consistent units across files
8. **Normalization**: Data will be normalized during training, but extreme outliers may affect learning

---

For more information, refer to:
- [Classification Dataset Format](./DatasetFormat_Timeseries_Classification.md)
- [ModelMaker Examples](../examples/)
- [TinyVerse Documentation](../../tinyml-tinyverse/README.md)
