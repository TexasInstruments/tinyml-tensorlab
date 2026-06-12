# Config File Creation Guide

## Prerequisites

Before creating any config file, **ALWAYS confirm with the user the path to their tinyML tensorlab installation.**

```
Example confirmation:
"What's the full path to your tinyML tensorlab directory? 
(e.g., /home/username/tinyml-tensorlab or /opt/tinyml-tensorlab)"
```

Store this path as `TINYML_BASE_PATH` for all subsequent operations.

---

## Config File Structure

Config files follow YAML format with these main sections (in order):

```yaml
common:
  # Task type, device, module, run name

dataset:
  # Data loading and splitting configuration

data_processing_feature_extraction:
  # Data transforms and feature engineering

training:
  # Model training parameters

testing:
  # Testing and validation configuration

compilation:
  # Compilation settings for device deployment
```

---

## File Path Requirements

**CRITICAL: Config files MUST be saved at:**

```
{TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/config.yaml
```

Where `{task_name}` is a slug (lowercase, underscores, no spaces) derived from the application name.

**Examples:**
- Motor fault detection → `motor_fault_detection`
- Generic timeseries classification → `generic_ts_classification`
- Custom anomaly detection → `custom_anomaly`

**Before saving, verify the directory exists:**

```bash
mkdir -p "{TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}"
```

---

## Common Section Rules

### Required Fields

| Field | Type | Valid Values | Example |
|-------|------|--------------|---------|
| `task_type` | string | See Task Types table below | `motor_fault` |
| `target_device` | string | See Device List below | `F28P55` |

### Optional Fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `target_module` | string | Auto-inferred from task_type | 'timeseries' or 'vision' |
| `run_name` | string | '{date-time}/{model_name}' | Supports `{date-time}` and `{model_name}` placeholders |

### Task Types

**Specific pre-built tasks** (use exactly as shown):
- `motor_fault`
- `ecg_classification`
- `arc_fault`
- `blower_imbalance`
- `pir_detection`

**Generic tasks** (choose based on data type):
- `generic_timeseries_classification`
- `generic_timeseries_regression`
- `generic_timeseries_forecasting`
- `generic_timeseries_anomalydetection`
- `image_classification`

### Device List

| Device | Variant | Type |
|--------|---------|------|
| TMS320F280013 | F280013 | C2000 |
| TMS320F280015 | F280015 | C2000 |
| TMS320F28003 | F28003 | C2000 |
| TMS320F28004 | F28004 | C2000 |
| TMS320F2837 | F2837 | C2000 |
| TMS320F28P55 | F28P55 | C2000 |
| TMS320F28P65 | F28P65 | C2000 |
| TMS320F29H85 | F29H85 | C2000 |
| TMS320F29P58 | F29P58 | C2000 |
| TMS320F29P32 | F29P32 | C2000 |
| MSPM0G3507 | MSPM0G3507 | MSPM0 |
| MSPM0G3519 | MSPM0G3519 | MSPM0 |
| MSPM0G5187 | MSPM0G5187 | MSPM0 |
| MSPM33C32 | MSPM33C32 | MSPM33 |
| MSPM33C34 | MSPM33C34 | MSPM33 |
| AM1335 | AM13E2 | Sitara |
| AM263 | AM263 | Sitara |
| AM263P | AM263P | Sitara |
| AM261 | AM261 | Sitara |
| CC2755 | CC2755 | SimpleLink |
| CC1352 | CC1352 | SimpleLink |
| CC1354 | CC1354 | SimpleLink |
| CC35X1 | CC35X1 | SimpleLink |

### Common Section Example

```yaml
common:
  task_type: motor_fault
  target_device: F28P55
  target_module: timeseries
  run_name: "motor_fault_{date-time}/{model_name}"
```

---

## Dataset Section Rules

### Required Fields

| Field | Type | Format | Notes |
|-------|------|--------|-------|
| `enable` | boolean | true/false | Must be true to load data, false for BYOM |
| `dataset_name` | string | slug | Used for output directory naming |
| `input_data_path` | string | path/URL/zip | Local path, HTTP URL, or .zip file |

### Optional Fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `split_type` | string | From params.py | 'amongst_files' or 'within_files' |
| `split_factor` | array | From params.py | [train, val, test] summing to 1.0 |

### Data Format Rules

Data must follow BYOD (Bring Your Own Data) format specifications:

**Classification:** 
```
dataset/
├── classes/
│   ├── class_1/
│   │   ├── sample_1.csv
│   │   └── sample_2.csv
│   └── class_2/
│       └── sample_3.csv
└── metadata.json
```

**Regression/Forecasting:**
```
dataset/
├── files/
│   ├── signal_1.csv
│   ├── signal_2.csv
│   └── ...
├── annotations.json
└── metadata.json
```

**Anomaly Detection:**
```
dataset/
├── classes/
|   ├── Normal/
|   │   ├── normal_1.csv
|   │   └── normal_2.csv
|   ├── Anomaly/
|       ├── fault_1.csv
|       └── fault_2.csv
└── metadata.json
```

### Dataset Section Example

```yaml
dataset:
  enable: true
  dataset_name: motor_fault_training_data
  input_data_path: /home/user/datasets/motor_data.zip
  split_type: amongst_files
  split_factor: [0.7, 0.2, 0.1]
```

---

## Data Processing & Feature Extraction Section

### Purpose

Defines transforms applied to raw data and feature extraction pipeline to prepare data for model input.

### Required for All Tasks

- At minimum: transforms needed by task type (e.g., SimpleWindow for forecasting)
- Feature extraction: either named preset or custom pipeline

### Common Transforms

| Transform | Task Types | Required | Parameters |
|-----------|-----------|----------|------------|
| SimpleWindow | Classification, Regression, Forecasting | Often | `frame_size`, `stride_size` |
| DownSample | All | Optional | `sampling_rate`, `new_sr` |
| Normalize | All | Optional | None |
| AddNoise | All | Optional | `std_dev` |

### Feature Extraction Presets

Presets are task-specific and variable-count dependent. Format: `Generic_{input_size}Input_{transform}_{output_size}Feature_{frames}Frame`

Example: `Generic_1024Input_FFTBIN_64Feature_8Frame`

### Feature Extraction Example

```yaml
data_processing_feature_extraction:
  data_proc_transforms: [SimpleWindow, Normalize]
  frame_size: 128
  stride_size: 0.5
  feature_extraction_name: Generic_1024Input_FFTBIN_64Feature_8Frame
  num_frame_concat: 8
```

---

## Training Section Rules

### Required Fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `enable` | boolean | true | true=train new model, false=BYOM |
| `model_name` | string | required | Model architecture from modelzoo |

### Optional Fields (Use defaults if not specified)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `batch_size` | int | From params.py | Training batch size |
| `training_epochs` | int | From params.py | Number of epochs |
| `num_gpus` | int | From params.py | GPU count |
| `learning_rate` | float | From params.py | Initial LR |

### Training Section Example

```yaml
training:
  enable: true
  model_name: ds_cnn
  batch_size: 32
  training_epochs: 100
  num_gpus: 1
  learning_rate: 0.001
```

---

## Testing Section Rules

### Optional Configuration

Most users can accept defaults (enable: true).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `enable` | boolean | true | Run testing phase |
| `skip_train` | boolean | false | Skip training, test pre-existing model |
| `device_inference` | boolean | false | Test on actual hardware |
| `test_data` | string | none | Custom test dataset path |
| `model_path` | string | required if skip_train=true | Pre-trained model path |

### Testing Section Example

```yaml
testing:
  enable: true
  skip_train: false
  device_inference: false
```

---

## Compilation Section Rules

### Presets (NPU device dependent)

| Preset | Type | Use Case | Devices |
|--------|------|----------|---------|
| `default_preset` | General | Standard inference | All |
| `forced_soft_npu_preset` | CPU-only | Anomaly/Forecasting | NPU devices |
| `compress_npu_layer_data` | Compressed | Tight SRAM | NPU devices only |

### Required Field

| Field | Type | Default |
|-------|------|---------|
| `enable` | boolean | true |

### Optional Fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `compile_preset_name` | string | system default | See Presets table |
| `compile_output_path` | string | run directory | Custom artifact location |
| `keep_libc_files` | boolean | false | Advanced integration |

### Compilation Section Example

```yaml
compilation:
  enable: true
  compile_preset_name: default_preset
  compile_output_path: ~/compiled_models/motor_fault
```

---

## Complete Config Example

```yaml
common:
  task_type: motor_fault
  target_device: F28P55
  target_module: timeseries
  run_name: "motor_fault_{date-time}/{model_name}"

dataset:
  enable: true
  dataset_name: motor_training_data
  input_data_path: /home/user/motor_data.zip
  split_type: amongst_files
  split_factor: [0.7, 0.2, 0.1]

data_processing_feature_extraction:
  data_proc_transforms: [SimpleWindow, Normalize]
  frame_size: 512
  stride_size: 0.5
  feature_extraction_name: Generic_1024Input_FFTBIN_64Feature_8Frame
  num_frame_concat: 8

training:
  enable: true
  model_name: ds_cnn
  batch_size: 32
  training_epochs: 100
  num_gpus: 1
  learning_rate: 0.001

testing:
  enable: true
  skip_train: false
  device_inference: false

compilation:
  enable: true
  compile_preset_name: default_preset
  compile_output_path: ~/compiled_models
```

---

## BYOM (Bring Your Own Model) Workflow

If using pre-trained ONNX model:

```yaml
dataset:
  enable: false  # Skip data loading

training:
  enable: false  # Skip training

testing:
  enable: true
  skip_train: true
  model_path: /path/to/model.onnx

compilation:
  enable: true
  compile_preset_name: default_preset
```

---

## Validation Before Saving

Before calling `generate_complete_config_file`, verify:

1. ✓ Path `{TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}` exists
2. ✓ All required fields present (common, dataset if enable=true, training if enable=true)
3. ✓ task_type matches valid task type list
4. ✓ target_device matches valid device list
5. ✓ input_data_path is accessible and formatted correctly
6. ✓ model_name (if training) exists in modelzoo
7. ✓ split_factor sums to 1.0 (if specified)
8. ✓ No conflicting settings (e.g., skip_train=true but model_path empty)
