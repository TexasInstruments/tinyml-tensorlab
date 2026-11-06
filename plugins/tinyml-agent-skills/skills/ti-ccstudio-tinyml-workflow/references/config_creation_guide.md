# Config File Creation Guide

## Prerequisites

Do not ask the user for a tinyml-tensorlab path. Session Setup (`SKILL.md`) already provides `PROJECT_PATH` (where every config file is saved), `TINYML_MODELZOO_PATH` and `MODELZOO_INSTALL_MODE` (`repo` or `wheel`). Reference example configs live in the skill's `references/examples/` (`$EXAMPLES_DIR`) and are read-only.

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
{PROJECT_PATH}/config.yaml
```

`PROJECT_PATH` is the imported project root (Step 3). Never write configs into tinyml-modelzoo or the bundled `references/examples/`. Where `{task_name}` is a slug (lowercase, underscores, no spaces) derived from the application name.

**Examples:**
- Motor fault detection → `motor_fault_detection`
- Generic timeseries classification → `generic_ts_classification`
- Custom anomaly detection → `custom_anomaly`

**Before saving, verify the directory exists:**

```bash
mkdir -p "{PROJECT_PATH}"
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
| SimpleWindow | Classification, Regression, Forecasting, Anomaly detection | Forecasting: required; others: when each file is one long recording | `frame_size`, `stride_size` (a FRACTION of `frame_size`: 1.0 = no overlap, 0.5 = 50% overlap; the default 0.01 is a 1-row step for large frames, so always set it) |
| Downsample | Timeseries tasks | Optional | `sampling_rate`, `new_sr` (`new_sr` <= `sampling_rate`, integer ratio) |

`data_proc_transforms` only knows `SimpleWindow` and `Downsample`, spelled exactly like that; any other entry (`DownSample`, `Normalize`, ...) is silently ignored. Value scaling is the separate `scale` key; noise and other augmenters belong in `training.augment_config`, not here. Rows per file: each sample needs `frame_size x num_frame_concat` rows (see `FE_and_Data_Processing_Transforms/selection-rules.md`).

### Feature Extraction Presets

Presets are task-specific and variable-count dependent. Format: `Generic_{input_size}Input_{transform}_{output_size}Feature_{frames}Frame`

Example: `Generic_1024Input_FFTBIN_64Feature_8Frame` (frame_size 1024, 64 features per frame, 8 frames, so 1024 x 8 = 8192 rows per file). Read the preset's values from modelmaker's `constants.py` instead of trusting the name, and let the preset supply `frame_size`, `feature_size_per_frame` and `num_frame_concat`.

### Feature Extraction Example

```yaml
data_processing_feature_extraction:
  data_proc_transforms: [SimpleWindow]
  stride_size: 0.5                      # frame_size (1024) and num_frame_concat (8) come from the preset
  feature_extraction_name: Generic_1024Input_FFTBIN_64Feature_8Frame
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
  train_output_path: /abs/project/run/20261009-153012/ds_cnn/training   # set by the skill in Step 13; MUST be inside training:
```

**STRICT:** `train_output_path` is valid only inside `training:`, never at the config root. Same for `compile_output_path`, which is valid only inside `compilation:`. When the skill runs training it sets both to `$PROJECT_PATH/run/<yyyymmdd-hhmmss>/<model_name>/training` and `.../compilation` (the modelzoo's own run layout), with `<model_name>` taken from `training.model_name`; do not hand-write them.

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
| `compile_output_path` | string | run directory | Custom artifact location. Only valid inside `compilation:`, never at the root |
| `keep_libc_files` | boolean | false | Advanced integration |

### Compilation Section Example

```yaml
compilation:
  enable: true
  compile_preset_name: default_preset
  compile_output_path: /abs/project/run/20261009-153012/ds_cnn/compilation
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
  data_proc_transforms: [SimpleWindow]
  stride_size: 0.5                      # frame_size (512) and num_frame_concat (8) come from the preset
  feature_extraction_name: Generic_512Input_FFTBIN_32Feature_8Frame

training:
  enable: true
  model_name: ds_cnn
  batch_size: 32
  training_epochs: 100
  num_gpus: 1
  learning_rate: 0.001
  train_output_path: /abs/project/run/20261009-153012/ds_cnn/training

testing:
  enable: true
  skip_train: false
  device_inference: false

compilation:
  enable: true
  compile_preset_name: default_preset
  compile_output_path: /abs/project/run/20261009-153012/ds_cnn/compilation
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

1. ✓ Path `{PROJECT_PATH}` exists
2. ✓ All required fields present (common, dataset if enable=true, training if enable=true)
3. ✓ task_type matches valid task type list
4. ✓ target_device matches valid device list
5. ✓ input_data_path is accessible and formatted correctly
6. ✓ model_name (if training) exists in modelzoo
7. ✓ split_factor sums to 1.0 (if specified)
8. ✓ No conflicting settings (e.g., skip_train=true but model_path empty)
9. ✓ `train_output_path` is inside `training:` and `compile_output_path` is inside `compilation:`; neither appears at the config root
