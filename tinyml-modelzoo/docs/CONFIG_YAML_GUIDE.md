# How to Create a ModelZoo Config YAML

Every example under [`examples/`](https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/) is driven by a single YAML file.
This guide explains the structure of that file and how to adapt an existing
example to your own dataset and target device.

- [1. Config Skeleton and 4-Step Recipe](#1-config-skeleton-and-4-step-recipe)
- [2. Feature Extraction: Preset vs. Custom](#2-feature-extraction-preset-vs-custom)
- [3. Useful Knobs](#3-useful-knobs)
- [Common Pitfalls](#common-pitfalls)

---

## 1. Config Skeleton and 4-Step Recipe

Almost every config follows the same 6-section skeleton
(based on [`generic_timeseries_classification/config.yaml`](https://github.com/TexasInstruments/tinyml-modelzoo/blob/main/examples/generic_timeseries_classification/config.yaml)):

```yaml
common:                          # WHAT task, WHICH device
  task_type: generic_timeseries_classification
  target_device: F28P55          # F28P55, MSPM0G5187, F29H85, CC2755, AM263 ...

dataset:                         # WHERE the data is
  dataset_name: my_dataset
  input_data_path: <URL | .zip | local folder>

data_processing_feature_extraction:   # HOW raw signal becomes model input
  feature_extraction_name: Generic_1024Input_FFTBIN_64Feature_8Frame  # preset
  variables: 1                   # no. of input channels

training:                        # WHICH model, HOW to train
  model_name: CLS_1k_NPU         # from ModelZoo
  batch_size: 256
  training_epochs: 20
  quantization: 2                # 0 = float, 1 = generic INT, 2 = TI-NPU INT

testing: {}                      # {} = use defaults
compilation: {}                  # {} = default_preset
```

| Section | Purpose |
|---------|---------|
| `common` | Task type, target device, (optionally) output location and run name |
| `dataset` | Dataset name and where to load it from |
| `data_processing_feature_extraction` | Windowing, transforms and feature extraction applied to raw data |
| `training` | Model architecture, training hyperparameters and quantization |
| `testing` | Evaluation of the trained model |
| `compilation` | Compilation into device-ready code with TI's Neural Network Compiler |

### 4-Step Recipe

1. **Copy the closest example.** Use the use-case folder if one matches your
   problem (e.g. [`motor_bearing_fault`](https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/motor_bearing_fault/)).
   If none does, use the generic example for your task:
   `generic_timeseries_classification`, `generic_timeseries_regression`,
   `generic_timeseries_forecasting` or `generic_timeseries_anomalydetection`.
2. **Set `common`.** Pick `task_type` and `target_device`. Many examples already
   ship device-specific variants such as `config_MSPM0.yaml`,
   `config_CC2755.yaml` or `config_CC1352.yaml`. Start from those when available.
3. **Point `dataset` at your data.** Set `input_data_path` (URL, `.zip` or local
   folder) and set `variables` to match the number of input channels in your data.
4. **Pick feature extraction and a model, then run:**

   ```bash
   ./run_tinyml_modelzoo.sh examples/<name>/config.yaml       # Linux
   run_tinyml_modelzoo.bat examples\<name>\config.yaml        # Windows
   ```

> **Tip:** Any key you leave out uses its default value, so `testing: {}` and
> `compilation: {}` are valid as written.

---

## 2. Feature Extraction: Preset vs. Custom

| Approach | When | Example |
|----------|------|---------|
| **Preset** | Classification, audio, image | `feature_extraction_name: Generic_512Input_FFTBIN_32Feature_8Frame` |
| **Custom transforms** | Fine-tuning a preset | `frame_size`, `feature_size_per_frame`, `num_frame_concat`, `feat_ext_transform: [FFT_FE, FFT_POS_HALF, DC_REMOVE, ABS, BINNING, LOG_DB, CONCAT]` |
| **Windowing only** | Regression, forecasting, anomaly detection | `data_proc_transforms: [SimpleWindow, Downsample]`, `frame_size`, `stride_size`, `sampling_rate`, `new_sr` |

### Preset example (classification)

```yaml
data_processing_feature_extraction:
  feature_extraction_name: Generic_1024Input_FFTBIN_64Feature_8Frame
  variables: 1
```

Other generic presets available for timeseries classification:

| Preset | Transform |
|--------|-----------|
| `Generic_1024Input_FFTBIN_64Feature_8Frame` | FFT, binned, 8 frames concatenated |
| `Generic_512Input_FFTBIN_32Feature_8Frame` | FFT, binned, 8 frames concatenated |
| `Generic_256Input_FFTBIN_16Feature_8Frame` | FFT, binned, 8 frames concatenated |
| `Generic_1024Input_FFT_512Feature_1Frame` | FFT, single frame |
| `Generic_512Input_FFT_256Feature_1Frame` | FFT, single frame |
| `Generic_256Input_FFT_128Feature_1Frame` | FFT, single frame |
| `Generic_512Input_RAW_512Feature_1Frame` | Raw samples, no transform |
| `Generic_256Input_RAW_256Feature_1Frame` | Raw samples, no transform |
| `Generic_128Input_RAW_128Feature_1Frame` | Raw samples, no transform |

### Custom transforms example

```yaml
data_processing_feature_extraction:
  frame_size: 1024
  feature_size_per_frame: 64
  num_frame_concat: 4
  feat_ext_transform: [ 'FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT' ]
  variables: 1
```

### Windowing-only example (anomaly detection)

```yaml
data_processing_feature_extraction:
  data_proc_transforms:
  - SimpleWindow
  - Downsample
  frame_size: 128
  sampling_rate: 100
  new_sr: 10
  variables: 1
```

### How the task type changes the config

| Task | What is different |
|------|-------------------|
| **Forecasting** | `SimpleWindow` is mandatory; add `forecast_horizon` (future timesteps to predict) and `target_variables` (column indices to predict). See [`generic_timeseries_forecasting`](https://github.com/TexasInstruments/tinyml-modelzoo/blob/main/examples/generic_timeseries_forecasting/config.yaml). |
| **Anomaly detection** | Uses `dataset.split_factor: [0.5, 0.1, 0.4]` and an `AD_*` autoencoder model. See [`generic_timeseries_anomalydetection`](https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/generic_timeseries_anomalydetection/). |
| **Regression** | Usually windowing only (`SimpleWindow`) with a `REGR_*` model. See [`generic_timeseries_regression`](https://github.com/TexasInstruments/tinyml-modelzoo/blob/main/examples/generic_timeseries_regression/config.yaml). |
| **Image** | Set `common.target_module: vision` and use a preset such as `Mnist_Default`. See [`MNIST_image_classification`](https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/MNIST_image_classification/). |

### Selecting input channels with `variables`

`variables` supports three formats (column indices are 0-indexed, counted after
time columns are removed):

```yaml
variables: 3                           # first N columns
variables: [0, 2, 4]                   # specific column indices
variables: ['accel_x', 'accel_y']      # column names (requires CSV header)
```

---

## 3. Useful Knobs

### Training

| Key | Description |
|-----|-------------|
| `early_stopping`, `early_stopping_patience` | Stop training when validation stops improving |
| `optimizer` | e.g. `adam` |
| `lr_scheduler` | e.g. `cosineannealinglr` |
| `learning_rate` | Initial learning rate |
| `num_gpus` | `0` for CPU training |

### Quantization

| Key | Description |
|-----|-------------|
| `quantization` | `0` = float, `1` = generic integer, `2` = TI-NPU integer |
| `quantization_method` | `QAT` (quantization-aware training) or `PTQ` (post-training quantization) |
| `quantization_weight_bitwidth` | `8`, `4` or `2` |
| `quantization_activation_bitwidth` | `8` |
| `auto_quantization` | Automatically select per-layer precision |

### Compilation

| Key | Description |
|-----|-------------|
| `compile_preset_name` | `default_preset`, `compress_npu_layer_data` or `forced_soft_npu_preset`. The last two apply only to NPU devices. |

See [NPU Configuration Guidelines](NPU_CONFIGURATION_GUIDELINES.md) for designing
models that run efficiently on the TI NPU.

### Output location

| Key | Description |
|-----|-------------|
| `common.projects_path` | Where artifacts are written. Default: `./data/projects/<project_name>/`, relative to the directory you run from |
| `common.run_name` | Run sub-folder name, e.g. `'{date-time}/{model_name}'` |

---

## Common Pitfalls

- **Be consistent with indentation.** The examples use either 2-space indents
  with bare values or 4-space indents with quoted values. Both are valid YAML,
  but every key within one file must use the same indentation.
- **Match `variables` to your data.** A mismatch between `variables` and the
  number of channels in your CSV files is a common source of errors.
- **NPU-only compile presets.** `compress_npu_layer_data` and
  `forced_soft_npu_preset` apply only to devices with the TinyEngine NPU
  (e.g. F28P55, MSPM0G5187, AM13E2).
