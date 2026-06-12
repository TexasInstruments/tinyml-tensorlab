# Tiny ML Tensorlab Constants Reference

This guide maps legacy skill assets to their live sources in tinyml-tensorlab. The skill no longer duplicates constants—it references the authoritative sources directly.

## Constants Location Map

### Timeseries Module

**Parameters** (hyperparameters, defaults, ranges):
- **Source:** `$TINYML_BASE_PATH/tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/params.py`
- **Contains:** `init_params()` function with full default config (training epochs, batch size, learning rate, quantization settings, NAS parameters, etc.)
- **When to check:** Step 8 (Training section generation) — for default hyperparameters and their ranges

**Constants** (enums, task types, device types):
- **Source:** `$TINYML_BASE_PATH/tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/constants.py`
- **Contains:** 
  - `TASK_TYPE_*` constants (MOTOR_FAULT, ECG_CLASSIFICATION, ARC_FAULT, etc.)
  - `COMPILATION_DEFAULT`, `COMPILATION_FORCED_SOFT_NPU`, etc.
  - Feature extraction preset definitions (FFT presets, raw presets, etc.)
  - Data processing transform definitions
- **When to check:** 
  - Step 2 (list supported task types)
  - Step 6B (feature extraction/data processing recommendations)
  - Step 10 (compilation presets)

### Vision Module

**Parameters** (hyperparameters, defaults):
- **Source:** `$TINYML_BASE_PATH/tinyml-modelmaker/tinyml_modelmaker/ai_modules/vision/params.py`

**Constants** (enums, task types):
- **Source:** `$TINYML_BASE_PATH/tinyml-modelmaker/tinyml_modelmaker/ai_modules/vision/constants.py`

### Model Descriptions and Models

**Model metadata and configurations** (supported models, architectures, parameter counts, complexity tiers):
- **Source:** `$TINYML_BASE_PATH/tinyml-modelzoo/tinyml_modelzoo/model_descriptions/` (Python modules: `classification.py`, `forecasting.py`, `regression.py`, `anomalydetection.py`)
- **Contains:** Model class definitions, architecture configurations, metadata
- **When to check:** Step 7 (Model selection) — for available models and their properties

**Model implementations** (actual neural network architectures):
- **Source:** `$TINYML_BASE_PATH/tinyml-modelzoo/tinyml_modelzoo/models/` (Python modules: `classification.py`, `forecasting.py`, `regression.py`, `anomalydetection.py`, `feature_extraction.py`, `image.py`)
- **Contains:** Model class implementations, layer definitions
- **When to check:** Advanced use cases — when understanding model architecture details

## Legacy Assets (Removed)

The following files were embedded in the skill and have been removed. They are no longer needed:

| Legacy File | Reason | Use Instead |
|---|---|---|
| `assets/timeseries_default_params.md` | Duplicated params.py | `$TINYML_BASE_PATH/tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/params.py` |
| `assets/vision_default_params.md` | Duplicated params.py | `$TINYML_BASE_PATH/tinyml-modelmaker/tinyml_modelmaker/ai_modules/vision/params.py` |
| `assets/timeseries_module_constants.md` | Duplicated constants.py | `$TINYML_BASE_PATH/tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/constants.py` |
| `assets/vision_module_constants.md` | Duplicated constants.py | `$TINYML_BASE_PATH/tinyml-modelmaker/tinyml_modelmaker/ai_modules/vision/constants.py` |
| `assets/timeseries_data_proc_feat_ext_consts.md` | Duplicated constants.py | `$TINYML_BASE_PATH/tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/constants.py` |
| `assets/vision_data_proc_feat_ext_consts.md` | Duplicated constants.py | `$TINYML_BASE_PATH/tinyml-modelmaker/tinyml_modelmaker/ai_modules/vision/constants.py` |
| `assets/model_descriptions/` | Duplicated modelzoo | `$TINYML_BASE_PATH/tinyml-modelzoo/tinyml_modelzoo/model_descriptions/` |
| `assets/models/` | Duplicated modelzoo | `$TINYML_BASE_PATH/tinyml-modelzoo/tinyml_modelzoo/models/` |

## Accessing Constants from Python

Within the skill runner scripts or user code, constants can be accessed directly:

```python
from tinyml_modelmaker.ai_modules.timeseries import constants
from tinyml_modelmaker.ai_modules.timeseries import params

# List task types
print(constants.TASK_TYPES)

# Get default params
default = params.init_params()
print(default['training']['training_epochs'])
```

## Environment Variable

Ensure `$TINYML_BASE_PATH` is set and exported before running the skill:

```bash
export TINYML_BASE_PATH=/path/to/tinyml-tensorlab
```

Verify installation:
```bash
ls $TINYML_BASE_PATH/tinyml-modelmaker/tinyml_modelmaker/ai_modules/
# Should list: timeseries/, vision/
```
