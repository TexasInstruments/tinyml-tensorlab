# Model Registration Reference

Reference for how models are defined, registered and resolved in Tiny ML
ModelZoo. For a step-by-step walkthrough that adds a model and trains it on an
example, see the **[Bring Your Own Model (BYOM) Guide](docs/BRING_YOUR_OWN_MODEL.md)**.

This reference describes the **permanent** route: model classes and descriptions registered inside
`tinyml-modelzoo`. `tinyml-tinyverse` and `tinyml-modelmaker` import them from `tinyml_modelzoo`.

To try a model without editing `tinyml-modelzoo`, use `training.custom_model_path` instead (see **Option A** in the
[BYOM Guide](docs/BRING_YOUR_OWN_MODEL.md)). The file only needs a class whose name equals `training.model_name`; no
description, `__all__` entry or `enabled_models_list` entry is required.

- [Files](#files)
- [Name Resolution](#name-resolution)
- [Model Classes](#model-classes)
- [Spec Layer Types](#spec-layer-types)
- [Model Descriptions](#model-descriptions)
- [Registry API](#registry-api)
- [Naming Conventions](#naming-conventions)
- [Test Snippets](#test-snippets)
- [Troubleshooting](#troubleshooting)

---

## Files

| Task type | Model classes (`tinyml_modelzoo/models/`) | Model descriptions (`tinyml_modelzoo/model_descriptions/`) |
|-----------|-------------------------------------------|------------------------------------------------------------|
| Time series classification | `classification.py` | `classification.py` |
| Time series regression | `regression.py` | `regression.py` |
| Time series anomaly detection | `anomalydetection.py` | `anomalydetection.py` |
| Time series forecasting | `forecasting.py` | `forecasting.py` |
| Feature extraction | `feature_extraction.py` | — |
| Image classification | `image.py` | — |
| Audio classification | `audio.py` | — |
| Radar | `radar.py` | — |

The model modules auto-discovered by the registry are listed in
`_MODEL_MODULES` in `tinyml_modelzoo/models/__init__.py`.

---

## Name Resolution

A model has two names:

| Name | Defined in | Used by |
|------|------------|---------|
| **Description name** (e.g. `CLS_1k_NPU`) | Key of `_model_descriptions` in `model_descriptions/<task>.py`, and listed in `enabled_models_list` | `training.model_name` in the config YAML, and the GUI |
| **Class name** (e.g. `CNN_TS_GEN_BASE_1K_NPU`) | Python class in `models/<task>.py`, listed in that file's `__all__` | `model_training_id` in the description; `get_model()` |

Resolution path when a config is run:

```
config.yaml: training.model_name ──► model_descriptions: _model_descriptions[<name>]   (must be in enabled_models_list)
                                         └─ training.model_training_id ──► models: model_dict[<class name>]
```

- `tinyml_modelmaker/run_tinyml_modelmaker.py` resolves `training.model_name`
  with `get_model_description()`. If no **enabled** description has that name,
  the run stops with:
  `please check if the given model_name is a supported one: <name>`.
- A description is therefore **required** for any model used from a config.
  A class with no description can only be used directly through `get_model()`.

---

## Model Classes

A model class must:

- be listed in the `__all__` list of its model file, and
- accept a `config` argument: `get_model()` instantiates every model as
  `ModelClass(config=config_dict)`, where `config_dict` contains
  `variables`, `num_classes`, `input_features` and `dual_op`.

Names in a module's optional `_REGISTRY_EXCLUDE` set are exported but not
added to `model_dict`.

Input tensor shape:

| Data | Shape |
|------|-------|
| Time series | `(batch, variables, input_features, 1)` |
| Images | `(batch, channels, height, width)` |

### Option A: Spec-based model (`GenericModelWithSpec`)

The network is an ordered dict of layer specs, built by `gen_model_spec()` and
instantiated by `_init_model_from_spec()`. `forward()` runs the layers in order.

```python
from ..utils import py_utils
from .base import GenericModelWithSpec


class MY_NEW_MODEL_2K_NPU(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=1, num_classes=2):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(5, 1), stride=(1, 1))}
        layers += {'2': dict(type='MaxPoolLayer', kernel_size=(2, 1), stride=(2, 1))}
        layers += {'3': dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(3, 1), stride=(1, 1))}
        layers += {'4': dict(type='AdaptiveAvgPoolLayer', output_size=(4, 1))}
        layers += {'5': dict(type='ReshapeLayer', ndim=2)}
        layers += {'6': dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        return dict(model_spec=layers)
```

`_init_model_from_spec()` overrides these values automatically:

| Layer position | Condition | Overridden value |
|----------------|-----------|------------------|
| First layer | `type='BatchNormLayer'` | `num_features = variables` |
| Second layer | `type` is `ConvLayer` or `ConvBNReLULayer` | `in_channels = variables` |
| Last layer | `type='LinearLayer'` | `out_features = num_classes` |
| Any `LinearLayer` | `in_features=None` | inferred from the previous layer's output size |

### Option B: Custom `torch.nn.Module`

```python
import torch.nn as nn


class MY_CUSTOM_MODEL(nn.Module):
    def __init__(self, config, input_features=128, variables=1, num_classes=3):
        super().__init__()
        if isinstance(config, dict):
            variables = config.get('variables', variables)
            num_classes = config.get('num_classes', num_classes)
            input_features = config.get('input_features', input_features)
        self.conv1 = nn.Conv2d(variables, 32, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        return self.fc(x.view(x.size(0), -1))
```

Any class listed in `__all__` that has a `forward` attribute is registered.

---

## Spec Layer Types

Valid `type` values are the layer factory functions in
`tinyml_modelzoo/layers/tinynn.py`.

| Type | PyTorch layer(s) | Parameters |
|------|------------------|------------|
| `IdentityLayer` | `Identity` | — |
| `BatchNormLayer` | `BatchNorm2d` | `num_features` |
| `ReLULayer` / `ReluLayer` | `ReLU` | — |
| `ConvLayer` | `Conv2d` | `in_channels`, `out_channels`, `kernel_size` (2-tuple), `stride=1`, `padding=None`, `dilation=1` |
| `ConvBNReLULayer` | `Conv2d` + `BatchNorm2d` + `ReLU` | `in_channels`, `out_channels`, `kernel_size` (2-tuple), `stride`, `padding=None`, `dilation=1` |
| `MaxPoolLayer` | `MaxPool2d` | `kernel_size` (2-tuple), `stride=1`, `padding=None`, `dilation=1` |
| `AvgPoolLayer` | `AvgPool2d` | `kernel_size` (2-tuple), `stride=1`, `padding=None` |
| `AdaptiveAvgPoolLayer` | `AdaptiveAvgPool2d` | `output_size` (2-tuple) |
| `ReshapeLayer` | Flatten to 2D | `ndim=2` (only 2 supported) |
| `LinearLayer` | `Linear` | `in_features` (`None` = inferred), `out_features` |
| `LinearBNReLULayer` | `Linear` + `BatchNorm1d` + `ReLU` | `in_features`, `out_features` |
| `DropoutLayer` | `Dropout` | `dropout_prob` |
| `TCDSBasicBlockLayer` | Depthwise-separable residual block | `in_channels`, `out_channels`, `kernel_size`, `stride=2` |
| `RNNLayer` | `RNN` / `LSTM` / `GRU` | `input_size`, `hidden_size`, `return_last_timestep=False`, `num_layers=1`, `dropout=0.1`, `batch_first=True`, `rnn_type='RNN'` |
| `LSTMLayer` | `LSTM` | Same as `RNNLayer` (without `rnn_type`) |
| `GRULayer` | `GRU` | Same as `RNNLayer` (without `rnn_type`) |

Conventions for `ConvLayer`, `ConvBNReLULayer` and the pooling layers:

- `padding=None` means "same" padding: `k // 2` per kernel dimension.
- An integer `padding`, `stride` or `dilation` applies to the height dimension
  only: `n` becomes `(n, 0)` for padding and `(n, 1)` for stride and dilation.

For layer configurations that the TI NPU can accelerate, see
[NPU Configuration Guidelines](docs/NPU_CONFIGURATION_GUIDELINES.md).

---

## Model Descriptions

Each `model_descriptions/<task>.py` defines:

| Symbol | Purpose |
|--------|---------|
| `template_model_description` | Default description for the task. New entries start from `deepcopy(template_model_description)` |
| `_model_descriptions` | Dict of description name → description |
| `enabled_models_list` | Description names that are visible. Entries not listed here are ignored |
| `get_model_descriptions()` / `get_model_description(name)` | Return enabled descriptions only |

Entry format:

```python
    'MY_CLS_2k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Short architecture description shown in the GUI.',
        ),
        'training': dict(
            model_training_id='MY_NEW_MODEL_2K_NPU',   # class name in models/
            model_name='MY_CLS_2k_NPU',                # same as the dict key
            target_devices=[
                constants.TARGET_DEVICE_F28P55,
                constants.TARGET_DEVICE_MSPM0G5187,
            ],
        ),
    }),
```

| Field | Required | Description |
|-------|----------|-------------|
| `training.model_training_id` | Yes | Class name, exactly as registered in `model_dict` |
| `training.model_name` | Yes | Same string as the `_model_descriptions` key |
| `training.target_devices` | Yes | List of `constants.TARGET_DEVICE_*` values the model is intended for |
| `common.model_details` | No | Free-text description shown in the GUI |
| `training.properties` | No | GUI parameter groups. Only used by the GUI; copy from an existing entry such as `CLS_1k_NPU` |

All other fields (task category, quantization default, batch size, training
devices, learning rate) are inherited from `template_model_description`.

---

## Registry API

Defined in `tinyml_modelzoo/models/__init__.py`:

| Symbol | Description |
|--------|-------------|
| `model_dict` | Dict of class name → model class, built from each module's `__all__` |
| `get_model(model_name, variables, num_classes, input_features=None, model_config=None, model_spec=None, dual_op=False)` | Instantiate a registered class by class name. Raises `ValueError` if not found. `model_config` is not implemented and only emits a warning |
| `list_models()` | Sorted list of registered class names |
| `get_model_count()` | Number of registered classes |

---

## Naming Conventions

| Name | Style | Examples |
|------|-------|----------|
| Class name | `SCREAMING_SNAKE_CASE`, model family + parameter count, `_NPU` suffix if NPU-compliant | `CNN_TS_GEN_BASE_1K_NPU`, `RES_ADD_CNN_TS_GEN_BASE_3K`, `REG_TS_GEN_BASE_1K`, `AE_CNN_TS_GEN_BASE_4K`, `FC_CNN_TS_GEN_BASE_13K` |
| Description name | Task prefix + parameter count, `_NPU` suffix if NPU-compliant | `CLS_1k_NPU`, `CLS_ResAdd_3k`, `REGR_2k`, `AD_17k`, `FCST_LSTM10` |

---

## Test Snippets

Registration and forward pass (time series):

```python
import torch
from tinyml_modelzoo.models import get_model, list_models

assert 'MY_NEW_MODEL_2K_NPU' in list_models()
model = get_model('MY_NEW_MODEL_2K_NPU', variables=1, num_classes=3, input_features=128)
y = model(torch.randn(1, 1, 128, 1))
print(tuple(y.shape))                                  # (1, 3)
print(sum(p.numel() for p in model.parameters()))      # parameter count
```

Description lookup:

```python
from tinyml_modelzoo.model_descriptions.classification import get_model_description

d = get_model_description('MY_CLS_2k_NPU')             # None if missing or not enabled
print(d['training']['model_training_id'])
```

ONNX export:

```python
import torch
from tinyml_modelzoo.models import get_model

model = get_model('MY_NEW_MODEL_2K_NPU', variables=1, num_classes=3, input_features=128).eval()
torch.onnx.export(model, torch.randn(1, 1, 128, 1), 'my_model.onnx', opset_version=11)
```

Full training run: set `training.model_name` to the **description name** in a
copy of an example config, then run it with `run_tinyml_modelzoo.sh` /
`run_tinyml_modelzoo.bat`. See the [BYOM Guide](docs/BRING_YOUR_OWN_MODEL.md).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `please check if the given model_name is a supported one: <name>` | `training.model_name` is not a key of `_model_descriptions`, or it is not in `enabled_models_list` | Add both; use the description name, not the class name |
| `Model '<name>' not found in registry or specified model_spec` | Class name not in `model_dict` | Check `__all__` and the spelling of `model_training_id` |
| Warning `Could not import model module '<module>': ...`, and that module's models are missing | `ImportError` while importing the model file | Run `python -c "import tinyml_modelzoo.models.<module>"` to see the error |
| `SyntaxError` on `import tinyml_modelzoo.models` | Syntax error in a model file (not caught by the registry) | Fix the reported line |
| Shape mismatch at a `LinearLayer` | `in_features` does not match the flattened previous output | Set `in_features=None` to infer it, or recompute channels × pooled height × pooled width |
| `AssertionError: kernel_size must be a tuple of size 2` | Integer `kernel_size` passed to a conv or pooling layer | Use a 2-tuple such as `(5, 1)` |
