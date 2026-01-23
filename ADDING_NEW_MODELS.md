# How to Add a New Model to TinyML ModelZoo

This guide explains how to add a new model to the TinyML ModelZoo. With the dynamic registration system, you only need to modify files in `tinyml-modelzoo` - **no changes are required in `tinyml-tinyverse` or `tinyml-modelmaker`**.

## Quick Summary

To add a new model:

1. Add your model class to the appropriate file in `tinyml_modelzoo/models/`
2. Add the class name to that file's `__all__` list
3. (Optional) Add device performance info to `device_info/run_info.py`
4. (Optional) Add a model description to `model_descriptions/` for GUI integration

That's it! The model is automatically registered and available everywhere. **No changes are required in `tinyml-tinyverse` or `tinyml-modelmaker`.**

## Step-by-Step Guide

### Step 1: Choose the Right Model File

Models are organized by task type in `tinyml_modelzoo/models/`:

| Task Type                  | File                    | Examples                                   |
|----------------------------|-------------------------|--------------------------------------------|
| Time series classification | `classification.py`     | CNN_TS_GEN_BASE_1K_NPU, HAR_TINIE_CNN_2K       |
| Time series regression     | `regression.py`         | REG_TS_GEN_BASE_1K, REG_TS_CNN_13K         |
| Anomaly detection          | `anomalydetection.py`   | AE_CNN_TS_GEN_BASE_4K, AD_CNN_TS_17K       |
| Time series forecasting    | `forecasting.py`        | FC_CNN_TS_GEN_BASE_13K, LSTM10_TS_GEN_BASE |
| Feature extraction         | `feature_extraction.py` | FEModel, FEModelLinear                     |
| Image classification       | `image.py`              | CNN_LENET5                                 |

### Step 2: Create Your Model Class

Your model should inherit from `GenericModelWithSpec` (for spec-based models) or `torch.nn.Module` (for custom architectures).

#### Option A: Spec-Based Model (Recommended)

Use this approach for models that can be defined as a sequence of standard layers:

```python
from ..utils import py_utils
from .base import GenericModelWithSpec


class MY_NEW_MODEL_2K(GenericModelWithSpec):
    """
    My new 2K parameter classification model.

    Architecture: 2 Conv+BN+ReLU layers + MaxPool + Linear
    """

    def __init__(self, config, input_features=128, variables=1, num_classes=3):
        super().__init__(config, input_features=input_features,
                        variables=variables, num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(
            model_spec=self.model_spec,
            variables=self.variables,
            input_features=self.input_features,
            num_classes=self.num_classes
        )

    def gen_model_spec(self):
        """Define the model architecture using layer specifications."""
        layers = py_utils.DictPlus()

        # Input normalization
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}

        # Conv block 1
        layers += {'1': dict(type='ConvBNReLULayer',
                            in_channels=self.variables,
                            out_channels=16,
                            kernel_size=(5, 1),
                            stride=(1, 1))}
        layers += {'2': dict(type='MaxPoolLayer',
                            kernel_size=(2, 1),
                            stride=(2, 1))}

        # Conv block 2
        layers += {'3': dict(type='ConvBNReLULayer',
                            in_channels=16,
                            out_channels=32,
                            kernel_size=(3, 1),
                            stride=(1, 1))}
        layers += {'4': dict(type='MaxPoolLayer',
                            kernel_size=(2, 1),
                            stride=(2, 1))}

        # Global pooling and classifier
        layers += {'5': dict(type='AdaptiveAvgPoolLayer', output_size=(4, 1))}
        layers += {'6': dict(type='ReshapeLayer', ndim=2)}
        layers += {'7': dict(type='LinearLayer',
                            in_features=32 * 4,
                            out_features=self.num_classes)}

        return dict(model_spec=layers)
```

#### Option B: Custom PyTorch Model

For more complex architectures:

```python
import torch
import torch.nn as nn


class MY_CUSTOM_MODEL(nn.Module):
    """Custom model with non-standard architecture."""

    def __init__(self, config, input_features=128, variables=1, num_classes=3):
        super().__init__()
        # Extract config values (may come from get_model() call)
        if isinstance(config, dict):
            variables = config.get('variables', variables)
            num_classes = config.get('num_classes', num_classes)
            input_features = config.get('input_features', input_features)

        self.variables = variables
        self.num_classes = num_classes
        self.input_features = input_features

        # Define your layers
        self.conv1 = nn.Conv2d(variables, 32, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (batch, variables, input_features) or (batch, variables, input_features, 1)
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # Add channel dimension

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

### Step 3: Add to `__all__`

At the bottom of the model file, add your class name to the `__all__` list:

```python
# Export all classification models
__all__ = [
    'CNN_TS_GEN_BASE_100',
    'CNN_TS_GEN_BASE_1K_NPU',
    # ... existing models ...
    'MY_NEW_MODEL_2K',  # <-- Add your model here
]
```

### Step 4: Verify Your Model

Run the test script to verify your model works:

```bash
source ~/.pyenv/versions/py310_tinyml/bin/activate
cd tinyml-modelzoo
./run_tests.sh --skip-training
```

You should see your model in the count and be able to instantiate it:

```python
from tinyml_modelzoo.models import get_model, list_models

# Check if model is registered
print('MY_NEW_MODEL_2K' in list_models())  # Should print True

# Instantiate the model
model = get_model('MY_NEW_MODEL_2K', variables=1, num_classes=3, input_features=128)
print(model)
```

### Step 5 (Optional): Add Device Performance Info

If you have benchmarked your model on target devices, add the info to `device_info/run_info.py`:

```python
DEVICE_RUN_INFO = {
    # ... existing models ...

    'MyModelName_ForGUI': {
        'F28P55': {'flash': 2500, 'inference_time_us': 150, 'sram': 1200},
        'F28P65': {'flash': 2500, 'inference_time_us': 400, 'sram': 1200},
        'F2837': {'flash': 2500, 'inference_time_us': 800, 'sram': 1200},
        # Add TBD for untested devices
        'MSPM0G3507': {'flash': 'TBD', 'inference_time_us': 'TBD', 'sram': 'TBD'},
    },
}
```

### Step 6 (Optional): Add GUI Model Description

If you want the model to appear in the TinyML Studio GUI, add a description to the appropriate file in `tinyml_modelzoo/model_descriptions/`:

| Task Type | Description File |
|-----------|------------------|
| Time series classification | `model_descriptions/classification.py` |
| Time series regression | `model_descriptions/regression.py` |
| Anomaly detection | `model_descriptions/anomalydetection.py` |
| Time series forecasting | `model_descriptions/forecasting.py` |

Add your model to the `_model_descriptions` dict and `enabled_models_list`:

```python
# In tinyml_modelzoo/model_descriptions/classification.py

from tinyml_modelzoo import constants
from tinyml_modelzoo.utils import deep_update_dict
from tinyml_modelzoo.device_info import DEVICE_RUN_INFO

_model_descriptions = {
    # ... existing models ...

    'My_Model_Name_2k_t': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='My new 2K model. 2 Conv+BN+ReLU layers.'
        ),
        'training': dict(
            model_training_id='MY_NEW_MODEL_2K',  # Must match class name in models/!
            model_name='My_Model_Name_2k_t',
            properties=[dict(type="group", dynamic=True, script="generictimeseriesclassification.py",
                           name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties,
            target_devices={
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=None) |
                    (DEVICE_RUN_INFO['My_Model_Name_2k_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=None) |
                    (DEVICE_RUN_INFO['My_Model_Name_2k_t'][constants.TARGET_DEVICE_F28P65]),
                # ... add other target devices ...
            },
        ),
    }),
}

enabled_models_list = [
    # ... existing models ...
    'My_Model_Name_2k_t',  # Add to enable in GUI
]
```

**Important fields:**
- `model_training_id`: Must exactly match your model class name in `models/`
- `model_name`: The display name shown in the GUI
- `model_details`: Brief description of the model architecture
- `target_devices`: Dict of supported devices with performance info from `DEVICE_RUN_INFO`
- `properties`: GUI properties for training parameters (use the template)

After adding, verify the description is generated correctly:

```bash
cd tinyml-modelmaker
python scripts/run_generate_description.py
# Check data/descriptions/description_timeseries.json for your model
```

## Available Layer Types

For spec-based models, you can use these layer types in `gen_model_spec()`:

| Layer Type | Description | Key Parameters |
|------------|-------------|----------------|
| `BatchNormLayer` | Batch normalization | `num_features` |
| `ConvBNReLULayer` | Conv2d + BatchNorm + ReLU | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` |
| `MaxPoolLayer` | Max pooling | `kernel_size`, `stride`, `padding` |
| `AvgPoolLayer` | Average pooling | `kernel_size`, `stride`, `padding` |
| `AdaptiveAvgPoolLayer` | Adaptive average pooling | `output_size` |
| `ReshapeLayer` | Flatten/reshape | `ndim` |
| `LinearLayer` | Fully connected | `in_features`, `out_features` |
| `ReluLayer` | ReLU activation | - |
| `SigmoidLayer` | Sigmoid activation | - |
| `LSTMLayer` | LSTM layer | `input_size`, `hidden_size` |
| `TransposeConvBNReLULayer` | Transposed conv | Same as ConvBNReLULayer |
| `ConvTranspose` | Transposed conv (no BN/ReLU) | Same as Conv2d |
| `CatLayer` | Concatenation | - |
| `AddLayer` | Element-wise addition | - |

## Naming Conventions

- **Class names** (in `models/`): Use SCREAMING_SNAKE_CASE with model type and parameter count
  - Classification: `CNN_TS_GEN_BASE_1K_NPU`, `RES_ADD_CNN_TS_GEN_BASE_3K`
  - Regression: `REG_TS_GEN_BASE_1K`, `REG_TS_CNN_13K`
  - Anomaly Detection: `AE_CNN_TS_GEN_BASE_4K`, `AD_CNN_TS_17K`
  - Forecasting: `FC_CNN_TS_GEN_BASE_13K`, `LSTM10_TS_GEN_BASE`

- **GUI names** (in `model_descriptions/`): Use `TimeSeries_Generic_Xk_t` or `TimeSeries_Generic_Xk_NPU_t` pattern
  - `TimeSeries_Generic_1k_NPU_t`, `TimeSeries_Generic_Regr_10k_t`, `TimeSeries_Generic_AD_4k_t`

## Testing Your Model

### Basic Import Test

```python
from tinyml_modelzoo.models import get_model

model = get_model('MY_NEW_MODEL_2K', variables=1, num_classes=3, input_features=128)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
```

### Forward Pass Test

```python
import torch

model = get_model('MY_NEW_MODEL_2K', variables=1, num_classes=3, input_features=128)
x = torch.randn(1, 1, 128)  # (batch, variables, features)
y = model(x)
print(f"Output shape: {y.shape}")  # Should be (1, 3)
```

### ONNX Export Test

```python
import torch
import torch.onnx

model = get_model('MY_NEW_MODEL_2K', variables=1, num_classes=3, input_features=128)
model.eval()
x = torch.randn(1, 1, 128, 1)
torch.onnx.export(model, x, "my_model.onnx", opset_version=11)
```

### Full Training Test

To test your model with an actual training run, use the training wrapper:

```bash
# Linux
./run_tinyml_modelzoo.sh examples/hello_world/config.yaml

# Windows
run_tinyml_modelzoo.bat examples\hello_world\config.yaml
```

You can modify an example config to use your new model by changing the `model_training_id` field.

## Troubleshooting

### Model not appearing in `model_dict`

1. Check that the class name is in the file's `__all__` list
2. Verify there are no import errors in your model file:
   ```bash
   python -c "from tinyml_modelzoo.models.classification import *"
   ```

### Model instantiation fails

1. Ensure your `__init__` accepts a `config` dict parameter
2. Handle both dict-style config and keyword arguments

### Model training fails

1. Verify input/output shapes match the data pipeline
2. For time series: input is typically `(batch, variables, features, 1)`
3. For images: input is typically `(batch, channels, height, width)`

## Summary Checklist

- [ ] Model class added to appropriate file in `tinyml_modelzoo/models/`
- [ ] Class name added to file's `__all__` list
- [ ] Model instantiates correctly via `get_model()`
- [ ] Forward pass produces correct output shape
- [ ] (Optional) Device performance info added to `device_info/run_info.py`
- [ ] (Optional) GUI description added to `model_descriptions/` and `enabled_models_list`
- [ ] (Optional) Verified with `run_generate_description.py`
