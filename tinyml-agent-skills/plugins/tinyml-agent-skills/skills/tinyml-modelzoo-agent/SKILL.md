---
name: tinyml-modelzoo-agent
description: Guides users through adding a new or custom neural network model to tinyml-modelzoo so it becomes available for training, quantization, and compilation. Trigger when user mentions: "add my model", "register a model", "bring your own model", "custom architecture", "new model to modelzoo", "BYOM", or wants to use a model not already in the zoo. Requires tinyml-modelzoo setup (TINYML_MODELZOO_PATH must be set). Always run /tinyml-agent-skills:setup first if not done.
---

# TinyML ModelZoo — Add Custom Model

Adds a new neural network architecture to tinyml-modelzoo so it works end-to-end with training, quantization, and compilation.

**No changes needed in tinyml-modelmaker or tinyml-tinyverse — modelzoo only.**

---

## Session Prerequisites

Load `TINYML_MODELZOO_PATH` from `~/.tinyml-agent-skills/.env` and activate the venv:
```bash
source ~/.tinyml-agent-skills/.env 2>/dev/null || true
source "$TINYML_MODELZOO_PATH/.venv/bin/activate" 2>/dev/null || true
```

If `TINYML_MODELZOO_PATH` is unset: stop and tell user to run `/tinyml-agent-skills:setup` first.

---

## Files Modified (Overview)

| File | Purpose |
|---|---|
| `tinyml_modelzoo/models/<task>.py` | Model class definition |
| `tinyml_modelzoo/device_info/run_info.py` | Device performance info |
| `tinyml_modelzoo/model_descriptions/<task>.py` | Pipeline + GUI integration |

---

## Step 1: Gather Requirements

Ask the user:

1. **Task type** — classification / regression / anomaly detection / forecasting / image classification?
2. **Model name** — choose a class name following SCREAMING_SNAKE_CASE with param count suffix:
   - Classification: `CNN_TS_MY_MODEL_2K` or `CNN_TS_MY_MODEL_2K_NPU` (if NPU-targeted)
   - Regression: `REG_TS_MY_MODEL_2K`
   - Anomaly detection: `AE_MY_MODEL_4K`
   - Forecasting: `FC_MY_MODEL_13K`
3. **Architecture approach** — spec-based (recommended, uses standard layer blocks) or custom PyTorch?
4. **NPU target?** — if yes, enforce NPU constraints (see Step 2C)
5. **Architecture details** — how many layers, what types, approximate param count?

Set session variables:
```bash
TASK_TYPE=<classification|regression|anomalydetection|forecasting|image>
MODEL_CLASS_NAME=<e.g., CNN_TS_MY_MODEL_2K>
MODEL_GUI_NAME=<e.g., My_Model_2k_t>   # snake_case_t suffix for GUI name
```

Task → file mapping:
| Task | Model file | Description file |
|---|---|---|
| classification | `models/classification.py` | `model_descriptions/classification.py` |
| regression | `models/regression.py` | `model_descriptions/regression.py` |
| anomalydetection | `models/anomalydetection.py` | `model_descriptions/anomalydetection.py` |
| forecasting | `models/forecasting.py` | `model_descriptions/forecasting.py` |
| image | `models/image.py` | `model_descriptions/classification.py` |

```bash
MODEL_FILE="$TINYML_MODELZOO_PATH/tinyml_modelzoo/models/${TASK_TYPE}.py"
DESC_FILE="$TINYML_MODELZOO_PATH/tinyml_modelzoo/model_descriptions/${TASK_TYPE}.py"
```

---

## Step 2: Write the Model Class

Open `$MODEL_FILE` and append the new class before the `__all__` list at the bottom.

### Option A: Spec-Based Model (Recommended)

Use this for models expressible as standard layer sequences. Avoids custom ONNX issues.

```python
from ..utils import py_utils
from .base import GenericModelWithSpec


class MY_NEW_MODEL_2K(GenericModelWithSpec):
    """
    <Brief description>. Architecture: <layer summary>.
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
        layers = py_utils.DictPlus()

        # Input normalization
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}

        # Conv block 1
        layers += {'1': dict(type='ConvBNReLULayer',
                            in_channels=self.variables, out_channels=16,
                            kernel_size=(5, 1), stride=(1, 1))}
        layers += {'2': dict(type='MaxPoolLayer', kernel_size=(2, 1), stride=(2, 1))}

        # Conv block 2
        layers += {'3': dict(type='ConvBNReLULayer',
                            in_channels=16, out_channels=32,
                            kernel_size=(3, 1), stride=(1, 1))}
        layers += {'4': dict(type='MaxPoolLayer', kernel_size=(2, 1), stride=(2, 1))}

        # Classifier
        layers += {'5': dict(type='AdaptiveAvgPoolLayer', output_size=(1, 1))}  # use (1,1) for ONNX safety
        layers += {'6': dict(type='ReshapeLayer', ndim=2)}
        layers += {'7': dict(type='LinearLayer',
                            in_features=32, out_features=self.num_classes)}

        return dict(model_spec=layers)
```

**Available layer types for `gen_model_spec()`:**

| Layer | Key params |
|---|---|
| `BatchNormLayer` | `num_features` |
| `ConvBNReLULayer` | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` |
| `MaxPoolLayer` | `kernel_size`, `stride`, `padding` |
| `AvgPoolLayer` | `kernel_size`, `stride`, `padding` |
| `AdaptiveAvgPoolLayer` | `output_size` — use `(1,1)` for ONNX safety |
| `ReshapeLayer` | `ndim` |
| `LinearLayer` | `in_features`, `out_features` |
| `ReluLayer` | — |
| `SigmoidLayer` | — |
| `LSTMLayer` | `input_size`, `hidden_size` |
| `CatLayer` | — |
| `AddLayer` | — |

### Option B: Custom PyTorch Model

For architectures that cannot be expressed as layer specs:

```python
import torch.nn as nn


class MY_CUSTOM_MODEL(nn.Module):
    """Custom model. Architecture: <describe>."""

    def __init__(self, config, input_features=128, variables=1, num_classes=3):
        super().__init__()
        if isinstance(config, dict):
            variables = config.get('variables', variables)
            num_classes = config.get('num_classes', num_classes)
            input_features = config.get('input_features', input_features)
        self.variables = variables
        self.num_classes = num_classes
        self.input_features = input_features

        self.conv1 = nn.Conv2d(variables, 32, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # Input: (batch, variables, input_features, 1) — 4D required
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
```

### Option C: NPU-Compatible Model

If user targets NPU devices (F28P55, MSPM0G5187, AM13E2), enforce these constraints in `gen_model_spec()`:
- First FCONV layer: `in_channels=1` (not `variables`)
- All `out_channels` must be multiples of 4
- All conv kernel heights ≤ 7
- MaxPool: at least one kernel dimension must be 1–4 (the other can be any)
- FC `in_features` ≥ 16 (8-bit weights) or ≥ 8 (4-bit weights) — source: `docs/NPU_CONFIGURATION_GUIDELINES.md`

---

## Step 3: Register in `__all__`

In `$MODEL_FILE`, find the `__all__` list at the bottom and add the class name:

```python
__all__ = [
    # ... existing entries ...
    'MY_NEW_MODEL_2K',   # ← add here
]
```

Show user the updated `__all__` block for confirmation before saving.

---

## Step 4: Verify Model Loads and Runs

```python
import sys
sys.path.insert(0, "$TINYML_MODELZOO_PATH")

from tinyml_modelzoo.models import get_model, list_models
import torch

# Registration check
assert 'MY_NEW_MODEL_2K' in list_models(), "Model not in registry — check __all__"

# Instantiation
model = get_model('MY_NEW_MODEL_2K', variables=1, num_classes=3, input_features=128)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass
x = torch.randn(1, 1, 128, 1)
y = model(x)
assert y.shape == (1, 3), f"Wrong output shape: {y.shape}"
print(f"Forward pass OK: {y.shape}")

# ONNX export
model.eval()
torch.onnx.export(model, x, "/tmp/test_model.onnx", opset_version=11)
print("ONNX export OK")
```

**If any check fails:** fix before proceeding. Common issues:
- Not in registry → class name not in `__all__`, or import error in model file
- Shape mismatch → wrong `in_features` in LinearLayer (calculate manually from pooling output)
- ONNX export fails → `AdaptiveAvgPoolLayer` with `output_size` other than `(1,1)`; switch to `(1,1)`

---

## Step 5: Add Device Performance Info

Open `$TINYML_MODELZOO_PATH/tinyml_modelzoo/device_info/run_info.py`.

Find the `DEVICE_RUN_INFO` dict and add an entry for the GUI model name (not the class name):

```python
DEVICE_RUN_INFO = {
    # ... existing entries ...

    'My_Model_Name_2k_t': {   # ← GUI name (MODEL_GUI_NAME)
        'F28P55':      {'flash': 2500,  'inference_time_us': 150,  'sram': 1200},
        'F28P65':      {'flash': 2500,  'inference_time_us': 400,  'sram': 1200},
        'F2837':       {'flash': 2500,  'inference_time_us': 800,  'sram': 1200},
        'MSPM0G3507':  {'flash': 'TBD', 'inference_time_us': 'TBD', 'sram': 'TBD'},
        'MSPM0G5187':  {'flash': 'TBD', 'inference_time_us': 'TBD', 'sram': 'TBD'},
        'AM263':       {'flash': 'TBD', 'inference_time_us': 'TBD', 'sram': 'TBD'},
    },
}
```

Ask user which devices to support. For untested devices, use `'TBD'` — this is fine and won't break anything.

---

## Step 6: Add Model Description (MANDATORY for training pipeline)

This step is **required** — without it, the model cannot be referenced by `model_name` in config files, and the training pipeline cannot find it.

Open `$DESC_FILE` and:

**A. Add entry to `_model_descriptions` dict:**

```python
_model_descriptions = {
    # ... existing entries ...

    'My_Model_Name_2k_t': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='My new 2K classification model. 2 Conv+BN+ReLU layers + MaxPool + Linear.'
        ),
        'training': dict(
            model_training_id='MY_NEW_MODEL_2K',   # MUST match class name exactly
            model_name='My_Model_Name_2k_t',
            properties=[dict(
                type="group", dynamic=True,
                script="generictimeseriesclassification.py",
                name="preprocessing_group",
                label="Preprocessing Parameters",
                default=[]
            )] + template_gui_model_properties,
            target_devices={
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=None) |
                    DEVICE_RUN_INFO['My_Model_Name_2k_t'][constants.TARGET_DEVICE_F28P55],
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=None) |
                    DEVICE_RUN_INFO['My_Model_Name_2k_t'][constants.TARGET_DEVICE_F28P65],
                # add the devices the user specified in Step 5
            },
        ),
    }),
}
```

**Key fields:**
- `model_training_id` — **must exactly match** the class name from `models/` (Step 2)
- `model_name` — the name users put in `config.yaml` under `training.model_name`
- `target_devices` — only include devices added to `DEVICE_RUN_INFO` in Step 5

**B. Add to `enabled_models_list`:**

```python
enabled_models_list = [
    # ... existing entries ...
    'My_Model_Name_2k_t',   # ← add here
]
```

Show user both additions for confirmation before saving.

---

## Step 7: Final Verification

```bash
cd "$TINYML_MODELZOO_PATH"
./run_tests.sh --skip-training 2>&1 | grep -E "PASS|FAIL|ERROR|MY_NEW_MODEL|found [0-9]+ model"
```

Check that model count increased and no errors appear.

**Verify model is usable in config:**
```yaml
# In any config.yaml
training:
  model_name: 'My_Model_Name_2k_t'   # GUI name from model_description
```

Or directly by class name:
```yaml
training:
  model_name: 'MY_NEW_MODEL_2K'       # Class name also works
```

---

## Summary Checklist

- [ ] Model class added to `tinyml_modelzoo/models/<task>.py`
- [ ] Class name added to `__all__` in same file
- [ ] Import test passes (`get_model()` returns instance)
- [ ] Forward pass produces correct output shape
- [ ] ONNX export succeeds
- [ ] Device info added to `device_info/run_info.py`
- [ ] Model description added to `model_descriptions/<task>.py`
- [ ] Added to `enabled_models_list`
- [ ] `run_tests.sh --skip-training` passes

Model is now available for use with `/tinyml-agent-skills:tinyml-workflow-agent`.
