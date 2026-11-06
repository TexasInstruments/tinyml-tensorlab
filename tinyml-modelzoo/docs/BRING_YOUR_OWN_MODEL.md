# Bring Your Own Model (BYOM)

This guide walks you through adding your own neural network to Tiny ML ModelZoo
and training it with an existing example. As a worked example, you will add a
small NPU-friendly classification model and train it on the
[DC arc fault](https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/dc_arc_fault/) dataset using
[`config_dsk.yaml`](https://github.com/TexasInstruments/tinyml-modelzoo/blob/main/examples/dc_arc_fault/config_dsk.yaml).

There are two ways to bring your own model. Choose the one that fits your situation.

## Two Ways to Bring Your Own Model

| | Option A — Quick Start | Option B — Permanent Addition |
|---|---|---|
| Edit modelzoo source? | **No** | Yes |
| Survives reinstall? | Only if you keep your `.py` file | Yes |
| Appears in GUI model picker? | No | Yes |
| Full train → ONNX → MCU pipeline? | **Yes** | Yes |

Use **Option A** to try a custom architecture quickly. Use **Option B** when the model is ready to ship to all users.

---

## Option A: Quick Start — Custom Model File

Point the config at a Python file containing your model class. No modelzoo edits needed.

> **Requirement:** `training.custom_model_path` is handled by `tinyml-modelmaker`. Use a modelmaker release that
> includes it (a `pip install -e .` of this repository pulls in the released wheel, which may be older). If your
> run stops with `please check if the given model_name is a supported one: MyModel`, the installed modelmaker
> predates this option. Update it, or install modelmaker from source.

### Step A-1: Write your model file

Choose one style. The pipeline detects which style you used automatically.

**Style 1 — Plain `nn.Module` (simpler)**

```python
# /path/to/my_model.py
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, variables=1, num_classes=2, input_features=512, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(variables * input_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # input shape: (batch, variables, input_features, 1)
```

Constructor contract:

| Parameter | Meaning | Provided by pipeline |
|---|---|---|
| `variables` | Number of input channels | Yes |
| `num_classes` | Number of output classes | Yes |
| `input_features` | Temporal length of input | Yes |
| `**kwargs` | Forward-compat catch-all | — |

**Style 2 — `GenericModelWithSpec` subclass (matches built-in modelzoo style)**

Use this style to leverage built-in layer types (`BatchNormLayer`, `ConvBNReLULayer`, etc.) and layer-spec YAML serialization.

```python
# /path/to/my_model.py
from tinyml_modelzoo.models.base import GenericModelWithSpec
from tinyml_modelzoo.utils import py_utils

class MyModel(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=1, num_classes=2, **kwargs):
        # Must forward shape params to super so self.variables etc. are set
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes, **kwargs)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(
            model_spec=self.model_spec,
            variables=self.variables,
            num_classes=self.num_classes,
            input_features=self.input_features,
        )

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1': dict(type='ReshapeLayer', ndim=2)}
        layers += {'2': dict(type='LinearLayer', in_features=self.variables * self.input_features,
                             out_features=self.num_classes)}
        return dict(model_spec=layers)
```

Both styles must end with an output of shape `(batch, num_classes)`. In Style 2 that is why a `ReshapeLayer`
precedes the `LinearLayer`: without it the linear layer is applied to every time step and the output keeps the
time axis. Time series input arrives as `(batch, variables, input_features, 1)`.

Constructor receives `config=dict(variables=..., num_classes=..., input_features=..., dual_op=...)`. Shape params (`variables`, `num_classes`, `input_features`) must be declared in `__init__` and forwarded to `super().__init__()` — that's how `self.variables` etc. get set. See `tinyml_modelzoo/layers/tinynn.py` for available layer types.

### Step A-2: Update your config

```yaml
training:
  model_name: MyModel                          # must match class name exactly
  custom_model_path: /path/to/my_model.py      # absolute path recommended
```

This works for both styles — no other config change needed.

### Step A-3: Run as normal

```bash
./run_tinyml_modelzoo.sh /path/to/your_config.yaml     # Linux / macOS, from the tinyml-modelzoo folder
python -m tinyml_modelmaker.run_tinyml_modelmaker /path/to/your_config.yaml   # any install, any folder
```

The full pipeline runs: training → ONNX export → TVM compilation → MCU C code. The model is read from your file
at run time, so keep the file in place until the run has finished. The resolved path is recorded in the run's
`run.yaml` (`training.model_spec`).

What you get:

- With `training.quantization` unset (`0`, float only; `config_dsk.yaml` does not set it) the run produces the float
  model under `training/base/` and compiles it into `compilation/artifacts/` (`mod.a`, `tvmgen_default.h`), plus
  golden vectors and plots.
- With `quantization: 2` (quantization-aware training) the run also produces `training/quantization/`, and the
  quantized model is what gets compiled. Add the line under `training:` in your config.
- The NPU preset follows the quantization setting, not the model's origin. A float run (`quantization` unset)
  compiles with the **soft** NPU preset (compiler log: `ti-npu type: soft`). A `quantization: 2` run on an NPU
  device compiles for the **hardware NPU** (`ti-npu type: hard`, with `skip_normalize` and `output_int` on), as it
  does for the built-in models. Layers that break the NPU rules still fall back to the CPU, so follow
  [Guidelines for Designing Your Model](#guidelines-for-designing-your-model).

### Common errors

| Error message | Cause | Fix |
|---|---|---|
| `custom_model_path does not exist or is not a file` | Path typo or file not deployed | Use an absolute path |
| `BYOM: model 'MyModel' not found in '/path/to/my_model.py'` | Class name doesn't match `model_name` | Class name and `training.model_name` must be identical |
| `UserWarning: BYOM model 'MyModel' __init__ did not accept variables/...` | Plain `nn.Module` constructor ignores shape params | Add `variables=1, num_classes=2, input_features=512, **kwargs` to `__init__` |

> **Note:** If your file also defines a `get_model(model_name)` function, that takes priority over bare class discovery.

When your model is validated and you want it available to all users and the GUI, follow **Option B** below.

---

## Option B: Permanent Addition to ModelZoo

Edit files inside `tinyml-modelzoo`. **No changes are needed in
`tinyml-tinyverse` or `tinyml-modelmaker`**: models are discovered automatically
from `tinyml_modelzoo`.

## Files You Will Change

| # | File | What you add | Required? |
|---|------|--------------|-----------|
| 1 | [`tinyml_modelzoo/models/classification.py`](../tinyml_modelzoo/models/classification.py) | Your model class, plus its name in `__all__` | Yes |
| 2 | [`tinyml_modelzoo/model_descriptions/classification.py`](../tinyml_modelzoo/model_descriptions/classification.py) | A model description entry, plus its name in `enabled_models_list` | Yes |
| 3 | `examples/<example>/config_<your_model>.yaml` | A copy of an example config with `training.model_name` pointing to your model | Yes |

For other task types, use the matching files instead of `classification.py`:

| Task type | Model file (`tinyml_modelzoo/models/`) | Description file (`tinyml_modelzoo/model_descriptions/`) |
|-----------|---------------------------------------|----------------------------------------------------------|
| Time series classification | `classification.py` | `classification.py` |
| Time series regression | `regression.py` | `regression.py` |
| Time series anomaly detection | `anomalydetection.py` | `anomalydetection.py` |
| Time series forecasting | `forecasting.py` | `forecasting.py` |

### How the names connect

Two different names are involved. Keeping them apart avoids most "model not
found" errors:

```
config.yaml                      model_descriptions/classification.py        models/classification.py
training:                        'MY_CLS_2k_NPU': {                          class MY_CNN_TS_2K_NPU(...)
  model_name: MY_CLS_2k_NPU ───►     model_name='MY_CLS_2k_NPU',
                                     model_training_id='MY_CNN_TS_2K_NPU' ─►
```

- **Description name** (`MY_CLS_2k_NPU`): what you put in `training.model_name`
  in the config. It must be a key in `_model_descriptions` **and** be listed in
  `enabled_models_list`.
- **Class name** (`MY_CNN_TS_2K_NPU`): the Python class. It must be listed in the
  model file's `__all__` and match `model_training_id` exactly.

---

## Prerequisites

- Tiny ML ModelZoo is installed in editable mode (`pip install -e .` from the
  `tinyml-modelzoo` directory), so your edits take effect without reinstalling.
  Check that Python imports `tinyml_modelzoo` from your clone:

  ```bash
  python -c "import tinyml_modelzoo; print(tinyml_modelzoo.__file__)"
  # Expected: <your clone>/tinyml-modelzoo/tinyml_modelzoo/__init__.py
  ```

- You can already run an unmodified example, e.g.
  `./run_tinyml_modelzoo.sh examples/dc_arc_fault/config_dsk.yaml`.

All commands below are run from the `tinyml-modelzoo` directory.

---

## Step 1: Add the Model Class

Open `tinyml_modelzoo/models/classification.py` and add your class. The
simplest approach is to inherit from `GenericModelWithSpec` and describe the
network as an ordered list of layers. `GenericModelWithSpec` and `py_utils` are
already imported at the top of the file.

```python
class MY_CNN_TS_2K_NPU(GenericModelWithSpec):
    """
    My NPU-compliant classification model (~1.7K parameters).

    Architecture: BatchNorm -> Conv5x1(8) -> MaxPool -> Conv5x1(16) -> MaxPool
                  -> Conv3x1(16) -> AdaptiveAvgPool(4x1) -> FC
    """

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
        layers += {'2': dict(type='MaxPoolLayer', kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))}
        layers += {'3': dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(5, 1), stride=(1, 1))}
        layers += {'4': dict(type='MaxPoolLayer', kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))}
        layers += {'5': dict(type='ConvBNReLULayer', in_channels=16, out_channels=16, kernel_size=(3, 1), stride=(1, 1))}
        layers += {'6': dict(type='AdaptiveAvgPoolLayer', output_size=(4, 1))}
        layers += {'7': dict(type='ReshapeLayer', ndim=2)}
        layers += {'8': dict(type='LinearLayer', in_features=16 * 4, out_features=self.num_classes)}
        return dict(model_spec=layers)
```

Notes:

- Keep the `__init__(self, config, input_features=..., variables=..., num_classes=...)`
  signature. The training pipeline fills in `variables`, `input_features` and
  `num_classes` from your dataset and feature extraction settings, so do not
  hard-code them inside the layers.
- Time series input reaches the model as `(batch, variables, input_features, 1)`,
  which is why kernels are written as `(k, 1)`.
- `padding=(0, 0)` on the pooling layers matters. `MaxPoolLayer` pads by
  `kernel_size // 2` by default, which would turn 256 → 129 → 65 here. ONNX
  export of `AdaptiveAvgPoolLayer(output_size=(4, 1))` requires its input height
  to be a multiple of 4, so 65 fails. With no padding the height goes
  256 → 128 → 64. In general, `input_features` must be a multiple of 16 for
  this model (two halvings, then a multiple of 4).

### Alternative: write it as a plain PyTorch module

If you prefer standard PyTorch, you can write the same model as a
`torch.nn.Module` instead. The version below has the same layers, the same
1,700 parameters and produces the same ONNX graph as the spec-based version
above. Use **one** of the two, not both. `torch` is already imported at the top
of `models/classification.py`.

```python
class MY_CNN_TS_2K_NPU(torch.nn.Module):
    def __init__(self, config, input_features=512, variables=1, num_classes=2):
        super().__init__()
        variables = config.get('variables', variables)
        num_classes = config.get('num_classes', num_classes)
        self.features = torch.nn.Sequential(
            torch.nn.BatchNorm2d(variables),
            torch.nn.Conv2d(variables, 8, kernel_size=(5, 1), padding=(2, 0)),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            torch.nn.Conv2d(8, 16, kernel_size=(5, 1), padding=(2, 0)),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            torch.nn.Conv2d(16, 16, kernel_size=(3, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 1)),
        )
        self.classifier = torch.nn.Linear(16 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.reshape(x.shape[0], -1))
```

Differences from the spec-based version:

- The pipeline passes a `config` dict, so read `variables` and `num_classes`
  from it.
- Plain PyTorch layers do not pad by default, so each `Conv2d` sets
  `padding=(k // 2, 0)` to match `ConvBNReLULayer`. `MaxPool2d` already
  defaults to no padding, so it needs nothing extra.

Then add the class name to the `__all__` list at the bottom of the same file:

```python
__all__ = [
    # ... existing models ...
    'MY_CNN_TS_2K_NPU',
]
```

### Check it

Use the same `input_features` that your feature extraction produces (256 for
the `FFT1024Input_256Feature_1Frame_Full_Bandwidth` preset in `config_dsk.yaml`),
and include the ONNX export. The training pipeline exports to ONNX after float
training, so a model that runs in PyTorch but cannot be exported fails there.

```bash
python -c "
import torch
from tinyml_modelzoo.models import get_model, model_dict
assert 'MY_CNN_TS_2K_NPU' in model_dict
m = get_model('MY_CNN_TS_2K_NPU', variables=1, num_classes=2, input_features=256).eval()
x = torch.randn(1, 1, 256, 1)
print('output:', tuple(m(x).shape))
print('params:', sum(p.numel() for p in m.parameters()))
torch.onnx.export(m, x, '/tmp/my_model.onnx', dynamo=False)
print('onnx export: OK')
"
```

Expected: `output: (1, 2)`, `params: 1700` and `onnx export: OK`.

---

## Step 2: Add the Model Description

The config file refers to models by their **description name**, not their class
name. ModelMaker rejects any `model_name` that has no enabled description with:
`please check if the given model_name is a supported one: <name>`.

Open `tinyml_modelzoo/model_descriptions/classification.py` and add an entry
to the `_model_descriptions` dict. Start from `template_model_description`
(defined at the top of the file) and override only what is specific to your
model:

```python
    'MY_CLS_2k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='My NPU-compliant classification model with ~1.7k params.\n'
                          '3 Conv+BatchNorm+ReLU layers + Linear layer.',
        ),
        'training': dict(
            model_training_id='MY_CNN_TS_2K_NPU',   # must match the class name exactly
            model_name='MY_CLS_2k_NPU',             # must match the dict key
            target_devices=[
                constants.TARGET_DEVICE_F28P55,
                constants.TARGET_DEVICE_MSPM0G5187,
                constants.TARGET_DEVICE_F29H85,
                # ... any other device you plan to compile for
            ],
        ),
    }),
```

The template marks the model as a generic time series classification model.
It does not have to match the `task_type` of the example config
(`config_dsk.yaml` uses `task_type: arc_fault`). Generic classification models
such as `CLS_1k_NPU` are used across all classification examples in the same way.

Then add the description name to `enabled_models_list` at the bottom of the
same file:

```python
enabled_models_list = [
    # ... existing models ...
    '...last existing entry...',   # <-- make sure this line ends with a comma
    'MY_CLS_2k_NPU',
]
```

> **Watch the comma.** If the last existing entry has no trailing comma and you
> append your name on the next line, Python silently joins the two strings into
> one. Your model is then reported as "not supported" with no other hint.

### Check it

```bash
python -c "
from tinyml_modelzoo.model_descriptions.classification import get_model_description
d = get_model_description('MY_CLS_2k_NPU')
assert d is not None, 'not found: check _model_descriptions and enabled_models_list'
print(d['training']['model_training_id'])
"
```

Expected: `MY_CNN_TS_2K_NPU`.

---

## Step 3: Create a Config That Uses Your Model

Copy the example config instead of editing it, so the original stays intact:

```bash
cp examples/dc_arc_fault/config_dsk.yaml examples/dc_arc_fault/config_dsk_my_model.yaml
```

In the copy, change `training.model_name` to your description name:

```yaml
training:
  model_name: MY_CLS_2k_NPU     # was: CLS_1k_NPU
  model_config: ''
  batch_size: 32
  training_epochs: 10
  num_gpus: 0
  early_stopping: true
  early_stopping_patience: 5
```

Leave the other sections unchanged. See the
[Config YAML Guide](CONFIG_YAML_GUIDE.md) for what each section does.

---

## Step 4: Train, Quantize and Compile

```bash
./run_tinyml_modelzoo.sh examples/dc_arc_fault/config_dsk_my_model.yaml      # Linux
run_tinyml_modelzoo.bat examples\dc_arc_fault\config_dsk_my_model.yaml       # Windows
```

Near the top of the log, check that your model was picked up. The run folder
is named after the description name, and the model summary shows your class
name and parameter count:

```
INFO: __main__          :  Run params is at: .../run/<date-time>/MY_CLS_2k_NPU/run.yaml
...
MY_CNN_TS_2K_NPU                         [1, 2]                    --
...
Total params: 1,700
```

The run downloads the dataset (first run only), trains a float model, trains a
quantized model, tests the quantized ONNX model and compiles it for the
`target_device` in the config.

For reference, on a 16-core CPU machine with F28P55 as the target the full run
took about 14 minutes. Float training took seconds; almost all the time went to
quantized training.

A few warnings are expected and harmless, for example
`model_config () was provided but is not implemented` (from `model_config: ''`)
and DataLoader warnings about the number of worker processes.

Compilation uses the TI compiler toolchain. By default it is looked up under
`~/ti` (for F28P55: `~/ti/ti-cgt-c2000_<version>`). If you installed the
toolchains elsewhere, set `TOOLS_PATH` to that directory before running.

---

## Step 5: Check the Results

Artifacts are written under `./data/projects/<dataset_name>/run/<run_name>/`,
relative to the directory you ran from. `<run_name>` defaults to
`<date-time>/<model_name>`. Look for:

| Path (inside the run folder) | Contents |
|------------------------------|----------|
| `run.yaml` | The fully resolved parameters used for this run |
| `training/base/` | Float model: `run.log` (the training log for the whole run), `checkpoint.pth`, `model.onnx`, `golden_vectors/`, PCA plots of the extracted features, `file_level_classification_summary.log`, and a `.zip` that packages the final (quantized) model with its golden vectors and analysis |
| `training/quantization/` | Quantized model: `checkpoint.pth`, `model.onnx`, `golden_vectors/`, `post_training_analysis/` |
| `compilation/` | `run.log`, `artifacts/mod.a` and `artifacts/tvmgen_default.h` (the compiled model library), and a `.zip` of the compiled artifacts |

Two log sections summarize the result:

- `Test Data Evaluation Accuracy: ...`: accuracy of the quantized ONNX model
  on the test split.
- `TI Model Library Memory Usage (mod.a)`: code, RO data, RW data and total
  size of the compiled model on the target device.

To judge your model, compare both with a run of the original `config_dsk.yaml`
(`CLS_1k_NPU`).
Note that the DC arc fault example is an easy dataset where small models already
reach very high accuracy, so it is mainly useful to prove your model trains and
compiles end to end. Use your own dataset to judge accuracy.

> A failed run leaves its own timestamped folder next to successful ones. Make
> sure you are looking at the latest `<date-time>` folder.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `please check if the given model_name is a supported one: <name>` | `training.model_name` is not an enabled description name | Add the entry to `_model_descriptions` **and** to `enabled_models_list`, and make sure the config uses the description name, not the class name |
| `KeyError` / model not found during training | `model_training_id` does not match a registered class | Check spelling against the class name, and that the class is listed in `__all__` |
| `SyntaxError` / `IndentationError` traceback as soon as anything imports `tinyml_modelzoo.models` | Typo in your edit to `models/classification.py` | Fix the line shown in the traceback |
| `Could not import model module 'classification'` warning, and no classification models are found | Your edit to `models/classification.py` imports something that is not available | Run `python -c "import tinyml_modelzoo.models.classification"` to see the full `ImportError` |
| `Unsupported: ONNX export of operator adaptive_avg_pool2d, output size that are not factor of input size` after float training (often preceded by the warning `Memory pre-flight: could not find the NN model memory report`) | The height reaching `AdaptiveAvgPoolLayer` is not a multiple of its `output_size`, usually because of the default padding in `MaxPoolLayer` | Set `padding=(0, 0)` on the pooling layers, or choose `output_size` / `input_features` so the heights divide evenly. The ONNX check in [Step 1](#check-it) catches this before a full run |
| Shape mismatch at the `LinearLayer` | `in_features` does not equal channels × pooled height | Recompute: with `AdaptiveAvgPoolLayer(output_size=(4, 1))` and 16 channels, `in_features = 16 * 4` |
| Edits have no effect | Python imports an installed copy of `tinyml_modelzoo`, not your clone | Reinstall with `pip install -e .` from `tinyml-modelzoo` (see [Prerequisites](#prerequisites)) |

---

## Guidelines for Designing Your Model

### Run efficiently on the NPU

Devices with the TinyEngine NPU (e.g. F28P55, MSPM0G5187, AM13E2) only
accelerate layers that meet the NPU constraints. The most important rules are:

- **Channels in multiples of 4.** Every convolution output (and every input
  after the first layer) should be 4, 8, 12, 16, ...
- **First convolution.** With a single input channel (`variables: 1`), the first
  layer is an FCONV: input channels = 1 and output channels a multiple of 4.
- **Kernel sizes.** Use 1D kernels `(k, 1)` for time series. Split large
  kernels into several smaller ones rather than using one big kernel.
- **MaxPool.** At least one kernel dimension must be 1–4, e.g. `(2, 1)` or `(3, 1)`.
- **Fully-connected input.** The `LinearLayer` needs **at least 16 input
  features** for 8-bit weights (8 for 4-bit). Pool to `(4, 1)` rather than
  `(1, 1)` when you have few channels.
- **Global average pooling** needs more than 2 spatial elements at its input.

The full rules, with valid and invalid examples, are in the
[NPU Configuration Guidelines](NPU_CONFIGURATION_GUIDELINES.md). By convention,
models that follow them use an `_NPU` suffix in both names.

### Size the model for your device

| Parameters | Typical target |
|------------|----------------|
| 100 – 1K | Ultra-minimal MCUs |
| 1K – 4K | Small MCUs (Cortex-M0+) |
| 4K – 6K | Standard MCUs (Cortex-M4) |
| 6K – 13K | Larger MCUs (Cortex-R5) |
| 13K+ | Devices with more memory |

Flash and SRAM use grow with the parameter count, and activations grow with
`input_features`. Start small, then grow the model only if accuracy requires it.

### Keep the model compiler-friendly

- Prefer the spec layers (`ConvBNReLULayer`, `MaxPoolLayer`,
  `AdaptiveAvgPoolLayer`, `ReshapeLayer`, `LinearLayer`, ...) used by existing
  models. They export cleanly to ONNX and are known to quantize and compile.
- If you write a plain `torch.nn.Module` instead, stick to standard
  Conv / BatchNorm / ReLU / Pool / Linear layers and avoid data-dependent
  control flow in `forward()`.
- Leave `quantization` at its default (`2`, TI-NPU integer) for NPU devices;
  see [Useful Knobs](CONFIG_YAML_GUIDE.md#3-useful-knobs) for other options.

### Naming

- **Class name:** `SCREAMING_SNAKE_CASE`, with the model family and size,
  e.g. `MY_CNN_TS_2K_NPU`.
- **Description name:** short and readable, with the size,
  e.g. `MY_CLS_2k_NPU`. This is what users put in their config.

### Optional: show the model in the GUI

The description entry you added in Step 2 is also what Edge AI Studio uses.
To expose the model there with editable training parameters, add the
`properties` field, following existing entries such as `CLS_1k_NPU` in
`model_descriptions/classification.py`.

For more on model internals (available layer types, custom `torch.nn.Module`
models, test commands), see [ADDING_NEW_MODELS.md](https://github.com/TexasInstruments/tinyml-modelzoo/blob/main/ADDING_NEW_MODELS.md).
