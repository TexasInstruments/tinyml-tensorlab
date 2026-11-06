=============
Common Errors
=============

Each entry gives the symptom (message text as produced by the code), the cause and the fix. Messages marked
"from a 1.5.0 test run" were captured from the ModelMaker functional test logs; the others were checked against the
source code.

.. contents::
   :local:
   :depth: 1

Setup and configuration
-----------------------

.. _err-unknown-model:

Unknown model name
^^^^^^^^^^^^^^^^^^

**Symptom**

.. code-block:: text

   ValueError: could not find pretrained model for <model_name>

**Cause** ``training.model_name`` is not in the ``enabled_models_list`` of any model description. Only the listed
models are selectable, so an older 1.4.0 name, a typo, or a model you defined but did not enable gives this error. The
lookup is case sensitive.

**Fix** Use a name from :doc:`/appendix/model_zoo_reference`. For your own model either set
``training.custom_model_path`` to a ``.py`` file that defines a class with that name, or add it to the description
file and to ``enabled_models_list``, as described in :doc:`/BRING_YOUR_OWN_MODEL`. If you already set
``custom_model_path`` and still get ``please check if the given model_name is a supported one``, the installed
``tinyml-modelmaker`` predates that option. A related error is
``task_type: <x> does not match the pretrained model``: the model belongs to a different task than
``common.task_type``.

.. _err-custom-model:

Custom model file (``custom_model_path``) errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**

.. code-block:: text

   training.custom_model_path does not exist or is not a file: /path/to/my_model.py
   ValueError: BYOM: model 'MyModel' not found in '/path/to/my_model.py'. Define a class named 'MyModel' or a get_model() function in that file.
   UserWarning: BYOM model 'MyModel' __init__ did not accept variables/num_classes/input_features kwargs. Instantiating with no arguments.

**Cause** The first message means the path is wrong (use an absolute path). The second means the file has no class
whose name equals ``training.model_name`` (the match is exact and case sensitive). The warning means a plain
``torch.nn.Module`` has an ``__init__`` that does not take ``variables``, ``num_classes`` and ``input_features``, so it was
built with no arguments and the model will not fit your data.

**Fix** Correct the path or the class name. Declare
``def __init__(self, variables=1, num_classes=2, input_features=512, **kwargs)``. The model must return
``(batch, num_classes)``. If the run instead stops with
``please check if the given model_name is a supported one: <name>``, the installed ``tinyml-modelmaker`` predates
``custom_model_path``; update it. See :doc:`/BRING_YOUR_OWN_MODEL`.

.. _err-python-311:

Python 3.11 or newer cannot compile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom** Training runs, but the compilation step fails with an import error for the TI compiler package
(``ti_mcu_nnc`` / ``tvm``), or ``pip`` did not install it.

**Cause** ``ti_mcu_nnc`` is only published for Python 3.10 (``cp310`` wheels). The dependency has the marker
``python_version < '3.11'`` in ``tinyml_modelmaker/pyproject.toml``, so on Python 3.11 to 3.14 everything else installs
(``requires-python = ">=3.10,<3.15"``) but the compiler does not.

**Fix** Create a Python 3.10 environment for the whole flow (see :doc:`/installation/index`). If you only need training and ONNX export, newer
Python versions work; set ``compilation.enable: False``.

.. _err-no-examples:

No ``examples/`` folder after a wheel install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom** ``./run_tinyml_modelzoo.sh examples/...`` or ``examples/<name>/config.yaml`` does not exist.

**Cause** The wheel (``pip install ...tinyml_modelzoo-1.5.0-py3-none-any.whl``) installs the Python package only. The
examples and the ``run_tinyml_modelzoo`` scripts are in the Git repository.

**Fix** Either clone ``https://github.com/TexasInstruments/tinyml-modelzoo`` and run ``pip install -e .`` there, or run
your own YAML directly with the ModelMaker entry point:

.. code-block:: bash

   python -m tinyml_modelmaker.run_tinyml_modelmaker config.yaml

See :doc:`/installation/user_installation`.

.. _err-compiler-tools:

Compiler tools not found (``TOOLS_PATH``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom** Training and testing succeed, then compilation fails because the cross compiler cannot be run
(for example the ``cl2000``, ``c29clang`` or ``tiarmclang`` binary is missing). The exact text comes from the
compiler step and is not produced by ModelZoo code, so it varies.

**Cause** ModelMaker looks for the TI code generation tools in ``$TOOLS_PATH`` (default ``~/ti``). The folder names
expected in 1.5.0 are ``ti-cgt-c2000_25.11.1.LTS`` (C28x), ``ti-cgt-c29_2.2.1.LTS`` (C29x) and
``ti-cgt-armllvm_5.1.1.LTS`` (MSPM0 and other Arm targets); see ``ai_modules/timeseries/constants.py``. Each root can be
overridden with ``C2000_CG_ROOT``, ``CG_TOOL_ROOT`` and ``ARM_LLVM_CGT_PATH``.

**Fix**

.. code-block:: bash

   export TOOLS_PATH=$HOME/ti        # folder that contains the ti-cgt-* folders
   ls $TOOLS_PATH

Install the compiler for your device family and use the exact versions above. See
:doc:`/installation/environment_variables`.

Training
--------

.. _err-nan-loss:

NaN or infinite loss
^^^^^^^^^^^^^^^^^^^^

**Symptom** (from a 1.5.0 test run, motor bearing fault anomaly detection for MSPM0)

.. code-block:: text

   RuntimeError: Training encountered NaN/Inf loss at Epoch 0. This indicates training instability.
   Common causes and fixes:
     1. Learning rate might be too high, try reducing learning_rate
     2. Batch size might be too large, try reducing batch_size

**Cause** The loss became ``NaN`` or ``Inf`` during training. 1.5.0 stops immediately instead of continuing.
Typical causes are a too high learning rate, unnormalized inputs, or a feature extraction chain that overflows.

**Fix** Lower ``training.learning_rate``, reduce ``training.batch_size``, check the dataset values and the feature
extraction settings (:doc:`/features/feature_extraction`).

.. _err-early-stopping:

Training stops before all epochs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**

.. code-block:: text

   Early stopping at epoch <N>: no improvement for <patience> epochs.

**Cause** Early stopping is on by default in 1.5.0 (``training.early_stopping: True``, patience 5, min delta 0.0).
This is not an error.

**Fix** To train all epochs set ``early_stopping: False``; to be less strict increase
``early_stopping_patience`` or set a small ``early_stopping_min_delta``. See :doc:`/features/early_stopping`.

.. _err-onnx-adaptive:

ONNX export fails at AdaptiveAvgPool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom** ONNX export stops with a PyTorch message of the form
``Unsupported: ONNX export of operator adaptive_avg_pool2d, output size that are not factor of input size``
(wording depends on the PyTorch version).

**Cause** In a custom model, ``AdaptiveAvgPool2d`` (``AdaptiveAvgPoolLayer`` in the spec layers) can only be
exported when the input height is a multiple of the output size. For example, with ``output_size=(4, 1)`` the input
height must be a multiple of 4. This often happens when pooling layers pad the height (for example 256 to 129 to 65).

**Fix** Use ``padding=(0, 0)`` on the pooling layers or change the layer sizes so the height stays a multiple of
the output size. Worked example: :doc:`/BRING_YOUR_OWN_MODEL`.

.. _err-qat-scale:

QAT: scale and zero-point dimension error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom** (from a 1.5.0 test run, DC arc fault anomaly detection ``config_anomaly_detection_dsk.yaml``)

.. code-block:: text

   RuntimeError: dimensions of scale and zero-point are not consistent with input tensor

The traceback ends in ``tinyml_torchmodelopt/quantization/base/fx/fake_quant_types.py`` (``forward``), calling
``torch.fake_quantize_per_channel_affine``.

**Cause** Known 1.5.0 failure in quantization-aware training of the anomaly detection model used by that config. It
was observed in test logs; the root cause was not analyzed here.

**Workaround** Train the float model and quantize later with PTQ (``training.quantization_method: PTQ``), or set
``training.quantization: 0``. Check the release notes of a newer version for a fix.

.. _err-cuda-qat:

QAT: tensors on different devices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom** (from a 1.5.0 test run, generic anomaly detection with an NPU model)

.. code-block:: text

   RuntimeError: Expected all tensors to be on the same device, but got scale is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA__fake_quantize_per_channel_affine_cachemask)

**Cause** Known 1.5.0 failure: the quantization scale tensor stays on the CPU while the model runs on CUDA during QAT.

**Workaround** Run on CPU with ``training.num_gpus: 0`` (and ``training_device: cpu``). On a CUDA machine this also stops the
automatic ``compile_model`` / ``native_amp`` switch-on.

.. _err-zip:

Dataset download: File is not a zip file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom** (from a 1.5.0 test run, ``hvac_indoor_temp_forecast``)

.. code-block:: text

   zipfile.BadZipFile: File is not a zip file

raised in ``utils.extract_files`` (``download_utils.py``).

**Cause** The file downloaded to ``common.download_path`` (default ``./data/downloads``) is not a valid zip, for
example an HTML error page from a proxy, a failed or partial download, or a wrong URL.

**Fix** Delete the file in ``data/downloads``, check the network or proxy, and re-run. For a local dataset set
``dataset.input_data_path`` to the extracted folder.

.. _err-list-index:

``IndexError: list index out of range`` when loading the dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom** (from a 1.5.0 test run, ``bms_soc_estimation`` for MSPM0)

.. code-block:: text

   list_to_load = glob(joined_path)[0]
   IndexError: list index out of range

**Cause** The dataset loader looks for a file list (train/validation/test list) with ``glob`` and found none. The
dataset folder does not have the expected layout, or the split lists were not created.

**Fix** Check the dataset layout for your task (:doc:`/byod/index`) and that ``dataset.dataset_path`` or
``input_data_path`` points to the right folder. Delete a half-created ``data/projects/<project>/dataset`` folder and
re-run.

Compilation and model design
----------------------------

.. _err-npu-constraints:

NPU layer constraint violations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom** Compilation for an NPU device (F28P55, AM13E2, MSPM0G5187) fails or falls back to the software path with a
message from the TI neural network compiler about a layer that is not supported by the NPU. The exact text comes
from the compiler and depends on its version.

**Cause** NPU models must follow layer rules: channels in multiples of 4, kernel height limits, fully connected
input of at least 16 features (8-bit), and others.

**Fix** Use a model whose name ends in ``_NPU`` (see :doc:`/appendix/model_zoo_reference`), or adapt your model with
:doc:`/NPU_CONFIGURATION_GUIDELINES`. To compile an NPU device model with the software NPU, use
``compilation.compile_preset_name: forced_soft_npu_preset``.

.. _err-golden-vector:

Golden vector mismatch on the device
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom** The output of the model in the SDK example differs from the golden vectors generated by the run.

**Cause** Usual reasons: the four generated files (model, feature extraction, test vectors) come from different runs,
the SDK example uses different feature extraction settings or input scaling, or the quantization output type
(``output_int``) does not match the code reading the result.

**Fix** Copy all generated files from the same run folder and follow :doc:`/DEPLOYING_TO_SDK_EXAMPLE`. Check the
deployment pages for your task, for example :doc:`/deploying_classification_models_from_modelmaker_to_device/readme`.

Platform
--------

.. _err-mps:

macOS (Apple Silicon, MPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom** Training on a Mac is slow, or an operator is reported as not implemented for the ``MPS`` device.

**Cause** The MPS backend does not implement every operator. ModelMaker sets ``PYTORCH_ENABLE_MPS_FALLBACK=1`` by default
(``run_tinyml_modelmaker.py``), so unsupported operators run on the CPU, which can be slower. ``torch.compile`` on MPS
uses the ``aot_eager`` backend.

**Fix** If MPS causes trouble, set ``training.training_device: cpu``. Update to the latest PyTorch that the wheel
allows. See :doc:`/installation/macos_setup`.

Other tips
----------

* ``ModuleNotFoundError`` for ``tinyml_modelmaker``, ``tinyml_tinyverse`` or ``tinyml_torchmodelopt``: reinstall with
  ``pip install -e .`` from the ``tinyml-modelzoo`` folder, or reinstall the wheel. No other repository needs to be
  installed. The older multi-repository install is no longer used.
* Out-of-memory on GPU: reduce ``training.batch_size``, or run with ``num_gpus: 0``.
