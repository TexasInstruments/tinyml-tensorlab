================
Compilation Only
================

If you already have a trained model in ONNX format, for example exported from PyTorch or another framework, you can
compile it for a TI device with ModelMaker and skip the dataset and training stages.

The flow is:

1. Train (and, for NPU devices, quantize) the model outside ModelZoo.
2. Export it to ONNX.
3. Run ModelMaker with only the compilation stage enabled and ``compilation.model_path`` pointing at the ONNX file.
4. Deploy the compiled artifacts to the device (see :doc:`/deployment/index`).

Use it when you have a model from another framework or training environment, or an architecture you do not want to
add to the ModelZoo. If you want ModelZoo to train and quantize your model, use :doc:`/BRING_YOUR_OWN_MODEL` instead.

Configuration
=============

Disable the dataset and training stages and give the path of the ONNX file under ``compilation``:

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'
     run_name: '{date-time}/{model_name}'

   dataset:
     enable: False           # no dataset needed
     dataset_name: my_model  # only used to name the output folders

   training:
     enable: False           # no training
     model_name: 'my_model'  # only used to name the output folders

   compilation:
     enable: True
     model_path: '/path/to/your/model.onnx'

When ``compilation.model_path`` points to an existing file, ModelMaker compiles that file. Otherwise it looks for the
``model.onnx`` that a training stage of the same run produced.

Quantized models and NPU devices
--------------------------------

For an NPU device the compiler needs a model quantized for the NPU. The compile target keeps the hard NPU mode only
when ``training.quantization`` is ``2`` (TINPU); for any other value the target is switched to the software NPU. If your
ONNX file is already an NPU-quantized model, set the level so that the NPU target is kept:

.. code-block:: yaml

   training:
     enable: False
     model_name: 'my_model'
     quantization: 2

A float model compiles for the software path and is not accelerated by the NPU. Quantize it first, for example with
the wrappers described in :doc:`/features/quantization`.

Running the compilation
=======================

Run the config like any other, from the ``tinyml-modelzoo`` folder (see :doc:`/getting_started/running_examples`):

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh byom_config.yaml

   .. tab:: Windows (cmd)

      .. code-block:: bat

         run_tinyml_modelzoo.bat byom_config.yaml

   .. tab:: Windows (PowerShell)

      .. code-block:: powershell

         ./run_tinyml_modelzoo.ps1 byom_config.yaml

The compiled artifacts are written to the ``compilation`` folder of the run
(``./data/projects/<project_name>/run/<date-time>/<model_name>/compilation/``). It contains the compiled library
``mod.a`` and the header ``tvmgen_default.h`` in ``artifacts/``, and a zip package of the compilation output.

.. note::

   The compile-only flow needs the compiler tools to be installed (see :doc:`/installation/index`). The golden vectors
   used by the deployment guides come from a training run, and a compile-only run does not create them. See
   :doc:`/DEPLOYING_TO_SDK_EXAMPLE`. This page describes what the 1.5.0 ModelMaker code supports; the flow was not
   re-run end to end for this guide.

Example: external PyTorch model
===============================

Export the trained model to ONNX with a batch size of 1:

.. code-block:: python

   import torch

   model.eval()
   dummy_input = torch.randn(1, 1, 512, 1)   # (batch, variables, features, 1)
   torch.onnx.export(model, dummy_input, 'my_model.onnx',
                     input_names=['input'], output_names=['output'],
                     opset_version=18)

The ModelZoo training scripts export with ONNX opset 18 by default, so using the same opset avoids surprises. Then use the
configuration above with ``model_path: 'my_model.onnx'``. A TensorFlow model can be converted to ONNX first (for
example with ``tf2onnx``) and compiled the same way.

NPU constraints
===============

For NPU devices the ONNX model must follow the NPU rules (channel counts that are multiples of 4, limits on kernel and
pooling sizes). See :doc:`/NPU_CONFIGURATION_GUIDELINES`. If the model does not meet them you can modify and retrain it,
target a device without NPU, or accept that those layers run on the CPU.

Troubleshooting
===============

* **Unsupported operator**: replace it with a supported operation or simplify the model.
* **Input shape mismatch**: check the input shape of the ONNX model against what the application feeds it.
* **Model not accelerated on an NPU device**: the model is not NPU-quantized, or ``training.quantization`` is not ``2``.

Best practices
==============

1. Test the ONNX model with ONNX Runtime before compiling.
2. Start by compiling for a device without NPU to debug the model.
3. Compare the outputs of the compiled model with the original.

Next steps
==========

* :doc:`adding_models`
* :doc:`/BRING_YOUR_OWN_MODEL`
* :doc:`/deployment/index`
