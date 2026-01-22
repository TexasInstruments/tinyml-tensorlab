======================
Bring Your Own Model
======================

This section explains how to extend Tiny ML Tensorlab with custom models
or use pre-trained models from external sources.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   adding_models
   compilation_only

Two Approaches
--------------

**1. Add a Custom Model to ModelZoo**

If you want to create a new neural network architecture that integrates
fully with the training pipeline:

* Add your model class to ``tinyml_modelzoo/models/``
* Register it in the module's ``__all__`` list
* Use it like any built-in model

See :doc:`adding_models` for step-by-step instructions.

**2. Compile an Existing ONNX Model**

If you already have a trained model (from PyTorch, TensorFlow, etc.)
and just want to compile it for TI MCUs:

* Export your model to ONNX format
* Use Tiny ML Tensorlab's compilation-only mode
* Skip the dataset and training steps

See :doc:`compilation_only` for instructions.

Model Requirements
------------------

**For NPU Devices** (F28P55, AM13E2, MSPM0G5187):

Models must follow NPU constraints for hardware acceleration:

* All channels must be multiples of 4
* Convolution kernel heights ≤ 7 for GCONV layers
* MaxPool kernels ≤ 4
* FC layer inputs ≥ 16 features (8-bit) or ≥ 8 features (4-bit)

See :doc:`/devices/npu_guidelines` for complete constraints.

**For Non-NPU Devices:**

More flexible architecture choices, but consider:

* Total parameter count (memory constraints)
* Layer types supported by the compiler
* Inference time requirements
