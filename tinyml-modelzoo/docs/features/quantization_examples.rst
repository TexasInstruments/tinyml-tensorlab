================================
Standalone Quantization Examples
================================

The ``tinyml-modeloptimization`` package ships standalone Python examples that use the quantization
wrappers directly, without the YAML-based toolchain. They are in the ``torchmodelopt/examples`` folder
of the tinyml-modeloptimization source package.

.. note::

   These examples are for users who want to add quantization to their own PyTorch training scripts.
   Most users should use the YAML flow described in :doc:`/features/quantization`.

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Example folder
     - Dataset
     - Highlights
   * - ``fmnist_image_classification``
     - Fashion MNIST
     - Beginner example: LinearReLU model, QAT, TI NPU export, node renaming, ONNX inference.
   * - ``audio_keyword_spotting``
     - Speech Commands
     - DSCNN model, PTQ, mixed precision, bias calibration.
   * - ``motor_fault_time_series_classification``
     - Motor fault CSV
     - CNN on time series, QAT/PTQ, 2/4/8-bit weights, ONNX inference.
   * - ``mnist_lenet5_classification``
     - MNIST
     - LeNet-5, QAT/PTQ, 4/8-bit quantization.
   * - ``torque_time_series_regression``
     - Torque measurement CSV
     - CNN regression, QAT, R2 and SMAPE metrics.

Common workflow
===============

All examples follow the same steps. In 1.5.0 the wrappers take a ``TinyMLQConfigType`` and an example input:

.. code-block:: python

   from tinyml_torchmodelopt.quantization import TINPUTinyMLQATFxModule
   from tinyml_torchmodelopt.quantization.common import TinyMLQConfigType

   # 1. train the float model as usual, then
   example_inputs = torch.rand((1, 1, 256, 1))
   qconfig_type = TinyMLQConfigType(weight_bitwidth=8, activation_bitwidth=8,
                                    auto_quantization=False)
   ti_model = TINPUTinyMLQATFxModule(model, example_inputs=example_inputs,
                                     qconfig_type=qconfig_type, total_epochs=5)

   # 2. QAT training loop (same loss and optimizer as float)
   ti_model.train()
   for epoch in range(5):
       for images, targets in train_loader:
           loss = criterion(ti_model(images), targets)
           loss.backward()
           optimizer.step()

   # 3. convert to integer operations and export
   ti_model.eval()
   ti_model = ti_model.convert()
   ti_model.export(example_inputs, 'model_int8.onnx', input_names=['input'])

Run an example from its folder, for example:

.. code-block:: bash

   cd torchmodelopt/examples/fmnist_image_classification
   python fmnist_tinpu_qat.py

The time series examples (motor fault, torque regression) are configured by editing constants at the top
of the script: the quantization method (``QAT`` or ``PTQ``), the weight and activation bit widths and the
device type (``TINPU`` or ``GENERIC``). The audio keyword spotting example passes per-layer
``weight_mixed_precision`` to ``TinyMLQConfigType`` (see :doc:`/features/mixed_precision_quantization`).

Guidance
========

**QAT versus PTQ.** PTQ with 8-bit weights and activations is a safe, fast default because it needs only a
calibration pass. Switch to QAT when PTQ loses accuracy, especially below 8 bits, using a small learning rate
and more epochs.

**Sub-8-bit.** Prefer QAT with per-channel weight quantization. The NPU prefers symmetric per-channel weights.

**PTQ calibration.** Use representative inputs (hundreds to a few thousand samples) that cover the full input
distribution; poor calibration causes large activation errors.

**Validate the ONNX model.** Check the exported model with ONNX Runtime before deploying:

.. code-block:: python

   import onnxruntime as ort

   session = ort.InferenceSession('model_int8.onnx')
   prediction = session.run(None, {'input': test_input.numpy()})

Next steps
==========

* :doc:`/features/quantization`
* :doc:`/features/auto_quantization`
