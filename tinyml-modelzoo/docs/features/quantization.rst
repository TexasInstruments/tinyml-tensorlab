============
Quantization
============

Quantization reduces model precision from 32-bit floating point to 8-, 4- or 2-bit integers. It
shrinks the model, speeds up inference and is required for the TI NPU.

Why quantize?

* **Smaller models**: 4x reduction from float32 to int8, more with 4- and 2-bit weights.
* **Faster inference**: integer operations are faster, and the NPU only runs quantized models.
* **Lower power**: less memory traffic.

What is new in 1.5.0
====================

* **Residual connection support** in the TINPU quantization: ``add`` (and ``cat``/``stack``) inputs get
  matching observers and a wider clipping range, so models with skip connections (ResNet style) can
  be quantized for the NPU.
* **Cleaner API**: the old ``qconfig_dict`` was replaced by a ``TinyMLQConfigType`` object. Auto-quantization is
  toggled with a single ``auto_quantization`` flag (``training.auto_quantization`` in the config YAML, on by default).
  See :doc:`/features/auto_quantization`.
* **Hessian-aware (HAWQ) automatic mixed precision** and **ternary-weight / 8-bit-activation QAT**:
  see :doc:`/features/mixed_precision_quantization`.
* **NPU quantization for anomaly detection models**: ``quantization: 2`` now works for the
  anomaly detection task type (previously these models were generic quantization only).
* Weight bias width is chosen automatically: 24-bit for 8-bit weights and 16-bit for 2/4-bit weights.

Configuration parameters
========================

Quantization is controlled from the ``training`` section of the config YAML:

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Option
     - Values
     - Description
   * - ``quantization``
     - ``0``, ``1``, ``2``
     - ``0`` = floating point training (no quantization). ``1`` = standard PyTorch (generic) quantization.
       ``2`` = TI style quantization for the NPU (TINPU).
   * - ``quantization_method``
     - ``'PTQ'``, ``'QAT'``
     - Post training quantization or quantization aware training (default ``QAT``).
   * - ``quantization_weight_bitwidth``
     - ``8``, ``4``, ``2``
     - Weight bit width (default 8).
   * - ``quantization_activation_bitwidth``
     - ``8``, ``4``, ``2``
     - Activation bit width (default 8).
   * - ``auto_quantization``
     - ``True``, ``False``
     - Default ``True``: bit widths are chosen automatically per layer and the two bit width keys above
       are overridden. Set ``False`` to use exactly the bit widths you specify.

.. note::

   The method and bit width keys apply only when ``quantization`` is ``1`` or ``2``. For models whose
   description in the ModelZoo is an NPU model (names ending in ``_NPU``), the default is already
   TINPU quantization; you only need to set these keys to change the behavior.

Quantization modes
==================

**Floating point (quantization: 0)**

.. code-block:: yaml

   training:
     model_name: 'CLS_4k_NPU'
     quantization: 0

**Generic PyTorch quantization (quantization: 1)** uses the standard PyTorch quantization APIs
(``GenericTinyMLQATFxModule`` / ``GenericTinyMLPTQFxModule``) and suits CPU deployment:

.. code-block:: yaml

   training:
     model_name: 'CLS_4k'
     quantization: 1
     quantization_method: 'QAT'
     auto_quantization: False
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

**TI style quantization (quantization: 2)** uses ``TINPUTinyMLQATFxModule`` / ``TINPUTinyMLPTQFxModule`` and
applies the constraints of the NPU accelerator. It is required for NPU deployment:

.. code-block:: yaml

   common:
     target_device: 'F28P55'

   training:
     model_name: 'CLS_4k_NPU'
     quantization: 2
     quantization_method: 'QAT'
     auto_quantization: False
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

Quantization methods
====================

* **PTQ** quantizes the trained float model after training using a calibration pass. It is fast and needs no
  retraining, but may lose accuracy on some models.
* **QAT** simulates quantization during training, which retains accuracy better but takes longer.

Bit widths
==========

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Weights
     - Size vs float32
     - Notes
   * - 8-bit
     - 4x smaller
     - Most common choice. Power-of-2 scales.
   * - 4-bit
     - 8x smaller
     - Soft-sigmoid quantizer during QAT.
   * - 2-bit
     - 16x smaller
     - Ternary (-1, 0, +1) weights through DBQ quantizer; see :doc:`/features/mixed_precision_quantization`.

Weight and activation bit widths are set independently, for example 4-bit weights with 8-bit activations.
Accuracy loss depends on the model and the task; always compare the float and quantized results in the
run output.

NPU constraints
===============

The NPU accelerates FCONV, GCONV, DWCONV, PWCONV, FC, AVGPOOL and MAXPOOL layers, each followed by a
BNORM sequence (bias, scale, shift, floor, clip). Channel counts must satisfy the NPU rules (multiples of 4
for most layers, and exactly 1 input channel for the first convolution); see :doc:`/NPU_CONFIGURATION_GUIDELINES`.

* Weights are quantized per channel, symmetric. Activations are per tensor, symmetric.
* Layers that do not meet the NPU constraints fall back to the CPU. Use ``quantization: 1`` for models that cannot satisfy them.

Output files
============

The quantized stage writes its results next to the float training output of the run:

.. code-block:: text

   .../<model_name>/training/
   |-- base/                       # float model
   |   |-- checkpoint.pth
   |   |-- model.onnx
   |   `-- memory_preflight/
   `-- quantization/               # quantized model
       |-- checkpoint.pth
       |-- model.onnx              # quantized ONNX, input for compilation
       |-- model_aux.h
       |-- golden_vectors/
       |-- memory_preflight/
       `-- post_training_analysis/ # see Post Training Analysis

See :doc:`/features/post_training_analysis` and :doc:`/features/memory_estimation`.

Troubleshooting accuracy loss
=============================

1. Use QAT instead of PTQ.
2. Use higher bit widths. Start at 8-bit and go lower only when the model needs to be smaller.
3. Keep activations at 8-bit with lower bit weights.
4. Leave ``auto_quantization: True`` (default) and tune the ``autoquant_tolerance_*`` keys
   (see :doc:`/features/auto_quantization`).
5. Increase the model size; a larger model usually tolerates quantization better.

QAT training performance
========================

QAT is slower than float training. Each layer gets ``FakeQuantize`` nodes with observers that run on
every batch, 4-bit and 2-bit weights use extra soft-quantization passes, and the QAT wrapper
traverses the module tree every epoch. If QAT time matters, reduce ``training_epochs`` for the quantized
stage or use PTQ. See :doc:`/features/training_performance` for the other training speed options.

Using the wrappers directly
===========================

The quantization wrappers can be used in your own PyTorch training scripts through
``tinyml_torchmodelopt`` (installed with the ModelZoo).

.. code-block:: python

   from tinyml_torchmodelopt.quantization import TINPUTinyMLQATFxModule
   from tinyml_torchmodelopt.quantization.common import TinyMLQConfigType

   model = MyNeuralNetwork()
   model.load_state_dict(torch.load('pretrained.pth'))
   example_inputs = torch.rand((1, 1, 256, 1))

   qconfig_type = TinyMLQConfigType(weight_bitwidth=8, activation_bitwidth=8,
                                    auto_quantization=False)
   model = TINPUTinyMLQATFxModule(model, example_inputs=example_inputs,
                                  qconfig_type=qconfig_type, total_epochs=epochs)

   model.train()
   for e in range(epochs):
       for images, target in train_loader:
           output = model(images)
           # loss, backward(), optimizer step as usual

   model.eval()
   model = model.convert()                   # integer operations
   model.export(example_inputs, 'model_int8.onnx', input_names=['input'])

For PTQ replace the QAT class with ``TINPUTinyMLPTQFxModule`` and run only a calibration pass
(``model.eval()`` with ``torch.no_grad()``) before ``convert()``. ``GenericTinyMLQATFxModule`` and
``GenericTinyMLPTQFxModule`` have the same API for generic quantization. To pass explicit
per-layer bit widths use ``weight_mixed_precision`` of ``TinyMLQConfigType`` (a dict of
bit width to a list of layer names) with ``auto_quantization=False``.

Wrapper arguments
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Argument
     - Type
     - Description
   * - ``model``
     - nn.Module
     - Model to quantize.
   * - ``total_epochs``
     - int
     - Total number of quantized training epochs (also drives soft-quantization temperatures).
   * - ``qconfig_type``
     - TinyMLQConfigType
     - Bit widths, auto-quantization flag and optional mixed precision (replaces the old ``qconfig_dict``).
   * - ``example_inputs``
     - Tensor
     - Example input with batch size 1.
   * - ``num_batch_norm_update_epochs``
     - bool/int/None
     - ``None`` (default) freezes BatchNorm statistics at the midpoint of training; ``False`` never
       freezes; an integer freezes after that epoch.
   * - ``num_observer_update_epochs``
     - bool/int/None
     - Epoch after which observers freeze.
   * - ``bias_calibration_factor``
     - float
     - Bias calibration factor (0.0 disables it).
   * - ``output_int``
     - bool
     - Quantized (integer) or dequantized output.
   * - ``verbose``
     - bool
     - Verbose logging.

More examples are in the ``torchmodelopt/examples`` folder of the tinyml-modeloptimization package.

Next steps
==========

* :doc:`/features/quantization_examples`
* :doc:`/features/auto_quantization`
* :doc:`/features/mixed_precision_quantization`
* :doc:`/features/feature_extraction`
