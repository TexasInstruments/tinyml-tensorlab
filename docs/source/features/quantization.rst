============
Quantization
============

Quantization reduces model precision from 32-bit floating point to lower
bit widths (8-bit, 4-bit, or 2-bit integers), dramatically reducing model
size and improving inference speed.

Overview
--------

Why quantize?

* **Smaller models**: 4x reduction (float32 → int8)
* **Faster inference**: Integer operations are faster
* **NPU requirement**: TI's NPU requires quantized models
* **Lower power**: Reduced memory bandwidth

Configuration Parameters
------------------------

Quantization in Tiny ML Tensorlab is controlled by four parameters in the
``training`` section of the config YAML:

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Option
     - Values
     - Description
   * - ``quantization``
     - ``0``, ``1``, ``2``
     - Quantization mode. ``0`` = floating point training (no quantization).
       ``1`` = standard PyTorch Quantization. ``2`` = TI style optimised
       Quantization.
   * - ``quantization_method``
     - ``'PTQ'``, ``'QAT'``
     - Quantization method. Only applicable when ``quantization`` is ``1``
       or ``2``.
   * - ``quantization_weight_bitwidth``
     - ``8``, ``4``, ``2``
     - Bit width for weight quantization. Only applicable when
       ``quantization`` is ``1`` or ``2``.
   * - ``quantization_activation_bitwidth``
     - ``8``, ``4``, ``2``
     - Bit width for activation quantization. Only applicable when
       ``quantization`` is ``1`` or ``2``.

.. note::

   ``quantization_method``, ``quantization_weight_bitwidth``, and
   ``quantization_activation_bitwidth`` are only used when ``quantization``
   is set to ``1`` or ``2``. When ``quantization`` is ``0`` (floating point
   training), these parameters have no effect.

Quantization Modes
------------------

**Floating Point Training (quantization: 0)**

Standard float32 training with no quantization applied:

.. code-block:: yaml

   training:
     model_name: 'CLS_4k_NPU'
     quantization: 0

**Standard PyTorch Quantization (quantization: 1)**

Uses standard PyTorch quantization APIs (``GenericTinyMLQATFxModule`` /
``GenericTinyMLPTQFxModule``). Suitable for general-purpose CPU deployment:

.. code-block:: yaml

   training:
     model_name: 'CLS_4k'
     quantization: 1
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

For more details on the underlying wrappers, see the
`tinyml-modeloptimization documentation <https://bitbucket.itg.ti.com/projects/TINYML-ALGO/repos/tinyml-modeloptimization/browse/torchmodelopt/README.md>`_.

**TI Style Optimised Quantization (quantization: 2)**

Uses TI's NPU-optimised quantization (``TINPUTinyMLQATFxModule`` /
``TINPUTinyMLPTQFxModule``). This incorporates the constraints of TI NPU
Hardware accelerator and is required for NPU deployment:

.. code-block:: yaml

   training:
     model_name: 'CLS_4k_NPU'
     quantization: 2
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

Quantization Methods
--------------------

**Post-Training Quantization (PTQ)**

Quantizes a trained float model after training:

.. code-block:: yaml

   training:
     model_name: 'CLS_4k_NPU'
     quantization: 2
     quantization_method: 'PTQ'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

* Pros: Fast, simple, no retraining required
* Cons: May lose accuracy for some models

**Quantization-Aware Training (QAT)**

Simulates quantization during training for better accuracy retention:

.. code-block:: yaml

   training:
     model_name: 'CLS_4k_NPU'
     quantization: 2
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

* Pros: Better accuracy retention
* Cons: Longer training time

Bit Widths
----------

**8-bit Quantization**

Most common choice, good accuracy retention:

.. code-block:: yaml

   training:
     quantization: 2
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

* Model size: 4x smaller than float32
* Accuracy loss: Usually <1%

**4-bit Quantization**

Aggressive compression for size-constrained devices:

.. code-block:: yaml

   training:
     quantization: 2
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 4
     quantization_activation_bitwidth: 4

* Model size: 8x smaller than float32
* Accuracy loss: 1-5% typical

**2-bit Quantization**

Maximum compression, limited use cases:

.. code-block:: yaml

   training:
     quantization: 2
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 2
     quantization_activation_bitwidth: 2

* Model size: 16x smaller than float32
* Accuracy loss: Can be significant

.. note::

   Weight and activation bit widths can be set independently. For example,
   you can use 8-bit activations with 4-bit weights:

   .. code-block:: yaml

      training:
        quantization: 2
        quantization_method: 'QAT'
        quantization_weight_bitwidth: 4
        quantization_activation_bitwidth: 8

NPU Quantization Requirements
-----------------------------

TI's NPU requires TI style optimised quantization (``quantization: 2``):

.. code-block:: yaml

   common:
     target_device: 'F28P55'

   training:
     model_name: 'CLS_4k_NPU'
     quantization: 2
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

**NPU Constraints:**

* Must use ``quantization: 2`` (TI style optimised)
* INT8 or INT4 bit widths recommended
* Symmetric quantization preferred
* Per-channel quantization for weights
* Per-tensor quantization for activations

Output Files
------------

After quantization, you'll find:

.. code-block:: text

   .../training/
   ├── base/
   │   └── best_model.pt          # Float32 model
   └── quantization/
       ├── best_model.onnx        # Quantized ONNX
       └── quantization_config.yaml

Accuracy Comparison
-------------------

Typical accuracy retention by bit width:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Precision
     - Size Reduction
     - Speed Improvement
     - Accuracy Drop
   * - Float32
     - 1x (baseline)
     - 1x (baseline)
     - 0%
   * - INT8
     - 4x
     - 2-4x
     - <1%
   * - INT4
     - 8x
     - 3-6x
     - 1-5%
   * - INT2
     - 16x
     - 4-8x
     - 5-15%

**Note:** Results vary by model and task.

Troubleshooting Accuracy Loss
-----------------------------

If quantization hurts accuracy:

**1. Try QAT instead of PTQ:**

.. code-block:: yaml

   training:
     quantization: 2
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

**2. Use higher bit widths:**

If using 4-bit or 2-bit, try 8-bit first:

.. code-block:: yaml

   training:
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

**3. Keep activations at higher precision:**

Use higher activation bit width with lower weight bit width:

.. code-block:: yaml

   training:
     quantization_weight_bitwidth: 4
     quantization_activation_bitwidth: 8

**4. Increase model size:**

A larger model may tolerate quantization better.

Best Practices
--------------

1. **Start with INT8**: Best balance of compression and accuracy
2. **Use QAT for critical applications**: When accuracy is paramount
3. **Use TI optimised quantization for NPU**: Set ``quantization: 2`` for NPU targets
4. **Compare float vs quantized**: Always measure accuracy drop
5. **Test on target device**: Verify behavior matches simulation

Example: Full Quantization Workflow
-----------------------------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'dc_arc_fault_example_dsk'

   data_processing_feature_extraction:
     feature_extraction_name: 'FFT1024Input_256Feature_1Frame_Full_Bandwidth'
     variables: 1

   training:
     model_name: 'CLS_4k_NPU'
     training_epochs: 30
     batch_size: 256
     quantization: 2
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

   testing:
     enable: True

   compilation:
     enable: True
     preset_name: 'compress_npu_layer_data'

**Expected Results:**

.. code-block:: text

   Float32 Model:
   Accuracy: 99.2%
   Size: 1.6 KB

   INT8 Quantized Model:
   Accuracy: 99.0%
   Size: 0.4 KB
   Speedup: 3.5x on NPU

Memory Savings
--------------

Quantization reduces memory at multiple levels:

**Model Weights:**

.. code-block:: text

   Float32: 4 bytes per parameter
   INT8:    1 byte per parameter
   INT4:    0.5 bytes per parameter

   Example: 4000 parameter model
   Float32: 16 KB
   INT8:    4 KB
   INT4:    2 KB

**Activations:**

Intermediate computations also benefit from reduced precision.

**Total Memory:**

For memory-constrained devices, quantization may be the difference
between fitting and not fitting.

Performance Impact
------------------

**NPU Performance:**

.. code-block:: text

   Model: CLS_4k_NPU on F28P55

   Float32 (CPU): ~5000 µs
   INT8 (NPU):    ~300 µs
   INT4 (NPU):    ~200 µs

**CPU Performance:**

Even without NPU, integer operations are faster:

.. code-block:: text

   Model: CLS_4k on F28P65 (no NPU)

   Float32: ~5000 µs
   INT8:    ~2000 µs

Quantization Wrapper Architecture
----------------------------------

Under the hood, Tiny ML Tensorlab uses quantization wrapper classes from
the ``tinyml-modeloptimization`` package. Understanding the wrapper
architecture helps when customizing quantization or debugging.

**Class Hierarchy:**

.. code-block:: text

   TinyMLQuantFxBaseModule (base class)
       ├── TINPUTinyMLQuantFxModule
       │   ├── TINPUTinyMLQATFxModule   (quantization: 2, QAT)
       │   └── TINPUTinyMLPTQFxModule   (quantization: 2, PTQ)
       │
       └── GenericTinyMLQuantFxModule
           ├── GenericTinyMLQATFxModule  (quantization: 1, QAT)
           └── GenericTinyMLPTQFxModule  (quantization: 1, PTQ)

**TINPUTinyML wrappers** (``quantization: 2``) incorporate the constraints
of TI NPU Hardware accelerator. They perform extensive graph transformations
including 13+ layer pattern replacements to produce NPU-compatible integer
operations. Key characteristics:

* Enforces power-of-2 scale factors (mandatory for 8-bit quantization)
* Transforms convolution, pooling, linear, and batch normalization layers
  to NPU-compatible patterns
* Implements the NPU BNORM sequence:
  ``Add (bias) → Mul (scale) → Div (2^n, right shift) → Floor → Clip``
* All operations in integer domain, no dequantization step

**GenericTinyML wrappers** (``quantization: 1``) use standard PyTorch
quantization APIs with minimal modifications, relying on ONNX Runtime for
optimization. Key characteristics:

* Flexible scaling (no power-of-2 constraint)
* Only 1 pattern replacement (permute + unsqueeze)
* Uses PyTorch's native quantized operations
* Relies on ONNX Runtime optimization for deployment

.. note::

   When using the toolchain via YAML configs, you do not need to interact
   with these wrapper classes directly. Setting ``quantization: 1`` or
   ``quantization: 2`` in the config selects the appropriate wrapper
   automatically.

NPU Hardware Constraints
------------------------

When using TI style optimised quantization (``quantization: 2``), the
following hardware constraints are enforced automatically by the TINPU
wrapper:

**Channel Alignment:**

Input and output channels must be multiples of 4. The NPU processes data
in SIMD fashion with 4-channel vectors.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Layer Type
     - Channel Requirement
     - Notes
   * - FCONV (First Conv)
     - Input: exactly 1, Output: multiple of 4
     - First layer in the network
   * - GCONV (Generic Conv)
     - Input and Output: multiple of 4
     - General convolution layers
   * - DWCONV (Depthwise Conv)
     - Input/Output: multiple of 4
     - Depthwise separable layers
   * - PWCONV (Pointwise Conv)
     - Input/Output: multiple of 4
     - 1x1 convolution layers
   * - FC (Fully Connected)
     - Input: multiple of 4
     - Dense/linear layers

**Power-of-2 Scaling:**

For 8-bit quantization, scale factors must be powers of 2. This enables
efficient implementation as bit shifts in hardware, avoiding expensive
division operations. For sub-8-bit quantization (4-bit, 2-bit),
non-power-of-2 scales are supported and may provide better accuracy.

**Bitwidth Constraints:**

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Parameter
     - Allowed Values
     - Notes
   * - Weight bitwidth
     - 2, 4, or 8 bits (signed)
     - Determines model compression ratio
   * - Activation bitwidth
     - 8 bits (signed or unsigned)
     - Fixed at 8 bits for NPU acceleration
   * - Bias
     - 16-bit (2b/4b weights), 24-bit (8b weights)
     - Automatically computed
   * - Scale
     - 8-bit unsigned (2b/4b), power-of-2 shift (8b)
     - Automatically computed

**Supported NPU Layer Patterns:**

The NPU accelerates the following layer types: FCONV, GCONV, DWCONV,
PWCONV, FC, AVGPOOL, MAXPOOL. Each layer includes a BNORM sequence
(bias → scale → shift → floor → clip) that maps directly to NPU hardware
units.

.. warning::

   Models with layers that do not meet NPU constraints will fall back to
   CPU execution for those layers. Use ``quantization: 1`` (Generic) for
   models that cannot satisfy these constraints.

Using Quantization Wrappers Directly
-------------------------------------

For advanced users who want to use the quantization wrappers outside the
Tiny ML Tensorlab toolchain (e.g., in custom PyTorch training scripts),
the wrappers can be imported and used directly.

**TINPU QAT Example:**

.. code-block:: python

   from tinyml_torchmodelopt.quantization import TINPUTinyMLQATFxModule

   # Create and pretrain your model
   model = MyNeuralNetwork()
   model.load_state_dict(torch.load('pretrained.pth'))

   # Wrap with TINPU quantization
   model = TINPUTinyMLQATFxModule(model, total_epochs=epochs)

   # Train the wrapped model (your usual training loop)
   model.train()
   for e in range(epochs):
       for images, target in train_loader:
           output = model(images)
           # loss, backward(), optimizer step as usual

   model.eval()

   # Convert to integer operations
   model = model.convert()

   # Export to ONNX
   dummy_input = torch.rand((1, 1, 256, 1))
   model.export(dummy_input, 'model_int8.onnx', input_names=['input'])

**Generic QAT Example:**

.. code-block:: python

   from tinyml_torchmodelopt.quantization import GenericTinyMLQATFxModule

   # Create and pretrain your model
   model = MyNeuralNetwork()
   model.load_state_dict(torch.load('pretrained.pth'))

   # Wrap with Generic quantization
   model = GenericTinyMLQATFxModule(model, total_epochs=epochs)

   # Train, convert, and export (same API as TINPU)
   # ...
   model = model.convert()
   model.export(dummy_input, 'model_int8.onnx', input_names=['input'])

**PTQ (Post-Training Quantization):**

For PTQ, replace the QAT module with the PTQ variant. PTQ only requires
a calibration pass (forward pass on representative data) instead of full
retraining:

.. code-block:: python

   from tinyml_torchmodelopt.quantization import TINPUTinyMLPTQFxModule

   model = TINPUTinyMLPTQFxModule(model, total_epochs=1)

   # Calibration pass (no backward, no optimizer)
   model.eval()
   with torch.no_grad():
       for images, _ in calibration_loader:
           model(images)

   model = model.convert()
   model.export(dummy_input, 'model_int8.onnx', input_names=['input'])

**Evaluating Exported ONNX Models:**

After exporting, you can evaluate the quantized ONNX model using ONNX
Runtime:

.. code-block:: python

   import onnxruntime as ort

   ort_session_options = ort.SessionOptions()
   ort_session_options.graph_optimization_level = (
       ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
   )

   ort_session = ort.InferenceSession('model_int8.onnx', ort_session_options)
   prediction = ort_session.run(None, {'input': example_input.numpy()})

Wrapper API Reference
---------------------

All quantization wrappers inherit from ``TinyMLQuantFxBaseModule``, which
accepts the following parameters:

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Parameter
     - Type
     - Description
   * - ``model``
     - nn.Module
     - The PyTorch model to quantize
   * - ``qconfig_type``
     - dict/None
     - QConfig mapping for quantization. ``None`` uses wrapper defaults.
   * - ``example_inputs``
     - Tensor
     - Example input with batch size 1
   * - ``is_qat``
     - bool
     - Toggle between QAT (True) and PTQ (False)
   * - ``backend``
     - str
     - Backend: ``'qnnpack'`` (Linux) or ``'fbgemm'`` (Windows).
       Automatically selected.
   * - ``total_epochs``
     - int
     - Total number of quantized training epochs
   * - ``num_batch_norm_update_epochs``
     - bool/int
     - BatchNorm freezing control (see below)
   * - ``num_observer_update_epochs``
     - bool/int
     - Observer freezing control (see below)
   * - ``bias_calibration_factor``
     - float
     - Bias calibration factor (0.0 = disabled)
   * - ``verbose``
     - bool
     - Enable verbose logging
   * - ``float_ops``
     - bool
     - Use float bias for Conv/Linear layers. Increases accuracy but
       disables BNORM on TINPU hardware.

**BatchNorm Freezing (``num_batch_norm_update_epochs``):**

* ``None`` (default): Freezes BatchNorm statistics at the midpoint of
  training
* ``False``: Never freezes BatchNorm (may cause overfitting)
* Integer value: Freezes after the specified epoch. Best results with
  half to 3/4 of total epochs.

**Observer Freezing (``num_observer_update_epochs``):**

* ``False`` (default): Observers remain active throughout training
* Integer value: Freezes observers after the specified epoch

.. tip::

   For best QAT results, set ``num_batch_norm_update_epochs`` to
   approximately half of ``total_epochs``. This allows the model to
   learn quantization-aware representations before freezing statistics.

Model Surgery
-------------

The ``tinyml-modeloptimization`` package includes model surgery utilities
that use ``torch.fx`` to replace unsupported modules with efficient
alternatives. This is useful for adapting existing models to meet NPU
constraints.

**Basic Usage:**

.. code-block:: python

   from tinyml_torchmodelopt.surgery import convert_to_lite_fx

   # Replace unsupported layers with default replacements
   model = convert_to_lite_fx(model)

**Custom Replacements:**

You can define custom replacement rules:

.. code-block:: python

   import copy
   from tinyml_torchmodelopt.surgery import (
       convert_to_lite_fx, get_replacement_dict_default
   )

   # Get and modify the default replacement dictionary
   replacement_dict = copy.deepcopy(get_replacement_dict_default())
   replacement_dict.update({torch.nn.GELU: torch.nn.ReLU})

   # Apply with custom replacements
   model = convert_to_lite_fx(model, replacement_dict=replacement_dict)

The replacement value can also be a function for complex transformations:

.. code-block:: python

   replacement_dict.update({'my_layer': my_replacement_function})
   model = convert_to_lite_fx(model, replacement_dict=replacement_dict)

Model surgery is applied automatically during the quantization pipeline
when needed. Direct usage is only necessary for custom workflows.

Next Steps
----------

* Explore :doc:`neural_architecture_search` for model optimization
* Learn about :doc:`feature_extraction` for input preparation
* Deploy quantized models: :doc:`/deployment/npu_device_deployment`
