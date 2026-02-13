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
     model_name: 'ArcFault_model_400_t'
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

Next Steps
----------

* Explore :doc:`neural_architecture_search` for model optimization
* Learn about :doc:`feature_extraction` for input preparation
* Deploy quantized models: :doc:`/deployment/npu_device_deployment`
