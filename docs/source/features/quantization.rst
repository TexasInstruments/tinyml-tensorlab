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

Quantization Types
------------------

**Post-Training Quantization (PTQ)**

Quantizes a trained float model after training:

.. code-block:: yaml

   training:
     model_name: 'CLS_4k_NPU'
     quantization_type: 'int8'
     ptq_calibration_samples: 1000

* Pros: Fast, simple
* Cons: May lose accuracy for some models

**Quantization-Aware Training (QAT)**

Simulates quantization during training:

.. code-block:: yaml

   training:
     model_name: 'CLS_4k_NPU'
     quantization_type: 'int8'
     qat_enabled: True
     training_epochs: 30

* Pros: Better accuracy retention
* Cons: Longer training time

Bit Widths
----------

**8-bit Quantization (INT8)**

Most common choice, good accuracy retention:

.. code-block:: yaml

   training:
     quantization_type: 'int8'

* Range: -128 to 127 (signed) or 0-255 (unsigned)
* Model size: 4x smaller than float32
* Accuracy loss: Usually <1%

**4-bit Quantization (INT4)**

Aggressive compression for size-constrained devices:

.. code-block:: yaml

   training:
     quantization_type: 'int4'

* Range: -8 to 7 (signed)
* Model size: 8x smaller than float32
* Accuracy loss: 1-5% typical

**2-bit Quantization (INT2)**

Maximum compression, limited use cases:

.. code-block:: yaml

   training:
     quantization_type: 'int2'

* Range: -2 to 1 (signed)
* Model size: 16x smaller than float32
* Accuracy loss: Can be significant

Enabling Quantization
---------------------

**Basic INT8 Quantization:**

.. code-block:: yaml

   training:
     model_name: 'CLS_4k_NPU'
     quantization_type: 'int8'

**QAT with INT8:**

.. code-block:: yaml

   training:
     model_name: 'CLS_4k_NPU'
     quantization_type: 'int8'
     qat_enabled: True
     qat_start_epoch: 10  # Start QAT after warmup

**Mixed Precision:**

Different layers can use different precisions:

.. code-block:: yaml

   training:
     quantization_type: 'mixed'
     mixed_precision_config:
       first_layer: 'int8'     # Sensitive layer
       hidden_layers: 'int4'   # Can tolerate lower precision
       last_layer: 'int8'      # Output layer

Calibration
-----------

PTQ requires calibration data to determine quantization parameters:

.. code-block:: yaml

   training:
     quantization_type: 'int8'
     ptq_calibration_samples: 1000
     ptq_calibration_method: 'minmax'  # or 'histogram', 'entropy'

**Calibration Methods:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - ``minmax``
     - Uses min/max values (fast, simple)
   * - ``histogram``
     - Uses value distribution (better accuracy)
   * - ``entropy``
     - Minimizes KL divergence (best accuracy)

**Calibration Best Practices:**

* Use representative data (similar to inference data)
* Include edge cases and variations
* More samples = better calibration (up to ~1000)

NPU Quantization Requirements
-----------------------------

TI's NPU requires specific quantization:

.. code-block:: yaml

   common:
     target_device: 'F28P55'

   training:
     model_name: 'CLS_4k_NPU'
     quantization_type: 'int8'  # Required for NPU

**NPU Constraints:**

* Must use INT8 or INT4
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
       ├── quantization_config.yaml
       └── calibration_stats.json

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
     qat_enabled: True
     training_epochs: 50

**2. Use more calibration data:**

.. code-block:: yaml

   training:
     ptq_calibration_samples: 2000

**3. Use histogram or entropy calibration:**

.. code-block:: yaml

   training:
     ptq_calibration_method: 'entropy'

**4. Keep sensitive layers at higher precision:**

.. code-block:: yaml

   training:
     quantization_type: 'mixed'
     sensitive_layers: ['first_conv', 'classifier']

**5. Increase model size:**

A larger model may tolerate quantization better.

Best Practices
--------------

1. **Start with INT8**: Best balance of compression and accuracy
2. **Use QAT for critical applications**: When accuracy is paramount
3. **Calibrate on representative data**: Match inference conditions
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

     # Quantization settings
     quantization_type: 'int8'
     qat_enabled: True
     qat_start_epoch: 15
     ptq_calibration_samples: 500

   testing:
     enable: True
     test_quantized: True  # Also test quantized model

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
