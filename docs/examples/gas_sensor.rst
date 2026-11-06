==========
Gas Sensor
==========

Identify gases from a metal-oxide sensor array, and learn how the ``output_int`` option changes the model
output format (quantized integer vs. dequantized float).

Example folder: ``examples/gas_sensor``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/gas_sensor>`_)

Overview
--------

The "Gas Sensor Array under Low Concentration" dataset from the UCI Machine Learning Repository contains
30-minute recordings sampled at 1 Hz from 10 metal-oxide semiconductor (MOS) sensors exposed to low
concentrations of six gases. The example uses three of them, each at 100 ppb: **ethyl acetate**,
**isopropanol** and **hexane**.

* **Application**: environmental monitoring, industrial safety, air quality
* **Task**: classification (``generic_timeseries_classification``), 3 classes
* **Data**: 10 variables (``Sensor_0`` to ``Sensor_9``), 64-sample windows
* **Device**: F28P55
* **Model**: ``CLS_1k_NPU``

Preparing the Dataset
---------------------

The config downloads ``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/gas_sensor_dataset.zip``
automatically. To regenerate the zip from the UCI source run the helper script and point
``dataset.input_data_path`` to ``examples/gas_sensor/gas_sensor_dataset.zip``:

.. code-block:: bash

   cd examples/gas_sensor
   python gas_sensor.py

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/gas_sensor/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\gas_sensor\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55
   dataset:
     dataset_name: gas_sensor
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/gas_sensor_dataset.zip
     split_type: within_files
   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 64
     stride_size: 0.02
     variables: 10
     scale: 0.00390625   # 1/256
   training:
     model_name: CLS_1k_NPU
     num_gpus: 0
     epochs: 20
     early_stopping: true
     early_stopping_patience: 5
     output_int: true
   testing: {}
   compilation: {}

Quantized vs. Dequantized Output (``output_int``)
-------------------------------------------------

After float training the model goes through quantization-aware or post-training quantization. The
example demonstrates the difference between the output of native PyTorch quantization and the TI quantization
wrappers (``TINPUTinyMLQATFxModule`` / ``TINPUTinyMLPTQFxModule``):

* ``output_int: true`` (default): no multiply block is added to the last layer, so the model output stays in
  quantized INT8 format. This is faster because everything stays in the integer domain.
* ``output_int: false``: a multiply block is added to the last layer of the ONNX model that dequantizes the
  output to floating point.

.. figure:: /_static/img/examples/gas_sensor/quantize_vs_dequantize.png
   :width: 600px
   :align: center
   :alt: Quantized vs dequantized outputs

   Quantized vs. dequantized model outputs

The effect can be inspected in ``training/quantization/golden_vectors/test_vector.c`` of the run folder. With
``output_int: false`` the scaling factor learned for this model is 0.03125, so a dequantized output of
{0.875, -0.4375, -0.40625} corresponds to the quantized output {28, -14, -13}.

Performance on Device
---------------------

The example readme benchmarks ``CLS_1k_NPU`` on the F28P55x, on the CPU and on the NPU (TINPU). These
numbers come from the example readme (they were measured with an earlier release):

.. list-table::
   :header-rows: 1
   :widths: 34 16 18 16 16

   * - Configuration
     - AI model cycles
     - Inference time (us)
     - Flash (B)
     - SRAM (B)
   * - CPU, ``output_int`` true
     - 665,234
     - 4,434.89
     - 4,143
     - 3,880
   * - CPU, ``output_int`` false
     - 665,801
     - 4,438.67
     - 4,169
     - 3,880
   * - NPU, ``output_int`` true
     - 80,339
     - 535.59
     - 3,529
     - 1,680
   * - NPU, ``output_int`` false
     - 80,354
     - 535.69
     - 3,560
     - 1,682

Results from the 1.5.0 Test Run
-------------------------------

In the automated 1.5.0 config test run, ``config.yaml`` reached a test accuracy of 100.00% with an AUC ROC of
1.000 (``CLS_1k_NPU``, 1,271 parameters).

See Also
--------

* :doc:`/DEPLOYING_TO_SDK_EXAMPLE`: use the generated golden vectors in an SDK example
* :doc:`generic_classification`
