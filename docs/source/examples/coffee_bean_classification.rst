==========================
Coffee Bean Classification
==========================

Coffee bean quality classification from images using MobileNetV1 on MSPM0G5187 with NPU.

Overview
--------

* **Task**: Image Classification
* **Application**: Coffee bean quality inspection
* **Dataset**: coffee_bean_classification
* **Model**: MobileNetV1_58k_NPU
* **Device**: MSPM0G5187 (NPU-accelerated)

Device Support
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 50 10

   * - Device
     - Notes
     - Configuration File
   * - ``MSPM0G5187``
     - MSPM0 with NPU
     - ``config_MSPM0.yaml``

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/coffee_bean_classification/config_MSPM0.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\coffee_bean_classification\config_MSPM0.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: 'image_classification'
     target_device: 'MSPM0G5187'

   dataset:
     dataset_name: 'coffee_bean_classification'
     input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/coffee_bean_classification.zip'

   data_processing_feature_extraction:
     feature_extraction_name: 'CoffeeBean_Default'

   training:
     model_name: 'MobileNetV1_58k_NPU'
     training_epochs: 30
     batch_size: 64
     learning_rate: 0.1
     quantization: 2

   testing: {}
   compilation: {}

Available Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Model
     - Parameters
     - Description
   * - ``MobileNetV1_58k_NPU``
     - ~58,000
     - MobileNetV1 optimized for NPU deployment

System Components
-----------------

**Hardware**

* `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ microcontroller with integrated NPU
* Camera interface for image capture

**Software**

* Code Composer Studio (CCS) 12.x or later
* MSPM0 SDK 2.10.04 or later

Next Steps
----------

* Browse similar examples: :doc:`machine_readable_code_classification`, :doc:`mnist_image_classification`
* Deploy to device: :doc:`/deployment/npu_device_deployment`
