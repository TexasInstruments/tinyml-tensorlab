==========================
MNIST Image Classification
==========================

Handwritten digit recognition (0-9) on the `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ microcontroller
using the classic LeNet-5 CNN.

Example folder: ``examples/MNIST_image_classification``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/MNIST_image_classification>`_)

Overview
--------

Handwritten digit recognition is a basic task in document processing and automation. Traditional template matching
struggles with handwriting variability, and edge AI allows recognition directly on a resource-constrained MCU
without cloud connectivity.

* **Task**: image classification (``image_classification``, ``target_module: vision``), 10 classes
* **Device**: MSPM0G5187 (``config_image_classification_mnist.yaml``, the only config in the folder)
* **Model**: ``Lenet5`` (about 60,000 parameters according to the readme)
* **Input**: 28x28 grayscale images
* **Demo setup in the readme**: the MCU talks to a host GUI over UART

The Dataset
-----------

The data is MNIST exported with torchvision as 28x28 grayscale PNG images in 10 class folders. The config
downloads ``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/mnist_classes.zip`` automatically. To recreate the
dataset, run ``mnist_dataset_creation.py`` in the example folder; see ``readme_dataset_creation.md`` there. The
layout is a ``classes/`` folder with one sub-folder per digit.

Feature Extraction
------------------

The config uses the preset ``Mnist_Default``. The preprocessing converts to grayscale (1 channel), resizes to
28x28, converts to a tensor and normalizes with mean 0.1307 and scale 0.3081.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/MNIST_image_classification/config_image_classification_mnist.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\MNIST_image_classification\config_image_classification_mnist.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: vision
     task_type: image_classification
     target_device: MSPM0G5187
   dataset:
     dataset_name: mnist_image_classification
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/mnist_classes.zip
   data_processing_feature_extraction:
     feature_extraction_name: Mnist_Default
   training:
     model_name: Lenet5
     batch_size: 64
     training_epochs: 14
     num_gpus: 1
     quantization: 2
     learning_rate: 0.1
   testing: {}
   compilation: {}

``quantization: 2`` gives INT8 quantization for a smaller model. ``num_gpus: 1`` expects a GPU; use 0 for CPU.
The image models enabled in 1.5.0 are ``Lenet5``, ``MobileNetV1_28k_NPU`` and ``MobileNetV2_58k_NPU``. For
the image task type see :doc:`/task_types/image_classification`.

Results
-------

The readme quotes an accuracy of about 99%. The automated 1.5.0 config test run reached a best quantized epoch of 3
with a validation accuracy of 98.94% (F1 0.989, AUC ROC 0.998) and a test accuracy of 98.93% (AUC ROC 0.999).

Output Location
---------------

Results are written to ``./data/projects/mnist_image_classification/run/<date-time>/Lenet5/``. See
:doc:`/getting_started/understanding_run_output` and
:doc:`/deploying_classification_models_from_modelmaker_to_device/readme` for deployment with the MSPM0 SDK
(2.08.00 or later) and Code Composer Studio.

See Also
--------

* :doc:`machine_readable_code_classification`
* :doc:`coffee_bean_classification`
* :doc:`image_classification_example`
