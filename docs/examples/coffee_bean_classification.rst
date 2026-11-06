==========================
Coffee Bean Classification
==========================

Classify coffee beans by roast level from images, using a MobileNetV1 model on the MSPM0G5187 with NPU.

Example folder: ``examples/coffee_bean_classification``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/coffee_bean_classification>`_)

Overview
--------

* **Task**: image classification (``image_classification``, ``target_module: vision``), 4 classes
* **Application**: coffee bean roast and quality inspection from color, texture and surface appearance
* **Device**: MSPM0G5187 (``config_MSPM0.yaml``, the only config in the folder). The folder contains only the config
  and no readme.
* **Model**: ``MobileNetV1_28k_NPU``

.. note::

   In 1.4.0 this example used ``MobileNetV1_58k_NPU``. In 1.5.0 the config uses ``MobileNetV1_28k_NPU``.

The Dataset
-----------

The data is the Kaggle
`Coffee Bean Dataset Resized (224 x 224) <https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224>`_
with four classes: Dark, Green, Light and Medium. Check the Kaggle page for the license and usage terms. The config
downloads the prepared dataset from
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/coffee_bean_classification.zip``. The layout is a ``classes/``
folder with one sub-folder per class.

Feature Extraction
------------------

The config uses the preset ``CoffeeBean_Default``: RGB images (3 channels) resized to 128x128 and normalized with the
ImageNet mean (0.485, 0.456, 0.406) and scale (0.229, 0.224, 0.225). The preset ``CoffeeBean_Augmentation_Default``
adds random horizontal and vertical flips, random rotation and color jitter if you want augmentation.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/coffee_bean_classification/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\coffee_bean_classification\config_MSPM0.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: vision
     task_type: image_classification
     target_device: MSPM0G5187
   dataset:
     dataset_name: coffee_bean_classification
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/coffee_bean_classification.zip
   data_processing_feature_extraction:
     feature_extraction_name: CoffeeBean_Default
   training:
     model_name: MobileNetV1_28k_NPU
     batch_size: 64
     training_epochs: 30
     num_gpus: 1
     quantization: 2
     learning_rate: 0.1
   testing: {}
   compilation: {}

``num_gpus: 1`` expects a GPU; use 0 for CPU. The image models enabled in 1.5.0 are ``Lenet5``,
``MobileNetV1_28k_NPU`` and ``MobileNetV2_58k_NPU``. For the image task type see
:doc:`/task_types/image_classification`.

Results
-------

There is no readme with published results. In the automated 1.5.0 config test run (test set of 160 images, 40 per
class) the quantized model reached a validation accuracy of 98.75% (F1 0.988, AUC ROC 0.972) and a test accuracy of
99.38% (AUC ROC 0.966). This is a small test set, so treat it as indicative.

Output Location
---------------

Results are written to ``./data/projects/coffee_bean_classification/run/<date-time>/MobileNetV1_28k_NPU/``. See
:doc:`/getting_started/understanding_run_output` and
:doc:`/deploying_classification_models_from_modelmaker_to_device/readme` for deployment with the MSPM0 SDK and a
camera interface for image capture.

See Also
--------

* :doc:`machine_readable_code_classification`
* :doc:`mnist_image_classification`
* :doc:`image_classification_example`
