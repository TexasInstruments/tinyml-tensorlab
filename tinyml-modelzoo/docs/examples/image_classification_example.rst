===================================
Image Classification Examples Guide
===================================

In 1.4.0 this page was a generic image classification tutorial that pointed to an ``examples/image_classification``
folder. That folder does not exist in 1.5.0. This page is now a short guide to the three image classification
examples and the settings they share. Each example has its own page with the full configuration. For the task type
itself see :doc:`/task_types/image_classification`.

Image Examples in 1.5.0
-----------------------

All use ``task_type: image_classification`` with ``target_module: vision`` on the MSPM0G5187 and are run from the
``tinyml-modelzoo`` folder with ``./run_tinyml_modelzoo.sh examples/<folder>/<config>.yaml`` (Windows: see
:doc:`/getting_started/running_examples`).

.. list-table::
   :header-rows: 1
   :widths: 36 16 18 14 16

   * - Config
     - Model
     - Input
     - Classes
     - Page
   * - ``MNIST_image_classification/config_image_classification_mnist.yaml``
     - ``Lenet5``
     - 28x28 grayscale
     - 10
     - :doc:`mnist_image_classification`
   * - ``machine_readable_code_classification/config_MSPM0.yaml``
     - ``Lenet5``
     - 28x28 binary
     - 3
     - :doc:`machine_readable_code_classification`
   * - ``coffee_bean_classification/config_MSPM0.yaml``
     - ``MobileNetV1_28k_NPU``
     - 128x128 RGB
     - 4
     - :doc:`coffee_bean_classification`

Dataset Format
--------------

Image datasets use the classification layout: one folder per class under ``classes/`` containing the images
(PNG or similar).

.. code-block:: text

   dataset.zip/
   |-- classes/
         |-- class_a/
         |     |-- img_001.png
         |-- class_b/
               |-- img_001.png

The training, validation and test split is made automatically.

Feature Extraction Presets
--------------------------

Image preprocessing is selected with ``feature_extraction_name``. The presets used by the examples are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Preset
     - Preprocessing
   * - ``Mnist_Default``
     - Grayscale, resize to 28x28, mean 0.1307, scale 0.3081
   * - ``MachineReadable_Default``
     - Grayscale, resize to 28x28, mean 0.5, scale 0.5
   * - ``CoffeeBean_Default``
     - RGB, resize to 128x128, ImageNet mean and scale
   * - ``CoffeeBean_Augmentation_Default``
     - As ``CoffeeBean_Default`` plus random flips, rotation and color jitter

Image size drives memory use: the input tensor is ``height x width x channels`` and is the largest early
activation. Small inputs (28x28) fit the smallest devices, while 128x128 RGB images need an NPU class model.

Available Models
----------------

The image models enabled in 1.5.0 are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Description
   * - ``Lenet5``
     - Classic LeNet-5 CNN, about 60,000 parameters according to the MNIST readme
   * - ``MobileNetV1_28k_NPU``
     - NPU-compliant MobileNetV1-inspired tiny model
   * - ``MobileNetV2_58k_NPU``
     - NPU-compliant MobileNetV2-inspired tiny model

Quantization is set with ``quantization: 2`` in all three examples.

Results
-------

From the automated 1.5.0 config test run: MNIST 98.93% test accuracy, machine readable code 100.00% test accuracy
(synthetic data), coffee bean 99.38% test accuracy on a small test set. See the example pages for details.

Limitations
-----------

These are tiny models for small, simple images. High resolution images, many classes or fine-grained detail need a
larger model than a microcontroller NPU can run. For deployment see
:doc:`/deploying_classification_models_from_modelmaker_to_device/readme`.
