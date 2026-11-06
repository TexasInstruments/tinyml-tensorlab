====================
Image Classification
====================

Image classification assigns labels to images based on their visual content.

Overview
--------

**What it does**: Takes an image and outputs a class label.

**Example**: Handwritten digit image -> "0", "1", "2", ... "9"

**Use cases**:

* Visual inspection and quality control
* Digit / character recognition
* Machine-readable code classification
* Simple image sorting (for example coffee-bean grades)

.. note::
   Image classification in Tiny ML ModelZoo targets small images and small models. For complex
   image tasks, consider TI's Edge AI tools.

Task Settings
-------------

.. list-table::
   :widths: 30 70

   * - ``task_type``
     - ``image_classification``
   * - ``target_module``
     - ``vision`` (inferred from ``task_type`` if you omit it)
   * - Data folder
     - ``classes/``
   * - Early stopping
     - On by default in 1.5.0; see :doc:`/features/early_stopping`

Configuration
-------------

This is ``examples/MNIST_image_classification/config_image_classification_mnist.yaml``:

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

Run it from the tinyml-modelzoo folder:

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/MNIST_image_classification/config_image_classification_mnist.yaml

See :doc:`/getting_started/quickstart` for the install options and :doc:`/examples/mnist_image_classification`
for the full walkthrough.

Dataset Format
--------------

Image datasets use a ``classes/`` folder structure:

.. code-block:: text

   my_image_dataset/
   └── classes/
       ├── class_0/
       │   ├── image1.png
       │   ├── image2.png
       │   └── ...
       ├── class_1/
       │   └── ...
       └── class_N/
           └── ...

* Supported formats: ``.png``, ``.jpg``, ``.jpeg``, ``.bmp``. Images must sit directly inside a class folder.
* Train/val/test lists are generated automatically if you do not provide ``annotations/``;
  see :doc:`/byod/data_splitting`.

Feature Extraction Presets
--------------------------

The preset sets image size, channels and normalization:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``feature_extraction_name``
     - Settings
   * - ``Mnist_Default``
     - Grayscale, 28x28, 1 channel
   * - ``MachineReadable_Default``
     - Grayscale, 28x28, 1 channel
   * - ``CoffeeBean_Default``
     - RGB, resized, 3 channels
   * - ``CoffeeBean_Augmentation_Default``
     - RGB, 128x128, with random flips, rotation and colour jitter

Available Models
----------------

Enabled in ``tinyml-modelmaker`` for this task (``enabled_models_list``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model Name
     - Description
   * - ``Lenet5``
     - Classic CNN for small grayscale images (MNIST example)
   * - ``MobileNetV1_28k_NPU``
     - NPU-friendly MobileNetV1 variant (about 28k parameters, used in the coffee-bean example)
   * - ``MobileNetV2_58k_NPU``
     - NPU-friendly MobileNetV2 variant (about 58k parameters)

Current Limitations
-------------------

* Small image sizes are recommended (28x28 up to about 128x128).
* Only classification is supported; there is no detection or segmentation task.

Examples
--------

* :doc:`/examples/mnist_image_classification` - Handwritten digits (``Lenet5``)
* :doc:`/examples/coffee_bean_classification` - Coffee bean classification (``MobileNetV1_28k_NPU``)
* :doc:`/examples/machine_readable_code_classification` - Machine-readable code classification

Preparing Image Data
--------------------

1. Resize images to the target size (for example 28x28).
2. Convert to grayscale if using a grayscale preset.
3. Organize the images into class folders.

.. code-block:: python

   # Example preprocessing script
   from PIL import Image
   import os

   def preprocess_images(input_dir, output_dir, size=(28, 28)):
       for class_name in os.listdir(input_dir):
           class_in = os.path.join(input_dir, class_name)
           class_out = os.path.join(output_dir, 'classes', class_name)
           os.makedirs(class_out, exist_ok=True)

           for img_file in os.listdir(class_in):
               img = Image.open(os.path.join(class_in, img_file))
               img = img.convert('L')  # Grayscale
               img = img.resize(size)
               img.save(os.path.join(class_out, img_file))

Tips
----

* Keep image sizes small for MCU deployment.
* Use grayscale when colour is not essential.
* Start from a shipped preset and only then customize.

See Also
--------

* :doc:`timeseries_classification` - Time series classification
* :doc:`/byod/data_splitting` - Splits
* `TI Edge AI <https://www.ti.com/tool/EDGE-AI-STUDIO>`_ - For advanced image tasks
