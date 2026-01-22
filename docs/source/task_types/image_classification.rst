====================
Image Classification
====================

Image classification assigns labels to images based on their visual content.

Overview
--------

**What it does**: Takes an image and outputs a class label.

**Example**: Handwritten digit image → "0", "1", "2", ... "9"

**Use cases**:

* Visual inspection
* Digit/character recognition
* Simple object detection
* Quality control

.. note::
   Image classification in Tiny ML Tensorlab is currently limited compared to
   time series tasks. For complex image tasks, consider TI's Edge AI tools.

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'image'
     task_type: 'image_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'mnist_digit'
     input_data_path: '/path/to/mnist'

   training:
     model_name: 'Lenet5'
     training_epochs: 10
     batch_size: 64

   testing: {}
   compilation: {}

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

Supported formats: PNG, JPG, BMP

Available Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model Name
     - Parameters
     - Description
   * - ``Lenet5``
     - ~60,000
     - Classic CNN for digits

Current Limitations
-------------------

* Limited model selection (primarily LeNet-5)
* Small image sizes recommended (28x28, 32x32)
* Grayscale images work best
* No data augmentation built-in

Example: MNIST Digit Recognition
--------------------------------

.. code-block:: yaml

   common:
     target_module: 'image'
     task_type: 'image_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'mnist'
     input_data_path: '/path/to/mnist_dataset'

   training:
     model_name: 'Lenet5'
     training_epochs: 10
     batch_size: 64

   testing: {}
   compilation: {}

Preparing Image Data
--------------------

1. Resize images to target size (e.g., 28x28)
2. Convert to grayscale if using grayscale models
3. Organize into class folders

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

* Keep image sizes small for MCU deployment
* Use grayscale when color isn't essential
* Consider converting image tasks to time-series if possible
  (e.g., row-by-row scanning)

See Also
--------

* :doc:`timeseries_classification` - More extensive support
* `TI Edge AI <https://www.ti.com/tool/EDGE-AI-STUDIO>`_ - For advanced image tasks
