==========================
MNIST Image Classification
==========================

Handwritten digit recognition using the MNIST dataset.

Overview
--------

This example demonstrates image classification on microcontrollers using
the classic MNIST handwritten digit dataset. It shows how to train and
deploy a CNN model for recognizing digits 0-9.

**Application**: OCR, document processing, embedded vision

**Task Type**: Image Classification

**Data Type**: Grayscale images (28x28 pixels)

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'image'
     task_type: 'image_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'mnist'
     input_data_path: '/path/to/mnist'

   training:
     model_name: 'Lenet5'
     training_epochs: 10
     batch_size: 64

   testing: {}
   compilation: {}

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/MNIST_image_classification/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\MNIST_image_classification\config.yaml

Dataset Details
---------------

**MNIST Dataset**:

* 60,000 training images
* 10,000 test images
* 28x28 grayscale images
* 10 classes (digits 0-9)

The dataset is automatically downloaded when you run the example.

**Input Format**:

* Single-channel (grayscale) images
* 28x28 pixel resolution
* Normalized pixel values (0-1)

Model Architecture
------------------

**LeNet-5**:

The classic LeNet-5 architecture is used:

.. code-block:: text

   Input (28x28x1)
   ├── Conv1: 6 filters, 5x5
   ├── MaxPool: 2x2
   ├── Conv2: 16 filters, 5x5
   ├── MaxPool: 2x2
   ├── FC1: 120 units
   ├── FC2: 84 units
   └── Output: 10 classes

Memory Requirements
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Component
     - Size
     - Notes
   * - Model parameters
     - ~60KB
     - LeNet-5
   * - Input buffer
     - 784 bytes
     - 28x28 pixels
   * - Inference RAM
     - ~10KB
     - Activations

See Also
--------

* :doc:`/task_types/image_classification` - Image classification overview
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/MNIST_image_classification>`_
