==========================
MNIST Image Classification
==========================

Edge AI solution for real-time handwritten digit recognition on the
`MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ microcontroller,
using a LeNet-5 CNN to classify single-digit handwritten character images (0-9).

Overview
--------

Handwritten digit recognition is a fundamental task for document processing and
automation workflows. Manual data entry is slow and error-prone, and traditional
template-matching approaches struggle with the natural variability in human
handwriting. Edge AI enables robust, real-time recognition directly on
resource-constrained microcontrollers, eliminating the need for cloud
connectivity.

This example trains and deploys a LeNet-5 convolutional neural network on the
classic MNIST dataset to classify grayscale 28x28 images of handwritten digits
(0-9) with approximately 99% accuracy.

* **Application**: OCR, document processing, embedded vision, automation
* **Task Type**: Image Classification
* **Data Type**: Grayscale images (28x28 pixels)

Key Targets
-----------

* Real-time classification of handwritten digit images
* High accuracy (~99%)
* Low memory footprint suitable for MCU deployment
* UART-based communication with host GUI for interactive demo

System Components
-----------------

**Hardware**

* `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ microcontroller
* UART connection to host PC

**Software**

* Code Composer Studio (CCS) 12.x or later
* `MSPM0 SDK <https://www.ti.com/tool/MSPM0-SDK>`_ 2.08.00 or later

Running the Example
-------------------

.. tabs::

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         .\run_tinyml_modelzoo.bat examples\MNIST_image_classification\config_image_classification_mnist.yaml

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/MNIST_image_classification/config_image_classification_mnist.yaml

Dataset
-------

The ``mnist_image_classification`` dataset is based on the classic
`MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset, obtained via
torchvision. It contains 28x28 grayscale PNG images across 10 classes
(digits 0-9).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Value
   * - Dataset name
     - ``mnist_image_classification``
   * - Number of classes
     - 10 (digits 0-9)
   * - Image size
     - 28x28 pixels, grayscale
   * - Image format
     - PNG
   * - Download URL
     - `mnist_classes.zip <https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/mnist_classes.zip>`_

Feature Extraction
------------------

The feature extraction pipeline prepares raw images for inference:

1. **Image Input** -- Read 28x28 image
2. **Grayscale Conversion** -- Convert to single-channel grayscale
3. **Resize** -- Resize to 28x28 (if needed)
4. **Tensor Conversion** -- Convert to tensor format
5. **Normalization** -- Normalize with mean=0.1307, scale=0.3081

Model Architecture
------------------

**LeNet-5** (Default)

The classic `LeNet-5 <https://en.wikipedia.org/wiki/LeNet>`_ CNN architecture
is used, with approximately 60,000 parameters.

.. code-block:: text

   Input (28x28x1)
   +-- Conv1: 6 filters, 5x5
   +-- MaxPool: 2x2
   +-- Conv2: 16 filters, 5x5
   +-- MaxPool: 2x2
   +-- FC1: 120 units
   +-- FC2: 84 units
   +-- Output: 10 classes

Training Configuration
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Value
   * - Model
     - ``Lenet5``
   * - Batch size
     - 64
   * - Learning rate
     - 0.1
   * - Epochs
     - 14
   * - Quantization
     - INT8
   * - Compilation target
     - `TI NNC <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_ for CPU inference

Expected Results
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Value
   * - Accuracy
     - ~99%

Supported Devices
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Device
     - Notes
   * - `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_
     - Primary target
   * - MSPM0G3507
     - Supported
   * - MSPM0G3519
     - Supported
   * - MSPM33C32
     - Supported

References
----------

* `MSPM0G5187 Product Page <https://www.ti.com/product/MSPM0G5187>`_
* `TI Neural Network Compiler (NNC) User Guide <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_
* `TinyML Tensorlab on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main>`_
* `MNIST Dataset <http://yann.lecun.com/exdb/mnist/>`_
* `LeNet Architecture <https://en.wikipedia.org/wiki/LeNet>`_
* `MSPM0 SDK <https://www.ti.com/tool/MSPM0-SDK>`_
