==============================================
Standalone Quantization Examples
==============================================

The ``tinyml-modeloptimization`` package includes standalone Python examples
demonstrating direct use of the quantization wrappers. These examples are
located in ``tinyml-modeloptimization/torchmodelopt/examples/`` and can be
run independently of the Tiny ML Tensorlab YAML-based toolchain.

.. note::

   These examples are for users who want to integrate quantization into
   their own PyTorch training scripts. For most users, the YAML-based
   toolchain (see :doc:`quantization`) is recommended.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Example
     - Dataset
     - Description
   * - FMNIST Image Classification
     - Fashion MNIST
     - Beginner example: LinearReLU model, QAT, TINPU export, ONNX
       inference, node renaming
   * - Audio Keyword Spotting
     - Speech Commands v2
     - DSCNN model, QAT/PTQ, mixed precision, bias calibration
       (MLPerf Tiny benchmark)
   * - Motor Fault Classification
     - Motor vibration CSV
     - CNN on time series, QAT/PTQ, 2b/4b/8b weights, confusion matrix
   * - MNIST Digit Classification
     - MNIST
     - LeNet-5, QAT/PTQ, 4/8-bit quantization, ONNX export
   * - Torque Regression
     - Torque measurement CSV
     - CNN regression, QAT/PTQ, R2/SMAPE metrics

FMNIST Image Classification
----------------------------

A beginner-friendly example using the Fashion MNIST dataset (60,000 training
+ 10,000 test samples, 10 classes).

**Model**: Simple neural network with Linear + ReLU layers.

**Workflow:**

1. Create train and test dataloaders from ``torchvision.datasets``
2. Define a classification neural network (LinearReLU stack)
3. Train and test the float model (5 epochs)
4. Wrap with ``TINPUTinyMLQATFxModule`` for quantization-aware training
5. Train and test the quantized model
6. Convert from PyTorch QDQ layers to TI NPU int8 layers
7. Rename input node to ``'input'`` for inference compatibility
8. Export as ``fmnist_int8.onnx``

**Key Code:**

.. code-block:: python

   from tinyml_torchmodelopt.quantization import TINPUTinyMLQATFxModule

   # After float training:
   ti_model = TINPUTinyMLQATFxModule(model, total_epochs=5)

   # QAT training loop (same loss/optimizer as float)
   for epoch in range(5):
       for images, targets in train_loader:
           output = ti_model(images)
           loss = criterion(output, targets)
           loss.backward()
           optimizer.step()

   # Convert and export
   ti_model.eval()
   ti_model = ti_model.convert()
   ti_model.export(dummy_input, 'fmnist_int8.onnx', input_names=['input'])

**Run:**

.. code-block:: bash

   cd tinyml-modeloptimization/torchmodelopt/examples/fmnist_image_classification
   python fmnist_tinpu_qat.py

Audio Keyword Spotting
-----------------------

An advanced example based on the MLPerf Tiny keyword spotting benchmark.
Uses a Depthwise Separable Convolutional Neural Network (DSCNN) to identify
10 keywords from the Google Speech Commands v2 dataset.

**Model**: DSCNN (2D conv + 4 depthwise separable conv blocks + global
average pooling + FC layer).

**Dataset**: Modified Speech Commands v2 with 12 classes (10 keywords +
"unknown" + "silence"). Training/validation/test split: 80%/10%/10%.

**Features Demonstrated:**

* QAT and PTQ workflows
* Mixed precision quantization
* Bias calibration (``bias_calibration_factor``)
* Cosine annealing LR scheduler
* ONNX export for NPU deployment

**Workflow:**

1. Download and prepare Speech Commands v2 dataset
2. Build DSCNN model with batch normalization and dropout
3. Train float model (10 epochs with cosine LR schedule)
4. Wrap with ``TINPUTinyMLQuantFxModule``
5. Perform QAT or PTQ calibration
6. Export as ``quant_kws.onnx``

**Run:**

.. code-block:: bash

   cd tinyml-modeloptimization/torchmodelopt/examples/audio_keyword_spotting
   python main.py

Motor Fault Time Series Classification
----------------------------------------

Demonstrates quantization for time series classification using motor
vibration sensor data. The CNN model classifies fault conditions from
3-axis accelerometer readings.

**Model**: Small CNN for vibration data classification.

**Data Format**: CSV with columns ``Vibx``, ``Viby``, ``Vibz``, ``Target``.
Rows are segmented into sliding windows controlled by ``WINDOW_LENGTH``
and ``WINDOW_OFFSET``.

**Features Demonstrated:**

* Both TINPU and Generic quantization device types
* QAT and PTQ workflows
* 2-bit, 4-bit, and 8-bit weight quantization
* Confusion matrix evaluation
* ONNX Runtime inference validation

**Configuration** (edit constants in script):

.. code-block:: python

   QUANTIZATION_METHOD = 'QAT'      # or 'PTQ'
   WEIGHT_BITWIDTH = 8              # 2, 4, or 8
   ACTIVATION_BITWIDTH = 8          # typically 8
   QUANTIZATION_DEVICE_TYPE = 'TINPU'  # or 'GENERIC'
   WINDOW_LENGTH = 256
   WINDOW_OFFSET = 64

**Script Structure:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Purpose
   * - ``get_dataset_from_csv()``
     - Load CSV data and create sliding windows
   * - ``get_nn_model()``
     - Create the CNN model
   * - ``train_model()``
     - Float and QAT training loop
   * - ``calibrate_model()``
     - PTQ calibration pass
   * - ``get_quant_model()``
     - Select TINPU/Generic and QAT/PTQ wrapper
   * - ``export_model()``
     - Export to ONNX format
   * - ``validate_saved_model()``
     - ONNX Runtime inference validation

**Run:**

.. code-block:: bash

   cd tinyml-modeloptimization/torchmodelopt/examples/motor_fault_time_series_classification
   python motor_fault_classification_tinpu_quant.py

MNIST Digit Classification
---------------------------

Uses the classic LeNet-5 architecture on MNIST, demonstrating a complete
TinyML workflow from training to quantization and ONNX export.

**Model**: LeNet-5 CNN:

* Conv1: 8 filters, 3x3, BatchNorm + ReLU + MaxPool
* Conv2: 16 filters, 3x3, BatchNorm + ReLU + MaxPool
* FC1: 400 → 120
* FC2: 120 → 84
* FC3: 84 → 10

**Features Demonstrated:**

* QAT and PTQ workflows
* 4-bit and 8-bit quantization
* Cosine annealing LR scheduler
* Float and quantized ONNX export
* Complete 14-epoch training pipeline

**Workflow:**

1. Download MNIST (28x28 grayscale, 60K train + 10K test)
2. Normalize with mean=0.1307, std=0.3081
3. Train LeNet-5 for 14 epochs (SGD + CosineAnnealingLR)
4. Wrap with ``TINPUTinyMLQuantFxModule``
5. Perform QAT or PTQ
6. Validate quantized model accuracy
7. Export both float and quantized ONNX models

**Run:**

.. code-block:: bash

   cd tinyml-modeloptimization/torchmodelopt/examples/mnist_lenet5_classification
   python main.py

Torque Time Series Regression
------------------------------

Demonstrates quantization for a regression task (continuous value
prediction) using sensor time-series data from motor torque measurements.

**Model**: Small CNN for torque prediction.

**Data**: CSV dataset with sensor columns and a ``torque`` target column.
Available from TI's public dataset server. Data is segmented into
sliding windows.

**Features Demonstrated:**

* QAT quantization for regression models
* R2 and SMAPE evaluation metrics
* TINPU and Generic device types
* ONNX Runtime inference validation

**Configuration** (edit constants in script):

.. code-block:: python

   QUANTIZATION_METHOD = 'QAT'
   WEIGHT_BITWIDTH = 8
   ACTIVATION_BITWIDTH = 8
   QUANTIZATION_DEVICE_TYPE = 'TINPU'

**Run:**

.. code-block:: bash

   cd tinyml-modeloptimization/torchmodelopt/examples/torque_time_series_regression
   python torque_regression_tinpu_quant.py

Quantization Guidance
---------------------

These best practices apply when using the wrappers directly:

**Choosing QAT vs PTQ:**

* **PTQ** (safe default): Use ``QUANTIZATION_METHOD = 'PTQ'`` with 8-bit
  weights and activations. Fast, requires only a calibration pass.
* **QAT** (better accuracy): Switch to QAT if PTQ accuracy degrades,
  especially for sub-8-bit quantization. Use a smaller learning rate
  and more epochs.

**Sub-8-bit Quantization:**

* For 4-bit or 2-bit weights: prefer QAT with per-channel weight
  quantization
* Careful tuning of calibration, clipping, and bias correction is
  important
* TINPU prefers symmetric per-channel weight quantization with
  power-of-two scales

**PTQ Calibration:**

* Use representative inputs (hundreds to a few thousand samples)
* Poor calibration causes large activation quantization errors
* Ensure calibration data covers the full input distribution

**ONNX Evaluation:**

Always validate the exported ONNX model before deploying to device:

.. code-block:: python

   import onnxruntime as ort

   session = ort.InferenceSession('model_int8.onnx')
   prediction = session.run(None, {'input': test_input.numpy()})

Next Steps
----------

* :doc:`quantization` - Quantization via the YAML-based toolchain
* :doc:`neural_architecture_search` - Automatic model architecture search
* :doc:`/deployment/npu_device_deployment` - Deploy quantized models to NPU devices
