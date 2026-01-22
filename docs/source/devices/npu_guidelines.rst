==============
NPU Guidelines
==============

This guide covers the constraints and best practices for designing models
that run on TI's Neural Processing Unit (TINPU).

NPU-Enabled Devices
-------------------

* **F28P55** - C2000 family
* **AM13E2** - MSPM33C family
* **MSPM0G5187** - MSPM0 family

Layer Constraints
-----------------

Models running on the NPU must follow these constraints:

**First Convolution Layer (FCONV)**

.. list-table::
   :widths: 30 70

   * - Input channels
     - Must be 1
   * - Output channels
     - Must be multiple of 4
   * - Kernel width
     - Maximum 8

**Generic Convolution Layer (GCONV)**

.. list-table::
   :widths: 30 70

   * - Input channels
     - Must be multiple of 4
   * - Output channels
     - Must be multiple of 4
   * - **Kernel height**
     - **Maximum 7** (critical constraint)
   * - Stride
     - Supported: (1,1), (2,1), (2,2)

**Depthwise Convolution (DWCONV)**

.. list-table::
   :widths: 30 70

   * - Groups
     - Must equal input channels (true depthwise)
   * - Kernel width
     - Maximum 7

**Pointwise Convolution (PWCONV)**

.. list-table::
   :widths: 30 70

   * - Kernel size
     - Must be (1, 1)
   * - Channels
     - Must be multiples of 4

**Pooling Layers**

.. list-table::
   :widths: 30 70

   * - MaxPool kernel
     - Maximum 4x4
   * - AvgPool (global)
     - Input size must satisfy (H × W) > 2

**Fully Connected (FC) Layer**

.. list-table::
   :widths: 30 70

   * - Input features (8-bit)
     - Minimum 16
   * - Input features (4-bit)
     - Minimum 8

Using NPU-Compatible Models
---------------------------

Use model names ending in ``_NPU``:

.. code-block:: yaml

   training:
     model_name: 'CLS_1k_NPU'    # NPU-compatible
     # not: model_name: 'CLS_1k'  # Non-NPU version

Available NPU models:

* Classification: ``CLS_100_NPU`` through ``CLS_55k_NPU``
* Regression: ``REGR_500_NPU`` through ``REGR_20k_NPU``
* Anomaly Detection: ``AD_500_NPU`` through ``AD_20k_NPU``
* Forecasting: ``FCST_500_NPU`` through ``FCST_20k_NPU``

Channel Multiples of 4
----------------------

All intermediate channels must be multiples of 4:

**Correct:**

.. code-block:: text

   Input: 1 channel
   Conv1: 1 → 4 channels
   Conv2: 4 → 8 channels
   Conv3: 8 → 16 channels
   FC: 16 → num_classes

**Incorrect:**

.. code-block:: text

   Input: 1 channel
   Conv1: 1 → 3 channels    # NOT multiple of 4
   Conv2: 3 → 6 channels    # NOT multiple of 4

Kernel Size Restrictions
------------------------

The most common issue is kernel height exceeding 7:

**Correct:**

.. code-block:: yaml

   # Kernel (5, 1) - height 5 is OK
   # Kernel (7, 1) - height 7 is OK (maximum)

**Incorrect:**

.. code-block:: yaml

   # Kernel (8, 1) - height 8 exceeds limit
   # Kernel (9, 1) - NOT supported

Compilation Preset
------------------

For NPU devices, use the appropriate compilation preset:

.. code-block:: yaml

   compilation:
     enable: True
     preset_name: 'compress_npu_layer_data'  # For NPU devices

The ``compress_npu_layer_data`` preset optimizes memory layout for NPU.

Custom NPU-Compatible Models
----------------------------

When creating custom models for NPU, follow this template:

.. code-block:: python

   class MY_NPU_MODEL(GenericModelWithSpec):
       def __init__(self, config, input_features=128, variables=1, num_classes=3):
           super().__init__(config, input_features=input_features,
                           variables=variables, num_classes=num_classes)
           self.model_spec = self.gen_model_spec()
           self._init_model_from_spec(...)

       def gen_model_spec(self):
           layers = py_utils.DictPlus()

           # First conv: in_channels=1 (variables), out_channels=4 (multiple of 4)
           layers += {'0': dict(type='ConvBNReLULayer',
                               in_channels=self.variables,  # Must be 1 for FCONV
                               out_channels=4,              # Multiple of 4
                               kernel_size=(5, 1),          # Height ≤ 7
                               stride=(1, 1))}

           # Subsequent convs: all channels multiple of 4
           layers += {'1': dict(type='ConvBNReLULayer',
                               in_channels=4,
                               out_channels=8,
                               kernel_size=(5, 1),
                               stride=(1, 1))}

           # MaxPool: kernel ≤ 4
           layers += {'2': dict(type='MaxPoolLayer',
                               kernel_size=(2, 1),
                               stride=(2, 1))}

           # FC: input features ≥ 16
           layers += {'3': dict(type='ReshapeLayer', ndim=2)}
           layers += {'4': dict(type='LinearLayer',
                               in_features=...,  # ≥ 16
                               out_features=self.num_classes)}

           return dict(model_spec=layers)

Troubleshooting NPU Compilation
-------------------------------

**"Channel count not multiple of 4"**

Adjust your model architecture to use channels that are multiples of 4.

**"Kernel size exceeds limit"**

Reduce kernel height to 7 or less. Use multiple smaller kernels instead.

**"Unsupported layer type"**

Check that all layers are in the supported list (Conv, Pool, FC, BN, ReLU).

**"FC input features too small"**

Ensure the FC layer receives at least 16 input features.

Performance Comparison
----------------------

Example inference times (approximate):

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Model
     - CPU (F28P55)
     - NPU (F28P55)
     - Speedup
   * - CLS_1k
     - 2000 µs
     - 150 µs
     - ~13x
   * - CLS_4k
     - 5000 µs
     - 300 µs
     - ~17x
   * - CLS_13k
     - 15000 µs
     - 600 µs
     - ~25x

Actual performance varies by model architecture and input size.
