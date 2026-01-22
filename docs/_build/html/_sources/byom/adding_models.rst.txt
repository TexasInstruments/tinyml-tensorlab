===========================
Adding Custom Models
===========================

This guide explains how to add your own neural network architectures
to Tiny ML Tensorlab's model zoo.

Overview
--------

You can extend the model zoo with custom architectures that:

* Fit your specific accuracy/size requirements
* Implement novel layer combinations
* Target specific hardware constraints

Model Definition Structure
--------------------------

Models are defined in the TinyVerse repository using PyTorch:

.. code-block:: text

   tinyml-tinyverse/
   └── tinyml_tinyverse/
       └── common/
           └── models/
               ├── generic_model_spec.py    # Base class
               ├── generic_timeseries.py    # Time series models
               └── your_custom_model.py     # Your model

Base Class: GenericModelWithSpec
--------------------------------

All models inherit from ``GenericModelWithSpec``:

.. code-block:: python

   from tinyml_tinyverse.common.models.generic_model_spec import GenericModelWithSpec
   from tinyml_tinyverse.common.utils import py_utils

   class MY_CUSTOM_MODEL(GenericModelWithSpec):
       def __init__(self, config, input_features=128, variables=1, num_classes=3):
           super().__init__(
               config,
               input_features=input_features,
               variables=variables,
               num_classes=num_classes
           )
           self.model_spec = self.gen_model_spec()
           self._init_model_from_spec(self.model_spec)

       def gen_model_spec(self):
           """Define model architecture."""
           layers = py_utils.DictPlus()

           # Add layers here

           return dict(model_spec=layers)

Layer Types
-----------

Available layer types for model specification:

**Convolutional Layers:**

.. code-block:: python

   # Standard convolution
   layers += {'conv1': dict(
       type='ConvBNReLULayer',
       in_channels=self.variables,
       out_channels=8,
       kernel_size=(5, 1),
       stride=(1, 1),
       padding=(2, 0)
   )}

   # Depthwise convolution
   layers += {'dwconv': dict(
       type='DWConvBNReLULayer',
       in_channels=8,
       out_channels=8,
       kernel_size=(3, 1),
       stride=(1, 1)
   )}

   # Pointwise convolution
   layers += {'pwconv': dict(
       type='PWConvBNReLULayer',
       in_channels=8,
       out_channels=16
   )}

**Pooling Layers:**

.. code-block:: python

   # Max pooling
   layers += {'pool1': dict(
       type='MaxPoolLayer',
       kernel_size=(2, 1),
       stride=(2, 1)
   )}

   # Average pooling
   layers += {'avgpool': dict(
       type='AvgPoolLayer',
       kernel_size=(4, 1),
       stride=(4, 1)
   )}

   # Global average pooling
   layers += {'gap': dict(
       type='GlobalAvgPoolLayer'
   )}

**Fully Connected Layers:**

.. code-block:: python

   # Reshape for FC
   layers += {'flatten': dict(
       type='ReshapeLayer',
       ndim=2
   )}

   # Linear layer
   layers += {'fc': dict(
       type='LinearLayer',
       in_features=64,
       out_features=self.num_classes
   )}

**Other Layers:**

.. code-block:: python

   # Batch normalization
   layers += {'bn': dict(
       type='BatchNormLayer',
       num_features=16
   )}

   # Dropout
   layers += {'dropout': dict(
       type='DropoutLayer',
       p=0.5
   )}

   # Activation
   layers += {'relu': dict(
       type='ReLULayer'
   )}

Complete Model Example
----------------------

A complete classification model:

.. code-block:: python

   class CLS_CUSTOM_1k(GenericModelWithSpec):
       """Custom 1k parameter classification model."""

       def __init__(self, config, input_features=128, variables=1, num_classes=3):
           super().__init__(
               config,
               input_features=input_features,
               variables=variables,
               num_classes=num_classes
           )
           self.model_spec = self.gen_model_spec()
           self._init_model_from_spec(self.model_spec)

       def gen_model_spec(self):
           layers = py_utils.DictPlus()

           # First conv: 1 input channel -> 4 output channels
           layers += {'0': dict(
               type='ConvBNReLULayer',
               in_channels=self.variables,
               out_channels=4,
               kernel_size=(5, 1),
               stride=(1, 1),
               padding=(2, 0)
           )}

           # Pooling
           layers += {'1': dict(
               type='MaxPoolLayer',
               kernel_size=(2, 1),
               stride=(2, 1)
           )}

           # Second conv
           layers += {'2': dict(
               type='ConvBNReLULayer',
               in_channels=4,
               out_channels=8,
               kernel_size=(3, 1),
               stride=(1, 1),
               padding=(1, 0)
           )}

           # Pooling
           layers += {'3': dict(
               type='MaxPoolLayer',
               kernel_size=(2, 1),
               stride=(2, 1)
           )}

           # Flatten
           layers += {'4': dict(
               type='ReshapeLayer',
               ndim=2
           )}

           # Calculate FC input size based on input_features and pooling
           fc_input = 8 * (self.input_features // 4)

           # Classifier
           layers += {'5': dict(
               type='LinearLayer',
               in_features=fc_input,
               out_features=self.num_classes
           )}

           return dict(model_spec=layers)

NPU-Compatible Model
--------------------

For NPU devices, follow these constraints:

.. code-block:: python

   class CLS_CUSTOM_NPU(GenericModelWithSpec):
       """NPU-compatible custom model."""

       def __init__(self, config, input_features=128, variables=1, num_classes=3):
           super().__init__(
               config,
               input_features=input_features,
               variables=variables,
               num_classes=num_classes
           )
           self.model_spec = self.gen_model_spec()
           self._init_model_from_spec(self.model_spec)

       def gen_model_spec(self):
           layers = py_utils.DictPlus()

           # First conv: variables must be 1, out_channels multiple of 4
           layers += {'0': dict(
               type='ConvBNReLULayer',
               in_channels=1,  # FCONV requires 1
               out_channels=4,  # Multiple of 4
               kernel_size=(5, 1),  # Height <= 7
               stride=(1, 1)
           )}

           # GCONV: all channels multiple of 4, kernel height <= 7
           layers += {'1': dict(
               type='ConvBNReLULayer',
               in_channels=4,
               out_channels=8,
               kernel_size=(5, 1),
               stride=(1, 1)
           )}

           # MaxPool: kernel <= 4x4
           layers += {'2': dict(
               type='MaxPoolLayer',
               kernel_size=(2, 1),
               stride=(2, 1)
           )}

           layers += {'3': dict(
               type='ConvBNReLULayer',
               in_channels=8,
               out_channels=16,
               kernel_size=(5, 1),
               stride=(1, 1)
           )}

           layers += {'4': dict(
               type='MaxPoolLayer',
               kernel_size=(2, 1),
               stride=(2, 1)
           )}

           layers += {'5': dict(
               type='ReshapeLayer',
               ndim=2
           )}

           # FC: in_features >= 16
           layers += {'6': dict(
               type='LinearLayer',
               in_features=16 * (self.input_features // 4),
               out_features=self.num_classes
           )}

           return dict(model_spec=layers)

Registering Your Model
----------------------

After defining the model, register it in the model registry:

**Step 1: Add to model file**

Add your class to the appropriate file in ``models/``.

**Step 2: Update __init__.py**

In ``models/__init__.py``:

.. code-block:: python

   from .your_custom_model import CLS_CUSTOM_1k, CLS_CUSTOM_NPU

   # Add to model registry
   MODEL_REGISTRY = {
       # ... existing models ...
       'CLS_CUSTOM_1k': CLS_CUSTOM_1k,
       'CLS_CUSTOM_NPU': CLS_CUSTOM_NPU,
   }

**Step 3: Test registration**

.. code-block:: python

   from tinyml_tinyverse.common.models import MODEL_REGISTRY

   # Verify model is registered
   print('CLS_CUSTOM_1k' in MODEL_REGISTRY)  # Should print True

Using Your Custom Model
-----------------------

Reference in configuration:

.. code-block:: yaml

   training:
     model_name: 'CLS_CUSTOM_1k'  # Your model name
     training_epochs: 30

Testing Your Model
------------------

Before deploying, test your model:

**Unit Test:**

.. code-block:: python

   import torch
   from your_custom_model import CLS_CUSTOM_1k

   # Create model
   config = {}  # Your config
   model = CLS_CUSTOM_1k(
       config,
       input_features=512,
       variables=1,
       num_classes=3
   )

   # Test forward pass
   x = torch.randn(1, 1, 512, 1)
   output = model(x)

   print(f"Input shape: {x.shape}")
   print(f"Output shape: {output.shape}")
   print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

**Export Test:**

.. code-block:: python

   import torch.onnx

   # Export to ONNX
   torch.onnx.export(
       model,
       x,
       "custom_model.onnx",
       input_names=['input'],
       output_names=['output'],
       opset_version=11
   )

   print("ONNX export successful")

Best Practices
--------------

1. **Follow naming conventions**: Use task prefix (``CLS_``, ``AD_``, ``FCST_``)
2. **Document your model**: Add docstrings and comments
3. **Test thoroughly**: Verify shapes and parameter counts
4. **Consider NPU constraints**: If targeting NPU devices
5. **Start from existing models**: Modify rather than create from scratch

Next Steps
----------

* See :doc:`compilation_only` to compile external models
* Review :doc:`/devices/npu_guidelines` for NPU constraints
* Check :doc:`/features/quantization` for quantization support
