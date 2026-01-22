=================
Compilation Only
=================

If you have a pre-trained model (ONNX format) from another framework,
you can use Tiny ML Tensorlab to compile it for TI microcontrollers.

Overview
--------

The compilation-only workflow:

1. Train your model externally (PyTorch, TensorFlow, etc.)
2. Export to ONNX format
3. Use Tiny ML Tensorlab to compile for target device
4. Deploy compiled model to MCU

This is useful when:

* You have existing models from other frameworks
* You prefer a different training environment
* You need specific model architectures not in ModelZoo

ONNX Model Requirements
-----------------------

Your ONNX model must meet these requirements:

**Supported Operations:**

* Convolution (Conv, ConvTranspose)
* Pooling (MaxPool, AveragePool, GlobalAveragePool)
* Fully Connected (Gemm, MatMul)
* Activation (ReLU, Sigmoid, Tanh, Softmax)
* Normalization (BatchNormalization)
* Arithmetic (Add, Sub, Mul, Div)
* Reshape (Reshape, Flatten, Squeeze, Unsqueeze)

**Data Types:**

* Float32 (will be quantized)
* Int8 (already quantized)

**Input/Output:**

* Single input tensor
* Single output tensor (or multiple for specific tasks)

Compilation Configuration
-------------------------

Create a YAML configuration for compilation:

.. code-block:: yaml

   common:
     task_type: 'byom_compilation'  # Compilation-only mode
     target_device: 'F28P55'        # Your target device

   byom:
     enable: True
     onnx_model_path: '/path/to/your/model.onnx'
     input_shape: [1, 1, 512, 1]    # Your model's input shape

   compilation:
     enable: True
     preset_name: 'compress_npu_layer_data'  # or 'default_preset'

   # Quantization (if model is float)
   quantization:
     enable: True
     type: 'ptq'  # Post-training quantization
     calibration_data: '/path/to/calibration/data.npy'
     num_samples: 500

Running Compilation
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelmaker
         python tinyml_modelmaker/run_tinyml_modelmaker.py byom_config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelmaker
         python tinyml_modelmaker\run_tinyml_modelmaker.py byom_config.yaml

Calibration Data
----------------

For PTQ, provide calibration data:

**Format:**

NumPy array file (.npy) containing representative inputs:

.. code-block:: python

   import numpy as np

   # Generate calibration data
   # Shape: [num_samples, channels, height, width]
   calibration_data = np.random.randn(500, 1, 512, 1).astype(np.float32)

   # Or load from your actual data
   calibration_data = np.load('my_data.npy')

   # Save for compilation
   np.save('calibration_data.npy', calibration_data)

**Configuration:**

.. code-block:: yaml

   quantization:
     calibration_data: 'calibration_data.npy'
     num_samples: 500
     calibration_method: 'minmax'  # or 'histogram', 'entropy'

Pre-Quantized Models
--------------------

If your model is already INT8 quantized:

.. code-block:: yaml

   byom:
     enable: True
     onnx_model_path: '/path/to/quantized_model.onnx'
     input_shape: [1, 1, 512, 1]
     already_quantized: True

   quantization:
     enable: False  # Skip quantization

Output Artifacts
----------------

After compilation:

.. code-block:: text

   .../byom_output/
   ├── mod.a                    # Compiled library
   ├── mod.h                    # Interface header
   ├── model_config.h           # Configuration
   └── compilation_log.txt      # Compilation details

NPU Compilation
---------------

For NPU devices, your ONNX model must follow NPU constraints:

**Channel Requirements:**

* All intermediate channels must be multiples of 4
* First layer input channels = 1

**Kernel Constraints:**

* Convolution kernel height ≤ 7
* MaxPool kernel ≤ 4x4

**Verification:**

.. code-block:: python

   import onnx

   model = onnx.load('your_model.onnx')

   # Check model structure
   for node in model.graph.node:
       if node.op_type == 'Conv':
           # Check kernel size
           for attr in node.attribute:
               if attr.name == 'kernel_shape':
                   kernel_h = attr.ints[0]
                   if kernel_h > 7:
                       print(f"Warning: Kernel height {kernel_h} > 7")

If your model doesn't meet NPU constraints, you have options:

1. Modify and retrain the model
2. Target a non-NPU device
3. Accept CPU-only inference on NPU device

Example: External PyTorch Model
-------------------------------

**Step 1: Train in PyTorch**

.. code-block:: python

   import torch
   import torch.nn as nn

   class MyModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv1 = nn.Conv2d(1, 4, kernel_size=(5, 1), padding=(2, 0))
           self.bn1 = nn.BatchNorm2d(4)
           self.pool = nn.MaxPool2d((2, 1))
           self.conv2 = nn.Conv2d(4, 8, kernel_size=(3, 1), padding=(1, 0))
           self.bn2 = nn.BatchNorm2d(8)
           self.fc = nn.Linear(8 * 128, 3)

       def forward(self, x):
           x = torch.relu(self.bn1(self.conv1(x)))
           x = self.pool(x)
           x = torch.relu(self.bn2(self.conv2(x)))
           x = self.pool(x)
           x = x.view(x.size(0), -1)
           return self.fc(x)

   # Train your model
   model = MyModel()
   # ... training code ...

**Step 2: Export to ONNX**

.. code-block:: python

   # Export
   dummy_input = torch.randn(1, 1, 512, 1)
   torch.onnx.export(
       model,
       dummy_input,
       'my_model.onnx',
       input_names=['input'],
       output_names=['output'],
       opset_version=11
   )

**Step 3: Create calibration data**

.. code-block:: python

   import numpy as np

   # Use samples from your training data
   calibration = train_data[:500]  # First 500 samples
   np.save('calibration.npy', calibration)

**Step 4: Configure and compile**

.. code-block:: yaml

   # byom_config.yaml
   common:
     task_type: 'byom_compilation'
     target_device: 'F28P55'

   byom:
     enable: True
     onnx_model_path: 'my_model.onnx'
     input_shape: [1, 1, 512, 1]

   quantization:
     enable: True
     type: 'ptq'
     calibration_data: 'calibration.npy'
     num_samples: 500

   compilation:
     enable: True
     preset_name: 'compress_npu_layer_data'

**Step 5: Run compilation**

.. code-block:: bash

   python tinyml_modelmaker/run_tinyml_modelmaker.py byom_config.yaml

Example: TensorFlow Model
-------------------------

**Step 1: Export from TensorFlow to ONNX**

.. code-block:: python

   import tensorflow as tf
   import tf2onnx

   # Your trained TF model
   tf_model = tf.keras.models.load_model('my_tf_model.h5')

   # Convert to ONNX
   spec = (tf.TensorSpec((1, 1, 512, 1), tf.float32, name="input"),)
   model_proto, _ = tf2onnx.convert.from_keras(
       tf_model,
       input_signature=spec,
       opset=11
   )

   # Save
   with open('tf_model.onnx', 'wb') as f:
       f.write(model_proto.SerializeToString())

**Step 2: Continue with compilation as above**

Troubleshooting
---------------

**Unsupported Operation:**

.. code-block:: text

   Error: Operation 'MyCustomOp' not supported

Solution: Replace with supported operations or simplify model.

**Shape Mismatch:**

.. code-block:: text

   Error: Input shape mismatch

Solution: Verify ``input_shape`` in config matches ONNX model.

**Quantization Error:**

.. code-block:: text

   Error: Quantization failed

Solutions:

* Provide more calibration data
* Use different calibration method
* Check for unsupported dynamic shapes

**NPU Constraint Violation:**

.. code-block:: text

   Error: Channel count 5 not multiple of 4

Solution: Modify model architecture to meet NPU requirements.

Best Practices
--------------

1. **Verify ONNX model first**: Use onnxruntime to test
2. **Match training preprocessing**: Calibration data should match inference
3. **Test on representative data**: Ensure accuracy after quantization
4. **Start with non-NPU**: Debug on CPU target first
5. **Compare outputs**: Validate compiled model matches original

Next Steps
----------

* See :doc:`adding_models` to add models to ModelZoo
* Review :doc:`/deployment/ccs_integration` for deployment
* Check :doc:`/devices/npu_guidelines` for NPU constraints
