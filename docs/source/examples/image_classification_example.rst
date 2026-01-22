============================
Image Classification Example
============================

This example demonstrates image classification using Tiny ML Tensorlab
to classify visual data on microcontrollers.

Overview
--------

* **Task**: Image classification (multi-class)
* **Application**: Visual quality inspection, object detection
* **Model**: MobileNet-based architectures
* **Input**: RGB or grayscale images

When to Use Image Classification
--------------------------------

Image classification is useful for:

* Visual quality inspection
* Simple object recognition
* Scene classification
* Presence/absence detection

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/image_classification/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\\image_classification\\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'image'
     task_type: 'generic_image_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'image_classification_example'
     input_data_path: '/path/to/image/dataset'

   data_processing_feature_extraction:
     image_size: [96, 96]  # Width x Height
     channels: 3           # RGB (3) or Grayscale (1)

   training:
     model_name: 'MobileNetV2_Small'
     training_epochs: 50
     batch_size: 32

   testing:
     enable: True

   compilation:
     enable: True

Dataset Format
--------------

Image datasets use folder-per-class structure:

.. code-block:: text

   my_image_dataset/
   ├── annotations.yaml
   └── classes/
       ├── class_a/
       │   ├── image_001.jpg
       │   ├── image_002.png
       │   └── ...
       ├── class_b/
       │   ├── image_001.jpg
       │   └── ...
       └── class_c/
           └── ...

**annotations.yaml:**

.. code-block:: yaml

   name: my_image_dataset
   description: Custom image classification dataset
   task_type: image_classification

**Supported formats:** JPEG, PNG, BMP

Image Size Considerations
-------------------------

Smaller images = faster inference but less detail:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Size
     - Memory
     - Inference Time
     - Detail
   * - 32x32
     - Very Low
     - Fastest
     - Low
   * - 64x64
     - Low
     - Fast
     - Moderate
   * - 96x96
     - Moderate
     - Moderate
     - Good
   * - 128x128
     - Higher
     - Slower
     - High

**Recommendation:** Start with 64x64 or 96x96 for most applications.

.. code-block:: yaml

   data_processing_feature_extraction:
     image_size: [64, 64]  # Start small, increase if needed
     channels: 3

Available Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Model
     - Parameters
     - Description
   * - ``MobileNetV2_Tiny``
     - ~10k
     - Minimal, simple tasks
   * - ``MobileNetV2_Small``
     - ~50k
     - Good balance
   * - ``MobileNetV2_Medium``
     - ~100k
     - Complex classification
   * - ``CustomCNN_Small``
     - ~20k
     - Simple custom architecture

**Note:** Image models are typically larger than time series models
due to 2D spatial processing requirements.

Expected Results
----------------

.. code-block:: text

   Training complete.

   Float32 Model:
   Accuracy: 95%+
   F1-Score: 0.94

   Quantized Model:
   Accuracy: 92%+

Grayscale vs RGB
----------------

**Grayscale (1 channel):**

* Smaller model input
* Faster inference
* Good when color is not important

.. code-block:: yaml

   data_processing_feature_extraction:
     channels: 1

**RGB (3 channels):**

* Full color information
* Larger model input
* Needed when color matters for classification

.. code-block:: yaml

   data_processing_feature_extraction:
     channels: 3

Data Augmentation
-----------------

Image augmentation improves model robustness:

.. code-block:: yaml

   data_processing_feature_extraction:
     augmentation:
       horizontal_flip: True
       vertical_flip: False
       rotation: 15          # degrees
       brightness: 0.2
       contrast: 0.2
       zoom: 0.1

**Common augmentations:**

* **Flip**: For symmetric objects
* **Rotation**: When orientation varies
* **Brightness/Contrast**: For lighting variation
* **Zoom/Crop**: For scale variation

Practical Applications
----------------------

**Visual Quality Inspection:**

Detect defects in manufactured parts:

.. code-block:: yaml

   common:
     target_module: 'image'
     task_type: 'generic_image_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'defect_inspection_dataset'

   data_processing_feature_extraction:
     image_size: [96, 96]
     channels: 1  # Grayscale for surface defects

   training:
     model_name: 'MobileNetV2_Small'

Classes: good, scratch, dent, contamination

**Presence Detection:**

Detect if an object is present:

.. code-block:: yaml

   common:
     task_type: 'generic_image_classification'

   dataset:
     dataset_name: 'presence_detection_dataset'
     # classes: present, absent

   data_processing_feature_extraction:
     image_size: [64, 64]
     channels: 3

   training:
     model_name: 'MobileNetV2_Tiny'

**Scene Classification:**

Classify environmental conditions:

.. code-block:: yaml

   common:
     task_type: 'generic_image_classification'

   dataset:
     dataset_name: 'scene_dataset'
     # classes: indoor, outdoor, low_light, etc.

   data_processing_feature_extraction:
     image_size: [96, 96]
     channels: 3

   training:
     model_name: 'MobileNetV2_Medium'

Memory Constraints
------------------

Image classification requires more memory than time series:

**Memory Budget:**

.. code-block:: text

   Input buffer: W × H × C × 4 bytes (float)
   Example: 96 × 96 × 3 × 4 = 110 KB

   Model weights: Varies by model
   Example: MobileNetV2_Small ≈ 200 KB

**Total:** Plan for 300-500 KB for image models

**Optimization Tips:**

* Use smaller image size
* Use grayscale if possible
* Choose quantized models (int8)
* Select device with sufficient memory

Inference Performance
---------------------

Image inference is slower than time series:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Device
     - Image Size
     - Model
     - Latency
   * - F28P55 (NPU)
     - 64x64
     - Small
     - ~5 ms
   * - F28P55 (NPU)
     - 96x96
     - Small
     - ~15 ms
   * - F28P55 (CPU)
     - 64x64
     - Small
     - ~50 ms

**Note:** Actual performance depends on model architecture.

Transfer Learning
-----------------

For better results with limited data, use pretrained models:

.. code-block:: yaml

   training:
     model_name: 'MobileNetV2_Small'
     pretrained: True        # Start from ImageNet weights
     freeze_backbone: False  # Fine-tune entire model
     training_epochs: 30

Transfer learning helps when:

* You have limited training images
* Your classes are similar to common objects
* You want faster convergence

Camera Integration
------------------

For device deployment, consider:

**Camera Interface:**

* DCMI/CSI for high-speed capture
* GPIO for simple cameras
* Frame buffer management

**Frame Rate:**

* Typical: 1-10 fps for classification
* Higher rates need faster inference

**Preprocessing:**

* Resize on device or camera
* Convert color space if needed
* Normalize pixel values

Troubleshooting
---------------

**Low accuracy:**

* Increase image size
* Use larger model
* Add more training data
* Apply appropriate augmentation

**Out of memory:**

* Reduce image size
* Use grayscale
* Choose smaller model
* Check device memory specs

**Slow inference:**

* Use NPU-compatible model
* Reduce image size
* Optimize model architecture

**Overfitting (train >> test accuracy):**

* Add more augmentation
* Reduce model complexity
* Increase training data

Limitations
-----------

Image classification on MCUs has limitations:

* **Resolution**: Limited to small images (32-128 pixels)
* **Complexity**: Cannot match server-side accuracy
* **Memory**: Large images exhaust RAM
* **Speed**: Real-time video difficult

**Best suited for:**

* Simple binary/few-class problems
* Controlled lighting conditions
* Fixed camera position
* Non-safety-critical applications

Next Steps
----------

* Review :doc:`/task_types/image_classification` details
* Learn about data preparation in :doc:`/byod/classification_format`
* Deploy to device: :doc:`/deployment/npu_device_deployment`
