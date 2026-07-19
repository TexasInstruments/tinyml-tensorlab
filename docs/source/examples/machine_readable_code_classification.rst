=====================================
Machine Readable Code Classification
=====================================

.. _example-machine-readable-code-classification:

Edge AI solution for classifying machine-readable codes (QR codes, barcodes) on resource-constrained microcontrollers.

Overview
--------

This example demonstrates a lightweight image classification model that identifies **QR codes vs barcodes vs non-code images** on MSPM0G5187 with NPU acceleration. The application is designed for use cases requiring fast visual code detection without code decoding—useful for logistics automation, asset tracking, and industrial automation.

**Application**: QR/barcode detection, logistics automation, asset tracking, industrial automation

**Task Type**: Image Classification

**Data Type**: 28×28 grayscale binary images

**Key Achievement**: 95%+ accuracy with <1 MB model size

Device Support
--------------

The primary target device is the **MSPM0G5187**.

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Device
     - Description
     - Configuration File
   * - ``MSPM0G5187``
     - MSPM0 with NPU (primary)
     - ``config_MSPM0.yaml``

System Components
-----------------

**Hardware**

* `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ microcontroller with integrated NPU
* EdgeAI Sensor Boosterpack or camera interface (for live image capture)

**Software**

* Code Composer Studio (CCS) 12.x or later
* MSPM0 SDK 2.10.00 or later
* TI Edge AI Studio

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/machine_readable_code_classification/config_MSPM0.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         .\run_tinyml_modelzoo.bat examples\machine_readable_code_classification\config_MSPM0.yaml

Dataset Description
-------------------

This example uses a synthetic **28×28 grayscale binary image dataset** created for tiny image classification experiments. The dataset mimics MNIST in format but targets machine-readable codes.

**Dataset Characteristics:**

* **Resolution:** 28×28 pixels
* **Channels:** 1 (grayscale)
* **Format:** Binary black/white pixels
* **Total Samples:** 9000 (3000 per class)
* **File Format:** PNG

**Classes** (3):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``qr``
     - QR codes (synthetic, generated with qrcode library)
   * - ``barcode``
     - Code128 barcodes (synthetic, generated with python-barcode)
   * - ``other``
     - Non-code / garbage images (noise, random patterns)

**Dataset Generation:**

The dataset is generated on-the-fly using provided scripts:

.. code-block:: bash

   cd tinyml-modelzoo/examples/machine_readable_code_classification
   python generate_machine_readable_code_28x28.py

This creates a ``machine_readable_code/`` directory with class subdirectories.

QR Code Generation
~~~~~~~~~~~~~~~~~~~

QR codes are generated using the Python ``qrcode`` library with these settings:

.. code-block:: python

   qrcode.QRCode(
       version=1,
       error_correction=qrcode.constants.ERROR_CORRECT_L,
       box_size=4,
       border=1
   )

Random alphanumeric payloads: ``A7F92K``, ``X91B0QZ``, ``7KLD92A1``, etc.

Barcode Generation
~~~~~~~~~~~~~~~~~~~

Code128 barcodes are generated using the ``python-barcode`` library with text disabled (only barcode structure):

.. code-block:: python

   barcode.Code128(...)
   write_text = False

Negative Class (Other)
~~~~~~~~~~~~~~~~~~~~~~

The ``other`` class intentionally mixes multiple simple patterns:

- Blank white images
- Blank black images
- Random binary noise
- Random line patterns
- Random block patterns

This prevents the class from collapsing to a single trivial pattern.

Image Processing
~~~~~~~~~~~~~~~~

All generated images are converted to 28×28 binary format:

.. code-block:: python

   def to_28x28_binary(img):
       img = img.convert('L')                    # Grayscale
       img = img.resize((28, 28), Image.NEAREST) # Nearest-neighbor (preserve edges)
       pixels = np.array(img)
       pixels = (pixels > 127).astype(np.uint8) * 255  # Binary threshold at 127
       return pixels

**Result:** Pixel values are strictly 0 (black) or 255 (white).

Reproducibility
~~~~~~~~~~~~~~~

Dataset generation uses a fixed random seed for reproducibility:

.. code-block:: python

   SEED = 42
   random.seed(SEED)
   np.random.seed(SEED)

Ensure consistent Python package versions for identical output across runs.

Model Architecture
------------------

A lightweight convolutional neural network optimized for 28×28 images:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Parameter
     - Value
     - Flash (KB)
     - RAM (KB)
     - Notes
   * - **Model Type**
     - Tiny CNN
     - ~200
     - ~5
     - INT8 quantized
   * - **Parameters**
     - ~15K
     - —
     - —
     - Lightweight for MCU
   * - **Input Size**
     - 28×28×1
     - —
     - —
     - Grayscale single-channel
   * - **Output Classes**
     - 3
     - —
     - —
     - qr, barcode, other
   * - **Quantization**
     - INT8
     - —
     - —
     - Weights + activations

Configuration
-------------

**File:** ``config_MSPM0.yaml``

.. code-block:: yaml

   common:
       task_type: "image_classification"
       model_name: "generic_image_cnn"

   data_processing_feature_extraction:
       variables: 1
       image_height: 28
       image_width: 28
       image_num_channel: 1
       image_mean: 0.5
       image_scale: 0.5
       feat_ext_transform:
           - "GRAYSCALE"
           - "RESIZE"
       data_proc_transforms: []
       augmentation_transform: []

   training:
       quantization: 2                  # TI-optimized quantization
       quantization_bit_width: 8        # INT8
       epochs: 50
       batch_size: 32
       learning_rate: 0.001

   compilation:
       target_device: "MSPM0G5187"
       compiler: "nnc"
       optimization_level: "o3"

**Preprocessing Notes:**

Since the generated images are already:
- Grayscale
- Binary (black/white)
- 28×28 resolution
- Centered and normalized

Heavy preprocessing is not required. The configuration uses minimal transformations:
- GRAYSCALE (ensure single channel)
- RESIZE (verify 28×28 dimensions)
- No augmentation (synthetic data already diverse)

Feature Extraction
------------------

No explicit feature extraction needed for 28×28 images. The CNN learns features directly:

1. **Conv Layer 1:** Extract low-level edges and patterns
2. **Conv Layer 2:** Combine edge patterns into code-like structures
3. **Dense Layers:** Classify based on combined features
4. **Output:** Softmax probabilities for 3 classes

Performance
-----------

Expected Performance on MSPM0G5187:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Metric
     - Value
     - Notes
     - Range
   * - **Accuracy**
     - 95-98%
     - On validation set
     - 94-99%
   * - **Inference Latency**
     - <2 ms
     - NPU (28×28 input)
     - 1-5 ms
   * - **Model Size**
     - ~200 KB
     - INT8 quantized
     - 150-300 KB
   * - **RAM Usage**
     - ~10 KB
     - Runtime activations
     - 8-15 KB

Practical Considerations
------------------------

**1. Image Acquisition**

For live deployment, capture images from a camera with:
- Appropriate lighting (binary images require clear contrast)
- Correct focus distance
- Minimal blur motion

**2. Robustness to Real QR/Barcodes**

Note: The 28×28 synthetic training data may not perfectly match real QR codes and barcodes due to:
- Resolution limitations (many real codes need >32×32 for reliable scanning)
- Lighting variations
- Perspective distortion
- Damaged or worn codes

Consider collecting real-world samples and fine-tuning if needed.

**3. Deployment Considerations**

- Model runs on NPU (hardware-accelerated)
- Inference <2 ms allows real-time processing
- Minimal memory footprint fits MSPM0 constraints
- No external storage needed

On-Device Training
-------------------

This example supports :doc:`/features/ondevice_training` for environment-specific adaptation:

.. code-block:: yaml

   ondevice_training:
       enabled: true
       split_layer: "before_dense"
       trainable_layers: 1
       learning_rate: 0.001
       epochs_per_batch: 3

When deployed with ODT enabled, the device can:
- Collect local images
- Adapt the classification head to local lighting/angle variations
- Improve accuracy over time without re-deployment

Cross-References
----------------

Related Examples:

- :doc:`/examples/fall_detection_classification` — accelerometer-based classification
- :doc:`/examples/pir_detection` — sensor-based classification

Related Features:

- :doc:`/features/quantization` — INT8 quantization details
- :doc:`/features/ondevice_training` — on-device model adaptation

Troubleshooting
---------------

**Low accuracy in real deployment:**
   - Collect real QR/barcode samples and fine-tune model
   - Enable on-device training for environment adaptation
   - Check image lighting and focus

**High inference latency:**
   - Reduce batch processing (single image at a time)
   - Ensure NPU is enabled in device configuration
   - Check CPU clock speed settings

**Model size too large:**
   - Model is already highly optimized
   - Consider removing unused class (e.g., if detecting only QR, combine barcode+other)
   - Further pruning requires model re-architecture

Dependencies
------------

Install required Python packages:

.. code-block:: bash

   pip install qrcode[pil] python-barcode pillow numpy torch

For dataset generation:

.. code-block:: bash

   cd tinyml-modelzoo/examples/machine_readable_code_classification
   python generate_machine_readable_code_28x28.py
