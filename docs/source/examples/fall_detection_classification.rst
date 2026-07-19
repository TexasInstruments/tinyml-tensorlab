==============================
Human Fall Detection
==============================

.. _example-fall-detection-classification:

Edge AI solution for real-time human fall detection using accelerometer data on MSPM0G5187 with NPU acceleration.

Overview
--------

This example demonstrates a safety-critical Edge AI application that classifies human movement as **Activities of Daily Living (ADL) or Fall** using accelerometer data. Falls are a leading cause of injury and death in elderly populations and industrial settings. The application provides real-time fall detection directly on a microcontroller without cloud connectivity or bulky external hardware.

**Application**: Elderly care, worker safety, wearable safety systems, occupancy monitoring

**Task Type**: Time Series Classification

**Data Type**: Multivariate (3-axis accelerometer)

**Key Achievement**: 97.65% accuracy with <1 ms inference latency

Device Support
--------------

The primary target device is the **MSPM0G5187** with accelerometer sensor.

.. list-table::
   :header-rows: 1
   :widths: 40 50 10

   * - Device
     - Hardware
     - Configuration File
   * - ``MSPM0G5187``
     - MSPM0 with NPU + BMI270 accelerometer (TIDA-010997)
     - ``config_MSPM0.yaml``

System Components
-----------------

**Hardware**

* `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ microcontroller with integrated NPU
* `TIDA-010997 <https://www.ti.com/tool/TIDA-010997>`_ EdgeAI Boosterpack with BMI270 3-axis accelerometer
* BMI270 specifications: 16-bit resolution, ±16g range, 100-200 Hz sampling rate

**Software**

* Code Composer Studio (CCS) 12.x or later
* MSPM0 SDK 2.11.00 or later
* TI Edge AI Studio

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/fall_detection_classification/config_MSPM0.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         .\run_tinyml_modelzoo.bat examples\fall_detection_classification\config_MSPM0.yaml

Dataset Description
-------------------

This example uses the **SisFall dataset**, a publicly available dataset of fall and normal movement patterns collected from multiple subjects.

**Dataset Source:**

* `SisFall: A Fall and Normal Movement Dataset <https://www.mdpi.com/1424-8220/17/1/198>`_ (open access publication)
* 34 subjects, multiple falls and ADL activities per subject
* Total: ~4500 recordings (≈70% training, 30% test)

**Sensor Data:**

The original SisFall dataset includes three sensors:

* ADXL345 — 13-bit triaxial accelerometer
* ITG3200 — triaxial gyroscope
* MMA8451Q — triaxial accelerometer

For this example, **only ADXL345 accelerometer data is used** to match the single-accelerometer BMI270 hardware on TIDA-010997.

**Classes** (2):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - **ADL**
     - Activities of Daily Living (walking, sitting, standing, lying, exercising)
   * - **Fall**
     - Actual fall events (forward fall, backward fall, lateral fall, etc.)

Data Preprocessing
-------------------

**1. Sensor Filtering**

Only ADXL345 (triaxial accelerometer) columns are extracted from SisFall:

.. code-block:: python

   # Extract only accelerometer (ADXL345)
   ax, ay, az = df['ADXL345_x'], df['ADXL345_y'], df['ADXL345_z']
   # Drop gyroscope (ITG3200) and other accelerometer (MMA8451Q)

**2. Bit-Depth Scaling**

ADXL345 is a 13-bit sensor, but BMI270 (deployed hardware) is 16-bit. Scale training data to match deployment:

.. code-block:: python

   # Scale from 13-bit to 16-bit resolution
   scale_factor = 2**16 / 2**13  # = 8
   scaled_value = raw_value * scale_factor

This ensures the training data distribution matches the resolution of live sensor data during inference.

**3. Format Conversion**

Convert SisFall CSV format to TinyML ModelZoo input format:

- Parse each recording file
- Extract 3-axis accelerometer values
- Label as ADL or Fall
- Export to standardized training dataset structure

Feature Extraction Pipeline
---------------------------

Raw 3-axis accelerometer data → Model input (64 features):

1. **Sensor Input** — 3-axis accelerometer (x, y, z) @ 100-200 Hz
2. **Windowing** — Segment into fixed-size windows (e.g., 256 samples = 1.3-2.5 sec)
3. **FFT Computation** — 256-point Real FFT per axis
4. **Magnitude Calculation** — Compute spectral magnitude
5. **DC Removal** — Remove DC bias component
6. **Binning** — Average 16 adjacent FFT bins → 8 features per axis
7. **Frame Concatenation** — Stack 8 frames (8 features × 3 axes × 3 frames = 72, reduced to 64)
8. **Model Input** — 64-element feature vector per window

**Output:** Time-domain and frequency-domain features capture both impulse (fall) and sustained (ADL) patterns.

Model Architecture
------------------

A generic time-series classification CNN optimized for MCU deployment:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 20 10

   * - Component
     - Value
     - Flash (KB)
     - RAM (KB)
     - Notes
   * - **Model**
     - CLS_6k (6,000 params)
     - 14
     - 0.7
     - INT8 quantized
   * - **Input Shape**
     - 64 features
     - —
     - —
     - FFT + time-domain
   * - **Layers**
     - Conv2D → MaxPool → Dense → Dense
     - —
     - —
     - Standard CNN
   * - **Quantization**
     - INT8
     - —
     - —
     - Weights + activations

**Performance Metrics (on LP-MSPM0G5187 Launchpad):**

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Metric
     - Value
     - Target
     - Notes
   * - **Accuracy**
     - 97.65%
     - ≥97.5%
     - On SisFall test set
   * - **Inference Latency**
     - 0.67 ms
     - <5 ms
     - On NPU
   * - **Model Size**
     - 14 KB
     - <50 KB
     - INT8 quantized
   * - **RAM Runtime**
     - 0.7 KB
     - <16 KB
     - Activations only

Configuration
-------------

**File:** ``config_MSPM0.yaml``

.. code-block:: yaml

   common:
       task_type: "timeseries_classification"
       model_name: "generic_timeseries_cnn"

   data_processing_feature_extraction:
       variables: 3                    # 3-axis accelerometer
       window_length: 256              # 256 samples per window
       frame_count: 8                  # 8 frames (3.2-6.4 sec @ 100-200 Hz)
       feat_ext_transform:
           - "FFT"
           - "MAGNITUDE"
           - "DC_REMOVAL"
           - "POOLING"
       data_proc_transforms:
           - "SCALING"
           - "NORMALIZATION"

   training:
       quantization: 2                 # TI-optimized quantization
       quantization_bit_width: 8       # INT8
       epochs: 100
       batch_size: 32
       learning_rate: 0.001

   compilation:
       target_device: "MSPM0G5187"
       compiler: "nnc"
       optimization_level: "o3"

Deployment Workflow
-------------------

**1. Real-Time Inference**

.. code-block:: c

   // MCU firmware loop
   while (true) {
       // Read 3-axis accelerometer from BMI270
       read_accelerometer(accel_x, accel_y, accel_z);

       // Apply feature extraction (FFT, DC removal, etc.)
       extract_features(accel_x, accel_y, accel_z, features);

       // Run inference on NPU
       classify(features, &class_probabilities);

       // Check for fall
       if (class_probabilities[FALL] > THRESHOLD) {
           trigger_alert();
       }
   }

**2. Alert Generation**

When fall is detected (confidence > 90%):
- Sound alarm
- Send alert to caregiver (via Bluetooth or network)
- Log event with timestamp
- Wait for user cancellation (false fall reset button)

**3. Sensor Calibration**

For new users/environments:
- Collect 30-60 seconds of ADL data
- Fine-tune model using :doc:`/features/ondevice_training`
- Improve accuracy for person-specific characteristics

On-Device Training
-------------------

This example supports on-device training for personalization:

.. code-block:: yaml

   ondevice_training:
       enabled: true
       split_layer: "before_dense"
       trainable_layers: 1
       learning_rate: 0.001
       epochs_per_batch: 5

With ODT enabled:
- Device collects local ADL samples during normal use
- Fine-tunes classification head to user's movement patterns
- Reduces false positives over first week of deployment
- Maintains high accuracy as user ages or activity changes

Practical Considerations
------------------------

**1. Sensor Placement**

- Mount accelerometer on belt or wrist
- Keep sensor level with ground (minimize static tilt)
- Ensure firm contact to avoid loose vibration

**2. Sampling Rate**

- Default: 100-200 Hz
- Nyquist limit for fall detection: ~50 Hz (capture impulses)
- Higher rates: better impulse fidelity, more power consumption

**3. False Positives / False Negatives**

- **False positives (ADL misclassified as fall):** Rapid movements (jumping, running)
  - Mitigate: Collect jumping/running samples, on-device training
- **False negatives (fall misclassified as ADL):** Slow falls (controlled descent)
  - Model trained on natural falls; handled by SisFall dataset diversity

**4. Battery Life**

- NPU inference: <10 mW @ 0.67 ms
- Continuous sampling @ 100 Hz: ~100 mW accelerometer
- Typical wearable: 24-48 hours on coin cell

**5. Privacy**

- All inference happens on-device
- No data transmission to cloud
- Raw accelerometer data never leaves device
- GDPR/HIPAA compliant

Cross-References
----------------

Related Examples:

- :doc:`/examples/pir_detection` — sensor-based classification
- :doc:`/examples/motor_bearing_fault` — anomaly detection (similar architecture)

Related Features:

- :doc:`/features/ondevice_training` — on-device personalization
- :doc:`/features/feature_extraction` — FFT and feature engineering
- :doc:`/features/quantization` — INT8 quantization details

Troubleshooting
---------------

**Low accuracy in deployment:**
   - Enable on-device training for user-specific adaptation
   - Verify sensor placement and calibration
   - Collect more fall samples if available

**Frequent false positives:**
   - Collect jumping/running ADL samples
   - Increase fall confidence threshold (e.g., 95% instead of 90%)
   - Fine-tune model with local ADL data

**High latency:**
   - Verify NPU is enabled in device config
   - Check CPU clock is not throttled
   - Reduce feature dimension if needed

**Sensor not reading:**
   - Check I2C bus (BMI270 uses 0x68 or 0x69 I2C address)
   - Verify TIDA-010997 firmware is up-to-date
   - Test accelerometer directly in CCS debugger

References
----------

- `SisFall Dataset Paper <https://www.mdpi.com/1424-8220/17/1/198>`_
- `BMI270 Datasheet <https://www.bosch-sensortec.com/products/motion-sensors/imus/bmi270/>`_
- `TIDA-010997 Documentation <https://www.ti.com/tool/TIDA-010997>`_
- `MSPM0 SDK Accelerometer Driver <https://github.com/TexasInstruments/mspm0_sdk>`_
