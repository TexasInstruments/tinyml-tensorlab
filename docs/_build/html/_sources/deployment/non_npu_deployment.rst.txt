====================
Non-NPU Deployment
====================

This guide covers deployment to TI devices without NPU hardware acceleration.
These devices run inference entirely on the CPU.

Non-NPU Devices
---------------

Devices without NPU include:

**C2000 Family:**

* F28P65, F29H85, F29P58, F29P32
* F2837, F28004, F28003
* F280013, F280015

**MSPM0 Family:**

* MSPM0G3507, MSPM0G3519

**MSPM33C Family:**

* MSPM33C32, MSPM33C34

**AM26x Family:**

* AM263, AM263P, AM261

**Connectivity:**

* CC2755, CC1352

Configuration
-------------

For non-NPU devices, use standard models:

.. code-block:: yaml

   common:
     target_device: 'F28P65'  # Non-NPU device

   training:
     model_name: 'CLS_4k'  # Standard model (no _NPU suffix)

   compilation:
     enable: True
     preset_name: 'default_preset'  # Standard compilation

Model Selection
---------------

Without NPU acceleration, choose smaller models:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Device Class
     - Recommended Size
     - Model Examples
   * - Entry-level (M0+)
     - 100-500 params
     - CLS_100, CLS_500
   * - Mid-range
     - 500-2k params
     - CLS_1k, CLS_2k
   * - High-performance
     - 2k-6k params
     - CLS_4k, CLS_6k
   * - AM26x (Cortex-R5)
     - Up to 13k params
     - CLS_6k, CLS_13k

CPU Inference Performance
-------------------------

Typical inference times (CPU-only):

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Model
     - F28P65
     - MSPM0G3507
     - AM263
     - CC2755
   * - CLS_500
     - 500 µs
     - 800 µs
     - 200 µs
     - 600 µs
   * - CLS_1k
     - 1000 µs
     - 1500 µs
     - 400 µs
     - 1200 µs
   * - CLS_4k
     - 4000 µs
     - 6000 µs
     - 1500 µs
     - 5000 µs

**Note:** Times are approximate and depend on clock frequency.

Compilation Artifacts
---------------------

Non-NPU compilation produces:

.. code-block:: text

   .../compilation/artifacts/
   ├── mod.a                    # Model library (CPU code)
   ├── mod.h                    # Model interface
   ├── model_config.h           # Configuration
   ├── feature_extraction.c     # Feature extraction
   └── inference_example.c      # Example code

CCS Project Setup
-----------------

**Import the Project:**

.. figure:: /_static/img/deployment/non_npu/import_project.png
   :width: 600px
   :align: center
   :alt: Import Project for Non-NPU

   Importing a project into CCS for non-NPU devices

**Build the Project:**

.. figure:: /_static/img/deployment/non_npu/build_project.png
   :width: 600px
   :align: center
   :alt: Build Project

   Building the project for non-NPU deployment

**Flash and Debug:**

.. figure:: /_static/img/deployment/non_npu/flash_application.png
   :width: 600px
   :align: center
   :alt: Flash Application

   Flashing the application to a non-NPU device

.. figure:: /_static/img/deployment/non_npu/debug_screen.png
   :width: 700px
   :align: center
   :alt: Debug Screen

   CCS Debug perspective for non-NPU deployment

Basic Integration
-----------------

.. code-block:: c

   #include "mod.h"
   #include "feature_extraction.h"

   float input_buffer[INPUT_SIZE];
   float feature_buffer[FEATURE_SIZE];
   float output_buffer[NUM_CLASSES];

   void run_inference(void) {
       // Collect data
       collect_sensor_data(input_buffer);

       // Extract features
       extract_features(input_buffer, feature_buffer);

       // Run CPU inference
       mod_inference(feature_buffer, output_buffer);

       // Get result
       int prediction = argmax(output_buffer, NUM_CLASSES);
       handle_result(prediction);
   }

Optimizing CPU Inference
------------------------

**1. Enable Compiler Optimizations:**

.. code-block:: text

   Project Properties → Build → Compiler → Optimization
   Level: 4 (Highest)
   Speed vs Size: Speed

**2. Use Fixed-Point When Possible:**

If your model supports fixed-point:

.. code-block:: yaml

   training:
     quantization: 1
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

INT8 operations are faster than float on many MCUs.

**3. Place Critical Code in Fast Memory:**

.. code-block:: c

   #pragma CODE_SECTION(mod_inference, ".TI.ramfunc")

**4. Optimize Feature Extraction:**

Use simpler feature extraction if possible:

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_256Input_RAW_256Feature_1Frame'

Memory Optimization
-------------------

Non-NPU devices may have limited RAM:

**Minimize Buffer Sizes:**

.. code-block:: yaml

   data_processing_feature_extraction:
     # Smaller input reduces buffers
     feature_extraction_name: 'Generic_256Input_FFTBIN_32Feature_4Frame'

**Use Static Allocation:**

.. code-block:: c

   // Static allocation - size known at compile time
   static float feature_buffer[FEATURE_SIZE];
   static float output_buffer[NUM_CLASSES];

**Memory Map Check:**

Verify model fits in available memory:

.. code-block:: text

   After building, check .map file:
   .text (code):    XX KB
   .const (weights): XX KB
   .bss (buffers):   XX KB

   Compare with device memory:
   Flash: XXX KB
   RAM:   XX KB

Power Optimization
------------------

For battery-powered devices:

**1. Duty Cycle Inference:**

.. code-block:: c

   void main(void) {
       while (1) {
           // Wake up
           wake_from_sleep();

           // Run inference
           run_inference();

           // Sleep
           enter_low_power_mode();
       }
   }

**2. Reduce Clock During Inference:**

Some devices allow dynamic clocking:

.. code-block:: c

   // Run at lower clock for power savings
   // (trades off latency for power)
   set_clock_speed(CLOCK_40MHZ);
   run_inference();

**3. Use Smallest Sufficient Model:**

.. code-block:: yaml

   training:
     model_name: 'CLS_500'  # Smaller = less energy

Real-Time Considerations
------------------------

For real-time applications:

**Worst-Case Execution Time (WCET):**

Measure inference time to ensure deadlines are met:

.. code-block:: c

   // Measure WCET
   uint32_t max_time = 0;
   for (int i = 0; i < 1000; i++) {
       uint32_t start = get_timer();
       run_inference();
       uint32_t elapsed = get_timer() - start;
       if (elapsed > max_time) max_time = elapsed;
   }
   // max_time is WCET estimate

**Interrupt Latency:**

Inference may block interrupts:

.. code-block:: c

   // Option 1: Run inference at low priority
   void low_priority_task(void) {
       run_inference();
   }

   // Option 2: Split inference into chunks
   void inference_chunk(int chunk_id) {
       mod_inference_partial(chunk_id, feature_buffer, output_buffer);
   }

Device-Specific Notes
---------------------

**C2000 (F28P65, F2837, etc.):**

* Strong floating-point unit
* Good for signal processing
* Use FPU-optimized libraries

.. code-block:: c

   // Enable FPU
   FPU_enableModule();

**MSPM0 (Cortex-M0+):**

* No FPU (software float)
* Prefer INT8 quantization
* Keep models small (<1k params)

.. code-block:: yaml

   training:
     model_name: 'CLS_500'
     quantization: 1
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

**AM26x (Cortex-R5):**

* High performance
* FPU available
* Can handle larger models

.. code-block:: yaml

   training:
     model_name: 'CLS_6k'  # or larger

**CC27xx/CC13xx (Connectivity):**

* Balance model vs wireless stack memory
* Consider inference frequency vs RF activity

Example: Vibration Monitoring on MSPM0G3507
-------------------------------------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'MSPM0G3507'

   dataset:
     dataset_name: 'vibration_dataset'

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_256Input_FFTBIN_32Feature_4Frame'
     variables: 1

   training:
     model_name: 'AD_500'  # Small model for M0+
     quantization: 1
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

   compilation:
     enable: True

**Application Code:**

.. code-block:: c

   #include "ti_msp_dl_config.h"
   #include "mod.h"
   #include "feature_extraction.h"

   #define SAMPLE_SIZE 256
   #define FEATURE_SIZE 128
   #define THRESHOLD 0.5f

   float adc_buffer[SAMPLE_SIZE];
   float feature_buffer[FEATURE_SIZE];
   float output;  // Reconstruction error

   int main(void) {
       SYSCFG_DL_init();

       // Initialize model
       mod_init();

       while (1) {
           // Collect vibration data
           for (int i = 0; i < SAMPLE_SIZE; i++) {
               DL_ADC12_startConversion(ADC0);
               while (!DL_ADC12_isConversionComplete(ADC0));
               adc_buffer[i] = DL_ADC12_getMemResult(ADC0, 0);
           }

           // Extract features
           extract_features(adc_buffer, feature_buffer);

           // Run anomaly detection
           mod_inference(feature_buffer, &output);

           // Check threshold
           if (output > THRESHOLD) {
               // Anomaly detected
               DL_GPIO_setPins(ALERT_PORT, ALERT_PIN);
           } else {
               DL_GPIO_clearPins(ALERT_PORT, ALERT_PIN);
           }

           // Enter low power until next sample period
           __WFI();
       }
   }

Comparison: NPU vs Non-NPU
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - NPU Devices
     - Non-NPU Devices
   * - Inference speed
     - 10-25x faster
     - Baseline
   * - Model size
     - Up to 60k params
     - Typically <6k params
   * - Power
     - Lower per inference
     - Higher per inference
   * - Model constraints
     - NPU-specific rules
     - More flexible
   * - Cost
     - Higher BOM
     - Lower BOM

Next Steps
----------

* See :doc:`ccs_integration` for detailed CCS setup
* Review :doc:`/devices/device_overview` for device selection
* Check :doc:`/troubleshooting/common_errors` for issues
