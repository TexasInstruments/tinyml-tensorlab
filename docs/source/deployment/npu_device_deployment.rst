======================
NPU Device Deployment
======================

This guide covers deployment to TI devices with Neural Processing Unit (NPU)
hardware acceleration.

NPU-Enabled Devices
-------------------

The following devices include TI's TINPU:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Device
     - Family
     - NPU Features
   * - F28P55
     - C2000
     - 8-bit/4-bit inference, up to 60k params
   * - AM13E2
     - MSPM33C
     - 8-bit inference, Cortex-M33 + NPU
   * - MSPM0G5187
     - MSPM0
     - 8-bit inference, ultra-low power

NPU Compilation
---------------

To compile for NPU, use the correct preset:

.. code-block:: yaml

   common:
     target_device: 'F28P55'  # NPU device

   training:
     model_name: 'CLS_4k_NPU'  # NPU-compatible model

   compilation:
     enable: True
     preset_name: 'compress_npu_layer_data'  # NPU optimization

The ``compress_npu_layer_data`` preset:

* Optimizes memory layout for NPU
* Compresses weight data
* Generates NPU-specific code

NPU Model Requirements
----------------------

Models must follow NPU constraints (see :doc:`/devices/npu_guidelines`):

* Use model names ending in ``_NPU``
* Channel counts must be multiples of 4
* Kernel heights ≤ 7
* Must use INT8 or INT4 quantization

NPU Compilation Artifacts
-------------------------

After compilation:

.. code-block:: text

   .../compilation/artifacts/
   ├── mod.a                       # Compiled library (includes NPU code)
   ├── mod.h                       # Model interface
   ├── model_config.h              # NPU configuration
   ├── npu_layer_data.bin          # NPU weight data
   ├── feature_extraction.c        # Feature extraction
   └── inference_example.c         # Example code

NPU Initialization
------------------

NPU requires initialization before inference:

.. code-block:: c

   #include "mod.h"
   #include "npu.h"

   void main(void) {
       // Initialize system
       System_Init();

       // Initialize NPU hardware
       NPU_Init();

       // Initialize model (loads weights to NPU)
       mod_init();

       // Now ready for inference
       while (1) {
           if (data_ready) {
               run_npu_inference();
           }
       }
   }

NPU Inference Code
------------------

.. code-block:: c

   #include "mod.h"
   #include "feature_extraction.h"

   // Buffers
   float input_buffer[INPUT_SIZE];
   float feature_buffer[FEATURE_SIZE];
   float output_buffer[NUM_CLASSES];

   void run_npu_inference(void) {
       // 1. Collect sensor data
       collect_sensor_data(input_buffer);

       // 2. Extract features (runs on CPU)
       extract_features(input_buffer, feature_buffer);

       // 3. Run NPU inference
       // NPU handles quantization internally
       mod_inference(feature_buffer, output_buffer);

       // 4. Get prediction
       int prediction = argmax(output_buffer, NUM_CLASSES);

       // 5. Act on result
       handle_prediction(prediction);
   }

NPU Memory Management
---------------------

NPU requires specific memory regions:

**Weight Memory:**

NPU weights are stored in dedicated memory:

.. code-block:: c

   // Linker command file
   MEMORY
   {
       NPU_WEIGHTS : origin = 0x00080000, length = 0x00010000
   }

   SECTIONS
   {
       .npu_weights : > NPU_WEIGHTS
   }

**Activation Memory:**

NPU uses scratch memory for intermediate results:

.. code-block:: c

   // Allocate NPU scratch buffer
   #pragma DATA_SECTION(npu_scratch, ".npu_scratch")
   uint8_t npu_scratch[NPU_SCRATCH_SIZE];

NPU Performance
---------------

Typical NPU performance on F28P55:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Model
     - CPU Time
     - NPU Time
     - Speedup
   * - CLS_1k_NPU
     - 2000 µs
     - 150 µs
     - 13x
   * - CLS_4k_NPU
     - 5000 µs
     - 300 µs
     - 17x
   * - CLS_13k_NPU
     - 15000 µs
     - 600 µs
     - 25x

**Note:** Actual performance depends on model architecture and input size.

NPU Power Considerations
------------------------

NPU can be power-managed:

.. code-block:: c

   // Disable NPU when not in use
   void enter_low_power(void) {
       NPU_Disable();  // Saves power
   }

   // Re-enable before inference
   void prepare_inference(void) {
       NPU_Enable();
       // May need small delay for NPU to stabilize
       delay_us(10);
   }

NPU Debugging
-------------

**Verify NPU Initialization:**

.. code-block:: c

   if (NPU_GetStatus() != NPU_STATUS_READY) {
       // NPU initialization failed
       handle_error();
   }

**Check Inference Results:**

Compare NPU results with expected values from training:

.. code-block:: c

   // Known test input
   float test_input[] = {...};
   float expected_output[] = {...};

   mod_inference(test_input, output_buffer);

   // Compare
   float max_error = 0;
   for (int i = 0; i < NUM_CLASSES; i++) {
       float error = fabs(output_buffer[i] - expected_output[i]);
       if (error > max_error) max_error = error;
   }

   // Quantization error should be small
   if (max_error > 0.1) {
       // Unexpected deviation
       debug_print("Max error: %f\n", max_error);
   }

NPU Error Handling
------------------

Handle NPU errors gracefully:

.. code-block:: c

   int run_safe_inference(float* features, float* output) {
       // Check NPU status
       if (NPU_GetStatus() != NPU_STATUS_READY) {
           NPU_Reset();
           if (NPU_GetStatus() != NPU_STATUS_READY) {
               return -1;  // NPU unavailable
           }
       }

       // Run inference
       int result = mod_inference(features, output);

       if (result != 0) {
           // Inference error
           NPU_Reset();
           return -2;
       }

       return 0;  // Success
   }

CCS Project Setup for NPU
-------------------------

**1. Include NPU Support Files:**

From your device SDK, add:

* NPU driver files
* NPU header files
* NPU configuration files

**2. Configure Linker:**

Ensure linker command file includes NPU memory regions.

**3. Add Compiler Defines:**

.. code-block:: text

   Project Properties → Build → Compiler → Predefined Symbols
   Add: NPU_ENABLED=1

Example: Arc Fault on F28P55 NPU
--------------------------------

Complete deployment example:

**Configuration:**

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'dc_arc_fault_example_dsk'

   training:
     model_name: 'ArcFault_model_400_t'
     quantization_type: 'int8'

   compilation:
     enable: True
     preset_name: 'compress_npu_layer_data'

**Main Application:**

.. code-block:: c

   #include "device.h"
   #include "mod.h"
   #include "feature_extraction.h"
   #include "npu.h"

   #define SAMPLE_SIZE 1024
   #define FEATURE_SIZE 256
   #define NUM_CLASSES 2  // Normal, Arc

   float adc_buffer[SAMPLE_SIZE];
   float feature_buffer[FEATURE_SIZE];
   float output_buffer[NUM_CLASSES];

   volatile uint8_t inference_flag = 0;

   void main(void) {
       // System initialization
       Device_init();
       Device_initGPIO();

       // Initialize ADC for current sensing
       ADC_Init();

       // Initialize NPU
       NPU_Init();

       // Initialize model
       mod_init();

       // Enable interrupts
       EINT;

       while (1) {
           if (inference_flag) {
               // Extract features
               extract_features(adc_buffer, feature_buffer);

               // Run NPU inference
               mod_inference(feature_buffer, output_buffer);

               // Check for arc fault
               if (output_buffer[1] > output_buffer[0]) {
                   // Arc detected!
                   GPIO_writePin(ALERT_PIN, 1);
                   trigger_protection();
               }

               inference_flag = 0;
           }
       }
   }

   __interrupt void ADC_ISR(void) {
       static uint16_t sample_idx = 0;

       adc_buffer[sample_idx++] = ADC_readResult();

       if (sample_idx >= SAMPLE_SIZE) {
           sample_idx = 0;
           inference_flag = 1;
       }

       ADC_clearInterruptStatus();
   }

Troubleshooting NPU Issues
--------------------------

**NPU Initialization Fails:**

* Check device is NPU-enabled
* Verify NPU clock is enabled
* Ensure NPU memory regions are defined

**Incorrect Results:**

* Verify model is NPU-compatible
* Check quantization settings match
* Compare with float model on same input

**NPU Hangs:**

* Check for memory conflicts
* Verify buffer alignments
* Reset NPU and retry

Next Steps
----------

* Review :doc:`/devices/npu_guidelines` for model constraints
* See :doc:`ccs_integration` for general CCS setup
* Check :doc:`/troubleshooting/common_errors` for issues
