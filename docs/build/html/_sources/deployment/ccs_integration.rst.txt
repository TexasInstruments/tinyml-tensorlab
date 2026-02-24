======================
CCS Integration Guide
======================

This guide explains how to deploy Tiny ML models to TI microcontrollers
using Code Composer Studio (CCS).

Prerequisites
-------------

Before deploying, ensure you have:

1. **Code Composer Studio** (CCS) installed

   * Download from: https://www.ti.com/tool/CCSTUDIO
   * Version 12.x or later recommended
   * Install support for your target device family

2. **Device-specific SDK**

   * C2000WARE for C2000 devices
   * MSPM0 SDK for MSPM0 devices
   * AM26x SDK for AM26x devices

3. **TI Compilers**

   * Included with CCS
   * Or download separately from TI website

4. **Trained and compiled model**

   * ``mod.a`` library file
   * ``mod.h`` header file
   * Feature extraction code

Compilation Output
------------------

After running Tiny ML Tensorlab with ``compilation: enable: True``:

.. code-block:: text

   .../compilation/artifacts/
   ├── mod.a                    # Compiled model library
   ├── mod.h                    # Model interface header
   ├── model_config.h           # Model configuration
   ├── feature_extraction.c     # Feature extraction code
   ├── feature_extraction.h     # Feature extraction header
   └── inference_example.c      # Example usage code

Creating a CCS Project
----------------------

**Step 1: Import Project from Resource Explorer**

1. Open CCS
2. Navigate to Resource Explorer
3. Find your target device SDK

.. figure:: /_static/img/deployment/resouce_explorer.png
   :width: 700px
   :align: center
   :alt: CCS Resource Explorer

   Resource Explorer in Code Composer Studio

4. Select and import the example project

.. figure:: /_static/img/deployment/import_project.png
   :width: 500px
   :align: center
   :alt: Import Project

   Importing a project into CCS

**Step 2: Create New Project (Alternative)**

1. Open CCS
2. File → New → CCS Project
3. Select your target device (e.g., F28P55x)
4. Choose "Empty Project" template
5. Name your project

**Step 2: Add Model Files**

Copy compiled artifacts to your project:

.. code-block:: text

   MyProject/
   ├── main.c                   # Your application
   ├── model/
   │   ├── mod.a               # Model library
   │   ├── mod.h               # Model header
   │   ├── model_config.h      # Configuration
   │   ├── feature_extraction.c
   │   └── feature_extraction.h
   └── ...

**Step 3: Configure Project**

Add model directory to include paths:

1. Right-click project → Properties
2. Build → Compiler → Include Options
3. Add: ``${PROJECT_ROOT}/model``

Add library to linker:

1. Build → Linker → File Search Path
2. Add library: ``${PROJECT_ROOT}/model/mod.a``

Integration Code
----------------

**Basic Inference Example:**

.. code-block:: c

   #include "mod.h"
   #include "feature_extraction.h"

   // Allocate buffers
   float input_buffer[INPUT_SIZE];
   float feature_buffer[FEATURE_SIZE];
   float output_buffer[NUM_CLASSES];

   void run_inference(void) {
       // 1. Collect sensor data into input_buffer
       collect_sensor_data(input_buffer);

       // 2. Extract features
       extract_features(input_buffer, feature_buffer);

       // 3. Run model inference
       mod_inference(feature_buffer, output_buffer);

       // 4. Get prediction
       int predicted_class = argmax(output_buffer, NUM_CLASSES);

       // 5. Take action based on prediction
       handle_prediction(predicted_class);
   }

   int argmax(float* array, int size) {
       int max_idx = 0;
       float max_val = array[0];
       for (int i = 1; i < size; i++) {
           if (array[i] > max_val) {
               max_val = array[i];
               max_idx = i;
           }
       }
       return max_idx;
   }

**Continuous Inference Loop:**

.. code-block:: c

   void main(void) {
       // Initialize hardware
       System_Init();
       ADC_Init();
       Timer_Init();

       // Initialize model
       mod_init();

       while (1) {
           // Wait for data ready
           if (data_ready_flag) {
               run_inference();
               data_ready_flag = 0;
           }
       }
   }

Memory Placement
----------------

For optimal performance, place buffers in fast memory:

.. code-block:: c

   // Place in fast RAM (device-specific syntax)
   #pragma DATA_SECTION(feature_buffer, ".ramgs0")
   float feature_buffer[FEATURE_SIZE];

   #pragma DATA_SECTION(output_buffer, ".ramgs0")
   float output_buffer[NUM_CLASSES];

Consult your device's memory map for available sections.

Linker Command File
-------------------

Modify your linker command file to allocate space:

.. code-block:: text

   MEMORY
   {
       /* Existing memory regions */

       /* Add space for model */
       MODEL_RAM  : origin = 0x00010000, length = 0x00002000
   }

   SECTIONS
   {
       /* Place model data */
       .model_weights : > MODEL_RAM
       .model_buffers : > MODEL_RAM
   }

Interrupt-Based Inference
-------------------------

For real-time applications, trigger inference from interrupts:

.. code-block:: c

   volatile uint16_t sample_count = 0;
   volatile uint8_t inference_ready = 0;

   // ADC interrupt - collect samples
   __interrupt void ADC_ISR(void) {
       input_buffer[sample_count++] = ADC_Result;

       if (sample_count >= INPUT_SIZE) {
           sample_count = 0;
           inference_ready = 1;
       }

       // Clear interrupt flag
       ADC_clearInterruptStatus();
   }

   void main(void) {
       // ... initialization ...

       while (1) {
           if (inference_ready) {
               run_inference();
               inference_ready = 0;
           }
       }
   }

Timing and Profiling
--------------------

Measure inference time:

.. code-block:: c

   #include "device.h"

   uint32_t start_time, end_time, inference_time;

   void profile_inference(void) {
       // Start timer
       start_time = CPUTimer_getTimerCount(CPUTIMER0_BASE);

       // Run inference
       mod_inference(feature_buffer, output_buffer);

       // Stop timer
       end_time = CPUTimer_getTimerCount(CPUTIMER0_BASE);

       // Calculate time (adjust for timer configuration)
       inference_time = start_time - end_time;  // Downcounting timer
   }

Debugging
---------

**Verify Model Output:**

1. Set breakpoint after inference
2. Examine output_buffer values
3. Compare with expected results from training

.. figure:: /_static/img/deployment/breakpoint.png
   :width: 700px
   :align: center
   :alt: Setting breakpoint

   Setting a breakpoint in CCS debugger

.. figure:: /_static/img/deployment/debug_screen.png
   :width: 700px
   :align: center
   :alt: Debug screen

   CCS Debug perspective showing variables

**Viewing Test Results:**

.. figure:: /_static/img/deployment/variable_test_result.png
   :width: 600px
   :align: center
   :alt: Variable test result

   Examining model output variables

.. figure:: /_static/img/deployment/test_result_value.png
   :width: 600px
   :align: center
   :alt: Test result values

   Verifying inference results

**Memory Debugging:**

1. Use Memory Browser in CCS
2. Verify buffers are properly allocated
3. Check for memory corruption

**Performance Debugging:**

1. Use Profiler in CCS
2. Identify bottlenecks
3. Optimize critical sections

Build Configurations
--------------------

**Debug Configuration:**

* Optimization: Off
* Debug symbols: Enabled
* Use for development

**Release Configuration:**

* Optimization: High (O3)
* Debug symbols: Optional
* Use for deployment

.. code-block:: text

   Build → Manage Build Configurations
   → Set Active Configuration → Release

Example Project Structure
-------------------------

Complete project layout:

.. code-block:: text

   ArcFaultDetection/
   ├── main.c                      # Application entry
   ├── hardware/
   │   ├── adc_config.c            # ADC setup
   │   ├── gpio_config.c           # GPIO setup
   │   └── timer_config.c          # Timer setup
   ├── model/
   │   ├── mod.a                   # Model library
   │   ├── mod.h                   # Model interface
   │   ├── model_config.h          # Model params
   │   ├── feature_extraction.c    # Feature code
   │   └── feature_extraction.h    # Feature header
   ├── app/
   │   ├── inference.c             # Inference logic
   │   └── fault_handler.c         # Response logic
   ├── F28P55x.cmd                 # Linker command file
   └── .project                    # CCS project file

Common Issues
-------------

**Linker Error: Undefined symbol**

Model library not linked properly:

* Verify mod.a is in linker search path
* Check library order in linker options

**Runtime Error: Hard Fault**

Memory access issue:

* Check buffer sizes match model requirements
* Verify memory sections are properly defined

**Incorrect Results**

Data format mismatch:

* Verify input data scaling matches training
* Check feature extraction parameters
* Ensure correct byte order

**Slow Inference**

Optimization needed:

* Enable compiler optimizations
* Use NPU if available
* Place buffers in fast memory

Testing on Hardware
-------------------

**1. Build the Project:**

.. figure:: /_static/img/deployment/build_project.png
   :width: 600px
   :align: center
   :alt: Build project

   Building the project in CCS

**2. Flash the Device:**

* Click Debug button in CCS
* Wait for code to load
* Click Resume to run

.. figure:: /_static/img/deployment/flash_application.png
   :width: 600px
   :align: center
   :alt: Flash application

   Flashing the application to the device

**3. Set Active Target Configuration:**

.. figure:: /_static/img/deployment/active_target.png
   :width: 500px
   :align: center
   :alt: Active target

   Selecting the active target configuration

**4. Verify Operation:**

* Monitor output variables
* Check GPIO/LED indicators
* Use serial output for status

**5. Validate Results:**

* Compare with training test set
* Test with known inputs
* Verify edge cases

Next Steps
----------

* See :doc:`npu_device_deployment` for NPU-specific details
* See :doc:`non_npu_deployment` for CPU-only devices
* Review :doc:`/troubleshooting/common_errors` for issues
