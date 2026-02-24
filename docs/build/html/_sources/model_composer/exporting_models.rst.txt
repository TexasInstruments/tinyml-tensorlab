=================
Exporting Models
=================

This guide covers how to export trained models from Model Composer for
deployment on TI microcontrollers.

Export Options
--------------

Model Composer provides several export formats:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Format
     - Description
   * - **CCS Project**
     - Complete Code Composer Studio project
   * - **Artifacts Only**
     - Model library and headers
   * - **ONNX Model**
     - Intermediate model format
   * - **Configuration**
     - YAML config for CLI tools

Exporting as CCS Project
------------------------

The CCS Project export creates a ready-to-build project:

**Step 1: Access Export**

1. Open your trained project in Model Composer
2. Navigate to **Export** tab
3. Select **CCS Project**

**Step 2: Configure Export**

1. Select target device family
2. Choose SDK version (if applicable)
3. Select example type:

   * Minimal: Basic inference loop
   * Full: With ADC, GPIO examples

**Step 3: Download**

1. Click **Export**
2. Wait for packaging
3. Download ZIP file

**Step 4: Import to CCS**

1. Open Code Composer Studio
2. File → Import → CCS Projects
3. Select archive file (the ZIP)
4. Click Finish

**Project Contents:**

.. code-block:: text

   MyProject_CCS/
   ├── .project              # CCS project file
   ├── .cproject             # CCS configuration
   ├── main.c                # Application entry
   ├── model/
   │   ├── mod.a            # Compiled model
   │   ├── mod.h            # Model interface
   │   └── model_config.h   # Configuration
   ├── features/
   │   ├── feature_extraction.c
   │   └── feature_extraction.h
   ├── device/
   │   └── device_config.c  # Device-specific init
   └── linker.cmd           # Memory layout

Exporting Artifacts Only
------------------------

For integration into existing projects:

**Step 1: Select Artifacts Only**

1. Go to **Export** tab
2. Select **Artifacts Only**

**Step 2: Choose Components**

Select which files to include:

* Model library (mod.a) - Required
* Model header (mod.h) - Required
* Feature extraction code - Recommended
* Test vectors - Optional
* Configuration header - Optional

**Step 3: Download**

1. Click **Export**
2. Download ZIP

**Artifact Contents:**

.. code-block:: text

   artifacts/
   ├── mod.a                    # Model library
   ├── mod.h                    # Interface header
   ├── model_config.h           # Model parameters
   ├── feature_extraction.c     # Feature code
   ├── feature_extraction.h     # Feature header
   └── test_vectors/            # (if selected)
       ├── input_0.csv
       └── expected_0.csv

**Using Artifacts:**

1. Copy to your existing CCS project
2. Add include paths for headers
3. Add library to linker settings
4. Call model functions from your code

Exporting ONNX Model
--------------------

For custom deployment or analysis:

**Step 1: Select ONNX**

1. Go to **Export** tab
2. Select **ONNX Model**

**Step 2: Choose Options**

* Float32 model: Original precision
* Quantized model: INT8 quantized
* Both: Download both versions

**Step 3: Download**

Receive ONNX file(s):

.. code-block:: text

   onnx_export/
   ├── model_float32.onnx    # Full precision
   └── model_int8.onnx       # Quantized

**Using ONNX:**

* Visualize with Netron
* Run inference with ONNX Runtime
* Custom compilation pipelines
* Cross-platform deployment

Exporting Configuration
-----------------------

Export YAML config for CLI tools:

**Step 1: Select Configuration**

1. Go to **Export** tab
2. Select **Configuration (YAML)**

**Step 2: Download**

Receive config file:

.. code-block:: yaml

   # Exported from Model Composer
   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'
     run_name: 'arc_fault_v1'

   dataset:
     dataset_name: 'dc_arc_fault_example_dsk'
     input_data_path: '/path/to/dataset'

   data_processing_feature_extraction:
     feature_extraction_name: 'FFT1024Input_256Feature_1Frame_Full_Bandwidth'
     variables: 1

   training:
     model_name: 'ArcFault_model_400_t'
     training_epochs: 30
     batch_size: 256
     quantization: 2
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

   testing:
     enable: True

   compilation:
     enable: True

**Using Configuration:**

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh exported_config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat exported_config.yaml

Batch Export
------------

Export multiple projects at once:

1. Select projects in dashboard (checkbox)
2. Click **Batch Export**
3. Choose export format
4. Download combined ZIP

Export History
--------------

Track all exports:

1. Go to **Project** → **History**
2. View **Exports** tab
3. Re-download previous exports
4. Compare export configurations

Validating Exports
------------------

After exporting, validate the model:

**Check Model Files:**

.. code-block:: bash

   # Verify library exists
   ls -la mod.a

   # Check header content
   head mod.h

**Test Vectors:**

If test vectors were exported:

.. code-block:: c

   // In your test code
   #include "test_vectors.h"

   void validate_model(void) {
       float output[NUM_CLASSES];

       // Run inference on test vector
       mod_inference(test_input_0, output);

       // Compare with expected
       for (int i = 0; i < NUM_CLASSES; i++) {
           float diff = fabs(output[i] - expected_output_0[i]);
           if (diff > 0.01) {
               // Unexpected difference
               printf("Mismatch at %d: %.4f vs %.4f\n",
                      i, output[i], expected_output_0[i]);
           }
       }
   }

Export Troubleshooting
----------------------

**Export Button Grayed Out:**

* Training must complete first
* Check for compilation errors
* Refresh the page

**Download Fails:**

* Check browser popup blocker
* Try different browser
* Check internet connection

**CCS Import Fails:**

* Verify CCS version compatibility
* Check SDK installation
* Try "Artifacts Only" instead

**Model Doesn't Work on Device:**

* Verify target device matches
* Check memory constraints
* Validate with test vectors

Best Practices
--------------

**Before Export:**

1. Verify training completed successfully
2. Check accuracy meets requirements
3. Test on holdout data
4. Document configuration

**Export Selection:**

* New to CCS: Use CCS Project export
* Existing project: Use Artifacts Only
* Custom pipeline: Use ONNX
* Reproducibility: Export Configuration

**After Export:**

1. Validate model files
2. Test with known inputs
3. Profile on target device
4. Document performance

Next Steps
----------

After exporting:

* :doc:`/deployment/ccs_integration` - Import into CCS
* :doc:`/deployment/npu_device_deployment` - NPU deployment
* :doc:`/deployment/non_npu_deployment` - CPU deployment
