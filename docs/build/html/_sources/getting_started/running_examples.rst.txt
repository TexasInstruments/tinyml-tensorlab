================
Running Examples
================

This guide explains how to find, understand, and run the example configurations
included with Tiny ML Tensorlab.

Finding Examples
----------------

Examples are located in two places:

1. **tinyml-modelzoo/examples/** - Primary location, recommended entry point
2. **tinyml-modelmaker/examples/** - Additional examples

List available examples:

.. code-block:: bash

   ls tinyml-modelzoo/examples/

   # Output:
   # ac_arc_fault/
   # blower_imbalance/
   # dc_arc_fault/
   # ecg_arrhythmia/
   # ecg_classification/
   # fan_fault/
   # grid_stability/
   # generic_timeseries_classification/
   # hvac_temp_forecast/
   # motor_bearing_fault/
   # motor_speed_regression/
   # pmsm_temp_forecast/
   # ...

Running an Example
------------------

**Basic Command:**

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/<example_name>/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\<example_name>\config.yaml

**Examples:**

.. code-block:: bash

   # Hello World (simplest)
   ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

   # DC Arc Fault Detection
   ./run_tinyml_modelzoo.sh examples/dc_arc_fault/config.yaml

   # Motor Bearing Fault Classification
   ./run_tinyml_modelzoo.sh examples/motor_bearing_fault/config.yaml

   # PMSM Temperature Forecasting
   ./run_tinyml_modelzoo.sh examples/pmsm_temp_forecast/config.yaml

Example Directory Structure
---------------------------

Each example folder typically contains:

.. code-block:: text

   example_name/
   ├── config.yaml          # Main configuration
   ├── config_device2.yaml  # Alternative device config (optional)
   └── readme.md            # Example-specific documentation (optional)

Understanding Example Configs
-----------------------------

Most examples have a similar structure. Here's how to read them:

**1. Identify the Task Type**

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'  # or regression, etc.

**2. Check the Dataset**

.. code-block:: yaml

   dataset:
     dataset_name: 'dc_arc_fault_example_dsk'
     # URL means data will be downloaded automatically
     input_data_path: 'https://software-dl.ti.com/...'

**3. Review Feature Extraction**

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'FFT1024Input_256Feature_1Frame_Full_Bandwidth'
     variables: 1  # Number of input channels

**4. Check the Model**

.. code-block:: yaml

   training:
     model_name: 'ArcFault_model_200_t'  # Model architecture
     training_epochs: 20

**5. Verify Target Device**

.. code-block:: yaml

   common:
     target_device: 'F28P55'  # Which MCU to compile for

Customizing Examples
--------------------

**Change Target Device:**

.. code-block:: yaml

   common:
     target_device: 'MSPM0G3507'  # Change to your device

**Use a Different Model:**

.. code-block:: yaml

   training:
     model_name: 'CLS_4k_NPU'  # Try a different model size

**Disable Compilation (for faster iteration):**

.. code-block:: yaml

   compilation:
     enable: False

**Train Longer:**

.. code-block:: yaml

   training:
     training_epochs: 50  # Increase from default

Output Location
---------------

All outputs go to::

   ../tinyml-modelmaker/data/projects/<dataset_name>/run/<run_name>/

For example, running ``dc_arc_fault`` example produces::

   ../tinyml-modelmaker/data/projects/dc_arc_fault_example_dsk/run/2024-01-15_14-30-00/ArcFault_model_200_t/

Example Categories
------------------

**Classification Examples:**

* ``generic_timeseries_classification`` - Waveform classification (beginner)
* ``dc_arc_fault`` - DC arc fault detection
* ``ac_arc_fault`` - AC arc fault detection
* ``motor_bearing_fault`` - 6-class bearing fault
* ``fan_fault`` - Fan blade fault detection
* ``ecg_classification`` - ECG arrhythmia detection
* ``grid_stability`` - Power grid stability

**Regression Examples:**

* ``motor_speed_regression`` - Predict motor speed
* ``torque_measurement`` - Predict motor torque

**Forecasting Examples:**

* ``pmsm_temp_forecast`` - PMSM rotor temperature
* ``hvac_temp_forecast`` - Indoor temperature prediction

**Anomaly Detection Examples:**

* ``dc_arc_fault_ad`` - Arc fault using autoencoder
* ``ecg_anomaly`` - ECG anomaly detection
* ``motor_bearing_ad`` - Bearing anomaly

Running Multiple Examples
-------------------------

Create a script to run multiple examples:

.. code-block:: bash

   #!/bin/bash
   # run_all_examples.sh

   examples=(
     "generic_timeseries_classification"
     "dc_arc_fault"
     "motor_bearing_fault"
   )

   for example in "${examples[@]}"; do
     echo "Running $example..."
     ./run_tinyml_modelzoo.sh examples/$example/config.yaml
   done

Troubleshooting Examples
------------------------

**"Dataset download failed"**

Check your internet connection. Try downloading manually:

.. code-block:: bash

   wget <dataset_url> -O dataset.zip
   unzip dataset.zip

Then update config with local path:

.. code-block:: yaml

   dataset:
     input_data_path: '/path/to/extracted/dataset'

**"Out of memory"**

Reduce batch size:

.. code-block:: yaml

   training:
     batch_size: 64  # Reduce from 256

**"Compilation failed"**

Check environment variables are set correctly.
Or skip compilation for now:

.. code-block:: yaml

   compilation:
     enable: False

Next Steps
----------

* :doc:`/byod/index` - Use your own data
* :doc:`/examples/index` - Detailed example guides
* :doc:`/features/index` - Advanced features
