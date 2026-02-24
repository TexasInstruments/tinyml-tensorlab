==========
Quickstart
==========

This guide gets you from zero to a trained model in 5 minutes.

Prerequisites
-------------

* Python 3.10.x installed
* Tiny ML Tensorlab installed (see :doc:`/installation/index`)

Step 1: Navigate to ModelZoo
----------------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-tensorlab/tinyml-modelzoo

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-tensorlab\tinyml-modelzoo

Step 2: Run Hello World Example
-------------------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml

This example:

1. Downloads a simple waveform dataset (sine, square, sawtooth waves)
2. Applies FFT-based feature extraction
3. Trains a small classification model (~1K parameters)
4. Quantizes the model for MCU deployment
5. Compiles for F28P55 (NPU device)

Step 3: View Results
--------------------

Training outputs are saved to::

   ../tinyml-modelmaker/data/projects/generic_timeseries_classification/run/<timestamp>/

**Directory Contents:**

.. code-block:: text

   <timestamp>/
   ├── training/
   │   ├── base/                    # Float32 training
   │   │   ├── best_model.pt        # Best model checkpoint
   │   │   ├── training_log.csv     # Loss/accuracy history
   │   │   └── *.png                # Visualizations
   │   └── quantization/            # Quantized model
   │       ├── best_model.onnx      # Final ONNX model
   │       └── golden_vectors/      # Test data for device
   ├── testing/                     # Test results
   └── compilation/
       └── artifacts/               # Device-ready files
           ├── mod.a                # Compiled model library
           └── tvmgen_default.h     # API header

Step 4: Understand the Config
-----------------------------

Open ``examples/generic_timeseries_classification/config.yaml``:

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55

   dataset:
     dataset_name: generic_timeseries_classification

   data_processing_feature_extraction:
     feature_extraction_name: Generic_1024Input_FFTBIN_64Feature_8Frame
     variables: 1

   training:
     model_name: CLS_1k_NPU
     batch_size: 256
     training_epochs: 20

   testing: {}
   compilation: {}

Key parameters:

* ``task_type``: What ML task to perform
* ``target_device``: Which MCU to compile for
* ``model_name``: Which model architecture to use
* ``feature_extraction_name``: How to preprocess data

Step 5: Try a Different Example
-------------------------------

Run the arc fault detection example:

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/dc_arc_fault/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         run_tinyml_modelzoo.bat examples\dc_arc_fault\config.yaml

Expected Results
----------------

For the hello world example, you should see:

* **Training accuracy**: ~98-100%
* **Quantized accuracy**: ~95-100%
* **Model size**: ~1K parameters
* **Training time**: 1-5 minutes (CPU)

Next Steps
----------

* :doc:`first_example` - Detailed walkthrough
* :doc:`understanding_config` - Config file reference
* :doc:`/byod/index` - Use your own data
* :doc:`/deployment/ccs_integration` - Deploy to a device
