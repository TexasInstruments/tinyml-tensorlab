===================
Hello World Example
===================

The Hello World example is the simplest way to get started with Tiny ML Tensorlab.
It classifies simple waveforms (sine, square, sawtooth) using a small neural network.

Overview
--------

* **Task**: Time series classification (3 classes)
* **Dataset**: Synthetic waveforms (auto-downloaded)
* **Model**: CLS_1k_NPU (~1,000 parameters)
* **Target**: F28P55 (NPU device)
* **Training time**: ~2-5 minutes on CPU

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/hello_world/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\hello_world\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55

   dataset:
     dataset_name: hello_world_example_dsg

   data_processing_feature_extraction:
     feature_extraction_name: Generic_1024Input_FFTBIN_64Feature_8Frame
     variables: 1

   training:
     model_name: CLS_1k_NPU
     batch_size: 256
     training_epochs: 20
     num_gpus: 0

   testing: {}
   compilation: {}

Understanding the Dataset
-------------------------

The dataset contains three classes of synthetic waveforms:

* **Sine wave** - Smooth sinusoidal pattern
* **Square wave** - Alternating high/low values
* **Sawtooth wave** - Linear ramp pattern

Each waveform is a 1D time series with 1024 samples.

Feature Extraction
------------------

The ``Generic_1024Input_FFTBIN_64Feature_8Frame`` preset:

1. Applies 1024-point FFT to convert to frequency domain
2. Bins the FFT output into 64 frequency features
3. Concatenates 8 frames for temporal context
4. Total input to model: 64 × 8 = 512 features

Expected Results
----------------

After training, you should see:

.. code-block:: text

   Float32 Model:
   Accuracy: 98-100%
   F1-Score: ~1.0

   Quantized Model:
   Accuracy: 95-100%

Output Location
---------------

Results are saved to::

   ../tinyml-modelmaker/data/projects/hello_world_example_dsg/run/<timestamp>/CLS_1k_NPU/

Key files:

* ``training/base/best_model.pt`` - Trained model
* ``training/quantization/best_model.onnx`` - Quantized ONNX
* ``compilation/artifacts/mod.a`` - Compiled for device

Variations to Try
-----------------

**Different Feature Extraction**

Try raw time domain instead of FFT:

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: Generic_512Input_RAW_512Feature_1Frame

**Different Model Size**

Try a smaller or larger model:

.. code-block:: yaml

   training:
     model_name: CLS_100_NPU  # Smaller
     # or
     model_name: CLS_4k_NPU   # Larger

**Different Target Device**

Compile for a different device:

.. code-block:: yaml

   common:
     target_device: MSPM0G3507

Next Steps
----------

After successfully running Hello World:

1. Try :doc:`arc_fault` - Real-world arc fault detection
2. Explore :doc:`/byod/index` - Use your own data
3. Read :doc:`/getting_started/understanding_config` - Learn all config options
