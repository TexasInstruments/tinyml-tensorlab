======================================
Generic Time Series Classification
======================================

This example serves as a **"Hello World" introduction** to time series classification
using the TinyML ModelMaker toolchain. It demonstrates how to use any generic
time series classification task with our toolchain.

Overview
--------

* **Task**: Time series classification (3 classes)
* **Dataset**: Synthetic waveforms (sine, square, sawtooth)
* **Model**: CLS_1k_NPU (~1,000 parameters)
* **Target**: F28P55 (NPU device)
* **Training time**: ~2-5 minutes on CPU

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml

Understanding the Dataset
-------------------------

The dataset contains three classes of synthetic waveforms:

* **Sine wave** - Smooth sinusoidal pattern
* **Square wave** - Alternating high/low values
* **Sawtooth wave** - Linear ramp pattern

Each waveform is a 1D time series with 1024 samples.

Dataset Format
--------------

For classification tasks, ModelMaker expects the dataset in this structure:

.. code-block:: text

   {dataset_name}.zip/
   |
   |-- classes/
         |-- class_1/
         |     |-- file1.csv
         |     |-- file2.csv
         |
         |-- class_2/
         |     |-- file1.csv
         |     |-- file2.csv
         |
         |-- class_N/
               |-- ...

.. note::

   For classification tasks, the folder names under ``classes/`` become the
   class labels. No separate annotation files are required.

Configuration
-------------

**common section:**

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55

**dataset section:**

.. code-block:: yaml

   dataset:
     dataset_name: generic_timeseries_classification

**data_processing_feature_extraction section:**

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: Generic_1024Input_FFTBIN_64Feature_8Frame
     variables: 1

**training section:**

.. code-block:: yaml

   training:
     model_name: CLS_1k_NPU
     batch_size: 256
     training_epochs: 20
     num_gpus: 0

   testing: {}
   compilation: {}

Feature Extraction
------------------

The ``Generic_1024Input_FFTBIN_64Feature_8Frame`` preset:

1. Applies 1024-point FFT to convert to frequency domain
2. Bins the FFT output into 64 frequency features
3. Concatenates 8 frames for temporal context
4. Total input to model: 64 x 8 = 512 features

Evaluation Metrics
------------------

**Accuracy**

Percentage of correctly classified samples.

* Range: 0% to 100%
* Ideal value: 100% (higher is better)

**F1-Score**

Harmonic mean of precision and recall.

* Range: 0 to 1
* Ideal value: 1 (higher is better)

Expected Results
----------------

.. code-block:: text

   Float32 Model:
   Accuracy: 98-100%
   F1-Score: ~1.0

   Quantized Model:
   Accuracy: 95-100%

Output Location
---------------

Results are saved to::

   ../tinyml-modelmaker/data/projects/generic_timeseries_classification/run/<timestamp>/CLS_1k_NPU/

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

After successfully running this generic example:

1. Try :doc:`arc_fault` - Real-world arc fault detection
2. Explore :doc:`/byod/classification_format` - Use your own classification data
3. Read :doc:`/getting_started/understanding_config` - Learn all config options
