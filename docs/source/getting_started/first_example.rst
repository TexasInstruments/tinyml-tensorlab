=============
First Example
=============

This guide provides a detailed walkthrough of running an arc fault detection
example from start to finish.

Overview
--------

DC Arc Fault Detection is a common application where we need to identify
dangerous electrical arcs in DC power systems (solar panels, batteries, etc.).

We'll train a model to classify current waveforms as either "normal" or "arc".

Step 1: Examine the Configuration
---------------------------------

Open ``examples/dc_arc_fault/config.yaml``:

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'
     run_name: '{date-time}/{model_name}'

   dataset:
     enable: True
     dataset_name: 'dc_arc_fault_example_dsk'
     input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/dc_arc_fault_example_dsk.zip'

   data_processing_feature_extraction:
     feature_extraction_name: 'FFT1024Input_256Feature_1Frame_Full_Bandwidth'
     variables: 1

   training:
     enable: True
     model_name: 'ArcFault_model_200_t'
     batch_size: 256
     training_epochs: 20

   testing:
     enable: True

   compilation:
     enable: True

**Configuration Breakdown:**

``common`` Section
   * ``target_module: 'timeseries'`` - Working with time series data
   * ``task_type: 'generic_timeseries_classification'`` - Binary/multi-class classification
   * ``target_device: 'F28P55'`` - Compile for F28P55 (NPU device)

``dataset`` Section
   * The dataset is automatically downloaded from the URL
   * Contains current waveforms labeled as "arc" or "normal"

``data_processing_feature_extraction`` Section
   * Uses a preset that applies 1024-point FFT
   * Extracts 256 frequency features per frame
   * ``variables: 1`` - Single channel (current only)

``training`` Section
   * Uses a specialized arc fault model (~200 parameters)
   * Trains for 20 epochs with batch size 256

Step 2: Run Training
--------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/dc_arc_fault/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\dc_arc_fault\config.yaml

You'll see output showing:

1. **Dataset download and extraction**
2. **Data preprocessing and feature extraction**
3. **Training progress** (loss, accuracy per epoch)
4. **Testing results**
5. **Quantization**
6. **Compilation**

Step 3: Understand Training Output
----------------------------------

During training, you'll see something like:

.. code-block:: text

   Epoch 1/20
   Train Loss: 0.6543 | Train Acc: 62.34%
   Val Loss: 0.4521 | Val Acc: 78.92%

   Epoch 10/20
   Train Loss: 0.0234 | Train Acc: 99.12%
   Val Loss: 0.0312 | Val Acc: 98.87%

   Epoch 20/20
   Train Loss: 0.0089 | Train Acc: 99.89%
   Val Loss: 0.0198 | Val Acc: 99.45%

   Best model saved at epoch 18 with val accuracy: 99.52%

Key metrics:

* **Train Loss/Acc**: Performance on training data
* **Val Loss/Acc**: Performance on held-out validation data
* **Best model**: Checkpoint with highest validation accuracy

Step 4: Examine Output Files
----------------------------

Navigate to the output directory::

   ../tinyml-modelmaker/data/projects/dc_arc_fault_example_dsk/run/<timestamp>/ArcFault_model_200_t/

**Training Outputs** (``training/base/``):

.. list-table::
   :widths: 30 70

   * - ``best_model.pt``
     - PyTorch checkpoint of best model
   * - ``training_log.csv``
     - Epoch-by-epoch metrics
   * - ``pca_on_feature_extracted_train_data.png``
     - PCA visualization of features
   * - ``One_vs_Rest_MultiClass_ROC_test.png``
     - ROC curve (if classification)
   * - ``Histogram_Class_Score_differences_test.png``
     - Class score distribution

**Quantization Outputs** (``training/quantization/``):

.. list-table::
   :widths: 30 70

   * - ``best_model.onnx``
     - Quantized ONNX model
   * - ``golden_vectors/test_vector.c``
     - Test inputs/outputs for device validation
   * - ``golden_vectors/user_input_config.h``
     - Feature extraction config for device

**Compilation Outputs** (``compilation/artifacts/``):

.. list-table::
   :widths: 30 70

   * - ``mod.a``
     - Compiled model library
   * - ``tvmgen_default.h``
     - C API for model inference

Step 5: Analyze Results
-----------------------

**Check Training Curves**

Open ``training/base/training_log.csv`` or the generated plots:

* Loss should decrease over epochs
* Accuracy should increase and stabilize
* Val metrics should track train metrics (no overfitting)

**Check Classification Performance**

The ROC curve shows true positive vs false positive rates:

* Area Under Curve (AUC) close to 1.0 indicates good performance
* Review threshold selection for your application needs

**Check Quantization Impact**

Compare float32 vs quantized accuracy in the console output:

* Small accuracy drop (<2%) is normal
* Large drop indicates quantization issues

Step 6: Customize the Example
-----------------------------

**Try a Different Model**

Edit the config to use a larger model:

.. code-block:: yaml

   training:
     model_name: 'ArcFault_model_1400_t'  # ~1400 parameters

**Try a Different Device**

Compile for a non-NPU device:

.. code-block:: yaml

   common:
     target_device: 'F2837'  # Non-NPU C2000 device

**Use Your Own Data**

Replace the dataset section with your own data path:

.. code-block:: yaml

   dataset:
     dataset_name: 'my_arc_fault_data'
     input_data_path: '/path/to/your/dataset'

See :doc:`/byod/index` for dataset format requirements.

Next Steps
----------

* :doc:`understanding_config` - Complete config reference
* :doc:`/deployment/ccs_integration` - Deploy to device
* :doc:`/examples/index` - More examples
