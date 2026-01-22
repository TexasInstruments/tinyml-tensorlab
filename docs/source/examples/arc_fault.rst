========================
Arc Fault Detection
========================

Arc fault detection is one of the primary applications for Tiny ML Tensorlab.
This example demonstrates detecting electrical arcs in DC power systems.

Overview
--------

* **Task**: Binary classification (Normal vs Arc)
* **Application**: Solar inverters, battery systems, DC distribution
* **Dataset**: Real current waveforms from arc fault experiments
* **Model**: ArcFault_model_200_t (~200 parameters)

Why Arc Fault Detection?
------------------------

Electrical arcs are dangerous:

* Fire hazard in residential and commercial buildings
* Equipment damage in industrial settings
* Safety risk in electric vehicles and battery systems

Traditional protection (fuses, breakers) doesn't detect arcs reliably.
AI-based detection can identify arc signatures in current waveforms.

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/dc_arc_fault/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\dc_arc_fault\config.yaml

Configuration
-------------

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

Dataset Description
-------------------

The dataset contains current waveform samples:

* **Normal class**: Regular DC current with minor noise
* **Arc class**: Current during arc fault conditions

Data characteristics:

* Single channel (current)
* High sampling rate (captures arc frequency components)
* Pre-processed and labeled

Feature Extraction
------------------

``FFT1024Input_256Feature_1Frame_Full_Bandwidth`` applies:

1. 1024-point FFT
2. Takes first 256 frequency bins (full bandwidth)
3. Single frame (no temporal concatenation)

This captures the frequency signature of arcs, which have distinct
high-frequency components.

Available Models
----------------

Multiple arc fault models are available:

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Model
     - Parameters
     - Description
   * - ``ArcFault_model_200_t``
     - ~200
     - Minimal, fast inference
   * - ``ArcFault_model_400_t``
     - ~400
     - Balanced
   * - ``ArcFault_model_800_t``
     - ~800
     - Higher accuracy
   * - ``ArcFault_model_1400_t``
     - ~1,400
     - Maximum accuracy

Expected Results
----------------

Typical results with default configuration:

.. code-block:: text

   Float32 Model:
   Accuracy: 99%+
   F1-Score: ~0.99

   Quantized Model:
   Accuracy: 98%+

Interpreting Results
--------------------

After training, check these outputs:

**ROC Curve** (``One_vs_Rest_MultiClass_ROC_test.png``):

.. figure:: /_static/img/examples/arc_fault/One_vs_Rest_MultiClass_ROC_test.png
   :width: 600px
   :align: center
   :alt: ROC Curve for Arc Fault Detection

   One-vs-Rest Multi-class ROC curves showing excellent class separation (AUC close to 1.0)

* AUC should be close to 1.0
* Shows trade-off between detection rate and false alarms

**Class Score Histogram** (``Histogram_Class_Score_differences_test.png``):

.. figure:: /_static/img/examples/arc_fault/Histogram_Class_Score_differences_test.png
   :width: 600px
   :align: center
   :alt: Class Score Histogram

   Distribution of class score differences showing clear separation between correct and incorrect predictions

* Shows how confidently the model separates classes
* Wide separation indicates robust classification

**FPR/TPR CSV** (``fpr_tpr_thresholds.csv``):

* Use to select operating threshold for your application
* Balance between catching arcs and avoiding false alarms

Deployment Considerations
-------------------------

For real-world deployment:

**False Positive Rate**

In safety applications, you may prefer:

* Higher sensitivity (catch all arcs, accept some false alarms)
* Or higher specificity (fewer false alarms, may miss subtle arcs)

Adjust the threshold in your device code accordingly.

**Inference Latency**

Arc detection needs to be fast:

* F28P55 with NPU: ~100-200 µs
* Sufficient for real-time protection

AC Arc Fault Detection
----------------------

For AC systems, use the AC arc fault example:

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/ac_arc_fault/config.yaml

AC arcs have different signatures due to the alternating current,
requiring different feature extraction settings.

Next Steps
----------

* Deploy to device: :doc:`/deployment/npu_device_deployment`
* Try with your data: :doc:`/byod/classification_format`
* Explore other examples: :doc:`motor_bearing_fault`
