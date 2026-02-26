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

.. important::

   The ``ArcFault_model_*`` models listed below are only available in
   TI's **Edge AI Studio** (GUI) and are not included in the Tensorlab
   package. In Tensorlab, use the generic ``CLS_*`` or ``CLS_*_NPU``
   models instead -- they offer similar or better performance. For
   example, ``CLS_500_NPU`` (~500 params) or ``CLS_1k_NPU`` (~1,000
   params) are suitable replacements.

**Edge AI Studio models:**

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

**Tensorlab alternatives:**

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Model
     - Parameters
     - Description
   * - ``CLS_500_NPU``
     - ~500
     - Small, NPU-optimized
   * - ``CLS_1k_NPU``
     - ~1,000
     - Baseline NPU model
   * - ``CLS_2k_NPU``
     - ~2,000
     - Medium NPU model
   * - ``CLS_4k_NPU``
     - ~4,000
     - Recommended NPU model

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

Overview
^^^^^^^^

AC arc fault detection is an Edge AI solution targeting the
`MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ microcontroller with
on-chip NPU. It detects series arc faults in residential and commercial
electrical systems -- the leading cause of electrical fires -- using machine
learning inference on data captured through the
`TIDA-010971 <https://www.ti.com/lit/df/slvrbz4/slvrbz4.pdf>`_ analog front
end with a PCB Rogowski coil. This enables Arc Fault Circuit Interrupter
(AFCI) products that comply with the National Electrical Code (NEC).

Why AC Arc Fault Detection?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Arc faults cause more than 30,000 home fires each year in the United States
alone, resulting in hundreds of deaths and over one billion dollars in property
damage. Traditional circuit breakers are not designed to detect low-level arc
faults and routinely fail to interrupt them before ignition occurs.

Edge AI addresses this gap by enabling:

* **Complex pattern recognition** -- identifying subtle arc signatures that
  rule-based approaches miss.
* **Multi-feature analysis** -- combining frequency-domain and temporal
  information in a single inference pass.
* **Adaptive detection** -- maintaining accuracy across diverse load types and
  operating conditions.

Key performance targets for the AC arc fault solution:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Target
   * - Response time
     - < 10 ms
   * - Detection accuracy
     - > 95%
   * - Compliance
     - `UL 1699 <https://code-authorities.ul.com/wp-content/uploads/2014/05/Dini2.pdf>`_

System Components
^^^^^^^^^^^^^^^^^

**Hardware**

* `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ -- Arm Cortex-M0+
  microcontroller with integrated NPU.
* `TIDA-010971 <https://www.ti.com/lit/df/slvrbz4/slvrbz4.pdf>`_ -- Analog
  front end reference design featuring a PCB Rogowski coil for non-intrusive
  current sensing.

**Software**

* Code Composer Studio (CCS) 12.x or later
* `MSPM0 SDK <https://www.ti.com/tool/MSPM0-SDK>`_ 2.08.00 or later
* TI Edge AI Studio / TI Tiny ML Tensorlab

Running the AC Arc Fault Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/ac_arc_fault/config_MSPM0.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         .\run_tinyml_modelzoo.bat examples\ac_arc_fault\config_MSPM0.yaml

AC Arc Fault Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   common:
     task_type: arc_fault
     target_device: MSPM0G5187

   dataset:
     dataset_name: ac_arc_fault_log300_example_dsk
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/ac_arc_fault_log300.zip

   data_processing_feature_extraction:
     feature_extraction_name: ArcFault_512Input_FE_RFFT_32Feature_8Frame_1InputChannel_removeDC_Full_Bandwidth
     store_feat_ext_data: false

   training:
     model_name: CLS_1k_NPU
     batch_size: 30
     training_epochs: 30
     num_gpus: 0
     quantization: 2
     learning_rate: 1e-5
     weight_decay: 1e-5

   testing: {}
   compilation: {}

AC Arc Fault Dataset
^^^^^^^^^^^^^^^^^^^^

The example dataset (``ac_arc_fault_log300.zip``) contains 100+ captures per
load type, organized into three categories:

**Normal operations**

* Inductive loads
* LED lighting
* Switched-mode power supplies (SMPS)
* Mixed loads

**Arc fault conditions**

* Series arc faults
* Parallel arc faults

**Masking loads** (loads whose signatures can resemble arcs)

* Vacuum cleaners
* Drills
* Dimmers
* Compressors
* Resistive loads

A dedicated labelling script, ``AFCI_LabellingScript.py``, is provided in the
example directory for preparing custom datasets. Refer to
``readme_labelling.md`` in the same directory for detailed instructions.

AC Feature Extraction
^^^^^^^^^^^^^^^^^^^^^

The default feature extraction pipeline
(``ArcFault_512Input_FE_RFFT_32Feature_8Frame_1InputChannel_removeDC_Full_Bandwidth``)
processes raw ADC samples through the following stages:

1. **ADC capture** -- 512 samples at 107 kSps (~4.76 ms per frame).
2. **512-point real FFT** -- computed with the ARM CMSIS-DSP library.
3. **Complex magnitude** -- converts complex FFT output to magnitudes.
4. **DC removal** -- discards the DC component (bin 0).
5. **Binning** -- groups 8 adjacent frequency bins into one, producing 32
   features per frame.
6. **INT8 normalization** -- scales features to the INT8 range.
7. **Frame concatenation** -- concatenates 8 consecutive frames to form a
   single input vector of 256 features.

AC Arc Fault Models
^^^^^^^^^^^^^^^^^^^

.. important::

   The ``ArcFault_model_*`` models listed below are only available in TI's
   **Edge AI Studio** (GUI) and are not included in the Tensorlab package. In
   Tensorlab, use the generic ``CLS_*_NPU`` models instead -- 11 CNN and
   ResNet architectures ranging from ~100 to ~20,000 parameters with NPU
   support are available via the Tensorlab CLI. See the tinyml-modelzoo README
   for the full list.

**Edge AI Studio models (GUI only):**

.. list-table::
   :header-rows: 1
   :widths: 30 15 18 17 12 8

   * - Model
     - Parameters
     - Flash (MSPM0G5187)
     - Inference Time
     - Accuracy
     - Note
   * - ``ArcFault_model_200_t``
     - ~200
     - 3.6 KB
     - --
     - 99.60%
     -
   * - ``ArcFault_model_300_t``
     - ~300
     - 3.9 KB
     - --
     - 99.60%
     -
   * - ``ArcFault_model_700_t``
     - ~800
     - 4.5 KB
     - --
     - 99.42%
     -
   * - ``ArcFault_model_1400_t``
     - ~1,600
     - 5.6 KB
     - 0.7 ms
     - 99.88%
     - Recommended

**Tensorlab alternatives:**

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Model
     - Parameters
     - Description
   * - ``CLS_500_NPU``
     - ~500
     - Small, NPU-optimized
   * - ``CLS_1k_NPU``
     - ~1,000
     - Baseline NPU model (default in config)
   * - ``CLS_2k_NPU``
     - ~2,000
     - Medium NPU model
   * - ``CLS_4k_NPU``
     - ~4,000
     - Recommended NPU model

AC Arc Fault Performance
^^^^^^^^^^^^^^^^^^^^^^^^

Performance measured on MSPM0G5187 with NPU using the recommended
configuration:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Metric
     - Value
   * - End-to-end latency (including 8-frame voting)
     - < 150 ms
   * - Detection accuracy
     - > 99%
   * - False positive rate
     - 0.01%
   * - Precision
     - 99.97%
   * - Recall
     - 99.72%
   * - F1-Score
     - 99.84%

AC Training Details
^^^^^^^^^^^^^^^^^^^

The default training configuration uses:

* **Batch size**: 50
* **Learning rate**: 0.04
* **Optimizer**: SGD
* **Quantization**: INT8 via Quantization-Aware Training (QAT) -- required for
  NPU deployment
* **Compilation**: TI Neural Network Compiler (NNC)

.. note::

   INT8 quantization is the default and is required for deployment on the
   MSPM0G5187 NPU. The configuration file sets ``quantization: 2`` to enable
   QAT during training.

AC Arc Fault References
^^^^^^^^^^^^^^^^^^^^^^^

* `UL 1699 -- Arc-Fault Circuit Interrupters <https://code-authorities.ul.com/wp-content/uploads/2014/05/Dini2.pdf>`_
* `TIDA-010971 Reference Design <https://www.ti.com/lit/df/slvrbz4/slvrbz4.pdf>`_
* `MSPM0G5187 Product Page <https://www.ti.com/product/MSPM0G5187>`_
* `MSPM0 SDK <https://www.ti.com/tool/MSPM0-SDK>`_

Next Steps
----------

* Deploy to device: :doc:`/deployment/npu_device_deployment`
* Try with your data: :doc:`/byod/classification_format`
* Explore other examples: :doc:`motor_bearing_fault`
