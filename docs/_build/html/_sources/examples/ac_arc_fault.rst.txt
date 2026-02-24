=============
AC Arc Fault
=============

Detect AC arc faults in electrical systems using current waveform analysis.

Overview
--------

AC arc faults occur when electrical current flows through an unintended path,
often caused by damaged insulation, loose connections, or worn conductors.
This example demonstrates how to detect these dangerous conditions using
machine learning on current sensor data.

**Application**: Electrical safety systems, circuit breakers, residential/commercial protection

**Task Type**: Time Series Classification

**Data Type**: Univariate (current waveform)

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'ac_arc_fault'

   training:
     model_name: 'ArcFault_model_700_t'
     training_epochs: 50
     batch_size: 32

   testing: {}
   compilation: {}

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/ac_arc_fault/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\ac_arc_fault\config.yaml

Dataset Details
---------------

The AC arc fault dataset contains current waveforms sampled during normal
operation and various arc fault conditions.

**Classes**:

* Normal operation
* Arc fault conditions

**Input Features**: Current waveform samples

Recommended Models
------------------

.. important::

   The ``ArcFault_model_*`` models listed below are only available in
   TI's **Edge AI Studio** (GUI) and are not included in Tensorlab.
   Use the generic ``CLS_*_NPU`` models (e.g., ``CLS_1k_NPU``,
   ``CLS_4k_NPU``) as equivalent alternatives in Tensorlab.

**Edge AI Studio models:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model
     - Parameters
     - Use Case
   * - ``ArcFault_model_200_t``
     - ~200
     - Minimal footprint
   * - ``ArcFault_model_300_t``
     - ~300
     - Balanced
   * - ``ArcFault_model_700_t``
     - ~700
     - Higher accuracy
   * - ``ArcFault_model_1400_t``
     - ~1,400
     - Maximum accuracy

**Tensorlab alternatives**: Use ``CLS_500_NPU``, ``CLS_1k_NPU``, or
``CLS_4k_NPU`` for equivalent performance.

See Also
--------

* :doc:`arc_fault` - DC arc fault detection
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/ac_arc_fault>`_
