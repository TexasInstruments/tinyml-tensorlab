===================================
NILM Appliance Usage Classification
===================================

Non-Intrusive Load Monitoring - identify active appliances from aggregate power data.

Overview
--------

Non-Intrusive Load Monitoring (NILM) disaggregates total household power
consumption to identify individual appliances. This example demonstrates
how to classify which appliances are currently active using aggregate
power measurements.

**Application**: Smart home, energy management, utility monitoring

**Task Type**: Time Series Classification

**Data Type**: Multivariate (power measurements)

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'nilm_appliance_usage_classification'

   training:
     model_name: 'CLS_4k_NPU'
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
         ./run_tinyml_modelzoo.sh examples/nilm_appliance_usage_classification/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\nilm_appliance_usage_classification\config.yaml

Dataset Details
---------------

**Input Variables**:

* Aggregate power consumption
* Voltage/current waveforms
* Power factor (optional)

**Classes**:

* Different appliance combinations
* Individual appliance states (on/off)

On-Device Results
-----------------

.. figure:: /_static/img/examples/nilm/ondevice_results.png
   :width: 600px
   :align: center
   :alt: On-Device Results

   NILM inference results on target device

PLAID Dataset Variant
---------------------

An alternative configuration using the PLAID (Plug-Level Appliance
Identification Dataset) is also available:

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/PLAID_nilm_classification/config.yaml

See Also
--------

* `NILM Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/nilm_appliance_usage_classification>`_
* `PLAID Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/PLAID_nilm_classification>`_
