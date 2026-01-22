================
Electrical Fault
================

Classify transmission line faults using voltage and current measurements.

Overview
--------

This example demonstrates classification of electrical faults in power
transmission lines. It uses voltage and current measurements to identify
different fault types including line-to-ground, line-to-line, and three-phase faults.

**Application**: Power grid protection, substation automation, fault localization

**Task Type**: Time Series Classification

**Data Type**: Multivariate (voltage and current signals)

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'electrical_fault'

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
         ./run_tinyml_modelzoo.sh examples/electrical_fault/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\electrical_fault\config.yaml

Dataset Details
---------------

**Input Variables**:

* Phase voltages (Va, Vb, Vc)
* Phase currents (Ia, Ib, Ic)

**Fault Classes**:

* No fault (normal)
* Line-to-ground fault (LG)
* Line-to-line fault (LL)
* Line-to-line-to-ground fault (LLG)
* Three-phase fault (LLL)

**Simulink Model:**

.. figure:: /_static/img/examples/electrical_fault/simulink.png
   :width: 700px
   :align: center
   :alt: Simulink Model for Electrical Fault

   Simulink model used for generating electrical fault training data

See Also
--------

* :doc:`grid_stability` - Power grid stability prediction
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/electrical_fault>`_
