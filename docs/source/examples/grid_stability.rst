==============
Grid Stability
==============

Predict power grid stability from node parameters.

Overview
--------

This example predicts the stability of a power grid based on node-level
parameters. It helps operators anticipate instability conditions and take
preventive actions before grid failures occur.

**Application**: Smart grid, power system operation, renewable energy integration

**Task Type**: Time Series Classification

**Data Type**: Multivariate (node parameters)

**Four-Node Star Network:**

.. figure:: /_static/img/examples/grid_stability/four_node_start.svg
   :width: 500px
   :align: center
   :alt: Four-Node Star Grid

   Four-node star network used for grid stability analysis

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'grid_stability'

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
         ./run_tinyml_modelzoo.sh examples/grid_stability/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\grid_stability\config.yaml

Dataset Details
---------------

**Input Variables**:

* Reaction time of participants
* Power consumed/produced
* Price elasticity coefficients
* Other node parameters

**Classes**:

* Stable
* Unstable

See Also
--------

* :doc:`electrical_fault` - Transmission line fault classification
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/grid_stability>`_
