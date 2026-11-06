==============
Grid Stability
==============

Predict the small-signal stability of a power grid from node-level parameters.

Example folder: ``examples/grid_stability``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/grid_stability>`_)

Overview
--------

The dataset comes from the UCI Machine Learning Repository. It is simulation-derived and describes a
4-node "star" network with one central generation node and three peripheral consumer nodes. For each node
there are three parameters:

* ``tau1`` to ``tau4``: reaction time of the node (larger values mean slower dynamics)
* ``p1`` to ``p4``: active power of the node (generation or consumption)
* ``g1`` to ``g4``: price elasticity coefficient (sensitivity to price or control signals)

The label is stable or unstable.

* **Application**: smart grid, power system operation, renewable energy integration
* **Task**: classification (``generic_timeseries_classification``)
* **Device**: F28P55 (``config.yaml``)
* **Model**: ``CLS_ResCat_3k``

.. figure:: /_static/img/examples/grid_stability/four_node_start.svg
   :width: 500px
   :align: center
   :alt: Four-node star grid

   Four-node star network used for the grid stability dataset

Preparing the Dataset
---------------------

The config downloads ``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/grid_stability_dataset.zip``
automatically. To regenerate the zip from the UCI source, run the helper script and point
``dataset.input_data_path`` at the output (``examples/grid_stability/grid_stability_dataset.zip``):

.. code-block:: bash

   cd examples/grid_stability
   python grid_stability.py

Because the data is simulated, make sure train and validation splits cover the range of operating conditions
and avoid overfitting to narrow parameter ranges.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/grid_stability/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\grid_stability\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55
   dataset:
     dataset_name: grid_stability
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/grid_stability_dataset.zip
   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 256
     stride_size: 0.01
     variables: 6
     feature_extraction_name: Custom_Default
   training:
     model_name: CLS_ResCat_3k
     num_gpus: 0
     early_stopping: true
     early_stopping_patience: 5
   testing: {}
   compilation: {}

Understanding the Outputs
-------------------------

The ``training/base`` folder holds the float model and feature data; ``training/quantization`` holds the INT8
model and its analysis. The PCA plots show whether the classes separate well in feature space. See
:doc:`/getting_started/understanding_run_output` for all generated files.

Results from the 1.5.0 Test Run
-------------------------------

In the automated 1.5.0 config test run, ``config.yaml`` reached a test accuracy of 100.00% with an AUC ROC of
1.000 (``CLS_ResCat_3k``, 1,018 parameters). The example readme does not publish accuracy numbers.

See Also
--------

* :doc:`electrical_fault`: transmission line fault classification
* :doc:`grid_fault_detection`
