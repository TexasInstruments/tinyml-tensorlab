===========================
Human Activity Recognition
===========================

Classify human activities from smartphone accelerometer data (WISDM dataset). The example also shows
support for residual connections and how to change model parameters with ``model_config``.

Example folder: ``examples/human_activity_recognition``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/human_activity_recognition>`_).
This folder was named ``branched_model_parameters`` in the 1.4.0 documentation.

Overview
--------

The WISDM (WIreless Sensor Data Mining) dataset is a public dataset of smartphone sensor recordings used
for activity recognition. See the `WISDM site <https://www.cis.fordham.edu/wisdm/dataset.php>`_ for the
activity list.

* **Application**: wearables, fitness trackers, smart home, elderly care
* **Task**: classification (``generic_timeseries_classification``)
* **Data**: 3 variables (accelerometer X, Y, Z), 128-sample windows
* **Dataset**: ``wisdm_example`` from ``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/wisdm_dataset.zip``
* **Device**: F28P55
* **Model**: ``CLS_ResCat_3k`` (residual/concatenation architecture)

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/human_activity_recognition/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\human_activity_recognition\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55
   dataset:
     dataset_name: wisdm_example
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/wisdm_dataset.zip
   data_processing_feature_extraction:
     data_proc_transforms:
     - SimpleWindow
     frame_size: 128
     stride_size: 0.01
     variables: 3
   training:
     model_name: CLS_ResCat_3k
     model_config: examples/human_activity_recognition/residual_network_config.yaml
     batch_size: 32
     training_epochs: 10
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 0
   testing: {}
   compilation: {}

Preparing the WISDM Dataset
---------------------------

The config downloads the prepared zip automatically. To rebuild it yourself, run
``python wisdm.py`` in the example folder (requires ``requests`` and ``pandas``): it downloads the raw
WISDM data, cleans it, splits each class into ``NUM_SPLITS`` files and zips the result in the format
expected by ModelZoo.

The ``CLS_ResCat_3k`` Model
---------------------------

.. figure:: /_static/img/examples/human_activity_recognition/model_onnx.png
   :width: 180px
   :align: center
   :alt: CLS_ResCat_3k ONNX graph

   ``CLS_ResCat_3k`` with residual connections

The model has an input batch normalization followed by two branches. The first branch has three
convolution layers, the second a single convolution. The branch outputs are concatenated, pooled,
flattened and fed to a fully connected layer.

The channel counts are configurable through ``training.model_config``, which points to
``residual_network_config.yaml``:

.. code-block:: yaml

   out_channel_layer1: 8
   out_channel_layer2: 16
   out_channel_layer3: 32

Comparison of Models
--------------------

The example ``readme.md`` compares three models on this dataset (results measured with an earlier release):

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Model
     - Parameters
     - Accuracy
     - AUC ROC
   * - ``CLS_1k_NPU``
     - 1,316
     - 94.92%
     - 0.995
   * - ``CLS_ResCat_3k``
     - 3,132
     - 93.84%
     - 0.965
   * - ``CLS_13k_NPU``
     - 14,124
     - 94.45%
     - 0.987

The purpose of the example is to demonstrate residual connections, not to show that ``CLS_ResCat_3k`` is
better than the other models.

In the automated 1.5.0 config test run, ``config.yaml`` reached a test accuracy of 88.87% with an AUC ROC of
0.931 (900 parameters reported in that log). The shipped config uses only 10 epochs, so expect results to
vary between runs.

See Also
--------

* :doc:`/CONFIG_YAML_GUIDE`: all configuration keys, including ``model_config``
* :doc:`/BRING_YOUR_OWN_MODEL`: add your own architecture
