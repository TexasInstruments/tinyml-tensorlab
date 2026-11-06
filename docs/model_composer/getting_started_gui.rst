=======================
Getting Started (GUI)
=======================

This page outlines how to train a first model with Edge AI Studio Model Composer.

.. note::

   Step names follow the 1.4.0 guide and were **not re-verified against the current web UI**. If a menu or button is
   named differently, follow the live application.

Prerequisites
-------------

1. A TI account (https://www.ti.com)
2. A current browser (Chrome, Firefox or Edge)
3. A dataset, or one of the built-in example datasets

Step 1: Open Model Composer
---------------------------

1. Go to https://dev.ti.com/edgeaistudio/ and log in.
2. Select **Model Composer** and create a new project.

Step 2: Create the project
--------------------------

1. Enter a project name (for example "Arc Fault Detection").
2. Select the task type (for example time series classification).
3. Select the target device (for example F28P55).

Step 3: Add a dataset
---------------------

Use an example dataset, or upload your own as a ZIP file. The expected layout of a custom classification dataset is
described in :doc:`/byod/classification_format` (other tasks: :doc:`/byod/index`). Preview the data to verify it
after upload.

Step 4: Feature extraction
--------------------------

Pick a feature extraction preset, or customize the parameters. Start with a preset. The presets correspond to the
``feature_extraction_name`` values of the command line flow, see :doc:`/features/feature_extraction`.

Step 5: Training settings
-------------------------

Select a model and set the number of epochs, batch size and learning rate. For NPU devices (F28P55, AM13E2,
MSPM0G5187) choose an NPU-compliant model and enable quantization. Model names and their target devices are listed in
:doc:`/appendix/model_zoo_reference`.

Step 6: Train
-------------

Start training and follow the progress and the loss graph. When it finishes, the GUI shows a results summary.

Step 7: Analyze and test
------------------------

Review the accuracy, per-class results and confusion matrix, then evaluate on the test set. If the test results are
much lower than training, the model may be overfitting.

Step 8: Export
--------------

Export the compiled artifacts or the configuration. See :doc:`exporting_models`.

Next steps
----------

* Try your own data and compare runs.
* Reproduce the run on the command line: :doc:`/getting_started/first_example`.
* Deploy to the device: :doc:`/deployment/ccs_integration` and :doc:`/DEPLOYING_TO_SDK_EXAMPLE`.
