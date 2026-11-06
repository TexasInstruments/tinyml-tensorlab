========================
Model Composer Overview
========================

Edge AI Studio Model Composer is a graphical front end for the same training and compilation flow that Tiny ML
ModelZoo runs from the command line.

.. note::

   The web UI is not part of this repository and could not be verified for the 1.5.0 guide. Treat the feature
   list as a general description and check the live application.

What it does
------------

* Upload and manage datasets
* Configure training visually
* Train models with progress visualization
* Export trained models and compiled artifacts for deployment

Accessing Model Composer
------------------------

Model Composer runs in the browser through Edge AI Studio: https://dev.ti.com/edgeaistudio/. You need a TI account.
Use a current version of Chrome, Firefox or Edge and an Internet connection.

Typical features
----------------

**Dataset management:** upload CSV or ZIP datasets, preview data, time series plots, format validation.

**Model configuration:** device selection, model selection, feature extraction presets.

**Training:** start training, loss graphs and progress indicators.

**Results:** accuracy metrics, confusion matrix and other plots (the command line flow produces comparable plots,
see :doc:`/getting_started/understanding_run_output`).

**Export:** trained model, compiled artifacts and a configuration file for the command line. See
:doc:`exporting_models`.

Limitations
-----------

Compared with the command line, the GUI has fewer configuration options, may not expose all presets, and has upload
size and processing time limits. For custom models (:doc:`/BRING_YOUR_OWN_MODEL`) or custom datasets
(:doc:`/byod/index`) use the command line.

Combining the GUI and the command line
---------------------------------------

A common flow is to prototype in Model Composer, export the configuration, then refine it with the command line. The
steps are in :doc:`exporting_models`.

Next steps
----------

* :doc:`getting_started_gui` for a step-by-step walkthrough
* :doc:`exporting_models` to get your model and configuration out of the GUI
* :doc:`/getting_started/quickstart` for the command line flow
