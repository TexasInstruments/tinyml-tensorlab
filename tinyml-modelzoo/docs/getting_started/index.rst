===============
Getting Started
===============

This section takes you from a fresh install to a trained, quantized and compiled model,
and explains what every file in the run folder means.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   first_example
   running_examples
   understanding_config
   /CONFIG_YAML_GUIDE
   understanding_run_output

5-Minute Quickstart
-------------------

After :doc:`/installation/index`, from the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

   .. tab:: Windows (cmd)

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml

   .. tab:: Windows (PowerShell)

      .. code-block:: powershell

         ./run_tinyml_modelzoo.ps1 examples/generic_timeseries_classification/config.yaml

This runs the "Hello World" example: it downloads a waveform dataset (sine, square,
sawtooth), extracts FFT features, trains a ~1K-parameter model (``CLS_1k_NPU``),
quantizes it and compiles it for the F28P55. Output is written to
``./data/projects/generic_timeseries_classification/run/<date-time>/CLS_1k_NPU/``.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Run the hello world example and find the results.

   .. grid-item-card:: First Example
      :link: first_example
      :link-type: doc

      Walk through DC arc fault detection end to end.

   .. grid-item-card:: Understanding the Run Output
      :link: understanding_run_output
      :link-type: doc

      What every file in a run folder is and which one to deploy.

   .. grid-item-card:: Config YAML Guide
      :link: /CONFIG_YAML_GUIDE
      :link-type: doc

      Create or adapt a config for your data and device.
