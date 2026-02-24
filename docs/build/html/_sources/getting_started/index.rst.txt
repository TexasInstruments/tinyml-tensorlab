===============
Getting Started
===============

This section helps you get up and running with Tiny ML Tensorlab quickly.
Follow these guides to train your first model and understand the toolchain workflow.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   first_example
   understanding_config
   running_examples

5-Minute Quickstart
-------------------

After completing the installation, you can train your first model in just a few commands:

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml

This runs the "Hello World" example, which:

1. Downloads a simple waveform classification dataset (sine, square, sawtooth)
2. Applies feature extraction (FFT)
3. Trains a small neural network (~1K parameters)
4. Quantizes the model for MCU deployment
5. Compiles for the target device (F28P55 by default)

Output artifacts are saved to ``../tinyml-modelmaker/data/projects/generic_timeseries_classification/``.
