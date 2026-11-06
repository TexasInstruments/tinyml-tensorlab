==============================
Edge AI Studio Model Composer
==============================

Edge AI Studio Model Composer is Texas Instruments' no-code web platform for training and deploying AI models on
MCUs. This section describes how it relates to Tiny ML ModelZoo and how to continue in the command line with a
Model Composer project.

.. note::

   The Model Composer web application is maintained separately from this repository. The pages in this section
   describe the general workflow carried over from the 1.4.0 user guide and were **not re-verified against the
   current web UI** for 1.5.0. Menu names, supported tasks and export options may differ.
   Start from https://dev.ti.com/edgeaistudio/ for the current interface, and see the
   `Edge AI Studio for MCUs <https://www.ti.com/tool/download/EDGE-AI-STUDIO-MCU/>`_ tool page.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   getting_started_gui
   exporting_models

What is Model Composer?
-----------------------

Model Composer provides a graphical interface for:

* Uploading and managing datasets
* Selecting and configuring models
* Training with progress visualization
* Downloading compiled artifacts for deployment

The models offered in the GUI come from the same model descriptions as the command line flow (see
:doc:`/appendix/model_zoo_reference`).

GUI vs command line
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - Model Composer (GUI)
     - Tiny ML ModelZoo (command line)
   * - Learning curve
     - Low, point and click
     - Medium, requires a config YAML
   * - Customization
     - Guided presets
     - Full flexibility, see :doc:`/appendix/config_reference`
   * - Model selection
     - Models exposed in the GUI
     - All enabled models
   * - Custom models
     - Not supported
     - Supported, see :doc:`/BRING_YOUR_OWN_MODEL`
   * - Automation
     - Manual
     - Scriptable

**Use Model Composer for** quick prototyping, first experiments and users who prefer a GUI.

**Use the command line for** custom datasets or models, advanced features (NAS, mixed-precision quantization, early
stopping settings), automated pipelines and full control over the configuration. Install it as described in
:doc:`/installation/index`.
