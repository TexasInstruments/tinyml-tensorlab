==============================
Edge AI Studio Model Composer
==============================

Edge AI Studio Model Composer is Texas Instruments' no-code web platform
for training and deploying AI models on MCUs. This section covers how to
use the GUI as an alternative to the command-line toolchain.

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
* Training with real-time progress visualization
* Downloading compiled artifacts for deployment

**Access URL**: https://dev.ti.com/modelcomposer/

GUI vs CLI Comparison
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - Model Composer (GUI)
     - Tiny ML Tensorlab (CLI)
   * - Learning Curve
     - Low - point and click
     - Medium - requires config files
   * - Customization
     - Limited presets
     - Full flexibility
   * - Model Selection
     - Curated models
     - All models in ModelZoo
   * - Task Types
     - Arc Fault, Motor Fault
     - All 5 task types
   * - Custom Models
     - Not supported
     - Fully supported
   * - Automation
     - Manual
     - Scriptable

**When to Use Model Composer:**

* Quick prototyping
* Non-technical users
* Standard arc fault or motor fault applications
* No Python environment available

**When to Use CLI:**

* Custom datasets or models
* Advanced features (NAS, custom quantization)
* Automated pipelines
* Full control over configuration
