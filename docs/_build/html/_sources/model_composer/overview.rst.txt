========================
Model Composer Overview
========================

Edge AI Studio Model Composer provides a graphical user interface (GUI)
for Tiny ML Tensorlab, making it easier to train and deploy models without
using the command line.

What is Model Composer?
-----------------------

Model Composer is TI's cloud-based or desktop GUI for:

* Uploading and managing datasets
* Configuring training parameters visually
* Training models with progress visualization
* Exporting trained models for deployment

It provides the same functionality as the CLI tools but through a
user-friendly web interface.

Model Composer vs CLI
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Model Composer (GUI)
     - CLI Tools
   * - Ease of use
     - Beginner-friendly
     - Requires CLI knowledge
   * - Flexibility
     - Guided workflows
     - Full configuration control
   * - Automation
     - Manual only
     - Scriptable
   * - Visualization
     - Built-in
     - Requires external tools
   * - Remote access
     - Cloud-accessible
     - Local only

**When to use Model Composer:**

* First-time users
* Quick experiments
* Visual dataset inspection
* Non-technical stakeholders

**When to use CLI:**

* Automation and scripting
* Advanced configuration
* Custom workflows
* Integration with CI/CD

Accessing Model Composer
------------------------

**Cloud Version:**

Access via Edge AI Studio:

1. Visit https://dev.ti.com/edgeaistudio/
2. Create or log into your TI account
3. Select "Model Composer" from the dashboard

**Local Version (if available):**

Some installations include a local GUI:

.. code-block:: bash

   # Start local Model Composer
   cd tinyml-mlbackend
   python app.py
   # Open http://localhost:5000 in browser

Key Features
------------

**Dataset Management:**

* Upload CSV or ZIP datasets
* Preview data in browser
* Visualize time series plots
* Automatic format validation

**Model Configuration:**

* Visual parameter selection
* Device dropdown menus
* Model size sliders
* Feature extraction presets

**Training:**

* One-click training start
* Real-time loss graphs
* Progress indicators
* Training history

**Results Analysis:**

* Confusion matrices
* ROC curves
* Accuracy metrics
* Model comparison

**Export:**

* Download trained models
* Get compilation artifacts
* Generate CCS projects

Supported Workflows
-------------------

Model Composer supports all Tiny ML Tensorlab workflows:

**Time Series Classification:**

* Upload labeled time series data
* Configure feature extraction
* Train classification models
* Evaluate with confusion matrix

**Time Series Regression:**

* Upload regression datasets
* Set target variables
* Train regression models
* Evaluate with scatter plots

**Anomaly Detection:**

* Upload normal-only data
* Train autoencoder models
* Set detection thresholds
* Evaluate reconstruction error

**Forecasting:**

* Upload sequential data
* Configure forecast horizon
* Train forecasting models
* Evaluate prediction accuracy

System Requirements
-------------------

**For Cloud Version:**

* Modern web browser (Chrome, Firefox, Edge)
* Internet connection
* TI account

**For Local Version:**

* Python 3.10 environment
* Tiny ML Tensorlab installed
* 8 GB RAM minimum
* GPU recommended for large models

Limitations
-----------

Model Composer may have some limitations compared to CLI:

* Fewer advanced configuration options
* May not support all presets
* File size limits for uploads
* Processing time limits

For advanced use cases, consider using the CLI tools directly.

Integration with CLI
--------------------

You can combine Model Composer and CLI workflows:

**Export Configuration:**

Model Composer can export YAML config files:

1. Configure your project in Model Composer
2. Click "Export Configuration"
3. Download the YAML file
4. Use with CLI for further customization

**Import Models:**

CLI-trained models can be analyzed in Model Composer:

1. Upload trained model files
2. View analysis results
3. Export for deployment

Getting Started
---------------

Ready to try Model Composer? See:

* :doc:`getting_started_gui` - Step-by-step tutorial
* :doc:`exporting_models` - Get your model for CCS

For CLI-based workflow, see :doc:`/getting_started/quickstart`.
