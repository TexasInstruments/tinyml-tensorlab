.. Tiny ML Tensorlab documentation master file

================================
Tiny ML Tensorlab User Guide
================================

Welcome to the **Tiny ML Tensorlab** documentation! This comprehensive guide covers
Texas Instruments' end-to-end AI toolchain for developing, training, optimizing,
and deploying machine learning models on resource-constrained microcontrollers.

.. note::
   This documentation is for Tiny ML Tensorlab version |release|.

----

Quick Links
-----------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting_started/index
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      New to Tiny ML Tensorlab? Start here with our quickstart guide and hello world example.

      +++
      :bdg-primary:`Beginner`

   .. grid-item-card:: Examples & Applications
      :link: examples/index
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Browse practical examples including arc fault detection, motor bearing fault classification, and more.

      +++
      :bdg-success:`Ready-to-Run`

   .. grid-item-card:: Edge AI Studio Model Composer
      :link: model_composer/index
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Prefer a GUI? Use our no-code web platform to train and deploy models.

      +++
      :bdg-info:`No-Code`

   .. grid-item-card:: Supported Devices
      :link: devices/index
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Find your target TI MCU and learn about NPU acceleration options.

      +++
      :bdg-warning:`20+ Devices`

----

What is Tiny ML Tensorlab?
--------------------------

Tiny ML Tensorlab is Texas Instruments' complete solution for bringing AI to microcontrollers.
The toolchain enables you to:

* **Train** machine learning models for time series and image classification tasks
* **Optimize** models using quantization (2-bit, 4-bit, 8-bit) for embedded deployment
* **Compile** models to run efficiently on TI MCUs, with optional NPU acceleration
* **Deploy** models using Code Composer Studio (CCS)

Supported Task Types
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Task Type
     - Description
   * - **Time Series Classification**
     - Categorize time-series data into discrete classes (e.g., fault detection, activity recognition)
   * - **Time Series Regression**
     - Predict continuous values from time-series inputs (e.g., torque estimation)
   * - **Time Series Forecasting**
     - Predict future values based on historical patterns (e.g., temperature prediction)
   * - **Anomaly Detection**
     - Identify abnormal patterns using autoencoder-based models (e.g., equipment monitoring)
   * - **Image Classification**
     - Categorize images into classes (e.g., visual inspection, digit recognition)

----

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Contents
   :hidden:

   introduction/index
   installation/index
   getting_started/index
   task_types/index
   byod/index
   devices/index
   examples/index
   features/index
   deployment/index
   model_composer/index
   byom/index
   troubleshooting/index
   appendix/index

**User Guide**
   Start with the :doc:`introduction/index` to understand the toolchain architecture,
   then follow the :doc:`installation/index` guide to set up your environment.

**Task Types**
   Learn about the different :doc:`task_types/index` supported and choose the right one for your application.

**Working with Data**
   The :doc:`byod/index` (Bring Your Own Data) section explains dataset formats and preparation.

**Target Devices**
   Browse :doc:`devices/index` to find specifications and capabilities for 20+ TI MCUs.

**Examples & Applications**
   The :doc:`examples/index` section provides ready-to-run configurations for common applications.

**Advanced Features**
   Explore :doc:`features/index` like Neural Architecture Search, quantization, and analysis tools.

**Deployment**
   The :doc:`deployment/index` section covers CCS integration and running models on devices.

**Edge AI Studio**
   Prefer a GUI? See :doc:`model_composer/index` for our no-code web platform.

**Extending the Toolchain**
   The :doc:`byom/index` (Bring Your Own Model) section covers adding custom models.

----

Additional Resources
--------------------

* `TI Neural Network Compiler Documentation <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_
* `Edge AI Studio Model Composer <https://dev.ti.com/modelcomposer/>`_
* `GitHub Repository <https://github.com/TexasInstruments/tinyml-tensorlab>`_
* `TI E2E Support Forum <https://e2e.ti.com/support/processors/>`_

----

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
