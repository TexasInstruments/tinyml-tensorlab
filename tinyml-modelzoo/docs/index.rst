.. Tiny ML ModelZoo documentation master file

================================
Tiny ML ModelZoo User Guide
================================

Welcome to the **Tiny ML ModelZoo** documentation! This guide covers
Texas Instruments' end-to-end AI toolchain for developing, training, optimizing,
and deploying machine learning models on resource-constrained microcontrollers.

.. note::
   This documentation is for Tiny ML ModelZoo version |release|. Starting with 1.5.0,
   ``tinyml-modelzoo`` is the standalone, pip-installable entry point to TI's MCU AI
   flow. It replaces cloning the full ``tinyml-tensorlab`` set of repositories.
   The 1.4.0 documentation is still available as the
   `Tiny ML Tensorlab User Guide <https://software-dl.ti.com/C2000/esd/mcu_ai/user_guide/index.html>`_.

----

Quick Links
-----------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Tiny ML ModelZoo on GitHub
      :link: https://github.com/TexasInstruments/tinyml-modelzoo
      :link-type: url
      :class-card: sd-border-0 sd-shadow-sm

      Models, examples, and configurations: your starting point for MCU AI development.

      +++
      :bdg-dark:`GitHub`

   .. grid-item-card:: Installation
      :link: installation/index
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Install with one ``pip install`` of the wheel, or clone ``tinyml-modelzoo`` to add your own models.

      +++
      :bdg-primary:`Start here`

   .. grid-item-card:: Getting Started
      :link: getting_started/index
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Run the hello world example, understand the config YAML, and read the run output.

      +++
      :bdg-primary:`Beginner`

   .. grid-item-card:: Examples & Applications
      :link: examples/index
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Ready-to-run examples: arc fault, motor bearing fault, wake word, radar and more.

      +++
      :bdg-success:`Ready-to-Run`

   .. grid-item-card:: Supported Devices
      :link: devices/index
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Find your target TI MCU and learn about NPU acceleration options.

      +++
      :bdg-warning:`30+ Devices`

   .. grid-item-card:: Deploy to an SDK Example
      :link: DEPLOYING_TO_SDK_EXAMPLE
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Copy the compiled model and golden vectors into an SDK example and verify on the board.

      +++
      :bdg-info:`On-Device`

----

What is Tiny ML ModelZoo?
-------------------------

Tiny ML ModelZoo is Texas Instruments' complete solution for bringing AI to microcontrollers.
The toolchain enables you to:

* **Train** machine learning models for time series, audio, image and radar tasks
* **Optimize** models using quantization (2-bit, 4-bit, 8-bit, mixed precision) for embedded deployment
* **Compile** models to run efficiently on TI MCUs, with optional NPU acceleration
* **Deploy** models using Code Composer Studio (CCS) and the device SDK examples

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
   Browse :doc:`devices/index` to find specifications and capabilities of the supported TI MCUs.

**Examples & Applications**
   The :doc:`examples/index` section provides ready-to-run configurations for common applications.

**Advanced Features**
   Explore :doc:`features/index` such as feature extraction, quantization, Neural Architecture Search and on-device training.

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
* `GitHub Repository <https://github.com/TexasInstruments/tinyml-modelzoo>`_
* `TI E2E Support Forum <https://e2e.ti.com/support/processors/>`_

----

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
