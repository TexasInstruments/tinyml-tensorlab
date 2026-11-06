============
Installation
============

Tiny ML ModelZoo |release| installs with ``pip``. There are exactly two supported
install paths, and neither needs more than one command to get the whole toolchain
(``tinyml_modelmaker``, ``tinyml_tinyverse`` and ``tinyml_torchmodelopt`` come in
automatically as prebuilt wheels from TI's download server).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   prerequisites
   user_installation
   developer_installation
   windows_setup
   linux_setup
   macos_setup
   environment_variables
   agentic_skill_setup

Which Install Should I Choose?
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Scenario
     - Use This Install
   * - I want to train and compile with the TI-provided models, using my own config files
     - **Wheel install** (:doc:`user_installation`)
   * - I want to use the model zoo as a Python library in my own scripts
     - **Wheel install** (:doc:`user_installation`)
   * - I want the ready-to-run ``examples/`` folder and the ``run_tinyml_modelzoo`` scripts
     - **Clone install** (:doc:`developer_installation`)
   * - I want to add or change a model architecture
     - **Clone install** (:doc:`developer_installation`)
   * - I want to bring my own model (BYOM), without editing ModelZoo (``custom_model_path``)
     - **Either install**
   * - I want to add my model to ModelZoo permanently (BYOM)
     - **Clone install** (:doc:`developer_installation`)

Quick Start
-----------

.. tabs::

   .. tab:: Wheel install (no model changes)

      .. code-block:: bash

         pip install http://software-dl.ti.com/C2000/esd/mcu_ai/wheel/tinyml_modelzoo-1.5.0-py3-none-any.whl

   .. tab:: Clone install (add or change models)

      .. code-block:: bash

         git clone https://github.com/TexasInstruments/tinyml-modelzoo.git
         cd tinyml-modelzoo
         pip install -e .

.. note::

   Starting with 1.5.0 you clone **only** ``tinyml-modelzoo``. You no longer clone
   ``tinyml-tensorlab`` or the separate modelmaker, tinyverse and modeloptimization
   repositories, and there is no ``setup_all`` script. ``tinyml-tensorlab`` is the
   1.4.0 entry point and is now deprecated.

After installing, continue with :doc:`environment_variables` (compiler setup, needed
only for compiling to a device) and then :doc:`/getting_started/quickstart`.
