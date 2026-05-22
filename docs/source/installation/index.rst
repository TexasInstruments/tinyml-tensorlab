============
Installation
============

This section guides you through installing Tiny ML Tensorlab on your system.
Choose the installation path that best fits your needs.

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

Which Install Should I Choose?
-------------------------------

Before diving into the prerequisites or detailed guides, use this table to pick
the right installation type for your situation:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Scenario
     - Use This Install
   * - I want to train models on my own data using TI-provided models
     - **User Install** (:doc:`user_installation`)
   * - I want to run the provided examples without modifying source code
     - **User Install** (:doc:`user_installation`)
   * - I want to use Tensorlab as a library in my own Python scripts
     - **User Install** (:doc:`user_installation`)
   * - I want to add custom model architectures to the model zoo
     - **Developer Install** (:doc:`developer_installation`)
   * - I want to modify or debug the training pipeline itself
     - **Developer Install** (:doc:`developer_installation`)
   * - I want to contribute to the Tensorlab codebase
     - **Developer Install** (:doc:`developer_installation`)
   * - I want to integrate new feature extraction transforms
     - **Developer Install** (:doc:`developer_installation`)

**In summary:**

* **User Install** -- Install and use. No source code changes. Best for the
  majority of users who want to train models on their own data.
* **Developer Install** -- Clone and edit. Gives full access to source code.
  Required only if you need to extend or modify the framework itself.

Quick Start
-----------

**For Users (Quick Install)**

If you want to use Tiny ML Tensorlab without modifying the source code:

.. code-block:: bash

   pip install git+https://github.com/TexasInstruments/tinyml-tensorlab.git@main#subdirectory=tinyml-modelmaker

**For Developers (Full Install)**

If you want to customize models, add features, or contribute:

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         git clone https://github.com/TexasInstruments/tinyml-tensorlab.git
         cd tinyml-tensorlab/tinyml-modelmaker
         ./setup_all.sh

   .. tab:: Windows

      .. code-block:: powershell

         git clone https://github.com/TexasInstruments/tinyml-tensorlab.git
         python -m pip install --editable tinyml-tensorlab\tinyml-modelmaker

   .. tab:: macOS

      .. code-block:: bash

         git clone https://github.com/TexasInstruments/tinyml-tensorlab.git
         cd tinyml-tensorlab/tinyml-modelmaker
         ./setup_all.sh

See the detailed guides in this section for complete instructions.
