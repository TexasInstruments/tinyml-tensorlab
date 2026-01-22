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
   environment_variables

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
         cd tinyml-tensorlab\tinyml-modelmaker
         python -m pip install --editable .

See the detailed guides in this section for complete instructions.
