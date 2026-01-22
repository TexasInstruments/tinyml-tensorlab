========
Appendix
========

Reference materials and supplementary information.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   config_reference
   model_zoo_reference
   changelog

Quick Reference
---------------

**Running Examples**

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/<example>/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\<example>\config.yaml

**Environment Variables**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``C2000_CG_ROOT``
     - Path to TI C2000 Codegen Tools
   * - ``C2000WARE_ROOT``
     - Path to C2000Ware SDK
   * - ``ARM_LLVM_CGT_PATH``
     - Path to TI Arm Clang Compiler

**Version Information**

* Tiny ML Tensorlab: 1.2.0
* Python: 3.10.x required
* PyTorch: 2.7.1
* TI Neural Network Compiler: 2.1.0

External Links
--------------

* `TI Neural Network Compiler Documentation <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_
* `Edge AI Studio Model Composer <https://dev.ti.com/modelcomposer/>`_
* `C2000Ware SDK <https://www.ti.com/tool/C2000WARE>`_
* `Digital Power SDK <https://www.ti.com/tool/C2000WARE-DIGITALPOWER-SDK>`_
* `Motor Control SDK <https://www.ti.com/tool/C2000WARE-MOTORCONTROL-SDK>`_
