=====================
Environment Variables
=====================

ModelMaker finds the TI compiler tools through environment variables. They are needed
only to **compile** a model for a device; training and quantization work without them.

.. warning::

   Set the variable for your target device **before** running an example, otherwise
   the compilation step fails.

How the Paths Are Resolved
--------------------------

Each variable below has a default under ``TOOLS_PATH``. If a variable is unset,
ModelMaker uses ``<TOOLS_PATH>/<default folder name>``.

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Variable
     - Default
     - Purpose
   * - ``TOOLS_PATH``
     - ``~/ti`` (``$HOME/ti``)
     - Parent folder searched for the compilers below
   * - ``C2000_CG_ROOT``
     - ``$TOOLS_PATH/ti-cgt-c2000_25.11.1.LTS``
     - TI C2000 CGT (``bin/cl2000``) for F28x devices
   * - ``CG_TOOL_ROOT``
     - ``$TOOLS_PATH/ti-cgt-c29_2.2.1.LTS``
     - TI C29 Clang (``bin/c29clang``) for F29x devices
   * - ``ARM_LLVM_CGT_PATH``
     - ``$TOOLS_PATH/ti-cgt-armllvm_5.1.1.LTS``
     - TI Arm Clang (``bin/tiarmclang``) for MSPM0, MSPM33C, AM13, AM26x and the CC13xx/CC27xx/CC35xx connectivity devices
   * - ``xWRL6432_ARM_LLVM_CGT_PATH``
     - ``$TOOLS_PATH/ti-cgt-armllvm_5.1.1.LTS``
     - Arm Clang for the xWRL6432 radar device (radar tasks only)

The defaults are in the ``constants.py`` of the ModelMaker modules
(``tinyml_modelmaker/ai_modules/timeseries``, ``audio``, ``radar``). If you install a
compiler in the default folder name under ``~/ti`` you do not need to set anything.
The compiler versions above are the ones 1.5.0 is built against.

Which Variable for Which Device
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Target devices
     - Variable
   * - F28P55, F28P65, F2837, F28004, F28003, other C2000 F28x
     - ``C2000_CG_ROOT``
   * - F29H85, F29P58, F29P32
     - ``CG_TOOL_ROOT``
   * - MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32/34, AM13E2, AM263, AM263P, AM261, CC2755, CC2745, CC1312, CC1314, CC1352, CC1354, CC35X1
     - ``ARM_LLVM_CGT_PATH``
   * - xWRL6432
     - ``xWRL6432_ARM_LLVM_CGT_PATH``

See :doc:`/devices/index` for the full device list.

Setting Variables
-----------------

.. tabs::

   .. tab:: Linux / macOS

      Per session, or add to ``~/.bashrc`` / ``~/.zshrc`` for permanence:

      .. code-block:: bash

         export TOOLS_PATH=$HOME/ti
         export C2000_CG_ROOT=$HOME/ti/ti-cgt-c2000_25.11.1.LTS
         export CG_TOOL_ROOT=$HOME/ti/ti-cgt-c29_2.2.1.LTS
         export ARM_LLVM_CGT_PATH=$HOME/ti/ti-cgt-armllvm_5.1.1.LTS

   .. tab:: Windows (PowerShell)

      .. code-block:: powershell

         $env:C2000_CG_ROOT = "C:\ti\ti-cgt-c2000_25.11.1.LTS"
         # Permanent for the current user:
         [Environment]::SetEnvironmentVariable("C2000_CG_ROOT", "C:\ti\ti-cgt-c2000_25.11.1.LTS", "User")

   .. tab:: Windows (cmd)

      .. code-block:: bat

         set C2000_CG_ROOT=C:\ti\ti-cgt-c2000_25.11.1.LTS

Installing the Compilers
------------------------

* **C2000 CGT**: https://www.ti.com/tool/C2000-CGT (version 25.11.1.LTS)
* **TI Arm Clang**: https://www.ti.com/tool/ARM-CGT (version 5.1.1.LTS)
* **TI C29 Clang**: from TI's compiler download page (version 2.2.1.LTS)
* **Code Composer Studio** also installs compilers under ``<CCS_INSTALL>/tools/compiler/``; point the variables there.

Install them with the vendor installer to any folder, then either set the variable or
install into ``~/ti``.

Verifying
---------

.. code-block:: bash

   $C2000_CG_ROOT/bin/cl2000 --version
   $ARM_LLVM_CGT_PATH/bin/tiarmclang --version

Troubleshooting
---------------

**Compiler not found during compilation**
   Check the variable points at the folder that contains ``bin/`` (``ls $C2000_CG_ROOT/bin``).

**Works in a terminal but not from an IDE**
   The IDE may not inherit your shell variables; set them in its run configuration.

**Other versions**
   The defaults name specific versions. Other versions may work but are not what 1.5.0 was validated with.

Also, ModelMaker sets ``PYTORCH_ENABLE_MPS_FALLBACK=1`` itself on startup (relevant to macOS; see :doc:`macos_setup`).

Next Steps
----------

* :doc:`/getting_started/quickstart`
* :doc:`/deployment/index`
