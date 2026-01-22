=====================
Environment Variables
=====================

This guide explains the environment variables needed for model compilation
targeting TI microcontrollers.

.. note::
   Environment variables are only required if you want to compile models
   for device deployment. Training and testing work without them.

Required Variables by Device Family
-----------------------------------

C2000 Devices (F28P55, F28P65, F2837, etc.)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``C2000_CG_ROOT``
     - Path to TI C2000 Code Generation Tools installation
   * - ``C2000WARE_ROOT``
     - Path to C2000Ware SDK installation

**Example Values (Linux):**

.. code-block:: bash

   export C2000_CG_ROOT="$HOME/ti/ccs/tools/compiler/ti-cgt-c2000_22.6.1.LTS"
   export C2000WARE_ROOT="$HOME/ti/c2000/C2000Ware_5_03_00_00"

**Example Values (Windows):**

.. code-block:: powershell

   $env:C2000_CG_ROOT = "C:\ti\ccs\tools\compiler\ti-cgt-c2000_22.6.1.LTS"
   $env:C2000WARE_ROOT = "C:\ti\c2000\C2000Ware_5_03_00_00"

MSPM0 Devices (MSPM0G3507, MSPM0G5187, etc.)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``ARM_LLVM_CGT_PATH``
     - Path to TI Arm Clang Compiler installation
   * - ``MSPM0_SDK_ROOT``
     - Path to MSPM0 SDK installation (optional)

**Example Values (Linux):**

.. code-block:: bash

   export ARM_LLVM_CGT_PATH="$HOME/ti/arm-clang/4.0.0.LTS"
   export MSPM0_SDK_ROOT="$HOME/ti/mspm0_sdk_2_01_00_03"

AM26x Devices (AM263, AM261, etc.)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``ARM_LLVM_CGT_PATH``
     - Path to TI Arm Clang Compiler installation
   * - ``MCU_PLUS_SDK_ROOT``
     - Path to MCU+ SDK installation (optional)

Setting Environment Variables
-----------------------------

**Linux (Permanent)**

Add to ``~/.bashrc`` or ``~/.profile``:

.. code-block:: bash

   # TI Toolchain Paths
   export C2000_CG_ROOT="$HOME/ti/ccs/tools/compiler/ti-cgt-c2000_22.6.1.LTS"
   export C2000WARE_ROOT="$HOME/ti/c2000/C2000Ware_5_03_00_00"
   export ARM_LLVM_CGT_PATH="$HOME/ti/arm-clang/4.0.0.LTS"

   # Add to PATH
   export PATH="$C2000_CG_ROOT/bin:$PATH"

Then reload:

.. code-block:: bash

   source ~/.bashrc

**Windows (Permanent)**

1. Open System Properties → Advanced → Environment Variables
2. Under "User variables", click "New"
3. Add each variable name and value

Or use PowerShell:

.. code-block:: powershell

   # Set for current user (permanent)
   [Environment]::SetEnvironmentVariable("C2000_CG_ROOT", "C:\ti\ccs\tools\compiler\ti-cgt-c2000_22.6.1.LTS", "User")
   [Environment]::SetEnvironmentVariable("C2000WARE_ROOT", "C:\ti\c2000\C2000Ware_5_03_00_00", "User")

**Per-Session (Temporary)**

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         export C2000_CG_ROOT="/path/to/compiler"

   .. tab:: Windows (CMD)

      .. code-block:: batch

         set C2000_CG_ROOT=C:\path\to\compiler

   .. tab:: Windows (PowerShell)

      .. code-block:: powershell

         $env:C2000_CG_ROOT = "C:\path\to\compiler"

Installing TI Tools
-------------------

**C2000 Code Generation Tools**

1. Download from https://www.ti.com/tool/C2000-CGT
2. Run installer and note installation path
3. Set ``C2000_CG_ROOT`` to the installation directory

**C2000Ware SDK**

1. Download from https://www.ti.com/tool/C2000WARE
2. Run installer and note installation path
3. Set ``C2000WARE_ROOT`` to the installation directory

**TI Arm Clang Compiler**

1. Download from https://www.ti.com/tool/ARM-CGT
2. Extract to your preferred location
3. Set ``ARM_LLVM_CGT_PATH`` to the extracted directory

**Code Composer Studio (CCS)**

When you install CCS, it can automatically install compilers:

1. Download CCS from https://www.ti.com/tool/CCSTUDIO
2. During installation, select the device families you need
3. CCS installs compilers to ``<CCS_INSTALL>/tools/compiler/``

Verifying Configuration
-----------------------

Check that environment variables are set:

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         echo $C2000_CG_ROOT
         echo $C2000WARE_ROOT
         echo $ARM_LLVM_CGT_PATH

         # Verify compiler is accessible
         $C2000_CG_ROOT/bin/cl2000 --version

   .. tab:: Windows

      .. code-block:: powershell

         echo $env:C2000_CG_ROOT
         echo $env:C2000WARE_ROOT

         # Verify compiler
         & "$env:C2000_CG_ROOT\bin\cl2000" --version

Troubleshooting
---------------

**"Compiler not found" during compilation**

Check the environment variable points to the correct directory:

.. code-block:: bash

   ls $C2000_CG_ROOT/bin/  # Should contain cl2000

**"SDK not found" errors**

Verify the SDK path contains expected subdirectories:

.. code-block:: bash

   ls $C2000WARE_ROOT/  # Should contain device/, libraries/, etc.

**Compilation works in terminal but not in IDE**

IDE may not inherit shell environment variables. Set them in the IDE's
run configuration or use absolute paths in your config.

**Different compiler versions**

The toolchain is tested with specific compiler versions. Using different
versions may cause compatibility issues. Recommended versions:

* C2000 CGT: 22.6.x
* Arm Clang: 4.0.x

Next Steps
----------

* :doc:`/getting_started/quickstart` - Train and compile your first model
* :doc:`/deployment/ccs_integration` - Deploy to your device
