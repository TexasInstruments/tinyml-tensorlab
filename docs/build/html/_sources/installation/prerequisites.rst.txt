=============
Prerequisites
=============

Before installing Tiny ML Tensorlab, ensure your system meets these requirements.

System Requirements
-------------------

**Operating System**

.. list-table::
   :widths: 30 70

   * - Linux
     - Ubuntu 20.04 or later (recommended)
   * - Windows
     - Windows 10/11 with WSL2 or native support
   * - macOS
     - macOS 12 or later (limited testing)

**Hardware**

.. list-table::
   :widths: 30 70

   * - CPU
     - Modern multi-core processor (Intel/AMD x86_64 or Apple Silicon)
   * - RAM
     - Minimum 8 GB (16 GB recommended)
   * - Storage
     - 10 GB free space for installation and datasets
   * - GPU
     - Optional but recommended for Neural Architecture Search (NAS)

Software Requirements
---------------------

**Python 3.10**

Tiny ML Tensorlab requires Python 3.10.x specifically. Other versions are not supported.

.. tabs::

   .. tab:: Linux

      We recommend using pyenv:

      .. code-block:: bash

         # Install pyenv
         curl https://pyenv.run | bash

         # Add to ~/.bashrc
         export PYENV_ROOT="$HOME/.pyenv"
         export PATH="$PYENV_ROOT/bin:$PATH"
         eval "$(pyenv init -)"

         # Install Python 3.10
         pyenv install 3.10.14
         pyenv global 3.10.14

         # Verify
         python --version  # Should show Python 3.10.14

   .. tab:: Windows

      We recommend using pyenv-win:

      .. code-block:: powershell

         # Install pyenv-win via PowerShell
         Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"

         # Restart terminal, then install Python
         pyenv install 3.10.14
         pyenv global 3.10.14

         # Verify
         python --version

**Git**

Git is required to clone the repository.

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         sudo apt-get install git

   .. tab:: Windows

      Download from https://git-scm.com/download/win

**pip**

pip should be included with Python. Ensure it's up to date:

.. code-block:: bash

   python -m pip install --upgrade pip

For Compilation (Optional)
--------------------------

To compile models for TI MCUs, you'll need the TI toolchain.
These are only required if you want to compile models for device deployment.

**For C2000 Devices**

* TI C2000 Code Generation Tools (CGT)
* C2000Ware SDK

Download from:

* CGT: https://www.ti.com/tool/C2000-CGT
* C2000Ware: https://www.ti.com/tool/C2000WARE

**For MSPM0 Devices**

* TI Arm Clang Compiler
* MSPM0 SDK

Download from:

* Arm Clang: https://www.ti.com/tool/ARM-CGT
* MSPM0 SDK: https://www.ti.com/tool/MSPM0-SDK

For Device Deployment (Optional)
--------------------------------

To flash and debug models on hardware:

**Code Composer Studio (CCS)**

TI's IDE for MCU development. Download from:
https://www.ti.com/tool/CCSTUDIO

Recommended version: 20.x or later

**TI LaunchPad or EVM**

A development board for your target device.

CUDA (Optional)
---------------

For Neural Architecture Search (NAS), a CUDA-capable GPU significantly
speeds up training:

* NVIDIA GPU with CUDA support
* CUDA Toolkit 11.x or 12.x
* cuDNN library

NAS can run on CPU but will be much slower.

Verification Checklist
----------------------

Before proceeding with installation, verify:

.. code-block:: bash

   # Python version (must be 3.10.x)
   python --version

   # pip is available
   pip --version

   # Git is available
   git --version

If all checks pass, proceed to :doc:`user_installation` or :doc:`developer_installation`.
