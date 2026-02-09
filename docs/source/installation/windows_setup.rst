=============
Windows Setup
=============

This guide provides Windows-specific instructions for installing and using
Tiny ML Tensorlab.

.. note::
   While Tiny ML Tensorlab works on Windows, Linux is the primary development
   platform. Some features may have better support on Linux.

Option 1: Native Windows Installation
-------------------------------------

**Step 1: Install Python 3.10**

Using pyenv-win (Recommended):

.. code-block:: powershell

   # Run PowerShell as Administrator
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

   # Install pyenv-win
   Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"
   & "./install-pyenv-win.ps1"

   # Restart PowerShell, then:
   pyenv install 3.10.14
   pyenv global 3.10.14

   # Verify
   python --version

**Step 2: Install Git**

Download and install Git from https://git-scm.com/download/win

During installation:

* Select "Git from the command line and also from 3rd-party software"
* Select "Checkout as-is, commit Unix-style line endings"

**Step 3: Clone and Install**

.. code-block:: powershell

   # Clone repository
   git clone https://github.com/TexasInstruments/tinyml-tensorlab.git
   cd tinyml-tensorlab

   # Create virtual environment
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Install components
   cd tinyml-modelmaker
   pip install -e .

   cd ..\tinyml-tinyverse
   pip install -e .

   cd ..\tinyml-modeloptimization\torchmodelopt
   pip install -e .

   cd ..\..\tinyml-modelzoo
   pip install -e .

**Step 4: Run Example**

.. code-block:: powershell

   cd tinyml-modelzoo
   run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml

Option 2: WSL2 (Recommended for Full Compatibility)
---------------------------------------------------

Windows Subsystem for Linux provides better compatibility with the toolchain.

**Step 1: Enable WSL2**

Open PowerShell as Administrator:

.. code-block:: powershell

   wsl --install

Restart your computer when prompted.

**Step 2: Install Ubuntu**

Open Microsoft Store and install "Ubuntu 22.04 LTS".

Launch Ubuntu and set up your username/password.

**Step 3: Install in Ubuntu**

.. code-block:: bash

   # Update packages
   sudo apt update && sudo apt upgrade -y

   # Install dependencies
   sudo apt install -y build-essential git curl

   # Install pyenv
   curl https://pyenv.run | bash

   # Add to ~/.bashrc
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
   echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init -)"' >> ~/.bashrc
   source ~/.bashrc

   # Install Python 3.10
   pyenv install 3.10.14
   pyenv global 3.10.14

   # Clone and install
   git clone https://github.com/TexasInstruments/tinyml-tensorlab.git
   cd tinyml-tensorlab/tinyml-modelmaker
   ./setup_all.sh

**Step 4: Run Examples**

.. code-block:: bash

   cd ~/tinyml-tensorlab/tinyml-modelzoo
   ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

Path Configuration
------------------

**Long Path Support (Native Windows)**

Enable long path support for deep directory structures:

.. code-block:: powershell

   # Run as Administrator
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

**Environment Variables**

Set up paths for TI tools (if compiling for devices):

.. code-block:: powershell

   # Add to your profile or set in System Properties
   $env:C2000_CG_ROOT = "C:\ti\ccs\tools\compiler\ti-cgt-c2000_22.6.1.LTS"
   $env:C2000WARE_ROOT = "C:\ti\c2000\C2000Ware_5_03_00_00"

Common Windows Issues
---------------------

**PowerShell Execution Policy**

If you get "running scripts is disabled":

.. code-block:: powershell

   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

**Path Too Long Errors**

Clone to a short path (e.g., ``C:\dev\tensorlab``) or enable long paths.

**Line Ending Issues**

Configure Git to handle line endings:

.. code-block:: powershell

   git config --global core.autocrlf true

**Virtual Environment Activation**

If ``venv\Scripts\Activate.ps1`` fails:

.. code-block:: powershell

   # Alternative activation
   .\venv\Scripts\activate.bat

   # Or run Python directly
   .\venv\Scripts\python.exe -m tinyml_modelmaker ...

GPU Support on Windows
----------------------

For CUDA support (useful for NAS):

1. Install NVIDIA drivers from https://www.nvidia.com/drivers
2. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
3. Verify with:

.. code-block:: powershell

   python -c "import torch; print(torch.cuda.is_available())"

WSL2 vs Native Windows Comparison
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Native Windows
     - WSL2
   * - Setup Complexity
     - Moderate
     - Higher initially
   * - Shell Scripts
     - Requires .bat files
     - Full bash support
   * - Compilation
     - Full support
     - Full support
   * - GPU Support
     - Native CUDA
     - WSL2 CUDA (newer)
   * - CCS Integration
     - Direct
     - Requires file sharing
   * - Recommended For
     - Quick start
     - Full development

Next Steps
----------

* :doc:`environment_variables` - Configure TI compiler paths
* :doc:`/getting_started/quickstart` - Train your first model
