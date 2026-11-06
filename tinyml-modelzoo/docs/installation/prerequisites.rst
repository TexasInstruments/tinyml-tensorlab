=============
Prerequisites
=============

Check these requirements before installing Tiny ML ModelZoo |release|.

System Requirements
-------------------

.. list-table::
   :widths: 30 70

   * - Linux
     - Ubuntu 20.04 or later recommended (x86_64)
   * - Windows
     - Windows 10/11, native or WSL2
   * - macOS
     - Apple Silicon (arm64) with MPS training support. See :doc:`macos_setup`.
   * - CPU / RAM / disk
     - Multi-core CPU, 8 GB RAM minimum (16 GB recommended), about 10 GB free disk for the install and datasets
   * - GPU
     - Optional. CUDA GPU on Linux/Windows, or MPS on Apple Silicon, to speed up training

Python
------

.. important::

   **Use Python 3.10 if you want to compile models for a device.**

   * ``tinyml_modelzoo`` declares ``requires-python >=3.10,<3.15`` and 1.5.0 added
     Python 3.14 / PyTorch 2.11 compatibility, so training works on Python 3.10 to 3.14.
   * The TI Neural Network Compiler wheel (``ti_mcu_nnc 2.1.2``) that ModelMaker uses for
     compilation is only installed on Python 3.10 (its dependency marker is
     ``python_version < '3.11'``). On Python 3.11 or later, pip skips it silently and the
     compile step is not available.
   * The project README lists "Python 3.10 environment" as the prerequisite.

Pinned dependencies installed by pip (from ``pyproject.toml``): ``torch==2.11.0``,
``PyYAML==6.0.3``, ``numpy==2.2.6`` (Python 3.10) or ``numpy==2.5.2`` (Python 3.11+).
Do not pre-install other versions of torch or numpy in the same environment.

.. tabs::

   .. tab:: Linux / macOS

      pyenv is a convenient way to get Python 3.10:

      .. code-block:: bash

         curl https://pyenv.run | bash       # macOS: brew install pyenv
         pyenv install 3.10.14
         pyenv global 3.10.14
         python --version

   .. tab:: Windows

      Install Python 3.10 from https://www.python.org/downloads/ or use pyenv-win:

      .. code-block:: powershell

         pyenv install 3.10.11
         pyenv global 3.10.11
         python --version

pip and Git
-----------

.. code-block:: bash

   python -m pip install --upgrade pip setuptools wheel
   git --version        # only needed for the clone install

Git is only needed for the clone install (:doc:`developer_installation`). The wheel
install (:doc:`user_installation`) needs only Python and pip.

TI Compiler Tools (for compilation)
-----------------------------------

Required only to compile models for a device. Download from TI and set the matching
environment variable (:doc:`environment_variables`).

* C2000 F28x: TI C2000 Code Generation Tools (https://www.ti.com/tool/C2000-CGT)
* C2000 F29x: TI C29 Clang compiler
* MSPM0, MSPM33C, AM13, AM26x, CC13xx/CC27xx/CC35xx, xWRL6432: TI Arm Clang compiler (https://www.ti.com/tool/ARM-CGT)

To flash and debug on hardware, install Code Composer Studio (https://www.ti.com/tool/CCSTUDIO)
and see :doc:`/DEPLOYING_TO_SDK_EXAMPLE`.

Verification Checklist
----------------------

.. code-block:: bash

   python --version     # 3.10.x for compilation support
   pip --version
   git --version        # clone install only

Then proceed to :doc:`user_installation` or :doc:`developer_installation`.
