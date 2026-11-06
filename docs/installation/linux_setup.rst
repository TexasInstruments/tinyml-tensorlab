===========
Linux Setup
===========

Linux-specific notes for Tiny ML ModelZoo |release|. Ubuntu 20.04 or later (x86_64) is
the primary development platform.

System Preparation
------------------

.. code-block:: bash

   sudo apt update && sudo apt upgrade -y
   sudo apt install -y build-essential git curl wget

Install Python 3.10
-------------------

.. code-block:: bash

   curl https://pyenv.run | bash
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
   echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init -)"' >> ~/.bashrc
   source ~/.bashrc
   pyenv install 3.10.14
   pyenv global 3.10.14

If ``pyenv install`` fails, install the build libraries it lists (for Ubuntu typically
``libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev``).
You can also use ``python3.10`` from your distribution or the deadsnakes PPA.

Installation
------------

Create a virtual environment, then choose one install path (details in
:doc:`user_installation` and :doc:`developer_installation`):

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel

   # Wheel install (no model changes)
   pip install http://software-dl.ti.com/C2000/esd/mcu_ai/wheel/tinyml_modelzoo-1.5.0-py3-none-any.whl

   # or clone install (add/change models, get examples and run scripts)
   git clone https://github.com/TexasInstruments/tinyml-modelzoo.git
   cd tinyml-modelzoo
   pip install -e .

Verification
------------

.. code-block:: bash

   python -c "import tinyml_modelmaker, tinyml_tinyverse, tinyml_torchmodelopt, tinyml_modelzoo; print('OK')"

Then configure the compiler (:doc:`environment_variables`) and run:

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

GPU Setup (Optional)
--------------------

Install the NVIDIA driver for your card (for example ``sudo apt install nvidia-driver-<version>``
or the driver from NVIDIA). The pip-installed ``torch==2.11.0`` ships its own CUDA
runtime, so a separate CUDA Toolkit install is normally not needed. Verify:

.. code-block:: bash

   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"

Enable it in the config with ``training.num_gpus: 1``. The default in the example
configs is ``num_gpus: 0`` (CPU).

Shell Configuration
-------------------

Add your compiler variables to ``~/.bashrc`` so they persist (see
:doc:`environment_variables`).

Troubleshooting
---------------

**Permission denied on run_tinyml_modelzoo.sh**
   ``chmod +x run_tinyml_modelzoo.sh``

**python: command not found / wrong version**
   Activate the virtual environment and check ``python --version``.

**Slow or interrupted downloads**
   The toolchain wheels are large (PyTorch). Re-run pip; it resumes from its cache.
   ``pip install --verbose ...`` shows progress.

Next Steps
----------

* :doc:`environment_variables`
* :doc:`/getting_started/quickstart`
