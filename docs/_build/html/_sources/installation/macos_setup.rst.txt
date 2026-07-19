===========
macOS Setup
===========

This guide provides macOS-specific instructions for installing and using
Tiny ML Tensorlab on macOS 12 (Monterey) or later.

.. note::
   Both Intel (x86_64) and Apple Silicon (M1/M2/M3) Macs are supported.
   All commands work on both architectures.

Prerequisites
-------------

**Step 1: Install Homebrew**

If you don't have Homebrew installed:

.. code-block:: bash

   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Follow the prompts. Apple Silicon users must also add Homebrew to PATH:

.. code-block:: bash

   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
   eval "$(/opt/homebrew/bin/brew shellenv)"

**Step 2: Install system dependencies**

.. code-block:: bash

   brew install git pyenv

**Step 3: Configure pyenv**

Add to ``~/.zshrc`` (or ``~/.bash_profile`` if using bash):

.. code-block:: bash

   export PYENV_ROOT="$HOME/.pyenv"
   export PATH="$PYENV_ROOT/bin:$PATH"
   eval "$(pyenv init -)"

Reload the shell:

.. code-block:: bash

   source ~/.zshrc

**Step 4: Install Python 3.10**

.. code-block:: bash

   pyenv install 3.10.14
   pyenv global 3.10.14

   # Verify
   python --version  # Should show Python 3.10.14

Installation
------------

**Step 1: Clone the repository**

.. code-block:: bash

   git clone https://github.com/TexasInstruments/tinyml-tensorlab.git
   cd tinyml-tensorlab

**Step 2: Create a virtual environment**

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate

**Step 3: Upgrade pip and build tools**

.. code-block:: bash

   python -m pip install --upgrade pip setuptools wheel

**Step 4: Install Tiny ML Tensorlab components**

.. code-block:: bash

   pip install -e tinyml-modelmaker
   pip install -e tinyml-tinyverse
   pip install -e tinyml-modeloptimization/torchmodelopt
   pip install -e tinyml-modelzoo

Alternatively, use the setup script:

.. code-block:: bash

   cd tinyml-modelmaker
   ./setup_all.sh

Environment Variables for Compilation
--------------------------------------

Environment variables are only needed if compiling models for TI MCU deployment.
Training and testing work without them.

For MSPM0 and Connectivity devices:

.. code-block:: bash

   export ARM_LLVM_CGT_PATH="$HOME/ti/arm-clang/4.0.0.LTS"

Add to ``~/.zshrc`` for persistence.

See :doc:`environment_variables` for complete device-specific instructions.

Running Examples
----------------

.. code-block:: bash

   # Activate environment
   source venv/bin/activate

   # Run an example
   cd tinyml-modelzoo
   ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

Verifying Installation
----------------------

.. code-block:: python

   import tinyml_modelmaker
   import tinyml_tinyverse
   import tinyml_torchmodelopt
   import tinyml_modelzoo

   print(f"ModelMaker: {tinyml_modelmaker.__version__}")
   print(f"Tinyverse: {tinyml_tinyverse.__version__}")
   print(f"ModelOpt: {tinyml_torchmodelopt.__version__}")
   print(f"ModelZoo: {tinyml_modelzoo.__version__}")

macOS-Specific Notes
--------------------

**Apple Silicon (M1/M2/M3)**

PyTorch works natively on Apple Silicon via MPS backend. CPU training
is used by default; GPU acceleration via MPS is not currently enabled
in the training pipeline but does not affect correctness.

**GPU / CUDA**

CUDA is not available on macOS. Neural Architecture Search (NAS) runs
on CPU, which is slower but fully functional.

**OpenMP**

Some operations use OpenMP. Install the runtime if needed:

.. code-block:: bash

   brew install libomp

Troubleshooting
---------------

**"SSL: CERTIFICATE_VERIFY_FAILED"**

Run the Python certificate installer:

.. code-block:: bash

   /Applications/Python\ 3.10/Install\ Certificates.command

Or via pip:

.. code-block:: bash

   pip install certifi
   /usr/local/opt/python@3.10/bin/python3.10 -c "import certifi; print(certifi.where())"

**"command not found: brew"**

Homebrew is not on PATH. For Apple Silicon, run:

.. code-block:: bash

   eval "$(/opt/homebrew/bin/brew shellenv)"

Then add this line to ``~/.zshrc``.

**pyenv: command not found after install**

Shell config not reloaded. Run ``source ~/.zshrc`` or open a new terminal.

Next Steps
----------

* :doc:`/getting_started/quickstart` — Train your first model
* :doc:`environment_variables` — Set up compilation toolchains
* :doc:`/examples/index` — Browse available examples
