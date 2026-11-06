===========
macOS Setup
===========

macOS-specific notes for Tiny ML ModelZoo |release|.

.. note::

   **New in 1.5.0: Apple Silicon (MPS).** Release 1.5.0 added macOS arm64 compatibility
   fixes across training, evaluation and quantization, and training can run on the
   Apple GPU through PyTorch's MPS backend.

Supported Hardware
------------------

Apple Silicon (arm64) is the supported Mac platform: the compiler wheel
(``ti_mcu_nnc``) used by ModelMaker is published for macOS arm64 (macOS 14 or later,
Python 3.10).

Prerequisites
-------------

.. code-block:: bash

   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
   eval "$(/opt/homebrew/bin/brew shellenv)"
   brew install git pyenv

Add pyenv to ``~/.zshrc``:

.. code-block:: bash

   export PYENV_ROOT="$HOME/.pyenv"
   export PATH="$PYENV_ROOT/bin:$PATH"
   eval "$(pyenv init -)"

then:

.. code-block:: bash

   source ~/.zshrc
   pyenv install 3.10.14
   pyenv global 3.10.14

Installation
------------

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel

   # Wheel install
   pip install http://software-dl.ti.com/C2000/esd/mcu_ai/wheel/tinyml_modelzoo-1.5.0-py3-none-any.whl

   # or clone install
   git clone https://github.com/TexasInstruments/tinyml-modelzoo.git
   cd tinyml-modelzoo
   pip install -e .

Run the hello world example from the clone:

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

Set the compiler variable first if you compile (:doc:`environment_variables`).

Using the Apple GPU (MPS)
-------------------------

Training uses the CPU by default (``training.num_gpus: 0``). Set ``num_gpus`` to 1 or
more in the config's ``training`` section: if PyTorch reports MPS as available, training
runs on ``mps``; otherwise it uses ``cuda``.

.. code-block:: yaml

   training:
     num_gpus: 1

MPS does not implement every PyTorch operator. ModelMaker sets
``PYTORCH_ENABLE_MPS_FALLBACK=1`` automatically (unless you already set it), so operators
that are missing on MPS fall back to the CPU. You do not need to export it yourself.

CUDA is not available on macOS.

Notes
-----

* Some operations use OpenMP; if you see a libomp error, run ``brew install libomp``.
* **SSL: CERTIFICATE_VERIFY_FAILED** while downloading datasets: run the "Install
  Certificates.command" shipped with the python.org installer, or ``pip install certifi``.
* **pyenv: command not found**: reload your shell (``source ~/.zshrc``).
* **brew: command not found** on Apple Silicon: run ``eval "$(/opt/homebrew/bin/brew shellenv)"``.

Next Steps
----------

* :doc:`environment_variables`
* :doc:`/getting_started/quickstart`
