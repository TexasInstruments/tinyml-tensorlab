===========
Linux Setup
===========

This guide provides Linux-specific instructions for installing Tiny ML Tensorlab.
Linux is the primary development platform for the toolchain.

System Preparation
------------------

**Ubuntu/Debian**

.. code-block:: bash

   # Update system
   sudo apt update && sudo apt upgrade -y

   # Install essential packages
   sudo apt install -y build-essential git curl wget \
       libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
       libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
       libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

**Fedora/RHEL**

.. code-block:: bash

   sudo dnf install -y git curl wget make gcc gcc-c++ \
       openssl-devel bzip2-devel libffi-devel zlib-devel \
       readline-devel sqlite-devel ncurses-devel xz-devel

Installing Python 3.10
----------------------

We recommend using pyenv for Python version management:

.. code-block:: bash

   # Install pyenv
   curl https://pyenv.run | bash

   # Add to ~/.bashrc (or ~/.zshrc for zsh users)
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
   echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init -)"' >> ~/.bashrc

   # Reload shell
   source ~/.bashrc

   # Install Python 3.10
   pyenv install 3.10.14

   # Set as global default
   pyenv global 3.10.14

   # Verify
   python --version  # Should show Python 3.10.14

Installation
------------

**Option A: Automated Setup (Recommended)**

.. code-block:: bash

   # Clone repository
   git clone https://github.com/TexasInstruments/tinyml-tensorlab.git
   cd tinyml-tensorlab/tinyml-modelmaker

   # Run setup script
   ./setup_all.sh

The script handles:

* Virtual environment creation
* Dependency installation
* Editable installation of all components

**Option B: Manual Installation**

.. code-block:: bash

   # Clone repository
   git clone https://github.com/TexasInstruments/tinyml-tensorlab.git
   cd tinyml-tensorlab

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate

   # Upgrade pip
   pip install --upgrade pip

   # Install components
   cd tinyml-modelmaker && pip install -e .
   cd ../tinyml-tinyverse && pip install -e .
   cd ../tinyml-modeloptimization/torchmodelopt && pip install -e .
   cd ../../tinyml-modelzoo && pip install -e .

Verification
------------

.. code-block:: bash

   # Activate environment
   source venv/bin/activate

   # Verify Python packages
   python -c "import tinyml_modelmaker; print('OK')"
   python -c "import tinyml_tinyverse; print('OK')"
   python -c "import tinyml_modelzoo; print('OK')"
   python -c "import tinyml_torchmodelopt; print('OK')"

   # Run hello world example
   cd tinyml-modelzoo
   ./run_tinyml_modelzoo.sh examples/hello_world/config.yaml

GPU Setup (Optional)
--------------------

For NVIDIA GPU support (useful for NAS):

**Install NVIDIA Drivers**

.. code-block:: bash

   # Ubuntu
   sudo apt install -y nvidia-driver-535  # Or latest version

   # Verify
   nvidia-smi

**Install CUDA Toolkit**

Download from https://developer.nvidia.com/cuda-downloads

Or use conda/pip:

.. code-block:: bash

   # PyTorch with CUDA is installed automatically
   # Verify CUDA availability
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

Shell Configuration
-------------------

**Alias for Quick Access**

Add to ``~/.bashrc``:

.. code-block:: bash

   # Tiny ML Tensorlab aliases
   alias tensorlab='cd ~/tinyml-tensorlab && source venv/bin/activate'
   alias run-tensorlab='./run_tinyml_modelzoo.sh'

**Auto-activate Environment**

For automatic activation when entering the directory:

.. code-block:: bash

   # Add to ~/.bashrc
   function cd() {
       builtin cd "$@"
       if [ -f "venv/bin/activate" ]; then
           source venv/bin/activate
       fi
   }

Permission Issues
-----------------

**Make Scripts Executable**

.. code-block:: bash

   chmod +x tinyml-modelzoo/run_tinyml_modelzoo.sh
   chmod +x tinyml-modelmaker/run_tinyml_modelmaker.sh
   chmod +x tinyml-modelmaker/setup_all.sh

**Fix Permission Denied Errors**

If you encounter permission errors:

.. code-block:: bash

   # Check file ownership
   ls -la

   # Fix ownership if needed
   sudo chown -R $USER:$USER ~/tinyml-tensorlab

Multiple Python Versions
------------------------

If you need to manage multiple Python projects:

.. code-block:: bash

   # Set local version for this project only
   cd ~/tinyml-tensorlab
   pyenv local 3.10.14

   # This creates a .python-version file
   # Python 3.10 will be used automatically in this directory

System Service (Optional)
-------------------------

For running training jobs as a background service:

Create ``/etc/systemd/system/tensorlab-training.service``:

.. code-block:: ini

   [Unit]
   Description=Tiny ML Training Job
   After=network.target

   [Service]
   Type=simple
   User=your_username
   WorkingDirectory=/home/your_username/tinyml-tensorlab/tinyml-modelzoo
   ExecStart=/home/your_username/tinyml-tensorlab/venv/bin/python -m tinyml_modelmaker config.yaml
   Restart=no

   [Install]
   WantedBy=multi-user.target

Troubleshooting
---------------

**"pyenv: command not found"**

Ensure pyenv is in your PATH:

.. code-block:: bash

   source ~/.bashrc
   # Or log out and log back in

**CUDA out of memory**

Reduce batch size or use CPU:

.. code-block:: yaml

   training:
     batch_size: 64  # Reduce from 256
     num_gpus: 0     # Use CPU

**Slow training without GPU**

This is expected. For faster training:

* Use a smaller model
* Reduce training epochs
* Use Google Colab with free GPU

Next Steps
----------

* :doc:`environment_variables` - Configure TI compiler paths
* :doc:`/getting_started/quickstart` - Train your first model
