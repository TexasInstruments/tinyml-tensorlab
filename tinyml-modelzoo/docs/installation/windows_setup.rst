=============
Windows Setup
=============

Windows-specific notes for Tiny ML ModelZoo |release|. Both native Windows and WSL2
work; Linux is the primary development platform.

Native Windows
--------------

1. Install Python 3.10 (see :doc:`prerequisites`).
2. Enable long paths in Git (only for the clone install), once per system:

   .. code-block:: powershell

      git config --global core.longpaths true

3. Create a virtual environment and install, using one of the two install paths:

   .. code-block:: powershell

      python -m venv venv
      .\venv\Scripts\Activate.ps1
      python -m pip install --upgrade pip setuptools wheel

      # Wheel install
      pip install http://software-dl.ti.com/C2000/esd/mcu_ai/wheel/tinyml_modelzoo-1.5.0-py3-none-any.whl

      # or clone install
      git clone https://github.com/TexasInstruments/tinyml-modelzoo.git
      cd tinyml-modelzoo
      pip install -e .

4. Set the compiler variable for your device (:doc:`environment_variables`).
5. Run the hello world example from the cloned folder:

   .. code-block:: bat

      run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml

   or in PowerShell: ``./run_tinyml_modelzoo.ps1 examples/generic_timeseries_classification/config.yaml``.
   With the wheel install, use ``python -m tinyml_modelmaker.run_tinyml_modelmaker config.yaml``
   (see :doc:`user_installation`).

ti_mcu_nnc (the compiler) is published for Windows x86_64 on Python 3.10.

WSL2
----

WSL2 gives a full Linux environment. See the
`WSL documentation <https://learn.microsoft.com/en-us/windows/wsl/about>`_.

.. code-block:: powershell

   wsl --install --web-download --distribution Ubuntu-22.04

Restart when prompted and run the command again to create the UNIX user. Then follow
:doc:`linux_setup` inside the WSL shell.

.. tip::

   Keep project files inside the WSL filesystem (``/home/<user>``), not on ``/mnt/c``,
   for much better performance. Browse WSL files from Windows at ``\\wsl.localhost\Ubuntu-22.04``.

Behind a Proxy (WSL)
~~~~~~~~~~~~~~~~~~~~

If ``apt`` or ``pip`` cannot reach the network, set proxy variables, for example in
``/etc/profile.d/proxy.sh``:

.. code-block:: bash

   export http_proxy="http://your.proxy.address:port"
   export https_proxy="http://your.proxy.address:port"
   export no_proxy="localhost,127.0.0.1"

and ``Acquire::http::proxy`` / ``Acquire::https::proxy`` in ``/etc/apt/apt.conf``. Run
``wsl --shutdown`` afterwards. If DNS fails, set a nameserver in ``/etc/resolv.conf`` and
``generateResolvConf = false`` under ``[network]`` in ``/etc/wsl.conf``.

USB devices (such as LaunchPads) can be attached to WSL with
`usbipd-win <https://github.com/dorssel/usbipd-win>`_.

Common Windows Issues
---------------------

**Path too long**
   Enable Git long paths (above) and clone to a short path such as ``C:\dev``.
   System-wide: set ``LongPathsEnabled`` to 1 in
   ``HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem`` (as Administrator).

**Running scripts is disabled**
   ``Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser``

**Virtual environment activation fails**
   Use ``.\venv\Scripts\activate.bat`` or call ``.\venv\Scripts\python.exe`` directly.

**CUDA GPU**
   Install the NVIDIA driver, then set ``training.num_gpus: 1`` in the config.
   Check with ``python -c "import torch; print(torch.cuda.is_available())"``.

Next Steps
----------

* :doc:`environment_variables`
* :doc:`/getting_started/quickstart`
