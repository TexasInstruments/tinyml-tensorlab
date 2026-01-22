===============
Troubleshooting
===============

This section helps you resolve common issues and find answers to
frequently asked questions.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   common_errors
   faq

Quick Fixes
-----------

**Installation Issues**

.. dropdown:: "ModuleNotFoundError" when importing

   Ensure you've installed in the correct order:

   .. code-block:: bash

      cd tinyml-modelmaker && pip install -e .
      cd ../tinyml-tinyverse && pip install -e .
      cd ../tinyml-modeloptimization/torchmodelopt && pip install -e .
      cd ../../tinyml-modelzoo && pip install -e .

.. dropdown:: Python version errors

   Tiny ML Tensorlab requires Python 3.10.x specifically:

   .. code-block:: bash

      python --version  # Should show 3.10.x

   Use pyenv to manage Python versions.

**Training Issues**

.. dropdown:: CUDA out of memory

   Reduce batch size in your config:

   .. code-block:: yaml

      training:
        batch_size: 128  # Try smaller values

.. dropdown:: Training accuracy is poor

   * Try different feature extraction presets
   * Use the Goodness of Fit test to evaluate your dataset
   * Ensure proper data normalization
   * Try a larger model

**Compilation Issues**

.. dropdown:: Compilation fails with NPU errors

   Ensure your model follows NPU constraints:

   * Channels must be multiples of 4
   * Kernel sizes within limits
   * See :doc:`/devices/npu_guidelines`

Getting Help
------------

* **GitHub Issues**: https://github.com/TexasInstruments/tinyml-tensorlab/issues
* **TI E2E Forum**: https://e2e.ti.com/support/processors/
