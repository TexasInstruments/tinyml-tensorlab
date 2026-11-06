===============
Troubleshooting
===============

Solutions to common problems when training, compiling and deploying with Tiny ML ModelZoo 1.5.0, and answers to
frequently asked questions.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   common_errors
   faq

Quick checks
------------

Most problems fall into one of these groups. Start with the matching entry in :doc:`common_errors`.

.. dropdown:: Is your Python version 3.10?

   Training works on Python 3.10 to 3.14, but compilation needs the TI neural network compiler (``ti_mcu_nnc``),
   which is only installed on Python 3.10. See :ref:`err-python-311`.

   .. code-block:: bash

      python --version

.. dropdown:: Is the model name valid for this release?

   ``training.model_name`` must be one of the models listed in :doc:`/appendix/model_zoo_reference`. See
   :ref:`err-unknown-model`.

.. dropdown:: Compilation fails but training works

   Check that the TI compiler for your device is installed under ``TOOLS_PATH`` (default ``~/ti``) and that the
   model follows the NPU layer rules. See :ref:`err-compiler-tools` and :ref:`err-npu-constraints`.

.. dropdown:: Outputs do not match on the device

   See :ref:`err-golden-vector` and :doc:`/DEPLOYING_TO_SDK_EXAMPLE`.

Getting help
------------

* Read the log files in the run folder, ``./data/projects/<project_name>/run/<date-time>/<model_name>/``. See
  :doc:`/getting_started/understanding_run_output`.
* Check the :doc:`faq`.
* Report issues at https://github.com/TexasInstruments/tinyml-modelzoo/issues, with the YAML you used and the
  log excerpt.
