===
FAQ
===

Frequently asked questions about Tiny ML ModelZoo 1.5.0.

General
-------

.. dropdown:: What is Tiny ML ModelZoo?

   The standalone, pip-installable entry point to TI's MCU AI flow: model zoo, example applications, and
   (through prebuilt wheels) ModelMaker, TinyVerse and ModelOptimization. In 1.4.0 and earlier this was the
   separate ``tinyml-tensorlab`` project, which is now deprecated as the entry point. See :doc:`/introduction/index`.

.. dropdown:: Which devices are supported?

   33 MCU and wireless devices in 1.5.0 (see the release notes in :doc:`/appendix/changelog` and
   :doc:`/devices/index`). Devices with a hardware NPU are F28P55, AM13E2 and MSPM0G5187.

.. dropdown:: Do I need a GPU?

   No. CPU training works for the small models in the zoo. With CUDA available, ``torch.compile`` and AMP are enabled
   automatically unless you set ``compile_model`` / ``native_amp`` yourself or ``num_gpus: 0``. On Apple Silicon the
   MPS backend is supported.

.. dropdown:: Which operating systems are supported?

   Linux, Windows and macOS. See :doc:`/installation/index`.

Installation
------------

.. dropdown:: Do I need to clone several repositories?

   No. Clone only ``tinyml-modelzoo`` and run ``pip install -e .``, or install the wheel
   if you do not change models. The other components come as prebuilt wheels. See :doc:`/installation/index`.

.. dropdown:: Which Python version?

   Python 3.10 for the full flow. Newer versions (up to 3.14) can train, but the TI compiler ``ti_mcu_nnc`` is only
   available for 3.10. See :ref:`err-python-311`.

.. dropdown:: Do I need the TI compilers to train?

   No. They are only needed for the compilation step. Set ``TOOLS_PATH`` (default ``~/ti``) when you compile. See
   :ref:`err-compiler-tools`.

.. dropdown:: I installed the wheel. Where are the examples?

   The examples are in the Git repository. See :ref:`err-no-examples`.

Models and training
-------------------

.. dropdown:: Which model should I choose?

   Choose a model that lists your target device in :doc:`/appendix/model_zoo_reference`. For NPU devices prefer
   models ending in ``_NPU``. Smaller models fit smaller devices; start with the example for your task.

.. dropdown:: Why did training stop before the last epoch?

   Early stopping is on by default. See :ref:`err-early-stopping` and :doc:`/features/early_stopping`.

.. dropdown:: Can I bring my own model?

   Yes. See :doc:`/BRING_YOUR_OWN_MODEL`. For your own dataset see :doc:`/byod/index`.

.. dropdown:: Where are the results?

   Under ``./data/projects/<project_name>/run/<date-time>/<model_name>/`` relative to where you started the run
   (configurable with ``common.projects_path``). See :doc:`/getting_started/understanding_run_output`.

.. dropdown:: How do I estimate memory before training?

   Use the ``estimate_memory`` command described in :doc:`/features/memory_estimation`.

Deployment
----------

.. dropdown:: How do I put the model on the device?

   Copy the generated files into an SDK example. See :doc:`/DEPLOYING_TO_SDK_EXAMPLE` and :doc:`/deployment/index`.
   If the device output differs from the golden vectors, see :ref:`err-golden-vector`.
