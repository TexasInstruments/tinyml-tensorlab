====================
Training Performance
====================

Tiny ML ModelZoo 1.5.0 adds options and safeguards that speed up training and catch problems early. Two of them
are switches in the config YAML, the others are automatic.

Options in the config YAML
==========================

Both keys go in the ``training`` section and apply to the time series task types (check the default
parameters of your task type if you use another module).

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Key
     - Default
     - Effect
   * - ``compile_model``
     - ``0``
     - ``1`` wraps the model with ``torch.compile``. It uses the ``inductor`` backend on CUDA and ``aot_eager`` on Apple MPS.
       A warm-up forward pass checks that compilation works on your hardware and toolchain; if it fails, training
       falls back to the normal (eager) model with a warning in the log.
   * - ``native_amp``
     - ``False``
     - ``True`` enables PyTorch native automatic mixed precision (``torch.amp.autocast``, with a gradient scaler on CUDA). It works
       on CUDA and MPS and is the choice for non-NVIDIA hardware.

.. code-block:: yaml

   training:
     compile_model: 1
     native_amp: True

.. note::

   ``torch.compile`` is skipped automatically for quantized runs (``quantization`` 1 or 2), because the FX-based quantization
   cannot trace a compiled model. It therefore only speeds up the float training stage. The log says when it is skipped.
   Both options are mainly beneficial on CUDA GPUs and are off by default.

Automatic behavior
==================

* **Persistent data loader workers.** When the number of data loading workers is greater than 0 (the default is 8 on Linux and macOS
  and 0 on Windows), the data loaders keep their worker processes alive between epochs instead of restarting them every epoch.
* **NaN/Inf loss detection.** Training stops with an error as soon as the loss becomes NaN or infinite, and the log
  suggests the usual fixes: reduce ``learning_rate`` or ``batch_size``. This catches instability early instead of wasting
  the rest of the run.
* **Early stopping.** See :doc:`/features/early_stopping`.
* **Memory pre-flight.** The model is exported and compiled before training starts to catch a model that is too
  large for the device. See :doc:`/features/memory_estimation`.

Apple Silicon
=============

macOS ARM64 (MPS) compatibility fixes were made across training, evaluation and quantization in 1.5.0. See
:doc:`/installation/macos_setup`.
