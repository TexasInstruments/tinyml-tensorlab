=================
Memory Estimation
=================

Tiny ML ModelZoo 1.5.0 can tell you how much Flash (ROM) and RAM a model and its feature
extraction will need on the device **before** you spend time on training. There are two ways to
get this estimate:

* automatically, as a *memory pre-flight* step in every training run, and
* on demand, with the standalone ``estimate_memory`` command.

Both export an untrained, randomly initialised copy of the model to ONNX and compile it with the
same compiler path used at the end of the pipeline. Memory use depends on the architecture's
shapes and data types, not on the trained weight values, so the numbers are representative.

The report has four lines per library: **Code**, **RO Data** (Flash), **RW Data** (RAM) and
**Total**.

Pre-flight during a training run
================================

When training starts, the training script creates the model, exports it and compiles it just to
read the memory report. An oversized model is therefore caught right after model creation, not
after a full training and compilation run. The files are written to a ``memory_preflight`` folder
next to the training output, for example::

   .../<model_name>/training/base/memory_preflight/
       compilation/
       onnx_export/
       memory_preflight_full.log

``memory_preflight_full.log`` holds the complete compiler output; the console and ``run.log`` show
only the cleaned-up memory blocks (``TI Model Library Memory Usage`` and
``TI Feature Extraction Library Memory Usage``). The quantized training stage has its own
``memory_preflight`` folder under ``training/quantization/``.

The pre-flight never blocks training. If the compiler tools (``ti_mcu_nnc``/TVM) cannot be loaded,
the step is skipped with a warning. The training script option ``--skip_memory_preflight`` turns it
off.

The Feature Extraction Library (FEL) is compiled and measured by the ModelMaker compilation
stage, using the same ``fel_memory`` tooling that ships with TinyVerse. That is how the RAM and Flash
cost of the feature extraction (FFT, windowing, and so on) is reported next to the neural network.

Standalone ``estimate_memory``
==============================

``estimate_memory`` resolves the model, feature extraction preset and target device from a
ModelMaker config and prints the estimate in seconds. It does not download a dataset, train, or run
the full compilation pipeline.

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./estimate_memory.sh examples/dc_arc_fault/config_dsk.yaml

   .. tab:: Any OS

      .. code-block:: bash

         python estimate_memory.py examples/dc_arc_fault/config_dsk.yaml

Run it from the ``tinyml-modelzoo`` folder, as for the other run scripts.

.. list-table:: Options
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning
   * - ``config_file``
     - Path to a ModelMaker YAML (or JSON) config (required).
   * - ``--target_device``
     - Override ``common.target_device``.
   * - ``--model_name``
     - Override ``training.model_name``.
   * - ``--variables``
     - Override the number of input variables (channels); skips config resolution.
   * - ``--input-features``
     - Override the model input feature dimension; skips config resolution.
   * - ``--num-classes``
     - Override the number of classes (model output size); avoids resolving a cached dataset.
   * - ``--opset-version``
     - ONNX opset used for the export (default 18, same as training).
   * - ``--quantization {0,1,2}``
     - Quantization scheme to estimate next to the float32 baseline: 0 none, 1 generic, 2 TINPU.
       Defaults to the config's own value (usually 2). Use ``0`` for float32 only.
   * - ``--quantization-method {QAT,PTQ}``
     - Quantization flavour when ``--quantization`` is non-zero (default ``QAT``).
   * - ``--weight-bitwidth {16,8,4,2}``
     - Weight bit width for the quantized estimate (default 8).
   * - ``--activation-bitwidth``
     - Activation bit width for the quantized estimate (default 8).
   * - ``--skip-fel``
     - Skip the Feature Extraction Library estimate and report only the neural network.
   * - ``--output-dir``
     - Scratch directory for the export and compile (a temporary folder by default).

The output lists the float32 base model, the FEL paired with it, then the quantized model, the FEL
paired with the quantized model and the percentage of memory saved by quantization.

Limitations
-----------

* Only ``timeseries`` task types are supported; other target modules exit with an error.
* Configs with ``training.nas_enabled: True`` are not supported, because NAS has no fixed
  architecture before the search runs.
* The FEL estimate is not available for every device; the tool then prints
  ``Not available for this device``.

Typical uses are checking whether a model fits a device's Flash and RAM, and comparing models or
feature extraction presets quickly. See also :doc:`/features/neural_architecture_search`
and :doc:`/features/quantization`.
