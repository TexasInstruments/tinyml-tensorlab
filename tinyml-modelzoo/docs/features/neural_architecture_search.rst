==========================
Neural Architecture Search
==========================

Neural Architecture Search (NAS) discovers a model architecture for your task and device constraints,
instead of using a model from the ModelZoo.

* It searches over layer configurations with a gradient-based method (DARTS).
* It optimizes for memory (parameter count) or compute (MACs).
* It generates a model for your dataset that can then be quantized and compiled like any other model.

.. warning::

   NAS is compute-intensive. A GPU is needed for practical use, and each NAS epoch can take several minutes.
   Choose ``nas_epochs`` with care.

When to use NAS
===============

Use it when you do not know the right model size for the task, when you want the smallest model that meets an accuracy
target, or when manual architecture tuning takes too long. Do not use it when a ModelZoo model already works, when time
is short, or when no GPU is available.

Supported task types
====================

In 1.5.0 the NAS arguments are passed to the training script of the **time series classification**, **image
classification**, **audio classification** and **radar classification** task types. Regression, forecasting and anomaly
detection do not use NAS.

NAS and the memory estimate
===========================

The standalone ``estimate_memory`` tool and the memory pre-flight need a fixed architecture, so they do not support
configs with ``nas_enabled: True``. See :doc:`/features/memory_estimation`.

How it works
============

1. The ``training`` section of the config YAML is parsed and the NAS parameters are read.
2. With ``nas_enabled: True`` no ModelZoo model is needed: ModelMaker creates a minimal generic model description, so
   ``model_name`` does not have to name a ModelZoo model.
3. The search runs for ``nas_epochs`` epochs, optimizing for the mode in ``nas_optimization_mode``.
4. The best architecture found is trained from scratch with the rest of the training configuration.
5. The result is tested, quantized and compiled as for any other model.

Configuration
=============

All NAS parameters are in the ``training`` section. The defaults are the ModelMaker defaults.

.. list-table::
   :header-rows: 1
   :widths: 30 12 38 20

   * - Parameter
     - Default
     - Description
     - Values
   * - ``nas_enabled``
     - ``False``
     - Enable NAS.
     - ``True``, ``False``
   * - ``nas_epochs``
     - 10
     - Number of epochs of the search.
     - integer
   * - ``nas_optimization_mode``
     - ``'Memory'``
     - ``'Memory'`` penalizes the parameter count (read-only data on the MCU). ``'Compute'`` penalizes MACs/FLOPs
       (peak SRAM usage and latency).
     - ``'Memory'``, ``'Compute'``
   * - ``nas_model_size``
     - ``None``
     - Preset search space size. If set, it overrides the customization parameters below.
     - ``'s'``, ``'m'``, ``'l'``, ``'xl'``

**Customization parameters**, used when ``nas_model_size`` is not set:

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Parameter
     - Default
     - Description
   * - ``nas_nodes_per_layer``
     - 4
     - Nodes (operations) per layer in the DAG.
   * - ``nas_layers``
     - 3
     - Number of layers; the minimum is 3.
   * - ``nas_init_channels``
     - 1
     - Initial feature map channels of the first convolution.
   * - ``nas_init_channel_multiplier``
     - 3
     - Channel multiplier after the first pre-processing step.
   * - ``nas_fanout_concat``
     - 4
     - Nodes per layer concatenated into the layer output (at most ``nas_nodes_per_layer``).

Model size presets
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 17 20 20 28

   * - Preset
     - Layers
     - Nodes/layer
     - Init channels
     - Channel multiplier
   * - ``s``
     - 3
     - 4
     - 1
     - 3
   * - ``m``
     - 10
     - 4
     - 1
     - 3
   * - ``l``
     - 12
     - 4
     - 4
     - 3
   * - ``xl``
     - 20
     - 4
     - 4
     - 3

All presets use ``nas_fanout_concat`` 4. The training script command line accepts ``s``, ``m``, ``l`` and ``xl``.

Usage
=====

**Preset mode (recommended):**

.. code-block:: yaml

   training:
     nas_enabled: True
     nas_epochs: 10
     nas_optimization_mode: 'Memory'
     nas_model_size: 'm'

**Customization mode:**

.. code-block:: yaml

   training:
     nas_enabled: True
     nas_epochs: 20
     nas_optimization_mode: 'Compute'
     nas_nodes_per_layer: 4
     nas_layers: 5
     nas_init_channels: 8
     nas_init_channel_multiplier: 2
     nas_fanout_concat: 3

Full example
============

.. code-block:: yaml

   common:
     task_type: 'arc_fault'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'dc_arc_fault_example_dsk'

   data_processing_feature_extraction:
     feature_extraction_name: 'FFT1024Input_256Feature_1Frame_Full_Bandwidth'
     variables: 1

   training:
     training_epochs: 15
     batch_size: 256
     nas_enabled: True
     nas_epochs: 10
     nas_optimization_mode: 'Memory'
     nas_model_size: 'm'

   testing: {}
   compilation: {}

Run it as any other config, for example ``./run_tinyml_modelzoo.sh <your_nas_config>.yaml`` (see
:doc:`/getting_started/quickstart`).

The training script can also load an architecture found earlier through the ``load_saved_model`` parameter (path of a
pre-searched NAS model).

Tips
====

* Start with a ModelZoo model, and use NAS only if you need a different size or accuracy trade-off.
* Start with presets ``s`` or ``m`` before larger ones.
* More ``nas_epochs`` can improve the result but lengthen the run.
* Use ``'Memory'`` for devices with limited Flash and RAM, ``'Compute'`` for latency-sensitive applications.
* Compare the NAS model with the hand-designed models; it does not always win.
* Validate the found model on your test data before deployment.

Search algorithm
================

The search uses **Differentiable Architecture Search (DARTS)**. Every edge of the search DAG has architecture parameters
(``alphas``), one per candidate operation, and a softmax over them gives a mixture of operations. The model weights and
the alphas are optimized alternately (bilevel optimization): weights on training data, alphas on validation data. After
the search, the operation with the highest alpha on each edge is selected, which gives a discrete architecture (the
genotype), and this is retrained from scratch.

The search includes a resource penalty: parameter count in ``'Memory'`` mode, MACs/FLOPs in ``'Compute'`` mode.

The candidate operations (``PRIMITIVES_CNN``) are 1D convolution and pooling primitives for time series:
``none``, ``avg_pool_3x1``, ``max_pool_3x1``, ``skip_connect``, ``conv_bn_relu_3x1``, ``conv_bn_relu_5x1`` and
``conv_bn_relu_7x1``. A found architecture consists of normal cells (which keep the dimensions) and reduction cells
(which downsample).

The NAS code is in the ``tinyml_torchmodelopt.nas`` package of tinyml-modeloptimization.

References
==========

* Liu, H., Simonyan, K., and Yang, Y. (2019). `DARTS: Differentiable Architecture Search <https://arxiv.org/abs/1806.09055>`_. ICLR 2019.
* Ye, P., et al. (2022). `beta-DARTS: Beta-Decay Regularization for Differentiable Architecture Search <https://arxiv.org/abs/2203.01665>`_.
* Bender, G., et al. (2020). `Balanced One-shot Neural Architecture Optimization <https://arxiv.org/abs/1909.10815>`_.

Next steps
==========

* :doc:`/features/quantization`
* :doc:`/features/feature_extraction`
