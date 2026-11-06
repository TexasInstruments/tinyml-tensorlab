======================================
Automatic Mixed Precision Quantization
======================================

Auto-quantization is a Hessian-aware (HAWQ) pipeline that assigns a quantization bit width to each
layer automatically. It is on by default and is the recommended approach for quantized models,
especially regression models, where uniform quantization can fail.

.. note::

   In 1.5.0 the old ``qconfig_dict`` was replaced by a ``TinyMLQConfigType`` object, and auto-quantization is
   switched on or off with one flag: ``training.auto_quantization`` in the YAML (``auto_quantization=`` in
   ``TinyMLQConfigType`` when you call the wrappers yourself). See also
   :doc:`/features/mixed_precision_quantization` for the ternary (2-bit) weight QAT and for the per-layer
   bit width allocation in more detail.

Overview
========

Uniform quantization uses the same bit width everywhere, but layers differ in how sensitive they are
to quantization error. Auto-quantization:

* estimates the sensitivity of each layer from the Hessian of the loss,
* assigns bit widths from ``{2, 4, 8, 32}`` per layer with a greedy algorithm that maximises error reduction per bit spent,
* picks the average bit width itself with a binary search against a metric tolerance.

Layers with 8, 4 or 2 bits run on the NPU. Layers left at 32 bits (float) run on the CPU.

``auto_quantization`` defaults to ``True`` for all task types. It has no effect when ``quantization: 0``,
and it works with both ``quantization: 1`` and ``quantization: 2``. When it is on,
``quantization_weight_bitwidth`` and ``quantization_activation_bitwidth`` are overridden, and the log
shows a warning that says so.

Algorithm
=========

**Stage 1: layer-wise sensitivity.** The sensitivity of a layer is the largest eigenvalue of the Hessian
of the training loss with respect to the layer weights, computed by power iteration with Hessian-vector
products on a single batch of data.

**Stage 2: greedy bit width assignment.** Starting with every layer at 2 bits, the layer upgrade with the
highest *error reduction per added bit* is applied repeatedly until the average bit width budget is
used up. Layers left at 32 bits are then probed, least sensitive first, and downgraded to 8 bits if the
metric still passes the threshold.

**Stage 3: automatic average bit width.** A binary search over the average bit width runs a fast
calibration pass (no full QAT retraining) for each candidate and picks the lowest one that stays within
the tolerance:

.. list-table::
   :header-rows: 1
   :widths: 22 14 12 22 30

   * - Task type
     - Search range
     - Metric
     - Default tolerance
     - Config key
   * - Classification
     - ``[4, 8]``
     - Accuracy
     - 5% drop (``0.05``)
     - ``autoquant_tolerance_classification``
   * - Regression
     - ``[4, 12]``
     - R2
     - 5% drop (``0.05``)
     - ``autoquant_tolerance_regression``
   * - Forecasting
     - ``[4, 32]``
     - SMAPE
     - 3x float baseline (``2.0``)
     - ``autoquant_tolerance_forecasting``
   * - Anomaly detection
     - ``[4, 8]``
     - MSE
     - 3x float baseline (``2.0``)
     - ``autoquant_tolerance_anomaly``

If the Hessian estimation cannot run (for example, no calibration data or no computable sensitivities),
the pipeline falls back to the uniform bit widths from ``quantization_weight_bitwidth`` and
``quantization_activation_bitwidth``.

Tolerance thresholds
====================

For accuracy and R2 (higher is better) the tolerance is the maximum fractional *drop* from the float
baseline: the quantized metric must stay above ``float_metric x (1 - tolerance)``. With a float accuracy
of 90% and ``0.05``, the threshold is 85.5%.

For SMAPE and MSE (lower is better) the metric must stay below ``float_metric x (1 + tolerance)``. The
default ``2.0`` therefore allows up to 3x the float baseline.

Defaults are set in the ModelMaker parameters and can be overridden under ``training`` in the YAML:

.. code-block:: yaml

   training:
     autoquant_tolerance_classification: 0.05   # max 5% accuracy drop vs float
     autoquant_tolerance_regression: 0.05       # max 5% R2 drop vs float
     autoquant_tolerance_forecasting: 2.0       # SMAPE must stay below 3x float
     autoquant_tolerance_anomaly: 2.0           # MSE must stay below 3x float

Configuration
=============

Because the default is ``True``, setting ``quantization`` is enough:

.. code-block:: yaml

   training:
     model_name: 'REGR_13k'
     training_epochs: 100
     quantization: 2

To disable it and use uniform bit widths, set ``auto_quantization: False``:

.. code-block:: yaml

   training:
     training_epochs: 100
     quantization: 2
     auto_quantization: False
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

To tighten a tolerance, for example allow at most a 2% R2 drop:

.. code-block:: yaml

   training:
     model_name: 'REGR_13k'
     quantization: 2
     auto_quantization: True
     autoquant_tolerance_regression: 0.02

Task-specific behavior
======================

* **Classification and anomaly detection**: the search runs over ``[4, 8]``.
* **Forecasting**: the search runs over ``[4, 32]`` with SMAPE as the metric, because sequence prediction
  models are more sensitive. The float SMAPE recorded at the end of float training is the reference.
* **Regression**: uniform 8-bit QAT can degrade regression metrics badly for some datasets. Auto-quantization
  gives higher bit widths to the sensitive layers (typically the first convolution and the last linear
  layer) and compresses the others. The range is extended to ``[4, 12]``.

Example config
==============

Regression with auto-quantization, from ``examples/washing_machine_load_weighing/config.yaml``:

.. code-block:: yaml

   common:
     task_type: generic_timeseries_regression
     target_device: F28P55

   dataset:
     dataset_name: washing_machine_load_weighing

   data_processing_feature_extraction:
     stride_size: 0.1
     data_proc_transforms:
       - SimpleWindow
     frame_size: 512
     variables: 6

   training:
     model_name: REGR_13k
     batch_size: 128
     training_epochs: 100
     quantization: 2
     auto_quantization: True

   testing: {}
   compilation: {}

Next steps
==========

* :doc:`/features/quantization`: uniform quantization reference
* :doc:`/features/mixed_precision_quantization`
* :doc:`/features/neural_architecture_search`
