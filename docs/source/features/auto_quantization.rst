==========================================
Automatic Mixed Precision Quantization
==========================================

Automatic Mixed Precision (AMP) is a Hessian-aware, fully automatic
pipeline that assigns per-layer quantization bit widths using a greedy
algorithm. It is enabled by setting ``auto_quantization: True`` in the
training config and is the recommended approach when deploying quantized
models, especially for regression tasks where uniform quantization can
fail.

Overview
--------

Standard uniform quantization applies the same bit width to every layer
in the model. This is suboptimal because layers differ in their
sensitivity to quantization error — some layers tolerate aggressive
compression while others require higher precision to preserve accuracy.

Automatic Mixed Precision solves this by:

* Estimating the sensitivity of each layer using the Hessian of the loss
* Assigning bit widths layer-by-layer via a greedy algorithm that
  maximises accuracy improvement per bit spent
* Automatically selecting the average bit width through binary search
  calibration — no manual tuning required

Layers assigned bit width ``8``, ``4``, or ``2`` execute on the NPU.
Layers assigned ``32`` (float) run on the CPU. This CPU+NPU split
applies across all task types.

The pipeline runs end-to-end without user intervention and is
generalised across classification, regression, anomaly detection, and
forecasting tasks.

.. note::

   ``auto_quantization`` defaults to ``True`` for all task types
   (classification, regression, anomaly detection, forecasting). It has
   no effect when ``quantization: 0`` (float training). It works with
   both ``quantization: 1``  and
   ``quantization: 2``

Algorithm
---------

The pipeline consists of three sequential stages:

**Stage 1 — Layer-wise Sensitivity Estimation**

The sensitivity of a layer is defined as the largest eigenvalue
(λ\ :sub:`max`) of the Hessian of the training loss with respect to that
layer's weights. A high eigenvalue indicates that small changes in the
weights cause large changes in the loss — i.e. the layer is sensitive to
quantization.

λ\ :sub:`max` is computed efficiently using **power iteration**:

1. Start with a random normalised vector *v*
2. Compute the first-order gradient *g* of the loss via backpropagation
3. Form the scalar product *gv*
4. Differentiate *gv* with respect to the weights to approximate the
   Hessian-vector product *Hv*
5. Normalise *Hv* to obtain the new *v*
6. Repeat until the Rayleigh quotient converges to λ\ :sub:`max`

This requires only a single batch of data and one forward-backward pass
per iteration, making it computationally practical.

**Stage 2 — Greedy Bit Width Assignment**

Given a target average bit width budget for the model, the algorithm
assigns bit widths from the set ``{2, 4, 8, 32}`` by maximising the following metric for each candidate
upgrade:

.. code-block:: text

   Efficiency = Error Reduction / Bit Cost

where *Error Reduction* is the decrease in quantization error when a
layer is upgraded from its current bit width to the next higher one, and
*Bit Cost* is the increase in average model bit width from that upgrade.

Starting from the lowest bit width, the layer with the highest
efficiency is upgraded at each step. This repeats until the total
average bit width budget is exhausted.

**Stage 3 — Automatic Average Bit Width via Calibration**

Rather than requiring the user to specify the average bit width budget,
the pipeline determines it automatically using binary search:

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 20 20

   * - Task Type
     - Search Range
     - Metric
     - Default Tolerance
     - Config Key
   * - Classification
     - ``[4, 8]``
     - Accuracy
     - 5% drop (``0.05``)
     - ``autoquant_tolerance_classification``
   * - Regression
     - ``[4, 12]``
     - R²
     - 5% drop (``0.05``)
     - ``autoquant_tolerance_regression``
   * - Forecasting
     - ``[4, 32]``
     - SMAPE
     - 3× float baseline (``2.0``)
     - ``autoquant_tolerance_forecasting``
   * - Anomaly Detection
     - ``[4, 8]``
     - MSE
     - 3× float baseline (``2.0``)
     - ``autoquant_tolerance_anomaly``

At each candidate average bit width, a fast calibration pass (no full
QAT retraining) is run and the metric is checked against the tolerance
threshold. The lowest average bit width that still passes is selected.
The binary search typically converges in two to three iterations.

If the Hessian estimation fails for any reason, the pipeline falls back
to standard uniform 8-bit QAT automatically.

Tolerance Thresholds
--------------------

The tolerance thresholds control how much metric degradation versus the
float baseline is acceptable during the binary-search calibration. They
are set in ``params.py`` and can be overridden per run in ``config.yaml``
under the ``training`` section.

For accuracy and R², **higher values are better**, so the tolerance is a
fraction representing the maximum allowed *drop* from the float baseline.
The quantized metric must stay above ``float_metric × (1 − tolerance)``.

For SMAPE and MSE, **lower values are better**, so the tolerance is a
value added to ``1.0`` to form a ceiling multiplier. The quantized metric
must stay below ``float_metric × (1 + tolerance)``.

**Classification — autoquant_tolerance_classification (default: 0.05)**

Accuracy is higher-is-better. ``0.05`` means the quantized model's
accuracy may drop by at most **5%** relative to the float model. For
example, if the float model achieves 90% accuracy, the threshold is
``90% × (1 − 0.05) = 85.5%``. Any candidate bit width that pushes
accuracy below that threshold is rejected and the algorithm tries a
higher bit width.

**Regression — autoquant_tolerance_regression (default: 0.05)**

R² is higher-is-better. ``0.05`` means the quantized model's R² may
drop by at most **5%** relative to the float baseline. For example, a
float R² of ``0.95`` sets a threshold of ``0.95 × (1 − 0.05) = 0.9025``.
Regression metrics are highly sensitive to quantization, so keeping
this tight ensures the selected bit width genuinely preserves model
quality.

**Forecasting — autoquant_tolerance_forecasting (default: 2.0)**

SMAPE is lower-is-better. The tolerance is used as an additive factor
to form a ceiling: ``threshold = float_SMAPE × (1 + 2.0) = 3 × float_SMAPE``.
So ``2.0`` means the quantized model's SMAPE may be **at most 3× the
float baseline** before the bit width is rejected. SMAPE is an unbounded
ratio metric, so a multiplicative ceiling is more meaningful than a
fixed fraction. The float SMAPE is recorded at the end of float training
and used as the reference.

**Anomaly Detection — autoquant_tolerance_anomaly (default: 2.0)**

MSE is lower-is-better. The same formula applies:
``threshold = float_MSE × (1 + 2.0) = 3 × float_MSE``. So ``2.0``
means the quantized model's reconstruction MSE may be **at most 3× the
float baseline** before the bit width is rejected. The absolute MSE
value is dataset-dependent, which is why a multiplier is used rather
than a fixed threshold.

Configuration
-------------

Since ``auto_quantization`` defaults to ``True``, no explicit config
change is needed — it is already active whenever ``quantization: 1``
or ``quantization: 2`` is set:

.. code-block:: yaml

   training:
     model_name: 'REGR_13k'
     training_epochs: 100
     quantization: 2

.. note::

   Explicitly setting ``auto_quantization: True`` is not required. The
   above config is sufficient to enable automatic mixed precision.

To disable it and fall back to uniform quantization, explicitly set it
to ``False``:

.. code-block:: yaml

   training:
     training_epochs: 100
     quantization: 2
     auto_quantization: False

.. note::

   When ``auto_quantization: True``, the ``quantization_weight_bitwidth``
   parameter is ignored. Bit widths are assigned per-layer by the greedy
   algorithm, not set uniformly.

**Overriding tolerance thresholds**

The tolerance thresholds have defaults set in ``params.py`` but can be
overridden in ``config.yaml`` under the ``training`` key. Only the keys
relevant to your task type need to be specified:

.. code-block:: yaml

   training:
     model_name: 'REGR_13k'
     training_epochs: 100
     quantization: 2
     auto_quantization: True
     # Tighten regression tolerance: allow at most 2% R² drop instead of 5%
     autoquant_tolerance_regression: 0.02

.. code-block:: yaml

   training:
     model_name: 'AD_17K'
     training_epochs: 100
     quantization: 2
     auto_quantization: True
     # Relax anomaly tolerance: allow up to 4x MSE increase
     autoquant_tolerance_anomaly: 3.0

All four keys and their defaults are:

.. code-block:: yaml

   training:
     autoquant_tolerance_classification: 0.05   # higher-is-better: max 5% accuracy drop vs float
     autoquant_tolerance_regression: 0.05        # higher-is-better: max 5% R² drop vs float
     autoquant_tolerance_forecasting: 2.0        # lower-is-better: SMAPE must stay below 3× float (1 + 2.0)
     autoquant_tolerance_anomaly: 2.0            # lower-is-better: MSE must stay below 3× float (1 + 2.0)

Task-Specific Behaviour
-----------------------

**Classification and Anomaly Detection**

The calibration search runs over ``[4, 8]``. For most classification
datasets the algorithm selects an average bit width in the range 4–6,
giving better compression than uniform 8-bit with comparable accuracy.

**Forecasting**

The calibration search runs over ``[4, 32]`` using SMAPE as the metric.
The wider range accounts for the sensitivity of sequence prediction models.
The float SMAPE is recorded during the float training run and used as the
reference for the binary search tolerance check.

**Regression**

Regression models are particularly sensitive to uniform quantization.
Uniform 8-bit QAT can catastrophically degrade regression metrics (e.g.
R² collapsing from 0.98 to −71 for some datasets). Automatic mixed
precision addresses this by assigning higher bit widths to the sensitive
layers (typically the first convolutional and last linear layers) while
compressing less sensitive intermediate layers.

The calibration search range is extended to ``[4, 12]`` to allow higher
average bit widths when the task demands it.

Example Configs
---------------

**Regression (Washing Machine Load Weighing):**

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

**Generic Timeseries Regression:**

.. code-block:: yaml

   common:
     task_type: generic_timeseries_regression
     target_device: F28P55

   training:
     model_name: REGR_13k
     quantization: 2
     auto_quantization: True

   testing: {}
   compilation: {}

Next Steps
----------

* :doc:`quantization` — Uniform quantization configuration reference
* :doc:`neural_architecture_search` — Automatic model architecture search
* :doc:`/deployment/npu_device_deployment` — Deploy quantized models to device
