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
   :widths: 30 20 20 30

   * - Task Type
     - Search Range
     - Metric
     - Tolerance
   * - Classification
     - ``[4, 8]``
     - Accuracy
     - 5% drop
   * - Anomaly Detection
     - ``[4, 8]``
     - MSE
     - 2x increase
   * - Forecasting
     - ``[4, 32]``
     - SMAPE
     - 2x increase
   * - Regression
     - ``[4, 12]``
     - R²
     - 5% drop

At each candidate average bit width, a fast calibration pass (no full
QAT retraining) is run and the metric is checked against the tolerance
threshold. The lowest average bit width that still passes is selected.
The binary search typically converges in two to three iterations.

If the Hessian estimation fails for any reason, the pipeline falls back
to standard uniform 8-bit QAT automatically.

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
