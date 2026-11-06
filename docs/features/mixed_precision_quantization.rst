============================
Mixed Precision Quantization
============================

Tiny ML ModelZoo 1.5.0 can quantize different layers of one model to different bit widths. Layers
that the loss is insensitive to use fewer bits (2 or 4), and sensitive layers keep more (8 or
more). This reduces Flash for the same accuracy compared with one uniform bit width.

Two 1.5.0 capabilities belong to this page:

* **Automatic mixed precision (HAWQ)**: a Hessian-aware search that picks the bit width per layer.
* **Ternary-weight / 8-bit-activation QAT**: 2-bit weights with three levels (-1, 0, +1) and 8-bit activations.

Both are part of :doc:`/features/quantization` and run on the quantized stage of a run
(quantization aware training, QAT, or post training quantization, PTQ). See
:doc:`/features/auto_quantization` for the overall auto-quantization flow.

Automatic mixed precision (HAWQ)
================================

Auto-quantization is **on by default** (``training.auto_quantization: True``) whenever quantization
is enabled. In that case ``quantization_weight_bitwidth`` and ``quantization_activation_bitwidth``
are overridden by the search, and a warning in the log says so. The search works as follows:

#. A Hessian-based sensitivity is computed for each layer (power iteration on Hessian-vector
   products, using a batch of the training data and the training loss).
#. A binary search finds the lowest *average* weight bit width whose quantized model still stays
   within a tolerance of the float model metric. The search range depends on the task type
   (classification and anomaly detection: 4 to 8 bits; regression: 4 to 12; forecasting: 4 to 32).
#. For a given average bit width, a greedy allocator assigns each layer one of 2, 4, 8 or 32 bits,
   giving more bits to layers with higher sensitivity per parameter.
#. Layers left at 32 bits are probed one by one (least sensitive first) and downgraded to 8 bits if
   the metric still passes the threshold, so that only layers that truly need float stay at 32 bits
   (these run on the CPU). The final per-layer table is written to the log.

If the search cannot run (for example, no calibration data loader or no computable sensitivities),
the uniform bit widths you configured are used instead.

To use fixed bit widths and disable the search, set ``auto_quantization: False``:

.. code-block:: yaml

   training:
     quantization: 2                       # 0 = none, 1 = generic, 2 = TINPU (NPU devices)
     quantization_method: 'QAT'            # or 'PTQ'
     auto_quantization: False
     quantization_weight_bitwidth: 4       # 8, 4 or 2
     quantization_activation_bitwidth: 8

Tolerances (``training`` section)
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Key
     - Default
     - Meaning
   * - ``autoquant_tolerance_classification``
     - 0.05
     - Largest tolerated accuracy drop, as a fraction (0.05 = 5%).
   * - ``autoquant_tolerance_regression``
     - 0.05
     - Largest tolerated R2 drop, as a fraction.
   * - ``autoquant_tolerance_forecasting``
     - 2.0
     - Maximum SMAPE is ``float_SMAPE x (1 + 2.0)``, that is 3x the float baseline.
   * - ``autoquant_tolerance_anomaly``
     - 2.0
     - Maximum MSE is ``float_MSE x (1 + 2.0)``, that is 3x the float baseline.

Ternary-weight, 8-bit-activation QAT
====================================

Setting ``quantization_weight_bitwidth: 2`` selects differentiable ternary quantization (DBQ) for
the weights: during QAT the weights are pushed towards the three levels -1, 0 and +1 with a learned
scale. Activations stay at ``quantization_activation_bitwidth`` (8 by default). 4-bit weights use a
soft-sigmoid quantizer during QAT in the same way.

.. code-block:: yaml

   training:
     quantization: 2
     quantization_method: 'QAT'
     auto_quantization: False
     quantization_weight_bitwidth: 2
     quantization_activation_bitwidth: 8

Models that mix bit widths per layer are used by the audio examples. For example, ``DSCNN_GB_NPU``
(glass break detection) uses 4-bit depthwise and pointwise convolutions with an 8-bit stem, and
``TCDS_ResNet_FB_NPU`` (wake word detection) uses a 2-bit backbone with 8-bit stem and head.

.. note::

   Mixed precision applies to the quantized (NPU) flow. Pick bit widths that the target device NPU
   supports; see :doc:`/NPU_CONFIGURATION_GUIDELINES`.

.. tip::

   Use :doc:`/features/memory_estimation` with ``--weight-bitwidth`` and ``--activation-bitwidth``
   to see the Flash effect of a bit width choice before training.
