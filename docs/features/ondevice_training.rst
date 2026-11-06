========================
On-Device Training (ODT)
========================

.. _feature-ondevice-training:

On-device training lets a model **continue training directly on the microcontroller** after deployment. Instead of
the classic flow (train on the PC, deploy a frozen model, inference only), the model is split into a frozen part and a
trainable part. The frozen part is compiled for inference, and the trainable part is deployed with the training
library, so the device can adapt to local data without a round trip to the PC.

.. code-block:: text

   Traditional:  Train on PC -> Compile -> Deploy -> Inference only

   On-device:    Train on PC -> Split model -> Deploy frozen + trainable parts
                 -> Continue training on the MCU -> Inference

Use it when:

* **Data drifts**: the installation differs from the lab where the model was trained.
* **Privacy matters**: raw sensor data must not leave the device.
* **Each installation is different**: personalize per motor, fan or machine without retraining on the PC.
* **Re-deployment is costly**: adapting avoids the collect, retrain, recompile, reflash cycle.

Two frameworks
==============

Tiny ML ModelZoo has two ODT frameworks. Each has a full Markdown guide.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Float on-device training
      :link: /ondevice_training/readme
      :link-type: doc

      The core training library works in 32-bit floating point and is device agnostic. It supports partial
      training (freeze the early layers, train the last k) and full training. The supported task is
      **anomaly detection** with autoencoder models. The example is the fan blade anomaly detection on
      the F29H85x (C29x) LaunchPad.

   .. grid-item-card:: Integer on-device training (new in 1.5.0)
      :link: /integer_ondevice_training/readme
      :link-type: doc

      Training with integer arithmetic only: INT8 weights, INT32 accumulators and integer gradient updates
      (NITI algorithm with adaptive shifts). It uses the 24K-parameter NPU classification model ``CLS_24k_NPU``,
      and the example is arc fault classification on the AM13E230X LaunchPad (M33 core).

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Aspect
     - Float ODT
     - Integer ODT
   * - YAML ``quantization``
     - ``0`` (float)
     - ``2`` (QAT is required)
   * - Weights
     - float32
     - INT8
   * - Gradients
     - float32
     - INT32 accumulate, INT8 store
   * - C library
     - ``ondevice_training_lib.h/.c``
     - ``integer_odl.h/.c``
   * - Example config
     - ``examples/fan_blade_fault_classification/fan_blade_anomaly_detection_ondevice_training.yaml``
     - ``examples/dc_arc_fault/config_dsi_ondevice_learning.yaml``

Configuration
=============

Both flows are switched on from the ``training`` section of the config YAML:

.. code-block:: yaml

   training:
     quantization: 2                       # required for integer ODT
     ondevice_training: true               # enable the ODT export pipeline
     trainable_layers_from_last: 3         # number of trailing layers that are trainable
     export_samples_per_class: '[10,5,5]'  # [train, val, test] samples per class exported as C arrays

For the real values for each flow, use the example configs above and the guides:

* Float ODT: :doc:`/ondevice_training/running_modelzoo_for_odt`
* Integer ODT: :doc:`/integer_ondevice_training/running_modelmaker_for_integer_odt`

Run the example with the usual script, for example
``./run_tinyml_modelzoo.sh examples/dc_arc_fault/config_dsi_ondevice_learning.yaml`` (see :doc:`/getting_started/quickstart`).
The run exports the model, the trainable model configuration and a small data set as C files, which you copy into the
device SDK example project in Code Composer Studio. The guides describe each file.

.. note::

   The ``tinyml-tensorlab`` ModelMaker environment mentioned in a few places of the integer ODT guide is the 1.4.0 way
   to run; since 1.5.0 you run ModelZoo as described in :doc:`/installation/index`.

Documentation
=============

.. toctree::
   :maxdepth: 1
   :caption: Float on-device training

   /ondevice_training/readme
   /ondevice_training/overview
   /ondevice_training/running_modelzoo_for_odt
   /ondevice_training/generated_artifacts_overview
   /ondevice_training/trainable_model_config
   /ondevice_training/ondevice_training_data
   /ondevice_training/ondevice_training_lib
   /ondevice_training/anomaly_detection_odt
   /ondevice_training/application_example
   /ondevice_training/faq

.. toctree::
   :maxdepth: 1
   :caption: Integer on-device training

   /integer_ondevice_training/readme
   /integer_ondevice_training/overview
   /integer_ondevice_training/running_modelmaker_for_integer_odt
   /integer_ondevice_training/generated_artifacts_overview
   /integer_ondevice_training/integer_model_config
   /integer_ondevice_training/training_data
   /integer_ondevice_training/integer_odl_lib
