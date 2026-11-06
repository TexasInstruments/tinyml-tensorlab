====================
Bring Your Own Model
====================

This section explains how to use your own neural network with Tiny ML ModelZoo 1.5.0. There are two approaches.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Add a custom model to the ModelZoo
      :link: /BRING_YOUR_OWN_MODEL
      :link-type: doc

      Create a new architecture that plugs into the full pipeline (training, quantization, compilation). Either point
      the config at your own ``.py`` file with ``custom_model_path`` (no source edits), or add a model class and a
      model description to ModelZoo. Follow the step-by-step guide.

   .. grid-item-card:: Compile an existing ONNX model
      :link: compilation_only
      :link-type: doc

      You already have a trained model from another framework. Skip the dataset and training steps and compile the
      ONNX file for a TI device.

Contents
========

.. toctree::
   :maxdepth: 2

   /BRING_YOUR_OWN_MODEL
   adding_models
   compilation_only

Which approach do I need?
=========================

* Use the **ModelZoo approach** when you want ModelMaker to train, quantize and test the model with your dataset.
  Start with a **custom model file** (``training.custom_model_path``) to try an architecture quickly, then add it to the
  ModelZoo itself when you want it to show up with the other ModelZoo models and in the GUI.
* Use **compilation only** when the model is already trained (and, for NPU devices, already quantized) and you only need the
  compiled library for the device.

.. important::

   A **custom model file** (``custom_model_path``) works with either install. Adding a model **permanently** to the
   ModelZoo edits files in the ``tinyml-modelzoo`` source tree, so it needs the **clone install**
   (``pip install -e .``). See :doc:`/installation/developer_installation`. Compile-only needs just the normal
   install (:doc:`/installation/user_installation`).

Model requirements
==================

**NPU devices** (for example F28P55): models have to follow the NPU rules to be accelerated, such as channel counts that
are multiples of 4 and limits on kernel sizes. The full list is in :doc:`/NPU_CONFIGURATION_GUIDELINES`. Layers that
do not meet the rules fall back to the CPU.

**Non-NPU devices**: the architecture is more flexible, but keep the total parameter count within the device memory,
and use layers the compiler supports. You can check the memory footprint before training with
:doc:`/features/memory_estimation`.
