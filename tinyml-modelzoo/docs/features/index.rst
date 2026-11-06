========
Features
========

Features of Tiny ML ModelZoo that help you prepare data, make models smaller and faster, and understand the results.

.. toctree::
   :maxdepth: 2
   :caption: Data and training

   feature_extraction
   goodness_of_fit
   early_stopping
   training_performance
   neural_architecture_search

.. toctree::
   :maxdepth: 2
   :caption: Quantization

   quantization
   auto_quantization
   mixed_precision_quantization
   quantization_examples

.. toctree::
   :maxdepth: 2
   :caption: Analysis and memory

   post_training_analysis
   memory_estimation

.. toctree::
   :maxdepth: 2
   :caption: On-device training

   ondevice_training

New in 1.5.0
============

* :doc:`early_stopping`: on by default, with patience and min-delta.
* :doc:`memory_estimation`: Flash and RAM estimate before training, and the standalone ``estimate_memory`` tool.
* :doc:`mixed_precision_quantization`: HAWQ automatic mixed precision and ternary-weight QAT.
* :doc:`quantization`: residual connection support, the cleaner ``auto_quantization`` API and NPU quantization for anomaly detection.
* :doc:`feature_extraction`: audio front ends (MFCC, LPC, FFT, learned filterbank) and new transforms.
* :doc:`ondevice_training`: new integer on-device training.
* :doc:`training_performance`: ``torch.compile``, native AMP and NaN loss detection.

Overview
========

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Feature extraction
      :link: feature_extraction
      :link-type: doc

      Presets, custom transforms, windowing and audio front ends.

   .. grid-item-card:: Quantization
      :link: quantization
      :link-type: doc

      8-, 4- and 2-bit QAT and PTQ for the NPU.

   .. grid-item-card:: Memory estimation
      :link: memory_estimation
      :link-type: doc

      Know the Flash and RAM cost before you train.

   .. grid-item-card:: Neural architecture search
      :link: neural_architecture_search
      :link-type: doc

      Let DARTS find a model for your dataset.

   .. grid-item-card:: Post-training analysis
      :link: post_training_analysis
      :link-type: doc

      ROC curves, thresholds and error plots.

   .. grid-item-card:: On-device training
      :link: ondevice_training
      :link-type: doc

      Continue training on the MCU.
