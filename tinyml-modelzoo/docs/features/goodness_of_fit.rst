===============
Goodness of Fit
===============

The Goodness of Fit (GoF) test helps you judge dataset quality and class separability before you spend time on
training. It answers these questions:

* Are my classes separable in feature space?
* Is my choice of frame size appropriate?
* Is it likely that a neural network can learn the patterns?
* Which classes might be confused?

GoF works on the raw dataset files of each class. It is available for classification task types (time series,
image, audio and radar classification).

Enabling the GoF test
=====================

Set ``gof_test: True`` and a ``frame_size`` in the ``data_processing_feature_extraction`` section:

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'your_dataset'

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_1024Input_FFTBIN_64Feature_8Frame'
     gof_test: True
     frame_size: 256

.. note::

   ``frame_size`` must be given in the YAML. Without it the GoF plot is skipped with a warning in the log.

Run the config as usual (see :doc:`/getting_started/quickstart`):

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/your_example/config.yaml

   .. tab:: Windows (cmd)

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\your_example\config.yaml

   .. tab:: Windows (PowerShell)

      .. code-block:: powershell

         ./run_tinyml_modelzoo.ps1 examples/your_example/config.yaml

To run only the data analysis and skip training, combine it with ``dont_train_just_feat_ext``:

.. code-block:: yaml

   data_processing_feature_extraction:
     gof_test: True
     dont_train_just_feat_ext: True
     frame_size: 256

The run then exits right after the plots (the PCA plots of the extracted features and the GoF plot) are written. See
:doc:`/features/feature_extraction`.

Output
======

The test writes one figure, ``GoF_frame_size_<frame_size>.png``, to the ``training/base`` folder of the run, next to
the ``pca_on_feature_extracted_*_data.png`` plots. The figure has 8 sub-plots.

The 8 plots
===========

The 8 plots come from three stages: **2 transforms x 2 scalings x 2 dimensionality reductions = 8**.

**Transforms**

* **FFT + Abs + Log**: moves each frame to the frequency domain, keeps the positive half of the FFT, takes the absolute value
  and applies a log to compress large magnitudes.
* **Wavelet transform (WT)**: analyzes time and frequency together, which captures localized events such as spikes.

**Scalings**

* **Standard scaler**: zero mean, unit variance.
* **Min-Max scaler**: scales each feature to [0, 1].

**Dimensionality reduction**

* **PCA**: linear projection that keeps as much variance as possible.
* **t-SNE**: non-linear projection that preserves local neighborhoods and reveals cluster structure.

.. code-block:: text

   Plot 1: FFT+Abs+Log + Standard scaler + PCA
   Plot 2: FFT+Abs+Log + Standard scaler + t-SNE
   Plot 3: FFT+Abs+Log + Min-Max scaler + PCA
   Plot 4: FFT+Abs+Log + Min-Max scaler + t-SNE
   Plot 5: WT          + Standard scaler + PCA
   Plot 6: WT          + Standard scaler + t-SNE
   Plot 7: WT          + Min-Max scaler  + PCA
   Plot 8: WT          + Min-Max scaler  + t-SNE

.. important::

   Not all 8 plots need to show separable clusters. Each one is a different view of the data. If **any one** of them shows
   separable clusters, that is a strong sign the dataset is suitable for classification.

**Example GoF figures**

.. figure:: /_static/img/goodness_of_fit/arc_fault_dsi_GoF_frame_size_256.png
   :width: 600px
   :align: center
   :alt: GoF plot for arc fault with frame size 256

   Arc fault, frame size 256

.. figure:: /_static/img/goodness_of_fit/arc_fault_dsi_GoF_frame_size_1024.png
   :width: 600px
   :align: center
   :alt: GoF plot for arc fault with frame size 1024

   Arc fault, frame size 1024

.. figure:: /_static/img/goodness_of_fit/motor_dsk_GoF_frame_size_256.png
   :width: 600px
   :align: center
   :alt: GoF plot for motor bearing fault

   Motor bearing fault, frame size 256

Interpreting the results
========================

* **Well-separated, tight clusters**: good separability.
* **Overlapping clusters**: potential confusion between those classes.
* **One class scattered while others are tight**: check for mislabeled samples, or split the class into sub-classes.
* **All classes overlap with no structure**: the frame size or the data may not carry discriminative information.

If classes overlap, try a different frame size, a different feature extraction preset, or review the labels
and add data for the affected classes.

Frame size sweeping
===================

If the plots are inconclusive, sweep ``frame_size``. A small frame captures only a thin slice of the signal, which
matters most for data with a high sampling frequency. In the 1.4.0 guide, the arc fault dataset (classes arc and normal) showed
no separation at 256 and 512 samples and clear separation in the wavelet plots only at 4096 samples.

Start with the frame size that matches the model input. If the plots are inconclusive, try 2x and 4x. The frame size
with the clearest separation indicates the minimum signal length needed for reliable classification.

Multiple clusters per class
===========================

A single class that appears as several clusters does not mean the data is bad. It often reflects real variation in
the collection: different sampling frequencies, equipment or sensors, environment or operating states. Check the
metadata, filter the data by the suspected variable and run GoF again. If the multi-cluster pattern goes away, decide
whether to train on the full mixed dataset or on a controlled subset.

Comparing feature extraction
============================

Run GoF with different settings and compare the figures, for example ``frame_size`` 256, 512 and 1024. Combined with
``dont_train_just_feat_ext: True`` this is a fast way to compare options.

Limitations
===========

* GoF is mostly a linear and visual analysis; neural networks can learn non-linear boundaries.
* Good GoF does not guarantee good accuracy, and poor GoF may still give an acceptable model.
* 2D projections can hide separability in higher dimensions.

Use GoF as a guide, not as a definitive answer.

Next steps
==========

* :doc:`/features/feature_extraction`
* :doc:`/features/post_training_analysis`
