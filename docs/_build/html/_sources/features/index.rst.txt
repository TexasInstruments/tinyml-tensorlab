=================
Advanced Features
=================

Tiny ML Tensorlab includes several advanced features to help you build
more accurate and efficient models. This section covers these capabilities in detail.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   neural_architecture_search
   quantization
   feature_extraction
   goodness_of_fit
   post_training_analysis

Feature Overview
----------------

**Neural Architecture Search (NAS)**

Automatically discover optimal neural network architectures for your dataset.
NAS can optimize for memory usage or computational efficiency.

* Preset sizes: ``s``, ``m``, ``l``, ``xl``, ``xxl``
* Optimization modes: Memory or Compute
* GPU recommended for practical use

**Quantization**

Reduce model size and improve inference speed through quantization:

* **QAT** (Quantization-Aware Training) - Best accuracy
* **PTQ** (Post-Training Quantization) - Faster, no retraining
* Weight bit-widths: 2-bit, 4-bit, 8-bit

**Feature Extraction**

Transform raw time-series data into meaningful features:

* FFT (Fast Fourier Transform)
* Binning and normalization
* Haar and Hadamard wavelets
* Logarithmic scaling

**Goodness of Fit Test**

Evaluate whether your dataset is suitable for classification before training.
Uses PCA and t-SNE visualization to assess class separability.

**Post-Training Analysis**

Understand model performance with:

* ROC curves for classification
* Confusion matrices
* FPR/TPR threshold analysis
* PCA visualization of feature-extracted data
