============
Introduction
============

This section introduces Tiny ML ModelZoo, Texas Instruments' toolchain for developing
machine learning models on microcontrollers.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   what_is_modelzoo
   architecture
   terminology

Overview
--------

Tiny ML ModelZoo enables end-to-end machine learning workflows for embedded systems:

1. **Data Preparation**: format and preprocess your datasets
2. **Model Training**: train neural networks sized for MCUs
3. **Model Optimization**: quantize models for efficient deployment
4. **Compilation**: generate device-ready code with TI's Neural Network Compiler
5. **Deployment**: run inference on TI microcontrollers

.. note::

   Starting with 1.5.0, ``tinyml-modelzoo`` is the standalone, pip-installable entry
   point. It replaces cloning the full ``tinyml-tensorlab`` set of repositories used up
   to 1.4.0. See :doc:`/installation/index`.
