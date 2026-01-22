=========
Changelog
=========

Version history and release notes for Tiny ML Tensorlab.

Version 1.2.0
-------------

*Current Release*

**New Features:**

* Added MSPM0G5187 NPU support
* New anomaly detection autoencoder architectures
* Enhanced Neural Architecture Search (NAS) algorithms
* Improved quantization-aware training (QAT)
* Edge AI Studio Model Composer integration
* Documentation overhaul with Sphinx

**Models:**

* Added CLS_20k_NPU and CLS_55k_NPU for complex tasks
* New forecasting models (FCST_* family)
* Updated motor fault models

**Bug Fixes:**

* Fixed INT4 quantization issues on F28P55
* Resolved memory allocation errors in large models
* Fixed GoF test visualization on Windows

**Breaking Changes:**

* Configuration file format updated for ``data_processing_feature_extraction``
* Model registry API changed (use ``MODEL_REGISTRY`` dict)

Version 1.1.0
-------------

**New Features:**

* AM13E2 device support
* Time series forecasting task type
* Multi-variable input support
* Goodness of Fit (GoF) test improvements
* Windows native support (without WSL)

**Models:**

* Added REGR_* regression model family
* New AD_* anomaly detection models
* Updated ArcFault models for better accuracy

**Improvements:**

* Faster compilation for NPU devices
* Reduced memory usage during training
* Better error messages

Version 1.0.0
-------------

*Initial Release*

**Features:**

* Support for 20+ TI microcontrollers
* Time series classification, regression, anomaly detection
* NPU support for F28P55
* Quantization (PTQ and QAT)
* Neural Architecture Search
* Feature extraction presets
* Post-training analysis tools

**Supported Devices:**

* C2000 family (F28P55, F28P65, F2837, etc.)
* MSPM0 family (MSPM0G3507, MSPM0G3519)
* AM26x family (AM263, AM263P, AM261)
* Connectivity devices (CC2755, CC1352)

**Models:**

* Classification models (CLS_100 through CLS_13k)
* NPU models (CLS_*_NPU variants)
* Arc fault detection models
* Motor fault detection models

Migration Guides
----------------

Migrating from 1.1.x to 1.2.x
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Configuration Changes:**

Old format:

.. code-block:: yaml

   feature_extraction:
     preset: 'my_preset'

New format:

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'my_preset'

**Model Names:**

Some model names have been updated for consistency:

.. code-block:: text

   Old: TimeSeries_Generic_1k_t
   New: CLS_1k

**API Changes:**

Model access changed:

.. code-block:: python

   # Old
   from models import get_model
   model = get_model('CLS_1k')

   # New
   from tinyml_tinyverse.common.models import MODEL_REGISTRY
   model_class = MODEL_REGISTRY['CLS_1k']

Deprecation Notices
-------------------

**Version 1.2.0:**

* ``feature_extraction`` config section renamed to ``data_processing_feature_extraction``
* Old preset names deprecated (will be removed in 1.4.0)

**Planned for 1.3.0:**

* Python 3.9 support will be dropped
* Legacy model names will be removed

Known Issues
------------

**Current:**

* Windows: Long path names may cause issues
* Large models (>50k params): May require increased stack size
* QAT: Minor accuracy variations between runs

**Workarounds:**

* Windows: Use shorter project paths
* Large models: Adjust linker settings for stack
* QAT: Set random seed for reproducibility

Roadmap
-------

**Planned Features:**

* Additional MCU family support
* Improved image classification capabilities
* Advanced hyperparameter tuning
* Cloud training integration
* TensorFlow Lite model import

**Community Requests:**

* RNN/LSTM support (under evaluation)
* Multi-task learning
* Model ensemble support

Contributing
------------

Tiny ML Tensorlab welcomes contributions:

* Report issues: https://github.com/TexasInstruments/tinyml-tensorlab/issues
* Submit PRs: https://github.com/TexasInstruments/tinyml-tensorlab/pulls
* Join discussions: TI E2E Forums

See CONTRIBUTING.md for guidelines.
