=========
Changelog
=========

Version history and release notes for Tiny ML ModelZoo (until 1.4.0 published as Tiny ML Tensorlab).

Version 1.6.0
-------------

*Current Release (2026-Oct)*

**Radar:**

* New example ``radar_surface_classification``: surface/material classification from 1-D FFT range-profile data (preset ``Generic_130Input_Radar_SurfaceClassification``, task type ``radar_surface_classification``)
* ``radar_pose_and_fall_detection`` renamed to ``radar_posture_and_fall_detection`` (task type ``radar_point_cloud_classification``, preset ``Generic_248Input_Posture_and_Fall_Detection``) with an annotation script and a detailed README
* New models: ``RF_MLP_3L_11K`` (10,982 parameters at 130 inputs), ``RF_MLP_4L_19K`` (19,298 at 248 inputs) and ``RF_DualStream_PointCloud_19K`` (about 18.9K at 248 inputs), replacing ``Pose_and_Fall_model``; all target xWRL6432
* New guide ``docs/RADAR_PROCESSING_PIPELINE.md`` covering the radar pre-processing presets and parameters
* Radar models total 3; enabled models total 85 (was 83)

Version 1.5.0
-------------

*Released 2026-Sep*

**Packaging and entry point:**

* ``tinyml-modelzoo`` is now the standalone, pip-installable entry point to TI's MCU AI flow. See :doc:`/installation/index`.
* ``tinyml-tensorlab`` is deprecated as the public entry point (its source is now private). Its history is kept in the older entries below.
* ModelZoo and TinyVerse examples run standalone through the published ModelMaker wheel dependency (TI wheel CDN).
* ``build_wheels.sh`` added to build the TinyVerse, ModelOptimization and ModelMaker wheels from local source.

**Device Support (33 MCU/wireless devices):**

* AM13E2: vision classification added, plus radar tasks
* ``fel_memory`` (feature-extraction library) support for AM13 and C28x (F280013x, F280015x, F28002x) devices, which enables on-device RAM/Flash estimation
* Gen3 F28x support reconciled: F2838x, F28P551x, F28002x and a new F28E12x device profile across regression, forecasting and anomaly detection. Over-broad device lists (for example arc fault and generic timeseries classification) were tightened to the verified per-model support.
* Target device lists for radar, image and audio classification are now derived from each model's real device support

**Applications (37 examples: 4 generic timeseries tasks + 33 specific applications):**

* New: cough detection (audio), WiFi CSI presence detection, Google speech command, glass break detection (audio), wake word detection (audio), BMS State-of-Charge estimation (regression)
* Washing machine load weighing regression example re-enabled and renamed to ``washing_machine_load_weighing`` (was ``reg_washing_machine``)

**Models (83 across classification, regression, forecasting and anomaly detection, timeseries and radar, plus vision and audio classification):**

* TCDSResNet and a compressed DSCNN for audio classification (cough detection), both NPU-compliant
* MobileNet_v1 compressed to fit the AM13 memory budget
* Radar Point Cloud Classification (new model and dataset flow)
* Learned-filterbank front-end (jointly trained Conv1D bank) with TCDS-ResNet for wake word detection, and a compact DSCNN variant for glass break detection, both NPU-compliant with mixed-precision quantization
* A 24K-parameter NPU classification model with integer on-device learning (ODL) support, plus new integer ODL documentation. See :doc:`/integer_ondevice_training/readme`.

See :doc:`model_zoo_reference` for the model list.

**Flows:**

* Early stopping for training (patience and min-delta, on by default) across all task types
* ModelMaker can report estimated RAM and Flash usage for feature extraction before training, including a standalone ``estimate_memory`` CLI that runs without a full training pass
* Preprocessing and publicly available datasets supported in the audio and radar flows
* NaN-loss detection during training to catch instability early

**Model Optimization:**

* Residual connection support in quantization (TINPU)
* ``qconfig_dict`` replaced by a cleaner API to toggle auto-quantization on and off
* Hessian-aware (HAWQ) automatic mixed-precision quantization, with fixes to the qconfig mapping and mixed-precision paths
* Automated ternary-weight / 8-bit-activation QAT
* NPU quantization enabled for anomaly detection models

**Reliability and compatibility:**

* Python 3.14 / PyTorch 2.11 compatibility
* macOS ARM64 (MPS, Apple Silicon) fixes across training, evaluation and quantization
* ``torch.compile`` safety: compiled models are unwrapped before ONNX/checkpoint export, with eager fallback on failure
* ONNX export no longer crashes inside PyInstaller/frozen builds
* Security hardening: safe checkpoint deserialization, safe YAML loading, safer cache-dataset handling
* Training performance: ``torch.compile``, AMP, persistent workers, more efficient evaluation loop
* Much larger automated test coverage across all repositories

**Acknowledgement:**

* Thanks to `@musicalplatypus <https://github.com/musicalplatypus>`_ for macOS/Apple Silicon (MPS) support, training-performance work, NAS bug fixes, security fixes, CI and test-suite improvements.

Version 1.4.0
-------------

**New Applications:**

* Gearbox Fault Detection — 2-class vibration classification for MSPM0G5187 NPU
* Electrical Fault 6-Class — fault type classification (G/C/B/A combinations)

**Device Support (23 MCUs):**

* CC1312 — added support for Fan Blade Fault Classification and PIR Detection
* CC1314, CC1352, CC1354 — added support for Fan Blade Fault Classification
* CC2755, CC35X1 — expanded support for Fan Blade Fault Classification
* MSPM0G5187 — added PMSM Rotor Forecasting config

**Model Optimization:**

* LSQ (Learned Step-size Quantization) observer for weights and activations
* MLP models with 2D input now fully supported for quantization and conversion

**Compilation:**

* TI MCU Neural Network Compiler upgraded to 2.1.2

**Platform:**

* macOS 12 or later officially supported

**Other Changes:**

* Timeseries Anomaly Detection re-enabled in default installation
* Reconstruction error plot description added to anomaly detection output
* F3 and WiFi SDK device descriptions updated
* AM13 SDK entry added to connectivity dependencies
* MSPM0L3 (M33) SDK version updated

Version 1.3.0
-------------

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
* Connectivity devices (CC2755, CC1352, CC1354, CC35X1)

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
   from tinyml_modelzoo.models import get_model
   model = get_model('CLS_1k', variables=1, num_classes=2, input_features=128)

Deprecation Notices
-------------------

**Version 1.3.0:**

* ``feature_extraction`` config section renamed to ``data_processing_feature_extraction``
* Old preset names deprecated (removed in 1.4.0)
* Python 3.9 support dropped
* Legacy model names removed

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

* Report issues: https://github.com/TexasInstruments/tinyml-modelzoo/issues
* Submit PRs: https://github.com/TexasInstruments/tinyml-modelzoo/pulls
* Join discussions: TI E2E Forums

See CONTRIBUTING.md for guidelines.
