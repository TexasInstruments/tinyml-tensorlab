# Tiny ML ModelZoo

Texas Instruments' central repository for AI models, examples, and configurations for microcontroller (MCU) applications. Clone this repo, install it, and run any example config against your target device — training, quantization, and compilation all happen automatically underneath.

Detailed User Guide: [TI Tiny ML Tensorlab User Guide](https://software-dl.ti.com/C2000/esd/mcu_ai/user_guide/index.html)

```
tinyml-modelzoo/
├── examples/               # Ready-to-run example configurations
├── tinyml_modelzoo/
│   ├── models/             # Neural network model definitions
│   ├── model_descriptions/ # Model metadata for GUI integration
│   └── device_info/        # Target device performance data
├── run_tinyml_modelzoo.sh  # Training wrapper (Linux)
├── run_tinyml_modelzoo.bat # Training wrapper (Windows)
└── ADDING_NEW_MODELS.md    # Guide for adding custom models
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Supported Target Devices](#supported-target-devices)
- [Supported Task Categories](#supported-task-categories)
- [Choosing an Example](#choosing-an-example)
- [Examples Reference](#examples-reference)
  - [Classification](#classification)
  - [Regression](#regression)
  - [Forecasting](#forecasting)
  - [Anomaly Detection](#anomaly-detection)
  - [Audio Classification](#audio-classification)
  - [Image Classification](#image-classification)
  - [Radar Point Cloud Classification](#radar-point-cloud-classification)
- [Available Models](#available-models)
- [Adding New Models](#adding-new-models)
- [About the Task Types](#about-the-task-types)
- [Release History](#release-history)
- [Additional Resources](#additional-resources)
- [License](#license)

---

## Quick Start

### Prerequisites

1. Python 3.10 environment
2. Clone **only** this repository, then install it:
   ```bash
   cd tinyml-modelzoo
   pip install -e .
   ```
   This pulls in the rest of the toolchain as prebuilt wheels automatically -
   no need to clone anything else.

### Running an Example

**Linux:**
```bash
cd tinyml-modelzoo
./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml
```

**Windows (cmd):**
```bat
cd tinyml-modelzoo
run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml
```

**Windows (PowerShell):**
```powershell
cd tinyml-modelzoo
./run_tinyml_modelzoo.ps1 examples/generic_timeseries_classification/config.yaml
```

### What Happens When You Run an Example?

1. **Dataset Download** - the required dataset is downloaded if not already present
2. **Data Processing** - feature extraction and preprocessing are applied
3. **Model Training** - the neural network is trained on your data
4. **Quantization** - the model is optimized for MCU deployment
5. **Compilation** - TI's Neural Network Compiler generates device-ready code

Output artifacts are saved to `./data/projects/<project_name>/`, relative to
whichever directory you ran `run_tinyml_modelzoo.sh`/`.bat`/`.ps1` from. To use
a different location, add this to the config's `common` section:
```yaml
common:
    projects_path: './your/choice'  # or an absolute path
```

---

## Supported Target Devices

### AM13 (Arm Cortex-M33)

* Devices with **TinyEngine NPU** (hardware accelerator): AM13E2

### MSPM0 (Arm Cortex-M0+)

* Devices with **TinyEngine NPU** (hardware accelerator): MSPM0G5187
* MSPM0G3507, MSPM0G3519

### Connectivity (Arm Cortex-M33/M4)

* CC2755, CC2745, CC1312, CC1314, CC1352, CC1354, CC35X1

### C2000 (C28 DSP)

* Devices with **TinyEngine NPU** (hardware accelerator): F28P55
* F2837, F2837xS, F2838x, F28P551x, F28003, F28004, F2807x, F28002x, F280013, F280015, F28E12, F28P65

### C2000 (C29 DSP)

* F29H85, F29P58, F29P32

### MSPM33C (Arm Cortex-M33)

* MSPM33C32, MSPM33C34

### AM26x (Arm Cortex-R5)

* AM263, AM263P, AM261

### Radar (Arm Cortex-M4)

* IWRL6432

---

## Supported Task Categories

| Task Category | Description | Use Cases |
|----------------|-------------|-----------|
| **Time Series Classification** | Categorize time-series data into discrete classes | Fault detection, activity recognition, anomaly classification |
| **Time Series Regression** | Predict continuous values from time-series inputs | Torque estimation, speed prediction, load measurement |
| **Time Series Forecasting** | Predict future values based on historical patterns | Temperature prediction, demand forecasting |
| **Time Series Anomaly Detection** | Identify abnormal patterns using autoencoder-based models | Equipment health monitoring, predictive maintenance |
| **Audio Classification** | Classify audio signals from MFCC features | Keyword spotting, voice commands, sound event detection |
| **Image Classification** | Categorize images into classes | Visual inspection, object recognition |
| **Radar Point Cloud Classification** | Classify point-cloud frames from radar sensors | Human pose detection, fall detection |

For the reasoning behind how these categories differ from one another, see
[About the Task Types](#about-the-task-types).

---

## Choosing an Example

1. **Look for your use case** in the [Examples Reference](#examples-reference) tables below. If one matches (e.g. `motor_bearing_fault`, `pir_detection`), start there — it ships with a dataset, a tuned model, and a device-specific config.
2. **If nothing matches**, use the generic example for your task type instead (the first row in each table below, e.g. [generic_timeseries_classification](examples/generic_timeseries_classification/)) and point it at your own dataset. This is also the recommended **first example to run** to learn the toolchain.

---

## Examples Reference

Each example links to its config directory under `examples/`. The first row
in each table is the **generic** example (a `generic_timeseries_*` config,
meant to be adapted to your own dataset); every other row is a **dedicated**,
purpose-built config for that specific use case.

### Classification

| Example | Data Type | Description |
|---------|-----------|--------------|
| [generic_timeseries_classification](examples/generic_timeseries_classification/) | — | Classify sine/square/sawtooth waveforms. **Start here** to learn the toolchain. |
| [dc_arc_fault](examples/dc_arc_fault/) | Current | Detect DC arc faults from current waveforms for electrical safety. |
| [ac_arc_fault](examples/ac_arc_fault/) | Current | Detect AC arc faults in electrical systems. |
| [motor_bearing_fault](examples/motor_bearing_fault/) | Vibration | Classify 5 bearing fault types + normal operation from vibration data. |
| [blower_imbalance](examples/blower_imbalance/) | Current | Detect blade imbalance in HVAC blowers using 3-phase motor currents. |
| [fan_blade_fault_classification](examples/fan_blade_fault_classification/) | Accelerometer | Detect faults in BLDC fans from accelerometer data. |
| [gearbox_fault_detection](examples/gearbox_fault_detection/) | Vibration | Classify gearbox operating conditions (healthy vs broken tooth) from vibration data. |
| [grid_fault_detection](examples/grid_fault_detection/) | Current | Detect electrical grid faults from sensor data. |
| [ecg_classification](examples/ecg_classification/) | ECG | Classify normal vs anomalous heartbeats from ECG signals. |
| [pir_detection](examples/pir_detection/) | PIR | Detect presence/motion using PIR sensor data. |
| [fall_detection_classification](examples/fall_detection_classification/) | Accelerometer | Detect and classify Human Fall vs Activities of Daily Living (ADL). |
| [dynamic_hand_gesture_recognition](examples/dynamic_hand_gesture_recognition/) | Accelerometer | Classify 4 dynamic hand gestures (circle, wave, tap, other) from 3-axis accelerometer data. |
| [electrical_fault](examples/electrical_fault/) | Voltage/Current | Classify transmission line faults using voltage and current (2-class and 6-class variants). |
| [grid_stability](examples/grid_stability/) | Simulated grid parameters | Predict power grid stability from node parameters. |
| [gas_sensor](examples/gas_sensor/) | Gas sensor array | Identify gas type and concentration from sensor array data. |
| [branched_model_parameters](examples/branched_model_parameters/) | Accelerometer/Gyroscope | Human Activity Recognition from accelerometer/gyroscope data. |
| [nilm_appliance_usage_classification](examples/nilm_appliance_usage_classification/) | Voltage/Current | Non-Intrusive Load Monitoring - identify active appliances. |
| [PLAID_nilm_classification](examples/PLAID_nilm_classification/) | Voltage/Current | Appliance identification using the PLAID dataset. |

### Regression

| Example | Data Type | Description |
|---------|-----------|--------------|
| [generic_timeseries_regression](examples/generic_timeseries_regression/) | — | Generic regression example for continuous value prediction. |
| [mosfet_temp_prediction](examples/mosfet_temp_prediction/) | Temperature/Power | Predict MOSFET temperature from electrical parameters. |
| [torque_measurement_regression](examples/torque_measurement_regression/) | Voltage/Current/Speed/Temperature | Predict PMSM motor torque from current measurements. |
| [induction_motor_speed_prediction](examples/induction_motor_speed_prediction/) | Voltage/Current | Predict induction motor speed from electrical signals. |
| [reg_washing_machine](examples/reg_washing_machine/) | Voltage/Current/Speed | Predict washing machine load weight. |

### Forecasting

| Example | Data Type | Description |
|---------|-----------|--------------|
| [generic_timeseries_forecasting](examples/generic_timeseries_forecasting/) | — | Generic forecasting example for time series prediction. |
| [forecasting_pmsm_rotor_temp](examples/forecasting_pmsm_rotor_temp/) | Voltage/Current | Forecast PMSM rotor winding temperature. |
| [hvac_indoor_temp_forecast](examples/hvac_indoor_temp_forecast/) | Temperature | Predict indoor temperature for HVAC control. |

### Anomaly Detection

| Example | Data Type | Description |
|---------|-----------|--------------|
| [generic_timeseries_anomalydetection](examples/generic_timeseries_anomalydetection/) | — | Generic anomaly detection example using autoencoders. |
| [dc_arc_fault (DSI)](examples/dc_arc_fault/config_anomaly_detection_dsi.yaml) | Current | Detect anomalous DC arc patterns using autoencoder (DSI dataset). |
| [dc_arc_fault (DSK)](examples/dc_arc_fault/config_anomaly_detection_dsk.yaml) | Current | Detect anomalous DC arc patterns using autoencoder (DSK dataset). |
| [ecg_classification](examples/ecg_classification/config_anomaly_detection.yaml) | ECG | Detect anomalous heartbeat patterns from ECG signals. |
| [fan_blade_fault_classification](examples/fan_blade_fault_classification/config_anomaly_detection.yaml) | Accelerometer | Detect anomalous fan blade behavior from accelerometer data. |
| [motor_bearing_fault](examples/motor_bearing_fault/config_anomaly_detection.yaml) | Vibration | Detect anomalous bearing behavior from vibration data. |

### Audio Classification

| Example | Data Type | Description |
|---------|-----------|--------------|
| [google_speech_command](examples/google_speech_command/) | Audio | 12-class keyword spotting from audio using MFCC + DSCNN model. |

### Image Classification

| Example | Data Type | Description |
|---------|-----------|--------------|
| [MNIST_image_classification](examples/MNIST_image_classification/) | Image | Handwritten digit recognition (MNIST dataset). |
| [machine_readable_code_classification](examples/machine_readable_code_classification/) | Image | Classify QR codes, barcodes, and other symbols (28×28 images). |
| [coffee_bean_classification](examples/coffee_bean_classification/) | Image | Classify coffee bean quality from images. |

### Radar Point Cloud Classification

| Example | Data Type | Description |
|---------|-----------|--------------|
| [radar_point_cloud_classification](examples/radar_point_cloud_classification/) | Radar point cloud | Human pose and fall detection from radar point-cloud frames. |

---

## Available Models

Models are organized by task type. The **NPU** column indicates hardware acceleration support on TI devices with NPU (F28P55, AM13E2, MSPM0G5187).

**NPU-optimized models** follow specific layer constraints for hardware acceleration:
- All channels are multiples of 4 (m4)
- Kernel heights ≤ 7 for GCONV layers
- MaxPool kernels ≤ 4
- FC layer inputs ≥ 16 features (8-bit) or ≥ 8 features (4-bit)

For detailed guidelines, see [NPU Configuration Guidelines](docs/NPU_CONFIGURATION_GUIDELINES.md).

**When to use NPU-optimized models:**
- Target device has NPU (F28P55, AM13E2, MSPM0G5187)
- You need maximum inference speed
- Standard models show "fallback to software" warnings during compilation

### Classification Models

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `CLS_100_NPU` | ~100 | CNN | Yes | Ultra-compact model |
| `CLS_500_NPU` | ~500 | CNN | Yes | Compact model |
| `CLS_1k_NPU` | ~1K | CNN | Yes | Lightweight 2-layer CNN |
| `CLS_1.2k_NPU` | ~1.2K | CNN | Yes | Compact model for ultra-low power devices |
| `CLS_1.5k_NPU` | ~1.5K | CNN | Yes | 3-layer model with balanced performance |
| `CLS_1.9k_NPU` | ~1.9K | CNN | Yes | Efficient 3-layer model |
| `CLS_2k_NPU` | ~2K | CNN | Yes | 2-layer model |
| `CLS_2.8k_NPU` | ~2.8K | CNN | Yes | Improved accuracy with compact footprint |
| `CLS_3.1k_NPU` | ~3.1K | CNN | Yes | Higher accuracy model |
| `CLS_ResAdd_3k` | ~3K | ResNet (Add) | No | Residual connections with addition |
| `CLS_ResCat_3k` | ~3K | ResNet (Cat) | No | Residual connections with concatenation |
| `CLS_3.9k_NPU` | ~3.9K | CNN | Yes | Advanced 3-layer model |
| `CLS_4k_NPU` | ~4K | CNN | Yes | Balanced model |
| `CLS_4.2k_NPU` | ~4.2K | CNN | Yes | Optimized 4-layer model |
| `CLS_5k_NPU` | ~5K | CNN | Yes | Mid-range model |
| `CLS_6k_NPU` | ~6K | CNN (DW-Sep) | Yes | Depthwise separable |
| `CLS_8k_NPU` | ~8K | CNN (DW-Sep) | Yes | Depthwise separable |
| `CLS_13k_NPU` | ~13K | CNN | Yes | Higher capacity |
| `CLS_20k_NPU` | ~20K | CNN | Yes | High capacity |
| `CLS_40k_NPU` | ~40K | CNN | Yes | Advanced model for complex tasks |
| `CLS_55k_NPU` | ~55K | CNN | Yes | Maximum accuracy |
| `ArcFault_model_200_t` | ~200 | Specialized | No | Arc fault detection |
| `ArcFault_model_300_t` | ~300 | Specialized | No | Arc fault with more capacity |
| `ArcFault_model_700_t` | ~700 | Specialized | No | Arc fault medium model |
| `ArcFault_model_1400_t` | ~1.4K | Specialized | No | Arc fault high accuracy |
| `GearboxFault_model_1.2k_t` | ~1.2K | CNN | Yes | Gearbox fault detection |
| `GearboxFault_model_1.5k_t` | ~1.5K | CNN | Yes | Gearbox fault with more capacity |
| `MotorFault_model_1_t` | Varies | Specialized | No | Motor bearing fault detection |
| `MotorFault_model_2_t` | Varies | Specialized | No | Motor fault variant 2 |
| `MotorFault_model_3_t` | Varies | Specialized | No | Motor fault variant 3 |
| `FanImbalance_model_1_t` | Varies | Specialized | No | Fan blade imbalance detection |
| `FanImbalance_model_2_t` | Varies | Specialized | No | Fan imbalance variant 2 |
| `FanImbalance_model_3_t` | Varies | Specialized | No | Fan imbalance variant 3 |
| `PIRDetection_model_1_t` | Varies | Specialized | No | PIR-based presence detection |

### Regression Models

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `REGR_500_NPU` | ~500 | CNN | Yes | Compact regression |
| `REGR_1k` | ~1K | CNN | No | Lightweight regression model |
| `REGR_2k_NPU` | ~2K | CNN | Yes | 2-layer model |
| `REGR_3k` | ~3K | MLP | No | 4-layer fully connected network |
| `REGR_4k` | ~4K | CNN | No | 2 Conv+BN+ReLU + Linear |
| `REGR_6k_NPU` | ~6K | CNN (DW-Sep) | Yes | Depthwise separable convolutions |
| `REGR_8k_NPU` | ~8K | CNN | Yes | 3-layer model |
| `REGR_10k` | ~10K | CNN | No | 3 Conv+BN+ReLU + 2 Linear |
| `REGR_13k` | ~13K | CNN | No | High capacity regression |
| `REGR_20k_NPU` | ~20K | CNN | Yes | High capacity with MaxPool |

### Anomaly Detection Models

Note: For NPU models, encoder convolutions are NPU-accelerated but decoder upsampling falls back to CPU.

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `AD_500_NPU` | ~500 | CNN AE | Yes | 2-layer autoencoder |
| `AD_1k` | ~1K | Autoencoder | No | Compact autoencoder |
| `AD_2k_NPU` | ~2K | CNN AE | Yes | 2-layer autoencoder |
| `AD_4k` | ~4K | Autoencoder | No | 3-layer CNN autoencoder |
| `AD_6k_NPU` | ~6K | CNN AE (DW-Sep) | Yes | Depthwise separable encoder |
| `AD_8k_NPU` | ~8K | CNN AE | Yes | 3-layer autoencoder |
| `AD_10k_NPU` | ~10K | CNN AE | Yes | 3-layer autoencoder |
| `AD_16k` | ~16K | Autoencoder | No | 4-layer CNN autoencoder |
| `AD_17k` | ~17K | Autoencoder | No | Fan blade anomaly detection |
| `AD_20k_NPU` | ~20K | CNN AE | Yes | High capacity autoencoder |
| `AD_Linear` | Varies | Linear AE | No | 3-layer deep linear autoencoder |
| `Ondevice_Trainable_AD_Linear` | Varies | Linear AE | No | On-device trainable variant |

### Forecasting Models

Note: LSTM models are not NPU-supported.

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `FCST_500_NPU` | ~500 | CNN | Yes | Compact forecasting |
| `FCST_1k_NPU` | ~1K | CNN | Yes | 2-layer model |
| `FCST_2k_NPU` | ~2K | CNN | Yes | 2-layer model |
| `FCST_3k` | ~3K | MLP | No | 4-layer fully connected |
| `FCST_4k_NPU` | ~4K | CNN | Yes | 3-layer model |
| `FCST_6k_NPU` | ~6K | CNN (DW-Sep) | Yes | Depthwise separable convolutions |
| `FCST_8k_NPU` | ~8K | CNN | Yes | 3-layer model |
| `FCST_10k_NPU` | ~10K | CNN | Yes | 3-layer model |
| `FCST_13k` | ~13K | CNN | No | 2 Conv+BN+ReLU + Linear |
| `FCST_20k_NPU` | ~20K | CNN | Yes | High capacity with MaxPool |
| `FCST_LSTM8` | Varies | LSTM | No | Single LSTM (hidden=8) + Linear |
| `FCST_LSTM10` | Varies | LSTM | No | Single LSTM (hidden=10) + Linear |

### Audio Classification Models

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `DSCNN_NPU` | ~9K | DSCNN | Yes | Depthwise separable CNN for keyword spotting; input (1, 49, 10) MFCC |

### Image Classification Models

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `Lenet5` | ~60K | LeNet-5 | No | Classic CNN for image classification |
| `MobileNetV1_58k_NPU` | ~58K | MobileNetV1-style DW-Sep | Yes | Compact NPU-optimized image classifier |
| `MobileNetV2_58k_NPU` | ~58K | MobileNetV2-style DW-Sep | Yes | Inverted residual image classifier |

### Radar Point Cloud Classification Models

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `Pose_and_Fall_model` | Varies | Linear (4-layer) | No | Human pose and fall detection from radar point-cloud data |

---

## Adding New Models

Want to add your own model? See the comprehensive guide: **[ADDING_NEW_MODELS.md](ADDING_NEW_MODELS.md)**

Key steps:
1. Add model class to `tinyml_modelzoo/models/`
2. Add class name to the file's `__all__` list
3. (Optional) Add device performance info to `device_info/run_info.py`
4. (Optional) Add model description to `model_descriptions/` for GUI integration

No changes required outside this repo.

---

## About the Task Types

**Classification** outputs a probability distribution over predefined classes. Best for: "Is this an A fault, B fault, or C fault?", "Which type of activity is this?"

**Regression** outputs a continuous numerical value. Best for: "What is the current torque?", "What will the temperature be?"

**Forecasting** predicts future values in a time series. Best for: "What will happen next?"

**Anomaly Detection** uses autoencoders to learn "normal" patterns; reconstruction error indicates anomalies. Best for: "Is this behavior normal?"

**Audio Classification** extracts MFCC features from a fixed-length audio window and classifies into keyword or sound categories. Best for: "What keyword was spoken?", "What sound event occurred?"

These categories can look similar from a distance, so here's how to tell them apart:

- **Anomaly Detection vs. Classification** — "Is it normal, or an anomaly?" is anomaly detection (binary outcome). "Is it normal, anomaly type A, type B, or type C?" is classification (multiple categories).
- **Classification vs. Regression** — predicting a **discrete** target (Class A / B / C, ...) from independent variables is classification; predicting a **continuous** target is regression.
- **Regression vs. Forecasting** — predicting a continuous target **Y** at the **same** time instant as its inputs is regression; predicting a variable's value at a **future** time instant is forecasting.

---

## Release History

Until **1.4.0** (2026-Jun), this history lived in the [tinyml-tensorlab](https://github.com/TexasInstruments/tinyml-tensorlab)
repository on GitHub; that repository's source is now private. Starting with
**1.5.0** (2026-Sep), `tinyml-modelzoo` (this repo) is the standalone, pip-installable
entry point to TI's MCU AI flow — a simplified, easy-to-install replacement
for what previously required cloning `tinyml-tensorlab`'s full set of
component repos.

- [2026-Sep] Release version 1.5.0 of the software
  <details>

  - Device Support:
    - Added AM13E2 support for vision classification and radar tasks
    - Added fel_memory support (feature-extraction library) for AM13 and C28x (F280013x, F280015x, F28002x) devices — enables on-device RAM/Flash estimation
  - Models:
    - Added TCDSResNet and a compressed DSCNN for audio classification (cough detection), both NPU-compliant
    - Compressed MobileNet_v1 to fit AM13 memory budget
    - Added Radar Point Cloud Classification support (new model + dataset flow across ModelMaker/TinyVerse/ModelZoo)
  - Flows:
    - ModelMaker can now report estimated RAM & Flash usage for Feature Extraction before training
    - Support for preprocessing and publicly-available datasets in audio/radar flows
    - NAS: can now accept a NAS model id directly
  - Model Optimization:
    - Residual connection support added to quantization (TINPU)
    - Replaced qconfig_dict with a cleaner API to toggle auto-quantization on/off
    - Hessian-aware auto-quantization bug fixes
  - Reliability & Compatibility:
    - Python 3.14 / PyTorch 2.13 compatibility across ModelMaker, ModelOptimization, TinyVerse
    - macOS ARM64 (MPS/Apple Silicon) compatibility fixes across training, evaluation, and quantization
    - torch.compile safety: unwraps compiled models correctly before ONNX/checkpoint export, falls back to eager on failure
    - Security/robustness hardening: safe checkpoint deserialization, safe YAML loading, safer cache-dataset handling
    - Training performance: torch.compile, AMP, persistent workers, more efficient eval loop
    - Large expansion of automated test coverage (functional test tiers, pytest suites) across all repos
  - Packaging:
    - Added build_wheels.sh to build TinyVerse/ModelOptimization/ModelMaker wheels from local source
    - ModelZoo and TinyVerse examples now runnable standalone via published ModelMaker wheel dependency (TI official wheel CDN)
    - `tinyml-tensorlab` deprecated as the public entry point; `tinyml-modelzoo` takes over as the standalone, pip-installable way to use TI's MCU AI flow
  - Documentation:
    - DEVICE_TASK_SUPPORT.md and NPU_CONFIGURATION_GUIDELINES.md updated
    - New how-to: publishing shared wheels
  - Special Acknowledgement:
    - Shoutout to [@musicalplatypus](https://github.com/musicalplatypus) for contributing towards a better, neater and more feature-rich toolchain by their additions such as full macOS/Apple Silicon (MPS) support, torch.compile+AMP training-performance optimizations, and NAS bug fixes. They also hardened the codebase with security fixes for unsafe deserialization/YAML loading, overhauled CI so tests actually run across all four packages, and expanded the test suite and architecture docs.

  </details>
- [2026-Jun] Release version 1.4.0 of the software
  <details>

    - Agent Skills with Claude Code supported for users to solve Edge AI/Tiny ML problems using natural language!
    - Device Support: 40 MCU devices supported:
      - AM1x: AM13E2
      - C2000 F28: F280013, F280015, F28003, F28004, F2837, F28P55, F28P65
      - C2000 F29: F29H85, F29P58, F29P32
      - MSP M0: MSPM0G3507, MSPM0G3519, MSPM0G5187
      - MSP M33: MSPM33C32,
      - Connectivity: CC2755, CC1352, CC1354, CC35X1, CC1312, CC1314
      - AM26x: AM263, AM263P, AM261
  - Flows:
    - Timeseries Anomaly Detection flow - More models
    - On Device Learning Mode Enabled - Expansive functionalities
  - Applications Supported
    - 31 (4 generic + 27 specific applications)
  - Models:
    - 50+ generic models added over classification, regression, forecasting and anomaly detection tasks.
  - Model Optimization:
    - Hessian Aware Quantization for automatic recommendation of quantization bitwidths for weights.
  - Compilation:
      - Upgraded TI MCU Neural Network Compiler for MCUs to 2.1.2

  </details>
- [2026-Feb] Release version 1.3.0 of the software
  <details>

  - Device Support: 22 MCU devices supported:
    - AM1x: AM13E2
    - C2000 F28: F280013, F280015, F28003, F28004, F2837, F28P55, F28P65
    - C2000 F29: F29H85, F29P58, F29P32
    - MSP M0: MSPM0G3507, MSPM0G3519, MSPM0G5187
    - MSP M33: MSPM33C32,
    - Connectivity: CC2755, CC1352, CC1354, CC35X1,
    - AM26x: AM263, AM263P, AM261
  - Flows:
    - Timeseries Anomaly Detection flow supported
    - On Device Learning Mode Enabled
  - Applications Supported
    - 22 (4 generic + 18 specific applications)
  - Models:
    - 50+ generic models added over classification, regression, forecasting and anomaly detection tasks.
  - Model Optimization:
    - Partial Quantization Supported to enable best of precision and latency for regression models.
  - Compilation:
      - Upgraded TI MCU Neural Network Compiler for MCUs to 2.1.1 LTS

  </details>
- [2025-Nov] Release version 1.2.0 of the software

    <details>

    - Device Support:
      - Added MSPM0 based MCUs: MSPM0G3507, MSPM0G5187
      - Added Connectivity device: CC2745R10-Q1, CC2755R10
    - General:
      - Supports simple gain augmentation for classification tasks
      - Prints dataset file level confusion matrix for classification tasks
      - Golden Test Vectors for Regression tasks
      - Run modelmaker with only the config, no more target device required in the input.
    - Flows:
      - Timeseries Forecasting flows supported
      - L1, L2 normalization can be enabled in regression using lambda_reg param
    - Model Optimization:
      - How to use: Documentation updated.
      - Example code for performing regression in modeloptimization
      - Fixing clipping of input data to int8 or uint8 based on dataset (zero_point) (only the input zero point is fixed and not the intermediate layers)
      - BatchNorm is supported by GENERIC quantization
      - Experimental features like additional QDQ at input of model and floating bias can be enabled individually
      - Residual Add supported for different scales, zero points, but not optimised for TINPU
    - Compilation:
      - Upgraded TI MCU Neural Network Compiler for MCUs to 2.1.0 LTS
      - Supported all layer configs with 8-bit activations and 8-/4-/2-bit weights that can be offloaded to TI-NPU
      - Supported all layer configs with 8-bit activations and 8-bit weights that can be accelerated using the M33 Custom Datapath Extension (CDE).

  </details>
- [2025-Aug] Release version 1.1.0 of the software
  <details>

  - General:
    - Generic Timeseries Classification is available with fixed point reference dataset.
    - Compatible with C2000Ware 6.0.0
  - Model Optimization:
    - Aggressive Quantization Modes for Weights & Activation: 2W8A, 4W4A, 4W8A --> massive speedup and memory saved
    - Neural network Architecture Search for generating a TINPU compatible model directly based on user's dataset
  - Dataset:
    - Dataset can be split into train-test-val on a file-by-file basis or within-a-file basis
  - Device Support:
    - Full Support for F280013x
    - Preliminary Support for F29H85x and MSPM0G3507x
  - Compilation:
    - Upgraded TI MCU Neural Network Compiler for MCUs to 2.0.0
  - Windows Platform Specific:
    - Major quantization accuracy improvements
  - Miscellaneous:
    - Fixed model performance data that appears on the terminal when a training is initiated
    - Added Model Descriptions for all models
    - Setup of the repos is now smoother and cleaner

  </details>
- [2025-Apr] Major feature updates (version 1.0.0) of the software
  <details>

  - General:
    - Tiny ML Modelmaker is now a pip installable package!
    - Existing models can be modified on the fly through a config file (check Tiny ML Modelmaker docs)
    - MPS (Metal Performance Shaders) backend support for Mac host devices!
  - Technology:
    - PTQ and QAT flows supported in tinyml-modelmaker, tinyml-modeloptimization
    - Ternary, 4 bit Quantization support in tinyml-modelmaker
  - Flows:
    - Regression ML tasks supported
    - Autoencoder based Anomaly Detection task supported
  - Feature Extraction:
    - Feature Extraction transforms are now modular and compatible with C2000Ware 5.05 only
    - Supports Haar and Hadamard Transform
    - Golden test vectors file has one set uncommented by default to work OOB
  - Data Visualisation:
    - Multiclass ROC-AUC graphs are autogenerated for better explainability of reports and help select thresholds based on false alarm/ sensitivity preference
    - PCA graphs are auto plotted for feature extracted data - Helps in identifying if the feature extraction actually helped
    - Run now begins with displaying inference time, sram usage and flash usage for all the devices for any model.
  - Dataset
    - Goodness of Fit of dataset now enabled.
  - Extensive Documentation & Know-How Examples to use Modelmaker

  </details>
- [2024-November] Updated (version 0.9.0) of the software
- [2024-August] Release version 0.8.0 of the software
- [2024-July] Release version 0.7.0 of the software
- [2024-June] Release version 0.6.0 of the software
- [2024-May] First public release (version 0.5.0) of the software

---

## Additional Resources

- [TI's Neural Network Compiler Documentation](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/)
- [NPU Configuration Guidelines](docs/NPU_CONFIGURATION_GUIDELINES.md) - Design models optimized for TI NPU acceleration
- [Edge AI Studio for MCUs](https://www.ti.com/tool/download/EDGE-AI-STUDIO-MCU/) - No-code GUI for data collection & model development

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

