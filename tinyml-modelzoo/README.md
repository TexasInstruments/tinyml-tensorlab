# TinyML ModelZoo

Welcome to the **TinyML ModelZoo** - Texas Instruments' central repository for AI models, examples, and configurations for microcontroller (MCU) applications.

---

## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Example Applications](#example-applications)
- [Supported Task Categories](#supported-task-categories)
- [Available Models](#available-models)
- [Adding New Models](#adding-new-models)
- [Additional Resources](#additional-resources)

---

## Introduction

### Texas Instruments MCU AI Toolchain

Texas Instruments provides a comprehensive toolchain for developing, training, and deploying machine learning models on resource-constrained microcontrollers. The toolchain consists of three main components:

| Component                       | Purpose                                | Who Should Use It                               |
|---------------------------------|----------------------------------------|-------------------------------------------------|
| **tinyml-modelzoo** (this repo) | Models, examples, and configurations   | **End customers** - This is your starting point |
| tinyml-modelmaker               | Training orchestration and compilation | Developers extending the toolchain              |
| tinyml-tinyverse                | Core training scripts and utilities    | Advanced developers only                        |
| tinyml-modeloptimization        | Quantization scripts and utilities     | Advanced developers only                        |

**As an end customer, you only need to work with `tinyml-modelzoo`.** The example configurations here will automatically use the underlying toolchain components to train and compile models for your target MCU.

### What's in ModelZoo?

```
tinyml-modelzoo/
├── examples/              # Ready-to-run example configurations
├── tinyml_modelzoo/
│   ├── models/            # Neural network model definitions
│   ├── model_descriptions/ # Model metadata for GUI integration
│   └── device_info/       # Target device performance data
├── run_tinyml_modelzoo.sh        # Training wrapper (Linux)
├── run_tinyml_modelzoo.bat       # Training wrapper (Windows)
└── ADDING_NEW_MODELS.md   # Guide for adding custom models
```

---

## Quick Start

### Prerequisites

1. Python 3.10 environment with the TinyML toolchain installed
2. Clone the [tinyml-tensorlab](https://github.com/TexasInstruments/tinyml-tensorlab) repository

### Running an Example

**Linux:**
```bash
# Activate your Python environment
source ~/.pyenv/versions/py310_tinyml/bin/activate

# Navigate to modelzoo
cd tinyml-modelzoo

# Run an example (e.g., hello_world)
./run_tinyml_modelzoo.sh examples/hello_world/config.yaml
```

**Windows:**
```powershell
# Navigate to modelzoo
cd tinyml-modelzoo

# Run an example
run_tinyml_modelzoo.bat examples\hello_world\config.yaml
```

### What Happens When You Run an Example?

1. **Dataset Download** - The toolchain downloads the required dataset (if not already present)
2. **Data Processing** - Feature extraction and preprocessing are applied
3. **Model Training** - The neural network is trained on your data
4. **Quantization** - The model is optimized for MCU deployment
5. **Compilation** - TI's Neural Network Compiler generates device-ready code

Output artifacts are saved to `./data/projects/<project_name>/`.

---

## Example Applications

The following ready-to-use examples demonstrate various AI applications for MCUs, organized by task type.

### Classification Examples

| No. | Example                                                                              | Data Type    | Description                                                                     |
|-----|--------------------------------------------------------------------------------------|--------------|---------------------------------------------------------------------------------|
| 1   | [hello_world](examples/hello_world/)                                                 | Univariate   | Classify sine/square/sawtooth waveforms. **Start here** to learn the toolchain. |
| 2   | [dc_arc_fault](examples/dc_arc_fault/)                                               | Univariate   | Detect DC arc faults from current waveforms for electrical safety.              |
| 3   | [ac_arc_fault](examples/ac_arc_fault/)                                               | Univariate   | Detect AC arc faults in electrical systems.                                     |
| 4   | [motor_bearing_fault](examples/motor_bearing_fault/)                                 | Multivariate | Classify 5 bearing fault types + normal operation from vibration data.          |
| 5   | [blower_imbalance](examples/blower_imbalance/)                                       | Multivariate | Detect blade imbalance in HVAC blowers using 3-phase motor currents.            |
| 6   | [fan_blade_fault_classification](examples/fan_blade_fault_classification/)           | Multivariate | Detect faults in BLDC fans from accelerometer data.                             |
| 7   | [electrical_fault](examples/electrical_fault/)                                       | Multivariate | Classify transmission line faults using voltage and current.                    |
| 8   | [grid_stability](examples/grid_stability/)                                           | Multivariate | Predict power grid stability from node parameters.                              |
| 9   | [gas_sensor](examples/gas_sensor/)                                                   | Multivariate | Identify gas type and concentration from sensor array data.                     |
| 10  | [branched_model_parameters](examples/branched_model_parameters/)                     | Multivariate | Human Activity Recognition from accelerometer/gyroscope data.                   |
| 11  | [ecg_classification](examples/ecg_classification/)                                   | Multivariate | Classify normal vs anomalous heartbeats from ECG signals.                       |
| 12  | [nilm_appliance_usage_classification](examples/nilm_appliance_usage_classification/) | Multivariate | Non-Intrusive Load Monitoring - identify active appliances.                     |
| 13  | [PLAID_nilm_classification](examples/PLAID_nilm_classification/)                     | Multivariate | Appliance identification using the PLAID dataset.                               |
| 14  | [pir_detection](examples/pir_detection/)                                             | Multivariate | Detect presence/motion using PIR sensor data.                                   |

### Regression Examples

| No. | Example | Data Type | Description |
|-----|---------|-----------|-------------|
| 1 | [torque_measurement_regression](examples/torque_measurement_regression/) | Multivariate | Predict PMSM motor torque from current measurements. |
| 2 | [induction_motor_speed_prediction](examples/induction_motor_speed_prediction/) | Multivariate | Predict induction motor speed from electrical signals. |
| 3 | [reg_washing_machine](examples/reg_washing_machine/) | Multivariate | Predict washing machine load weight. |

### Anomaly Detection Examples

| No. | Example | Data Type | Description |
|-----|---------|-----------|-------------|
| 1 | [dc_arc_fault (DSI)](examples/dc_arc_fault/config_anomaly_detection_dsi.yaml) | Univariate | Detect anomalous DC arc patterns using autoencoder (DSI dataset). |
| 2 | [dc_arc_fault (DSK)](examples/dc_arc_fault/config_anomaly_detection_dsk.yaml) | Univariate | Detect anomalous DC arc patterns using autoencoder (DSK dataset). |
| 3 | [ecg_classification](examples/ecg_classification/config_anomaly_detection.yaml) | Multivariate | Detect anomalous heartbeat patterns from ECG signals. |
| 4 | [fan_blade_fault_classification](examples/fan_blade_fault_classification/config_anomaly_detection.yaml) | Multivariate | Detect anomalous fan blade behavior from accelerometer data. |
| 5 | [motor_bearing_fault](examples/motor_bearing_fault/config_anomaly_detection.yaml) | Multivariate | Detect anomalous bearing behavior from vibration data. |

### Forecasting Examples

| No. | Example | Data Type | Description |
|-----|---------|-----------|-------------|
| 1 | [forecasting_pmsm_rotor](examples/forecasting_pmsm_rotor/) | Multivariate | Forecast PMSM rotor winding temperature. |
| 2 | [hvac_indoor_temp_forecast](examples/hvac_indoor_temp_forecast/) | Multivariate | Predict indoor temperature for HVAC control. |

### Image Classification Examples

| No. | Example | Data Type | Description |
|-----|---------|-----------|-------------|
| 1 | [MNIST_image_classification](examples/MNIST_image_classification/) | Image | Handwritten digit recognition (MNIST dataset). |

---

## Supported Task Categories

TinyML ModelZoo supports the following AI task categories:

| Task Category                     | Description                                               | Use Cases                                                     |
|-----------------------------------|-----------------------------------------------------------|---------------------------------------------------------------|
| **Time Series Classification**    | Categorize time-series data into discrete classes         | Fault detection, activity recognition, anomaly classification |
| **Time Series Regression**        | Predict continuous values from time-series inputs         | Torque estimation, speed prediction, load measurement         |
| **Time Series Forecasting**       | Predict future values based on historical patterns        | Temperature prediction, demand forecasting                    |
| **Time Series Anomaly Detection** | Identify abnormal patterns using autoencoder-based models | Equipment health monitoring, predictive maintenance           |
| **Image Classification**          | Categorize images into classes                            | Visual inspection, object recognition                         |

### Understanding Each Task

**Classification** - The model outputs a probability distribution over predefined classes. Best for: "Is this a A fault or B fault or C fault?", "Which type of activity is this?"

**Regression** - The model outputs a continuous numerical value. Best for: "What is the current torque?", "What will the temperature be?"

**Anomaly Detection** - Uses autoencoders to learn "normal" patterns. Reconstruction error indicates anomalies. Best for: "Is this behavior normal?"

**Forecasting** - Predicts future values in a time series. Best for: "What will happen next?"

---

## Available Models

Models are organized by task type. The **NPU** column indicates hardware acceleration support on TI devices with NPU (F28P55, F28P65, F29H85, F29P58, F29P32).

**NPU-optimized models** follow specific layer constraints for hardware acceleration:
- All channels are multiples of 4 (m4)
- Kernel heights ≤ 7 for GCONV layers
- MaxPool kernels ≤ 4
- FC layer inputs ≥ 16 features (8-bit) or ≥ 8 features (4-bit)

For detailed guidelines, see [NPU Configuration Guidelines](docs/NPU_CONFIGURATION_GUIDELINES.md).

**When to use NPU-optimized models:**
- Target device has NPU (F28P55, F28P65, F29H85, etc.)
- You need maximum inference speed
- Standard models show "fallback to software" warnings during compilation

### Classification Models

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `CLS_100_NPU` | ~100 | CNN | Yes | Ultra-compact model |
| `CLS_500_NPU` | ~500 | CNN | Yes | Compact model |
| `CLS_1k_NPU` | ~1K | CNN | Yes | Lightweight 2-layer CNN |
| `CLS_2k_NPU` | ~2K | CNN | Yes | 2-layer model |
| `CLS_ResAdd_3k` | ~3K | ResNet (Add) | No | Residual connections with addition |
| `CLS_ResCat_3k` | ~3K | ResNet (Cat) | No | Residual connections with concatenation |
| `CLS_4k_NPU` | ~4K | CNN | Yes | Balanced model |
| `CLS_6k_NPU` | ~6K | CNN (DW-Sep) | Yes | Depthwise separable |
| `CLS_8k_NPU` | ~8K | CNN (DW-Sep) | Yes | Depthwise separable |
| `CLS_13k_NPU` | ~13K | CNN | Yes | Higher capacity |
| `CLS_20k_NPU` | ~20K | CNN | Yes | High capacity |
| `CLS_55k_NPU` | ~55K | CNN | Yes | Maximum accuracy |
| `ArcFault_model_200_t` | ~200 | Specialized | No | Arc fault detection |
| `ArcFault_model_300_t` | ~300 | Specialized | No | Arc fault with more capacity |
| `ArcFault_model_700_t` | ~700 | Specialized | No | Arc fault medium model |
| `ArcFault_model_1400_t` | ~1.4K | Specialized | No | Arc fault high accuracy |
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

### Image Classification Models

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `Lenet5` | ~60K | LeNet-5 | No | Classic CNN for image classification |

---

## Adding New Models

Want to add your own model? See the comprehensive guide: **[ADDING_NEW_MODELS.md](ADDING_NEW_MODELS.md)**

Key steps:
1. Add model class to `tinyml_modelzoo/models/`
2. Add class name to the file's `__all__` list
3. (Optional) Add device performance info to `device_info/run_info.py`
4. (Optional) Add model description to `model_descriptions/` for GUI integration

**No changes required in tinyml-tinyverse or tinyml-modelmaker!**

---

## Additional Resources

- [TI's Neural Network Compiler Documentation](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/)
- [NPU Configuration Guidelines](docs/NPU_CONFIGURATION_GUIDELINES.md) - Design models optimized for TI NPU acceleration
- [Edge AI Studio Model Composer](https://dev.ti.com/modelcomposer/) - No-code GUI for model development
- [Understanding the Config File](../tinyml-modelmaker/docs/UnderstandingConfigFile.md)
- [Dataset Format Guide](../tinyml-modelmaker/docs/DatasetFormat_Timeseries_Classification.md)

---

## Supported Target Devices

| Device | NPU | Flash | SRAM | Notes |
|--------|-----|-------|------|-------|
| F28P55 | Yes | High | High | Recommended for complex models |
| F28P65 | Yes | High | High | NPU-accelerated inference |
| F2837 | No | Medium | Medium | General purpose MCU |
| F28003 | No | Low | Low | Cost-optimized |
| F28004 | No | Low | Low | Cost-optimized |
| MSPM0G3507 | No | Low | Low | Ultra-low power |

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

---

**Questions or Issues?** Open an issue on [GitHub](https://github.com/TexasInstruments/tinyml-tensorlab/issues).
