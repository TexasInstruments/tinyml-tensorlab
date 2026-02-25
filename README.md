# Tiny ML ModelZoo

Welcome to the **Tiny ML ModelZoo** - Texas Instruments' central repository for AI models, examples, and configurations for microcontroller (MCU) applications.

---

## Table of Contents

- [Introduction](#introduction)
- [Supported Target Devices](#supported-target-devices)
- [Quick Start](#quick-start)
- [Supported Task Categories](#supported-task-categories)
- [Example Applications](#example-applications)
  - [Generic Timeseries Applications](#generic-timeseries-applications)
  - [Application-Specific Examples](#application-specific-examples)
  - [Detailed Examples by Task Type](#detailed-examples-by-task-type)
- [Available Models](#available-models)
- [Adding New Models](#adding-new-models)
- [Additional Resources](#additional-resources)
- [License](#license)

---

## Introduction

### Texas Instruments MCU AI Toolchain

Texas Instruments provides a comprehensive toolchain for developing, training, and deploying machine learning models on resource-constrained microcontrollers. The toolchain consists of three main components:
<hr>

Detailed User Guide: [TI Tiny ML Tensorlab User Guide](https://software-dl.ti.com/C2000/esd/mcu_ai/user_guide/index.html)
<hr>

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

## Supported Target Devices

### C2000 DSP Family (Texas Instruments)

| Device | NPU | Description | Notes |
|--------|-----|-------------|-------|
| F28P55 | Yes | C2000 32-bit MCU | Recommended for complex models |
| F28P65 | No | C2000 32-bit MCU, 150 MHz | High performance |
| F29H85 | No | C2000 64-bit MCU with C29x core | High capacity |
| F29P58 | No | C2000 64-bit MCU with C29x core | High capacity |
| F29P32 | No | C2000 64-bit MCU with C29x core | High capacity |
| F2837 | No | C2000 32-bit dual-core MCU, 200 MHz | General purpose |
| F28003 | No | C2000 32-bit MCU, 100 MHz | Cost-optimized |
| F28004 | No | C2000 32-bit MCU, 100 MHz | Cost-optimized |
| F280013 | No | C2000 32-bit MCU, 100 MHz | Entry-level |
| F280015 | No | C2000 32-bit MCU, 120 MHz | Entry-level |

### MSPM0 Family (Arm Cortex-M0+)

| Device | NPU | Description | Notes |
|--------|-----|-------------|-------|
| MSPM0G3507 | No | 80 MHz Arm Cortex-M0+ | Ultra-low power, classification only |
| MSPM0G3519 | No | 80 MHz Arm Cortex-M0+ | Ultra-low power |
| MSPM0G5187 | Yes | 80 MHz Arm Cortex-M0+ | Ultra-low power, NPU-accelerated |

### MSPM33C Family (Arm Cortex-M33)

| Device | NPU | Description | Notes |
|--------|-----|-------------|-------|
| MSPM33C32 | No | 160 MHz Arm Cortex-M33, TrustZone | 1MB flash, 256kB SRAM |
| MSPM33C34 | No | 160 MHz Arm Cortex-M33 | High performance |


### AM13 Family (Arm Cortex-M33)

| Device | NPU | Description        | Notes                     |
|--------|-----|--------------------|---------------------------|
| AM13E2 | Yes | Arm Cortex-M33 MCU | NPU-accelerated |

### AM26x Family (Arm Cortex-R5)

| Device | NPU | Description | Notes |
|--------|-----|-------------|-------|
| AM263 | No | Quad-core Arm Cortex-R5F, 400 MHz | High performance |
| AM263P | No | Quad-core Arm Cortex-R5F, 400 MHz | High performance |
| AM261 | No | Single-core Arm Cortex-R5F, 400 MHz | Cost-optimized |

### Connectivity Devices (Wireless)

| Device | NPU | Description | Notes |
|--------|-----|-------------|-------|
| CC2755 | No | 96 MHz Arm Cortex-M33 wireless MCU | Optimized for PIR/wireless apps |
| CC1352 | No | Arm Cortex-M4 wireless MCU | Sub-1GHz and 2.4GHz |
| CC1354 | No | Arm Cortex-M33 wireless MCU | Sub-1GHz and 2.4GHz |
| CC35X1 | No | Arm Cortex-M33 wireless MCU | Wi-Fi + BLE combo |

---

## Quick Start

### Prerequisites

1. Python 3.10 environment with the Tiny ML toolchain installed
2. Clone the [tinyml-tensorlab](https://github.com/TexasInstruments/tinyml-tensorlab) repository

### Running an Example

**Linux:**
```bash
# Activate your Python environment

# Navigate to modelzoo
cd tinyml-modelzoo

# Run an example (e.g., generic_timeseries_classification)
./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml
```

**Windows:**
```powershell
# Navigate to modelzoo
cd tinyml-modelzoo

# Run an example
run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml
```

### What Happens When You Run an Example?

1. **Dataset Download** - The toolchain downloads the required dataset (if not already present)
2. **Data Processing** - Feature extraction and preprocessing are applied
3. **Model Training** - The neural network is trained on your data
4. **Quantization** - The model is optimized for MCU deployment
5. **Compilation** - TI's Neural Network Compiler generates device-ready code

* Output artifacts are saved to `../tinyml-modelmaker/data/projects/<project_name>/`.

* You can choose to save the output artifacts in your own custom directory, by specifying in the respective `config.yaml` by adding this under the common section:
    ```yaml
  common:
      projects_path: './your/choice'  # or absolute path
      # ... other settings
    ```

---

## Supported Task Categories

Tiny ML ModelZoo supports the following AI task categories:

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

**Forecasting** - Predicts future values in a time series. Best for: "What will happen next?"

**Anomaly Detection** - Uses autoencoders to learn "normal" patterns. Reconstruction error indicates anomalies. Best for: "Is this behavior normal?"

* The main difference between Anomaly Detection v/s Classification can be understood with the below example:
  * Is it Normal? or an anomaly? --> Anomaly Detection (binary outcome)
  * Is it Normal? or anomaly type A? or anomaly type B? or anomaly type C? --> Classification (multiple categories)

* The main difference between Classification v/s Regression can be understood with the below example:
  * Using independent variables Xa, Xb, Xc to predict dependent **discrete** variable (target) Y --> Classification 
    * Y can produce discrete values that indicate if it stands for Class A / Class B / Class C .... so on
  * Using independent variables Xa, Xb, Xc to predict dependent **continuous** variable (target) Y --> Regression

* The main difference between Regression v/s Forecasting can be understood with the below example:
  * Using independent variables Xa, Xb, Xc to predict dependent continuous variable (target) **Y** at the **same** time instant --> Regression
  * Using independent variables Xa, Xb, Xc to predict dependent continuous variable (target) **Xa** (or Xb or Xc) for the **next** time instant--> Forecasting

---

There are two ways to proceed using this toolchain. 
1. If you, as a user, find that there is an application under the `Example Applications` section below, then you can proceed with it.
2. However, if you do not find any applications that are of your direct interest, you may as well use the toolchain to do either of the [Supported Task Categories](#supported-task-categories) as mentioned above referring to the generic example for each of them:


| Generic Example Type         | Example                                                                              | Description                                                                     |
|------------------------------|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| Timeseries Classification    | [generic_timeseries_classification](examples/generic_timeseries_classification/)     | Classify sine/square/sawtooth waveforms. **Start here** to learn the toolchain. |
| Timeseries Regression        | [generic_timeseries_regression](examples/generic_timeseries_regression/)             | Generic regression example for continuous value prediction.                     |
| Timeseries Forecasting       | [generic_timeseries_forecasting](examples/generic_timeseries_forecasting/)           | Generic forecasting example for time series prediction.                         |
| Timeseries Anomaly Detection | [generic_timeseries_anomalydetection](examples/generic_timeseries_anomalydetection/) | Generic anomaly detection example using autoencoders.                           |

---

## Example Applications

The following ready-to-use examples demonstrate various AI applications for MCUs, organized by task type.

### Generic Timeseries Applications

These applications use generic task types that can be adapted to your custom datasets. All generic timeseries applications support **all 22 target devices.**
> **All 22 generic timeseries applications support all 22 devices:** F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29P58, F29P32, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32, F29H85, CC2755, CC1352, CC1354, CC35X1, AM263, AM263P, AM261, AM13E2

| Example Name                                                                 | Task Type | Description |
|------------------------------------------------------------------------------|-----------|-------------|
| **generic_timeseries_classification**                                        | generic_timeseries_classification | Classify sine/square/sawtooth waveforms - **Start here** to learn the toolchain |
| **generic_timeseries_regression**                                            | generic_timeseries_regression | Generic regression example for continuous value prediction |
| **generic_timeseries_forecasting**                                           | generic_timeseries_forecasting | Generic forecasting example for time series prediction |
| **generic_timeseries_anomalydetection**                                      | generic_timeseries_anomalydetection | Generic anomaly detection example using autoencoders |
| The below examples demonstrate the above AI task types with real world data: |                                     |                                                      |
| **branched_model_parameters**                                                | generic_timeseries_classification | Human Activity Recognition from accelerometer/gyroscope data |
| **electrical_fault**                                                         | generic_timeseries_classification | Classify transmission line faults using voltage and current |
| **gas_sensor**                                                               | generic_timeseries_classification | Identify gas type and concentration from sensor array data |
| **grid_fault_detection**                                                     | generic_timeseries_classification | Detect electrical grid faults from sensor data |
| **grid_stability**                                                           | generic_timeseries_classification | Predict power grid stability from node parameters |
| **nilm_appliance_usage_classification**                                      | generic_timeseries_classification | Non-Intrusive Load Monitoring - identify active appliances |
| **PLAID_nilm_classification**                                                | generic_timeseries_classification | Appliance identification using the PLAID dataset |
| **induction_motor_speed_prediction**                                         | generic_timeseries_regression | Predict induction motor speed from electrical signals |
| **mosfet_temp_prediction**                                                   | generic_timeseries_regression | Predict MOSFET temperature from electrical parameters |
| **reg_washing_machine**                                                      | generic_timeseries_regression | Predict washing machine load weight |
| **torque_measurement_regression**                                            | generic_timeseries_regression | Predict PMSM motor torque from current measurements |
| **forecasting_pmsm_rotor_temp**                                              | generic_timeseries_forecasting | Forecast PMSM rotor winding temperature |
| **hvac_indoor_temp_forecast**                                                | generic_timeseries_forecasting | Predict indoor temperature for HVAC control |
| **dc_arc_fault** (anomaly detection - DSI)                                   | generic_timeseries_anomalydetection | Detect anomalous DC arc patterns using autoencoder (DSI dataset) |
| **dc_arc_fault** (anomaly detection - DSK)                                   | generic_timeseries_anomalydetection | Detect anomalous DC arc patterns using autoencoder (DSK dataset) |
| **ecg_classification** (anomaly detection)                                   | generic_timeseries_anomalydetection | Detect anomalous heartbeat patterns from ECG signals |
| **fan_blade_fault_classification** (anomaly detection)                       | generic_timeseries_anomalydetection | Detect anomalous fan blade behavior from accelerometer data |
| **motor_bearing_fault** (anomaly detection)                                  | generic_timeseries_anomalydetection | Detect anomalous bearing behavior from vibration data |


### Application-Specific Examples

These applications are designed for specific use cases with optimized models and datasets.

| Example Name | Task Type | Supported Devices | Description |
|-------------|-----------|-------------------|-------------|
| **ac_arc_fault** | arc_fault | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32, F29H85, AM13E2, AM263 | Detect AC arc faults in electrical systems |
| **dc_arc_fault** | arc_fault | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32, F29H85, AM13E2, AM263 | Detect DC arc faults from current waveforms for electrical safety |
| **ecg_classification** | ecg_classification | MSPM0G3507, MSPM0G5187, MSPM0G3519 | Classify normal vs anomalous heartbeats from ECG signals |
| **blower_imbalance** | motor_fault | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32, F29H85, AM13E2, AM263 | Detect blade imbalance in HVAC blowers using 3-phase motor currents |
| **fan_blade_fault_classification** | motor_fault | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32, F29H85, AM13E2, AM263 | Detect faults in BLDC fans from accelerometer data |
| **motor_bearing_fault** | motor_fault | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32, F29H85, AM13E2, AM263 | Classify 5 bearing fault types + normal operation from vibration data |
| **pir_detection** | pir_detection | CC2755, CC1352, CC1354, CC35X1, MSPM0G5187, MSPM0G3507, MSPM0G3519, MSPM33C32 | Detect presence/motion using PIR sensor data |
| **MNIST_image_classification** | image_classification | MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32 | Handwritten digit recognition (MNIST dataset) |

### Summary by Task Type:
- **Generic Timeseries Tasks** (22 examples): Support all target devices and can be adapted to your custom datasets
  - Classification: 8 examples (1 base + 7 real-world applications)
  - Regression: 5 examples (1 base + 4 real-world applications)
  - Forecasting: 3 examples (1 base + 2 real-world applications)
  - Anomaly Detection: 6 examples (1 base + 5 application variants)
- **Application-Specific Tasks** (8 examples): arc_fault (2), motor_fault (3), pir_detection (1), ecg_classification (1), image_classification (1)

---

### Detailed Examples by Task Type

### Classification Examples

| No. | Example                                                                              | Data Type    | Description                                                                     |
|-----|--------------------------------------------------------------------------------------|--------------|---------------------------------------------------------------------------------|
| 1   | [dc_arc_fault](examples/dc_arc_fault/)                                               | Univariate   | Detect DC arc faults from current waveforms for electrical safety.              |
| 2   | [ac_arc_fault](examples/ac_arc_fault/)                                               | Univariate   | Detect AC arc faults in electrical systems.                                     |
| 3   | [motor_bearing_fault](examples/motor_bearing_fault/)                                 | Multivariate | Classify 5 bearing fault types + normal operation from vibration data.          |
| 4   | [blower_imbalance](examples/blower_imbalance/)                                       | Multivariate | Detect blade imbalance in HVAC blowers using 3-phase motor currents.            |
| 5   | [fan_blade_fault_classification](examples/fan_blade_fault_classification/)           | Multivariate | Detect faults in BLDC fans from accelerometer data.                             |
| 6   | [electrical_fault](examples/electrical_fault/)                                       | Multivariate | Classify transmission line faults using voltage and current.                    |
| 7   | [grid_stability](examples/grid_stability/)                                           | Multivariate | Predict power grid stability from node parameters.                              |
| 8   | [gas_sensor](examples/gas_sensor/)                                                   | Multivariate | Identify gas type and concentration from sensor array data.                     |
| 9   | [branched_model_parameters](examples/branched_model_parameters/)                     | Multivariate | Human Activity Recognition from accelerometer/gyroscope data.                   |
| 10  | [ecg_classification](examples/ecg_classification/)                                   | Multivariate | Classify normal vs anomalous heartbeats from ECG signals.                       |
| 11  | [nilm_appliance_usage_classification](examples/nilm_appliance_usage_classification/) | Multivariate | Non-Intrusive Load Monitoring - identify active appliances.                     |
| 12  | [PLAID_nilm_classification](examples/PLAID_nilm_classification/)                     | Multivariate | Appliance identification using the PLAID dataset.                               |
| 13  | [pir_detection](examples/pir_detection/)                                             | Multivariate | Detect presence/motion using PIR sensor data.                                   |

### Regression Examples

| No. | Example                                                                        | Data Type    | Description                                            |
|-----|--------------------------------------------------------------------------------|--------------|--------------------------------------------------------|
| 1   | [torque_measurement_regression](examples/torque_measurement_regression/)       | Multivariate | Predict PMSM motor torque from current measurements.   |
| 2   | [induction_motor_speed_prediction](examples/induction_motor_speed_prediction/) | Multivariate | Predict induction motor speed from electrical signals. |
| 3   | [reg_washing_machine](examples/reg_washing_machine/)                           | Multivariate | Predict washing machine load weight.                   |

### Forecasting Examples

| No. | Example                                                          | Data Type    | Description                                  |
|-----|------------------------------------------------------------------|--------------|----------------------------------------------|
| 1   | [forecasting_pmsm_rotor](examples/forecasting_pmsm_rotor/)       | Multivariate | Forecast PMSM rotor winding temperature.     |
| 2   | [hvac_indoor_temp_forecast](examples/hvac_indoor_temp_forecast/) | Multivariate | Predict indoor temperature for HVAC control. |

### Anomaly Detection Examples

| No. | Example                                                                                                 | Data Type    | Description                                                       |
|-----|---------------------------------------------------------------------------------------------------------|--------------|-------------------------------------------------------------------|
| 1   | [dc_arc_fault (DSI)](examples/dc_arc_fault/config_anomaly_detection_dsi.yaml)                           | Univariate   | Detect anomalous DC arc patterns using autoencoder (DSI dataset). |
| 2   | [dc_arc_fault (DSK)](examples/dc_arc_fault/config_anomaly_detection_dsk.yaml)                           | Univariate   | Detect anomalous DC arc patterns using autoencoder (DSK dataset). |
| 3   | [ecg_classification](examples/ecg_classification/config_anomaly_detection.yaml)                         | Multivariate | Detect anomalous heartbeat patterns from ECG signals.             |
| 4   | [fan_blade_fault_classification](examples/fan_blade_fault_classification/config_anomaly_detection.yaml) | Multivariate | Detect anomalous fan blade behavior from accelerometer data.      |
| 5   | [motor_bearing_fault](examples/motor_bearing_fault/config_anomaly_detection.yaml)                       | Multivariate | Detect anomalous bearing behavior from vibration data.            |


### Image Classification Examples

| No. | Example                                                            | Data Type | Description                                    |
|-----|--------------------------------------------------------------------|-----------|------------------------------------------------|
| 1   | [MNIST_image_classification](examples/MNIST_image_classification/) | Image     | Handwritten digit recognition (MNIST dataset). |

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

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

---

**Questions or Issues?** Open an issue on [GitHub](https://github.com/TexasInstruments/tinyml-tensorlab/issues).
