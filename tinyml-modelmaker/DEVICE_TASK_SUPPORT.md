# Tiny ML Device and Task Support Matrix

This document provides a comprehensive overview of supported tasks/applications and the devices that support them in the Texas Instruments Tiny ML ecosystem.

## Table of Contents
- [Supported Tasks Overview](#supported-tasks-overview)
- [Supported Device Families](#supported-device-families)
- [Task Support Matrix](#task-support-matrix)
- [Device Support Details](#device-support-details)
- [Task Descriptions](#task-descriptions)

---

## Supported Tasks Overview

The Tiny ML framework supports the following task categories:

### Task Categories
1. **Timeseries Classification** - Classify time-series data into categories
2. **Timeseries Regression** - Predict continuous values from time-series data
3. **Timeseries Anomaly Detection** - Detect anomalies in time-series data
4. **Timeseries Forecasting** - Predict future values in time-series data

### Specific Applications
1. **Arc Fault Detection** - Detect electrical arc faults in power systems
2. **Motor Fault Detection** - Identify faults in motor operation
3. **Blower/Fan Imbalance Detection** - Detect imbalance in rotating equipment
4. **PIR (Passive Infrared) Detection** - Motion/presence detection applications
5. **Generic Timeseries Classification** - Custom classification tasks
6. **Generic Timeseries Regression** - Custom regression tasks
7. **Generic Timeseries Anomaly Detection** - Custom anomaly detection tasks
8. **Generic Timeseries Forecasting** - Custom forecasting tasks

---

## Supported Device Families

### C2000 DSP Family (Texas Instruments)
- **F280013** - C2000 32-bit MCU with 100 MHz, FPU, CLA
- **F280015** - C2000 32-bit MCU with 120 MHz, FPU, CLA
- **F28003** - C2000 32-bit MCU with 100 MHz, FPU, CLA
- **F28004** - C2000 32-bit MCU with 100 MHz, FPU, CLA
- **F2837** - C2000 32-bit dual-core MCU with 200 MHz
- **F28P55** - C2000 32-bit MCU with hardware NPU
- **F28P65** - C2000 32-bit MCU with 150 MHz, hardware NPU
- **F29H85** - C2000 64-bit MCU with C29x core
- **F29P58** - C2000 64-bit MCU with C29x core
- **F29P32** - C2000 64-bit MCU with C29x core

### MSPM0 Family (Arm Cortex-M0+)
- **MSPM0G3507** - 80 MHz Arm Cortex-M0+ MCU with hardware NPU
- **MSPM0G5187** - 80 MHz Arm Cortex-M0+ MCU with hardware NPU

### MSPM33C Family (Arm Cortex-M33)
- **MSPM33C32** - 160 MHz Arm Cortex-M33 MCU with TrustZone, 1MB flash, 256kB SRAM
- **MSPM33C34** - 160 MHz Arm Cortex-M33 MCU with hardware NPU
- **AM13E2** - Arm Cortex-M33 MCU (additional device, CLI only)

### AM26x Family (Arm Cortex-R5)
- **AM263** - Quad-core Arm Cortex-R5F MCU up to 400 MHz
- **AM263P** - Quad-core Arm Cortex-R5F MCU up to 400 MHz
- **AM261** - Single-core Arm Cortex-R5F MCU up to 400 MHz

### Connectivity Devices (Wireless)
- **CC2755** - 96 MHz Arm Cortex-M33 wireless MCU with NPU
- **CC1352** - Arm Cortex-M4 wireless MCU for sub-1GHz and 2.4GHz

---

## Task Support Matrix

### By Task Type

| Task / Application | Supported Devices | Example Projects |
|-------------------|-------------------|-----------------|
| **Arc Fault Detection** | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P58, F29P32, MSPM0G3507, MSPM0G5187, MSPM33C32, MSPM33C34, AM13E2, AM263, AM263P, AM261 | `ac_arc_fault`, `dc_arc_fault` |
| **Motor Fault Detection** | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P58, F29P32, MSPM0G3507, MSPM0G5187, MSPM33C32, MSPM33C34, AM13E2, AM263, AM263P, AM261 | `motor_bearing_fault`, `fan_blade_fault_classification`, `blower_imbalance` |
| **Blower Imbalance Detection** | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P58, F29P32, MSPM33C32, MSPM33C34, AM13E2, AM263, AM263P, AM261 | `blower_imbalance` |
| **PIR Detection** | CC2755, CC1352 | `pir_detection` |
| **Generic Timeseries Classification** | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P58, F29P32, MSPM0G3507, MSPM0G5187, MSPM33C32, MSPM33C34, AM13E2, CC2755, CC1352, AM263, AM263P, AM261 | `hello_world`, `ecg_classification`, `electrical_fault`, `gas_sensor`, `grid_stability`, `nilm_appliance_usage_classification`, `PLAID_nilm_classification`, `branched_model_parameters` |
| **Generic Timeseries Regression** | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P58, F29P32, MSPM33C32, MSPM33C34, AM13E2, CC2755, CC1352, AM263, AM263P, AM261 | `induction_motor_speed_prediction`, `reg_washing_machine`, `torque_measurement_regression` |
| **Generic Timeseries Anomaly Detection** | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P58, F29P32, MSPM33C32, MSPM33C34, AM13E2, CC2755, CC1352, AM263, AM263P, AM261 | `dc_arc_fault_anomaly_detection`, `motor_bearing_fault_anomaly_detection`, `fan_blade_anomaly_detection`, `ecg_anomaly_detection` |
| **Generic Timeseries Forecasting** | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P58, F29P32, MSPM33C32, MSPM33C34, AM13E2, CC2755, CC1352, AM263, AM263P, AM261 | `forecasting_pmsm_rotor`, `hvac_indoor_temp_forecast` |
| **Image Classification** | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P58, F29P32 | `MNIST_image_classification` |

### Summary by Device Capability

| Device Category | All Generic Tasks | Specialized Tasks | Hardware NPU |
|----------------|-------------------|-------------------|--------------|
| **C2000 F28x (non-NPU)** | ✅ | Arc Fault, Motor Fault, Blower Imbalance | ❌ |
| **C2000 F28Px (NPU)** | ✅ | Arc Fault, Motor Fault, Blower Imbalance | ✅ (Hard) |
| **C2000 F29x (C29 core)** | ✅ | Arc Fault, Motor Fault, Blower Imbalance | ❌ |
| **MSPM0G (NPU)** | Classification only | Arc Fault, Motor Fault | ✅ (Hard) |
| **MSPM33C32** | ✅ | Arc Fault, Motor Fault, Blower Imbalance | ❌ (Soft) |
| **MSPM33C34** | ✅ | Arc Fault, Motor Fault, Blower Imbalance | ✅ (Hard) |
| **AM13E2** | ✅ | Arc Fault, Motor Fault, Blower Imbalance | ❌ (Soft) |
| **AM26x Series** | ✅ | Arc Fault, Motor Fault, Blower Imbalance | ❌ |
| **CC2755/CC1352** | ✅ | PIR Detection | Soft NPU |

---

## Device Support Details

### Full Support Devices
These devices support **all** timeseries tasks (classification, regression, anomaly detection, forecasting) plus specialized applications:

#### C2000 Family
- **F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P58, F29P32**
  - Generic Tasks: Classification, Regression, Anomaly Detection, Forecasting
  - Specialized: Arc Fault, Motor Fault, Blower Imbalance
  - Compilation: Soft NPU (F28Px has Hard NPU option, F29x uses C29 core)

#### ARM Cortex-R5 Family
- **AM263, AM263P, AM261**
  - Generic Tasks: Classification, Regression, Anomaly Detection, Forecasting
  - Specialized: Arc Fault, Motor Fault, Blower Imbalance
  - Compilation: Soft NPU (no hardware accelerator)

#### ARM Cortex-M33 Family
- **MSPM33C32, MSPM33C34, AM13E2**
  - Generic Tasks: Classification, Regression, Anomaly Detection, Forecasting
  - Specialized: Arc Fault, Motor Fault, Blower Imbalance
  - Compilation: Soft NPU (MSPM33C34 has Hard NPU option)

### Partial Support Devices

#### MSPM0G Family (Classification Focus)
- **MSPM0G3507, MSPM0G5187**
  - Generic Tasks: Classification only
  - Specialized: Arc Fault, Motor Fault
  - Compilation: Hard NPU available
  - Note: Limited to classification tasks due to memory constraints

#### Wireless/Connectivity Devices
- **CC2755, CC1352**
  - Generic Tasks: Classification, Regression, Anomaly Detection, Forecasting
  - Specialized: PIR Detection
  - Compilation: Soft NPU
  - Note: Optimized for wireless/connectivity applications

---

## Task Descriptions

### 1. Generic Timeseries Classification
**Purpose:** Classify time-series sensor data into predefined categories.

**Example Use Cases:**
- Activity recognition (walking, running, sitting)
- Gesture recognition
- Equipment state classification
- Pattern recognition in sensor data

**Available Models:**
- TimeSeries_Generic_13k_t (13K parameters)
- TimeSeries_Generic_6k_t (6K parameters)
- TimeSeries_Generic_4k_t (4K parameters)
- TimeSeries_Generic_1k_t (1K parameters)
- TimeSeries_Generic_100_t (100 parameters)
- TimeSeries_Generic_55k_t (55K parameters)
- Res_Add_TimeSeries_Generic_3k_t (Residual addition, 3K parameters)
- Res_Cat_TimeSeries_Generic_3k_t (Residual concatenation, 3K parameters)

**Key Features:**
- Multiple model sizes for different memory constraints
- Support for multi-channel sensor inputs
- Configurable window sizes and feature extraction

---

### 2. Generic Timeseries Regression
**Purpose:** Predict continuous values from time-series data.

**Example Use Cases:**
- Energy consumption prediction
- Temperature prediction
- Load forecasting
- Sensor calibration

**Available Models:**
- TimeSeries_Generic_Regr_13k_t (13K parameters, CNN-based)
- TimeSeries_Generic_Regr_10k_t (10K parameters)
- TimeSeries_Generic_Regr_4k_t (4K parameters, CNN-based)
- TimeSeries_Generic_Regr_3k_t (3K parameters, MLP-based)
- TimeSeries_Generic_Regr_1k_t (1K parameters)

**Key Features:**
- Multiple architectures (CNN, MLP)
- Optimized for real-time prediction
- Support for multi-target regression

---

### 3. Generic Timeseries Anomaly Detection
**Purpose:** Identify unusual patterns or outliers in time-series data.

**Example Use Cases:**
- Equipment health monitoring
- Predictive maintenance
- Quality control
- Security monitoring

**Available Models:**
- TimeSeries_Generic_AD_17k_t (17K parameters)
- TimeSeries_Generic_AD_16k_t (16K parameters)
- TimeSeries_Generic_AD_4k_t (4K parameters)
- TimeSeries_Generic_AD_1k_t (1K parameters)
- TimeSeries_Generic_Linear_AD (Linear model)
- Ondevice_Trainable_TimeSeries_Generic_Linear_AD (On-device trainable)

**Key Features:**
- Unsupervised and semi-supervised approaches
- Real-time anomaly scoring
- On-device learning capability (selected models)

---

### 4. Generic Timeseries Forecasting
**Purpose:** Predict future values in time-series sequences.

**Example Use Cases:**
- Energy demand forecasting
- Resource planning
- Predictive scheduling
- Trend prediction

**Available Models:**
- TimeSeries_Generic_Forecasting_13k_t (13K parameters, CNN-based)
- TimeSeries_Generic_Forecasting_3k_t (3K parameters, MLP-based)
- TimeSeries_Generic_Forecasting_LSTM10 (LSTM with hidden size 10)
- TimeSeries_Generic_Forecasting_LSTM8 (LSTM with hidden size 8)

**Key Features:**
- Multiple forecasting horizons
- CNN and LSTM architectures
- Support for multi-variate forecasting

---

### 5. Arc Fault Detection
**Purpose:** Detect dangerous electrical arc faults in power distribution systems.

**Example Use Cases:**
- Electrical safety monitoring
- Circuit breaker applications
- Power quality monitoring

**Available Models:**
- ArcFault_model_1400_t (1400 parameters)
- ArcFault_model_700_t (700 parameters)
- ArcFault_model_300_t (300 parameters)
- ArcFault_model_200_t (200 parameters)

**Supported Devices:** F28x, MSPM0G, MSPM33C, AM13E2, AM26x series

---

### 6. Motor Fault Detection
**Purpose:** Identify mechanical and electrical faults in motors.

**Example Use Cases:**
- Predictive maintenance
- Motor health monitoring
- Bearing fault detection

**Available Models:**
- MotorFault_model_3_t
- MotorFault_model_2_t
- MotorFault_model_1_t

**Supported Devices:** F28x, MSPM0G, MSPM33C, AM13E2, AM26x series

---

### 7. Blower/Fan Imbalance Detection
**Purpose:** Detect imbalance in rotating equipment like fans and blowers.

**Example Use Cases:**
- HVAC system monitoring
- Industrial fan monitoring
- Vibration analysis

**Available Models:**
- FanImbalance_model_3_t
- FanImbalance_model_2_t
- FanImbalance_model_1_t

**Supported Devices:** F28x (except MSPM0G), MSPM33C, AM13E2, AM26x series

---

### 8. PIR Detection
**Purpose:** Motion and presence detection using passive infrared sensors.

**Example Use Cases:**
- Occupancy detection
- Security systems
- Smart lighting control

**Available Models:**
- PIRDetection_model_1_t

**Supported Devices:** CC2755, CC1352 (wireless connectivity devices)

---

## Model Parameter Constraints

Models are sized to fit different MCU memory constraints:

| Parameter Count | Target MCU Class | Example Devices |
|----------------|------------------|-----------------|
| 100-1K params | Ultra-minimal | MSPM0G series |
| 1K-4K params | Small MCUs | F280013, MSPM0G |
| 4K-6K params | Standard MCUs | F28003, F28004 |
| 6K-13K params | Larger MCUs | F28P55, F28P65, MSPM33C |
| 13K-16K params | Edge devices | F29H85, AM26x |
| 55K+ params | High-end MCUs | F29H85, AM263P |

---

## Hardware NPU (TinyEngine) Support

### Hard NPU (Hardware Accelerator)
- **F28P55** 
- **MSPM0G5187**
- **AM13E2**
- **MSPM33C34**

### Soft NPU (Software Implementation)
- All other devices use optimized software NPU implementation
- Compilation target: `ti-npu type=soft`

---

## Notes

1. **GUI vs CLI Devices:**
   - All listed devices except MSPM33C34 and AM13E2 are available in the GUI
   - MSPM33C34 and AM13E2 are in `TARGET_DEVICES_ADDITIONAL` (CLI only)

2. **Compilation Profiles:**
   - Each device has optimized compilation profiles
   - NPU type (soft/hard) is configured per device capability
   - Space optimization available for devices with hard NPU

3. **Feature Extraction:**
   - All tasks support configurable feature extraction (FFT, wavelets, etc.)
   - Preprocessing parameters are task and dataset specific

4. **Quantization:**
   - Models support quantization-aware training
   - Post-training quantization available
   - Quantization optimized for target device architecture

---

## Example Projects

The Tiny ML ecosystem includes comprehensive example projects demonstrating various use cases. All examples are located in `/tinyml-modelmaker/examples/`.

### Timeseries Classification Examples

#### hello_world
- **Task Type:** Generic Timeseries Classification
- **Description:** Introductory example demonstrating basic timeseries classification workflow
- **Use Case:** Learning the Tiny ML framework basics
- **Recommended Devices:** All devices supporting classification
- **Key Features:** Simple dataset, fast training, ideal for getting started

#### ecg_classification
- **Task Type:** Generic Timeseries Classification
- **Description:** ECG (electrocardiogram) signal classification for cardiac health monitoring
- **Use Case:** Medical device applications, heart rhythm analysis
- **Recommended Devices:** F28P55, F28P65, MSPM33C34 (hardware NPU for real-time processing)
- **Key Features:** Multi-class classification, signal processing, medical diagnostics

#### electrical_fault
- **Task Type:** Generic Timeseries Classification
- **Description:** Electrical fault detection and classification in power systems
- **Use Case:** Power distribution monitoring, fault diagnosis
- **Recommended Devices:** F280013, F280015, F28003, F28004 (optimized for power applications)
- **Key Features:** Multi-fault classification, real-time detection

#### gas_sensor
- **Task Type:** Generic Timeseries Classification
- **Description:** Gas sensor data classification for environmental monitoring
- **Use Case:** Air quality monitoring, gas leak detection
- **Recommended Devices:** CC2755, CC1352 (wireless connectivity for IoT deployment)
- **Key Features:** Multi-gas classification, sensor fusion

#### grid_stability
- **Task Type:** Generic Timeseries Classification
- **Description:** Power grid stability prediction and classification
- **Use Case:** Smart grid applications, grid health monitoring
- **Recommended Devices:** F2837, F28P65 (dual-core or high-performance MCUs)
- **Key Features:** Real-time grid monitoring, stability prediction

#### nilm_appliance_usage_classification
- **Task Type:** Generic Timeseries Classification
- **Description:** Non-Intrusive Load Monitoring (NILM) for appliance usage detection
- **Use Case:** Smart home energy management, appliance recognition
- **Recommended Devices:** F28P55, F28P65, AM263, AM263P
- **Key Features:** Energy disaggregation, appliance signature detection

#### PLAID_nilm_classification
- **Task Type:** Generic Timeseries Classification
- **Description:** NILM using the PLAID (Plug Load Appliance Identification Dataset)
- **Use Case:** Advanced energy monitoring, appliance-level consumption tracking
- **Recommended Devices:** F29H85, F29P58, F29P32, AM263P (high-parameter models)
- **Key Features:** Large-scale appliance database, high-accuracy classification

#### branched_model_parameters
- **Task Type:** Generic Timeseries Classification
- **Description:** Demonstrates branched neural network architectures with shared feature extraction
- **Use Case:** Multi-task learning, parameter-efficient models
- **Recommended Devices:** All devices supporting classification
- **Key Features:** Model architecture experimentation, parameter sharing

### Specialized Fault Detection Examples

#### ac_arc_fault
- **Task Type:** Arc Fault Detection
- **Description:** AC (alternating current) arc fault detection for electrical safety
- **Use Case:** Circuit breaker applications, electrical panel monitoring
- **Recommended Devices:** F280013, F280015, F28003, F28004, MSPM0G3507, MSPM0G5187
- **Key Features:** Real-time arc detection, low-latency inference, safety-critical application

#### dc_arc_fault
- **Task Type:** Arc Fault Detection / Anomaly Detection
- **Description:** DC (direct current) arc fault detection with anomaly detection variants
- **Use Case:** Solar panel systems, EV charging stations, DC power distribution
- **Recommended Devices:** F28P55, F28P65 (hardware NPU for fast processing)
- **Key Features:** Both classification and anomaly detection modes, DC-specific features

#### motor_bearing_fault
- **Task Type:** Motor Fault Detection
- **Description:** Motor bearing fault classification using vibration data
- **Use Case:** Predictive maintenance, motor health monitoring
- **Recommended Devices:** F28003, F28004, F2837, AM263
- **Key Features:** Vibration signal analysis, multi-fault classification

#### fan_blade_fault_classification
- **Task Type:** Motor Fault Detection
- **Description:** Fan blade fault detection and classification
- **Use Case:** HVAC systems, industrial fans, cooling equipment
- **Recommended Devices:** F280013, F280015, MSPM33C32
- **Key Features:** Acoustic/vibration analysis, imbalance detection

#### blower_imbalance
- **Task Type:** Blower/Fan Imbalance Detection
- **Description:** Blower imbalance detection using current/vibration signatures
- **Use Case:** Industrial blowers, HVAC monitoring, rotating equipment
- **Recommended Devices:** F28P65, F29H85, AM263P
- **Key Features:** Real-time imbalance quantification, preventive maintenance

### Timeseries Regression Examples

#### induction_motor_speed_prediction
- **Task Type:** Generic Timeseries Regression
- **Description:** Induction motor speed prediction from current/voltage measurements
- **Use Case:** Motor control, sensorless speed estimation
- **Recommended Devices:** F280013, F280015, F28003, F28004 (motor control MCUs)
- **Key Features:** Real-time speed estimation, cost reduction (no speed sensor needed)

#### reg_washing_machine
- **Task Type:** Generic Timeseries Regression
- **Description:** Washing machine parameter regression for smart control
- **Use Case:** Smart home appliances, energy optimization
- **Recommended Devices:** MSPM33C32, CC2755 (connectivity-enabled devices)
- **Key Features:** Multi-parameter regression, appliance optimization

#### torque_measurement_regression
- **Task Type:** Generic Timeseries Regression
- **Description:** Motor torque estimation from electrical measurements
- **Use Case:** Motor control, torque sensor replacement
- **Recommended Devices:** F2837, F28P55, F28P65, AM263
- **Key Features:** High-accuracy torque estimation, sensor cost reduction

### Timeseries Forecasting Examples

#### forecasting_pmsm_rotor
- **Task Type:** Generic Timeseries Forecasting
- **Description:** PMSM (Permanent Magnet Synchronous Motor) rotor position forecasting
- **Use Case:** Motor control, predictive control algorithms
- **Recommended Devices:** F280015, F28004, F2837 (real-time control)
- **Key Features:** Multi-step forecasting, control loop optimization

#### hvac_indoor_temp_forecast
- **Task Type:** Generic Timeseries Forecasting
- **Description:** HVAC indoor temperature forecasting for predictive climate control
- **Use Case:** Smart buildings, energy-efficient HVAC systems
- **Recommended Devices:** MSPM33C32, AM263P, CC2755
- **Key Features:** Multi-variate forecasting, energy optimization

### Timeseries Anomaly Detection Examples

#### dc_arc_fault_anomaly_detection
- **Task Type:** Generic Timeseries Anomaly Detection
- **Description:** DC arc fault detection using anomaly detection approach with two variants (DSI and DSK datasets)
- **Use Case:** Solar panel systems, EV charging stations, DC power distribution safety monitoring
- **Recommended Devices:** F28P55, F28P65 (hardware NPU for fast anomaly scoring), F29H85, AM263P
- **Key Features:** Unsupervised learning, real-time anomaly scoring, works with limited labeled data
- **Configurations:** `config_anomaly_detection_dsi.yaml`, `config_anomaly_detection_dsk.yaml`

#### motor_bearing_fault_anomaly_detection
- **Task Type:** Generic Timeseries Anomaly Detection
- **Description:** Motor bearing fault detection using anomaly detection for predictive maintenance
- **Use Case:** Industrial motors, predictive maintenance, early fault detection without labeled failure data
- **Recommended Devices:** F28003, F28004, F2837, AM263, AM263P
- **Key Features:** Vibration analysis, normal behavior modeling, unsupervised anomaly detection
- **Configuration:** `config_anomaly_detection.yaml`

#### fan_blade_anomaly_detection
- **Task Type:** Generic Timeseries Anomaly Detection
- **Description:** Fan blade fault detection using anomaly detection with on-device training capability
- **Use Case:** HVAC systems, industrial fans, condition monitoring with adaptive learning
- **Recommended Devices:** F28P65, F29H85, MSPM33C32, AM263P
- **Key Features:** On-device trainable model, adaptive learning, continuous monitoring
- **Configurations:** `config_anomaly_detection.yaml`, `fan_blade_anomaly_detection_ondevice_training.yaml`

#### ecg_anomaly_detection
- **Task Type:** Generic Timeseries Anomaly Detection
- **Description:** ECG signal anomaly detection for cardiac health monitoring
- **Use Case:** Medical devices, wearable health monitors, arrhythmia detection
- **Recommended Devices:** F28P55, F28P65, MSPM33C34 (hardware NPU for low-latency detection)
- **Key Features:** Real-time anomaly detection, medical-grade signal processing, low-power operation
- **Configuration:** `config_anomaly_detection.yaml`

### Wireless/Connectivity Examples

#### pir_detection
- **Task Type:** PIR Detection
- **Description:** Passive Infrared (PIR) sensor-based motion and presence detection
- **Use Case:** Occupancy sensing, security systems, smart lighting
- **Recommended Devices:** CC2755, CC1352 (wireless connectivity devices)
- **Key Features:** Low-power operation, wireless reporting, edge AI inference

### Image Classification Examples

#### MNIST_image_classification
- **Task Type:** Image Classification
- **Description:** Classic MNIST handwritten digit classification
- **Use Case:** Learning image classification workflow, digit recognition
- **Recommended Devices:** F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P58, F29P32
- **Key Features:** Image preprocessing, CNN architectures, quantization demo

---

## Getting Started

To train a model for a specific task and device:

```bash
# Using ModelMaker CLI
cd tinyml-modelmaker
./run_tinyml_modelmaker.sh examples/<task_type>/config.yaml
```

For more information, refer to:
- `./tinyml-modelmaker/examples/` - Example configurations
