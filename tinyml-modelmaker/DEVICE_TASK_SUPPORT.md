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
5. **ECG Classification** - Cardiac signal classification (its own `task_type`, distinct from Generic Timeseries Classification)
6. **Generic Timeseries Classification** - Custom classification tasks
7. **Generic Timeseries Regression** - Custom regression tasks
8. **Generic Timeseries Anomaly Detection** - Custom anomaly detection tasks
9. **Generic Timeseries Forecasting** - Custom forecasting tasks

---

## Supported Device Families

### C2000 DSP Family (Texas Instruments)
- **F280013** - C2000 32-bit MCU with 100 MHz, FPU, CLA
- **F280015** - C2000 32-bit MCU with 120 MHz, FPU, CLA
- **F28003** - C2000 32-bit MCU with 100 MHz, FPU, CLA
- **F28004** - C2000 32-bit MCU with 100 MHz, FPU, CLA
- **F2837** - C2000 32-bit dual-core MCU with 200 MHz (xD variant)
- **F2837xS** - C2000 32-bit single-core MCU with 200 MHz (xS variant)
- **F2838x** - C2000 32-bit dual-core MCU 200 MHz C28x + Arm Cortex-M4
- **F2807x** - C2000 32-bit MCU 120 MHz C28x + CLA, 512-KB Flash
- **F28002x** - C2000 32-bit MCU 100 MHz C28x + CLA, 256-KB Flash
- **F28P55** - C2000 32-bit MCU with hardware NPU
- **F28P65** - C2000 32-bit MCU with 150 MHz, hardware NPU
- **F28P551x** - C2000 32-bit MCU 150 MHz C28x + CLA, 1.1-MB Flash
- **F29H85** - C2000 64-bit MCU with C29x core
- **F29P58** - C2000 64-bit MCU with C29x core
- **F29P32** - C2000 64-bit MCU with C29x core

### MSPM0 Family (Arm Cortex-M0+)
- **MSPM0G3507** - 80 MHz Arm Cortex-M0+ MCU with hardware NPU
- **MSPM0G3519** - 80 MHz Arm Cortex-M0+ MCU with hardware NPU
- **MSPM0G5187** - 80 MHz Arm Cortex-M0+ MCU with hardware NPU

### MSPM33C Family (Arm Cortex-M33)
- **MSPM33C32** - 160 MHz Arm Cortex-M33 MCU with TrustZone, 1MB flash, 256kB SRAM
- **MSPM33C34** - 160 MHz Arm Cortex-M33 MCU with hardware NPU

### AM13 Family (Arm Cortex-M33)
- **AM13E2** - Arm Cortex-M33 MCU

### AM26x Family (Arm Cortex-R5)
- **AM263** - Quad-core Arm Cortex-R5F MCU up to 400 MHz
- **AM263P** - Quad-core Arm Cortex-R5F MCU up to 400 MHz
- **AM261** - Single-core Arm Cortex-R5F MCU up to 400 MHz

### Connectivity Devices (Wireless)
- **CC2755** - 96 MHz Arm Cortex-M33 2.4 GHz wireless MCU with CDE (Custom Datapath Extension)
- **CC2745** - 96 MHz Arm Cortex-M33 2.4 GHz wireless MCU with CDE (Custom Datapath Extension, Automotive)
- **CC1312** - Arm Cortex-M4F sub-1 GHz wireless MCU
- **CC1314** - Arm Cortex-M33 sub-1 GHz wireless MCU
- **CC1352** - Arm Cortex-M4F sub-1 GHz wireless MCU
- **CC1354** - Arm Cortex-M33 sub-1 GHz wireless MCU
- **CC35X1** - Arm Cortex-M33 Wi-Fi wireless MCU with CDE (Custom Datapath Extension)

---

## Task Support Matrix

> **This table is derived, not hand-maintained.** For every timeseries `task_type`, the "Supported Devices" column below is the union of the `target_devices` declared by each model belonging to that `task_type` in `tinyml-modelzoo` (`tinyml_modelzoo/model_descriptions/{classification,regression,anomalydetection,forecasting}.py`). It is computed by `_get_task_target_devices_from_models()` / `_task_target_devices()` in `tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/constants.py`, and exposed as `TASK_DESCRIPTIONS[task_type]['target_devices']`. Because `tinyml-modelzoo` model descriptions are the single source of truth, this table stays in sync automatically as models are added, removed, or given new `target_devices` — it does **not** need to be hand-edited when models change. To regenerate the values below, run:
> ```python
> import sys
> sys.path.insert(0, '.')  # from tinyml-modelmaker/
> sys.path.insert(0, '../tinyml-modelzoo')
> from tinyml_modelmaker.ai_modules.timeseries import constants
> for tt in constants.TASK_TYPES:
>     print(tt, sorted(constants.TASK_DESCRIPTIONS[tt]['target_devices']))
> ```
> The **Image Classification** row is the one exception: it belongs to a separate `target_module='image'` code path that is not part of this derivation, so its device list below remains hand-maintained.

### By Task Type

| Task / Application | Supported Devices | Example Projects |
|-------------------|-------------------|-----------------|
| **Arc Fault Detection** | AM13E2, AM263, F280013, F280015, F28002x, F28003, F28004, F2807x, F2837, F2837xS, F2838x, F28P55, F28P551x, F28P65, F29H85, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32 | `ac_arc_fault`, `dc_arc_fault` |
| **ECG Classification** | F280013, F280015, F28002x, F28003, F28004, F2807x, F2837, F2837xS, F2838x, F28P551x, MSPM0G3507, MSPM0G3519, MSPM0G5187 | `ecg_classification` |
| **Motor Fault Detection** | AM13E2, AM263, CC1312, CC1314, CC1352, CC1354, CC2745, CC2755, CC35X1, F280013, F280015, F28002x, F28003, F28004, F2807x, F2837, F2837xS, F2838x, F28P55, F28P551x, F28P65, F29H85, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32 | `motor_bearing_fault`, `fan_blade_fault_classification` |
| **Blower Imbalance Detection** | AM13E2, AM263, F280013, F280015, F28002x, F28003, F28004, F2807x, F2837, F2837xS, F2838x, F28P55, F28P551x, F28P65, F29H85, MSPM33C32 | `blower_imbalance` |
| **PIR Detection** | CC1312, CC1314, CC1352, CC1354, CC2745, CC2755, CC35X1, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32 | `pir_detection` |
| **Generic Timeseries Classification** | AM13E2, AM261, AM263, AM263P, CC1312, CC1314, CC1352, CC1354, CC2745, CC2755, CC35X1, F280013, F280015, F28002x, F28003, F28004, F2807x, F2837, F2837xS, F2838x, F28P55, F28P551x, F28P65, F29H85, F29P32, F29P58, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32 | `hello_world`, `electrical_fault`, `gas_sensor`, `grid_stability`, `nilm_appliance_usage_classification`, `PLAID_nilm_classification`, `human_activity_recognition` |
| **Generic Timeseries Regression** | AM13E2, AM261, AM263, AM263P, F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P32, F29P58, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32 | `induction_motor_speed_prediction`, `washing_machine_load_weighing`, `torque_measurement_regression` |
| **Generic Timeseries Anomaly Detection** | AM13E2, AM261, AM263, AM263P, F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P32, F29P58, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32 | `dc_arc_fault_anomaly_detection`, `motor_bearing_fault_anomaly_detection`, `fan_blade_anomaly_detection`, `ecg_anomaly_detection` |
| **Generic Timeseries Forecasting** | AM13E2, AM261, AM263, AM263P, F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P32, F29P58, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32 | `forecasting_pmsm_rotor`, `hvac_indoor_temp_forecast` |
| **Image Classification**¹ | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P58, F29P32 | `MNIST_image_classification` |

¹ Not derived from `tinyml-modelzoo` (separate `target_module='image'` path) — hand-maintained.

**Notable corrections from the previous hand-maintained version:** `MSPM33C34` is a defined device constant but currently has **no** `tinyml-modelzoo` model targeting it for any task, so it no longer appears anywhere below. `AM261`/`AM263P` support the generic tasks but not Arc Fault/Motor Fault/Blower Imbalance (only `AM263` does among the AM26x family). `F29P32`/`F29P58` support the generic tasks but not Arc Fault/Motor Fault/Blower Imbalance (only `F29H85` does among the C29x family). Motor Fault now also includes the wireless connectivity devices (CC1312/CC1314/CC1352/CC1354/CC2745/CC2755/CC35X1) — these were added to the `MotorFault_model_1/2/3_t` `target_devices` in `tinyml-modelzoo` so the `fan_blade_fault_classification` CC-device example configs resolve to a real model instead of an empty list. MSPM0G devices (`MSPM0G3507`, `MSPM0G3519`, `MSPM0G5187`) now support all generic tasks (previously thought to be classification-only) plus Arc Fault, Motor Fault, ECG Classification, and PIR Detection, but not Blower Imbalance. `MSPM0G3519` was previously missing from this document entirely.

### Summary by Device Capability

Per-device support, derived the same way as the [Task Support Matrix](#task-support-matrix) above (see the note there). "Regr./AD/Fcst" covers Regression, Anomaly Detection, and Forecasting together since, for every device, all three are either all supported or all unsupported. For hardware-NPU-vs-software-NPU compilation details per device, see [Hardware NPU (TinyEngine) Support](#hardware-npu-tinyengine-support) — that categorization is independent of the `target_devices` derivation described above and is unchanged by this refactor.

| Device | Classification | Regr./AD/Fcst | Arc Fault | Motor Fault | Blower Imbalance | ECG Classification | PIR Detection |
|--------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| F280013 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| F280015 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| F28003 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| F28004 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| F2837 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| F2837xS | ✅ | — | ✅ | ✅ | ✅ | ✅ | — |
| F2838x | ✅ | — | ✅ | ✅ | ✅ | ✅ | — |
| F2807x | ✅ | — | ✅ | ✅ | ✅ | ✅ | — |
| F28002x | ✅ | — | ✅ | ✅ | ✅ | ✅ | — |
| F28P551x | ✅ | — | ✅ | ✅ | ✅ | ✅ | — |
| F28P55 | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| F28P65 | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| F29H85 | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| F29P58 | ✅ | ✅ | — | — | — | — | — |
| F29P32 | ✅ | ✅ | — | — | — | — | — |
| MSPM0G3507 | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| MSPM0G3519 | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| MSPM0G5187 | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| MSPM33C32 | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| MSPM33C34 | — | — | — | — | — | — | — |
| AM13E2 | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| AM263 | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| AM263P | ✅ | ✅ | — | — | — | — | — |
| AM261 | ✅ | ✅ | — | — | — | — | — |
| CC2755 | ✅ | — | — | ✅ | — | — | ✅ |
| CC2745 | ✅ | — | — | ✅ | — | — | ✅ |
| CC1312 | ✅ | — | — | ✅ | — | — | ✅ |
| CC1314 | ✅ | — | — | ✅ | — | — | ✅ |
| CC1352 | ✅ | — | — | ✅ | — | — | ✅ |
| CC1354 | ✅ | — | — | ✅ | — | — | ✅ |
| CC35X1 | ✅ | — | — | ✅ | — | — | ✅ |

`MSPM33C34` has an all-`—` row because it is a defined device constant with no `tinyml-modelzoo` model currently targeting it for any task (see the note above the Task Support Matrix).

---

## Device Support Details

The groupings below follow from the per-device table in [Summary by Device Capability](#summary-by-device-capability), which is itself derived from `tinyml-modelzoo` model descriptions (see the note at the top of [Task Support Matrix](#task-support-matrix)).

### Full Support Devices
These devices support **all four** generic timeseries tasks (classification, regression, anomaly detection, forecasting) plus Arc Fault, Motor Fault, and Blower Imbalance:

#### C2000 Family
- **F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85**
  - Generic Tasks: Classification, Regression, Anomaly Detection, Forecasting
  - Specialized: Arc Fault, Motor Fault, Blower Imbalance
  - Compilation: Soft NPU (F28P55/F28P65 have a Hard NPU option, F29H85 uses the C29 core)
  - Note: `F29P32`/`F29P58` (also C29 core) are **not** in this group — they support the generic tasks but not Arc Fault/Motor Fault/Blower Imbalance; see "Generic-Task-Only Devices" below.

#### ARM Cortex-R5 Family
- **AM263**
  - Generic Tasks: Classification, Regression, Anomaly Detection, Forecasting
  - Specialized: Arc Fault, Motor Fault, Blower Imbalance
  - Compilation: Soft NPU (no hardware accelerator)
  - Note: `AM263P`/`AM261` are **not** in this group — see "Generic-Task-Only Devices" below.

#### ARM Cortex-M33 Family
- **MSPM33C32, AM13E2**
  - Generic Tasks: Classification, Regression, Anomaly Detection, Forecasting
  - Specialized: Arc Fault, Motor Fault, Blower Imbalance (MSPM33C32 also supports PIR Detection)
  - Compilation: Soft NPU
  - Note: `MSPM33C34` is a defined device constant but currently has **no** `tinyml-modelzoo` model targeting it for any task — it is unsupported until a model adds it to `target_devices`.

### Generic-Task-Only Devices
These devices support all four generic timeseries tasks but none of Arc Fault, Motor Fault, or Blower Imbalance:
- **F29P32, F29P58** (C29 core, Soft NPU)
- **AM263P, AM261** (Cortex-R5, Soft NPU)

### Partial Support Devices

#### C2000 Classification-Only Devices
- **F2837xS, F2838x, F2807x, F28002x, F28P551x**
  - Generic Tasks: Classification only (no Regression, Anomaly Detection, or Forecasting models target these devices)
  - Specialized: Arc Fault, Motor Fault, Blower Imbalance, ECG Classification
  - Compilation: Soft NPU

#### MSPM0G Family
- **MSPM0G3507, MSPM0G3519, MSPM0G5187**
  - Generic Tasks: Classification, Regression, Anomaly Detection, Forecasting (all four — not classification-only)
  - Specialized: Arc Fault, Motor Fault, ECG Classification, PIR Detection (**not** Blower Imbalance)
  - Compilation: Hard NPU available
  - Note: `MSPM0G3519` was missing from earlier revisions of this document.

#### Wireless/Connectivity Devices
- **CC2755, CC2745, CC1312, CC1314, CC1352, CC1354, CC35X1**
  - Generic Tasks: Classification only (no Regression, Anomaly Detection, or Forecasting models target these devices)
  - Specialized: PIR Detection (all seven), Motor Fault (all seven)
  - Compilation: CDE (CC2755, CC2745, CC35X1) / Soft NPU (CC1312, CC1314, CC1352, CC1354)
  - Note: Optimized for wireless/connectivity applications. `MotorFault_model_1/2/3_t` in `tinyml-modelzoo` explicitly target these devices so the `fan_blade_fault_classification` CC-device example configs resolve to a real model.

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
- CLS_100_NPU (100 parameters)
- CLS_500_NPU (500 parameters)
- CLS_1k_NPU (1K parameters)
- CLS_2k_NPU (2K parameters)
- CLS_4k_NPU (4K parameters)
- CLS_6k_NPU (6K parameters)
- CLS_8k_NPU (8K parameters)
- CLS_13k_NPU (13K parameters)
- CLS_20k_NPU (20K parameters)
- CLS_55k_NPU (55K parameters)
- CLS_ResAdd_3k (Residual addition, 3K parameters)
- CLS_ResCat_3k (Residual concatenation, 3K parameters)

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
- REGR_1k (1K parameters)
- REGR_2k (2K parameters)
- REGR_3k (3K parameters, MLP-based)
- REGR_4k (4K parameters, CNN-based)
- REGR_10k (10K parameters)
- REGR_13k (13K parameters, CNN-based)
- REGR_500_NPU (500 parameters, NPU)
- REGR_2k_NPU (2K parameters, NPU)
- REGR_6k_NPU (6K parameters, NPU)
- REGR_8k_NPU (8K parameters, NPU)
- REGR_20k_NPU (20K parameters, NPU)

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
- AD_1k (1K parameters)
- AD_4k (4K parameters)
- AD_16k (16K parameters)
- AD_17k (17K parameters)
- AD_Linear (Linear model)
- AD_500_NPU (500 parameters, NPU)
- AD_2k_NPU (2K parameters, NPU)
- AD_6k_NPU (6K parameters, NPU)
- AD_8k_NPU (8K parameters, NPU)
- AD_10k_NPU (10K parameters, NPU)
- AD_20k_NPU (20K parameters, NPU)
- Ondevice_Trainable_AD_Linear (On-device trainable)

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
- FCST_3k (3K parameters, MLP-based)
- FCST_13k (13K parameters, CNN-based)
- FCST_LSTM8 (LSTM with hidden size 8)
- FCST_LSTM10 (LSTM with hidden size 10)
- FCST_500_NPU (500 parameters, NPU)
- FCST_1k_NPU (1K parameters, NPU)
- FCST_2k_NPU (2K parameters, NPU)
- FCST_4k_NPU (4K parameters, NPU)
- FCST_6k_NPU (6K parameters, NPU)
- FCST_8k_NPU (8K parameters, NPU)
- FCST_10k_NPU (10K parameters, NPU)
- FCST_20k_NPU (20K parameters, NPU)

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

**Supported Devices:** AM13E2, AM263, F280013, F280015, F28002x, F28003, F28004, F2807x, F2837, F2837xS, F2838x, F28P55, F28P551x, F28P65, F29H85, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32 (derived from `tinyml-modelzoo`; see [Task Support Matrix](#task-support-matrix). Not included: AM261, AM263P, F29P32, F29P58, MSPM33C34, and the wireless/connectivity devices.)

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

**Supported Devices:** AM13E2, AM263, CC1312, CC1314, CC1352, CC1354, CC2745, CC2755, CC35X1, F280013, F280015, F28002x, F28003, F28004, F2807x, F2837, F2837xS, F2838x, F28P55, F28P551x, F28P65, F29H85, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32 (derived from `tinyml-modelzoo`; see [Task Support Matrix](#task-support-matrix). Same as Arc Fault plus the CC13xx/CC27xx/CC35X1 wireless devices, which `MotorFault_model_1/2/3_t` explicitly target. Not included: AM261, AM263P, F29P32, F29P58, MSPM33C34.)

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

**Supported Devices:** AM13E2, AM263, F280013, F280015, F28002x, F28003, F28004, F2807x, F2837, F2837xS, F2838x, F28P55, F28P551x, F28P65, F29H85, MSPM33C32 (derived from `tinyml-modelzoo`; see [Task Support Matrix](#task-support-matrix). Unlike Arc Fault/Motor Fault, this task has no MSPM0G model, so `MSPM0G3507`/`MSPM0G3519`/`MSPM0G5187` are **not** supported. Also not included: AM261, AM263P, F29P32, F29P58, MSPM33C34.)

---

### 8. PIR Detection
**Purpose:** Motion and presence detection using passive infrared sensors.

**Example Use Cases:**
- Occupancy detection
- Security systems
- Smart lighting control

**Available Models:**
- PIRDetection_model_1_t

**Supported Devices:** CC1312, CC1314, CC1352, CC1354, CC2745, CC2755, CC35X1, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32 (derived from `tinyml-modelzoo`; see [Task Support Matrix](#task-support-matrix). In addition to the wireless connectivity devices, MSPM0G and MSPM33C32 also support PIR Detection — this was missing from earlier revisions of this document.)

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
   - All listed devices except MSPM33C34 are available in the GUI
   - MSPM33C34 is in `TARGET_DEVICES_ADDITIONAL` (CLI only)

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

The Tiny ML ecosystem includes comprehensive example projects demonstrating various use cases. All examples are located in `tinyml-modelzoo/examples/` (a sibling repository to `tinyml-modelmaker`; there is no `examples/` directory inside `tinyml-modelmaker` itself).

### Timeseries Classification Examples

#### hello_world
- **Task Type:** Generic Timeseries Classification
- **Description:** Introductory example demonstrating basic timeseries classification workflow
- **Use Case:** Learning the Tiny ML framework basics
- **Recommended Devices:** All devices supporting classification
- **Key Features:** Simple dataset, fast training, ideal for getting started

#### ecg_classification
- **Task Type:** ECG Classification (`ecg_classification`) — its own task_type, distinct from Generic Timeseries Classification; see the [Task Support Matrix](#task-support-matrix)
- **Description:** ECG (electrocardiogram) signal classification for cardiac health monitoring
- **Use Case:** Medical device applications, heart rhythm analysis
- **Recommended Devices:** F280013, F280015, F28003, F28004, MSPM0G3507, MSPM0G3519, MSPM0G5187 (this task has no `tinyml-modelzoo` model targeting F28P55/F28P65/MSPM33C34, unlike the other C2000/NPU-based tasks in this document)
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

#### human_activity_recognition
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
- **Recommended Devices:** F28P65, F29H85, AM263 (`AM263P` is not supported for this task — see [Task Support Matrix](#task-support-matrix))
- **Key Features:** Real-time imbalance quantification, preventive maintenance

### Timeseries Regression Examples

#### induction_motor_speed_prediction
- **Task Type:** Generic Timeseries Regression
- **Description:** Induction motor speed prediction from current/voltage measurements
- **Use Case:** Motor control, sensorless speed estimation
- **Recommended Devices:** F280013, F280015, F28003, F28004 (motor control MCUs)
- **Key Features:** Real-time speed estimation, cost reduction (no speed sensor needed)

#### washing_machine_load_weighing
- **Task Type:** Generic Timeseries Regression
- **Description:** Washing machine parameter regression for smart control
- **Use Case:** Smart home appliances, energy optimization
- **Recommended Devices:** MSPM33C32
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
- **Recommended Devices:** MSPM33C32, AM263P
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
- **Recommended Devices:** F28P55, F28P65 (hardware NPU for low-latency detection); `MSPM33C34` is not supported (no `tinyml-modelzoo` model targets it — see [Task Support Matrix](#task-support-matrix))
- **Key Features:** Real-time anomaly detection, medical-grade signal processing, low-power operation
- **Configuration:** `config_anomaly_detection.yaml`

### Wireless/Connectivity Examples

#### pir_detection
- **Task Type:** PIR Detection
- **Description:** Passive Infrared (PIR) sensor-based motion and presence detection
- **Use Case:** Occupancy sensing, security systems, smart lighting
- **Recommended Devices:** CC2755, CC1312, CC1352, CC1354, CC35X1 (wireless connectivity devices)
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
# Using ModelMaker CLI (examples live in the sibling tinyml-modelzoo repo)
cd tinyml-modelmaker
./run_tinyml_modelmaker.sh ../tinyml-modelzoo/examples/<project_name>/config.yaml
```

For more information, refer to:
- `../tinyml-modelzoo/examples/` (relative to `tinyml-modelmaker/`) - Example configurations, one subdirectory per example project (e.g. `ac_arc_fault`, `blower_imbalance`, `ecg_classification`, `generic_timeseries_classification`, `pir_detection`, etc.)
