# Battery Management System (BMS) - State of Charge Estimation

## Overview

  The BMS SOC Estimation is an Edge AI solution that runs on the MSPM0G5187 microcontroller with integrated Neural Processing Unit (NPU). It estimates the State of Charge (SOC) of a lithium-ion battery cell from time-series sensor readings, enabling accurate battery management without dedicated SOC-estimation hardware or lookup-table-based coulomb counting alone. This application enables engineers to develop battery management products with on-device, NPU-accelerated SOC estimation.

## Problem and Solution

  - Accurate SOC estimation is critical for battery management systems (BMS) to prevent over-charge/over-discharge, estimate remaining runtime, and extend battery life
  - Traditional coulomb-counting and OCV (open-circuit voltage) lookup methods drift over time and are sensitive to sensor noise and battery aging
  - A short window of raw voltage/current/temperature alone is insufficient — SOC depends on charge/discharge history, not just an instantaneous snapshot
  - Edge AI enables a lightweight regression model, informed by both instantaneous readings and accumulated charge, to run directly on a resource-constrained MCU

## Key Performance Targets

  - Test RMSE < 2% SOC (stretch target; current best model achieves 2.56%)
  - R² > 0.99 on held-out test data
  - Model footprint suitable for MSPM0G5187 flash/SRAM budget

## System Components

1. Hardware:
    - MSPM0G5187 microcontroller with integrated NPU [Link](https://www.ti.com/product/MSPM0G5187)

2. Software:
    - Code Composer Studio 12.x or later
    - MSPM0 SDK 2.08.00 or later
    - tinyml-tensorlab (CLI)

## Dataset Operations

  The model is trained on the LG SoC dataset (`LG_18650HG2_Liion_Battery_Data`), a public battery cell time-series dataset containing charge/discharge cycles logged at fixed intervals.

  - Source dataset: 4491 raw files

  Each sample file provides 4 input variables, in fixed column order:
  - `voltage_V`
  - `current_A`
  - `temperature_C`
  - `capacity_Ah` 

  Target variable: `SOC` (State of Charge, as a fraction/percentage).

## Feature Extraction Pipeline

  1. Windowing: raw `SimpleWindow`, frame_size = 128 samples, stride_size = 0.05 (dense overlap)
  2. 4 input channels per window: voltage, current, temperature, capacity (coulomb count)

## Model Architecture

| Model | Parameters | Flash (MSPM0G5187) | Quantization | Notes |
|-------|------------|---------------------|--------------|-------|
| REGR_20k_NPU | ~20,000 | 26.39 KB total (Code 2.43KB + RO 21.52KB + RW 2.45KB) | Mode 2 (NPU) | **Recommended** - trained and validated on the full 4491-file dataset |

## Performance Metrics (REGR_20k_NPU, full dataset, best run)

  - Test RMSE: 2.56% (0.0256)
  - Test R²: 0.99
  - Best-epoch R² during training: 0.987
  - Training epochs: 50, stride_size: 0.05
  - Device: MSPM0G5187, quantization mode 2 (NPU)

## Training and Deployment Process

NOTE: Running the config yaml takes care of everything including feature extraction, training, quantization and compilation.

1. Training:
   - Use tinyml-tensorlab (CLI) via the tinyml-workflow-agent skill or directly with the config file
   - Quantization mode 2 (TI NPU-optimized) enabled by default in the config file
2. Quantization:
   - INT8 NPU-optimized quantization, enabled via `quantization: 2` in the config
3. Compilation:
   - TI Neural Network Compiler converts the trained model
   - Generates `mod.a`, `tvmgen_default.h`, and configuration for MSPM0G5187 with NPU offload (GCONV/DWCONV/PWCONV/AVGPOOL/FC confirmed)

## How to Run

After completing the repository setup, run the following command from the `tinyml-modelzoo` directory:

**Linux:**
```bash
./run_tinyml_modelzoo.sh examples/bms_soc_estimation/config_MSPM0.yaml
```

**Windows:**
```bash
.\run_tinyml_modelzoo.bat examples\bms_soc_estimation\config_MSPM0.yaml
```

## References

- MSPM0G5187 Technical Reference Manual [Link](https://www.ti.com/product/MSPM0G5187)
- [TI Neural Network Compiler Guide](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/)
- TI Model Training Guide: [tinyml-tensorlab](https://github.com/TexasInstruments/tinyml-tensorlab/tree/main)
- MSPM0 SDK: https://www.ti.com/tool/MSPM0-SDK
