# Blower Imbalance Detection

## Overview

This example classifies blower/fan imbalance conditions from 3-channel vibration time series data using the `fan_blower_imbalance_dsh` dataset preset.

## Configuration (config.yaml)

- Task type: `motor_fault`
- Target device: `F28P55`
- Dataset: `fan_blower_imbalance_dsh`
- Feature extraction: `Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1` (3 variables, no extra data processing transforms)
- Model: `CLS_1k_NPU`
- Training: batch size 256, 20 epochs, early stopping enabled (patience 5), `num_gpus: 0`
- Testing and compilation: default settings

## How to Run

Run the following command from the `tinyml-modelzoo` directory:

**Windows:**
```bash
.\run_tinyml_modelzoo.bat examples\blower_imbalance\config.yaml
```

**Linux:**
```bash
./run_tinyml_modelzoo.sh examples/blower_imbalance/config.yaml
```
