# DC Arc Fault Detection

## Overview

This folder contains example configurations for DC arc fault detection (classification and anomaly detection) on two kinds of sensing front ends: DSK and DSI datasets. Each YAML file is a complete pipeline (data loading, feature extraction, training, quantization and compilation).

## Configurations

| Config | Task type | Target device | Dataset | Model | Training settings |
|--------|-----------|---------------|---------|-------|-------------------|
| `config_dsk.yaml` | `arc_fault` | F28P55 | `dc_arc_fault_example_dsk` | `CLS_1k_NPU` | batch 32, 10 epochs, early stopping (patience 5); feature extraction `FFT1024Input_256Feature_1Frame_Full_Bandwidth` |
| `config_dsk_my_model.yaml` | `arc_fault` | F28P55 | `dc_arc_fault_example_dsk` | `MY_CLS_2k_NPU` (user-defined model example) | batch 32, 10 epochs, early stopping (patience 5); same feature extraction as `config_dsk.yaml` |
| `config_dsi.yaml` | `arc_fault` | F28P55 | `arc_fault_example_dsi` (`arc_fault_classification_dsi.zip`) | `CLS_13k_NPU` | batch 2048, 10 epochs; Downsample (313000 -> 3130 Hz) + SimpleWindow, frame size 256, stride 0.01 |
| `config_dsi_ondevice_learning.yaml` | `arc_fault` | AM13E2 | `arc_fault_example_dsi_odl` (`arc_fault_classification_dsi.zip`) | `CLS_24k_NPU` | batch 2048, 1 epoch, on-device training enabled (last 3 layers trainable), quantization 2; frame size 128, stride 0.01 |
| `config_anomaly_detection_dsk.yaml` | `generic_timeseries_anomalydetection` | F28P55 | `arc_fault_example_dsk_ad` (`arc_fault_dsk_anomaly_detection.zip`) | `AD_17k` | batch 1024, 250 epochs, learning rate 0.0001, quantization 1; SimpleWindow + Downsample (100 -> 1), frame size 1024; `forced_soft_npu_preset` |
| `config_anomaly_detection_dsi.yaml` | `generic_timeseries_anomalydetection` | F28P55 | `arc_fault_example_dsi_ad` (`arc_fault_dsi_anomaly_detection.zip`) | `AD_17k` | batch 256, 100 epochs, quantization 1; SimpleWindow + Downsample (100 -> 1), frame size 256; `forced_soft_npu_preset` |

The DSI classification configs apply gain variations as augmentation (arc: 0.9 to 1.1, normal: 0.8 to 1.2).

## How to Run

Run the following command from the `tinyml-modelzoo` directory (replace the config name with any of the files above):

**Windows:**
```bash
.\run_tinyml_modelzoo.bat examples\dc_arc_fault\config_dsk.yaml
```

**Linux:**
```bash
./run_tinyml_modelzoo.sh examples/dc_arc_fault/config_dsk.yaml
```
