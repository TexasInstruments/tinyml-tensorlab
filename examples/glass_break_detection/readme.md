# Glass Break Detection

### TinyML ModelZoo Team

## Overview

The Glass Break Detection application is an Edge AI solution that runs on the MSPM0G5187 microcontroller with integrated Neural Processing Unit (NPU). It detects glass breaking events in security and home automation systems by analyzing acoustic signatures. This application combines FFT-based feature extraction with deep learning inference to achieve high detection accuracy while maintaining low power consumption and fast response times. This enables engineers to develop intelligent security systems, intrusion detection devices, and smart home safety applications leveraging the power efficiency and performance of Edge AI on microcontrollers.

## Problem and Solution

- Glass break detection is critical for security systems and home automation
- Traditional threshold-based detectors suffer from high false positive rates
- Edge AI enables robust acoustic pattern recognition and multi-feature analysis
- On-device processing ensures privacy and eliminates cloud dependency

## Key Performance Targets

- Less than 100ms response time
- Greater than 95% detection accuracy
- Low power consumption for battery-operated devices

## System Components

1. Hardware:
   - MSPM0G5187 microcontroller with integrated NPU [Link](https://www.ti.com/product/MSPM0G5187)
   - Audio input (microphone/acoustic sensor)

2. Software:
   - Code Composer Studio 12.x or later
   - MSPM0 SDK 2.08.00 or later
   - TI Edge AI Studio

## Dataset Operations

Data collection requires multiple scenarios:
- Glass Breaking: Various glass types (tempered, laminated, plate glass)
- Background Noise: Normal household sounds, music, voices
- Similar Sounds: Dishes breaking, metal impacts, door slams

Example dataset includes:
- 100+ captures per class
- Classes: Glass Break, Background Noise, Similar Non-Glass Sounds

All data prior to augmentation is sourced from the following public datasets:
- DEMAND (noise): [Link](https://zenodo.org/records/1227121)
- FSDK50K: [Link](https://zenodo.org/records/4060432)
- ESC-50: [Link](https://github.com/karolpiczak/esc-50)
- Google Speech Commands

## Feature Extraction Pipeline

This model uses **FFT-based feature extraction** over a 1-second audio window.

1. ADC Sampling: 1-second window per inference, split into frames of 512 samples (1 frame = 512 samples)
2. Real FFT: 512-point FFT using ARM CMSIS-DSP, computed per frame
3. Complex Magnitude Calculation
4. DC Removal
5. Binning: Average adjacent FFT bins → 32 features per frame
6. Normalization to INT8 range
7. Frame Concatenation: 86 frames are stacked and fed to the model (86 × 32 features)

## Model Architecture

The DSCNN_GB_NPU model is a lightweight DSCNN architecture:

| Model | Parameters | Flash | Inference Time | Expected Accuracy | Notes |
|-------|------------|-------|----------------|-------------------|-------|
| DSCNN_GB_NPU | ~6K | ~8 KB | <10 ms | >95% | **Recommended** - Optimized for MSPM0G5187 |

**Architecture:**
- Input: (N, 1, 86, 32) [86 time frames × 32 FFT features]
- BatchNorm → Conv 8×8/s4 → [DW 4×4 + PW 1×1] × 2 → Dropout → GlobalAvgPool → FC
- Output: 3 classes (Glass Break, Background, Similar Sounds)
- DW/PW convolutions quantized at 4-bit (mixed precision), activations at 8-bit

**Training hyperparameters:**
- Epochs: 20, Batch size: 32, Learning rate: 0.001 (Adam)

## Performance Metrics for MSPM0G5187 with NPU

- End-to-end latency: <100ms
- Detection accuracy: >95%
- False positive rate: <5%
- Memory footprint: <32KB SRAM, <128KB Flash

## Training and Deployment Process

NOTE: Running the config yaml takes care of everything including feature extraction, training, quantization and compilation.

1. Training:
   - Use TI Edge AI Studio (GUI) or tinyml-tensorlab (CLI)
   - Epochs: 20, Batch size: 32, Learning rate: 0.001, Optimizer: Adam
   - Enable Quantize-Aware Training (QAT) for quantized accuracy
2. Quantization:
   - Mixed precision quantization - Enabled in the config file by default as well as in Edge AI Studio
   - Activations: 8-bit; DW/PW convolution weights: 4-bit
   - ~4-8x reduction in model size vs. FP32
3. Compilation:
   - TI Neural Network Compiler converts trained model
   - Generates model.a, interface headers, and configuration

## How to Run

After completing the repository setup, run the following command from the `tinyml-modelzoo` directory:

**Windows:**
```bash
.\run_tinyml_modelzoo.bat examples\glass_break_detection\config_MSPM0.yaml
```

**Linux:**
```bash
./run_tinyml_modelzoo.sh examples/glass_break_detection/config_MSPM0.yaml
```

## References

- MSPM0G5187 Technical Reference Manual [Link](https://www.ti.com/product/MSPM0G5187)
- [TI Neural Network Compiler Guide](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/)
- TI Model Training Guide: [tinyml-tensorlab](https://github.com/TexasInstruments/tinyml-tensorlab/tree/main)
- EdgeAI Software Guide: https://dev.ti.com/tirex/explore/node?node=A__AKCnvqDed-Plz2JO5Umb3Q__MSPM0-SDK__a3PaaoK__LATEST
- MSPM0 SDK: https://www.ti.com/tool/MSPM0-SDK
- DEMAND dataset: https://zenodo.org/records/1227121
- FSDK50K dataset: https://zenodo.org/records/4060432
- ESC-50 dataset: https://github.com/karolpiczak/esc-50
