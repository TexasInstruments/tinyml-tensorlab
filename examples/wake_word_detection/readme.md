# Wake Word Detection

### TinyML ModelZoo Team

## Overview

The Wake Word Detection application is an Edge AI solution that runs on the MSPM0G5187 microcontroller with integrated Neural Processing Unit (NPU). It listens continuously for the wake word **"OK Kilby"** and triggers downstream voice-command processing only when the phrase is detected, keeping the always-on audio path fully on-device. This enables engineers to build low-power, privacy-preserving voice interfaces for smart home, industrial, and wearable devices without depending on cloud connectivity.

## Problem and Solution

- Always-on cloud-based wake word detection raises privacy concerns and requires constant connectivity
- Continuous audio streaming to the cloud is power-hungry and unsuitable for battery-operated devices
- Edge AI enables on-device, low-latency wake word detection with no audio leaving the device
- A dedicated NPU allows the wake word model to run continuously at low power while the rest of the system stays in a low-power state until triggered

## Key Performance Targets

- Less than 200ms detection latency from end of utterance
- High wake word detection rate with low false-accept rate on non-wake-word speech and background noise
- Low average power consumption suitable for always-on, battery-operated listening

## System Components

1. Hardware:
   - MSPM0G5187 microcontroller with integrated NPU [Link](https://www.ti.com/product/MSPM0G5187)
   - Audio input (microphone)

2. Software:
   - Code Composer Studio 12.x or later
   - MSPM0 SDK 2.08.00 or later
   - TI Edge AI Studio

## Dataset Operations

**Wake word:** "OK Kilby"

**Dataset source:** This is a TI-internal dataset, collected specifically for this application. It is **not** derived from any public or open-source dataset.

Data collection covers multiple scenarios:
- Wake Word: Multiple speakers saying "OK Kilby" with varying tone, distance, and speaking rate
- Background Noise: Household, office, and ambient environmental sounds
- Non-Wake-Word Speech: General conversational speech and other phrases, to reduce false accepts

Classes:
- Wake Word ("OK Kilby")
- Non-Wake-Word / Background

Since this dataset is TI-internal, it is not publicly downloadable and is not sourced from any third-party or open dataset.

**Dataset split included in this example:**

| Class | Files |
|-------|-------|
| Wake Word ("OK Kilby") | 6000 |
| Non-Wake-Word / Background | 4600 |

**Note:** This split is a small subset of the actual TI-internal dataset, which exceeds 50GB of data. For production-quality accuracy and robustness (different speakers, accents, distances, microphones, and noise conditions), customers should collect and add their own wake word and background data before training a model for deployment.

## Feature Extraction Pipeline

This model uses **learnable filterbanks** for feature extraction instead of a fixed MFCC/FFT front-end - the filterbank weights are trained jointly with the classifier rather than hand-designed.

1. Audio Sampling: 8000 Hz (8 kHz), 2-second window per inference (16000 samples total)
2. Framing: each frame covers a 20 ms context, i.e. 160 samples (20 ms at 8 kHz)
3. Learned Filterbank Front-End: strided Conv1D bank (kernel 64, stride 4) slides over the raw 8 kHz audio and projects it into 64 channels; a max-pool over each 160-sample (20 ms) context then produces one feature vector per frame. Filterbank weights are learned jointly during training, not fixed
4. Filterbank weights quantized to 2-bit; input audio quantized to 16-bit
5. Output features feed directly into the TCDS-ResNet backbone

## Model Architecture

The TCDS_ResNet_FB_NPU model pairs a learned filterbank front-end with a Temporal Channel-Decoupled Separable ResNet:

| Model | Notes |
|-------|-------|
| TCDS_ResNet_FB_NPU | **Recommended** - Filterbank front-end + TCDS-ResNet, optimized for MSPM0G5187 NPU |

**Architecture:**
- Front-end: Learned Filterbank (Conv1D, kernel 64, stride 4, 64 output channels, 2-bit weights)
- Stem: Conv3x1 -> BatchNorm -> ReLU (32 channels)
- Backbone: 3 x TCDSBasicBlock (channels 32 -> 48 -> 64 -> 96), each block:
  - Main branch: Depthwise(9x1) -> Pointwise(1x1) -> Depthwise(9x1) -> Pointwise(1x1)
  - Skip branch: 1x1 projection when stride != 1 or channels change, else Identity
  - Output: ReLU(main + skip)
- Head: Dropout -> Flatten -> Fully Connected
- Output: 2 classes (Wake Word, Non-Wake-Word/Background)
- Mixed precision: stem and head layers at 8-bit, residual blocks at 2-bit

## Performance Metrics for MSPM0G5187 with NPU

- Detection latency: <200ms end-to-end
- Memory footprint suitable for always-on operation within MSPM0G5187 SRAM/Flash budget
- Low average power consumption due to on-device NPU inference and 2-bit backbone quantization

## Training and Deployment Process

NOTE: Running the config yaml takes care of everything including feature extraction, training, quantization and compilation.

1. Training:
   - Use TI Edge AI Studio (GUI) or tinyml-tensorlab (CLI)
   - Batch size: 256, Learning rate: 0.001, Weight decay: 1e-5
   - Enable Quantize-Aware Training for INT8/mixed-precision accuracy
2. Quantization:
   - Mixed precision quantization: 8-bit stem/head, 2-bit residual blocks, 2-bit filterbank front-end
3. Compilation:
   - TI Neural Network Compiler converts trained model
   - Generates model.a, interface headers, and configuration

## How to Run

After completing the repository setup, run the following command from the `tinyml-modelzoo` directory:

**Windows:**
```bash
.\run_tinyml_modelzoo.bat examples\wake_word_detection\config_MSPM0.yaml
```

**Linux:**
```bash
./run_tinyml_modelzoo.sh examples/wake_word_detection/config_MSPM0.yaml
```

## References

- MSPM0G5187 Technical Reference Manual [Link](https://www.ti.com/product/MSPM0G5187)
- [TI Neural Network Compiler Guide](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/)
- TI Model Training Guide: [tinyml-tensorlab](https://github.com/TexasInstruments/tinyml-tensorlab/tree/main)
- EdgeAI Software Guide: https://dev.ti.com/tirex/explore/node?node=A__AKCnvqDed-Plz2JO5Umb3Q__MSPM0-SDK__a3PaaoK__LATEST
- MSPM0 SDK: https://www.ti.com/tool/MSPM0-SDK
