# Wi-Fi CSI Presence Detection
### -Pranav A, Abhijeet Pal, Adithya Thonse

## Overview

The Wi-Fi CSI Presence Detection application is an Edge AI solution that detects human presence using Channel State Information (CSI) from Wi-Fi signals. It runs on the CC35X1 microcontroller and classifies 2-second CSI windows into two states — **no presence** and **presence** — in real-time, enabling device-free occupancy sensing without any body-worn sensors.

## Problem and Solution

- Traditional PIR and radar sensors require line-of-sight and miss stationary occupants
- Wi-Fi CSI is pervasive, passive, and sensitive to subtle environmental changes caused by human bodies (even at rest)
- Manual threshold-based CSI detectors are brittle across rooms, orientations, and furniture arrangements
- Edge AI on Wi-Fi SoCs enables always-on, privacy-preserving presence sensing with no extra hardware

## Key Performance Targets

- Binary presence/no-presence classification in real-time
- Low-latency inference on CC35X1
- Robust to stationary occupants (sitting, typing, still)

## System Components

**1. Hardware:**

- CC35X1 Wi-Fi microcontroller [Link](https://www.ti.com/product/CC3531)

**2. Software:**

- Code Composer Studio 12.x or later
- TI Edge AI Studio
- tinyml-tensorlab CLI

## Dataset Information

Two dataset packages are provided for this example. Both contain CSI recordings at **128 Hz** across **52 usable subcarriers** (hardware indices 0–25 and 27–52; index 26 is the DC subcarrier and is excluded). Labels are binary: **class 0 = no presence**, **class 1 = presence**.

### wifi_presence_detection_dsi.zip

Compact in-house dataset and the default for the example `config.yaml`. Point `input_data_path` in `config.yaml` to the extracted dataset directory and run directly — no preprocessing required.

**Hardware:**
- Receiver: CC35X1 LP with XDS110 Debugger
- Transmitter: TP-Link AC1200 Mesh Wi-Fi Router (Archer C6)

**Collection environment:**
- Indoor conference room, area < 300 sq ft

**Data:**
- 400 × 2 s windows at 128 Hz per class (800 windows total)
- Presence class covers 4 activities: sitting, standing, walking, waving — activity labels are discarded; diversity is the only purpose

### wifi_presence_detection_dsd.zip

Larger, more diverse dataset intended for robust model evaluation. Requires running `preprocess_dsd.py` before training — update `IN_ROOT` and `OUT_ROOT` in the script, run it, then point `input_data_path` in `config.yaml` to the output directory.

- 650 MB+ compressed; over 6.4 million data points
- Captured across 8 recording days
- 13 fine-grained activity conditions collapsed to binary presence/no-presence
- Generalises well across different indoor rooms regardless of layout or size

### Preprocessing

**DSI dataset — `preprocess.py`:**

1. Reads CSVs from a `classes/no_presence` and `classes/presence` folder hierarchy
2. Drops metadata columns (`tx_mac`, `rx_mac`, `packet_no`, `hw_seq`)
3. Normalises timestamps (microseconds → seconds)
4. Detects session boundaries (gap > 0.5 s resets a session)
5. Resamples each session to a uniform grid at 128 Hz via linear interpolation
6. Writes output CSVs with 52 subcarrier magnitude columns

**Note:** The provided `wifi_presence_detection_dsi/` dataset is already preprocessed and ready to use — no need to run  `preprocess.py` on it again.

**DSD dataset — `preprocess_dsd.py`:**

1. Reads CSVs recursively; parses label and date from filename tokens (13 activity classes collapsed to binary)
2. Drops metadata columns and skips first **10 seconds** of each recording (transient removal)
3. Extracts 256-sample (2 s) windows with **30% overlap** stride (~179 samples)
4. Resamples each window to a uniform grid via linear interpolation
5. Assigns train/val/test splits by recording date:
   - **Train pool:** 2025-08-01, 2025-08-02, 2025-08-03, 2025-12-21, 2025-12-22, 2026-01-22
   - **Test:** 2025-12-20, 2026-01-23
   - **Val:** 20% of train pool via `GroupShuffleSplit` (grouped by date)
6. Writes output CSVs, file lists (`instances_train_list.txt`, `instances_val_list.txt`, `instances_test_list.txt`), and `metadata.csv`

## Feature Extraction Pipeline

The pipeline converts raw 52-subcarrier CSI magnitude windows into a 2-D time-frequency feature map:

| Step | Operation | Output shape |
|------|-----------|--------------|
| 1 | Downsample 128 Hz → 64 Hz (factor 2) | 64 samples × 52 subcarriers |
| 2 | Sliding window (2 s @ 64 Hz, 70% stride) | 128 × 52 |
| 3 | L2-normalise each time step across subcarriers (`L2_NORM_ROW`) | 128 × 52 |
| 4 | Real FFT along time axis (`FFT_FE`) | 65 × 52 |
| 5 | Keep positive-frequency half, drop Nyquist (`FFT_POS_HALF`) | 64 × 52 |
| 6 | Compute magnitude (`ABS`) | 64 × 52 |
| 7 | Log-magnitude: log₁₀(1.0 + \|F\|) (`LOG_DB`) | 64 × 52 |
| 8 | Real FFT across subcarrier axis, keep positive half (`FFT_COL`) | 64 × 26 |

**Final model input:** `(1, 26, 1, 64)` per window (batch × subcarrier-freq bins × 1 × time-freq bins).

## Model Architecture

| Model | Parameters | Task |
|-------|-----------|------|
| **SimpleCNN2D_BN_t** | ~4.7K | Binary presence classification |

The model accepts input of shape `(1, 26, 1, 64)` (batch × subcarrier-frequency bins × 1 × time-frequency bins) and uses a compact 2-D CNN with batch normalisation suitable for deployment on CC35X1.

## AI Model Performance

**`wifi_presence_detection_dsi.zip`**

| Metric | Float model | 8W8A Quantized (QAT) | Test set |
|--------|------------|----------------|----------|
| **Accuracy (Acc@1)** | 99.17% | 99.17% | 98.75% |
| **F1-Score** | 0.992 | 0.992 | — |
| **AUC ROC** | 0.983 | 0.978 | 0.985 |

**`wifi_presence_detection_dsd.zip`**

| Metric | Float model | 8W8A quantized (QAT) | Test set |
|--------|------------|----------------|----------|
| **Accuracy (Acc@1)** | 95.58% | 95.62% | 98.90% |
| **F1-Score** | 0.956 | 0.956 | — |
| **AUC ROC** | 0.986 | 0.986 | 0.997 |

## Training and Deployment Process

NOTE: Running the config YAML handles everything including feature extraction, training, quantization, and compilation.

1. **Preprocessing (one-time):**
   - DSI: `python preprocess.py` (update `IN_ROOT` and `OUT_ROOT` in the script)
   - DSD: `python preprocess_dsd.py` (update `IN_ROOT` and `OUT_ROOT` in the script)
   - Set `input_data_path` in `config.yaml` to the preprocessed output directory

2. **Training:**
   - Use TI Edge AI Studio (GUI) or tinyml-tensorlab (CLI)
   - Batch size: 32, Learning rate: 0.00005, Optimizer: Adam
   - Training epochs: 50

3. **Quantization:**
   - 8-bit Weight, 8-bit Activation (8W8A) quantization for reduced model size - default quantization setting.

4. **Compilation:**
   - TI Neural Network Compiler compiles the trained model for M33CDE on CC35X1
   - Generates `model.a` and `tvmgen_default.h` for firmware integration

## How to Run

After completing the repository setup, run the following command from the `tinyml-modelzoo` directory:

**Windows:**
```bash
.\run_tinyml_modelzoo.bat examples\wifi_csi_presence_detection\config.yaml
```

**Linux:**
```bash
./run_tinyml_modelzoo.sh examples/wifi_csi_presence_detection/config.yaml
```

## References

- [TI Model Training Guide](https://github.com/TexasInstruments/tinyml-tensorlab/tree/main)
- [TI Neural Network Compiler User Guide](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/)
- [CC35X1 Product Page](https://www.ti.com/product/CC3531)
