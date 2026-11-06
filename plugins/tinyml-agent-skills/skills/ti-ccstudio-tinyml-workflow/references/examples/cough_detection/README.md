# Cough Detection on MSPM0G5187

Binary audio classification (**cough** vs **other**) running fully on-device on the
[LP-MSPM0G5187 LaunchPad](https://www.ti.com/tool/LP-MSPM0G5187) using the TinyEngine NPU.
No cloud, no OS, no external ML framework.

---

## Table of Contents

1. [Hardware & Toolchain](#1-hardware--toolchain)
2. [Data Collection](#2-data-collection)
3. [Dataset Preparation (v2)](#3-dataset-preparation-v2)
4. [Training Configuration](#4-training-configuration)
   - 4.1 [MSPM0_LPC_8820 Feature Extraction Preset](#41-mspm0_lpc_8820-feature-extraction-preset)
   - 4.2 [TCDS_ResNet_NPU Model](#42-tcds_resnet_npu-model)
5. [Training Results](#5-training-results)
6. [Memory Footprint](#6-memory-footprint)
7. [Quick Start](#7-quick-start)

---

## 1. Hardware & Toolchain

| Item | Detail |
|------|--------|
| MCU | MSPM0G5187 — Arm Cortex-M0+ @ 80 MHz |
| NPU | TinyEngine™ — 2.56 GOPS @ 80 MHz, INT8 weights |
| Memory | 128 KB flash, 32 KB SRAM |
| Audio interface | Native I2S/TDM peripheral |
| Power | 1.5 µA standby, 103 µA/MHz active |
| Data capture | TI EdgeAI Studio (browser GUI, no firmware required) |
| Training | TI TensorLab (YAML-configured) |
| Compiler | TI Neural Network Compiler (NNC) → `model.a` |
| IDE | Code Composer Studio (CCS), no RTOS |

---

## 2. Data Collection

Audio was captured with **TI EdgeAI Studio** connected to the LP-MSPM0G5187 via USB.
EdgeAI Studio streams the raw I2S FIFO directly to a labeled CSV file — no custom
firmware or Python environment needed at this stage.

### Signal chain

```
PCM1809 MEMS mic
    │  I2S stereo, 44.1 kHz BCLK
    ▼
MSPM0G5187 I2S peripheral
    │  ISR deinterleaves L channel, raw >> 10, clamp → int16
    │  Decimates 5:1  →  8820 Hz mono
    ▼
EdgeAI Studio (USB)
    │  Streams raw FIFO samples
    ▼
Labeled CSV file  (one file per recording, one label per session)
```

The FIFO delivers interleaved L+R words at **88,200 words/s**. After
deinterleaving and 5:1 decimation the effective audio rate is **8820 Hz** — the
same rate the firmware uses at inference time. This is a simple integer decimation
(take every 5th sample, no anti-alias filter), which is critical: using a
polyphase/LP-filtered resampler during training would produce different spectral
aliasing than the firmware's plain decimation, causing a train/inference mismatch.

### Recordings

| File | Class | Duration |
|------|-------|----------|
| coughing (male 1) | cough | ~57 s |
| coughing (male 2) | cough | ~57 s |
| coughing (female) | cough | ~57 s |
| hello / greetings | other | ~57 s |
| meeting speech | other | ~57 s |
| people chatting | other | ~57 s |

Total raw audio: **~6 minutes** across 6 recordings.
Silence and white noise were also recorded and used exclusively as a noise bank
for augmentation — they are never added as training clips.

---

## 3. Dataset Preparation (v2)

`prepare_dataset.py` converts the raw EdgeAI CSV files to segmented 8820 Hz WAV
clips ready for TensorLab. It mirrors the firmware signal path exactly so that
every clip the model trains on is byte-for-byte equivalent to what the firmware
would produce from the same audio.

### Pipeline (one CSV file)

```
Raw CSV  (int32 FIFO samples at 88200 words/s)
    │
    │  Step 1: raw >> 10, clamp to [-32768, 32767]  → int16
    │          matches firmware ISR:  s32 = word >> 10; clamp
    │
    │  Step 2: deinterleave  →  L-channel only  (44100 Hz)
    │          matches firmware ISR:  read L word, discard R
    │
    │  Step 3: decimate by 5  →  8820 Hz
    │          matches firmware ISR:  dec++; if dec >= 5 → emit sample
    │
    │  Step 4: segment into 2-second clips  (17640 samples)
    │          50% overlap  →  stride = 8820 samples (1 s)
    │          ~55 clips per 57-second recording
    │
    │  Step 5: augmentation  (cough class only)
    │          gain=1.0   →  original clip (written as-is)
    │          gain=0.15  →  attenuated + 10 dB SNR noise mix
    │                        forces RMS overlap with "other" class
    ▼
labeled WAVs:  cough/cough_NNNNN.wav
               other/other_NNNNN.wav
```

### Augmentation rationale

A cough is inherently louder than background speech. Without augmentation the model
learns to threshold on RMS energy — it classifies "loud" as cough and "quiet" as
other. The `gain=0.15` copies of each cough clip bring the cough RMS into the same
range as speech clips (RMS ~550-1160 for attenuated, ~2000-15000 for normal).
With both classes spanning the same RMS buckets, the model is forced to discriminate
on **spectral shape**, not loudness. This is the primary reason cough recall reached
100% in validation.

Silence and white noise are mixed in at a fixed 10 dB SNR relative to the attenuated
signal. They are never used as standalone "other" clips — they contain no useful
spectral content and would confuse the class boundary.

### Output dataset structure

```
dataset/
└── cough_vs_other/
    └── classes/
        ├── cough/        # 330 clips (165 original + 165 augmented)
        └── other/        # 165 clips (no augmentation)
```

---

## 4. Training Configuration

The full training run is specified in `config.yaml`:

```yaml
common:
  target_module: audio
  task_type: audio_classification
  target_device: MSPM0G5187

dataset:
  dataset_name: cough_detection
  input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/cough_detection.zip

data_processing_feature_extraction:
  feature_extraction_name: MSPM0_LPC_8820

training:
  model_name: TCDS_ResNet_NPU
  batch_size: 64
  training_epochs: 40
  num_gpus: 0
  learning_rate: 0.01
  weight_decay: 4e-4

testing: {}

compilation:
  compile_preset_name: default_preset
```

| Field | Value | Meaning |
|-------|-------|---------|
| `target_device` | `MSPM0G5187` | NNC compiles specifically for this NPU |
| `feature_extraction_name` | `MSPM0_LPC_8820` | Selects the firmware-matched LPC preset (see below) |
| `model_name` | `TCDS_ResNet_NPU` | Temporal CNN with depthwise-separable convolutions |
| `training_epochs` | `40` | Sufficient for this dataset size; increase to 100+ for larger datasets |
| `learning_rate` | `0.01` | Conservative — appropriate for a small dataset (~500 clips) |
| `compile_preset_name` | `default_preset` | Runs NNC to produce `model.a` + `tvmgen_default.h` |

---

### 4.1 MSPM0_LPC_8820 Feature Extraction Preset

This preset is the critical link between training and firmware. Every parameter
matches a constant in the C firmware (`lpc_defs.h`, `feature_extraction.c`).
If any parameter differs between training and firmware, the model sees a different
feature distribution at inference time and accuracy collapses.

#### Parameters

| Parameter | Value | Firmware equivalent |
|-----------|-------|---------------------|
| `sampling_rate` | 8820 Hz | `SAMPLE_RATE_OUT = 44100 / DECIMATE` |
| `audio_duration_ms` | 2000 ms | 100 frames × 20 ms = 2 s context window |
| `audio_feature` | `"LPC"` | Linear Predictive Coding |
| `lpc_order` | 10 | `LPC_ORDER = 10` in `lpc_defs.h` |
| `nlpc` | 70 | `NUM_FREQ = 70` frequency bins |
| `frame_length_ms` | 30 ms | `WINDOW_LEN = 264` samples @ 8820 Hz |
| `frame_step_ms` | 20 ms | `FRONTEND_FRAME_LEN = 176` samples @ 8820 Hz |

The input tensor delivered to the model is shape **(1, 1, 100, 70)** in `int8`:
100 time frames × 70 frequency bins, one audio channel, batch size 1.

#### What happens inside the preset (per 20 ms frame)

```
Raw audio frame  (176 new samples + 88 overlap from previous frame = 264 samples)
    │
    │  1. Pre-emphasis
    │        y[n] = x[n] - x[n-1]        high-pass to flatten vocal tract slope
    │
    │  2. Q15 Hamming window
    │        frame[i] = (frame[i] * HAMMING_FX[i] + 16384) >> 15
    │        HAMMING_FX: 264-entry Q15 fixed-point Hamming table
    │        identical table used in both Python training and C firmware
    │
    │  3. Autocorrelation  (lags 0..10)
    │        r[k] = Σ frame[n] · frame[n+k]
    │        normalised by r[0];  off-diagonal terms × (1/1.1)  (noise floor regularisation)
    │
    │  4. Levinson-Durbin recursion  (order 10)
    │        solves for LPC coefficients a[1..10] and gain G
    │        models the vocal tract as a 10th-order all-pole IIR filter
    │
    │  5. Frequency response at 70 bins  (112 Hz – 3946 Hz, log-spaced)
    │        H(f) = G / |1 + Σ a[k]·e^{-j2πfk/Fs}|
    │        magnitude squared evaluated at each of the 70 frequencies
    │
    │  6. Log scale
    │        feature[i] = 10 · log10( |H(fi)|² )      dB spectral envelope
    │
    │  7. Quantise to int8  (for NPU input tensor)
    │        scale to Q8 int16:  lpc_q8 = feature * 256
    │        map to int8:        int8_val = clip((lpc_q8 + 256) * 157 >> 13, -128, 127)
    ▼
70 int8 values — one row of the (100 × 70) input tensor
```

The result is a compact representation of the **spectral envelope** of each audio
frame. Cough produces a broadband, relatively flat envelope. Speech produces
narrow peaks (formants) at 700-3000 Hz. LPC captures this shape difference in
just 70 numbers, making it ideal for a tiny embedded model.

---

### 4.2 TCDS_ResNet_NPU Model

**Full name:** Temporal Channel-Decoupled Separable ResNet for NPU

#### Design principles

The model is designed around three constraints of the MSPM0G5187:

1. **INT8 only** — the TinyEngine NPU executes INT8 multiply-accumulate. No
   floating-point at inference time.
2. **14.45 KB activation SRAM** — the largest intermediate tensor must fit within
   the 32 KB SRAM alongside firmware buffers.
3. **1D temporal convolutions only** — the input is permuted from `(N, 1, T, F)` to
   `(N, F, T, 1)` before any convolution. This places the 70 LPC coefficients on
   the channel axis and time on the spatial axis, so every convolution kernel is
   `K×1` (temporal only). This halves the parameter count vs a 2D approach.

#### Input permutation

```
Input from LPC preset:  (N, 1, 100, 70)   — batch × channel × time × features
        │
        │  BatchNorm on raw input
        │  Permute axes:  (N, 1, T, F) → (N, F, T, 1)
        ▼
                         (N, 70, 100, 1)   — LPC bins become channels
```

#### Architecture

| Stage | Block | In channels | Out channels | Stride | Kernel |
|-------|-------|-------------|--------------|--------|--------|
| Stem | Conv2d | 70 | 16 | 1 | 3×1 |
| Block 1 | TCDSBasicBlock | 16 | 24 | 2 | 9×1 |
| Block 2 | TCDSBasicBlock | 24 | 24 | 1 | 9×1 |
| Block 3 | TCDSBasicBlock | 24 | 32 | 2 | 9×1 |
| Block 4 | TCDSBasicBlock | 32 | 32 | 1 | 9×1 |
| Block 5 | TCDSBasicBlock | 32 | 48 | 2 | 9×1 |
| Block 6 | TCDSBasicBlock | 48 | 48 | 1 | 9×1 |
| Head | GAP + Dropout(0.5) + FC | 48 | num_classes | — | — |

Each **TCDSBasicBlock** is a depthwise-separable convolution block:
- Depthwise conv `K×1` (spatial, per-channel) — captures temporal patterns
- Pointwise conv `1×1` — mixes channels
- BatchNorm + ReLU after each conv
- No residual/skip connections — avoids quantized-add incompatibility with TI QAT

Depthwise-separable factorisation reduces parameter count by ~8-10× vs standard
convolutions at the same receptive field, which is the key to fitting INT8 weights
in 45 KB of flash.

#### Model I/O

| | Shape | Type |
|--|-------|------|
| Input | (1, 1, 100, 70) | int8 |
| Output | (1, 2) | int8 — scores for [cough, other] |
| Inference call | `tvmgen_default_run(inputs, outputs)` | TVM NPU runtime |

---

## 5. Training Results

Results on the v2 dataset (562 clips total, 80/20 train/val split):

| Metric | Value |
|--------|-------|
| Best validation accuracy | **88.5%** |
| Cough recall | **100%** (0 false negatives) |
| Cough precision | ~85% |
| Cough F1 | ~92% |
| Quantization | INT8 (NPU-ready, no accuracy penalty) |

### Confusion matrix (validation set)

```
                  Predicted
                  Cough    Other
Actual  Cough  [   99        0  ]   ← 100% recall
        Other  [   17       32  ]
```

Zero missed coughs. The 17 false positives are speech segments that
are acoustically similar to cough — addressed in the v3 dataset by adding
more diverse speech and extending augmentation to speech classes.

### Why 100% cough recall matters

For health monitoring use cases (hospital, wearable), missing a cough event is
a more serious error than a false alarm. The model's conservative decision boundary
— it leans toward calling something cough when uncertain — is intentional. The
debounce logic in firmware (`DETECT_THRESH = 8` frames = 160 ms of consecutive
cough predictions) filters out isolated false positives at the system level.

---

## 6. Memory Footprint

Reported by TI NNC after compiling `model.a` for MSPM0G5187:

| Segment | Size | Goes to |
|---------|------|---------|
| Code (firmware glue) | 4.58 KB | Flash |
| RO Data (weights + params) | 45.04 KB | Flash |
| RW Data (activations) | 14.45 KB | SRAM |
| **Total** | **64.08 KB** | |

| Resource | Used | Available | Utilisation |
|----------|------|-----------|-------------|
| Flash | 49.62 KB | 128 KB | **39%** |
| SRAM | 14.45 KB | 32 KB | **45%** |

The remaining flash and SRAM headroom is available for firmware logic, audio
buffers, and future model improvements.

---

## 7. Quick Start

### Prerequisites

- TI TensorLab installed and on `PATH`
- Python 3.10+, `numpy`, `pandas`, `scipy`

### Step 1 — Prepare the dataset (if using your own recordings)

Run `prepare_dataset.py` with `--data-dir` pointing to your EdgeAI Studio CSV
exports and `--output-dir` for the output location:

```bash
python prepare_dataset.py --data-dir /path/to/csvs --output-dir /path/to/output
```

The script prints a summary of clips per class and an RMS distribution table.
Verify that both classes have clips in overlapping RMS buckets before training.

### Step 2 — Train

```bash
tinyml_modelmaker config.yaml
```

TensorLab will:
1. Load and preprocess WAVs using the `MSPM0_LPC_8820` preset
2. Train `TCDS_ResNet_NPU` for 40 epochs
3. Quantize to INT8
4. Compile via NNC to `model.a` and `tvmgen_default.h`

### Step 3 — Deploy

Copy `model.a` and `tvmgen_default.h` into the CCS project's `model/` directory
and rebuild. The firmware is structured as a five-stage bare-metal pipeline:

```
I2S ISR  →  Buffer Manager  →  Feature Extraction  →  NPU Inference  →  Post-processing
(8820 Hz)   (sliding window)   (LPC → int8 tensor)   (tvmgen_default)   (debounce + LED)
```

Inference runs every **20 ms** (one LPC frame). The LED triggers after
8 consecutive cough frames (160 ms of sustained detection).

---

*For a full description of the firmware architecture, LPC mathematics, and
quantization chain, see `TI_Cough_Detection_Technical_Document.docx`.*
