# Anomaly Detection On-Device Training Library

## Table of Contents

- [1. Overview](#1-overview)
- [2. State Machine](#2-state-machine)
- [3. Data Structures](#3-data-structures)
  - [3.1 AnomalyDetectionConfig_t](#31-anomalydetectionconfig_t--user-configuration)
  - [3.2 AnomalyDetectionContext_t](#32-anomalydetectioncontext_t--internal-state)
  - [3.3 Result and Metrics Structures](#33-result-and-metrics-structures)
- [4. API Reference](#4-api-reference)
  - [4.1 AnomalyDetection_Init](#41-anomalydetection_init)
  - [4.2 AnomalyDetection_StartTraining](#42-anomalydetection_starttraining)
  - [4.3 AnomalyDetection_StopTraining](#43-anomalydetection_stoptraining)
  - [4.4 AnomalyDetection_ProcessTrainingFrame](#44-anomalydetection_processtrainingframe)
  - [4.5 AnomalyDetection_ProcessThresholdFrame](#45-anomalydetection_processthresholdframe)
  - [4.6 AnomalyDetection_ProcessInferenceFrame](#46-anomalydetection_processinferenceframe)
  - [4.7 AnomalyDetection_ForwardPass](#47-anomalydetection_forwardpass)
  - [4.8 AnomalyDetection_GetMode / GetMetrics](#48-anomalydetection_getmode--anomalydetection_getmetrics)
- [5. Training Features — Detailed Explanation](#5-training-features--detailed-explanation)
  - [5.1 EMA-Smoothed Validation Loss](#51-ema-smoothed-validation-loss)
  - [5.2 Early Stopping](#52-early-stopping)
  - [5.3 Best Weight Checkpointing](#53-best-weight-checkpointing)
- [6. Threshold Calculation — Detailed Explanation](#6-threshold-calculation--detailed-explanation)
- [7. Logging System](#7-logging-system)
- [8. API Quick Reference](#8-api-quick-reference)
- [9. Usage Examples](#9-usage-examples)
  - [9.1 Minimal API Patterns](#91-minimal-api-patterns)
  - [9.2 Initialization and Pre-Training Inference](#92-initialization-and-pre-training-inference)
  - [9.3 The Complete Training Loop](#93-the-complete-training-loop)
  - [9.4 Threshold Calculation](#94-threshold-calculation)
  - [9.5 Post-Training Inference](#95-post-training-inference)
  - [9.6 Putting It All Together](#96-putting-it-all-together)
- [10. Further Reading](#10-further-reading)

---

## 1. Overview

The `anomaly_detection_odt` library is a **task-specific library** built on top of the core `ondevice_training_lib`. It provides a complete anomaly detection workflow for autoencoder-based models running on TI microcontrollers.

While the core library handles the raw neural network operations (forward, backward, weight update), this library manages everything else needed for a working anomaly detection system:

- A **state machine** with three operating modes (training, threshold calculation, inference)
- **Training orchestration** with epoch structure, train/validate phase switching, and batch management
- **EMA-smoothed early stopping** with configurable patience and improvement delta
- **Threshold calculation** using percentile-based or Gaussian-based methods
- **Majority voting inference** with sliding window to reduce false positives
- **Rolling statistics** for runtime monitoring

### Where it fits

    ┌─────────────────────────────────────────┐
    │         Application Code                │
    │    (application_main.c)                 │  ← Calls this library's API
    ├─────────────────────────────────────────┤
    │      Anomaly Detection ODT              │
    │    (anomaly_detection_odt)              │  ← THIS LIBRARY: state machine, training loop
    ├─────────────────────────────────────────┤
    │      Core Training Library              │
    │    (ondevice_training_lib)              │  ← Forward, backward, SGD, loss
    ├─────────────────────────────────────────┤
    │      Model Configuration                │
    │    (trainable_model_config.h/.c)        │  ← Layer definitions, weights, buffers
    └─────────────────────────────────────────┘

### Source files

| File | Location | Purpose |
|------|----------|---------|
| `anomaly_detection_odt.h` | `tinyml-sdk/c29/ai/common/ondevice_training/` | API declarations, all data structures, mode/phase enums |
| `anomaly_detection_odt.c` | `tinyml-sdk/c29/ai/common/ondevice_training/` | All function implementations |

---

## 2. State Machine

The library operates as a state machine with three modes. The application drives transitions by calling the appropriate API functions.

    ┌──────────────────────────┐    StartTraining()     ┌──────────────────────────┐
    │                          │ ──────────────────────►│                          │
    │      MODE_INFERENCE      │                        │      MODE_TRAINING       │
    │                          │                        │                          │
    │  Detect anomalies using  │                        │  ┌──────────────────┐    │
    │  trained model +         │                        │  │   PHASE_TRAIN    │    │
    │  majority voting         │                        │  │  forward+backward│    │
    │                          │                        │  │  weight updates  │    │
    └──────────────────────────┘                        │  └────────┬─────────┘    │
               ▲                                        │           │ epoch done   │
               │                                        │           ▼              │
               │                                        │  ┌──────────────────┐    │
               │                                        │  │  PHASE_VALIDATE  │    │
               │                                        │  │  forward only    │    │
               │                                        │  │  EMA + early stop│    │
               │                                        │  └──────────────────┘    │
               │                                        │                          │
               │                                        └────────────┬─────────────┘
               │                                                     │
               │                                                     │ StopTraining()
               │                                                     │ or early stopping
               │                                                     ▼
               │                                        ┌──────────────────────────┐
               │          Auto-transition after         │                          │
               │          threshold_samples collected   │   MODE_THRESHOLD_CALC    │
                ◄───────────────────────────────────────│                          │
                                                        │  Collect normal errors,  │
                                                        │  compute threshold       │
                                                        └──────────────────────────┘

**Transitions:**

| From | To | Trigger |
|------|----|---------|
| `MODE_INFERENCE` | `MODE_TRAINING` | `AnomalyDetection_StartTraining()` |
| `MODE_TRAINING` | `MODE_THRESHOLD_CALC` | `AnomalyDetection_StopTraining()` or early stopping |
| `MODE_THRESHOLD_CALC` | `MODE_INFERENCE` | Automatic, after `threshold_samples` are collected |

**Sub-states within MODE_TRAINING:**

Training alternates between two phases within each epoch:
- `PHASE_TRAIN` — forward + backward pass, weight updates
- `PHASE_VALIDATE` — forward pass only, loss evaluation, early stopping check

One epoch = `batches_per_epoch` training batches + 1 validation batch.

---

## 3. Data Structures

### 3.1 AnomalyDetectionConfig_t — User Configuration

This structure is provided by the application at initialization time. It contains all configurable hyperparameters.

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `learning_rate` | `float` | 0.0001 | SGD learning rate. Passed to `ODT_Init` |
| `patience` | `uint16_t` | 10 | Epochs without improvement before early stopping. Must be > 0 |
| `min_improvement_delta` | `float` | 0.5 | Minimum decrease in EMA validation loss to count as improvement. Prevents tiny fluctuations from resetting the patience counter |
| `ema_alpha` | `float` | 0.5 | EMA smoothing factor for validation loss. Range: (0, 1]. Higher = more reactive to recent values. Lower = more smoothing |
| `batches_per_epoch` | `uint16_t` | 10 | Number of training batches per epoch. Total training samples per epoch = `batches_per_epoch × TRAIN_BATCH_SIZE` |
| `initial_threshold` | `float` | 6.98 | Initial anomaly threshold from PC-side training (`RECONSTRUCTION_ERROR_THRESHOLD`). Used for inference before on-device threshold calculation |
| `threshold_percentile` | `float` | 0.99 | Percentile for threshold calculation (e.g., 0.99 = 99th percentile). Range: [0, 1] |
| `gaussian_std_multiplier` | `float` | 4.0 | Number of standard deviations above mean for Gaussian threshold. Higher = fewer false positives but may miss subtle anomalies |
| `use_percentile_threshold` | `uint16_t` | 1 | Threshold method selector: 1 = percentile, 0 = Gaussian |
| `threshold_samples` | `uint16_t` | 500 | Number of normal samples to collect for threshold calculation. Max value limited to 1000 for now |
| `inference_window_size` | `uint16_t` | 1 | Sliding window size for majority voting. 1 = no voting (each frame decides independently). Range: 1–10 |
| `stats_window_size` | `uint16_t` | 100 | Frames per statistics reporting window. After this many frames, stats are logged and reset |

**Validation at init:** All fields are validated. Invalid values cause `AnomalyDetection_Init` to return -1 with an error log.

### 3.2 AnomalyDetectionContext_t — Internal State

This is the main context structure that holds all internal state. The application creates one instance and passes it to every API call. Key groups of fields:

**Model context:**
- `model_ctx` — the `ModelContext_t` from `ondevice_training_lib`, containing layers, weights, buffers

**Mode and phase:**
- `current_mode` — `MODE_INFERENCE`, `MODE_TRAINING`, or `MODE_THRESHOLD_CALC`
- `current_phase` — `PHASE_TRAIN` or `PHASE_VALIDATE` (meaningful only in training mode)

**Training counters:**
- `train_sample_count` — samples accumulated in current training batch
- `val_sample_count` — samples accumulated in current validation batch
- `epoch_num` — total epochs completed
- `batch_num_curr_epoch` — training batches completed in current epoch

**Loss tracking:**
- `train_loss_sum` — accumulated training loss for current epoch
- `val_loss_sum` — accumulated validation loss for current batch
- `best_val_loss` — best EMA validation loss achieved (initialized to 1e10)
- `best_epoch_num` — epoch where best validation loss was achieved

**Early stopping:**
- `no_improvement_count` — consecutive epochs without improvement
- `ema_val_loss` — EMA-smoothed validation loss
- `ema_initialized` — flag indicating whether EMA has been initialized (first epoch sets it directly, subsequent epochs apply the EMA formula)

**Threshold calculation:**
- `threshold_frame_count` — frames collected so far
- `threshold_mean_sum` — running sum of errors for mean calculation
- `threshold_errors[1000]` — buffer storing reconstruction errors
- `threshold` — the active anomaly detection threshold

**Inference:**
- `inference_error_window[10]` — circular buffer of recent reconstruction errors
- `inference_window_index` — current write position
- `inference_window_filled` — flag indicating the window has wrapped at least once
- `stats_frame_count`, `stats_anomaly_count`, `stats_normal_count` — rolling statistics

### 3.3 Result and Metrics Structures

**`AnomalyDetectionResult_t`** — returned by `ProcessInferenceFrame`:

| Field | Type | Description |
|-------|------|-------------|
| `is_anomaly` | `uint16_t` | 0 = normal, 1 = anomaly (after majority voting) |
| `reconstruction_error` | `float` | Raw MSE reconstruction error for this frame |
| `threshold` | `float` | Current threshold used for the decision |



**`AnomalyDetectionTrainingResult_t`** — returned by `StopTraining`:

| Field | Type | Description |
|-------|------|-------------|
| `final_train_loss` | `float` | Average training loss from last epoch |
| `final_val_loss` | `float` | Average validation loss from last epoch |
| `best_val_loss` | `float` | Best EMA validation loss achieved during training |
| `epochs_trained` | `uint16_t` | Total epochs completed |
| `quit_mode` | enum | `EARLY_STOPPED` or `USER_STOPPED` |



**`AnomalyDetectionMetrics_t`** — returned by `GetMetrics`:

| Field | Type | Description |
|-------|------|-------------|
| `current_train_loss` | `float` | Average training loss so far in current epoch |
| `current_val_loss` | `float` | Average validation loss so far in current batch |
| `best_val_loss` | `float` | Best EMA validation loss |
| `epoch_num` | `uint16_t` | Current epoch number |
| `no_improvement_count` | `uint16_t` | Epochs without improvement |
| `total_frames_processed` | `uint16_t` | Frames in current statistics window |
| `anomaly_count` | `uint16_t` | Anomalies in current statistics window |
| `normal_count` | `uint16_t` | Normal frames in current statistics window |
| `anomaly_percentage` | `float` | Anomaly percentage in current window |
| `normal_percentage` | `float` | Normal percentage in current window |
| `threshold` | `float` | Current anomaly detection threshold |

---

## 4. API Reference

### 4.1 AnomalyDetection_Init

    int AnomalyDetection_Init(AnomalyDetectionContext_t* ctx, AnomalyDetectionConfig_t* config);

Initializes the anomaly detection system. Must be called once before any other function. Validates all configuration fields, sets up the underlying neural network model via `ODT_Init()`, and starts the system in `MODE_INFERENCE` with the initial threshold from `config->initial_threshold`.

The `config` struct is copied internally — the caller can reuse or free it after this call.

**Returns:** 0 on success, -1 on failure.

**Common errors** (returns -1):
- NULL `ctx` or `config`
- `learning_rate` <= 0, `patience` == 0, `batches_per_epoch` == 0
- `ema_alpha` not in (0, 1]
- `threshold_samples` not in 1–1000, `inference_window_size` not in 1–10
- `ODT_Init` failure (model config mismatch — see [ondevice_training_lib](ondevice_training_lib.md#3-initialization))

**Example:**

    AnomalyDetectionContext_t ad_ctx;
    AnomalyDetectionConfig_t config = {
        .learning_rate = 0.0001f,
        .patience = 10,
        .min_improvement_delta = 0.5f,
        .ema_alpha = 0.5f,
        .batches_per_epoch = 10,
        .initial_threshold = RECONSTRUCTION_ERROR_THRESHOLD,
        .threshold_percentile = 0.99f,
        .gaussian_std_multiplier = 4.0f,
        .use_percentile_threshold = 1,
        .threshold_samples = 500,
        .inference_window_size = 1,
        .stats_window_size = 100
    };
    
    int status = AnomalyDetection_Init(&ad_ctx, &config);
    if (status != 0) {
        // Handle error — check console for specific validation failure
    }

### 4.2 AnomalyDetection_StartTraining

    int AnomalyDetection_StartTraining(AnomalyDetectionContext_t* ctx);

Transitions from `MODE_INFERENCE` to `MODE_TRAINING`. Resets all training state (counters, losses, early stopping) and saves the current weights as the initial "best" checkpoint. After this call, the system expects training samples via `ProcessTrainingFrame`.

**Precondition:** Must be in `MODE_INFERENCE`. Returns -1 if called from any other mode.

**Returns:** 0 on success, -1 on error.

### 4.3 AnomalyDetection_StopTraining

    int AnomalyDetection_StopTraining(AnomalyDetectionContext_t* ctx, AnomalyDetectionTrainingResult_t* result);

Ends training, restores the best weights (from the epoch with lowest validation loss, not the last epoch), and transitions to `MODE_THRESHOLD_CALC`. The `result` struct is filled with a training summary including final losses, best loss, epochs trained, and whether training was early-stopped or user-stopped.

**Precondition:** Must be in `MODE_TRAINING`. Returns -1 if called from any other mode.

**Returns:** 0 on success, -1 on error.

This function is also called automatically by `ProcessTrainingFrame` when early stopping triggers. You can call it manually at any time to stop training early (e.g., after a fixed number of epochs).

### 4.4 AnomalyDetection_ProcessTrainingFrame

    void AnomalyDetection_ProcessTrainingFrame(AnomalyDetectionContext_t* ctx, float* frozen_output, float* target);

The main training function. Call it once per sample — the library handles everything internally: batching, epoch transitions, train/validate phase switching, EMA-smoothed loss tracking, weight checkpointing, and early stopping.

**Parameters:**

| Parameter | Size | Description |
|-----------|------|-------------|
| `frozen_output` | `FROZEN_OUTPUT_SIZE` | Output from the frozen model for this sample. This is the input to the trainable model. |
| `target` | `FINAL_OUTPUT_SIZE` | Ground truth that the model tries to reconstruct. For autoencoders, this is the original feature input — the model learns to reconstruct the input features from the frozen model's output.|

**Precondition:** Must be in `MODE_TRAINING`.

**How it works:**

The library internally alternates between two phases each epoch. You don't need to manage this — just keep calling `ProcessTrainingFrame` with your data:

- **Training phase**: Runs forward + backward pass, updates weights. After `batches_per_epoch × TRAIN_BATCH_SIZE` samples, automatically switches to validation.
- **Validation phase**: Runs forward pass only (no weight updates), computes EMA-smoothed loss, checks for improvement, and saves best weights if the model improved. After `VAL_BATCH_SIZE` samples, increments the epoch counter and switches back to training.

**Early stopping:** If validation loss hasn't improved for `patience` consecutive epochs, the library automatically calls `StopTraining`

**Epoch structure:**

    Epoch N:
    ├── Training:    batches_per_epoch × TRAIN_BATCH_SIZE samples  (weights updated)
    └── Validation:  VAL_BATCH_SIZE samples                        (loss evaluated)
                     → EMA smoothing → improvement check → save best weights
    
    Epoch N+1: (repeats until early stopping or manual stop)

**What to keep in mind:**
- The application is responsible for feeding data in the right order — training samples during the training phase, validation samples during the validation phase. Get the current phase to know whether to pass training or validation data.
- Pass `frozen_output` as the input and the original feature input as `target`. 
- Early stopping can trigger at any epoch. After it triggers, the mode changes to `MODE_THRESHOLD_CALC` — subsequent calls to `ProcessTrainingFrame` will be rejected.

### 4.5 AnomalyDetection_ProcessThresholdFrame

    void AnomalyDetection_ProcessThresholdFrame(AnomalyDetectionContext_t* ctx, float* frozen_output, float* target);

Collects reconstruction errors from normal samples to compute the anomaly detection threshold. Call it repeatedly with normal (non-anomalous) data. After `threshold_samples` have been collected, the next call triggers threshold calculation and automatically transitions to `MODE_INFERENCE`.

**Precondition:** Must be in `MODE_THRESHOLD_CALC`.

Only feed **normal** (non-anomalous) samples. The threshold characterizes the distribution of normal reconstruction errors — feeding anomalous data here will produce an incorrect threshold.

**How it works internally:**
- Each call runs a forward pass, computes the reconstruction error, and stores it
- After collecting `threshold_samples` errors, it computes the threshold using the configured method (percentile or Gaussian — see [Section 6](#6-threshold-calculation--detailed-explanation) for details)
- Both methods are always computed and logged for comparison, regardless of which one is selected
- The system then auto-transitions to `MODE_INFERENCE` — no additional call needed
- The threshold is stored in `AnomalyDetectionContext_t.threshold` and used automatically during inference

**Example console output:**

    Threshold Calculation Results:
      Mean error: 14
      Std error: 6
      Gaussian (mean + 4*std): 39
      95th percentile: 25
      99th percentile: 25
      Configured percentile (0): 25
      Method used: Percentile
      FINAL THRESHOLD: 25

### 4.6 AnomalyDetection_ProcessInferenceFrame

    void AnomalyDetection_ProcessInferenceFrame(AnomalyDetectionContext_t* ctx, float* frozen_output, float* target, AnomalyDetectionResult_t* result);

Performs real-time anomaly detection on a single frame. Computes the reconstruction error, applies majority voting over a sliding window, and returns the anomaly decision in `result`.

**Precondition:** Must be in `MODE_INFERENCE`.

**Output** (`AnomalyDetectionResult_t`):
- `is_anomaly` — 0 (normal) or 1 (anomaly), after majority voting
- `reconstruction_error` — raw MSE error for this frame
- `threshold` — the threshold used for the decision

**Majority voting:** When `inference_window_size = 1`, each frame is classified independently. When > 1 (e.g., 5), the library maintains a sliding window of recent errors. An anomaly is declared only if **more than half** the window exceeds the threshold. This reduces false positives from isolated spikes.

    Window = [12.3, 8.1, 45.2, 7.5, 9.0], threshold = 25.0
    Exceeds threshold: 1 out of 5 → Normal (not majority)
    
    Window = [45.2, 38.1, 42.0, 7.5, 31.0], threshold = 25.0
    Exceeds threshold: 4 out of 5 → Anomaly (majority)

**Rolling statistics:** The library tracks anomaly/normal counts over a configurable window (`stats_window_size`). Statistics are logged to console periodically and then reset. Use `GetMetrics` to read them programmatically.

### 4.7 AnomalyDetection_ForwardPass

    int AnomalyDetection_ForwardPass(AnomalyDetectionContext_t* ctx, float* frozen_output, float* target, float* output, float* error);

Utility function that runs a forward pass and computes the reconstruction error, without any side effects — no weight updates, no state changes, no threshold or voting logic. Useful for golden vector validation and manual testing.

**Returns:** 0 on success, -1 on error (NULL parameters).

**Can be called in any mode.**

**Example (golden vector validation):**

    float output[FINAL_OUTPUT_SIZE];
    float error;
    AnomalyDetection_ForwardPass(&ad_ctx, frozen_output, target, output, &error);
    
    // Compare output against golden_output element by element
    int matched = 0, not_matched = 0;
    for (int i = 0; i < FINAL_OUTPUT_SIZE; i++) {
        if (fabsf(output[i] - golden_output[i]) < 1e-3f) {
            matched++;
        } else {
            not_matched++;
        }
    }

### 4.8 AnomalyDetection_GetMode / AnomalyDetection_GetMetrics

    int AnomalyDetection_GetMode(AnomalyDetectionContext_t* ctx, AnomalyDetectionTrainingMode_t* mode);
    
    void AnomalyDetection_GetMetrics(AnomalyDetectionContext_t* ctx, AnomalyDetectionMetrics_t* metrics);

**GetMode:** Returns the current operating mode (`MODE_INFERENCE`, `MODE_TRAINING`, or `MODE_THRESHOLD_CALC`).

**GetMetrics:** Fills the metrics structure with current state. Training metrics (losses, epoch) are meaningful in `MODE_TRAINING`. Inference metrics (frame counts, anomaly percentage) are meaningful in `MODE_INFERENCE`.

---

## 5. Training Features — Detailed Explanation

### 5.1 EMA-Smoothed Validation Loss

With small validation datasets (e.g., 6 samples), the raw validation loss can fluctuate significantly from epoch to epoch. EMA smoothing reduces this noise.

**Formula:**

    First epoch:  ema = raw_val_loss                      (initialize)
    Subsequent:   ema = alpha × raw_val_loss + (1 - alpha) × previous_ema

**Effect of `ema_alpha`:**

| Alpha | Behavior |
|-------|----------|
| 1.0 | No smoothing — EMA equals the raw value |
| 0.5 | Equal weight to current and history — moderate smoothing |
| 0.2 | Strong smoothing — slow to react to changes |

### 5.2 Early Stopping

Early stopping prevents overfitting by monitoring the EMA validation loss and stopping training when it stops improving.

**How it works:**
- After each validation epoch, check: `ema_val_loss < best_val_loss - min_improvement_delta`
- If yes: reset `no_improvement_count` to 0, save best weights
- If no: increment `no_improvement_count`
- If `no_improvement_count >= patience`: trigger early stopping

**The `min_improvement_delta` parameter** prevents tiny improvements from resetting the patience counter. For example, with `delta = 0.5`, a loss decrease from 100.0 to 99.9 does NOT count as improvement, but 100.0 to 99.4 does.

### 5.3 Best Weight Checkpointing

Every time a new best validation loss is achieved, the current weights are saved via `ODT_SaveBestWeights`. When training ends (either by early stopping or manual stop), the best weights are restored via `ODT_LoadBestWeights`.

This means the final model uses the weights from the **best epoch**, not the last epoch. 

---

## 6. Threshold Calculation — Detailed Explanation

After training, the model's notion of "normal" has changed, so the anomaly detection threshold must be recalculated using the updated model.

### Process

1. Feed `threshold_samples` normal samples through the trained model
2. Collect each sample's reconstruction error in `threshold_errors[]`
3. After all samples collected, compute the threshold using one of two methods

### Percentile method

Sort all errors in ascending order. Pick the error at the configured percentile index.

    sorted_errors = [3.2, 5.1, 7.0, 8.4, 9.1, 12.3, ..., 25.1]
    99th percentile index = 0.99 × 500 = 495
    threshold = sorted_errors[495]

This method directly says: "only 1% of normal samples have error above this threshold."

### Gaussian method

Assume errors follow a normal distribution. Set threshold at `mean + k × std`.

    mean = sum(errors) / N
    std = sqrt(sum((error - mean)²) / N)
    threshold = mean + gaussian_std_multiplier × std

With `gaussian_std_multiplier = 4`, this covers 99.997% of a normal distribution.

### Which method to choose

| Method | Pros | Cons |
|--------|------|------|
| Percentile | No distribution assumption, robust to outliers | Needs enough samples for stable percentile estimate |
| Gaussian | Works with fewer samples, mathematically grounded | Assumes normal distribution, sensitive to outliers |

Set `use_percentile_threshold = 1` for percentile (default) or `0` for Gaussian. Both are computed and logged regardless of selection, so you can compare them.

---

## 7. Logging System

Same pattern as the core library. Uses `AD_LOG` macro:

- **DEBUG build**: compiles to `printf`
- **DEBUG + interrupt safe**: wraps in `DINT`/`EINT`
- **Release build**: compiles to nothing (zero overhead)

Key log messages during normal operation:

    Epoch 0: Train Loss = 5571
    Epoch 0: Val Loss = 6109 (raw), 6109 (smoothed)  Best model, weights saved!
    Epoch 1: Train Loss = 5547
    Epoch 1: Val Loss = 5818 (raw), 5963 (smoothed)  Best model, weights saved!
    ...
    Epoch 121: Val Loss = 13 (raw), 16 (smoothed)  No improvement (10/10)
    EARLY STOPPING TRIGGERED

---

## 8. API Quick Reference

| Function | Mode Required | Purpose |
|----------|--------------|---------|
| `AnomalyDetection_Init(ctx, config)` | Any | Initialize system, validate config, reset state |
| `AnomalyDetection_StartTraining(ctx)` | INFERENCE | Transition to TRAINING mode |
| `AnomalyDetection_StopTraining(ctx, result)` | TRAINING | End training, load best weights, transition to THRESHOLD_CALC |
| `AnomalyDetection_ProcessTrainingFrame(ctx, input, target)` | TRAINING | Process one training/validation sample |
| `AnomalyDetection_ProcessThresholdFrame(ctx, input, target)` | THRESHOLD_CALC | Collect error for threshold calculation |
| `AnomalyDetection_ProcessInferenceFrame(ctx, input, target, result)` | INFERENCE | Detect anomaly with majority voting |
| `AnomalyDetection_ForwardPass(ctx, input, target, output, error)` | Any | Forward pass + MSE error (no side effects) |
| `AnomalyDetection_GetMode(ctx, mode)` | Any | Query current mode |
| `AnomalyDetection_GetMetrics(ctx, metrics)` | Any | Get training/inference statistics |

---

## 9. Usage Examples

This section walks through how to use the library with simple, self-contained examples. Each example builds on the previous one. All examples assume:
- The model artifacts (`trainable_model_config.h/.c`, `mod.a`) are already generated
- Training data (`ondevice_training_data.h/.c`) is available
- A `prepare_frozen_output()` function exists that takes a raw sample, runs feature extraction + frozen model, and fills `frozen_output[]` and `model_input[]` (the feature-space input before the frozen model)

For a complete working implementation, see the [fan blade example application](application_example.md).

### 9.1 Minimal API Patterns

These snippets show the bare-minimum API calls for each operation — no error handling, no prints, just the essential calls.

**Initialize:**

    AnomalyDetectionContext_t ctx;
    AnomalyDetectionConfig_t config = { /* fill fields */ };
    AnomalyDetection_Init(&ctx, &config);

**Run inference on one sample:**

    AnomalyDetectionResult_t result;
    AnomalyDetection_ProcessInferenceFrame(&ctx, frozen_output, model_input, &result);
    // result.is_anomaly → 0 or 1

**Train (full loop until early stopping):**

    AnomalyDetection_StartTraining(&ctx);

    AnomalyDetectionTrainingMode_t mode;
    AnomalyDetection_GetMode(&ctx, &mode);
    while (mode == MODE_TRAINING) {
        if (ctx.current_phase == PHASE_TRAIN)
            prepare_frozen_output(train_data[i++], frozen_output, model_input);
        else
            prepare_frozen_output(val_data[j++], frozen_output, model_input);

        AnomalyDetection_ProcessTrainingFrame(&ctx, frozen_output, model_input);
        AnomalyDetection_GetMode(&ctx, &mode);
    }

**Calculate threshold (after training):**

    AnomalyDetection_GetMode(&ctx, &mode);
    while (mode == MODE_THRESHOLD_CALC) {
        prepare_frozen_output(normal_data[k++], frozen_output, model_input);
        AnomalyDetection_ProcessThresholdFrame(&ctx, frozen_output, model_input);
        AnomalyDetection_GetMode(&ctx, &mode);
    }
    // Now in MODE_INFERENCE with new threshold

**Forward pass only (no side effects):**

    float output[FINAL_OUTPUT_SIZE];
    float error;
    AnomalyDetection_ForwardPass(&ctx, frozen_output, model_input, output, &error);

The following subsections show these patterns in full context with complete working code.

### 9.2 Initialization and Pre-Training Inference

Before training, you can run inference using the initial weights and the PC-exported threshold to establish a baseline accuracy.

    #include "anomaly_detection_odt.h"
    #include "ondevice_training_data.h"

    AnomalyDetectionContext_t ad_ctx;
    float frozen_output[FROZEN_OUTPUT_SIZE];
    float model_input[FE_STACKING_CHANNELS * FE_STACKING_FRAME_WIDTH];

    void run_baseline(void) {

        // Step 1: Configure and initialize
        AnomalyDetectionConfig_t config = {
            .learning_rate          = 0.0001f,
            .patience               = 10,
            .min_improvement_delta  = 0.5f,
            .ema_alpha              = 0.5f,
            .batches_per_epoch      = 10,
            .initial_threshold      = RECONSTRUCTION_ERROR_THRESHOLD,
            .threshold_percentile   = 0.99f,
            .gaussian_std_multiplier = 4.0f,
            .use_percentile_threshold = 1,
            .threshold_samples      = 500,
            .inference_window_size  = 1,
            .stats_window_size      = 100
        };

        if (AnomalyDetection_Init(&ad_ctx, &config) != 0) {
            printf("Init failed!\n");
            return;
        }

        // Step 2: Run inference on test data with initial (PC-trained) weights
        uint16_t correct = 0;
        for (uint16_t i = 0; i < NUM_TEST_SAMPLES; i++) {
            prepare_frozen_output(TEST_INPUTS[i], frozen_output, model_input);

            AnomalyDetectionResult_t result;
            AnomalyDetection_ProcessInferenceFrame(&ad_ctx, frozen_output, model_input, &result);

            // label=0 means anomaly, label=1 means normal
            uint16_t expected_anomaly = (TEST_LABELS[i] == 0) ? 1 : 0;
            if (result.is_anomaly == expected_anomaly) correct++;

            printf("Sample %d: error=%.2f threshold=%.2f -> %s\n", i, result.reconstruction_error, result.threshold, result.is_anomaly ? "ANOMALY" : "NORMAL");
        }
        printf("Baseline accuracy: %d/%d\n", correct, NUM_TEST_SAMPLES);
    }

### 9.3 The Complete Training Loop

Training is driven by calling `ProcessTrainingFrame` in a loop. The library handles epoch management, phase switching, and early stopping internally. You just need to feed the right data based on the current phase.

    void run_training(void) {

        // Start training (must currently be in MODE_INFERENCE)
        AnomalyDetection_StartTraining(&ad_ctx);

        // Simple indices to cycle through data
        uint16_t train_idx = 0;
        uint16_t val_idx = 0;

        // Loop until training ends (early stopping or manual stop)
        AnomalyDetectionTrainingMode_t mode;
        AnomalyDetection_GetMode(&ad_ctx, &mode);

        while (mode == MODE_TRAINING) {

            // Feed the right data based on current phase
            if (ad_ctx.current_phase == PHASE_TRAIN) {
                prepare_frozen_output(TRAIN_INPUTS[train_idx], frozen_output, model_input);
                train_idx = (train_idx + 1) % NUM_TRAIN_SAMPLES;
            } else {
                prepare_frozen_output(VALIDATION_INPUTS[val_idx], frozen_output, model_input);
                val_idx = (val_idx + 1) % NUM_VALIDATION_SAMPLES;
            }

            // One call per sample — the library handles: batching, forward/backward, loss tracking,  smoothing, checkpointing, early stopping
            AnomalyDetection_ProcessTrainingFrame( &ad_ctx, frozen_output, model_input);

            // Check if training ended (early stop -> MODE_THRESHOLD_CALC)
            AnomalyDetection_GetMode(&ad_ctx, &mode);
        }

        printf("Training complete!\n");
    }

**Key points:**
- Check `ad_ctx.current_phase` to know whether to feed training or validation data
- You don't manage epochs, batches, or early stopping — the library does all of that
- The loop exits when the mode changes from `MODE_TRAINING` 

### 9.4 Threshold Calculation

After training, the model's reconstruction errors have changed, so the threshold must be recalculated. Feed normal samples until the library has enough and auto-transitions to inference.

    void run_threshold_calculation(void) {

        uint16_t val_idx = 0;

        AnomalyDetectionTrainingMode_t mode;
        AnomalyDetection_GetMode(&ad_ctx, &mode);

        while (mode == MODE_THRESHOLD_CALC) {
            // Feed ONLY normal (non-anomalous) samples
            prepare_frozen_output(VALIDATION_INPUTS[val_idx], frozen_output, model_input);
            val_idx = (val_idx + 1) % NUM_VALIDATION_SAMPLES;

            AnomalyDetection_ProcessThresholdFrame(&ad_ctx, frozen_output, model_input);

            AnomalyDetection_GetMode(&ad_ctx, &mode);
        }

        // Mode is now MODE_INFERENCE with the new threshold
        printf("Threshold calculated! Ready for inference.\n");
    }

**Key points:**
- Only feed **normal** data — anomalous samples will corrupt the threshold
- The library auto-transitions to `MODE_INFERENCE` after computing the threshold
- Both percentile and Gaussian thresholds are logged for comparison

### 9.5 Post-Training Inference

After training and threshold calculation, inference works identically to pre-training — but now with updated weights and a recalculated threshold.

    void run_post_training_inference(void) {

        uint16_t correct = 0;
        for (uint16_t i = 0; i < NUM_TEST_SAMPLES; i++) {
            prepare_frozen_output(TEST_INPUTS[i], frozen_output, model_input);

            AnomalyDetectionResult_t result;
            AnomalyDetection_ProcessInferenceFrame(&ad_ctx, frozen_output, model_input, &result);

            uint16_t expected_anomaly = (TEST_LABELS[i] == 0) ? 1 : 0;
            if (result.is_anomaly == expected_anomaly) correct++;

            printf("Sample %d: error=%.2f threshold=%.2f -> %s\n", i, result.reconstruction_error, result.threshold, result.is_anomaly ? "ANOMALY" : "NORMAL");
        }
        printf("Post-training accuracy: %d/%d\n", correct, NUM_TEST_SAMPLES);
    }


### 9.6 Putting It All Together

The complete workflow in `main()`:

    int main(void) {

        init_board();  // Hardware, peripherals, feature extraction

        // 1. Initialize and check baseline accuracy
        run_baseline();

        // 2. Train the model on-device (runs until early stopping)
        run_training();

        // 3. Recalculate the anomaly threshold with updated model
        run_threshold_calculation();

        // 4. Evaluate with the trained model + new threshold
        run_post_training_inference();
    }

This is the same flow used in the SDK fan blade example — see [Application Example](application_example.md) for the full implementation with feature extraction, golden vector validation, and debugger-triggered training.

---

## 10. Further Reading

| Document | Description |
|----------|-------------|
| [On-Device Training Library](ondevice_training_lib.md) | Core library that this task library is built on |
| [Trainable Model Configuration](trainable_model_config.md) | Model architecture and weight files used by both libraries |
| [Application Example — Fan Blade](application_example.md) | Complete working example showing this library in action |
| [Overview](overview.md) | High-level architecture and concepts |