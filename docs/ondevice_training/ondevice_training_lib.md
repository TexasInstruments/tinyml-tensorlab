# On-Device Training Library 

## Table of Contents

- [1. Overview](#1-overview)
- [2. Data Structures](#2-data-structures)
  - [2.1 ModelContext_t](#21-modelcontext_t--the-central-training-state)
  - [2.2 TrainingPhase_t](#22-trainingphase_t)
- [3. Initialization — ODT_Init](#3-initialization)
- [4. Core API — Forward Pass](#4-core-api--forward-pass)
- [5. Core API — Backward Pass](#5-core-api--backward-pass)
- [6. Core API — Weight Update and Gradient Management](#6-core-api--weight-update-and-gradient-management)
- [7. Core API — Weight Checkpointing](#7-core-api--weight-checkpointing)
- [8. Loss Functions](#8-loss-functions)
- [9. Layer Operations](#9-layer-operations)
  - [9.1 Linear Layer](#91-linear-layer)
  - [9.2 ReLU Layer](#92-relu-layer)
- [10. Batch Size Optimization](#10-batch-size-optimization--detailed-explanation)
- [11. Logging System](#11-logging-system)
- [12. Writing a Custom Training Loop](#12-writing-a-custom-training-loop)
- [13. How to Add a New Layer Type](#13-how-to-add-a-new-layer-type)
- [14. API Quick Reference](#14-api-quick-reference)

---

## 1. Overview

The `ondevice_training_lib` is the **core neural network training engine** for on-device training on TI microcontrollers. It provides the fundamental building blocks — forward pass, backward pass, weight updates, loss computation — that any task-specific library can use to implement on-device training.

**Key design principles:**

- **Task-agnostic**: This library knows nothing about classification, anomaly detection etc. It only knows about layers, weights, gradients, and loss.
- **Zero dynamic allocation**: All memory is statically allocated via `trainable_model_config.c`. The library only uses pointers into these pre-allocated arrays.
- **Flat array architecture**: All weights, buffers, and gradients live in single contiguous arrays accessed via offsets.
- **Compile-time optimization**: Batch size = 1 eliminates gradient accumulation code entirely via `#if USE_GRADIENT_ACCUMULATION`.

### Where it fits

    ┌─────────────────────────────────────────┐
    │         Application Code                │
    │    (application_main.c)                 │
    ├─────────────────────────────────────────┤
    │      Task Library                       │
    │    (anomaly_detection_odt)              │  ← Orchestrates training workflow
    ├─────────────────────────────────────────┤
    │      Core Training Library              │
    │    (ondevice_training_lib)              │  ← THIS LIBRARY: forward, backward, SGD
    ├─────────────────────────────────────────┤
    │      Model Configuration                │
    │    (trainable_model_config.h/.c)        │  ← Layer definitions, weights, buffers
    └─────────────────────────────────────────┘

### Source files

| File | Location | Purpose |
|------|----------|---------|
| `ondevice_training_lib.h` | `tinyml-sdk/c29/ai/common/ondevice_training/` | API declarations, ModelContext_t structure, logging macros |
| `ondevice_training_lib.c` | `tinyml-sdk/c29/ai/common/ondevice_training/` | All function implementations |

---

## 2. Data Structures

### 2.1 ModelContext_t — The Central Training State

Every function in this library takes a `ModelContext_t*` as its first argument. This structure holds the complete state of the trainable model.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `layers[]` | `LayerParams_t[NUM_TRAINABLE_LAYERS]` | Layer descriptors copied from `LAYER_PARAMS_INIT` during init |
| `num_layers` | `uint16_t` | Number of layers in trainable model |
| `intermediate_buffers[]` | `float*[NUM_TRAINABLE_LAYERS + 1]` | Pointers into `INTERMEDIATE_BUFFERS`. Index 0 = frozen model output (input to first layer). Index i = output of layer i-1 |
| `gradient_buffers[]` | `float*[NUM_TRAINABLE_LAYERS + 1]` | Pointers into `GRADIENT_BUFFERS`. Index i+1 = gradient flowing backward into layer i (input to layer i's backward pass). Index i = gradient flowing backward out of layer i (output of layer i's backward pass). Last index is gradient of loss function w.r.t outputs. Index 0 is  gradient w.r.t. inputs of trainable model|
| `current_weights` | `float*` | Points to `ALL_WEIGHTS` — the active model parameters being updated during training |
| `best_weights` | `float*` | Points to `ALL_BEST_WEIGHTS` — checkpoint storage for the best-performing weights |
| `weight_gradients` | `float*` | Points to `ALL_WEIGHT_GRADS` — accumulated gradients for batch training. Only exists when `USE_GRADIENT_ACCUMULATION` is 1 |
| `batch_sample_count` | `uint16_t` | Number of samples processed in the current batch |
| `is_training_mode` | `TrainingPhase_t` | Current phase: `PHASE_TRAINING` or `PHASE_INFERENCE` |
| `learning_rate` | `float` | SGD learning rate for weight updates |

**How memory is organized:**

All three flat arrays follow the same indexing pattern, illustrated here with a simple 3-layer example: **Linear(4→3), ReLU(3), Linear(3→2)** and a loss function(FROZEN_OUTPUT_SIZE=4, FINAL_OUTPUT_SIZE=2, NUM_TRAINABLE_LAYERS=3).

**ALL_WEIGHTS** — all trainable parameters packed sequentially:

    ALL_WEIGHTS[TOTAL_PARAMS]   (TOTAL_PARAMS = 4×3 + 3 + 3×2 + 2 = 23)
    ┌──────────────────┬─────────────┬──────────────────┬─────────────┐
    │ Layer 0 weights  │ Layer 0     │ Layer 2 weights  │ Layer 2     │
    │ W[3×4] = 12      │ biases = 3  │ W[2×3] = 6       │ biases = 2  │
    └──────────────────┴─────────────┴──────────────────┴─────────────┘
     offset=0           offset=12    offset=15          offset=21

    (Layer 1 is ReLU — no weights, so it occupies no space in ALL_WEIGHTS)

**INTERMEDIATE_BUFFERS** — forward pass activations, one slot per layer boundary:

    INTERMEDIATE_BUFFERS[TOTAL_INTERMEDIATE_BUFFER_SIZE]   (= 4 + 3 + 3 + 2 = 12)
    ┌─────────────┬────────────────┬────────────────┬────────────────┐
    │  buffer[0]  │   buffer[1]    │   buffer[2]    │   buffer[3]    │
    │  4 floats   │   3 floats     │   3 floats     │   2 floats     │
    │  (input to  │   (output of   │   (output of   │   (output of   │
    │   layer 0)  │    layer 0)    │    layer 1)    │    layer 2)    │
    └─────────────┴────────────────┴────────────────┴────────────────┘
     offset=0      offset=4        offset=7         offset=10

**GRADIENT_BUFFERS** — backward pass gradients, same layout as intermediate:

    GRADIENT_BUFFERS[TOTAL_GRADIENT_BUFFER_SIZE]   (same size = 12)
    ┌─────────────┬────────────────┬────────────────┬────────────────┐
    │  grad[0]    │   grad[1]      │   grad[2]      │   grad[3]      │
    │  4 floats   │   3 floats     │   3 floats     │   2 floats     │
    │  grad w.r.t │   grad flowing │   grad flowing │   loss gradient│
    │  input      │   into         │   into         │   (dL/d_output)│
    │             │   layer 0      │   layer 1      │   placed here  │
    └─────────────┴────────────────┴────────────────┴────────────────┘
     offset=0      offset=4        offset=7         offset=10

    Backward flow:  grad[3] => Layer 2 => grad[2] => Layer 1 => grad[1] => Layer 0 => grad[0]
                                                                             

Each layer reads from `intermediate_buffers[i]` and writes to `intermediate_buffers[i+1]` during the forward pass. During the backward pass, layer i reads the incoming gradient from `gradient_buffers[i+1]` (gradient flowing backward from the layer above) and writes the outgoing gradient to `gradient_buffers[i]` (gradient flowing backward to the layer below). The loss gradient is placed in `gradient_buffers[num_layers]` to start the backward pass. `gradient_buffers[0]` receives the gradient w.r.t. the trainable model input, which is computed but not used.

### 2.2 TrainingPhase_t

    typedef enum {
        PHASE_TRAINING,   // Model is in training mode
        PHASE_INFERENCE   // Model is in inference mode
    } TrainingPhase_t;

This enum is stored in `ModelContext_t.is_training_mode`

---

## 3. Initialization

### ODT_Init

**Signature:**

    int ODT_Init(ModelContext_t* ctx, float learning_rate);

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `ModelContext_t*` | Model context to initialize |
| `learning_rate` | `float` | Learning rate for SGD (e.g., 0.0001). Must be > 0 |

**Returns:** 0 on success, -1 on error.

Prepares the model context for use. This is a one-time setup that must be called before any other function in the library. It reads the auto-generated model configuration (`LAYER_PARAMS_INIT`, `ALL_WEIGHTS`, `BUFFER_OFFSETS`, etc.), sets up all internal pointers, and validates that the model architecture is consistent (layer dimensions chain correctly, offsets are within bounds, first layer matches `FROZEN_OUTPUT_SIZE`, last layer matches `FINAL_OUTPUT_SIZE`).

After this call, the model context is ready for forward passes, backward passes, and weight updates. The model starts in `PHASE_INFERENCE` mode.

**Example:**

    ModelContext_t model_ctx;
    int status = ODT_Init(&model_ctx, 0.0001f);
    if (status != 0) {
        // Handle initialization error
    }

**Console output (DEBUG build):**

    ========================================
    ODT INITIALIZATION
    ========================================
      Layer 0: Linear(48→24)
      Layer 1: ReLU(24)
      Layer 2: Linear(24→12)
      Layer 3: ReLU(12)
      Layer 4: Linear(12→6)
      Layer 5: ReLU(6)
      Layer 6: Linear(6→12)
      Layer 7: ReLU(12)
      Layer 8: Linear(12→24)
      Layer 9: ReLU(24)
      Layer 10: Linear(24→48)

**Common errors** (returns -1 ):
- NULL context or non-positive learning rate
- Layer dimension mismatch (e.g., layer 2 expects 12 inputs but layer 1 outputs 24)
- Weight offsets out of bounds (usually means `trainable_model_config` files are corrupted or mismatched)
- First/last layer size doesn't match `FROZEN_OUTPUT_SIZE` / `FINAL_OUTPUT_SIZE`

---

## 4. Core API — Forward Pass

### ODT_Forward

**Signature:**

    int ODT_Forward(ModelContext_t* ctx, float* input, float* output);

**Parameters:**

| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `ctx` | `ModelContext_t*` | — | Initialized model context |
| `input` | `float*` | `FROZEN_OUTPUT_SIZE` | Input from frozen model (or raw input for full model training) |
| `output` | `float*` | `FINAL_OUTPUT_SIZE` | Buffer to receive model output |

**Returns:** 0 on success, -1 on error.


Runs the input through all trainable layers in sequence and writes the final result to `output`. Intermediate activations are saved in internal buffers — these are needed if you call `ODT_Backward` afterward.

**Data flow diagram** (using the same 3-layer example: Linear(4→3), ReLU(3), Linear(3→2)):

    input (4 floats) ──copy──► intermediate_buffers[0] (4 floats)
                                       │
                                       ▼
                                Layer 0: Linear(4→3)
                                       │
                                       ▼
                               intermediate_buffers[1] (3 floats)
                                       │
                                       ▼
                                Layer 1: ReLU(3)
                                       │
                               intermediate_buffers[2] (3 floats)
                                       │
                                       ▼
                                Layer 2: Linear(3→2)
                                       │
                                       ▼
                               intermediate_buffers[3] (2 floats) ──copy──► output (2 floats)

Each `intermediate_buffers[i]` is a pointer to a region of the flat `INTERMEDIATE_BUFFERS` array. The number in parentheses is the number of floats at that position. Layer i reads from `intermediate_buffers[i]` and writes its result to `intermediate_buffers[i+1]`.

**Important:** The intermediate buffers are preserved after the forward pass. They are needed by the backward pass to compute gradients. Do not modify them between a forward and backward call during training.

**Example:**

    float model_input[input_size];  
    float model_output[output_size];

    ODT_Forward(&model_ctx, model_input, model_output);
    // model_output now contains the trainable model's output

---

## 5. Core API — Backward Pass

### ODT_Backward

**Signature:**

    int ODT_Backward(ModelContext_t* ctx, float* loss_gradient);

**Parameters:**

| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `ctx` | `ModelContext_t*` | — | Model context (must have valid intermediate buffers from a prior forward pass) |
| `loss_gradient` | `float*` | `FINAL_OUTPUT_SIZE` | Gradient of the loss with respect to the model output. Computed by `ODT_MSEGradient` or similar |

**Returns:** 0 on success, -1 on error.


Propagates gradients backward through all layers in reverse order, using the saved activations from the most recent forward pass. For each layer, it computes the gradient flowing to the previous layer.

For Linear layers, the backward pass also handles weight updates:
- **Batch size = 1**: Weights are updated immediately during the backward pass (no separate update step needed)
- **Batch size > 1**: Gradients are accumulated into a buffer; call `ODT_UpdateWeights` after the batch is complete

**Data flow diagram** (using the same 3-layer example, in reverse):

    loss_gradient (2 floats) ──copy──► gradient_buffers[3] (2 floats)
                                       │
                                       ▼
                                Layer 2: Linear(3→2) backward
                                       │
                                       ▼
                               gradient_buffers[2] (3 floats)
                                       │
                                       ▼
                                Layer 1: ReLU(3) backward
                                       │
                                       ▼
                               gradient_buffers[1] (3 floats)
                                       │
                                       ▼
                                Layer 0: Linear(4→3) backward
                                       │
                                       ▼
                               gradient_buffers[0] (4 floats)  

Each layer's backward pass does two things:
1. **Reads** the incoming gradient from `gradient_buffers[i+1]` and the saved activation from `intermediate_buffers[i]`
2. **Writes** the outgoing gradient to `gradient_buffers[i]`

For model layers, the backward pass additionally computes weight gradients and either updates weights immediately (batch_size=1) or accumulates gradients for later update (batch_size>1). 

**Important:** `ODT_Backward` must be called after `ODT_Forward` on the same sample. The intermediate buffers from the forward pass contain saved activations needed for gradient computation.

**Example (complete training step for batch_size=1):**

    // 1. Forward pass
    ODT_Forward(&model_ctx, model_input, model_output);

    // 2. Compute loss
    float loss;
    ODT_MSELoss(model_output, target, output_size, &loss);

    // 3. Compute loss gradient
    float loss_grad[output_size];
    ODT_MSEGradient(model_output, target, loss_grad, output_size);

    // 4. Backward pass (weights updated immediately when batch_size=1)
    ODT_Backward(&model_ctx, loss_grad);

    // Done! For batch_size=1, gradients are calculated and applied immediately.
    // For batch_size>1, gradients are accumulated in the gradient buffer for later update.

---

## 6. Core API — Weight Update and Gradient Management

### ODT_UpdateWeights

**Signature:**

    int ODT_UpdateWeights(ModelContext_t* ctx, uint16_t batch_size);

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `ModelContext_t*` | Model context |
| `batch_size` | `uint16_t` | Number of samples accumulated in the batch (for averaging) |

**Returns:** 0 on success, -1 on error.

**Behavior:**

- **When `USE_GRADIENT_ACCUMULATION` is 0 (batch_size=1):** This function is a no-op. Weights were already updated immediately during `ODT_Backward`.
- **When `USE_GRADIENT_ACCUMULATION` is 1 (batch_size>1):** Applies accumulated gradients to all weights:

      for each parameter i:
          current_weights[i] -= (learning_rate / batch_size) * weight_gradients[i]

  The division by `batch_size` averages the gradients across all samples in the batch.

### ODT_ZeroGradients

**Signature:**

    int ODT_ZeroGradients(ModelContext_t* ctx);

**Behavior:**

- **When `USE_GRADIENT_ACCUMULATION` is 0 (batch_size=1):** No-op. No gradient accumulators exist.
- **When `USE_GRADIENT_ACCUMULATION` is 1 (batch_size>1):** Zeros the entire `weight_gradients` array via `memset`. Must be called at the start of each new batch.

**Example (complete training step for batch_size>1):**

    // Start of batch
    ODT_ZeroGradients(&model_ctx);

    // Accumulate gradients over batch
    for (int s = 0; s < batch_size; s++) {
        ODT_Forward(&model_ctx, inputs[s], output);
        ODT_MSELoss(output, targets[s], output_size, &loss);
        ODT_MSEGradient(output, targets[s], loss_grad, output_size);
        ODT_Backward(&model_ctx, loss_grad);
        // Gradients accumulated but NOT applied yet
    }

    // Apply averaged gradients
    ODT_UpdateWeights(&model_ctx, batch_size);

---

## 7. Core API — Weight Checkpointing

### ODT_SaveBestWeights

**Signature:**

    int ODT_SaveBestWeights(ModelContext_t* ctx);

Saves a snapshot of all current model weights as the "best" checkpoint.

**When to call:** After a validation epoch shows a new best loss. Call it to store the current model and load back the weights when training is over. 

### ODT_LoadBestWeights

**Signature:**

    int ODT_LoadBestWeights(ModelContext_t* ctx);

Restores the previously saved "best" weights as the active model weights.

**When to call:** When training ends. This ensures the final model uses the best-performing checkpoint rather than the last epoch's weights, which may have overfit.

**Example:**

    // During training, after each validation epoch:
    if (val_loss < best_val_loss) {
        best_val_loss = val_loss;
        ODT_SaveBestWeights(&model_ctx);
    }

    // When training ends:
    ODT_LoadBestWeights(&model_ctx);
    // model now uses the best weights, not the last epoch's weights

---

## 8. Loss Functions

### ODT_MSELoss

**Signature:**

    int ODT_MSELoss(float* prediction, float* target, uint16_t size, float* loss);

Calculates the mean squared error from prediction and target and return the average loss. 

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `prediction` | `float*` | Model output |
| `target` | `float*` | Ground truth / expected output |
| `size` | `uint16_t` | Number of elements |
| `loss` | `float*` | Pointer to store the computed loss value |

**Formula:**

    MSE = (1/N) × Σ(prediction[i] - target[i])²

### ODT_MSEGradient

**Signature:**

    int ODT_MSEGradient(float* prediction, float* target, float* grad_output, uint16_t size);

Computes the gradient of the MSE loss with respect to each element of the prediction. This gradient vector is passed to `ODT_Backward` to initiate backpropagation through the network.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `prediction` | `float*` | Model output |
| `target` | `float*` | Ground truth |
| `grad_output` | `float*` | Output buffer for the gradient |
| `size` | `uint16_t` | Number of elements |

**Formula:**

    grad_output[i] = (2/N) × (prediction[i] - target[i])


**Example (computing loss and gradient together):**

    float loss;
    float loss_grad[output_size];

    ODT_MSELoss(model_output, target, output_size, &loss);
    ODT_MSEGradient(model_output, target, loss_grad, output_size);

    printf("MSE Loss: %f\n", loss);
    // loss_grad is ready to pass to ODT_Backward

---

## 9. Layer Operations

These are internal functions called by `ODT_Forward` and `ODT_Backward` — users do not call them directly. They are documented here for understanding and for extending the library with new layer types.
### 9.1 Linear Layer

A linear (fully connected) layer performs the standard matrix-vector operation: `output = W × input + bias`.

#### ODT_LinearForward

    void ODT_LinearForward(float* input, float* output, LayerParams_t* layer);

Computes the weighted sum of inputs plus bias for each output neuron. The weight matrix is stored in row-major format `[output_size × input_size]` in the flat `ALL_WEIGHTS` array, starting at `layer->weight_offset`.

#### ODT_LinearBackward

    void ODT_LinearBackward(ModelContext_t* ctx, float* grad_output, float* grad_input, float* input, LayerParams_t* layer);

Performs three operations during the backward pass:

1. **Input gradient**: Computes `grad_input = W^T × grad_output` — the gradient flowing to the previous layer. This is always computed regardless of batch size.

2. **Weight/bias update** (behavior depends on batch size):
   - **Batch size = 1** (`USE_GRADIENT_ACCUMULATION = 0`): Computes weight gradients and applies them immediately using SGD (`W -= lr × gradient`). No gradient storage buffer needed.
   - **Batch size > 1** (`USE_GRADIENT_ACCUMULATION = 1`): Accumulates weight gradients into a separate buffer. The actual weight update happens later when `ODT_UpdateWeights` is called.

The two paths are selected at compile time via `#if USE_GRADIENT_ACCUMULATION`, so the unused path is completely eliminated from the binary. The batch_size=1 path saves both memory (no `ALL_WEIGHT_GRADS` array) and compute (no separate accumulation or update pass).

### 9.2 ReLU Layer

ReLU (Rectified Linear Unit) applies an element-wise non-linearity. It has no trainable parameters.

#### ODT_ReLUForward

    void ODT_ReLUForward(float* input, float* output, uint16_t size);

Applies `output[i] = max(0, input[i])` for each element. Negative values are zeroed out, positive values pass through unchanged.

#### ODT_ReLUBackward

    void ODT_ReLUBackward(float* grad_output, float* grad_input, float* forward_output, uint16_t size);

Masks the gradient using the saved forward output: where the forward output was positive, the gradient passes through unchanged; where it was zero (input was negative), the gradient is blocked. No weight updates occur since ReLU has no parameters.

---

## 10. Batch Size Optimization — Detailed Explanation

The batch size optimization is one of the memory optimization and here is how it works:

### Batch size = 1

**Compile-time configuration:**

    #define TRAIN_BATCH_SIZE 1
    // Results in: USE_GRADIENT_ACCUMULATION = 0

**What happens:**
- `ALL_WEIGHT_GRADS` is **not allocated** (compile-time elimination)
- `weight_grad` and `bias_grad` fields **do not exist** in `LayerParams_t`
- `ODT_LinearBackward` computes gradient AND updates weight in a single fused loop
- `ODT_UpdateWeights` is a no-op
- `ODT_ZeroGradients` is a no-op

**Training loop (simplified):**

    for each sample:
        ODT_Forward(ctx, input, output)
        ODT_MSELoss(output, target, size, &loss)
        ODT_MSEGradient(output, target, loss_grad, size)
        ODT_Backward(ctx, loss_grad)          // ← weights updated HERE
        // No ODT_UpdateWeights needed
        // No ODT_ZeroGradients needed

**Memory savings:** For a model with P total parameters, this saves P × 4 bytes of RAM. For example, a model with 3,150 parameters saves 3,150 × 4 = 12,600 bytes.

### Batch size > 1

**Compile-time configuration:**

    #define TRAIN_BATCH_SIZE 4
    // Results in: USE_GRADIENT_ACCUMULATION = 1

**What happens:**
- `ALL_WEIGHT_GRADS[TOTAL_PARAMS]` is allocated
- `ODT_LinearBackward` accumulates gradients without updating weights
- `ODT_UpdateWeights` applies the averaged accumulated gradients
- `ODT_ZeroGradients` must be called before each batch

**Training loop (simplified):**

    for each batch:
        ODT_ZeroGradients(ctx)                // ← Reset accumulators
        for each sample in batch:
            ODT_Forward(ctx, input, output)
            ODT_MSELoss(output, target, size, &loss)
            ODT_MSEGradient(output, target, loss_grad, size)
            ODT_Backward(ctx, loss_grad)      // ← gradients accumulated, NOT applied
        ODT_UpdateWeights(ctx, batch_size)    // ← Apply averaged gradients

---

## 11. Logging System

The library uses compile-time controlled logging via the `ODT_LOG` macro:

**DEBUG build (`-DDEBUG`):**

    #define ODT_LOG(...) printf(__VA_ARGS__)

All log statements compile to `printf` calls, producing detailed console output.

**DEBUG build with interrupt safety (`-DDEBUG -DUSE_INTERRUPT_SAFE_LOGGING`):**

    #define ODT_LOG(...) do { DINT; printf(__VA_ARGS__); EINT; } while(0)

Same as above, but disables interrupts during printing to prevent garbled output when logging from interrupt context.

**Release build (no `-DDEBUG`):**

    #define ODT_LOG(...) ((void)0)

All log statements compile to nothing — zero code size, zero runtime overhead.

---

## 12. Writing a Custom Training Loop

This section shows how to use the library directly to implement your own training loop. This is useful if you want full control over the training process.

### Understanding the data flow

Before writing the training loop, it helps to understand how data flows from raw input to trainable model output:

    Raw sensor data (e.g., accelerometer samples)
            │
            ▼
    Feature Extraction (FE)  ← User must implement if they need additional ones that are not offered by Tinyml-Tensorlab
            │
            ▼
    feature_input [FE_NN_OUT_SIZE]  (extracted features)
            │
            ▼
    Frozen Model (tvmgen_default_run)  ← Compiled model from mod.a
            │
            ▼
    frozen_output [FROZEN_OUTPUT_SIZE]  (input to trainable model)
            │
            ▼
    Trainable Model (ODT_Forward)  
            │
            ▼
    trainable_output [FINAL_OUTPUT_SIZE]  (model prediction)

The **feature extraction** and **frozen model inference** steps are application-specific — you must implement them based on your sensor data and model architecture. For the fan blade example, this involves FFT-based feature extraction followed by `tvmgen_default_run()`. See `application_main.c` in the SDK example for a complete implementation.

The ODT library only handles the trainable model portion — everything from `frozen_output` onward.

### Complete training loop example

This example assumes batch_size=1 (the default and recommended setting for on-device training).

    #include "ondevice_training_lib.h"
    #include "trainable_model_config.h"
    #include "ondevice_training_data.h"
    #include "tvmgen_default.h"

    // ---------------------------------------------------------------
    // Buffers
    // ---------------------------------------------------------------
    // frozen_output: the frozen model's output, which becomes the trainable model's input. Size = FROZEN_OUTPUT_SIZE.
    float frozen_output[FROZEN_OUTPUT_SIZE];

    // trainable_output: the trainable model's prediction. Size = FINAL_OUTPUT_SIZE.
    float trainable_output[FINAL_OUTPUT_SIZE];

    // loss_gradient: gradient of the loss w.r.t. trainable_output. Passed to ODT_Backward to start backpropagation.
    float loss_gradient[FINAL_OUTPUT_SIZE];

    // target: the ground truth for loss computation.
    // The application is responsible for providing the correct target for each sample.
    float target[FINAL_OUTPUT_SIZE];

    // ---------------------------------------------------------------
    // Application-specific: prepare frozen_output and target from one raw sample
    // ---------------------------------------------------------------
    // You MUST implement this function for your application.
    // It should:
    //   1. Run feature extraction on the raw sample data (e.g., FFT, mel spectrogram — depends on your pipeline)
    //   2. Run the frozen model on the extracted features (tvmgen_default_run from the compiled mod.a)
    //   3. Write the frozen model's output into frozen_out[]
    //   4. Write the appropriate target into target[]
    //
    // Pseudocode:
    //   void prepare_sample(const float* raw_sample, float* frozen_out, float* target) {
    //       float features[FEATURE_SIZE];
    //       run_feature_extraction(raw_sample, features);
    //       // Set target based on your task (application-specific)
    //       set_target(target, ...);
    //       struct tvmgen_default_inputs  in  = { features };
    //       struct tvmgen_default_outputs out = { frozen_out };
    //       tvmgen_default_run(&in, &out);
    //   }
    void prepare_sample(const float* raw_sample, float* frozen_out, float* target);

    void train_model(void) {

        // =========================================================
        // Step 1: Initialize the trainable model
        // =========================================================
        // ODT_Init reads the auto-generated model config, sets up all weight/buffer pointers, and validates the architecture.
        ModelContext_t ctx;
        ODT_Init(&ctx, 0.0001f);  // learning rate = 1e-4

        float best_val_loss = 1e30f;
        int epochs_without_improvement = 0;
        int max_epochs = 200;
        int patience = 10;

        for (int epoch = 0; epoch < max_epochs; epoch++) {

            // =========================================================
            // Step 2: Training phase — learn from training samples
            // =========================================================
            float epoch_train_loss = 0.0f;

            for (int s = 0; s < NUM_TRAIN_SAMPLES; s++) {

                // (a) Prepare input: raw data → FE → frozen model → frozen_output + target
                prepare_sample(TRAIN_INPUTS[s], frozen_output, target);

                // (b) Forward pass: frozen_output → trainable layers → trainable_output
                ODT_Forward(&ctx, frozen_output, trainable_output);

                // (c) Compute loss: how different is the prediction from the target?
                float sample_loss;
                ODT_MSELoss(trainable_output, target, FINAL_OUTPUT_SIZE, &sample_loss);
                epoch_train_loss += sample_loss;

                // (d) Compute loss gradient: dLoss/d(trainable_output)
                ODT_MSEGradient(trainable_output, target, loss_gradient, FINAL_OUTPUT_SIZE);

                // (e) Backward pass: propagate gradient through all layers and update weights immediately (batch_size=1)
                ODT_Backward(&ctx, loss_gradient);

                // Note: For batch_size=1, weights are already updated inside ODT_Backward.
                // For batch_size>1, you would call ODT_UpdateWeights(&ctx, batch_size) after accumulating gradients over all samples in the batch.
            }

            // =========================================================
            // Step 3: Validation phase — evaluate without learning
            // =========================================================
            // Same forward + loss as training, but NO backward pass. We only measure loss to decide if the model is improving.
            float epoch_val_loss = 0.0f;

            for (int s = 0; s < NUM_VALIDATION_SAMPLES; s++) {

                prepare_sample(VALIDATION_INPUTS[s], frozen_output, target);

                ODT_Forward(&ctx, frozen_output, trainable_output);

                float sample_loss;
                ODT_MSELoss(trainable_output, target, FINAL_OUTPUT_SIZE, &sample_loss);
                epoch_val_loss += sample_loss;
            }
            float avg_val_loss = epoch_val_loss / NUM_VALIDATION_SAMPLES;

            // =========================================================
            // Step 4: Checkpointing and early stopping
            // =========================================================
            if (avg_val_loss < best_val_loss - 0.5f) {
                best_val_loss = avg_val_loss;
                epochs_without_improvement = 0;
                ODT_SaveBestWeights(&ctx);  // Snapshot current weights
            } else {
                epochs_without_improvement++;
                if (epochs_without_improvement >= patience) {
                    break;  // Stop — model is no longer improving
                }
            }
        }

        // =========================================================
        // Step 5: Restore best weights
        // =========================================================
        // The last epoch may have overfit. Load the checkpoint from the epoch with the lowest validation loss.
        ODT_LoadBestWeights(&ctx);

        // Model is now ready for inference with best-performing weights.
    }

### How this example works

**Where the data comes from:** `TRAIN_INPUTS`, `VALIDATION_INPUTS`, and their corresponding labels are defined in `ondevice_training_data.h/.c` — auto-generated arrays containing a small number of samples exported from the PC-side training pipeline. Each sample is a flat float array of `RAW_INPUT_SIZE` elements representing raw sensor data.

**Data flow for one training sample:**

1. `prepare_sample()` — Your application-specific function takes one raw sample, runs feature extraction (FFT, binning, etc.), runs the frozen model via `tvmgen_default_run()`, and sets the `target` appropriate for your task. The frozen model output is written to `frozen_output[]`, a `FROZEN_OUTPUT_SIZE`-element float array. This is the only part you need to implement yourself. See `application_main.c` in the SDK example for a complete implementation.

2. `ODT_Forward()` — Takes `frozen_output` as input, passes it through all trainable layers (Linear, ReLU, etc.), and writes the prediction to `trainable_output[]`. Intermediate activations are saved internally for the backward pass.

3. `ODT_MSELoss()` — Compares `trainable_output` to `target` and computes the loss value. 

4. `ODT_MSEGradient()` — Computes how each element of `trainable_output` contributed to the loss. This produces `loss_gradient[]`, which tells the backward pass which direction to adjust the weights.

5. `ODT_Backward()` — Propagates `loss_gradient` backward through all layers in reverse order. With batch_size=1, weights are updated immediately inside this call — no separate update step is needed.

**Training vs. validation:** Both phases run steps 1–3 (prepare → forward → loss). The critical difference is that **training also runs steps 4–5** (gradient → backward) to update weights, while **validation does not**. Never call `ODT_Backward` during validation — it would modify the weights and corrupt the model.

**Early stopping and checkpointing:** After each epoch, the average validation loss is compared against the best seen so far. If it improved, `ODT_SaveBestWeights` snapshots the current weights. If it hasn't improved for `patience` consecutive epochs, training stops early. At the end, `ODT_LoadBestWeights` restores the best checkpoint — this is important because the last epoch's weights may have started overfitting.

### Key rules

1. **Always call `ODT_Init` first.** It validates the model configuration and sets up all internal pointers.
2. **Forward before backward.** The backward pass reads activations saved during the forward pass. Calling backward without a prior forward is undefined behavior.
3. **Never call backward during validation.** Validation is read-only — forward + loss computation only.
4. **Save best weights on improvement, load best at end.** This ensures the final model is the best-performing one, not the last one.
5. **For batch_size=1:** `ODT_ZeroGradients` and `ODT_UpdateWeights` are not needed — weights are updated inside `ODT_Backward`.
6. **For batch_size>1:** Call `ODT_ZeroGradients` before each batch and `ODT_UpdateWeights` after each batch.

---

## 13. How to Add a New Layer Type

To extend the library with a new layer type (e.g., Conv1D):

**Step 1:** Add the new enum value in `trainable_model_config.h`:

    typedef enum {
        LAYER_TYPE_LINEAR,
        LAYER_TYPE_RELU,
        LAYER_TYPE_CONV1D,    // ← New
    } LayerType_t;

**Step 2:** Add shape metadata to the `LayerParams_t` union (already has a placeholder):

    union {
        struct { 
            uint16_t rows;
            uint16_t cols;
        } linear;
        struct { 
            uint16_t out_channels;
            uint16_t in_channels;
            uint16_t kernel_size;
            uint16_t stride;
            uint16_t padding;
         } conv1d;       // add new layer info
    } shape;

**Step 3:** Implement the forward function:

    void ODT_Conv1DForward(float* input, float* output, LayerParams_t* layer);

**Step 4:** Implement the backward function:

    void ODT_Conv1DBackward(ModelContext_t* ctx, float* grad_output, float* grad_input, float* input, LayerParams_t* layer);

**Step 5:** Add cases to `ODT_Forward` and `ODT_Backward` switch statements:

    case LAYER_TYPE_CONV1D:
        ODT_Conv1DForward(layer_input, layer_output, layer);
        break;

**Step 6:** Update the Python export pipeline (`ondevice_training.py`) to parse the new ONNX op type and generate the appropriate `LAYER_PARAMS_INIT` entries.

---

## 14. API Quick Reference

| Function | Purpose | Returns |
|----------|---------|---------|
| `ODT_Init(ctx, lr)` | Initialize model context, validate config, set up pointers | 0 or -1 |
| `ODT_Forward(ctx, input, output)` | Forward pass through all trainable layers | 0 or -1 |
| `ODT_Backward(ctx, loss_gradient)` | Backward pass, compute gradients, update weights (batch=1) | 0 or -1 |
| `ODT_UpdateWeights(ctx, batch_size)` | Apply accumulated gradients (batch>1 only, no-op for batch=1) | 0 or -1 |
| `ODT_ZeroGradients(ctx)` | Reset gradient accumulators (batch>1 only, no-op for batch=1) | 0 or -1 |
| `ODT_SaveBestWeights(ctx)` | Copy current weights to best weights checkpoint | 0 or -1 |
| `ODT_LoadBestWeights(ctx)` | Restore best weights to current weights | 0 or -1 |
| `ODT_MSELoss(pred, target, size, loss)` | Compute Mean Squared Error loss | 0 or -1 |
| `ODT_MSEGradient(pred, target, grad, size)` | Compute MSE gradient: 2/N × (pred - target) | 0 or -1 |
| `ODT_LinearForward(in, out, layer)` | Linear layer: out = W×in + b | void |
| `ODT_LinearBackward(ctx, grad_out, grad_in, in, layer)` | Linear backward: input grad + weight update/accumulate | void |
| `ODT_ReLUForward(in, out, size)` | ReLU: out = max(0, in) | void |
| `ODT_ReLUBackward(grad_out, grad_in, fwd_out, size)` | ReLU backward: pass grad where fwd > 0 | void |

---