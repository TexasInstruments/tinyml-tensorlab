# Integer ODL Library: API Reference

## Table of Contents

- [1. Overview](#1-overview)
- [2. Public API](#2-public-api)
  - [2.1 INTODT_Infer](#21-intodt_infer)
  - [2.2 INTODT_Forward](#22-intodt_forward)
  - [2.3 INTODT_Backward](#23-intodt_backward)
- [4. The User Loss Function](#4-the-user-loss-function)
- [5. Writing a Complete Training Loop](#5-writing-a-complete-training-loop)
- [6 Writing an Inference-Only Loop](#6-writing-an-inference-only-loop)
- [7. Tuning mu_weight and mu_offset](#7-tuning-mu_weight-and-mu_offset)

---

## 1. Overview

The integer ODL library (`integer_odl.h` / `integer_odl.c`) provides three public functions that implement integer neural network forward pass, backward pass, and inference. Everything else (loss computation, training loop structure, epoch management) is your responsibility in `main.c`.

**Files:**
- Header: `integer_ondevice_training/integer_odl.h`
- Implementation: `integer_ondevice_training/integer_odl.c`
- Both files are linked into your CCS project from the SDK `common/` directory

**Dependencies:** `integer_model_config.h`, `<stdio.h>`, `<string.h>`, `<math.h>`, `<inttypes.h>`

**No initialization required.** `INT_MODEL` in `integer_model_config.c` is statically initialized. Call any of the three functions immediately after flash.

### Input Type: Which Mode Are You In?

The `input_type` field in `IntModelCtx_t` controls how the library interprets the first input to the layer stack. This is set automatically by ModelMaker in `integer_model_config.c`; you do not set it manually. But you need to understand it to write correct `main.c` code.

| `input_type` | Value | When set | What you pass to INTODT_Forward / Infer | Where PREQUANT lives |
|-------------|-------|----------|----------------------------------------|---------------------|
| `INT_INPUT_INT8` | 1 | **Full model ODT** | `int8_t*` TVM output (`FROZEN_OUTPUT_SIZE` bytes) | In TVM `mod.a` (PREQUANT-only frozen model) |
| `INT_INPUT_UINT8` | 2 | **Partial model ODT** | `uint8_t*` TVM output (`FROZEN_OUTPUT_SIZE` bytes) | In TVM `mod.a` (PREQUANT + frozen compute layers) |

**For ModelMaker-generated projects on device:** you are always in `INT_INPUT_INT8` (full model) or `INT_INPUT_UINT8` (partial model) mode. You always run TVM first, then pass TVM's output buffer to the library.

---

## 2. Public API

### 2.1 INTODT_Infer

```c
void INTODT_Infer(IntModelCtx_t *ctx, const void *input, int8_t *logit_output);
```

Runs a **single-sample** inference pass through all layers. Does not write to `act_buf` or `relu_mask`. Does not modify weights. Safe to call at any time, including between batches during training.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `IntModelCtx_t*` | Model context. Pass `&INT_MODEL`. |
| `input` | `const void*` | Pointer to one sample. `int8_t*` if `INT_INPUT_INT8` (full model ODT, TVM PREQUANT output). `uint8_t*` if `INT_INPUT_UINT8` (partial model ODT, TVM frozen output). |
| `logit_output` | `int8_t*` | Caller-allocated buffer of `out_size` bytes (= `N_CLASSES`). Receives the final layer's output. |

Internally, it uses `ctx->grad_buf[0/1]` as ping-pong temporary storage for layer activations (since `act_buf` is not written during inference).

**Typical usage:**
```c
int8_t logits[N_CLASSES];
INTODT_Infer(&INT_MODEL, frozen_model_output, logits);

```

---

### 2.2 INTODT_Forward

```c
void INTODT_Forward(IntModelCtx_t *ctx, const void *batch_input, int32_t batch_size);
```

Runs a **batch** forward pass through all layers. Writes each layer's output to `layer->act_buf` and sets `layer->relu_mask` bits. Must be called before `INTODT_Backward`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `IntModelCtx_t*` | Model context. Pass `&INT_MODEL`. |
| `batch_input` | `const void*` | Pointer to batch data. For full model ODT: `int8_t*` pointing to `batch_size × FROZEN_OUTPUT_SIZE` int8 values (TVM output). For partial model ODT: `uint8_t*` pointing to `batch_size × FROZEN_OUTPUT_SIZE` uint8 values. |
| `batch_size` | `int32_t` | Number of samples in this batch. Must be > 0 and ≤ `BATCH_SIZE`.|

For each sample `b` from 0 to `batch_size-1`:
   - Iterates through all layers from first to last
   - Calls `conv_forward` or `linear_forward` for each
   - Writes output to `layer->act_buf + b * layer->out_size`
   - Writes ReLU mask bits for RELU layers


```c
// Step 1: Prepare batch (float → TVM → INT8 or UINT8)
for (int b = 0; b < batch_size; b++) {
    struct tvmgen_default_inputs  tvm_in  = { .input  = (void *)TRAIN_INPUTS[start + b] };
    struct tvmgen_default_outputs tvm_out = { .output = (void *)(frozen_out + b * FROZEN_OUTPUT_SIZE) };
    tvmgen_default_run(&tvm_in, &tvm_out);
}

// Step 2: Forward pass on the batch
INTODT_Forward(&INT_MODEL, frozen_out, batch_size);
```

Where `frozen_out` is a static buffer of `BATCH_SIZE * FROZEN_OUTPUT_SIZE` bytes.

---

### 2.3 INTODT_Backward

```c
void INTODT_Backward(IntModelCtx_t *ctx, const void *batch_input, int32_t batch_size);
```

Runs a **batch** backward pass. Reads loss gradient `dy` from `ctx->grad_buf[0]`, propagates deltas backward through all layers, and updates weights and offsets in-place.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `IntModelCtx_t*` | Model context. `ctx->grad_buf[0]` must be filled with `dy` before calling. |
| `batch_input` | `const void*` | Same pointer as passed to `INTODT_Forward` for this batch. Used to read input activations for weight gradient computation of the first layer. |
| `batch_size` | `int32_t` | Same value as passed to `INTODT_Forward`. |

`ctx->grad_buf[0]` must contain the loss gradient before calling. Each element `grad_buf[0][b * N_CLASSES + c]` is the gradient of the loss with respect to `logit[b][c]`, saturated to INT8 range. Your `compute_loss` function is responsible for writing this.

The function:
1. Iterates layers from last to first
2. For each layer:
   a. Computes delta propagation to the lower layer (NITI-shifted, written to the other grad_buf)
   b. If the layer below has ReLU: applies relu_mask to the propagated delta
   c. Updates offsets using NITI-shifted sum of deltas
   d. Updates weights using NITI-shifted outer product of input activations and deltas
3. Flips between grad_buf[0] and grad_buf[1] for each layer (ping-pong)

**After this call:** All weights and offsets in `INT_LAYERS[]` have been updated. The next `INTODT_Forward` call uses the updated weights automatically.

---


## 4. The User Loss Function

The loss function is the bridge between `INTODT_Forward` and `INTODT_Backward`. You write it in `main.c`. It must:
1. Read logits from the last layer's `act_buf`
2. Compute gradients per sample per class
3. Write gradients (clamped to INT8) into `ctx->grad_buf[0]`
4. Return total loss

### Reference Implementation: Per-Class MSE Loss

```c
#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))

static int32_t compute_loss(IntModelCtx_t *ctx, const int8_t *labels, int32_t batch_size)
{
    // Get pointers to last layer's output and grad_buf[0]
    IntLayerDesc_t *out_layer  = &ctx->layers[ctx->n_layers - 1];
    int32_t         n_classes  = (int32_t)out_layer->out_size;   // == N_CLASSES
    int8_t         *output_buf = (int8_t *)out_layer->act_buf;   // logits: int8
    int8_t         *dy         = ctx->grad_buf[0];               // write gradients here

    int32_t total_loss = 0;

    for (int32_t b = 0; b < batch_size; b++) {
        for (int32_t c = 0; c < n_classes; c++) {
            int8_t  logit  = output_buf[b * n_classes + c];

            // Target: what this logit should ideally be
            int32_t target = (labels[b] == (int8_t)c)
                             ? (int32_t)TARGET_MAG_CORRECT[c]
                             : (int32_t)TARGET_MAG_WRONG[c];

            int32_t grad = (int32_t)logit - target;

            // Accumulate squared error for logging
            total_loss += grad * grad;

            // Write INT8 gradient, clamp to prevent overflow in backward
            dy[b * n_classes + c] = (int8_t)CLAMP(grad, -128, 127);
        }
    }

    return total_loss;
}
```

This is an MSE loss where the target is not a one-hot label but a class-specific logit magnitude. The model is pushed to output logits near `TARGET_MAG_CORRECT` for the correct class and near `TARGET_MAG_WRONG` for other classes. These targets are calibrated to the model's actual distribution, making convergence fast even on small datasets.

The clamp to INT8 is required because `INTODT_Backward` reads from `grad_buf[0]` as `int8_t*`. The gradient `(logit - target)` could be up to ±255 (e.g., logit=127, target=-128 → grad=255). Clamping to [-128, 127] before writing prevents silent truncation.

---

## 5. Writing a Complete Training Loop

Here is the complete standard pattern for a training loop in `main.c`:

```c
#include <stdio.h>
#include <stdint.h>
#include "integer_odl.h"
#include "integer_model_config.h"
#include "train_data.h"

#define NUM_EPOCHS      30
#define CLAMP(x,lo,hi)  ((x)<(lo)?(lo):(x)>(hi)?(hi):(x))

// Static buffer for TVM frozen model output (INT8 for full model ODT)
// Size: BATCH_SIZE × FROZEN_OUTPUT_SIZE
static int8_t frozen_out[BATCH_SIZE * FROZEN_OUTPUT_SIZE];

static int32_t compute_loss(IntModelCtx_t *ctx, const int8_t *labels, int32_t batch_size)
{
    IntLayerDesc_t *out_layer = &ctx->layers[ctx->n_layers - 1];
    int8_t *logit_buf = (int8_t *)out_layer->act_buf;
    int8_t *dy        = ctx->grad_buf[0];
    int32_t total_loss = 0;

    for (int32_t b = 0; b < batch_size; b++) {
        for (int32_t c = 0; c < N_CLASSES; c++) {
            int32_t grad = (int32_t)logit_buf[b * N_CLASSES + c]
                         - (int32_t)((labels[b] == c) ? TARGET_MAG_CORRECT[c] : TARGET_MAG_WRONG[c]);
            total_loss += grad * grad;
            dy[b * N_CLASSES + c] = (int8_t)CLAMP(grad, -128, 127);
        }
    }
    return total_loss;
}

static int32_t eval_accuracy(void)
{
    int8_t logits[N_CLASSES];
    int32_t correct = 0;

    for (int32_t i = 0; i < N_TRAIN_SAMPLES; i++) {
        // Run TVM frozen model on one sample
        struct tvmgen_default_inputs  tvm_in  = { .input  = (void *)TRAIN_INPUTS[i] };
        struct tvmgen_default_outputs tvm_out = { .output = (void *)frozen_out };
        tvmgen_default_run(&tvm_in, &tvm_out);

        // Run integer ODT inference
        INTODT_Infer(&INT_MODEL, frozen_out, logits);

        // Argmax
        int32_t pred = 0;
        for (int32_t c = 1; c < N_CLASSES; c++)
            if (logits[c] > logits[pred]) pred = c;

        if (pred == (int32_t)TRAIN_LABELS[i]) correct++;
    }
    return correct;
}

int main(void)
{
    // ── Board and peripheral initialization (device-specific) ────────────────
    // Device_init(); SYSCFG_init(); UART_init(); ...

    // ── Print config ──────────────────────────────────────────────────────────
    printf("INPUT_SIZE=%d  N_CLASSES=%d  BATCH_SIZE=%d  EPOCHS=%d  SAMPLES=%d\n",
           INPUT_SIZE, N_CLASSES, BATCH_SIZE, NUM_EPOCHS, N_TRAIN_SAMPLES);

    // ── Pre-training accuracy ─────────────────────────────────────────────────
    int32_t acc = eval_accuracy();
    printf("Accuracy BEFORE: %d / %d  (%.1f%%)\n", acc, N_TRAIN_SAMPLES, 100.0f * acc / N_TRAIN_SAMPLES);

    // ── Compute batch structure ───────────────────────────────────────────────
    int32_t num_full_batches = N_TRAIN_SAMPLES / BATCH_SIZE;
    int32_t remainder        = N_TRAIN_SAMPLES % BATCH_SIZE;
    int32_t num_batches      = num_full_batches + (remainder > 0 ? 1 : 0);

    // ── Training loop ─────────────────────────────────────────────────────────
    for (int32_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        int32_t epoch_loss = 0;

        for (int32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            int32_t batch_size  = (batch_idx < num_full_batches) ? BATCH_SIZE : remainder;
            int32_t sample_start = batch_idx * BATCH_SIZE;

            // Step 1: Run TVM on all samples in this batch
            for (int32_t b = 0; b < batch_size; b++) {
                struct tvmgen_default_inputs  tvm_in  = { .input  = (void *)TRAIN_INPUTS[sample_start + b] };
                struct tvmgen_default_outputs tvm_out = { .output = (void *)(frozen_out + b * FROZEN_OUTPUT_SIZE) };
                tvmgen_default_run(&tvm_in, &tvm_out);
            }

            // Step 2: Integer ODT forward pass
            INTODT_Forward(&INT_MODEL, frozen_out, batch_size);

            // Step 3: Compute loss — writes dy to grad_buf[0]
            epoch_loss += compute_loss(&INT_MODEL, TRAIN_LABELS + sample_start, batch_size);

            // Step 4: Integer ODT backward pass — reads dy from grad_buf[0], updates weights
            INTODT_Backward(&INT_MODEL, frozen_out, batch_size);
        }

        printf("Epoch [%d/%d]  Loss: %.1f\n", epoch + 1, NUM_EPOCHS, (float)epoch_loss / num_batches);
    }

    // ── Post-training accuracy ────────────────────────────────────────────────
    acc = eval_accuracy();
    printf("Accuracy AFTER: %d / %d  (%.1f%%)\n", acc, N_TRAIN_SAMPLES, 100.0f * acc / N_TRAIN_SAMPLES);

    return 0;
}
```

---

## 5b. Partial Model ODT: Training Loop Variant

For partial model ODT (`IS_FULL_MODEL_RETRAIN == 0`), the TVM frozen model runs the PREQUANT block **plus frozen compute layers** and outputs UINT8. The C library (`INT_INPUT_UINT8`) receives that UINT8 directly.

The training loop pattern is nearly identical to full model ODT. The only difference is TVM's output is UINT8 instead of INT8. The `INT_MODEL.input_type` is set to `INT_INPUT_UINT8` in `integer_model_config.c` automatically; you do not need to change any library call.

```c
// Partial model ODT — training loop in main.c
// FROZEN_OUTPUT_SIZE = size of TVM output (last frozen layer's RELU output, UINT8)

static uint8_t frozen_out[BATCH_SIZE * FROZEN_OUTPUT_SIZE];  // uint8 for partial model

for (int32_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    int32_t epoch_loss = 0;

    for (int32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int32_t batch_size   = (batch_idx < num_full_batches) ? BATCH_SIZE : remainder;
        if (batch_size == 0) break;
        int32_t sample_start = batch_idx * BATCH_SIZE;

        // Step 1: Run TVM on each sample — outputs UINT8 (RELU output of last frozen layer)
        for (int32_t b = 0; b < batch_size; b++) {
            struct tvmgen_default_inputs  tvm_in  = { .input  = (void *)TRAIN_INPUTS[sample_start + b] };
            struct tvmgen_default_outputs tvm_out = { .output = (void *)(frozen_out + b * FROZEN_OUTPUT_SIZE) };
            tvmgen_default_run(&tvm_in, &tvm_out);
            // frozen_out[b * FROZEN_OUTPUT_SIZE] now holds UINT8 activations for sample b
        }

        // Step 2–4: same as full model — library uses INT_INPUT_UINT8 automatically
        INTODT_Forward (&INT_MODEL, frozen_out, batch_size);
        epoch_loss += compute_loss(&INT_MODEL, TRAIN_LABELS + sample_start, batch_size);
        INTODT_Backward(&INT_MODEL, frozen_out, batch_size);
    }

    printf("Epoch [%d/%d]  Loss: %.1f\n", epoch + 1, NUM_EPOCHS, (float)epoch_loss / num_batches);
}
```

**Key difference from full model loop:** `frozen_out` is `uint8_t*` not `int8_t*`. That is the only change in your `main.c`.


---

## 6. Writing an Inference-Only Loop

If you only need inference

```c
// Allocate per-call output buffer
static int8_t frozen_out[FROZEN_OUTPUT_SIZE];   // single sample, not batched
int8_t logits[N_CLASSES];

// Run inference on new data from sensor
float sensor_data[INPUT_SIZE];
// ... fill sensor_data from ADC/DMA ...

// Feature extraction
float feature_buf[INPUT_SIZE];
// run featature extraction

// TVM frozen model
struct tvmgen_default_inputs  tvm_in  = { .input  = (void *)feature_buf };
struct tvmgen_default_outputs tvm_out = { .output = (void *)frozen_out };
tvmgen_default_run(&tvm_in, &tvm_out);

// Integer ODT inference
INTODT_Infer(&INT_MODEL, frozen_out, logits);

```
---

## 7. Tuning mu_weight and mu_offset

### Changing mu at Runtime

You can change mu between epochs to implement a learning rate schedule:

```c
    INT_MODEL.mu_weight = 4; 
    INT_MODEL.mu_offset = 4;
```