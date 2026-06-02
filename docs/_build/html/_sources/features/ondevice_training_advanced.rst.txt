=============================================
On-Device Training — Advanced API & Config
=============================================

.. _feature-ondevice-training-advanced:

Deep dive into ODT library API and trainable model configuration.

Library Architecture
====================

The ODT system has three layers:

.. code-block:: text

   ┌─────────────────────────────┐
   │   Application Code          │  User's training loop
   │   (application_main.c)      │
   └─────────────────────────────┘
                 ↓
   ┌─────────────────────────────┐
   │   Task-Specific Library     │  Anomaly detection, classification
   │   (e.g., anomaly_odt.c)     │  Orchestrates training workflow
   └─────────────────────────────┘
                 ↓
   ┌─────────────────────────────┐
   │   Core Training Library     │  Forward pass, backward pass, SGD
   │   (ondevice_training_lib.c) │  Task-agnostic
   └─────────────────────────────┘
                 ↓
   ┌─────────────────────────────┐
   │   Model Configuration       │  Auto-generated
   │   (trainable_model_config)  │  Weights, buffers, layer defs
   └─────────────────────────────┘

**Core Library Files:**

- ``ondevice_training_lib.h`` — API declarations, structures
- ``ondevice_training_lib.c`` — Forward/backward pass, SGD, weight updates
- Location: ``{device-sdk}/ai/common/ondevice_training/``

**Key Principles:**

- **Task-agnostic** — library knows nothing about classification vs anomaly detection
- **Zero dynamic allocation** — all memory pre-allocated statically
- **Flat array architecture** — weights, gradients, buffers in contiguous arrays
- **Compile-time optimization** — batch size = 1 eliminates accumulation code

ModelContext_t — Central Training State
========================================

Every ODT function takes ``ModelContext_t*`` as first argument.

**Structure fields:**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``layers[]``
     - ``LayerParams_t[NUM_TRAINABLE_LAYERS]``
     - Layer descriptors (copied from config during init)
   * - ``num_layers``
     - ``uint16_t``
     - Number of trainable layers
   * - ``intermediate_buffers[]``
     - ``float*[NUM_TRAINABLE_LAYERS + 1]``
     - Forward pass activation storage. Index 0 = frozen model output. Index i = output of layer i-1
   * - ``gradient_buffers[]``
     - ``float*[NUM_TRAINABLE_LAYERS + 1]``
     - Backward pass gradients. Index i+1 = gradient into layer i. Index i = gradient out of layer i
   * - ``current_weights``
     - ``float*``
     - Points to all trainable weights (being updated)
   * - ``best_weights``
     - ``float*``
     - Checkpoint storage for best-performing weights
   * - ``weight_gradients``
     - ``float*``
     - Accumulated gradients (only if ``USE_GRADIENT_ACCUMULATION=1``)
   * - ``batch_sample_count``
     - ``uint16_t``
     - Samples processed in current batch
   * - ``is_training_mode``
     - ``TrainingPhase_t``
     - ``PHASE_TRAINING`` or ``PHASE_INFERENCE``
   * - ``learning_rate``
     - ``float``
     - SGD learning rate (e.g., 0.001)

Memory Layout
=============

All three flat arrays follow the same indexing:

**Example:** Linear(4→3) → ReLU(3) → Linear(3→2)

.. code-block:: text

   ALL_WEIGHTS[23]  (total params = 4×3 + 3 + 3×2 + 2 = 23)
   ┌──────────────────┬─────────┬──────────────────┬─────────┐
   │ Layer 0 weights  │ biases  │ Layer 2 weights  │ biases  │
   │ W[3×4] = 12      │ b = 3   │ W[2×3] = 6       │ b = 2   │
   └──────────────────┴─────────┴──────────────────┴─────────┘
    offset=0          offset=12 offset=15          offset=21

   INTERMEDIATE_BUFFERS[12]  (4 + 3 + 3 + 2)
   ┌──────┬────────┬────────┬────────┐
   │  in  │ L0_out │ L1_out │ L2_out │
   │  (4) │  (3)   │  (3)   │  (2)   │
   └──────┴────────┴────────┴────────┘

   GRADIENT_BUFFERS[12]  (same layout as intermediate)
   ┌──────┬────────┬────────┬────────┐
   │ grad │ grad   │ grad   │  loss  │
   │input │ into L0│ into L1│gradient│
   └──────┴────────┴────────┴────────┘

Forward/Backward Flow:

- **Forward:** frozen_output → buffer[0] → Layer 0 → buffer[1] → Layer 1 → buffer[2] → Layer 2 → buffer[3]
- **Backward:** loss_grad → grad_buffer[3] → Layer 2 → grad_buffer[2] → Layer 1 → grad_buffer[1] → Layer 0 → grad_buffer[0]

Core API
========

Initialization
--------------

.. code-block:: c

   int ODT_Init(ModelContext_t* ctx, float learning_rate);

**Parameters:**

- ``ctx`` — Model context to initialize
- ``learning_rate`` — SGD learning rate (e.g., 0.0001). Must be > 0

**Returns:** 0 on success, non-zero on error

**Actions:**
- Copies layer parameters from ``LAYER_PARAMS_INIT`` to ``ctx->layers[]``
- Initializes all buffer pointers from offsets table
- Sets ``is_training_mode = PHASE_INFERENCE``

Forward Pass
------------

.. code-block:: c

   int ODT_Forward(ModelContext_t* ctx);

Computes activations for all layers sequentially:

1. Read input from ``intermediate_buffers[0]`` (frozen model output)
2. For each layer i:
   - Read input from ``intermediate_buffers[i]``
   - Compute layer operation (linear, ReLU, etc.)
   - Write output to ``intermediate_buffers[i+1]``
3. Final output in ``intermediate_buffers[num_layers]``

**Usage:**

.. code-block:: c

   ODT_Forward(&ctx);
   float* output = ctx.intermediate_buffers[ctx.num_layers];

Backward Pass
-------------

.. code-block:: c

   int ODT_Backward(ModelContext_t* ctx);

Computes weight gradients for all layers:

1. Loss gradient placed in ``gradient_buffers[num_layers]``
2. For each layer i (reverse order):
   - Read incoming gradient from ``gradient_buffers[i+1]``
   - Compute gradients w.r.t. weights using input from ``intermediate_buffers[i]``
   - Accumulate into ``weight_gradients`` (if enabled)
   - Compute outgoing gradient (for layer below)
   - Write outgoing gradient to ``gradient_buffers[i]``

**Requires:** Forward pass already completed, ``gradient_buffers[num_layers]`` populated with loss gradient

Weight Update (SGD)
-------------------

.. code-block:: c

   int ODT_SGDUpdate(ModelContext_t* ctx);

Updates all weights using accumulated gradients:

.. code-block:: c

   current_weights[i] -= learning_rate * weight_gradients[i]

Then resets gradient accumulators to zero.

**When to call:** After ``ODT_Backward()`` completes

Loss Functions
--------------

**Classification (CrossEntropy):**

.. code-block:: c

   float ODT_LossCrossEntropy(const float* output, const uint16_t* targets,
                              uint16_t num_classes, uint16_t batch_size);

**Anomaly Detection (MSE):**

.. code-block:: c

   float ODT_LossMSE(const float* output, const float* target,
                     uint16_t size, uint16_t batch_size);

Weight Checkpointing
--------------------

Save/restore best weights seen during training:

.. code-block:: c

   void ODT_SaveBestWeights(ModelContext_t* ctx);
   void ODT_RestoreBestWeights(ModelContext_t* ctx);

**Usage pattern:**

.. code-block:: c

   for (int epoch = 0; epoch < max_epochs; epoch++) {
       ODT_Forward(&ctx);
       float loss = ODT_LossMSE(...);
       if (loss < best_loss) {
           best_loss = loss;
           ODT_SaveBestWeights(&ctx);
       }
       ODT_Backward(&ctx);
       ODT_SGDUpdate(&ctx);
   }
   ODT_RestoreBestWeights(&ctx);  // Use best weights for inference

Trainable Model Configuration
=============================

**Auto-generated files:**

- ``trainable_model_config.h`` — Architecture defines, layer descriptors
- ``trainable_model_config.c`` — Weight storage, buffers

**Generated by:** ModelZoo Python pipeline with ``ondevice_training: enabled``

trainable_model_config.h
------------------------

**Architecture Defines:**

.. code-block:: c

   #define NUM_TRAINABLE_LAYERS 11
   #define FROZEN_OUTPUT_SIZE 48           // Input to first trainable layer
   #define FINAL_OUTPUT_SIZE 48            // Output of last trainable layer
   #define TOTAL_PARAMS 3150               // All weights + biases
   #define TOTAL_INTERMEDIATE_BUFFER_SIZE 252
   #define TOTAL_GRADIENT_BUFFER_SIZE 252

**Enumerations:**

.. code-block:: c

   typedef enum {
       LAYER_TYPE_LINEAR,
       LAYER_TYPE_RELU,
   } LayerType_t;

   typedef enum {
       TASK_TYPE_CLASSIFICATION,
       TASK_TYPE_ANOMALY_DETECTION,
   } TaskType_t;

   typedef enum {
       LOSS_FUNCTION_MSE,
       LOSS_FUNCTION_CROSSENTROPY,
   } LossFunction_t;

**Task Configuration:**

.. code-block:: c

   #define TASK_TYPE TASK_TYPE_ANOMALY_DETECTION
   #define LOSS_FUNCTION LOSS_FUNCTION_MSE
   #define TRAIN_BATCH_SIZE 1
   #define VAL_BATCH_SIZE 1

**Layer Descriptors (auto-generated):**

.. code-block:: c

   static const LayerParams_t LAYER_PARAMS_INIT[NUM_TRAINABLE_LAYERS] = {
       // Layer 0: Linear(48 -> 24)
       {
           .layer_type = LAYER_TYPE_LINEAR,
           .input_size = 48,
           .output_size = 24,
           .weights_offset = 0,
           .num_weights = 48 * 24 + 24  // 1176 (weights + biases)
       },
       // Layer 1: ReLU(24)
       {
           .layer_type = LAYER_TYPE_RELU,
           .input_size = 24,
           .output_size = 24,
           .weights_offset = -1,          // No weights for ReLU
           .num_weights = 0
       },
       // ...more layers...
   };

**Buffer Offsets (auto-generated):**

.. code-block:: c

   static const uint16_t BUFFER_OFFSETS[NUM_TRAINABLE_LAYERS + 1] = {
       0,      // buffer[0]: frozen output (48 floats)
       48,     // buffer[1]: Layer 0 output (24 floats)
       72,     // buffer[2]: Layer 1 output (24 floats)
       // ...more offsets...
   };

trainable_model_config.c
------------------------

**Weight Storage:**

.. code-block:: c

   // All trained weights live here (NOT reinitialized)
   float ALL_WEIGHTS[TOTAL_PARAMS] = { ... };

   // Backup for checkpointing
   float ALL_BEST_WEIGHTS[TOTAL_PARAMS];

   // Gradient accumulation (only if USE_GRADIENT_ACCUMULATION=1)
   float ALL_WEIGHT_GRADS[TOTAL_PARAMS];

**Buffer Storage:**

.. code-block:: c

   // Forward pass activations
   float INTERMEDIATE_BUFFERS[TOTAL_INTERMEDIATE_BUFFER_SIZE];

   // Backward pass gradients
   float GRADIENT_BUFFERS[TOTAL_GRADIENT_BUFFER_SIZE];

**Memory Sections:**

Can be placed in specific MCU memory regions:

.. code-block:: c

   #pragma DATA_SECTION(ALL_WEIGHTS, ".trainable_weights")
   #pragma DATA_SECTION(INTERMEDIATE_BUFFERS, ".fast_ram")
   #pragma DATA_SECTION(GRADIENT_BUFFERS, ".fast_ram")

Custom Training Loop Example
=============================

**Complete epoch:**

.. code-block:: c

   // Initialize
   ModelContext_t ctx;
   ODT_Init(&ctx, learning_rate=0.001);

   // Training epoch
   for (int batch = 0; batch < num_batches; batch++) {
       // Load batch
       float* frozen_output = get_frozen_model_output();
       memcpy(ctx.intermediate_buffers[0], frozen_output, FROZEN_OUTPUT_SIZE * sizeof(float));

       // Forward pass
       ODT_Forward(&ctx);

       // Compute loss
       float* target = get_target();
       float loss = ODT_LossMSE(
           ctx.intermediate_buffers[ctx.num_layers],
           target,
           FINAL_OUTPUT_SIZE,
           TRAIN_BATCH_SIZE
       );

       // Log
       printf("Batch %d: loss = %.4f\n", batch, loss);

       // Backward pass
       float loss_gradient = 1.0;  // dL/dOutput
       memcpy(ctx.gradient_buffers[ctx.num_layers], &loss_gradient, sizeof(float));
       ODT_Backward(&ctx);

       // Weight update
       ODT_SGDUpdate(&ctx);
   }

Adding a New Layer Type
=======================

To add support for a new layer (e.g., Conv2D):

**1. Update enumerations in ``trainable_model_config.h``:**

.. code-block:: c

   typedef enum {
       LAYER_TYPE_LINEAR,
       LAYER_TYPE_RELU,
       LAYER_TYPE_CONV2D,  // NEW
   } LayerType_t;

**2. Add layer struct in ``ondevice_training_lib.h``:**

.. code-block:: c

   typedef struct {
       uint16_t kernel_height;
       uint16_t kernel_width;
       uint16_t in_channels;
       uint16_t out_channels;
       uint16_t stride;
       uint16_t padding;
       // ... more params
   } Conv2DParams_t;

**3. Implement forward/backward in ``ondevice_training_lib.c``:**

.. code-block:: c

   void Conv2D_Forward(const float* input, float* output, const Conv2DParams_t* params, const float* weights);
   void Conv2D_Backward(const float* input, const float* grad_output, float* grad_input,
                        float* grad_weights, const Conv2DParams_t* params);

**4. Call from main forward/backward dispatcher:**

.. code-block:: c

   case LAYER_TYPE_CONV2D:
       Conv2D_Forward(input, output, &layer->params.conv2d, weights);
       break;

Batch Size Optimization
=======================

**Compile-time flag (``trainable_model_config.h``):**

.. code-block:: c

   #define USE_GRADIENT_ACCUMULATION (TRAIN_BATCH_SIZE > 1)

**When ``TRAIN_BATCH_SIZE = 1``:**
- Gradient accumulation code is eliminated via ``#if``
- Weight updates happen immediately after backward pass
- Saves memory (no ``ALL_WEIGHT_GRADS`` array)
- Slightly faster (fewer accumulation operations)

**When ``TRAIN_BATCH_SIZE > 1``:**
- Gradients accumulated over batch
- One weight update per batch (not per sample)
- Better gradient estimates
- More memory required

Logging System
==============

**Macros for debugging:**

.. code-block:: c

   ODT_LOG("format string", args);      // Info level
   ODT_LOG_DEBUG("format string", args); // Debug level
   ODT_LOG_ERROR("format string", args); // Error level

**Control verbosity via:**

.. code-block:: c

   #define ODT_LOG_LEVEL ODT_LOG_DEBUG  // Set in trainable_model_config.h

**Output:** Logged data useful for on-device training diagnostics

API Quick Reference
====================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Purpose
   * - ``ODT_Init(ctx, lr)``
     - Initialize model context with learning rate
   * - ``ODT_Forward(ctx)``
     - Compute forward pass activations
   * - ``ODT_Backward(ctx)``
     - Compute gradients (requires loss gradient in grad_buffers)
   * - ``ODT_SGDUpdate(ctx)``
     - Update weights using accumulated gradients
   * - ``ODT_LossMSE(out, target, size, batch)``
     - Compute MSE loss (anomaly detection)
   * - ``ODT_LossCrossEntropy(out, targets, classes, batch)``
     - Compute cross-entropy loss (classification)
   * - ``ODT_SaveBestWeights(ctx)``
     - Checkpoint current weights to best_weights
   * - ``ODT_RestoreBestWeights(ctx)``
     - Restore best-performing weights
   * - ``ODT_SetTrainingMode(ctx, PHASE_TRAINING)``
     - Switch to training mode (enable dropout, batch norm, etc.)
   * - ``ODT_SetInferenceMode(ctx, PHASE_INFERENCE)``
     - Switch to inference mode

Related Documentation
=====================

- :doc:`/features/ondevice_training` — High-level ODT overview
- :doc:`/examples/fall_detection_classification` — ODT usage in fall detection
- :doc:`/examples/motor_bearing_fault` — ODT for fault detection
