============================
On-Device Training (ODT)
============================

.. _feature-ondevice-training:

Overview
--------

On-Device Training (ODT) enables machine learning models to **continue training directly on microcontrollers** after deployment. Rather than deploying frozen inference-only models, ODT sends both a frozen feature-extraction backbone and trainable model head to the device, allowing it to adapt to local data in real-world operating conditions.

**Traditional workflow:**

.. code-block:: text

   Train on PC → Compile model → Deploy to MCU → Inference only

**On-device training workflow:**

.. code-block:: text

   Train on PC → Split model → Compile frozen part → Export trainable part
   → Deploy to MCU → Continue training on MCU → Inference

Use Cases
---------

ODT is essential when:

**1. Data Drift**
   Deployment environment differs from training environment. Example: Fan blade anomaly detector trained in a lab encounters different vibration characteristics when installed on a factory floor. ODT allows the model to adapt to the new operating conditions without re-deployment.

**2. Privacy and Security**
   Raw sensor data cannot leave the device due to regulatory or security constraints (e.g., medical devices, industrial systems). ODT allows local adaptation without transmitting sensitive data.

**3. Personalization**
   Each installation has unique characteristics. A motor vibration model trained on one motor type may need to adapt to a different motor. ODT enables per-installation customization without individual model training.

**4. Reduced Re-deployment Cost**
   Without ODT, adapting to a new environment requires: collect data → ship to PC → retrain → recompile → reflash. ODT eliminates this round-trip entirely.

**5. Zero-shot Deployment**
   The trainable portion can deploy with **zero epochs** of PC-side training, allowing the device to train entirely from scratch using locally collected data.

Architecture: Frozen + Trainable Split
--------------------------------------

The model is split into two parts at deployment time:

**Frozen Part (Feature Extractor)**
   - Contains pre-trained convolutional layers, embeddings, or backbone
   - Deployed as compiled inference code
   - Not modified during on-device training
   - Typically 70-90% of model parameters

**Trainable Part (Classification Head)**
   - Lightweight dense layers or simple linear classifier
   - Deployed as weights + gradient computation code
   - Updated during on-device training
   - Typically 10-30% of model parameters

Memory implications:

- Frozen part: weights only (inference)
- Trainable part: weights + activations + gradients (training)
- Total memory ≈ smaller trainable head + accumulated gradient buffers

Supported Task Types
--------------------

ODT is available for:

- **Time Series Classification** — accelerometer, audio, sensor signals
- **Time Series Regression** — forecasting, sensor readings
- **Time Series Anomaly Detection** — detecting out-of-distribution patterns
- **Image Classification** — visual recognition (with reduced input size)

Each task type has its own trainable architecture optimized for MCU memory constraints.

Workflow
--------

**Phase 1: PC-side Preparation**

1. Train full model on PC with training dataset
2. Extract frozen backbone (feature extractor) and trainable head (classifier)
3. Compile frozen backbone to MCU code (NPU or CPU inference)
4. Export trainable head weights in quantized format
5. Generate trainable architecture code for MCU

**Phase 2: MCU Deployment**

1. Flash frozen backbone + trainable head to device
2. Device runs inference with frozen backbone
3. When adaptation needed, device collects local data
4. Device fine-tunes trainable head using local data (SGD, Adam, etc.)
5. Updated weights remain on device

**Phase 3: Continuous Adaptation (Optional)**

1. Device periodically retrains on new data batches
2. Frozen backbone remains unchanged
3. Trainable head converges to new environment characteristics
4. Inference accuracy improves with local adaptation

Configuration
--------------

Enable ODT in your config file:

.. code-block:: yaml

   common:
       task_type: "timeseries_classification"  # or regression, anomaly_detection
       model_name: "generic_timeseries_cnn"

   ondevice_training:
       enabled: true
       split_layer: "before_dense"              # where to split model
       trainable_layers: 2                      # number of trainable layers
       training_method: "sgd"                   # sgd, adam, rmsprop
       learning_rate: 0.001
       epochs_per_batch: 5                      # epochs when training on device
       batch_size: 32
       optimizer_state_size: "minimal"          # minimal, full

Supported Configurations:

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Parameter
     - Description
     - Example Values
   * - ``split_layer``
     - Which layer to split at (frozen before, trainable after)
     - "before_dense", "before_classifier"
   * - ``trainable_layers``
     - Number of trainable layers
     - 1, 2, 3
   * - ``training_method``
     - Optimizer algorithm
     - "sgd", "adam", "rmsprop"
   * - ``learning_rate``
     - Learning rate for on-device training
     - 0.0001 to 0.1
   * - ``epochs_per_batch``
     - Epochs per training batch on MCU
     - 1 to 20
   * - ``batch_size``
     - Training batch size
     - 8, 16, 32, 64
   * - ``optimizer_state_size``
     - Memory mode for optimizer state
     - "minimal", "full"

Memory Considerations
---------------------

ODT requires additional MCU memory for:

1. **Trainable weights** — typically 1-10 KB
2. **Activations** — forward pass outputs (5-50 KB)
3. **Gradients** — backprop gradients (5-50 KB)
4. **Optimizer state** — learning rate schedules, momentum (2-20 KB)

**Total ODT overhead:** 15-130 KB depending on configuration

Devices with sufficient memory:

- MSPM0G5187 (160 KB SRAM) — recommended
- CC1312, CC1314, CC1352, CC1354, CC2755 (20-60 KB SRAM) — limited configs
- CC35X1 (512 KB SRAM) — full support

Limitations
-----------

1. **Frozen backbone is immutable** — only trainable head adapts
2. **Memory-constrained training** — smaller models and batch sizes than PC training
3. **Limited dataset** — device collects data incrementally, not large static datasets
4. **No distributed training** — single-device training only
5. **Lower numerical precision** — quantized weights and activations

Best Practices
--------------

**1. Choose the right split point**
   - Split after feature extraction, before classification
   - Frozen part should be robust to environment variations
   - Trainable part should be small enough for MCU memory

**2. Pre-train the frozen backbone thoroughly**
   - Use large diverse training dataset on PC
   - Ensure backbone generalizes well
   - Frozen part quality determines ceiling performance

**3. Start training early**
   - Begin on-device training shortly after deployment
   - Device needs representative local data to converge
   - Don't wait for accuracy degradation to trigger retraining

**4. Monitor training convergence**
   - Log training loss on device (periodic dumps to host)
   - Stop training if loss plateaus
   - Retrain with different learning rate if needed

**5. Use minimal optimizer state when memory is tight**
   - SGD with minimal state (just gradients, no momentum)
   - Adam requires full optimizer state (2× memory)
   - Trade optimizer capability for memory savings

Examples
--------

See the following examples for ODT workflows:

- :doc:`/examples/fan_blade_fault_classification` — anomaly detection with on-device adaptation
- :doc:`/examples/motor_bearing_fault` — fault detection with environment-specific training

Related Features
----------------

- :doc:`quantization` — quantization compatible with ODT
- :doc:`feature_extraction` — frozen backbone design patterns
- :doc:`auto_quantization` — preparing models for on-device training

FAQ
---

**Q: Can I train the frozen part on device?**

A: No. The frozen part is compiled to MCU code and cannot be modified. Only the trainable head can be updated.

**Q: How much accuracy improvement can I expect?**

A: Typical improvements: 2-5% accuracy gain after 100-500 device training iterations with local data.

**Q: What if device runs out of memory during training?**

A: Reduce batch size, epochs per batch, or trainable layer count. Use minimal optimizer state (SGD instead of Adam).

**Q: Can I update the trainable weights remotely?**

A: Yes. Export trained weights from device, send to host, verify, send updated weights back to device.

**Q: Is ODT compatible with NPU inference?**

A: Yes. Frozen NPU inference + on-device trainable head training (CPU) both supported.

Troubleshooting
---------------

**Training loss not decreasing:**
   - Increase learning rate (start with 0.01)
   - Ensure local data is representative
   - Check if trainable head has sufficient parameters

**Out of memory during training:**
   - Reduce batch size to 8 or 16
   - Reduce epochs per batch to 1-3
   - Use minimal optimizer state (SGD)
   - Reduce trainable layer count to 1

**Inference accuracy dropped after training:**
   - Frozen backbone may not generalize to new data
   - Retrain frozen backbone on PC with larger dataset
   - Reduce learning rate to prevent overfitting on small device dataset

Further Reading
---------------

For in-depth documentation, see:

- `TinyML ModelZoo: On-Device Training Overview <https://github.com/TexasInstruments/tinyml-modelzoo/blob/main/docs/ondevice_training/overview.md>`_
- `On-Device Training Library <https://github.com/TexasInstruments/tinyml-modelzoo/blob/main/docs/ondevice_training/ondevice_training_lib.md>`_
- `On-Device Training Data Handling <https://github.com/TexasInstruments/tinyml-modelzoo/blob/main/docs/ondevice_training/ondevice_training_data.md>`_
- `Trainable Model Configuration <https://github.com/TexasInstruments/tinyml-modelzoo/blob/main/docs/ondevice_training/trainable_model_config.md>`_
- `On-Device Training Application Examples <https://github.com/TexasInstruments/tinyml-modelzoo/blob/main/docs/ondevice_training/application_example.md>`_
