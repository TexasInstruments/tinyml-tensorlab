# On-Device Training — FAQ & Troubleshooting

## Table of Contents

- [1. Training Issues](#1-training-issues)
- [2. Deployment & Build Issues](#2-deployment--build-issues)
- [3. Threshold & Inference Issues](#3-threshold--inference-issues)
- [4. Configuration & YAML Issues](#4-configuration--yaml-issues)
- [5. General Questions](#5-general-questions)

---

## 1. Training Issues

### Training loss is not decreasing

**Symptoms:** Training loss stays flat or decreases extremely slowly across epochs.

**Possible causes and fixes:**
- **Learning rate too low.** Try increasing by 10× (e.g., 0.0001 → 0.001). If loss starts decreasing, you found the issue.
- **Data not reaching the trainable model correctly.** Verify golden vectors match first. If they don't, the frozen model or feature extraction is misconfigured — fix that before training.
- **Too few training samples.** With very few samples (e.g., 5), the model may not have enough signal to learn. Increase `export_samples_per_class` in the YAML and regenerate.
- **Wrong target data.** Ensure the target passed to the loss function is correct for your task.

---

### Training loss explodes or becomes very large

**Symptoms:** Loss increases rapidly, may reach very large values or become `inf`/`NaN`.

**Possible causes and fixes:**
- **Learning rate too high.** This is the most common cause. Reduce by 10× (e.g., 0.001 → 0.0001) and retry.
- **Data contains extreme values.** Check that your training data and feature extraction output are within reasonable ranges. Very large input values can cause gradient explosion.

---

### Training stops too early (early stopping triggers prematurely)

**Symptoms:** Training stops after a few epochs with the message `EARLY STOPPING TRIGGERED`, but the model hasn't converged.

**Possible causes and fixes:**
- **Patience too low.** Increase `patience` (e.g., 10 → 20 or 30). This gives the model more epochs to improve before stopping.
- **`min_improvement_delta` too high.** If set too high, small but genuine improvements are ignored. Try reducing it (e.g., 0.5 → 0.1).
- **Validation loss is noisy due to few samples.** Reduce `ema_alpha` (e.g., 0.5 → 0.3) for more aggressive smoothing, which stabilizes the early stopping signal.

---

### Training runs for too many epochs without converging

**Symptoms:** Training continues for hundreds of epochs, loss decreases very slowly or oscillates without clear convergence.

**Possible causes and fixes:**
- **Learning rate too low.** Increase by 10×. A model that takes hundreds of epochs to converge often just needs a higher learning rate.
- **Patience too high.** Reduce `patience` to stop sooner when the model plateaus.
- **`min_improvement_delta` too low.** Very tiny improvements keep resetting the patience counter. Increase it so only meaningful improvements count.

---

### Validation loss oscillates wildly between epochs

**Symptoms:** Validation loss swings significantly from epoch to epoch (e.g., 100 → 500 → 80 → 600).

**Possible causes and fixes:**
- **Too few validation samples.** With 3–6 samples, validation loss is inherently noisy. Increase `export_samples_per_class` validation count or reduce `ema_alpha` for heavier smoothing.
- **`ema_alpha` too high.** A value of 1.0 means no smoothing at all. Try 0.3–0.5 for more stable EMA tracking.

---

## 2. Deployment & Build Issues

### Golden vectors don't match

**Symptoms:** Console shows `Golden vectors matched: X not matched: Y` where Y > 0.

**Possible causes and fixes:**
- **Mismatched artifacts.** The most common cause is using `mod.a` from one ModelZoo run and `trainable_model_config.c` from another. All artifacts must come from the same run — regenerate and recopy all files together.
- **Wrong feature extraction config.** `user_input_config.h` must match the feature extraction used during training. If you changed the YAML feature extraction settings, regenerate all artifacts.
- **Corrupted flash.** Try a clean flash erase and re-program. Partial flash writes can corrupt data arrays.
- **Compiler/toolchain mismatch.** Ensure the same CGT version is used. Different compiler versions may produce slightly different floating-point results.

---

### Build fails with undefined symbols

**Symptoms:** Linker errors like `undefined symbol: tvmgen_default_run` or `undefined symbol: ALL_WEIGHTS`.

**Possible causes and fixes:**
- **`mod.a` not added to linker.** Ensure `mod.a` is in the project's library search path and included in the linker settings.
- **`trainable_model_config.c` not in the build.** This file defines `ALL_WEIGHTS`, `ALL_BEST_WEIGHTS`, and buffer arrays. Make sure it is added to the project source files, not just the `artifacts/` directory.
- **`ondevice_training_data.c` not in the build.** This file defines the training/validation/test data arrays.

---

### RAM overflow / memory allocation failure

**Symptoms:** Linker error about section overflow, or the device crashes at startup.

**Possible causes and fixes:**
- **Model too large for on-device training.** On-device training requires roughly 2–3× the parameter memory compared to inference only (active weights + best weights + optional gradient accumulators). Check the [Memory Considerations](overview.md#8-memory-considerations) section.
- **Reduce trainable layers.** Use a smaller `trainable_layers_from_last` value to train fewer layers, which reduces weight and buffer sizes.
- **Use batch_size=1.** This eliminates the `ALL_WEIGHT_GRADS` buffer, saving `TOTAL_PARAMS × 4` bytes of RAM. Set `TRAIN_BATCH_SIZE 1` in `trainable_model_config.h`.
- **Check linker command file.** Ensure the memory sections (`trainable_parameters`, `trainable_best_weights`, `intermediate_buffers`, `gradient_buffers`) are mapped to memory regions with sufficient space.
- **Reduce embedded training data.** Lower `export_samples_per_class` values to reduce the size of `ondevice_training_data.c`.

---

## 3. Threshold & Inference Issues

### All samples classified as anomaly after training

**Symptoms:** Every inference frame returns `is_anomaly = 1`, even for known normal data.

**Possible causes and fixes:**
- **Threshold too low.** The model may have trained well (low reconstruction errors on normal data), but the threshold was calculated incorrectly. Check the threshold calculation console output — if the threshold is unreasonably low, the cause is likely bad data during threshold calculation.
- **Anomaly data fed during threshold calculation.** Only normal (non-anomalous) samples must be used for threshold calculation. If anomaly samples are mixed in, the threshold will be inflated in unpredictable ways. Ensure only validation data with label=1 (normal) is fed during `MODE_THRESHOLD_CALC`.
- **Model didn't train properly.** Check training loss — if it's still high, the model hasn't learned to reconstruct normal patterns. See [Training loss is not decreasing](#training-loss-is-not-decreasing).

---

### All samples classified as normal after training

**Symptoms:** Every inference frame returns `is_anomaly = 0`, even for known anomaly data.

**Possible causes and fixes:**
- **Threshold too high.** If using Gaussian method with a high `gaussian_std_multiplier` (e.g., 6+), the threshold may be too permissive. Try reducing it or switching to percentile method.
- **Model overfitting to all data.** If the model was inadvertently trained on anomaly data, it learns to reconstruct anomalies too, reducing the separation. Ensure only normal data is used for training.
- **Too few threshold samples.** With very few samples, the percentile estimate may be inaccurate. Increase `threshold_samples`.

---

### Excessive false positives during inference

**Symptoms:** Normal operation occasionally triggers anomaly alerts.

**Possible causes and fixes:**
- **Transient spikes in reconstruction error.** Increase `inference_window_size` (e.g., 1 → 5) to enable majority voting. This requires multiple consecutive high-error frames before declaring an anomaly.
- **Threshold too tight.** Switch from 95th to 99th percentile, or increase `gaussian_std_multiplier`. A stricter threshold reduces false positives at the cost of potentially missing subtle anomalies.
- **Insufficient threshold data.** Increase `threshold_samples` to better characterize the normal error distribution.

---

## 4. Configuration & YAML Issues

### `quantization` not set to 0

**Symptoms:** Model exports incorrectly or produces wrong results on-device.

**Fix:** On-device training requires float32 for the trainable path. Always set `quantization: 0` in the YAML. See [Running ModelZoo for ODT — YAML Configuration](running_modelzoo_for_odt.md#2-odt-specific-yaml-configuration).

---

### `trainable_layers_from_last` set too high

**Symptoms:** RAM overflow, or the frozen model (`mod.a`) is unexpectedly tiny (just a Flatten operation).

**Explanation:** When `trainable_layers_from_last` equals the total number of main layers, the entire model becomes trainable and the frozen part contains only a pass-through layer. This is valid for small models but may exceed RAM budget for larger ones.

**Fix:** Reduce `trainable_layers_from_last` to keep more layers frozen. This reduces the trainable parameter count and all associated buffers. See [Trainable Model Configuration — How trainable_layers_from_last Maps](trainable_model_config.md#4-how-trainable_layers_from_last-maps-to-these-files).

---

### ModelZoo export step fails or skips ODT export

**Symptoms:** No `trainable_model_config.h/.c` files generated, or the on-device training export log messages don't appear.

**Possible causes and fixes:**
- **`ondevice_training: True` not set.** This flag must be present under the `training:` section in the YAML.
- **Model architecture not supported.** The model must produce a linear (sequential) ONNX graph. Models with skip connections, branches, or unsupported layer types will fail the split. Check the console for error messages during the export step.

---

## 5. General Questions

### Can I train with live sensor data instead of embedded arrays?

**Yes.** The library APIs (`ProcessTrainingFrame`, `ProcessThresholdFrame`, `ProcessInferenceFrame`) accept data buffers as function parameters. They don't care where the data comes from. The embedded arrays in `ondevice_training_data.c` are just a convenience for development and testing. For production, replace the data source with live sensor readings — the API calls remain the same.

---

### Can I retrain the model multiple times?

**Yes.** Call `AnomalyDetection_StartTraining()` again from `MODE_INFERENCE` to begin a new training cycle. The library resets all training state (counters, losses, early stopping) and starts fresh. The model will train starting from its current weights (which may be the result of a previous training cycle).

---

### How do I choose the right learning rate?

Start with **0.0001** (1e-4) as a baseline. Then observe:

| Observation | Action |
|------------|--------|
| Loss not decreasing at all | Increase by 10× (try 0.001) |
| Loss decreasing but very slowly | Increase by 2–5× |
| Loss decreasing then suddenly exploding | Decrease by 10× |
| Loss decreasing smoothly to convergence | Good — keep this value |

For on-device training with SGD and small datasets, learning rates in the range **1e-4 to 1e-3** typically work well. If training the entire model from scratch (0 PC-side epochs), you may need a slightly higher learning rate than when fine-tuning.

---

### What happens if I interrupt training (e.g., power loss)?

The model weights in RAM will be lost. On next boot, the model starts from the initial weights stored in `ALL_WEIGHTS[]` (the PC-trained or initial values baked into `trainable_model_config.c`). There is no automatic persistence of trained weights to non-volatile memory. If you need persistence, you would need to implement a mechanism to save `ALL_WEIGHTS[]` to Flash or EEPROM after training completes.

---

### How many training samples do I need?

There is no fixed answer — it depends on data complexity, model size, and how different the deployment environment is from the training environment. As a rough guide:

| Samples | Expectation |
|---------|-------------|
| 5–10 | Minimum viable. May work for simple adaptation tasks. |
| 20–50 | Reasonable for most fine-tuning scenarios. |
| 100+ | Better convergence and more representative threshold calculation. |

More samples improve training quality but consume more Flash memory. See [On-Device Training Data — Memory Impact](ondevice_training_data.md#6-memory-impact) for the memory formula.

---

### Can I use this with a model that has skip connections or branches?

**Not currently.** The on-device training framework requires a **linear (sequential) ONNX graph** — no branches, skip connections, or parallel paths. The split mechanism and the forward/backward pass implementations assume a simple chain of layers. Supporting graph-structured models would require significant changes to both the Python export pipeline and the C training library.