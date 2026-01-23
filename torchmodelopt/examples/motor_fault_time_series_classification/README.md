# Motor Fault Time Series Classification (Example)

1) Purpose
----------
Compact, runnable example that trains a small CNN on motor vibration data, demonstrates quantization (QAT/PTQ), exports an ONNX model, and evaluates accuracy.

2) Quickstart
-------------
- Prepare a CSV with columns: `Vibx`, `Viby`, `Vibz`, `Target` (see Data format below).
- Run the example:

```bash
python motor_fault_classification_tinpu_quant.py
```

- To enable quantization, edit the constants at the bottom of the script: `QUANTIZATION_METHOD`, `WEIGHT_BITWIDTH`, `ACTIVATION_BITWIDTH`, `QUANTIZATION_DEVICE_TYPE`.

3) Data format
--------------
One row per time sample. Required columns (comma-separated):

```
Vibx, Viby, Vibz, Target
0.12, -0.04, 0.01, 0
```

The script concatenates rows into sliding windows controlled by `WINDOW_LENGTH` and `WINDOW_OFFSET`.

4) Key configuration (in-script)
--------------------------------
- `WINDOW_LENGTH`, `WINDOW_OFFSET`, `BATCH_SIZE` — windowing and batching
- `WEIGHT_BITWIDTH`, `ACTIVATION_BITWIDTH`, `QUANTIZATION_METHOD` — quantization settings
- `QUANTIZATION_DEVICE_TYPE` — `'TINPU'` or `'GENERIC'`
- `NORMALIZE_INPUT` — whether to apply batch normalization to inputs

5) How the script maps to functions (quick developer guide)
---------------------------------------------------------
- Data loading and windowing: `get_dataset_from_csv()` and `MotorFaultDataset`
- Dataloader creation: `get_dataloader()`
- Model definition: `get_nn_model()` (returns `NeuralNetwork` class instance)
- Training loop: `train()` and `train_model()`
- PTQ calibration: `calibrate()` and `calibrate_model()`
- Quantization wrapper selection: `get_quant_model()` (chooses TINPU/GENERIC and QAT/PTQ)
- Export: `export_model()` (exports ONNX and optionally converts quant modules)
- Evaluation: `validate_model()` (PyTorch) and `validate_saved_model()` (ONNX Runtime)

6) Quantization guidance (practical)
-----------------------------------
- Default (safe): use `QUANTIZATION_METHOD = 'PTQ'` with `WEIGHT_BITWIDTH = 8` and `ACTIVATION_BITWIDTH = 8`.
- If accuracy degrades and you can retrain: switch to `QAT`, use a smaller learning rate and more fine-tuning epochs (recommended for 4-bit/2-bit).
- For `WEIGHT_BITWIDTH <= 4` or `ACTIVATION_BITWIDTH < 8`: prefer `QAT`, per-channel weight quantization, and careful tuning (calibration, clipping, bias correction).
- Device-specific: `TINPU` commonly prefers symmetric per-channel weight quantization and power-of-two scales; check device constraints.
- PTQ calibration: use representative inputs (hundreds to a few thousand windows). Poor calibration causes large activation errors.

7) Output & troubleshooting
---------------------------
- Trained models and an ONNX export are written to the current working directory (see `MODEL_NAME`).
- The script prints training progress, confusion matrix, and final accuracy metrics.
- Quick tests: reduce `WINDOW_LENGTH`, `BATCH_SIZE`, and `NUM_EPOCHS` to iterate faster.
- If memory is constrained, run on CPU or reduce `BATCH_SIZE`.

8) Files
--------
- `motor_fault_classification_tinpu_quant.py` — runnable example script
- `motor_fault_dataset.csv` (expected) — input dataset (not included)

9) Notes
--------
This is a compact educational example. Adapt preprocessing, augmentation and model size for production or larger datasets.