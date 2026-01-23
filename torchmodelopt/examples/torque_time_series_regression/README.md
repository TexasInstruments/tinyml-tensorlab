# Torque Time Series Regression (Example)

1) Purpose
----------
Compact example demonstrating training and quantization of a small CNN for torque regression from sensor time-series measurements. The script trains a float model, optionally performs quantization (QAT/PTQ), exports an ONNX model and evaluates regression metrics (R2, SMAPE).

2) Quickstart
-------------
- Obtain the CSV dataset (the example script points to a public URL by default). The CSV must contain a `torque` column and sensor feature columns.
- Run the example:

```bash
python torque_regression_tinpu_quant.py
```

- To enable quantization, edit the constants at the bottom of the script: `QUANTIZATION_METHOD`, `WEIGHT_BITWIDTH`, `ACTIVATION_BITWIDTH`, `QUANTIZATION_DEVICE_TYPE`.

3) Data format
--------------
- Required target column: `torque`.
- Input features: sensor columns (the example uses all columns up to `stator_yoke`). The script extracts features with `df.loc[:, :'stator_yoke']` and the target with `df['torque']`.

Example minimal rows (comma-separated):

```
sensor_1, sensor_2, ..., stator_yoke, torque
0.123, -0.045, ..., 0.012, 1.234
```

The script segments samples into windows using `WINDOW_LENGTH` and `WINDOW_OFFSET`.

4) Key configuration (in-script)
--------------------------------
- `WINDOW_LENGTH`, `WINDOW_OFFSET`, `BATCH_SIZE` — windowing and batching
- `NUM_EPOCHS`, `LEARNING_RATE` — training schedule
- `WEIGHT_BITWIDTH`, `ACTIVATION_BITWIDTH`, `QUANTIZATION_METHOD` — quantization settings
- `QUANTIZATION_DEVICE_TYPE` — `'TINPU'` or `'GENERIC'`
- `NORMALIZE_INPUT` — whether to apply batch normalization to inputs

5) How the script maps to functions (developer guide)
---------------------------------------------------
- Data loading and windowing: `get_dataset_from_csv()` and `TorqueMeasurementDataset`
- Dataloader creation: `get_dataloader()`
- Model definition: `get_nn_model()` (returns `NeuralNetwork` instance)
- Training loop: `train()` and `train_model()`
- PTQ calibration: `calibrate()` and `calibrate_model()`
- Quantization wrapper selection: `get_quant_model()` (chooses TINPU/GENERIC and QAT/PTQ)
- Export: `export_model()` (exports ONNX and converts quant modules when requested)
- Evaluation: `validate_model()` (PyTorch) and `validate_saved_model()` (ONNX Runtime)

6) Quantization guidance (practical)
-----------------------------------
- Default (safe): use `QUANTIZATION_METHOD = 'PTQ'` with `WEIGHT_BITWIDTH = 8` and `ACTIVATION_BITWIDTH = 8`.
- If accuracy degrades and you can retrain: switch to `QAT`, use a smaller learning rate and more fine-tuning epochs (recommended for 4-bit and 2-bit regimes).
- For `WEIGHT_BITWIDTH <= 4` or `ACTIVATION_BITWIDTH < 8`: prefer `QAT`, use per-channel weight quantization and careful tuning (calibration, clipping, bias correction).
- Device-specific: `TINPU` often prefers symmetric per-channel weight quantization and power-of-two scales; check device constraints.
- PTQ calibration: use representative inputs (hundreds to a few thousand windows). Poor calibration can cause large activation errors.

7) Output & troubleshooting
---------------------------
- The script writes trained/quantized models and an ONNX export to the working directory (see `MODEL_NAME`).
- The script prints training progress and reports R2 and SMAPE metrics for float and quantized models.
- Quick tests: reduce `WINDOW_LENGTH`, `BATCH_SIZE`, and `NUM_EPOCHS` to iterate faster.
- If memory is constrained, run on CPU or reduce `BATCH_SIZE`.

8) Files
--------
- `torque_regression_tinpu_quant.py` — runnable example script
- `torque_measurement.csv` (expected) — input dataset (not included in this repo)

9) License & notes
-------------------
Test bench measurements collected by LEA Department at Paderborn University.
License: CC BY-SA 4.0.

This is an educational example — adapt preprocessing, augmentation and model size for production or larger datasets.