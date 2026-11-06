# Deploying a ModelZoo Run to an SDK Example (Golden-Vector Check)

A ModelZoo run produces everything you need to prove that the model on the
board gives the same answer as the model on your PC. This guide shows how to take
the run's compiled model, feature-extraction config and golden vectors, drop them
into a ready-made SDK example, and see **`All tests have passed.`** on the device.

> Prefer slides? See [slides/deploying_to_sdk_example.html](slides/deploying_to_sdk_example.html).
> New to run outputs? Read [Understanding the Run Output](slides/understanding_run_output.html) first.

---

## At a Glance: the 4 Files

| File | Take it from the run folder | Put it in the imported CCS project | What it does |
|------|-----------------------------|-------------------------------------|--------------|
| `mod.a` | `compilation/artifacts/mod.a` | `artifacts/mod.a` | The compiled model (device-specific library) |
| `tvmgen_default.h` | `compilation/artifacts/tvmgen_default.h` | `artifacts/tvmgen_default.h` | Model API (`tvmgen_default_run`) plus the input normalization (bias/scale/shift) parameters |
| `user_input_config.h` | `training/quantization/golden_vectors/user_input_config.h` | `user_input_config.h` | `#define`s that configure the SDK feature-extraction library to match training |
| `test_vector.c` | `training/quantization/golden_vectors/test_vector.c` | `test_vector.c` | Raw input, expected model input and expected model output (`golden_output`) for a few samples |

The run folder is `data/projects/<dataset>/run/<date-time>/<model_name>/`.

> **Shortcut:** `compilation/<date-time>_<model>_<device>.zip` already bundles the matching set:
> `artifacts/{mod.a, tvmgen_default.h}`, `golden_vectors/{test_vector.c, user_input_config.h}`,
> plus `model_aux.h` (class names) and the post-training-analysis plots. Unzip it and copy from there.

### The one rule: all four files come from the **same run**

The golden outputs and the normalization parameters are tied to the trained weights. Two runs
with the *same* config still differ. For example, the SDK's `arc_fault` example and a fresh
`dc_arc_fault/config_dsk.yaml` run (CLS_1k_NPU) have **identical** `user_input_config.h` defines.
Their `golden_output` differs, though (`{43, -52}` vs `{61, -73}`), and so does the bias in
`tvmgen_default.h` (`20.71` vs `20.52`). If you mix files from different runs, the check fails.

Use **`training/quantization/golden_vectors`** (not `training/base/golden_vectors`). The compiled
`mod.a` is built from `training/quantization/model.onnx`, and the quantized golden output is
`int8_t`, matching the device. The base golden vectors are float outputs of the float model.

---

## Step 1: Pick the SDK and the Example

**SDK by `common.target_device`:**

| `target_device` | SDK (recommended version) | Example location after install | Device folder |
|-----------------|---------------------------|--------------------------------|---------------|
| F28P55, F28P65, F28004, F28003 | C2000Ware 6.01.00.00 | `C2000Ware_6_01_00_00/libraries/ai/examples/<example>/` | `f28p55x/`, `f28p65x/`, `f28004x/`, `f28003x/` |
| F29H85 | F29H85x SDK 1.04 | `f29h85x-sdk_1_04_00_00/examples/ai/<example>/` | `f29h85x/` |
| MSPM33C32 | MSPM33 SDK 1.03.00.00 | MSPM33 SDK `examples/ai/<example>/` | `mspm33c321a/` |
| AM13E2 | MCU-SDK 1.00 | `mcu_sdk/examples/ai/<example>/` | (single folder) |
| AM263 | MCU-PLUS-SDK 11.03 | `mcu_plus_sdk_am263x_11_03_00_00/examples/ai/<example>/` | `am263x_lp/r5fss0-0_nortos/` |

`mod.a` is compiled for one device (instruction set, NPU or not). Use the device folder that
matches `target_device` in your config.

**Example by `common.task_type`:**

| Task | Examples (any of them works) |
|------|------------------------------|
| Classification | `generic_timeseries_classification`, `arc_fault`, `motor_fault`, `blower_imbalance` |
| Regression | `generic_timeseries_regression`, `torque_measurement`, `washing_machine_load_weighing` |
| Forecasting | `generic_timeseries_forecasting`, `forecasting_pmsm_rotor_temp`, `hvac_indoor_temp_forecast` |
| Anomaly detection | `generic_timeseries_anomalydetection`, `fan_blade_anomalydetection` |

Within a task, the examples share the same `application_main.c`. Everything that makes them
"arc fault" or "motor fault" lives in the four files you are about to replace. Pick the one
whose name matches your application, or the `generic_*` one.

---

## Step 2: Import the Example into CCS

1. Open **Code Composer Studio** (20.4.0 or later). Go to **View → Resource Explorer**.
2. Filter by your board (e.g. **LAUNCHXL-F28P55X**) and search the example name (e.g. `f28p55x_arc_fault`).
3. Click **Import**, and install any dependencies it asks for.

The project spec **copies** `artifacts/`, `test_vector.c`, `user_input_config.h` and
`application_main.c` into your workspace. You edit the **workspace copy**; the SDK install stays untouched.

```
f28p55x_arc_fault/            <- imported CCS project (workspace)
├── application_main.c        (leave as is)
├── artifacts/
│   ├── mod.a                 <- replace
│   └── tvmgen_default.h      <- replace
├── test_vector.c             <- replace
├── user_input_config.h       <- replace
├── c2000.syscfg              (leave as is: board, NPU init)
└── lnk.cmd                   (leave as is)
```

---

## Step 3: Replace the 4 Files

Copy the files over the ones in the imported project (keep the same names):

```bash
RUN=tinyml-modelmaker/data/projects/<dataset>/run/<date-time>/<model_name>
PRJ=~/workspace_ccstheia/f28p55x_arc_fault          # your imported project

cp $RUN/compilation/artifacts/mod.a                          $PRJ/artifacts/mod.a
cp $RUN/compilation/artifacts/tvmgen_default.h               $PRJ/artifacts/tvmgen_default.h
cp $RUN/training/quantization/golden_vectors/test_vector.c        $PRJ/test_vector.c
cp $RUN/training/quantization/golden_vectors/user_input_config.h  $PRJ/user_input_config.h
```

On Windows, drag and drop the files in Explorer, or into the CCS Project Explorer.

> **AM13E2:** the project links `artifacts/mod_m33_ti_arm_clang.a`, but ModelZoo writes `mod.a`.
> Copy `mod.a` in **as `mod_m33_ti_arm_clang.a`**.

Quick sanity check before building: the header of `tvmgen_default.h` names the model it was
compiled from (`input model file: .../<model_name>/training/quantization/model.onnx`). That
path should point at the same run as your golden vectors.

---

## Step 4: Build, Flash, Check

1. **Project → Build Project.**
2. Set the active target configuration to the LaunchPad one
   (e.g. `TMS320F28P550SJ9_LaunchPad.ccxml` instead of `TMS320F28P550SJ9.ccxml`).
3. Connect the LaunchPad. **Run → Debug Project** (this flashes the board).
4. Add `test_result` to the **Watch** window. Press **Continue**.

**Expected console (CIO) output for classification / regression / forecasting:**

```
Feature extraction mismatches 0
All tests have passed.
Golden vectors matched: 2 not matched: 0
```

`test_result` reads **1** (PASS). The "matched" count equals the number of model outputs
(`FE_NN_OUT_SIZE`, e.g. 2 classes for arc fault).

### What the example checks

```
raw_input_test ──► FE_runFeatureExtract ──► model_input ──► tvmgen_default_run ──► model_output
                                              │                                    │
                     compare (±1) vs model_test_input       compare (±1) vs golden_output
                     "Feature extraction mismatches N"      "Golden vectors matched: X not matched: Y"
```

- **Check 1 (feature extraction):** the SDK feature-extraction library, configured by
  `user_input_config.h`, must reproduce the features the PC computed. A non-zero count points to
  `user_input_config.h` / `test_vector.c`.
- **Check 2 (model):** the compiled model must reproduce the PC model's output. A mismatch here,
  with check 1 at 0, points to `mod.a` / `tvmgen_default.h`.
- On NPU devices, the app waits for `tvmgen_default_finished` before comparing. The NPU is set up by
  SysConfig in the C2000 examples; the F29H85 examples call `TI_NPU_init()` in `application_main.c`.

**Anomaly detection is different.** Each output may deviate by up to **4**. Then the example
computes the reconstruction error and compares it with `RECONSTRUCTION_ERROR_THRESHOLD`
(defined in `user_input_config.h`). Here `test_result` means **1 = anomaly, 0 = normal**, not
pass/fail. The console prints the error, the threshold and "The test sample is a Normal/Anomaly sample".

---

## Step 5: Try the Other Test Sets

`test_vector.c` holds several sets (e.g. 4 for the arc-fault run: 2 per class). Only **SET 0**
is active.

1. Wrap SET 0 in `/* ... */`, and remove the `/*` and `*/` around another set. Keep exactly one set active.
2. Rebuild, flash, and check the console again.

Each array has a comment naming its class and sample index (e.g. `//Class: class_0_normal (Index: 472)`). `model_aux.h` (in the zip)
maps the output index to the class name (`classIdToName[]`).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Feature extraction mismatches` > 0 | `user_input_config.h` and `test_vector.c` from different runs/configs, or an SDK older than the ModelZoo version | Copy both from the same `golden_vectors/` folder; use the recommended SDK version |
| FE mismatches 0, but `not matched` > 0 | `mod.a`/`tvmgen_default.h` from a different run than `test_vector.c`, or `base/golden_vectors` used with the quantized `mod.a` | Copy all four from the same run, golden vectors from `training/quantization/` (or use the zip) |
| Build errors on `model_input_t` / `model_output_t` types | `user_input_config.h` from another run: `SKIP_NORMALIZE` / `OUTPUT_INT` switch the types to `int8_t` | Use the `user_input_config.h` that matches `mod.a` |
| Linker: undefined `tvmgen_default_run`, or incompatible object | `mod.a` built for another device, or (AM13) wrong file name | Re-run with the right `target_device`; on AM13 rename to `mod_m33_ti_arm_clang.a` |
| `RECONSTRUCTION_ERROR_THRESHOLD` undeclared (anomaly detection) | `user_input_config.h` is missing its defines | Check the file has the FE defines and the threshold; regenerate the run if it is empty |
| Hangs at `while (!tvmgen_default_finished)` | NPU model, but the NPU is not initialized | Keep the NPU module in the example's `.syscfg` (C2000); keep `TI_NPU_init()` (F29H85) |
| Linker: section does not fit (e.g. `FFT_buffer_1`, `.bss`) | Your model or feature extraction is larger than the example's | Start from an example with a similar FE preset; adjust `lnk.cmd` memory placement |
| Output off by more than 1 on the device but not on PC | Not the files: possibly a stale build | **Project → Clean**, rebuild, re-flash |

---

## After the Golden Vectors Pass

The golden-vector check proves that the pipeline is bit-exact. To go live, keep the same
feature-extraction and `tvmgen_default_run` calls in `application_main.c`, and feed
`fe_handle->input_buffer` from your ADC/sensor instead of `raw_input_test`.

Per-task walkthroughs with screenshots:

- [Classification](deploying_classification_models_from_modelmaker_to_device/readme.md)
- [Regression](deploying_regression_models_from_modelmaker_to_device/readme.md)
- [Forecasting](deploying_forecasting_models_from_modelmaker_to_device/readme.md)
- [Anomaly detection](deploying_anomaly_detection_models_from_modelzoo_to_device/readme.md)
