# Device Deployment Guide

## Overview

After training and compilation, deploy the model to the target MCU. The workflow
has 5 scripted steps (A–D) followed by one manual CCS step (flash).

```
ModelMaker artifacts  →  CCS project  →  build  →  flash  →  verify
     (Step A)             (Step B)       (Step C)  (Step D)   (debug)
```

---

## Prerequisites

Before Step 13:
1. ✓ Training and compilation completed (`run_example` returned success)
2. ✓ `RUN_ID`, `MODEL_ID`, `QUANTIZATION` recorded from Step 12 logs
3. ✓ Code Composer Studio installed (version 12.x or later recommended)
4. ✓ Device-specific SDK installed (C2000Ware / MSPM0 SDK / AM26x SDK)
5. ✓ `CCS_INSTALL_PATH` known — e.g., `/opt/ti/ccs1260`

### Find run_id and model_id from logs

```bash
# List run directories
ls $TINYML_BASE_PATH/tinyml-modelmaker/data/projects/$TASK_TYPE/run/
# Output: 20240515_143022  ← this is RUN_ID

# model_id = model_name from training config
grep model_name $CONFIG_DIR/config.yaml
```

---

## Step A: Verify artifacts exist

```bash
python3 $SCRIPTS_DIR/runner.py find_run_artifacts '{
  "tinyml_base_path": "$TINYML_BASE_PATH",
  "task_type": "$TASK_TYPE",
  "run_id": "$RUN_ID",
  "model_id": "$MODEL_ID",
  "quantization": $QUANTIZATION
}'
```

Checks for all 4 required files:

| File | Source in ModelMaker output |
|---|---|
| `mod.a` | `compilation/artifacts/mod.a` |
| `tvmgen_default.h` | `compilation/artifacts/tvmgen_default.h` |
| `test_vector.c` | `training/quantization/golden_vectors/test_vector.c` |
| `user_input_config.h` | `training/quantization/golden_vectors/user_input_config.h` |

If `success: false`: check that both training and compilation completed without errors.

---

## Step B: Set up CCS project

**Device/SDK reference:** See `assets/deployment_sdk_reference.md` for device family → SDK mapping, device types, and installation paths.

### B1: Check SDK installation

```bash
python3 $SCRIPTS_DIR/runner.py check_sdk_installation \
  '{"target_device": "$TARGET_DEVICE"}'
# Returns: sdk_name, sdk_root, ai_examples_path
# If found: false → errors field has download URL
```

Store returned `ai_examples_path` as `$AI_EXAMPLES_PATH`.

If SDK not found: download from URL in errors field, install, then re-run.

### B2: Create CCS project (auto-copies artifacts)

```bash
python3 $SCRIPTS_DIR/runner.py create_ccs_project '{
  "project_name": "<name>",
  "device_type": "<e.g., f28p55x>",
  "target_device": "$TARGET_DEVICE",
  "run_id": "$RUN_ID",
  "task_type": "$TASK_TYPE",
  "quantization": $QUANTIZATION,
  "model_id": "$MODEL_ID",
  "tinyml_base_path": "$TINYML_BASE_PATH",
  "ccs_templates_path": "$AI_EXAMPLES_PATH"
}'
```

On success: returns `project_path`. Set `CCS_PROJECT_PATH=$project_path`.

`create_ccs_project` automatically copies:
- Artifacts from `{run_path}/compilation/artifacts/` → project
- Golden vectors from `{run_path}/training/[quantization|base]/golden_vectors/` → project

### B3: Verify project structure

After `create_ccs_project`, verify complete file structure. Navigate to `$CCS_PROJECT_PATH` and confirm:

```
$CCS_PROJECT_PATH/
├── application_main.c
└── {device_type}/  ← e.g., f28p55x
    ├── artifacts/
    │   ├── mod.a                  (from compilation artifacts)
    │   └── tvmgen_default.h       (from compilation artifacts)
    ├── c2000.syscfg               (from template)
    ├── CCS/
    │   └── {device_type}_{project_name}.projectspec
    ├── lnk.cmd                    (from template)
    ├── test_vector.c              (from golden vectors)
    └── user_input_config.h        (from golden vectors)
```

**Critical files to verify exist:**

| File | Source | Purpose |
|---|---|---|
| `mod.a` | `{run_path}/compilation/artifacts/mod.a` | Compiled model object |
| `tvmgen_default.h` | `{run_path}/compilation/artifacts/tvmgen_default.h` | Model header |
| `test_vector.c` | `{run_path}/training/[quantization\|base]/golden_vectors/test_vector.c` | Golden test data |
| `user_input_config.h` | `{run_path}/training/[quantization\|base]/golden_vectors/user_input_config.h` | Feature config |

### B4: Manual fallback (if auto-copy failed)

If after running `create_ccs_project` any files are missing:

1. Verify the artifact paths exist:
   ```bash
   ls "$TINYML_BASE_PATH/tinyml-modelmaker/data/projects/$TASK_TYPE/run/$RUN_ID/$MODEL_ID/compilation/artifacts/"
   ls "$TINYML_BASE_PATH/tinyml-modelmaker/data/projects/$TASK_TYPE/run/$RUN_ID/$MODEL_ID/training/quantization/golden_vectors/"
   ```
**If "$CCS_PROJECT_PATH/{device_type}/artifacts/" does not exist, create the directory first**
2. Copy manually:
   ```bash
   # Copy artifacts
   cp "$TINYML_BASE_PATH/tinyml-modelmaker/data/projects/$TASK_TYPE/run/$RUN_ID/$MODEL_ID/compilation/artifacts/"* \
      "$CCS_PROJECT_PATH/{device_type}/artifacts/"
   
   # Copy golden vectors
   cp "$TINYML_BASE_PATH/tinyml-modelmaker/data/projects/$TASK_TYPE/run/$RUN_ID/$MODEL_ID/training/quantization/golden_vectors/test_vector.c" \
      "$CCS_PROJECT_PATH/{device_type}/"
   cp "$TINYML_BASE_PATH/tinyml-modelmaker/data/projects/$TASK_TYPE/run/$RUN_ID/$MODEL_ID/training/quantization/golden_vectors/user_input_config.h" \
      "$CCS_PROJECT_PATH/{device_type}/"
   ```

3. Re-verify structure matches diagram above before proceeding to Step C.

## Step C: Build the project

```bash
python3 $SCRIPTS_DIR/runner.py build_ccs_project '{
  "ccs_project_path": "$CCS_PROJECT_PATH",
  "ccs_install_path": "$CCS_INSTALL_PATH"
}'
```

Returns `out_file` path (e.g., `<project>/Debug/<name>.out`) on success.

**If headless build fails**, build manually in CCS:
- **Project → Build Project** (Ctrl+B)
- Watch Console for errors
- Common fix: missing linker path for `mod.a`

Set active target configuration before flashing:
- Right-click the `.ccxml` file in Project Explorer
- **Set as Active Target Configuration**
- Use the **LaunchPad** variant (e.g., `TMS320F28P550SJ9_LaunchPad.ccxml`)

---

## Step D: Flash to device

```bash
# Connect device via USB/JTAG first

python3 $SCRIPTS_DIR/runner.py flash_ccs_project '{
  "ccs_project_path": "$CCS_PROJECT_PATH",
  "ccs_install_path": "$CCS_INSTALL_PATH"
}'
```

Pass `"ccxml_path"` explicitly if auto-detection picks the wrong file:
```bash
python3 $SCRIPTS_DIR/runner.py flash_ccs_project '{
  "ccs_project_path": "$CCS_PROJECT_PATH",
  "ccs_install_path": "$CCS_INSTALL_PATH",
  "ccxml_path": "<project>/CCS/TMS320F28P550SJ9_LaunchPad.ccxml"
}'
```

**If dslite flash fails**, flash manually in CCS:
- **Run → Flash Project**

---

## Verify model on device

After flashing, CCS opens the Debug perspective automatically:

1. Click **Debug** icon to start a session
2. Set a breakpoint on the line **after** the inference call in `application_main.c`
3. Click **Resume** (F8)
4. Add variable `test_result` to the **Watch** window
5. Check value:
   - `test_result == 1` → inference passed (output matches golden vector) ✓
   - `test_result == 0` → inference failed — see troubleshooting below

---

## Device type → CCS variant mapping

| Device | CCS device_type | Target ccxml |
|---|---|---|
| F28P55 | f28p55x | TMS320F28P550SJ9_LaunchPad.ccxml |
| F28P65 | f28p65x | TMS320F28P650DH9.ccxml |
| F28004 | f28004x | TMS320F280049C_LaunchPad.ccxml |
| MSPM0G3507 | mspm0g3507 | MSPM0G3507.ccxml |
| AM263 | am263 | AM263.ccxml |
| CC2755 | cc2755 | CC2755.ccxml |

---

## NPU vs non-NPU deployment

| Aspect | NPU devices (F28P55, AM13E2, MSPM0G5187) | Non-NPU |
|---|---|---|
| Model suffix | `_NPU` required | No suffix |
| Quantization | `quantization: 2` required | `quantization: 1` |
| Compilation preset | `compress_npu_layer_data` | `default_preset` |
| Inference speed | 10-25× faster | Baseline |
| CCS example | Available in Resource Explorer | Manual import |

---

## Troubleshooting

### `test_result == 0` (inference mismatch)

1. Verify `quantization` param matches training config
2. Check feature extraction parameters match between training and `user_input_config.h`
3. Recompile and redo create_ccs_project / build / flash

### Flash fails: "Cannot connect to device"

1. Check USB connection
2. Verify device power is on
3. Install USB drivers (`lsusb` on Linux to check device is seen)
4. Try a different USB port
5. Restart CCS

### Build error: "Undefined symbol mod_inference"

`mod.a` not linked. In CCS:
- Project Properties → Build → Linker → File Search Path
- Add: `${PROJECT_ROOT}/artifacts/mod.a`

### Device out of memory

Use compression preset in config:
```yaml
compilation:
  compile_preset_name: compress_npu_layer_data
```
Or select a smaller model and retrain.


## Contingency Plan:
If you are unable to use the above or if the the above steps fail for any reason:
Use the **ccs mcp servers** to import and build project. Flashing to device can be done by the user.
Again, note that this is a **contingency** plan for importing and building the project if the regular guide given above fails.