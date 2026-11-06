# Device Deployment Reference

Supporting reference for `SKILL.md` **Step 14** (14A–14E). The deployment steps 14A–14E and gates (G2–G5) live in `SKILL.md`; the CCStudio `.cproject` PASS-check script and the build/flash details live in `deployment_procedure.md`. This file only holds the lookup tables and troubleshooting that support them.

Prerequisites (all set earlier in the workflow): `PROJECT_PATH` (Step 3), compiled artifacts in `$RUN_DIR/compilation/artifacts/` and golden vectors in `$RUN_DIR/training/[training_quantization|training_base]/golden_vectors/` (Step 13/13A), where `$RUN_DIR` is `$PROJECT_PATH/run/<yyyymmdd-hhmmss>/<model_name>`.

Files the project needs after Step 14C. These replace the template's placeholder files in place and keep the placeholder's file name (e.g. the trained `mod.a` is written as `model.a` if the project links `model.a`):

| File | Source (inside `$RUN_DIR`) |
|---|---|
| `mod.a` | `compilation/artifacts/mod.a` |
| `tvmgen_default.h` | `compilation/artifacts/tvmgen_default.h` |
| `test_vector.c` | `training/[training_quantization\|training_base]/golden_vectors/test_vector.c` |
| `user_input_config.h` | `training/[training_quantization\|training_base]/golden_vectors/user_input_config.h` |

**Set the active target configuration before flashing.** Use the project's LaunchPad/board `.ccxml` that matches the hardware (e.g. `TMS320F28P550SJ9_LaunchPad.ccxml`) via your IDE's mechanism. In CCStudio's UI: right-click the `.ccxml` in Project Explorer → **Set as Active Target Configuration**.

---

## Verify model on device

After flashing, with a debug session open in the IDE (the agent drives it with the IDE's debug tools, or guides the user):

1. Start a debug session if one is not already running
2. Set a breakpoint on the line **after** the inference call in `application_main.c`
3. Resume the program
4. Watch the variable `test_result`
5. Check value:
   - `test_result == 1` → inference passed (output matches golden vector) ✓
   - `test_result == 0` → inference failed — see troubleshooting below

---

## Device type → target configuration mapping

| Device | device_type | Target ccxml |
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
| Model suffix | NPU models usually end in `_NPU` (rule of thumb; there may be exceptions) | Usually no suffix |
| Quantization | `quantization: 2` required | `quantization: 1` |
| Compilation preset | `compress_npu_layer_data` | `default_preset` |
| Inference speed | 10-25× faster | Baseline |
| SDK example project | Available from the SDK / IDE example browser | Manual import |

---

## Troubleshooting

### `test_result == 0` (inference mismatch)

1. Verify `quantization` param matches training config
2. Check feature extraction parameters match between training and `user_input_config.h`
3. Re-run training/compilation if the config changed, then redo Step 14C (copy artifacts), 14D (build) and 14E (flash)

### Flash fails: "Cannot connect to device"

1. Check USB connection
2. Verify device power is on
3. Install USB drivers (`lsusb` on Linux to check device is seen)
4. Try a different USB port
5. Restart the IDE

### Build error: "Undefined symbol mod_inference"

The library is not linked. First confirm Step 14C replaced the project's placeholder library under the name the linker expects (e.g. `artifacts/mod.a` on C2000, `model/model.a` on MSPM0; `grep` the project's `.cproject` for the linked name) and that Step 14B did **not** leave `compilation/` or `training/` in the build. Then in the project's linker settings (CCStudio: Project Properties → Build → Linker → File Search Path):
- Add: `${PROJECT_ROOT}/artifacts/mod.a` (use the library path and name your project actually links)

### Device out of memory

Use compression preset in config:
```yaml
compilation:
  compile_preset_name: compress_npu_layer_data
```
Or select a smaller model and retrain.
