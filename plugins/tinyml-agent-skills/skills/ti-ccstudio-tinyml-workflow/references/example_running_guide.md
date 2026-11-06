# Running Examples Guide

## Prerequisites

Values come from Session Setup in `SKILL.md` — do not ask the user for paths again:

| Variable | Meaning |
|----------|---------|
| `PROJECT_PATH` | Imported project; holds `config.yaml` and all outputs |
| `TRAIN_DIR` / `TRAIN_CMD` | Repo install: `$TINYML_MODELZOO_PATH` and `./run_tinyml_modelzoo.sh`. Wheel install: `$HOME/.tinyml-agent-skills` and `$VENV_PYTHON -m tinyml_modelmaker.run_tinyml_modelmaker` |
| `EXAMPLES_DIR` | Bundled example configs (`references/examples/`), read-only. Copy from here into `$PROJECT_PATH`; never run or write there |

---

## Overview

After config.yaml is created and saved to `{PROJECT_PATH}/config.yaml`, the example runs through three phases:

1. **Training**: Model trains on dataset
2. **Compilation**: Model compiled to target device
3. **Testing** (optional): Model tested on test split or hardware

---

## Complete Run Command

### Syntax

`run_training.py` starts every training run: it first calls `new_run_dir.py`, which points `train_output_path` / `compile_output_path` at a new `run/<yyyymmdd-hhmmss>/<model_name>/` folder in the project, so re-runs never overwrite earlier ones.

```bash
$VENV_PYTHON "$SCRIPTS_DIR/run_training.py" "$PROJECT_PATH" "$TRAIN_DIR" -- \
  $TRAIN_CMD "$PROJECT_PATH/config.yaml"
```

### Example

Repo install, `PROJECT_PATH=/home/user/workspace/motor_fault_f28p55`:

```bash
/home/user/.tinyml-agent-skills/venv/bin/python <SCRIPTS_DIR>/run_training.py /home/user/workspace/motor_fault_f28p55 /home/user/tinyml-modelzoo -- \
  ./run_tinyml_modelzoo.sh /home/user/workspace/motor_fault_f28p55/config.yaml
```

Wheel install (same config, run through modelmaker directly):

```bash
~/.tinyml-agent-skills/venv/bin/python <SCRIPTS_DIR>/run_training.py /home/user/workspace/motor_fault_f28p55 ~/.tinyml-agent-skills -- \
  ~/.tinyml-agent-skills/venv/bin/python3 -m tinyml_modelmaker.run_tinyml_modelmaker /home/user/workspace/motor_fault_f28p55/config.yaml
```

Compiler env vars (e.g. `C2000_CGT_ROOT`) must be set in the same shell call — see Gate G1 and Step 13 in `SKILL.md`. Use `C2000_CGT_ROOT` for C2000 devices and `ARM_LLVM_CGT_PATH` for ARM devices.

### Breaking Down the Command

| Component | Purpose | Value |
|-----------|---------|-------|
| `cd "$TRAIN_DIR"` | Run from the modelzoo repo, or `~/.tinyml-agent-skills` for wheels | Required |
| `$TRAIN_CMD` | `./run_tinyml_modelzoo.sh` (repo) or `$VENV_PYTHON -m tinyml_modelmaker.run_tinyml_modelmaker` (wheel) | Required |
| Config path | YAML config in the project | Absolute path to `$PROJECT_PATH/config.yaml` |

---

## Pre-Run Checks

**Before executing the run command, verify:**

1. ✓ `$TRAIN_DIR` exists
2. ✓ The entry point exists: repo install → `ls -l "$TRAIN_DIR/run_tinyml_modelzoo.sh"` (and `chmod +x` if needed); wheel install → `$VENV_PYTHON -c "import tinyml_modelmaker"`
3. ✓ Config file exists at correct path:
   ```bash
   ls -la {PROJECT_PATH}/config.yaml
   ```
4. ✓ Dataset is accessible (if `dataset.enable: true`):
   ```bash
   ls -la {input_data_path}
   ```
5. ✓ Output directory is writable:
   ```bash
   touch {PROJECT_PATH}/.write_test && rm {PROJECT_PATH}/.write_test
   ```

---

## Running Examples

### Standard Full Run (Training + Compilation)

```bash
$VENV_PYTHON "$SCRIPTS_DIR/run_training.py" "$PROJECT_PATH" "$TRAIN_DIR" -- \
  $TRAIN_CMD "$PROJECT_PATH/config.yaml"
```

**Expected output:**
```
[INFO] Loading config from /home/user/workspace/motor_fault_f28p55/config.yaml
[INFO] Phase 1: Data loading...
[INFO] Phase 2: Training model...
[INFO] Training complete. Model saved to: /path/to/run/model.pt
[INFO] Phase 3: Compilation...
[INFO] Compilation complete. Artifacts saved to: /path/to/run/compilation
[SUCCESS] Run completed successfully!
```

### BYOM Run (Compilation Only, Skip Training)

If config has `training.enable: false` and `dataset.enable: false`:

```bash
$VENV_PYTHON "$SCRIPTS_DIR/run_training.py" "$PROJECT_PATH" "$TRAIN_DIR" -- \
  $TRAIN_CMD "$PROJECT_PATH/config.yaml"
```

**Expected output:**
```
[INFO] Skipping data loading (dataset.enable: false)
[INFO] Skipping training (training.enable: false)
[INFO] Phase: Compilation...
[INFO] Loading model from: /path/to/model.onnx
[INFO] Compilation complete.
[SUCCESS] Run completed successfully!
```

### With GPU (Multi-GPU Training)

Ensure `training.num_gpus` is set correctly in config. The run script auto-detects GPUs:

```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi
```

---

## Output Directory Structure

After a successful run, outputs are under `$PROJECT_PATH/run/<yyyymmdd-hhmmss>/<model_name>/` (`$RUN_DIR`), the same `run/<date-time>/<model_name>/` layout the modelzoo uses. The skill writes `training.train_output_path` and `compilation.compile_output_path` there (always nested in their sections, never at the config root) before each run; `config.yaml` itself stays in `$PROJECT_PATH`. The subfolder names inside `training/` can vary by version — locate files with `find "$RUN_DIR" -name run.log`. Typical layout:

```
{PROJECT_PATH}/
├── config.yaml
└── run/
    └── {yyyymmdd-hhmmss}/
        └── {model_name}/                  # = $RUN_DIR
            ├── training/
            │   ├── training_base/
            │   │   ├── run.log            # Training metrics & losses
            │   │   ├── model.onnx / checkpoint.pth
            │   │   └── golden_vectors/
            │   └── training_quantization/ (if quantization is on; has golden_vectors/)
            ├── compilation/
            │   ├── run.log                # Compilation metrics
            │   └── artifacts/             # mod.a, tvmgen_default.h, ...
            └── testing/
```

---

## Monitoring & Logging

### Real-time Monitoring

While run is in progress:

```bash
# Watch training logs
tail -f $RUN_DIR/training/training_base/run.log

# Watch compilation logs
tail -f $RUN_DIR/compilation/run.log

# Monitor system resources
watch -n 1 'ps aux | grep python; nvidia-smi'
```

### Key Metrics from Logs

**Training log (`run.log`):**
- Loss values per epoch
- Validation accuracy
- Learning rate schedule
- Training time

**Compilation log (`compilation/run.log`):**
- Model size post-compilation
- Memory footprint (Flash, RAM)
- Inference latency estimates
- Quantization details (if applicable)

---

## Extracting Results

### Training Summary

```bash
# Extract final metrics
grep -E "Final|Accuracy|Loss|Epoch" \
  $RUN_DIR/training/training_base/run.log
```

### Compilation Summary

```bash
# Extract compilation metrics
grep -E "Memory|Size|Latency|Model" \
  $RUN_DIR/compilation/run.log
```

### Full Results Summary

Parse from JSON results (if available):

```bash
# List all results files
find {PROJECT_PATH}/ -name "results.json"

# Parse with jq (if installed)
jq . $RUN_DIR/testing/results.json
```

---

## Troubleshooting

### Run Fails to Start

**Problem:** Script not found error
```
bash: /path/to/run_tinyml_modelzoo.sh: No such file or directory
```

**Solution:**
```bash
# Repo install: verify the script and make it executable
ls -la "$TINYML_MODELZOO_PATH/run_tinyml_modelzoo.sh"
chmod +x "$TINYML_MODELZOO_PATH/run_tinyml_modelzoo.sh"

# Wheel install: there is no run_tinyml_modelzoo.sh — run the modelmaker module
$VENV_PYTHON -c "import tinyml_modelmaker; print(tinyml_modelmaker.__file__)"
```

### Config File Not Found

**Problem:**
```
[ERROR] Config file not found: /path/to/config.yaml
```

**Solution:**
```bash
# Verify config exists
ls -la {PROJECT_PATH}/config.yaml

# If missing, go back to Step 12 (assemble config) or Step 3.5 (copy a bundled example)
```

### Dataset Loading Error

**Problem:**
```
[ERROR] Failed to load dataset from {input_data_path}
```

**Solution:**
```bash
# Verify data path exists and is readable
ls -la {input_data_path}

# Check format matches BYOD specification
# See config_creation_guide.md for format details

# If remote URL, verify connectivity
curl -I {input_data_path}
```

### GPU/Memory Issues

**Problem:**
```
CUDA out of memory. Tried to allocate X.XX GiB
```

**Solution:**
```bash
# Reduce batch size in config
# batch_size: 32 → 16

# Or disable GPU
# num_gpus: 0 (forces CPU)

# Check available memory
nvidia-smi
```

### Compilation Failures

**Problem:**
```
[ERROR] Compilation failed: Model too large for device
```

**Solution:**
```bash
# Try compression preset
# compile_preset_name: compress_npu_layer_data

# Or force software NPU
# compile_preset_name: forced_soft_npu_preset

# Consult device memory specs in config guide
```

---

## Performance Tuning

### Faster Training

```yaml
training:
  batch_size: 64          # Increase batch size
  training_epochs: 50     # Reduce epochs
  num_gpus: 2             # Use more GPUs
  learning_rate: 0.01     # Increase initial LR
```

### Smaller Model

```yaml
training:
  model_name: mobilenet_v2  # Use lighter model

compilation:
  compile_preset_name: compress_npu_layer_data  # Enable compression
```

### Better Accuracy

```yaml
training:
  batch_size: 16          # Smaller batches
  training_epochs: 200    # More epochs
  learning_rate: 0.0001   # Smaller learning rate
```

---

## Run Time Expectations

Typical run times (varies by dataset size & hardware):

| Phase | Small Dataset | Medium Dataset | Large Dataset |
|-------|---------------|----------------|---------------|
| Training | 5-10 min | 30-60 min | 2-6 hours |
| Compilation | 2-5 min | 5-15 min | 10-30 min |
| **Total** | **10-15 min** | **40-75 min** | **2-7 hours** |

---

## Resuming Failed Runs

If run is interrupted, restart from last checkpoint:

```bash
# Every run starts in a NEW run/<timestamp>/<model_name>/ folder (new_run_dir.py), so a re-run never resumes or overwrites an earlier one
$VENV_PYTHON "$SCRIPTS_DIR/run_training.py" "$PROJECT_PATH" "$TRAIN_DIR" -- \
  $TRAIN_CMD "$PROJECT_PATH/config.yaml"
```

A re-run trains again from scratch in its own new folder; the interrupted run's folder is kept.

---

## Next Steps After Run

After successful completion:

1. **Review training metrics** — check accuracy/loss in `$RUN_DIR/training/training_base/run.log`
2. **Verify compiled artifacts** — confirm compilation succeeded and model size is acceptable
3. **Test on hardware** — see `device_deployment_guide.md` for device flashing
4. **Iterate** — modify config and rerun to improve performance
