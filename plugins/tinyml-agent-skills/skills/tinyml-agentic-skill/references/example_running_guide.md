# Running Examples Guide

## Prerequisites

**CRITICAL: Confirm TinyML tensorlab path with user before running any commands.**

Store the path as `TINYML_BASE_PATH` for reference in all commands below.

---

## Overview

After config.yaml is created and saved to `{TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/config.yaml`, the example runs through three phases:

1. **Training**: Model trains on dataset
2. **Compilation**: Model compiled to target device
3. **Testing** (optional): Model tested on test split or hardware

---

## Complete Run Command

### Syntax

```bash
bash {TINYML_BASE_PATH}/tinyml-modelzoo/run_tinyml_modelzoo.sh {TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/config.yaml
```

### Example

If `TINYML_BASE_PATH="/home/user/tinyml-tensorlab"` and `task_name="motor_fault"`:

```bash
bash /home/user/tinyml-tensorlab/tinyml-modelzoo/run_tinyml_modelzoo.sh /home/user/tinyml-tensorlab/tinyml-modelzoo/examples/motor_fault/config.yaml
```

### Breaking Down the Command

| Component | Purpose | Value |
|-----------|---------|-------|
| `bash` | Execute shell script | Required |
| `{TINYML_BASE_PATH}/tinyml-modelzoo/run_tinyml_modelzoo.sh` | Entry point script | Full path to run script |
| Config path | YAML config file location | Full path to config.yaml |

---

## Pre-Run Checks

**Before executing the run command, verify:**

1. ✓ `{TINYML_BASE_PATH}` directory exists and contains `tinyml-modelzoo/`
2. ✓ `{TINYML_BASE_PATH}/tinyml-modelzoo/run_tinyml_modelzoo.sh` is executable:
   ```bash
   ls -l {TINYML_BASE_PATH}/tinyml-modelzoo/run_tinyml_modelzoo.sh
   chmod +x {TINYML_BASE_PATH}/tinyml-modelzoo/run_tinyml_modelzoo.sh
   ```
3. ✓ Config file exists at correct path:
   ```bash
   ls -la {TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/config.yaml
   ```
4. ✓ Dataset is accessible (if `dataset.enable: true`):
   ```bash
   ls -la {input_data_path}
   ```
5. ✓ Output directory is writable:
   ```bash
   touch {TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/.write_test && rm {TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/.write_test
   ```

---

## Running Examples

### Standard Full Run (Training + Compilation)

```bash
bash /home/user/tinyml-tensorlab/tinyml-modelzoo/run_tinyml_modelzoo.sh \
  /home/user/tinyml-tensorlab/tinyml-modelzoo/examples/motor_fault/config.yaml
```

**Expected output:**
```
Starting tinyML ModelMaker...
[INFO] Loading config from /home/user/tinyml-tensorlab/tinyml-modelzoo/examples/motor_fault/config.yaml
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
bash /home/user/tinyml-tensorlab/tinyml-modelzoo/run_tinyml_modelzoo.sh \
  /home/user/tinyml-tensorlab/tinyml-modelzoo/examples/byom_deployment/config.yaml
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

After successful run, check output at:

```
{TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/runs/
└── {run_name}/
    ├── training/
    │   ├── base/
    │   │   ├── run.log                    # Training metrics & losses
    │   │   ├── model.pt                   # Trained PyTorch model
    │   │   └── checkpoint_*.pt
    │   └── quantized/ (if applicable)
    ├── compilation/
    │   ├── run.log                        # Compilation metrics
    │   ├── artifacts/
    │   │   ├── model.onnx
    │   │   ├── model_compiled.so
    │   │   ├── golden_vectors.bin
    │   │   └── ...
    │   └── memory_report.txt
    └── testing/
        ├── results.json
        └── metrics.txt
```

---

## Monitoring & Logging

### Real-time Monitoring

While run is in progress:

```bash
# Watch training logs
tail -f {TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/runs/*/training/base/run.log

# Watch compilation logs
tail -f {TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/runs/*/compilation/run.log

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
  {TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/runs/*/training/base/run.log
```

### Compilation Summary

```bash
# Extract compilation metrics
grep -E "Memory|Size|Latency|Model" \
  {TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/runs/*/compilation/run.log
```

### Full Results Summary

Parse from JSON results (if available):

```bash
# List all results files
find {TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/runs/ -name "results.json"

# Parse with jq (if installed)
jq . {TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/runs/*/testing/results.json
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
# Verify path is correct
ls -la {TINYML_BASE_PATH}/tinyml-modelzoo/run_tinyml_modelzoo.sh

# Make executable
chmod +x {TINYML_BASE_PATH}/tinyml-modelzoo/run_tinyml_modelzoo.sh
```

### Config File Not Found

**Problem:**
```
[ERROR] Config file not found: /path/to/config.yaml
```

**Solution:**
```bash
# Verify config exists
ls -la {TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/config.yaml

# Create directory if missing
mkdir -p {TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}
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
# Run command is idempotent — just re-execute
bash {TINYML_BASE_PATH}/tinyml-modelzoo/run_tinyml_modelzoo.sh \
  {TINYML_BASE_PATH}/tinyml-modelzoo/examples/{task_name}/config.yaml
```

System automatically detects checkpoints and resumes training.

---

## Next Steps After Run

After successful completion:

1. **Review training metrics** — check accuracy/loss in `training/base/run.log`
2. **Verify compiled artifacts** — confirm compilation succeeded and model size is acceptable
3. **Test on hardware** — see `device_deployment_guide.md` for device flashing
4. **Iterate** — modify config and rerun to improve performance
