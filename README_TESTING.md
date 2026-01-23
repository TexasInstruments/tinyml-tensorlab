# ModelMaker Config Testing Guide

## Overview

This testing infrastructure automates testing of all 32 ModelMaker config files in the `examples/` directory. It runs each config, tracks execution time, and **only saves detailed logs for failed runs** to conserve disk space.

## Quick Start

```bash
# Ensure environment is activated
source ~/.pyenv/versions/py310_tinyml/bin/activate

# Navigate to modelmaker directory
cd ./tinyml-modelmaker

# Test all configs (default 20-min timeout, continues on errors)
./run_tests.sh

# Or test specific configs
./run_tests.sh --filter hello_world
```

## Usage Options

### Basic Testing

```bash
# Test all configs
./run_tests.sh

# Test with custom timeout (in seconds)
./run_tests.sh --timeout 900

# Test specific configs by name pattern
./run_tests.sh --filter motor
./run_tests.sh --filter arc_fault
./run_tests.sh --filter hello_world
```

### Advanced Options

```bash
# Stop testing on first failure (default is to continue)
./run_tests.sh --stop-on-error

# Combine options
./run_tests.sh --filter forecast --timeout 1800 --stop-on-error

# Use Python script directly for more control
python test_all_configs.py --timeout 1800 --filter mnist
```

## What Gets Tested

The test script:
1. ✓ Discovers all YAML config files in `examples/` directory
2. ✓ Runs each config through `run_tinyml_modelmaker.py`
3. ✓ Tracks execution time per config
4. ✓ Captures stdout/stderr for analysis
5. ✓ **Saves detailed logs ONLY for failed runs**
6. ✓ Generates summary report for all runs
7. ✓ Provides color-coded console output

## Output Structure

### Real-time Console Output

```
[1/32] Testing examples/hello_world/config.yaml                  ✓ PASS (120.4s)
[2/32] Testing examples/motor_bearing_fault/config.yaml          ✓ PASS (145.2s)
[3/32] Testing examples/mnist/config_image_classification.yaml   ✗ FAIL (89.1s)
```

### Log Files

All logs are saved in the `test_logs/` directory:

#### Summary File (Always Created)
```
test_logs/20251230_143022_summary.txt
```
- Contains results for ALL configs
- Shows pass/fail status and duration
- Lists paths to failure logs (if any)

#### Failure Logs (Only for Failed Runs)
```
test_logs/20251230_143022_examples_mnist_config_image_classification_FAILED.log
```
- Complete stdout/stderr output
- Config path and details
- Return code
- Timeout status
- Duration

**Example Log Structure**:
```
test_logs/
├── 20251230_143022_summary.txt
├── 20251230_143022_examples_mnist_config_FAILED.log
└── 20251230_150000_summary.txt
```

## Config Categories Tested

The script automatically tests all 32 configs across these categories:

| Category | Count | Examples |
|----------|-------|----------|
| Hello World | 4 | config.yaml, config_MSPM0.yaml, config_CC1352.yaml, config_CC2755.yaml |
| Motor Fault | 3 | config.yaml, config_MSPM0.yaml, config_anomaly_detection.yaml |
| Fan Blade | 3 | config.yaml, config_anomaly_detection.yaml, ondevice_training.yaml |
| Arc Fault | 5 | AC/DC variants with dsi/dsk |
| ECG | 2 | config.yaml, config_anomaly_detection.yaml |
| NILM | 2 | nilm_appliance, PLAID_nilm |
| PIR Detection | 2 | config.yaml, config_CC1352.yaml |
| Forecasting | 2 | HVAC, PMSM rotor |
| Regression | 2 | washing_machine, torque |
| Vision | 1 | MNIST image classification |
| Other | 6 | gas_sensor, grid_stability, electrical_fault, blower_imbalance, branched_model |

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--timeout` | int | 1200 | Timeout per config in seconds (20 minutes) |
| `--stop-on-error` | flag | false | Stop testing if a config fails (default: continue) |
| `--filter` | string | "" | Only test configs matching this substring |

## Filtering Examples

```bash
# Test by example type
./run_tests.sh --filter hello_world     # All hello_world variants
./run_tests.sh --filter motor           # All motor-related configs
./run_tests.sh --filter arc             # All arc fault configs
./run_tests.sh --filter anomaly         # All anomaly detection configs

# Test by device
./run_tests.sh --filter MSPM0           # All MSPM0 configs
./run_tests.sh --filter CC1352          # All CC1352 configs
./run_tests.sh --filter CC2755          # All CC2755 configs

# Test by task type
./run_tests.sh --filter forecast        # Forecasting configs
./run_tests.sh --filter regression      # Regression configs
./run_tests.sh --filter classification  # Classification configs
```

## Performance Estimates

| Scenario | Typical Duration |
|----------|------------------|
| Single config | 1-3 minutes |
| All 32 configs (sequential) | 30-100 minutes |
| Quick smoke test (--filter hello_world) | 5-10 minutes |

## Common Use Cases

### 1. Quick Validation After Code Changes
```bash
# Test a representative subset
./run_tests.sh --filter hello_world --timeout 300
```

### 2. Full Regression Test
```bash
# Test everything (default behavior continues on errors)
./run_tests.sh --timeout 1800
```

### 3. Test Specific Feature
```bash
# Test only forecasting configs
./run_tests.sh --filter forecast

# Test only anomaly detection configs
./run_tests.sh --filter anomaly
```

### 4. Debug a Specific Failure
```bash
# Test just one config
./run_tests.sh --filter "hello_world/config.yaml"

# Check the failure log
cat test_logs/*_FAILED.log
```

## Exit Codes

- `0`: All tested configs passed ✓
- `1`: One or more configs failed ✗

## Troubleshooting

### Environment Not Activated
```bash
# Activate the environment first
source ~/.pyenv/versions/py310_tinyml/bin/activate

# Verify it's active
which python
python --version  # Should show Python 3.10.14
```

### Scripts Not Executable
```bash
chmod +x run_tests.sh test_all_configs.py
```

### All Tests Timeout
```bash
# Increase timeout to 15 minutes
./run_tests.sh --timeout 900

# Or test a single config manually
python tinyml_modelmaker/run_tinyml_modelmaker.py examples/hello_world/config.yaml
```

### Test Logs Not Found
```bash
# Logs are only created for failures
# Check if any tests actually failed
cat test_logs/*_summary.txt
```

### PYTHONPATH Issues
The test script automatically sets `PYTHONPATH=.:$PYTHONPATH`, but if you encounter import errors:
```bash
# Set it manually
export PYTHONPATH=.:$PYTHONPATH

# Then run tests
./run_tests.sh
```

## Integration with Existing Workflow

The test script complements the existing workflow:

```bash
# Normal training (existing method)
./run_tinyml_modelmaker.sh examples/hello_world/config.yaml

# Automated testing (new method)
./run_tests.sh --filter hello_world

# Manual config run (existing method)
python tinyml_modelmaker/run_tinyml_modelmaker.py examples/motor_bearing_fault/config.yaml

# Automated batch testing (new method)
./run_tests.sh --filter motor
```

## CI/CD Integration

Example CI pipeline:

```bash
#!/bin/bash
# Quick smoke test for CI (test hello_world variants)
cd ./tinyml-modelmaker
source ~/.pyenv/versions/py310_tinyml/bin/activate

./run_tests.sh --filter hello_world --timeout 300 --stop-on-error
exit_code=$?

# Upload logs if failed
if [ $exit_code -ne 0 ]; then
    tar -czf test_logs.tar.gz test_logs/
    # Upload to CI artifact storage
fi

exit $exit_code
```

## Advantages Over Manual Testing

1. **Automated**: No manual intervention needed
2. **Comprehensive**: Tests all 32 configs systematically
3. **Timed**: Tracks performance per config
4. **Selective Logging**: Only saves logs for failures (saves disk space)
5. **Resumable**: Can filter and re-test specific configs
6. **CI-Ready**: Easy to integrate into automated pipelines
7. **Detailed**: Full stdout/stderr captured for failures
8. **Consistent**: Same environment and parameters for all runs

## File Locations

```
./tinyml-modelmaker/
├── test_all_configs.py       # Main test script
├── run_tests.sh              # Convenience wrapper
├── README_TESTING.md         # This file
├── examples/                 # Config files to test
│   ├── hello_world/
│   ├── motor_bearing_fault/
│   ├── ...
│   └── torque_measurement_regression/
└── test_logs/                # Generated logs (only failures + summary)
    ├── YYYYMMDD_HHMMSS_summary.txt
    └── YYYYMMDD_HHMMSS_examples_*_FAILED.log
```

## Next Steps

1. **Run First Test**:
   ```bash
   cd ./tinyml-modelmaker
   ./run_tests.sh --filter hello_world
   ```

2. **Check Results**:
   ```bash
   cat test_logs/*_summary.txt
   ```

3. **Review Failures** (if any):
   ```bash
   cat test_logs/*_FAILED.log
   ```

4. **Run Full Suite**:
   ```bash
   ./run_tests.sh --continue-on-error
   ```

## Summary

This testing infrastructure provides:
- ✓ Automated testing of all 32 ModelMaker configs
- ✓ Timing information for each run
- ✓ Selective logging (only failures saved)
- ✓ Easy filtering and configuration
- ✓ CI/CD ready
- ✓ Comprehensive documentation

All files are ready to use in `./tinyml-modelmaker/`
