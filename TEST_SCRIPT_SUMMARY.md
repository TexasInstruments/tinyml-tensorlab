# ModelMaker Test Script Summary

## Created Files

### 1. `test_all_configs.py` - Main Test Script
**Location**: `./tinyml-modelmaker/test_all_configs.py`

**Features**:
- Tests all 32 configs in the examples directory
- Tracks execution time for each config
- Logs only failed runs (saves disk space)
- Generates summary report for all runs
- Supports filtering, timeouts
- Color-coded console output
- Detailed error logs with stdout/stderr

**Key Functions**:
- `get_all_configs()`: Discovers all YAML files in examples subdirectories
- `run_config()`: Executes a config with timeout handling
- `save_failure_log()`: Saves detailed logs only for failures
- `format_duration()`: Human-readable time formatting

**Key Differences from MLBackend Version**:
- Searches for configs in `examples/**/*.yaml` (subdirectories)
- Calls `run_tinyml_modelmaker.py` instead of `run.py`
- Sets `PYTHONPATH=.:$PYTHONPATH` environment variable
- Config paths include subdirectory (e.g., `examples/hello_world/config.yaml`)

### 2. `run_tests.sh` - Convenience Wrapper
**Location**: `./tinyml-modelmaker/run_tests.sh`

**Features**:
- Automatically activates py310_tinyml environment (commented out, assumes pre-activated)
- Validates environment before running
- Color-coded status messages
- Passes all arguments to test_all_configs.py
- Provides clear success/failure exit codes

### 3. `README_TESTING.md` - Complete Documentation
**Location**: `./tinyml-modelmaker/README_TESTING.md`

**Contents**:
- Quick start guide
- All usage options and examples
- Output format explanation
- Troubleshooting tips
- CI/CD integration examples
- Config category breakdown

## Usage Examples

### Basic Testing

```bash
# Test all 32 configs (default 20-min timeout, continues on errors)
./run_tests.sh

# Test only specific configs
./run_tests.sh --filter motor
./run_tests.sh --filter generic_timeseries_classification
./run_tests.sh --filter forecast
```

### Advanced Testing

```bash
# Stop on first failure (default is to continue)
./run_tests.sh --stop-on-error

# Custom timeout (30 minutes)
./run_tests.sh --timeout 1800

# Test specific category
./run_tests.sh --filter anomaly
./run_tests.sh --filter MSPM0

# Combine options
./run_tests.sh --filter arc --timeout 1800 --stop-on-error
```

## Output Structure

### Console Output (Real-time)
```
[1/32] Testing examples/hello_world/config.yaml                  ✓ PASS (120.4s)
[2/32] Testing examples/motor_bearing_fault/config.yaml          ✓ PASS (145.2s)
[3/32] Testing examples/mnist/config_image.yaml                  ✗ FAIL (89.1s)
```

### Log Files (test_logs/ directory)

**Summary File** (always created):
```
test_logs/20251230_143022_summary.txt
```
- Contains results for ALL configs
- Shows pass/fail status, duration
- Lists log files for failed configs

**Failure Logs** (only for failed runs):
```
test_logs/20251230_143022_examples_mnist_config_image_FAILED.log
```
- Complete stdout/stderr
- Config path and details
- Return code
- Timeout status

**Example Log Structure**:
```
test_logs/
├── 20251230_143022_summary.txt
├── 20251230_143022_examples_mnist_config_FAILED.log
└── 20251230_150000_summary.txt
```

## What Gets Tested

For each config, the script:
1. ✓ Loads YAML file from examples subdirectory
2. ✓ Calls `run_tinyml_modelmaker.py` with config path
3. ✓ Sets PYTHONPATH environment variable
4. ✓ Runs full ModelMaker pipeline (train + compile)
5. ✓ Captures all output
6. ✓ Records timing
7. ✓ Saves logs only if failed

## Config Categories (32 Total)

| Category | Count | Examples |
|----------|-------|----------|
| Hello World | 4 | F28P55, MSPM0, CC1352, CC2755 variants |
| Motor Fault | 3 | Classification, anomaly detection, MSPM0 |
| Fan Blade | 3 | Classification, anomaly, on-device training |
| Arc Fault | 5 | AC MSPM0, DC dsi/dsk variants |
| ECG | 2 | Classification, anomaly detection |
| NILM | 2 | Appliance usage, PLAID |
| PIR Detection | 2 | Default, CC1352 |
| Forecasting | 2 | HVAC, PMSM rotor |
| Regression | 2 | Washing machine, torque |
| Vision | 1 | MNIST image classification |
| Other | 6 | Gas, grid, electrical, blower, branched |

## Exit Codes

- `0`: All tests passed ✓
- `1`: One or more tests failed ✗

## Performance Characteristics

### Timing Estimates
- **All 32 configs**: ~30-100 minutes (varies by hardware)
- **Single config**: ~1-3 minutes
- **Hello world variants** (4 configs): ~5-10 minutes

### Resource Usage
- **CPU**: High during training
- **GPU**: If available and configured in config
- **Memory**: ~2-8 GB depending on model
- **Disk**: Minimal (only failure logs saved)

## Filtering Examples

```bash
# Test by example type
./run_tests.sh --filter generic_timeseries_classification     # All 4 generic_timeseries_classification variants
./run_tests.sh --filter motor           # All motor-related (3 configs)
./run_tests.sh --filter arc             # All arc fault (5 configs)

# Test by device
./run_tests.sh --filter MSPM0           # MSPM0 configs
./run_tests.sh --filter CC1352          # CC1352 configs

# Test by task
./run_tests.sh --filter forecast        # Forecasting (2 configs)
./run_tests.sh --filter anomaly         # Anomaly detection configs
./run_tests.sh --filter regression      # Regression (2 configs)
```

## Common Use Cases

### 1. Quick Validation After Changes
```bash
# Test representative generic_timeseries_classification configs (fastest)
./run_tests.sh --filter generic_timeseries_classification --timeout 300
```

### 2. Full Regression Test
```bash
# Test everything, save all results (default continues on errors)
./run_tests.sh --timeout 1800
```

### 3. Specific Feature Testing
```bash
# Test only forecasting
./run_tests.sh --filter forecast

# Test only vision/image
./run_tests.sh --filter MNIST
```

### 4. CI/CD Pipeline
```bash
#!/bin/bash
# Quick smoke test for CI
cd ./tinyml-modelmaker
source ~/.pyenv/versions/py310_tinyml/bin/activate

./run_tests.sh --filter generic_timeseries_classification --timeout 300
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
4. **Selective Logging**: Only saves logs for failures (saves space)
5. **Resumable**: Can filter and re-test specific configs
6. **CI-Ready**: Easy to integrate into automated pipelines
7. **Detailed**: Full stdout/stderr captured for failures
8. **Consistent**: Same environment for all runs

## Troubleshooting

### Script Not Found
```bash
cd ./tinyml-modelmaker
chmod +x run_tests.sh test_all_configs.py
```

### Environment Issues
```bash
# Verify environment
source ~/.pyenv/versions/py310_tinyml/bin/activate
python -c "import torch; print('OK')"
```

### All Tests Timeout
```bash
# Increase timeout to 15 minutes
./run_tests.sh --timeout 900

# Or test individual config manually
python tinyml_modelmaker/run_tinyml_modelmaker.py examples/generic_timeseries_classification/config.yaml
```

### PYTHONPATH Errors
```bash
# Set it manually if needed
export PYTHONPATH=.:$PYTHONPATH
./run_tests.sh
```

## Integration with Existing Workflow

The test script works alongside your existing tools:

```bash
# Normal training (existing method)
./run_tinyml_modelmaker.sh examples/generic_timeseries_classification/config.yaml

# Automated testing (new method)
./run_tests.sh --filter generic_timeseries_classification

# Manual run (existing method)
python tinyml_modelmaker/run_tinyml_modelmaker.py examples/motor_bearing_fault/config.yaml

# Automated batch testing (new method)
./run_tests.sh --filter motor
```

## Key Differences from MLBackend Test Script

| Aspect | MLBackend | ModelMaker |
|--------|-----------|------------|
| Config location | `configs/*.yaml` (flat) | `examples/**/*.yaml` (nested) |
| Script called | `run.py` | `run_tinyml_modelmaker.py` |
| Arguments | `--run_type <type> --run_args <config>` | `<config_path>` |
| Environment | Custom env vars | `PYTHONPATH=.:$PYTHONPATH` |
| Config count | 39 configs | 32 configs |
| Directory structure | Flat | Hierarchical (subdirectories) |

## Next Steps

1. **Run First Test**:
   ```bash
   cd ./tinyml-modelmaker
   ./run_tests.sh --filter generic_timeseries_classification
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
   ./run_tests.sh
   ```

## Summary

The testing infrastructure provides:
- ✓ Automated testing of all 32 ModelMaker configs
- ✓ Timing information for each run
- ✓ Selective logging (only failures saved)
- ✓ Easy filtering and configuration
- ✓ CI/CD ready
- ✓ Comprehensive documentation

All files are ready to use in:
`./tinyml-modelmaker/`
