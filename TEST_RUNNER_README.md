# TinyML ModelZoo Config Test Runner with HTML Report

Automated test runner for all 63 configs across 31 example folders in `tinyml-modelzoo/examples/`. Generates interactive HTML report with sortable table, charts, and detailed test results.

## Installation

The script requires **Jinja2** (for template rendering). Install with:
```bash
pip install jinja2
```

## Quick Start

### Test all configs (default 2400s timeout per config)
```bash
python test_runner_with_report.py
```

### Test specific configs (pattern matching, case-insensitive substring)
```bash
# Test motor_bearing_fault configs
python test_runner_with_report.py --config-pattern "*motor*"

# Test only MSPM0 variants
python test_runner_with_report.py --config-pattern "*MSPM0*"

# Test CC-series MCU configs
python test_runner_with_report.py --config-pattern "*CC*"
```

### Test with verbose output and logs
```bash
python test_runner_with_report.py --verbose
```

### Custom output file and short timeout
```bash
python test_runner_with_report.py --timeout 300 --output-file quick_test.html
```

### Skip saving logs (faster, no log links in report)
```bash
python test_runner_with_report.py --no-save-logs
```

### Test with each config's own device setting (no GPU override)
```bash
# By default the runner forces training.num_gpus=1 (via a throwaway config
# copy) so quantization/hessian-search paths run on GPU instead of CPU.
# Use --no-gpu-override to instead run each config exactly as authored.
python test_runner_with_report.py --no-gpu-override
```

## CLI Options

```
usage: test_runner_with_report.py [-h] [--timeout TIMEOUT]
                                  [--config-pattern CONFIG_PATTERN]
                                  [--output-file OUTPUT_FILE]
                                  [--no-save-logs] [--verbose]
                                  [--workers WORKERS]
                                  [--examples-dir EXAMPLES_DIR]
                                  [--fresh-datasets]
                                  [--num-gpus NUM_GPUS]
                                  [--no-gpu-override]

options:
  -h, --help            show this help message and exit
  --timeout TIMEOUT     Timeout per config in seconds (default: 2400)
  --config-pattern CONFIG_PATTERN
                        Only test configs matching this substring
                        (e.g., "*motor*")
  --output-file OUTPUT_FILE
                        Output HTML filename (default: tinyml_test_report.html)
  --no-save-logs        Skip saving detailed logs (default: save all logs)
  --verbose             Print progress for each config
  --workers WORKERS     Number of parallel workers (default: 1, sequential)
  --examples-dir EXAMPLES_DIR
                        Examples directory (default: ../tinyml-modelzoo/examples)
  --fresh-datasets      Delete each config's cached data/projects/<dataset_name>/dataset
                        directory before running it, forcing a fresh download+extract
                        instead of reusing a previous run's cache
  --num-gpus NUM_GPUS   Force training.num_gpus to this value in every config before
                        running it (default: 1). The original example config files on
                        disk are never modified -- a throwaway override copy is used
                        instead (written next to the original and deleted afterward,
                        pass/fail/timeout).
  --no-gpu-override     Run each config with its own num_gpus value as-is,
                        instead of forcing --num-gpus
```

## Output

### HTML Report
- **Location**: `test_logs/{output_file}` (or `test_logs/tinyml_test_report.html` by default)
- **Features**:
  - **Summary Stats**: Total configs, pass/fail counts, pass rate %, total duration
  - **Pie Chart**: Pass/fail/timeout distribution (doughnut chart)
  - **Bar Chart**: Pass count by folder (grouped results)
  - **Sortable Table**: Folder | Config Name | Status | Duration | Error Message | Log Link
  - **Status Badges**: Color-coded (green=PASS, red=FAIL, yellow=TIMEOUT)
  - **Interactive**: DataTables.js sorting/filtering, Chart.js visualization

### Log Files (with `--save-logs`)
- **Location**: `test_logs/{timestamp}_{config_name}_{status}.log`
- **Content**: Config path, status, duration, return code, full stdout/stderr
- **Pattern**: `{YYYYMMDD}_{HHMMSS}_{relative_path}_{PASS|FAIL|TIMEOUT}.log`

### Summary in Console
```
================================================================================
TEST SUMMARY
================================================================================
Total:    63
Passed:   0 (0.0%)
Failed:   63
Timeout:  0
Duration: 7.4s
================================================================================
```

## Examples

### Example 1: Quick validation of motor fault configs
```bash
python test_runner_with_report.py \
  --timeout 120 \
  --config-pattern "*motor*" \
  --output-file motor_validation.html \
  --verbose
```

### Example 2: Full test run with logs for troubleshooting
```bash
python test_runner_with_report.py \
  --timeout 3600 \
  --output-file full_validation.html
```

### Example 3: Test only MSPM0 variants (faster, no logs)
```bash
python test_runner_with_report.py \
  --timeout 600 \
  --config-pattern "*MSPM0*" \
  --no-save-logs \
  --output-file mspm0_report.html \
  --verbose
```

## Report Structure

The HTML report includes:

1. **Header** with timestamp and test parameters
2. **Statistics Grid** showing:
   - Total Configs
   - Passed (with percentage)
   - Failed
   - Timeout
   - Total Duration

3. **Charts Section**:
   - Doughnut chart (pass/fail/timeout)
   - Bar chart (results grouped by folder)

4. **Results Table**:
   - Sorted by folder name, then config name
   - Columns: Folder, Config Name, Status, Duration, Error Message, Log Link
   - Color-coded status badges
   - Sortable and filterable via DataTables.js
   - Pagination (25 rows per page)

5. **Responsive Design**: Works on desktop and mobile

## Error Pattern Detection

The runner detects 25 common error patterns:
- Python exceptions: AssertionError, Traceback, RuntimeError, ValueError, etc.
- Error messages: Error:, ERROR:, FAILED, Failed to, Cannot, etc.
- Timeout detection via subprocess.TimeoutExpired

Success = return code 0 AND no error patterns detected

## Performance

- **All 63 configs with 30s timeout**: ~7 seconds (mostly I/O)
- **Subset with verbose output**: ~0.1s per config execution
- **HTML generation**: <1 second
- **Report size**: ~14-30KB depending on config count

## Troubleshooting

### No configs found
- Ensure `tinyml-modelzoo/examples/` directory exists
- Check `--examples-dir` path if using custom directory
- Verify pattern matching (e.g., try without `--config-pattern`)

### All configs fail with quick timeout
- Increase timeout: `--timeout 600` or higher
- Check ModelMaker dependencies (PYTHONPATH, Python 3.10)
- Ensure `tinyml_modelmaker/run_tinyml_modelmaker.py` exists

### HTML report not opening
- Ensure CDN links are accessible (Chart.js, DataTables.js, jQuery)
- Check browser console for JavaScript errors
- Fallback: Table works without JS; charts require CDN access

## Integration with CI/CD

Return codes:
- `0` = all configs passed
- `1` = at least one config failed

Example GitHub Actions integration:
```yaml
- name: Run ModelZoo Tests
  run: |
    cd tinyml-modelmaker
    python test_runner_with_report.py \
      --timeout 1800 \
      --output-file ci_report.html
  
- name: Upload Test Report
  if: always()
  uses: actions/upload-artifact@v2
  with:
    name: test-report
    path: tinyml-modelmaker/test_logs/ci_report.html
```

## Created by Plan

Built using ConfigTestRunner class + HTMLReportGenerator class (Jinja2 + Chart.js).
Reuses error patterns and subprocess execution logic from `tinyml-modelmaker/test_all_configs.py`.

---

**Version**: 1.0  
**Date**: July 20, 2026  
**Python**: 3.10+  
**Dependencies**: jinja2
