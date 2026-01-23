#!/usr/bin/env python3
"""
Test All ModelZoo Configs
Runs all example configs and validates models work correctly.
Only saves detailed logs for failed runs.

Usage:
    source ~/.pyenv/versions/py310_tinyml/bin/activate
    python test_all_configs.py [--timeout 2400] [--filter STRING] [--stop-on-error]
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
EXAMPLES_DIR = SCRIPT_DIR / "examples"
LOGS_DIR = SCRIPT_DIR / "test_logs"

# ModelMaker run script location (relative to parent of modelzoo)
MODELMAKER_DIR = SCRIPT_DIR.parent / "tinyml-modelmaker"
RUN_SCRIPT = MODELMAKER_DIR / "tinyml_modelmaker" / "run_tinyml_modelmaker.py"

# Create logs directory
LOGS_DIR.mkdir(exist_ok=True)


def get_all_configs():
    """Get all YAML config files from examples directory."""
    configs = sorted(EXAMPLES_DIR.glob(os.path.join("**", "*.yaml")))
    return configs


def run_config(config_path, timeout=600):
    """
    Run a single config and return results.

    Args:
        config_path: Path to config file
        timeout: Timeout in seconds

    Returns:
        dict with keys: success, duration, stdout, stderr, return_code
    """
    start_time = time.time()

    # Construct command - ModelMaker accepts config file path as argument
    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        str(config_path)
    ]

    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{MODELMAKER_DIR}:{env.get('PYTHONPATH', '')}"

    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=MODELMAKER_DIR,
            env=env
        )

        duration = time.time() - start_time

        # Check for errors in output even if return code is 0
        error_patterns = [
            'AssertionError',
            'assert ',
            'Assertion failed',
            'Traceback (most recent call last)',
            'Exception:',
            'Error:',
            'ERROR:',
            'FAILED',
            'Failed to',
            'RuntimeError',
            'ValueError',
            'KeyError',
            'TypeError',
            'AttributeError',
            'ImportError',
            'ModuleNotFoundError',
            'FileNotFoundError',
            'MemoryError',
            'IndexError',
            'NameError',
            'ZeroDivisionError',
            'Cannot',
            'could not',
            'No such file',
            'permission denied',
        ]

        has_error = any(pattern in result.stderr or pattern in result.stdout
                       for pattern in error_patterns)

        # Success only if return code is 0 AND no error patterns detected
        success = result.returncode == 0 and not has_error

        return {
            'success': success,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode,
            'timeout_exceeded': False,
            'error_detected': has_error
        }

    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        return {
            'success': False,
            'duration': duration,
            'stdout': e.stdout.decode() if e.stdout else "",
            'stderr': e.stderr.decode() if e.stderr else "",
            'return_code': -1,
            'timeout_exceeded': True,
            'error_detected': False
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            'success': False,
            'duration': duration,
            'stdout': "",
            'stderr': str(e),
            'return_code': -1,
            'timeout_exceeded': False,
            'error_detected': True
        }


def save_failure_log(config_path, result, timestamp):
    """Save detailed log for failed runs."""
    config_name = str(config_path).replace('/', '_').replace('.yaml', '')
    log_file = LOGS_DIR / f"{timestamp}_{config_name}_FAILED.log"

    with open(log_file, 'w') as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"FAILED CONFIG TEST\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Config:         {config_path}\n")
        f.write(f"Timestamp:      {timestamp}\n")
        f.write(f"Duration:       {result['duration']:.2f}s\n")
        f.write(f"Return Code:    {result['return_code']}\n")
        f.write(f"Timeout:        {result['timeout_exceeded']}\n")
        f.write(f"Error Detected: {result.get('error_detected', False)}\n")
        f.write(f"\n" + "=" * 80 + "\n")
        f.write(f"STDOUT:\n")
        f.write(f"=" * 80 + "\n")
        f.write(result['stdout'])
        f.write(f"\n\n" + "=" * 80 + "\n")
        f.write(f"STDERR:\n")
        f.write(f"=" * 80 + "\n")
        f.write(result['stderr'])
        f.write(f"\n")

    return log_file


def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def test_model_imports():
    """Test that all models can be imported correctly."""
    print("Testing model imports...")
    try:
        from tinyml_modelzoo.models import model_dict, get_model, list_models
        model_count = len(model_dict)
        print(f"  ✓ {model_count} models registered in model_dict")

        # Test instantiation of a few models
        test_models = ['CNN_TS_GEN_BASE_1K_NPU', 'REG_TS_GEN_BASE_1K', 'AE_CNN_TS_GEN_BASE_1K']
        for model_name in test_models:
            if model_name in model_dict:
                try:
                    model = get_model(model_name, 1, 2, 64)
                    print(f"  ✓ {model_name} instantiated successfully")
                except Exception as e:
                    print(f"  ✗ {model_name} failed: {e}")
                    return False

        # Test backward compatibility through tinyverse
        from tinyml_tinyverse.common.models import model_dict as tv_dict
        if len(tv_dict) == model_count:
            print(f"  ✓ TinyVerse re-exports {len(tv_dict)} models (backward compatible)")
        else:
            print(f"  ! Warning: TinyVerse has {len(tv_dict)} models vs modelzoo {model_count}")

        return True
    except Exception as e:
        print(f"  ✗ Model import test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test all ModelZoo configs')
    parser.add_argument('--timeout', type=int, default=2400,
                       help='Timeout per config in seconds (default: 2400)')
    parser.add_argument('--stop-on-error', action='store_true',
                       help='Stop testing if a config fails')
    parser.add_argument('--filter', type=str, default='',
                       help='Only test configs matching this substring')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training tests, only test imports')

    args = parser.parse_args()

    print("=" * 80)
    print(f"TinyML ModelZoo Test Suite")
    print("=" * 80)
    print()

    # First, test model imports
    if not test_model_imports():
        print("\n✗ Model import tests failed!")
        return 1
    print()

    if args.skip_training:
        print("Skipping training tests (--skip-training flag set)")
        return 0

    # Check if modelmaker exists
    if not RUN_SCRIPT.exists():
        print(f"Warning: ModelMaker not found at {RUN_SCRIPT}")
        print("Skipping training tests. Run with --skip-training to suppress this warning.")
        return 0

    # Get all configs
    all_configs = get_all_configs()

    # Filter if requested
    if args.filter:
        all_configs = [c for c in all_configs if args.filter in str(c)]

    if not all_configs:
        print("No configs found in examples directory!")
        return 1

    # Start timestamp
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print(f"Training Test Suite")
    print("=" * 80)
    print(f"Timeout:        {args.timeout}s")
    print(f"Total Configs:  {len(all_configs)}")
    print(f"Logs Dir:       {LOGS_DIR}")
    print(f"Timestamp:      {start_timestamp}")
    print("=" * 80)
    print()

    # Results tracking
    results = []
    passed = 0
    failed = 0
    total_duration = 0

    # Run each config
    for i, config_path in enumerate(all_configs, 1):
        display_name = str(config_path.relative_to(SCRIPT_DIR))
        print(f"[{i}/{len(all_configs)}] Testing {display_name:55}", end=" ", flush=True)

        result = run_config(config_path, args.timeout)
        total_duration += result['duration']

        if result['success']:
            passed += 1
            status = "✓ PASS"
            color = "\033[92m"
        else:
            failed += 1
            status = "✗ FAIL"
            color = "\033[91m"
            log_file = save_failure_log(config_path, result, start_timestamp)
            result['log_file'] = log_file

        duration_str = format_duration(result['duration'])
        reset = "\033[0m"
        error_note = ""
        if not result['success'] and result.get('error_detected') and result['return_code'] == 0:
            error_note = " [Error in output]"
        print(f"{color}{status}{reset} ({duration_str}){error_note}")

        results.append({
            'config': str(config_path),
            'success': result['success'],
            'duration': result['duration'],
            'timeout_exceeded': result['timeout_exceeded'],
            'error_detected': result.get('error_detected', False),
            'return_code': result['return_code'],
            'log_file': result.get('log_file')
        })

        if args.stop_on_error and not result['success']:
            print(f"\n✗ Stopping on first failure (--stop-on-error flag set).")
            break

    # Summary
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Configs:  {len(results)}")
    print(f"✓ Passed:       {passed} ({passed/len(results)*100:.1f}%)")
    print(f"✗ Failed:       {failed} ({failed/len(results)*100:.1f}%)")
    print(f"Total Time:     {format_duration(total_duration)}")
    print(f"Avg Time:       {format_duration(total_duration/len(results))}")
    print()

    if failed > 0:
        print("FAILED CONFIGS:")
        print("-" * 80)
        for r in results:
            if not r['success']:
                timeout_note = " (TIMEOUT)" if r['timeout_exceeded'] else ""
                error_note = " (Error in output)" if r.get('error_detected') and r.get('return_code') == 0 else ""
                print(f"  ✗ {r['config']:55} {format_duration(r['duration']):>10}{timeout_note}{error_note}")
                if r.get('log_file'):
                    print(f"    Log: {r['log_file']}")
        print()

    # Save summary report
    summary_file = LOGS_DIR / f"{start_timestamp}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TinyML ModelZoo Test Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp:      {start_timestamp}\n")
        f.write(f"Total Configs:  {len(results)}\n")
        f.write(f"Passed:         {passed}\n")
        f.write(f"Failed:         {failed}\n")
        f.write(f"Total Time:     {format_duration(total_duration)}\n")
        f.write(f"\n" + "=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            status = "PASS" if r['success'] else "FAIL"
            timeout_note = " (TIMEOUT)" if r['timeout_exceeded'] else ""
            error_note = " (Error in output)" if r.get('error_detected') and r.get('return_code') == 0 else ""
            f.write(f"{status:6} {r['config']:55} {format_duration(r['duration']):>10}{timeout_note}{error_note}\n")
            if not r['success'] and r.get('log_file'):
                f.write(f"       Log: {r['log_file']}\n")

    print(f"Summary saved to: {summary_file}")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
