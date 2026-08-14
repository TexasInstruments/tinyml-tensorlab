#!/usr/bin/env python3
"""
Test All ModelZoo Configs with HTML Report Generation
Runs all configs from tinyml-modelzoo/examples and generates an interactive HTML report
with sortable table, charts, and detailed test results.

Usage:
    source ~/.pyenv/versions/py310_tinyml/bin/activate
    python test_runner_with_report.py [--timeout 2400] [--config-pattern PATTERN] \\
        [--output-file NAME.html] [--save-logs] [--verbose]

Example:
    python test_runner_with_report.py --timeout 300 --config-pattern "*motor*" --save-logs
"""

import os
import sys
import subprocess
import time
import argparse
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from jinja2 import Environment, Template
except ImportError:
    print("ERROR: jinja2 not found. Install with: pip install jinja2")
    sys.exit(1)


# Configuration
_SCRIPT_DIR = Path(__file__).parent
MODELZOO_DIR = _SCRIPT_DIR.parent / "tinyml-modelzoo"
EXAMPLES_DIR = MODELZOO_DIR / "examples"
LOGS_DIR = _SCRIPT_DIR / "test_logs"
RUN_SCRIPT = _SCRIPT_DIR / "tinyml_modelmaker" / "run_tinyml_modelmaker.py"

# Create logs directory
LOGS_DIR.mkdir(exist_ok=True)

# Error patterns from ModelMaker test suite
ERROR_PATTERNS = [
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
    'please check if the given model_name',  # silent exit: unrecognized model_name
]

# Lines containing these substrings are known-harmless noise on Python 3.12+ and
# must not trigger error detection even if they match an ERROR_PATTERN above.
NOISE_SUPPRESSIONS = [
    '/loky-',                           # Python 3.12+ multiprocessing resource_tracker
    'resource_tracker',                 # same
    'cache[rtype].remove',              # same
    'onnxscript.version_converter',     # non-fatal ONNX opset downgrade attempt
    '/project/onnx/version_converter',  # non-fatal ONNX C++ assertion in downgrade
    'axes_input_to_attribute',          # same
    'BaseConverter.h',                  # same
    'FutureWarning',                    # deprecation notices
    'isinstance(treespec, LeafSpec)',   # torch internal FutureWarning
]


def _has_real_error(text: str) -> bool:
    """Return True only if a real error pattern appears on a non-noise line."""
    for line in text.splitlines():
        if any(n in line for n in NOISE_SUPPRESSIONS):
            continue
        if any(p in line for p in ERROR_PATTERNS):
            return True
    return False


class ConfigTestRunner:
    """Discovers and runs all configs, capturing results."""

    def __init__(self, examples_dir: Path, timeout: int = 2400, save_logs: bool = False, verbose: bool = False, workers: int = 1):
        self.examples_dir = Path(examples_dir)
        self.timeout = timeout
        self.save_logs = save_logs
        self.verbose = verbose
        self.workers = workers

    def discover_configs(self, pattern: str = None) -> List[Dict]:
        """
        Discover all YAML configs, optionally filtered by pattern.
        Returns list of dicts: {config_absolute, config_relative, folder}
        """
        import yaml as _yaml
        configs = []
        for config_path in sorted(self.examples_dir.glob("**/*.yaml")):
            # Skip hidden files
            if config_path.name.startswith('.'):
                continue

            # Skip non-training YAMLs (model-spec / parameter files that have no
            # top-level 'common:' key and would crash with KeyError: 'common').
            try:
                with open(config_path) as _f:
                    _data = _yaml.safe_load(_f)
                if not isinstance(_data, dict) or 'common' not in _data:
                    continue
            except Exception:
                continue

            config_relative = config_path.relative_to(self.examples_dir)
            config_relative_str = str(config_relative)

            # Apply pattern filter if provided (substring match on relative path)
            if pattern:
                # Remove wildcards from pattern for substring matching
                clean_pattern = pattern.replace('*', '')
                if clean_pattern not in config_relative_str:
                    continue

            folder = config_relative.parent.name

            configs.append({
                'config_absolute': config_path,
                'config_relative': config_relative_str,
                'folder': folder,
                'config_name': config_path.name,
            })

        return configs

    def run_single_config(self, config_info: Dict) -> Dict:
        """
        Execute one config and return result dict.
        Returns: {config_relative, folder, config_name, status, duration, error_message,
                  return_code, log_file, timestamp}
        """
        config_path = config_info['config_absolute']

        # Build command
        cmd = [
            sys.executable,
            str(RUN_SCRIPT),
            str(config_path)
        ]

        # Note: epochs override currently not supported via CLI.
        # Would require modifying YAML file or post-processing config loading.
        # User can manually edit config files or use --epochs to modify configs before running.

        # Set environment
        env = os.environ.copy()
        env['PYTHONPATH'] = f".:{env.get('PYTHONPATH', '')}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Start timer right before subprocess (not at function entry to avoid queue delay)
            start_time = time.time()

            # Run config
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=_SCRIPT_DIR,
                env=env
            )

            duration = time.time() - start_time

            # Check for error patterns, ignoring known-harmless noise lines
            combined_output = result.stderr + result.stdout
            has_error = _has_real_error(combined_output)

            # Determine status
            if result.returncode == 0 and not has_error:
                status = 'PASS'
                error_message = ''
            else:
                status = 'FAIL'
                error_message = self._extract_error_message(combined_output)

            log_file = None
            if self.save_logs:
                log_file_path = self._save_log(config_info, result, status, timestamp, duration)
                # Store as filename only (logs are in same directory as report)
                log_file = log_file_path.name

            return {
                'config_relative': config_info['config_relative'],
                'config_name': config_info['config_name'],
                'folder': config_info['folder'],
                'status': status,
                'duration': duration,
                'error_message': error_message,
                'return_code': result.returncode,
                'log_file': log_file,
                'timestamp': timestamp,
            }

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_message = f"Timeout after {self.timeout}s"

            log_file = None
            if self.save_logs:
                # Create timeout log
                log_file_path = self._save_timeout_log(config_info, duration, timestamp)
                # Store as filename only
                log_file = log_file_path.name

            return {
                'config_relative': config_info['config_relative'],
                'config_name': config_info['config_name'],
                'folder': config_info['folder'],
                'status': 'TIMEOUT',
                'duration': duration,
                'error_message': error_message,
                'return_code': -1,
                'log_file': log_file,
                'timestamp': timestamp,
            }

        except Exception as e:
            duration = time.time() - start_time
            error_message = str(e)[:200]

            return {
                'config_relative': config_info['config_relative'],
                'config_name': config_info['config_name'],
                'folder': config_info['folder'],
                'status': 'FAIL',
                'duration': duration,
                'error_message': error_message,
                'return_code': -1,
                'log_file': None,
                'timestamp': timestamp,
            }

    def run_all(self, pattern: str = None) -> Tuple[List[Dict], Dict]:
        """
        Run all discovered configs, return (results, summary).
        """
        configs = self.discover_configs(pattern)

        if not configs:
            print("ERROR: No configs found!")
            return [], {}

        print("=" * 80)
        print("TinyML ModelZoo Config Test Runner with HTML Report")
        print("=" * 80)
        print(f"Timeout per config:  {self.timeout}s")
        print(f"Total configs:       {len(configs)}")
        print(f"Workers:             {self.workers}")
        print(f"Save logs:           {self.save_logs}")
        print(f"Logs directory:      {LOGS_DIR}")
        print("=" * 80)
        print()

        results = []
        start_total = time.time()

        if self.workers == 1:
            # Sequential execution
            for i, config_info in enumerate(configs, 1):
                if self.verbose:
                    print(f"[{i}/{len(configs)}] Running {config_info['config_relative']:60}", end=" ", flush=True)

                result = self.run_single_config(config_info)
                results.append(result)

                if self.verbose:
                    status_icon = "✓" if result['status'] == 'PASS' else "✗" if result['status'] == 'FAIL' else "⏱"
                    print(f"{status_icon} {result['duration']:.1f}s")
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(self.run_single_config, config_info): (i, config_info)
                          for i, config_info in enumerate(configs, 1)}

                for future in as_completed(futures):
                    i, config_info = futures[future]
                    result = future.result()
                    results.append(result)

                    if self.verbose:
                        status_icon = "✓" if result['status'] == 'PASS' else "✗" if result['status'] == 'FAIL' else "⏱"
                        print(f"[{i}/{len(configs)}] {config_info['config_relative']:60} {status_icon} {result['duration']:.1f}s")

        total_duration = time.time() - start_total

        # Compute summary
        passed = sum(1 for r in results if r['status'] == 'PASS')
        failed = sum(1 for r in results if r['status'] == 'FAIL')
        timeout = sum(1 for r in results if r['status'] == 'TIMEOUT')

        summary = {
            'total_configs': len(results),
            'passed': passed,
            'failed': failed,
            'timeout': timeout,
            'total_duration': total_duration,
            'avg_duration': total_duration / len(results) if results else 0,
            'pass_rate': (passed / len(results) * 100) if results else 0,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        # Print summary
        print()
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total:    {summary['total_configs']}")
        print(f"Passed:   {summary['passed']} ({summary['pass_rate']:.1f}%)")
        print(f"Failed:   {summary['failed']}")
        print(f"Timeout:  {summary['timeout']}")
        print(f"Duration: {self._format_duration(summary['total_duration'])}")
        print("=" * 80)
        print()

        return results, summary

    def _extract_error_message(self, output: str, max_chars: int = 200) -> str:
        """Extract first error message from output."""
        for line in output.split('\n'):
            if any(pattern in line for pattern in ERROR_PATTERNS):
                return line.strip()[:max_chars]
        return "Unknown error"[:max_chars]

    def _save_log(self, config_info: Dict, result, status: str, timestamp: str, duration: float) -> Path:
        """Save detailed log for this config run."""
        config_name = config_info['config_relative'].replace('/', '_').replace('.yaml', '')
        log_file = LOGS_DIR / f"{timestamp}_{config_name}_{status}.log"

        with open(log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"CONFIG TEST LOG\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Config:      {config_info['config_relative']}\n")
            f.write(f"Status:      {status}\n")
            f.write(f"Duration:    {duration:.2f}s\n")
            f.write(f"Return Code: {result.returncode}\n")
            f.write(f"\n" + "=" * 80 + "\n")
            f.write("STDOUT:\n")
            f.write("=" * 80 + "\n")
            f.write(result.stdout)
            f.write(f"\n\n" + "=" * 80 + "\n")
            f.write("STDERR:\n")
            f.write("=" * 80 + "\n")
            f.write(result.stderr)
            f.write("\n")

        return log_file

    def _save_timeout_log(self, config_info: Dict, duration: float, timestamp: str) -> Path:
        """Save timeout log."""
        config_name = config_info['config_relative'].replace('/', '_').replace('.yaml', '')
        log_file = LOGS_DIR / f"{timestamp}_{config_name}_TIMEOUT.log"

        with open(log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CONFIG TIMEOUT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Config:      {config_info['config_relative']}\n")
            f.write(f"Status:      TIMEOUT\n")
            f.write(f"Duration:    {duration:.2f}s (timeout: {self.timeout}s)\n")

        return log_file

    @staticmethod
    def _format_duration(seconds: float) -> str:
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


class HTMLReportGenerator:
    """Generates interactive HTML report from test results."""

    def __init__(self):
        self.env = Environment()

    def generate(self, results: List[Dict], summary: Dict, output_file: str = 'tinyml_test_report.html') -> str:
        """
        Generate HTML report and save to file.
        Returns path to generated report.
        """
        # Group results by folder
        results_by_folder = self._group_by_folder(results)

        # Compute folder statistics
        folder_stats = self._compute_folder_stats(results)

        # Render HTML
        html_content = self._render_template(results_by_folder, folder_stats, summary)

        # Save to file
        output_path = LOGS_DIR / output_file
        with open(output_path, 'w') as f:
            f.write(html_content)

        return str(output_path)

    def _group_by_folder(self, results: List[Dict]) -> Dict:
        """Group results by folder name."""
        grouped = defaultdict(list)
        for result in results:
            grouped[result['folder']].append(result)
        return dict(sorted(grouped.items()))

    def _compute_folder_stats(self, results: List[Dict]) -> List[Dict]:
        """Compute pass/fail stats per folder."""
        folder_stats = {}
        for result in results:
            folder = result['folder']
            if folder not in folder_stats:
                folder_stats[folder] = {'passed': 0, 'failed': 0, 'timeout': 0}

            if result['status'] == 'PASS':
                folder_stats[folder]['passed'] += 1
            elif result['status'] == 'TIMEOUT':
                folder_stats[folder]['timeout'] += 1
            else:
                folder_stats[folder]['failed'] += 1

        # Convert to list with totals
        stats_list = []
        for folder in sorted(folder_stats.keys()):
            stats = folder_stats[folder]
            stats_list.append({
                'folder_name': folder,
                'passed': stats['passed'],
                'failed': stats['failed'],
                'timeout': stats['timeout'],
                'total': stats['passed'] + stats['failed'] + stats['timeout'],
            })

        return stats_list

    def _render_template(self, results_by_folder: Dict, folder_stats: List[Dict], summary: Dict) -> str:
        """Render Jinja2 template with data."""
        template_str = self._get_html_template()
        template = Template(template_str)

        # Prepare data for template
        template_data = {
            'summary': summary,
            'results_by_folder': results_by_folder,
            'folder_stats': folder_stats,
            'format_duration': self._format_duration,
        }

        return template.render(**template_data)

    @staticmethod
    def _get_html_template() -> str:
        """Return Jinja2 HTML template."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TinyML ModelZoo Test Report</title>
    <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.x/css/jquery.dataTables.min.css">
    <script src="https://cdn.datatables.net/1.13.x/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        header p {
            opacity: 0.9;
            font-size: 0.95em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .stat-card h3 {
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }
        .stat-card .value {
            font-size: 2.2em;
            font-weight: bold;
            color: #333;
        }
        .stat-card.passed {
            border-left-color: #28a745;
        }
        .stat-card.passed .value {
            color: #28a745;
        }
        .stat-card.failed {
            border-left-color: #dc3545;
        }
        .stat-card.failed .value {
            color: #dc3545;
        }
        .stat-card.timeout {
            border-left-color: #ffc107;
        }
        .stat-card.timeout .value {
            color: #ff9800;
        }
        .section {
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section h2 {
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
            color: #333;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .chart-container {
            position: relative;
            height: 350px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }
        thead {
            background: #f8f9fa;
        }
        th {
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #555;
            border-bottom: 2px solid #ddd;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #eee;
        }
        tr:hover {
            background: #f9f9f9;
        }
        tbody tr:nth-child(even) {
            background: #fafafa;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge.PASS {
            background: #d4edda;
            color: #155724;
        }
        .badge.FAIL {
            background: #f8d7da;
            color: #721c24;
        }
        .badge.TIMEOUT {
            background: #fff3cd;
            color: #856404;
        }
        .error-msg {
            color: #666;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
            max-width: 400px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        a {
            color: #667eea;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .folder-row {
            background: #f0f0f0;
            font-weight: 600;
            color: #555;
        }
        .footer {
            text-align: center;
            color: #999;
            font-size: 0.9em;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        .dataTables_wrapper {
            padding: 0 !important;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧪 TinyML ModelZoo Test Report</h1>
            <p>Automated testing and validation of all example configurations</p>
            <p style="margin-top: 10px; font-size: 0.9em;">Generated: {{ summary.timestamp }}</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Configs</h3>
                <div class="value">{{ summary.total_configs }}</div>
            </div>
            <div class="stat-card passed">
                <h3>Passed</h3>
                <div class="value">{{ summary.passed }} <span style="font-size: 0.6em;">({{ "%.1f"|format(summary.pass_rate) }}%)</span></div>
            </div>
            <div class="stat-card failed">
                <h3>Failed</h3>
                <div class="value">{{ summary.failed }}</div>
            </div>
            <div class="stat-card timeout">
                <h3>Timeout</h3>
                <div class="value">{{ summary.timeout }}</div>
            </div>
            <div class="stat-card">
                <h3>Total Duration</h3>
                <div class="value" style="font-size: 1.4em;">{{ format_duration(summary.total_duration) }}</div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Results Overview</h2>
            <div class="charts-grid">
                <div class="chart-container">
                    <canvas id="pieChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="barChart"></canvas>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>📋 Detailed Results by Folder</h2>
            <table id="resultsTable" class="display">
                <thead>
                    <tr>
                        <th>Folder</th>
                        <th>Config Name</th>
                        <th>Status</th>
                        <th>Duration (s)</th>
                        <th>Error Message</th>
                        <th>Log</th>
                    </tr>
                </thead>
                <tbody>
                    {% for folder_name, configs in results_by_folder.items() %}
                        {% for config in configs %}
                        <tr>
                            <td><strong>{{ folder_name }}</strong></td>
                            <td>{{ config.config_name }}</td>
                            <td><span class="badge {{ config.status }}">{{ config.status }}</span></td>
                            <td style="text-align: right;">{{ "%.2f"|format(config.duration) }}</td>
                            <td class="error-msg" title="{{ config.error_message }}">{{ config.error_message }}</td>
                            <td>
                                {% if config.log_file %}
                                    <a href="{{ config.log_file }}" target="_blank">View</a>
                                {% else %}
                                    —
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Generated by TinyML Test Runner with Report Generator</p>
        </div>
    </div>

    <script>
        // Pie chart
        new Chart(document.getElementById('pieChart'), {
            type: 'doughnut',
            data: {
                labels: ['Passed', 'Failed', 'Timeout'],
                datasets: [{
                    data: [{{ summary.passed }}, {{ summary.failed }}, {{ summary.timeout }}],
                    backgroundColor: ['#28a745', '#dc3545', '#ffc107'],
                    borderColor: ['#20c997', '#c82333', '#e0a800'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { padding: 20, font: { size: 12 } }
                    }
                }
            }
        });

        // Bar chart by folder
        new Chart(document.getElementById('barChart'), {
            type: 'bar',
            data: {
                labels: [{% for fs in folder_stats %}'{{ fs.folder_name }}'{{ "," if not loop.last }}{% endfor %}],
                datasets: [
                    {
                        label: 'Passed',
                        data: [{% for fs in folder_stats %}{{ fs.passed }}{{ "," if not loop.last }}{% endfor %}],
                        backgroundColor: '#28a745',
                        borderColor: '#20c997',
                        borderWidth: 1
                    },
                    {
                        label: 'Failed',
                        data: [{% for fs in folder_stats %}{{ fs.failed }}{{ "," if not loop.last }}{% endfor %}],
                        backgroundColor: '#dc3545',
                        borderColor: '#c82333',
                        borderWidth: 1
                    },
                    {
                        label: 'Timeout',
                        data: [{% for fs in folder_stats %}{{ fs.timeout }}{{ "," if not loop.last }}{% endfor %}],
                        backgroundColor: '#ffc107',
                        borderColor: '#e0a800',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'x',
                scales: {
                    x: {
                        stacked: false,
                    },
                    y: {
                        stacked: false,
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { padding: 15, font: { size: 12 } }
                    }
                }
            }
        });

        // DataTables initialization
        $(document).ready(function() {
            $('#resultsTable').DataTable({
                paging: true,
                pageLength: 25,
                order: [[0, 'asc'], [1, 'asc']],
                columnDefs: [
                    { orderable: true, targets: '_all' }
                ]
            });
        });
    </script>
</body>
</html>
'''

    @staticmethod
    def _format_duration(seconds: float) -> str:
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


def main():
    parser = argparse.ArgumentParser(
        description='Test all ModelZoo configs and generate HTML report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Test all configs
  python test_runner_with_report.py

  # Test only motor_bearing_fault configs with 1 epoch (flow testing)
  python test_runner_with_report.py --timeout 300 --config-pattern "*motor*" --epochs 1

  # Test with verbose output and custom output file
  python test_runner_with_report.py --verbose --output-file custom_report.html --epochs 2
        '''
    )
    parser.add_argument('--timeout', type=int, default=2400,
                       help='Timeout per config in seconds (default: 2400)')
    parser.add_argument('--config-pattern', type=str, default=None,
                       help='Only test configs matching this substring (e.g., "*motor*")')
    parser.add_argument('--output-file', type=str, default='tinyml_test_report.html',
                       help='Output HTML filename (default: tinyml_test_report.html)')
    parser.add_argument('--no-save-logs', action='store_true',
                       help='Skip saving detailed logs (default: save all logs)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print progress for each config')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1, sequential)')
    parser.add_argument('--examples-dir', type=str, default=None,
                       help=f'Examples directory (default: {EXAMPLES_DIR})')

    args = parser.parse_args()

    # Validate examples directory
    examples_dir = Path(args.examples_dir) if args.examples_dir else EXAMPLES_DIR
    if not examples_dir.exists():
        print(f"ERROR: Examples directory not found: {examples_dir}")
        return 1

    # Run tests
    runner = ConfigTestRunner(
        examples_dir=examples_dir,
        timeout=args.timeout,
        save_logs=not args.no_save_logs,
        verbose=args.verbose,
        workers=args.workers
    )

    results, summary = runner.run_all(pattern=args.config_pattern)

    if not results:
        return 1

    # Generate HTML report
    print("Generating HTML report...")
    generator = HTMLReportGenerator()
    report_path = generator.generate(results, summary, args.output_file)
    print(f"✓ Report saved to: {report_path}")
    print()

    return 0 if summary['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
