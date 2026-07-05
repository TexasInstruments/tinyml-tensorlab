#!/usr/bin/env python3
"""
Dispatcher for setup skill functions.

Usage:
    python3 runner.py <function_name> '<json_args>'
    python3 runner.py set_update_mode '{"mode": "auto"}'
"""

import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from update_setup import set_update_mode, save_config


FUNCTION_MAP = {
    "set_update_mode": set_update_mode,
    "save_config": save_config,
}


def _check_installation(kwargs):
    """Built-in function: verify a tinyml-tensorlab installation path."""
    base = kwargs.get("tinyml_base_path", "")
    if not base:
        return {"success": False, "errors": ["tinyml_base_path is required"]}

    base = os.path.expanduser(base)
    checks = {
        "tinyml-modelzoo":         os.path.join(base, "tinyml-modelzoo"),
        "tinyml-modelzoo/examples": os.path.join(base, "tinyml-modelzoo", "examples"),
        "run_script":              os.path.join(base, "tinyml-modelzoo", "run_tinyml_modelzoo.sh"),
        "tinyml-modelmaker":       os.path.join(base, "tinyml-modelmaker"),
    }
    results = {}
    all_ok = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        results[name] = {"path": path, "exists": exists}
        if not exists:
            all_ok = False

    return {
        "success": all_ok,
        "tinyml_base_path": base,
        "checks": results,
        "errors": [] if all_ok else [
            f"Missing: {name} at {info['path']}"
            for name, info in results.items() if not info["exists"]
        ],
        "hint": (
            "Installation looks good." if all_ok
            else "Verify the path is the root of the tinyml-tensorlab checkout, "
                 "containing tinyml-modelzoo/ and tinyml-modelmaker/ subdirectories."
        ),
    }

def _output_result(result, save_yaml_path, save_result_path):
    """Print result as JSON and optionally write yaml/result to files."""
    output = json.dumps(result, indent=2, default=str)
    print(output)

    if save_yaml_path and isinstance(result, dict) and result.get("yaml"):
        _write_file(save_yaml_path, result["yaml"])
        print(f"[runner] YAML saved to: {save_yaml_path}", file=sys.stderr)

    if save_result_path:
        _write_file(save_result_path, output)
        print(f"[runner] Result saved to: {save_result_path}", file=sys.stderr)

def _write_file(path, content):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

def main():

    save_yaml_path = None
    save_result_path = None

    raw_args = sys.argv[1:]

    if not raw_args:
        print(json.dumps({"error": "Usage: runner.py <function_name> [<json_args>]"}))
        sys.exit(1)

    func_name = raw_args[0]

    if func_name == "--list":
        print(json.dumps({"available_functions": list(FUNCTION_MAP.keys())}))
        return
    
    if func_name == "check_installation":
        args_json = raw_args[1] if len(raw_args) > 1 else "{}"
        try:
            kwargs = json.loads(args_json)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON args: {e}"}))
            sys.exit(1)
        result = _check_installation(kwargs)
        _output_result(result, save_yaml_path, save_result_path)
        if not result.get("success"):
            sys.exit(1)
        return

    if func_name not in FUNCTION_MAP:
        print(json.dumps({
            "error": f"Unknown function: '{func_name}'",
            "available": list(FUNCTION_MAP.keys()),
        }))
        sys.exit(1)

    args_json = raw_args[1] if len(raw_args) > 1 else "{}"

    try:
        kwargs = json.loads(args_json)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON args: {e}"}))
        sys.exit(1)

    try:
        fn = FUNCTION_MAP[func_name]
        result = fn(**kwargs)
        print(json.dumps(result, indent=2, default=str))

        if isinstance(result, dict) and result.get("success") is False:
            sys.exit(1)

    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "type": type(e).__name__,
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
