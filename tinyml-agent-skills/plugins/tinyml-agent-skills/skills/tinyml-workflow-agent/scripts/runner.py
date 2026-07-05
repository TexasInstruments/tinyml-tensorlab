#!/usr/bin/env python3
"""
Dispatcher for tinyml-workflow-agent skill functions.

Usage:
    python3 runner.py <function_name> '<json_args>' [--save-yaml <file>] [--save-result <file>]
    python3 runner.py --list
    python3 runner.py check_installation '{"tinyml_base_path": "/home/user/tinyml-tensorlab"}'

Flags (must come after json_args):
    --save-yaml <file>    Extract the 'yaml' field from the result and write it to <file>.
                          Use this to persist section YAMLs to disk instead of keeping them
                          in memory. Pass WORK_DIR/<section>.yaml as the target file.
    --save-result <file>  Write the full JSON result to <file>.

Examples:
    python3 runner.py generate_common_section_yaml \
        '{"task_type": "motor_fault", "target_device": "F28P55"}' \
        --save-yaml /tmp/tinyml_work/common.yaml

    python3 runner.py generate_complete_config_file \
        '{"task_name": "demo", "common_yaml_file": "/tmp/tinyml_work/common.yaml", ...}' \
        --save-result /tmp/tinyml_work/config_result.json
"""

import sys
import json
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from update_manager import check_updates, get_update_status, do_update

FUNCTION_MAP = {
    # common_section_tools
    "validate_common_section":                ("common_section_tools",    "validate_common_section"),
    "generate_common_section_yaml":           ("common_section_tools",    "generate_common_section_yaml"),
    "list_supported_values":                  ("common_section_tools",    "list_supported_values"),
    # dataset_section_tools
    "validate_dataset_section":               ("dataset_section_tools",   "validate_dataset_section"),
    "generate_dataset_section_yaml":          ("dataset_section_tools",   "generate_dataset_section_yaml"),
    # "analyze_dataset_for_model_guidance":     ("dataset_section_tools",   "analyze_dataset_for_model_guidance"),
    # dataset_format_tools
    "validate_dataset_format":                ("dataset_format_tools",    "validate_dataset_format"),
    # "reorganize_dataset_stage_1":                     ("dataset_format_tools",    "reorganize_dataset_stage_1"),
    # "reorganize_dataset_stage_2":                     ("dataset_format_tools",    "reorganize_dataset_stage_2"),
    #dataset analysis
    "analyse_dataset":                        ("dataset_analysis",      "analyse_dataset"),
    # feature_extraction
    "get_data_proc_feat_ext_recommendations": ("feature_extraction",      "get_data_proc_feat_ext_recommendations"),
    "get_transform_context":                  ("feature_extraction",      "get_transform_context"),
    "validate_feat_ext_data_shape":           ("feature_extraction",      "validate_feat_ext_data_shape"),
    "generate_feat_ext_section_yaml":         ("feature_extraction",      "generate_feat_ext_section_yaml"),
    # model_selection_tools
    "select_model_for_task":                  ("model_selection_tools",   "select_model_for_task"),
    "list_available_models":                  ("model_selection_tools",   "list_available_models"),
    # training_section_tools
    "get_training_recommendations":           ("training_section_tools",  "get_training_recommendations"),
    "validate_training_section":              ("training_section_tools",  "validate_training_section"),
    "generate_training_section_yaml":         ("training_section_tools",  "generate_training_section_yaml"),
    # testing
    "validate_testing_section":               ("testing",                 "validate_testing_section"),
    "generate_testing_section_yaml":          ("testing",                 "generate_testing_section_yaml"),
    # compilation
    "get_compilation_preset_recommendations": ("compilation",             "get_compilation_preset_recommendations"),
    "validate_compilation_section":           ("compilation",             "validate_compilation_section"),
    "generate_compilation_section_yaml":      ("compilation",             "generate_compilation_section_yaml"),
    # config_file_tools
    "generate_complete_config_file":          ("config_file_tools",       "generate_complete_config_file"),
    "run_example":                            ("config_file_tools",       "run_example"),
    # device_deployment
    "check_sdk_installation":                 ("device_deployment",       "check_sdk_installation"),
    "find_run_artifacts":                     ("device_deployment",       "find_run_artifacts"),
    # "copy_artifacts_to_ccs":                  ("device_deployment",       "copy_artifacts_to_ccs"),
    "create_ccs_project":                     ("device_deployment",       "create_ccs_project"),
    "build_ccs_project":                      ("device_deployment",       "build_ccs_project"),
    "flash_ccs_project":                      ("device_deployment",       "flash_ccs_project"),
}



def _get_required_params(fn):
    """Return list of required (no-default) parameter names for a function."""
    sig = inspect.signature(fn)
    return [
        name for name, param in sig.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def main():
    raw_args = sys.argv[1:]

    # Extract flags
    save_yaml_path = None
    save_result_path = None
    positional = []
    i = 0
    while i < len(raw_args):
        if raw_args[i] == "--save-yaml" and i + 1 < len(raw_args):
            save_yaml_path = raw_args[i + 1]
            i += 2
        elif raw_args[i] == "--save-result" and i + 1 < len(raw_args):
            save_result_path = raw_args[i + 1]
            i += 2
        else:
            positional.append(raw_args[i])
            i += 1

    if not positional:
        print(json.dumps({"error": "Usage: runner.py <function_name> [<json_args>] [--save-yaml <file>] [--save-result <file>]"}))
        sys.exit(1)

    func_name = positional[0]

    # for skill to list available functions
    if func_name == "--list":
        fns = sorted(list(FUNCTION_MAP.keys()) + [
            "get_update_status", "check_updates", "do_update"
        ])
        print(json.dumps({"available_functions": fns}))
        return

    # Built-in functions
    if func_name == "get_update_status":
        result = get_update_status()
        _output_result(result, save_yaml_path, save_result_path)
        return

    if func_name == "check_updates":
        result = check_updates()
        _output_result(result, save_yaml_path, save_result_path)
        return

    if func_name == "do_update":
        args_json = positional[1] if len(positional) > 1 else "{}"
        try:
            kwargs = json.loads(args_json)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON args: {e}"}))
            sys.exit(1)
        if "tinyml_base_path" not in kwargs:
            print(json.dumps({"error": "Missing required parameter: tinyml_base_path"}))
            sys.exit(1)
        result = do_update(kwargs["tinyml_base_path"])
        _output_result(result, save_yaml_path, save_result_path)
        if not result.get("success"):
            sys.exit(1)
        return

    if func_name not in FUNCTION_MAP:
        print(json.dumps({
            "error": f"Unknown function: '{func_name}'",
            "available": sorted(list(FUNCTION_MAP.keys()) + ["get_update_status", "check_updates", "do_update"]),
        }))
        sys.exit(1)

    args_json = positional[1] if len(positional) > 1 else "{}"

    module_name, fn_name = FUNCTION_MAP[func_name]

    try:
        kwargs = json.loads(args_json)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON args: {e}. Make sure the JSON is quoted correctly."}))
        sys.exit(1)

    try:
        import importlib
        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)

        # Catch missing required args before calling, to give clear guidance
        required = _get_required_params(fn)
        missing = [p for p in required if p not in kwargs]
        if missing:
            print(json.dumps({
                "error": f"Missing required parameter(s): {missing}",
                "function": func_name,
                "required_params": required,
                "provided_params": list(kwargs.keys()),
                "hint": f"Add the missing parameter(s) to your JSON args object.",
            }))
            sys.exit(1)

        result = fn(**kwargs)

        _output_result(result, save_yaml_path, save_result_path)

        if isinstance(result, dict) and result.get("success") is False:
            sys.exit(1)

    except ImportError as e:
        print(json.dumps({"error": f"Module import failed: {e}", "type": "ImportError"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "type": type(e).__name__,
            "hint": "This is an unexpected internal error. Check the arguments and try again.",
        }))
        sys.exit(1)


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


if __name__ == "__main__":
    main()
