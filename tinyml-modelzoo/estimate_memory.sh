#!/usr/bin/env bash
#
# Tiny ML ModelZoo memory pre-flight check.
# Uses tinyml-modelmaker, installed as a Python package (wheel or editable
# install) - it does NOT need to be cloned as a sibling directory.
#
# Usage:
#   ./estimate_memory.sh examples/dc_arc_fault/config_dsk.yaml [options]

#################################################################################
if [ $# -le 0 ]; then
    echo "help:"
    echo "$0 config_file [--target_device DEVICE] [--model_name NAME] [--variables N] \\"
    echo "               [--input-features N] [--num-classes N] [--skip-fel] [--quantization N]"
    exit 0
fi
#################################################################################
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# tinyml_modelmaker is a dependency of this repo's pyproject.toml, so
# `pip install -e .` here pulls it in (and transitively tinyverse +
# torchmodelopt) automatically.
if ! python -c "import tinyml_modelmaker" >/dev/null 2>&1; then
    echo "Error: tinyml_modelmaker is not installed in this Python environment."
    echo "Run: pip install -e $SCRIPT_DIR"
    exit 1
fi

# Folder holding the TI compiler tools. Same default as tinyml-modelmaker
# (~/ti); override by exporting TOOLS_PATH before running this script.
export TOOLS_PATH=${TOOLS_PATH:-$HOME/ti}

# Relative config paths are resolved against the current directory first, then
# against this repo (e.g. examples/...), like run_tinyml_modelzoo.sh does.
CONFIG_FILE="$1"
shift
if [[ ! "$CONFIG_FILE" = /* ]] && [ ! -f "$CONFIG_FILE" ] && [ -f "$SCRIPT_DIR/$CONFIG_FILE" ]; then
    CONFIG_FILE="$SCRIPT_DIR/$CONFIG_FILE"
fi

python "$SCRIPT_DIR/estimate_memory.py" "$CONFIG_FILE" "$@"

#################################################################################
# Standalone, fast memory pre-flight check: resolves a config's model
# architecture + input shape and reports estimated ROM/RAM without
# downloading a dataset, training, or running the full compilation pipeline.
# See estimate_memory.py for details.
#################################################################################
