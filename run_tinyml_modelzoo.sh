#!/bin/bash
#
# Tiny ML ModelZoo Training Wrapper
# Delegates training to tinyml-modelmaker, installed as a Python package
# (wheel or editable install) - it does NOT need to be cloned as a sibling
# directory.
#
# Usage:
#   ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml
#   ./run_tinyml_modelzoo.sh /absolute/path/to/config.yaml

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# tinyml_modelmaker is a dependency of this repo's pyproject.toml, so
# `pip install -e .` here pulls it in (and transitively tinyverse +
# torchmodelopt) automatically.
if ! python -c "import tinyml_modelmaker" >/dev/null 2>&1; then
    echo -e "${RED}Error: tinyml_modelmaker is not installed in this Python environment.${NC}"
    echo "Run: pip install -e $SCRIPT_DIR"
    exit 1
fi

# Check arguments
if [ $# -lt 1 ]; then
    echo "Tiny ML ModelZoo Training Wrapper"
    echo ""
    echo "Usage: $0 <config_file> [additional_args...]"
    echo ""
    echo "Examples:"
    echo "  $0 examples/hello_world/config.yaml"
    echo "  $0 examples/motor_bearing_fault/config.yaml"
    echo ""
    echo "Available example configs:"
    if [ -d "$SCRIPT_DIR/examples" ]; then
        find "$SCRIPT_DIR/examples" -name "*.yaml" -type f | sort | while read -r cfg; do
            echo "  ${cfg#$SCRIPT_DIR/}"
        done
    fi
    exit 1
fi

CONFIG_FILE="$1"
shift  # Remove first argument, keep the rest

# Convert relative path to absolute if needed
if [[ ! "$CONFIG_FILE" = /* ]]; then
    # Check if it's relative to current directory
    if [ -f "$CONFIG_FILE" ]; then
        CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
    # Check if it's relative to script directory (e.g., examples/...)
    elif [ -f "$SCRIPT_DIR/$CONFIG_FILE" ]; then
        CONFIG_FILE="$SCRIPT_DIR/$CONFIG_FILE"
    else
        echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
        exit 1
    fi
fi

# Verify config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}Tiny ML ModelZoo Training${NC}"
echo "========================================"
echo "Config: $CONFIG_FILE"
echo "========================================"
echo ""
echo "Output artifacts will be written under ./data/projects relative to"
echo "the directory you ran this script from (override with common.projects_path"
echo "in the config)."
echo ""

# Run training via modelmaker. Invoked as a module (-m), not by script path -
# run_tinyml_modelmaker.py has a chdir('..') special-case for the latter that
# assumes a repo-checkout layout and would land in the wrong directory here.
python -m tinyml_modelmaker.run_tinyml_modelmaker "$CONFIG_FILE" "$@"
