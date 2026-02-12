#!/bin/bash
#
# Tiny ML ModelZoo Training Wrapper
# Delegates training to tinyml-modelmaker
#
# Usage:
#   ./run_tinyml_modelzoo.sh examples/hello_world/config.yaml
#   ./run_tinyml_modelzoo.sh /absolute/path/to/config.yaml
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ModelMaker location (sibling directory)
MODELMAKER_DIR="$SCRIPT_DIR/../tinyml-modelmaker"
RUN_SCRIPT="$MODELMAKER_DIR/tinyml_modelmaker/run_tinyml_modelmaker.py"

# Check if modelmaker exists
if [ ! -f "$RUN_SCRIPT" ]; then
    echo -e "${RED}Error: ModelMaker not found at $RUN_SCRIPT${NC}"
    echo "Make sure tinyml-modelmaker is installed alongside tinyml-modelzoo"
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
            echo "  $(realpath --relative-to="$SCRIPT_DIR" "$cfg")"
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
echo "Config:      $CONFIG_FILE"
echo "ModelMaker:  $RUN_SCRIPT"
echo "========================================"
echo ""

# Run training via modelmaker
cd "$MODELMAKER_DIR"
python "$RUN_SCRIPT" "$CONFIG_FILE" "$@"
