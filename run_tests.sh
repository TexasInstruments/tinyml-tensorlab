#!/bin/bash
#
# Wrapper script to test all ModelZoo configs
# This script tests that all example configs work correctly with modelmaker.
#
# Usage:
#   source ~/.pyenv/versions/py310_tinyml/bin/activate
#   ./run_tests.sh [--timeout 2400] [--filter STRING] [--stop-on-error]
#

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "TinyML ModelZoo Test Suite"
echo "======================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for examples directory
if [ ! -d "$SCRIPT_DIR/examples" ]; then
    echo -e "${RED}Error: examples directory not found${NC}"
    echo "Expected: $SCRIPT_DIR/examples"
    exit 1
fi

echo "Python: $(which python)"
echo "Version: $(python --version)"
echo "Script Dir: $SCRIPT_DIR"
echo ""

# Run the test script with all arguments passed through
echo -e "${YELLOW}Starting tests...${NC}"
echo ""
python "$SCRIPT_DIR/test_all_configs.py" "$@"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✓ All tests completed successfully${NC}"
else
    echo -e "${RED}✗ Some tests failed (exit code: $exit_code)${NC}"
fi

exit $exit_code
