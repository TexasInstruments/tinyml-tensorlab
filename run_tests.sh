#!/bin/bash
#
# Wrapper script to test all ModelMaker configs
# Automatically activates the py310_tinyml environment
#

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "ModelMaker Config Test Suite"
echo "======================================================================"
echo ""

# Check if environment exists
#if [ ! -d "$HOME/.pyenv/versions/py310_tinyml" ]; then
#    echo -e "${RED}Error: py310_tinyml environment not found${NC}"
#    echo "Please ensure the environment exists at: ~/.pyenv/versions/py310_tinyml"
#    exit 1
#fi

# Activate environment
#echo -e "${YELLOW}Activating py310_tinyml environment...${NC}"
#source ~/.pyenv/versions/py310_tinyml/bin/activate

#if [ $? -ne 0 ]; then
#    echo -e "${RED}Failed to activate environment${NC}"
#    exit 1
#fi

#echo -e "${GREEN}✓ Environment activated${NC}"
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo ""

# Change to modelmaker directory
cd "$(dirname "$0")"

# Run the test script with all arguments passed through
echo -e "${YELLOW}Starting tests...${NC}"
echo ""
python test_all_configs.py "$@"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✓ All tests completed successfully${NC}"
else
    echo -e "${RED}✗ Some tests failed (exit code: $exit_code)${NC}"
fi

exit $exit_code
