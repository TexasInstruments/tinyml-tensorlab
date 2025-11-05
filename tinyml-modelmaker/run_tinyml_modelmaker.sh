#!/usr/bin/env bash

#################################################################################
if [ $# -le 0 ]; then
    echo "help:"
    echo "$0 config_file"
    exit 0
fi
#################################################################################
HOME_DIR=${HOME}
HOME_DIR=$(realpath $HOME_DIR)

WORK_DIR=${WORK_DIR:-"./"}
DATA_DIR=${WORK_DIR:-"./data"}
export TOOLS_PATH=${TOOLS_PATH:-$HOME_DIR/bin}
CONFIG_FILE=${1:-dc_arc_fault/config_dsk.yaml}
export PYTHONPATH=.:$PYTHONPATH

echo "PYTHONPATH                        : ${PYTHONPATH}"
python tinyml_modelmaker/run_tinyml_modelmaker.py ${CONFIG_FILE}

#################################################################################
# This script was changed to remove target device as an argument on 4th October 2025 and is effective from 1.2 release
# Check TINYML_ALGO-448
#################################################################################