#!/usr/bin/env bash

#################################################################################
if [ $# -le 1 ]; then
    echo "help:"
    echo "$0 target_device config_file"
    exit 0
fi

#################################################################################
HOME_DIR=${HOME}
HOME_DIR=$(realpath $HOME_DIR)

WORK_DIR=${WORK_DIR:-"./"}
DATA_DIR=${WORK_DIR:-"./data"}

export TOOLS_PATH=${TOOLS_PATH:-$HOME_DIR/bin}

TARGET_SOC=${1:-F28P55}
CONFIG_FILE=${2:-config_timeseries_classification_kilby.yaml}

#################################################################################
export PYTHONPATH=.:$PYTHONPATH

#################################################################################
# print some settings
echo "Target device                     : ${TARGET_SOC}"
echo "PYTHONPATH                        : ${PYTHONPATH}"

#################################################################################
python tinyml_modelmaker/run_tinyml_modelmaker.py ${CONFIG_FILE} --target_device ${TARGET_SOC}
