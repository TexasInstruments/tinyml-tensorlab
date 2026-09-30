#!/usr/bin/env bash

#################################################################################
if [ $# -le 0 ]; then
    echo "help:"
    echo "$0 config_file [--target_device DEVICE] [--model_name NAME] [--variables N] \\"
    echo "               [--input-features N] [--num-classes N] [--skip-fel] [--quantization N]"
    exit 0
fi
#################################################################################
# Portable realpath for macOS (GNU coreutils realpath not installed by default)
if ! command -v realpath &>/dev/null; then
    realpath() { python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"; }
fi

HOME_DIR=${HOME}
HOME_DIR=$(realpath $HOME_DIR)

WORK_DIR=${WORK_DIR:-"./"}
DATA_DIR=${WORK_DIR:-"./data"}
export TOOLS_PATH=${TOOLS_PATH:-$HOME_DIR/bin}
export PYTHONPATH=../tinyml-modelmaker:../tinyml-modeloptimization:../tinyml-modeloptimization/torchmodelopt:../tinyml-tinyverse:.:$PYTHONPATH

echo "PYTHONPATH                        : ${PYTHONPATH}"
python estimate_memory.py "$@"

#################################################################################
# Standalone, fast memory pre-flight check: resolves a config's model
# architecture + input shape and reports estimated ROM/RAM without
# downloading a dataset, training, or running the full compilation pipeline.
# See estimate_memory.py for details.
#################################################################################
