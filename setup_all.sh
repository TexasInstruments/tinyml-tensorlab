#!/usr/bin/env bash

#################################################################################
# Copyright (c) 2018-2024, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################

#################################################################################
CUR_DIR=$(pwd)
PARENT_DIR=$(realpath ${CUR_DIR}/..)
HOME_DIR=${HOME}
HOME_DIR=$(realpath $HOME_DIR)

#################################################################################
# internal or external repositories
USE_INTERNAL_REPO=0

if [[ ${USE_INTERNAL_REPO} -eq 1 ]]; then
    SOURCE_LOCATION="ssh://git@bitbucket.itg.ti.com/tinyml-algo/"
else
    SOURCE_LOCATION="https://github.com/TexasInstruments/tinyml-tensorlab/"
fi
# print
echo "SOURCE_LOCATION="${SOURCE_LOCATION}

#################################################################################
# clone
echo "cloning/updating git repositories. this may take some time..."
echo "if there is any issue, please remove these folders and try again ${PARENT_DIR}/tinyml-tinyverse"
if [[ ! -d ${PARENT_DIR}/tinyml-tinyverse ]]; then git clone --branch main ${SOURCE_LOCATION}tinyml-tinyverse.git ${PARENT_DIR}/tinyml-tinyverse; else ls ${PARENT_DIR}/tinyml-tinyverse; fi
if [[ ! -d ${PARENT_DIR}/tinyml-modeloptimization ]]; then git clone --branch main ${SOURCE_LOCATION}tinyml-modeloptimization.git ${PARENT_DIR}/tinyml-modeloptimization; else ls ${PARENT_DIR}/tinyml-modeloptimization; fi

cd ${PARENT_DIR}/tinyml-modelmaker
echo "cloning/updating done."


#################################################################################
# upgrade pip
pip install --no-input --upgrade pip setuptools
pip install --no-input --upgrade wheel # cython numpy==1.23.0

#################################################################################
# install code get tools
# Note: this will need sudo access.
./setup_cg_tools.sh

#################################################################################
echo "preparing environment..."
# for setup.py develop mode to work inside docker environment, this is required
git config --global --add safe.directory $(pwd)

echo "installing repositories..."

echo "installing: tinyml-modeloptimization"
cd ${PARENT_DIR}/tinyml-modeloptimization/torchmodelopt
./setup.sh
cd ..

echo "installing: tinyml-tinyverse"
cd ${PARENT_DIR}/tinyml-tinyverse
./setup_cpu.sh
# Uncomment below line and comment the above line to install GPU version of torch
# ./setup.sh

echo "installing tinyml-modelmaker"
cd ${PARENT_DIR}/tinyml-modelmaker
./setup.sh

#################################################################################
ls -d ${PARENT_DIR}/tinyml-*

echo "installation done."
