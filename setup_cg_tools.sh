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

HOME_DIR=${HOME}
HOME_DIR=$(realpath $HOME_DIR)

INSTALLER_PATH=${INSTALLER_PATH:-$HOME_DIR/installer}
TOOLS_PATH=${TOOLS_PATH:-$HOME_DIR/bin}
mkdir ${INSTALLER_PATH}
mkdir ${TOOLS_PATH}

# multiarch i386 is needed for c2000ware
sudo apt update
sudo apt install -y binutils-multiarch binutils
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install -y binutils:i386

#################################################################################
# c2000 cgtools
C2000_CGT_INSTALLER=ti_cgt_c2000_22.6.0.LTS_linux-x64_installer.bin
C2000_CGT_INSTALLER_FILE=${INSTALLER_PATH}/${C2000_CGT_INSTALLER}
rm -f ${C2000_CGT_INSTALLER_FILE}
wget https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-xqxJ05PLfM/22.6.0.LTS/${C2000_CGT_INSTALLER} -O ${C2000_CGT_INSTALLER_FILE}
chmod +x ${C2000_CGT_INSTALLER_FILE}
${C2000_CGT_INSTALLER_FILE} --mode unattended --prefix ${TOOLS_PATH}

# now install c2000ware
C2000WARE_INSTALLER=C2000Ware_5_02_00_00_setup.run
C2000WARE_INSTALLER_FILE=${INSTALLER_PATH}/${C2000WARE_INSTALLER}
rm -f ${C2000WARE_INSTALLER_FILE}
wget  http://c2000-ubuntu-build-pc.dhcp.ti.com/c2000ware-builds/5.02.00.00/RC1/${C2000WARE_INSTALLER} -O ${C2000WARE_INSTALLER_FILE}
chmod +x ${C2000WARE_INSTALLER_FILE}
${C2000WARE_INSTALLER_FILE} --mode unattended --prefix ${TOOLS_PATH}

sudo apt update && sudo apt install -y binutils g++ gcc
