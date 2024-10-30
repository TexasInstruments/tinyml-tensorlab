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
mkdir -p ${INSTALLER_PATH}
mkdir -p ${TOOLS_PATH}

# multiarch i386 is needed for c2000ware
sudo apt update
sudo apt install -y binutils-multiarch binutils
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install -y binutils:i386

#################################################################################

# now install c2000ware
C2000WARE_INSTALLER=C2000Ware_5_04_00_00_setup.run
C2000WARE_INSTALLER_FILE=${INSTALLER_PATH}/${C2000WARE_INSTALLER}
# rm -f ${C2000WARE_INSTALLER_FILE}
wget http://c2000-ubuntu-build-pc.dhcp.ti.com/c2000ware-builds/5.04.00.00/RC4/${C2000WARE_INSTALLER} -O ${C2000WARE_INSTALLER_FILE}
if [ -e ${C2000WARE_INSTALLER_FILE} ]; then
    echo -e "\033[0;32mDownloaded ${C2000WARE_INSTALLER_FILE} from TI remote repository!\033[0m"
    chmod +x ${C2000WARE_INSTALLER_FILE}
    ${C2000WARE_INSTALLER_FILE} --mode unattended --prefix ${TOOLS_PATH}
else
  echo -e "\033[31;5mCRITICAL WARNING! C2000Ware could not be downloaded! \033[0m"  # Blinking Text
  echo -e "\033[0;33mPlease note that C2000Ware requires SW Export License permissions. \033[0m"
  echo -e "\033[0;33mIf you are installing this outside TI network, then you will need to separately download the Linux version: ${C2000WARE_INSTALLER} from: https://www.ti.com/tool/download/C2000WARE/ \033[0m"
  echo -e "\033[0;33mAnd then run the following 2 commands:\033[0m"
  echo "chmod +x ${C2000WARE_INSTALLER}"
  echo "${C2000WARE_INSTALLER} --mode unattended --prefix ${TOOLS_PATH}"
  echo -e "\033[0;33mPlease hit Ctrl+C repeatedly and manually download the C2000Ware and install it. \033[0m"
  echo -e "\033[0;33mContinuing installation nevertheless- meaning you can do AI training, but not model compilation. \033[0m"
  echo -e "\033[0;32mIgnore the above messages if you have already manually installed C2000Ware \033[0m"
  echo -e "\033[33;5mWaiting for 90 seconds to let you make a decision.\033[0m"
  sleep 90
fi

sudo apt update && sudo apt install -y binutils g++ gcc
