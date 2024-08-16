#!/usr/bin/env bash

# system packages
sudo apt-get install -y libjpeg-dev zlib1g-dev cmake libffi-dev protobuf-compiler

######################################################################
# upgrade pip
pip3 install --no-input --upgrade pip setuptools
######################################################################
echo "installing pytorch - use the appropriate index-url from https://pytorch.org/get-started/locally/"
pip3 install --no-input torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

echo 'Installing python packages...'
pip3 install --no-input cython wheel numpy==2.0.1
pip3 install --no-input torchinfo==1.8.0 pycocotools opencv-python
pip uninstall --yes pillow
#pip uninstall --no-input pillow-simd

echo "installing requirements"
pip3 install --no-input -r requirements/requirements.txt
######################################################################
python3 -m pip install --editable .


