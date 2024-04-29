#!/usr/bin/env bash

# system packages
sudo apt-get install -y libjpeg-dev zlib1g-dev cmake libffi-dev protobuf-compiler

######################################################################
# upgrade pip
pip3 install --no-input --upgrade pip setuptools
######################################################################
echo "installing pytorch - use the appropriate index-url from https://pytorch.org/get-started/locally/"
pip3 install --no-input torch==2.1.1+cpu torchvision==0.16.1+cpu torchaudio==2.1.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

echo 'Installing python packages...'
pip3 install --no-input cython wheel numpy==1.23.0
pip3 install --no-input torchinfo pycocotools opencv-python
pip uninstall --yes pillow
#pip uninstall --no-input pillow-simd

echo "installing requirements"
pip3 install --no-input -r requirements/requirements.txt

######################################################################
# can we move this inside the requirements file is used.
#pip3 install --no-input protobuf==3.20.2 onnx==1.13.0

######################################################################
python3 setup.py develop

######################################################################
# setup the edgeai_xvision package, which is inside references/edgeailite
#pip3 install --no-input -r ./references/edgeailite/requirements.txt
#pip3 install -e ./references/edgeailite/

