#!/usr/bin/env bash

# system packages
sudo apt-get install -y libjpeg-dev zlib1g-dev cmake libffi-dev protobuf-compiler

######################################################################
# upgrade pip
python -m ensurepip --upgrade
python -m pip install --no-input --upgrade pip setuptools
######################################################################
echo "installing pytorch - use the appropriate index-url from https://pytorch.org/get-started/locally/"
python -m pip install --no-input torch==2.3.0 --index-url https://download.pytorch.org/whl/cu124

echo 'Installing python packages...'
python -m pip install --no-input cython wheel numpy==1.26.4
python -m pip install --no-input torchinfo==1.8.0 pycocotools opencv-python
python -m pip uninstall --yes pillow
#python -m pip uninstall --no-input pillow-simd

echo "installing requirements"
python -m pip install --no-input -r requirements/requirements.txt
python -m pip install --no-input -r requirements/requirements_ti_packages.txt

######################################################################
# can we move this inside the requirements file is used.
#python -m pip install --no-input protobuf onnx==1.13.0

######################################################################
python -m pip install --editable . --use-pep517

######################################################################
# setup the edgeai_xvision package, which is inside references/edgeailite
#python -m pip install --no-input -r ./references/edgeailite/requirements.txt
#python -m pip install -e ./references/edgeailite/

