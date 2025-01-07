# Instructions to run motor_fault_classification.py

## Install standard modules

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnx torchinfo tabulate
```

## Install TI modules

### Install TVM (known externally as TI MCU Neural Network Compiler)

```bash
# Install the NNC compiler
pip install https://software-dl.ti.com/mctools/esd/tvm/mcu/ti_mcu_nnc-1.2.0-cp310-cp310-linux_x86_64.whl
```

### Install tinyml-modeloptimization

```bash
git clone ssh://git@bitbucket.itg.ti.com/tinyml-algo/tinyml-tensorlab.git
cd tinyml-tensorlab/tinyml-modeloptimization/torchmodelopt
./setup.sh
```

## Install TVM compiler dependencies

TODO

## Train the model 

```bash
# Perform Training, QAT, Export & Inference
python3 motor_fault_classification.py
```

