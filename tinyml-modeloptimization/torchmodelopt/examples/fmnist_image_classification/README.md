# Instructions to run fmnist_ti_qat.py

## Set up Python environment

```bash
python3 -m venv .venv
source ./.venv/bin/activate
```

## Install standard modules

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxscript
pip install tabulate
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

E.g., for MSP M0:

```bash
# Download and install TI MSPM0 SDK: https://www.ti.com/tool/MSPM0-SDK
export MSPM0_SDK_PATH=$HOME/scratch/ml/ti-sdks/mspm0_sdk_2_02_00_05
# Download and install TI Arm Clang Compiler: https://www.ti.com/tool/download/ARM-CGT-CLANG
export PATH=$HOME/scratch/ml/ti-compilers/ti-cgt-armllvm_4.0.0.LTS/bin:$PATH
export TVMC_OPTIONS="--output-format=a --runtime=crt --executor=aot --executor-aot-interface-api=c --executor-aot-unpacked-api=1 --pass-config tir.disable_vectorize=1 --pass-config tir.usmp.algorithm=hill_climb"
export TIARMCLANG_OPTIONS="-D__MSPM0G3507__ -Os -mcpu=cortex-m0plus -march=thumbv6m -mtune=cortex-m0plus -mthumb -mfloat-abi=soft -I${MSPM0_SDK_PATH}/source/third_party/CMSIS/Core/Include -I${MSPM0_SDK_PATH}/source -Wno-return-type -I."
```

## Perform QAT and run the model with the ONNX runtime on the host

```bash
# Perform QAT
python3 fmnist_ti_qat.py
# Inference
python3 run_fmnist.py --model-name fmnist.onnx
python3 run_fmnist.py --model-name fmnist_int8.onnx
```

## Compile model with TVM

Detailed [User Guide](https://software-dl.ti.com/mctools/esd/tvm/mcu/docs/index.html) for model types supported by TVM and general know-how.

Note: Compile is currently broken for FMIST, TVM fails to compile the model because the TI-NPU prepare pass is expecting:
1x784  Add  1
Or
1x784 Add  1x1

But in this case, it gets:
1x784  Add  1x1x1x1  =>  1x1x1x784

This will be fixed in the next RC release of TVM (1.3.0rc4)

```bash
tvmc compile --output-format=a --runtime=crt --executor=aot --executor-aot-interface-api=c --executor-aot-unpacked-api=1 --pass-config tir.disable_vectorize=1 --pass-config tir.usmp.algorithm=hill_climb --target="c, ti-npu type=soft" --target-c-mcpu=cortex-m0plus ./fmnist_int8.onnx -o artifacts_m0_soft/mod.a --cross-compiler="tiarmclang" --cross-compiler-options="$TIARMCLANG_OPTIONS -Iartifacts_m0_soft
```