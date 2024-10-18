# Tiny ML ModelMaker

Tiny ML Modelmaker is an end-to-end model development tool that contains dataset handling, model training and compilation. Currently, it doesn't have an integrated feature to annotate data, but can accept annotated Dataset from a tool such as [Label Studio](https://labelstud.io/)

This repository stitches individual core scripts and utilities present in Tiny ML Tinyverse to make a simple and consistent interface for model development. This does not support all the models that can be trained and compiled using our tools, but only a subset. This is a commandline tool and requires a Linux PC.

The following are the key functionality supported by this tool:
- Dataset handling: This dataset formats supported by this tool is described in a section below. This can convert dataset formats and can automatically split the given dataset into train and validation sets (if it is not already split).
- Model training: Model training repositories such as [tinyml-tinyverse](https://github.com/TexasInstruments/tinyml-tinyverse) built using [PyTorch](https://pytorch.org/) and [torchaudio](https://pytorch.org/audio/stable/index.html) are integrated. Several models with pretrained checkpoints are incorporated for each of these repositories. 
- Model compilation: Model compilation tools TI's [neo-tvm](https://bitbucket.itg.ti.com/projects/MCTOOLS/repos/neo-tvm/) for TI's microcontrollers have been integrated.


Tasks and Models

| Task | Modality                  | Example Application                                     | Description |
| --- |---------------------------|---------------------------------------------------------| --- | 
| Classification | Timeseries (Univariate)   | Arc Detection, Fall Detection                           | 1 to n-class classification of timeseries data |
| Classification | Timeseries (Multivariate) | Arc Detection, Motor Bearing Fault Detection, Gyroscope | 1 to n-class classification of timeseries data |



These functionalities that are supported are fully integrated and the user can control it by setting  parameters in the config file.

## Step 1: OS & Environment 

This repository can be used from native Ubuntu bash terminal directly or from within a docker environment.

#### Step 1, Option 1: With native Ubuntu environment and pyenv (recommended)
We have tested this tool in Ubuntu 22.04 and with Python 3.10
* If you rather want to try this offering out on a Windows PC, then a [user guide](./docs/Windows_Subsytem_for_Linux.md) to use this toolchain on Windows Subsystem for Linux has been provided.

In this option, we describe using this repository with the pyenv environment manager. 

Step 1.1a: Make sure that you are using bash shell. If it is not bash shell, change it to bash. Verify it by typing:
```
echo ${SHELL}
```

Step 1.2a: Install system dependencies
```
sudo apt update
sudo apt install build-essential curl libbz2-dev libffi-dev liblzma-dev libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev libxml2-dev libxmlsec1-dev llvm make tk-dev xz-utils wget curl
sudo apt install -y libffi-dev libjpeg-dev zlib1g-dev graphviz graphviz-dev protobuf-compiler
```

Step 1.3a: Install pyenv using the following commands
```
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

echo '# pyenv settings ' >> ${HOME}/.bashrc
echo 'command -v pyenv >/dev/null || export PATH=":${HOME}/.pyenv/bin:$PATH"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv init -)"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ${HOME}/.bashrc
echo '' >> ${HOME}/.bashrc

exec ${SHELL}
```

Further details on pyenv installation are given here https://github.com/pyenv/pyenv and https://github.com/pyenv/pyenv-installer


Step 1.4a: Install Python 3.10 in pyenv and create an environment
```
pyenv install 3.10
pyenv virtualenv 3.10 py310
pyenv rehash
pyenv activate py310
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools
```

Step 1.5a: **Activate the Python environment.** This activation step needs to be done everytime one starts a new terminal or shell. (Alternately, this also can be written to the .bashrc, so that this will be the default penv environment).
```
pyenv activate py310
```

#### Step 1, Option 2: With docker environment

Step 1.1b: Install docker if you don't have it already. The following steps are for installation on Ubuntu 22.04
```
./docker/docker_setup.sh
```

Step 1.2b: Build docker image:
```
./docker/docker_build.sh
```

Step 1.3b: Run docker container to bring up the container terminal on docker:
```
./docker/docker_run.sh
```

Source .bashrc to update the PATH
```
source /opt/.bashrc
```

Step 1.4b: During docker run, we map the parent directory of this folder to /opt/code This is to easily share code and data between the host and the docker container. Inside the docker terminal, change directory to where this folder is mapped to:
```
cd /opt/code/tinyml-modelmaker
```



## Step 2: Setup the model training and compilation repositories

Firstly, clone this repo into your PC.
This tool depends on several repositories that we have published at https://github.com/TexasInstruments
The following setup script can take care of cloning the required repositories and running their setup scripts.

**NOTE: Please download and install [C2000Ware](https://www.ti.com/tool/C2000WARE)**
* Please set the installed path in your terminal: `export C2000WARE_PATH="/path/to/C2000Ware_5_03_00_00"`

```
./setup_all.sh 
```
If you get permission denied error, then try:  
```
bash setup_all.sh
```

If the script runs sucessfully, you should have this directory structure: 

```
parent_directory
    |
    |--tinyml-modelmaker
    |--tinyml-tinyverse
    |--tinyml-modeloptimization

```
Your python environment will have several model compilation python packages installed. See it by running:
```
pip list | grep 'onnxruntime\|tvm'
```

Also, PyTorch and its related packages will be installed. See it by running:
```
pip list | grep 'torch\|torchaudio'
```

## Step 3: Run the ready-made examples

```
run_tinyml_modelmaker.sh  <target_device> <config_file>
```

#### Examples: 
 
Timeseries Classification (Arc fault detection) example
```
run_tinyml_modelmaker.sh F28P55 config_timeseries_classification_dsi.yaml
(or)
run_tinyml_modelmaker.sh F28P55 config_timeseries_classification_dsk.yaml
```

Audio classification example
```
run_tinyml_modelmaker.sh F28P55 config_audio_classification.yaml
```
Where F28P55 above is an example of target_device supported.

#### Disclaimer
Although dataset loading and training are agnostic to the device, for the compilation aspect we do require compilers to be installed for cross-compilation.
That is, for C28ST, you would need to download and install [cgt2000](https://www.ti.com/tool/C2000-CGT#downloads) .
(Not supported as of April 2024) That is, for AM263, you would need to download and install [tiarmclang](https://www.ti.com/tool/download/ARM-CGT-CLANG/2.1.3.LTS) . 


#### Target devices supported
The list of target devices supported depends on the neo-tvm installed. Currently, **F28P55, F28P65, F2837, F28003** are supported.


## Step 4: Dataset format
- The dataset format is similar to that of the [Google Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) dataset, but there are some changes as explained below.
 

####  Dataset format
The dataset should have the following structure. 

<pre>
data/projects/<dataset_name>/dataset
                             |
                             |--classes
                             |     |-- the directories should be here
                             |     |-- class1
                             |     |-- class2
                             |
                             |--annotations
                                   |--instances_train_list.txt
                                   |--instances_val_list.txt
</pre>

- Use a suitable dataset name instead of dataset_name
- Look at the example dataset [Arc Fault Classification](https://software-dl.ti.com/jacinto7/esd/modelzoo/other/tinyml/00_05_00/datasets/arc_fault_classification_dsk.zip) to understand further.
- In the config file, provide the name of the dataset (dataset_name in this example) in the field dataset_name and provide the path or URL in the field input_data_path.
- Then the ModelMaker tool can be invoked with the config file.


#### Notes
If the dataset has already been split into train and validation set already, it is possible to provide those paths separately as a tuple in input_data_path.

After the model compilation, the compiled models will be available in a folder inside [./data/projects](./data/projects)

The config file can be in .yaml or in .json format


## Step 6: Accelerated Training using GPUs (Optional) 

Note: **This section is for advanced users only**. Familiarity with NVIDIA GPU and CUDA driver installation is assumed.

This tool can train models either on CPU or on GPUs. By default, CPU based training is used. 

It is possible to speedup model training significantly using GPUs (with CUDA support) - if you have those GPUs in the PC. The PyTorch version that we install by default is not capable of supporting CUDA GPUs. There are additional steps to be followed to enable GPU support in training. 
- In the file setup_all.sh, we are using setup_cpu.sh for several of the repositories that we are using. These will have to be changed to setup.sh before running setup_all.sh
- Install GPU driver and other tools as described in the sections below.
- In the config file, set a value for num_gpus to a value greater than 0 (should not exceed the number of GPUs in the system) to enable GPU based training.

#### Option 1: When using Native Ubuntu Environment

The user has to install an appropriate NVIDIA GPU driver that supports the GPU being used.

The user also has to install CUDA Toolkit. See the [CUDA download instructions](https://developer.nvidia.com/cuda-downloads). 

#### Option 2: When using docker environment

Enabling CUDA GPU support inside a docker environment requires several additional steps. Please follow the instructions given in: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Once CUDA is installed, you will be able to model training much faster.

### Additional Docs
- [Bring Your Own Model for Compilation](./docs/BYOM_for_Compilation.md)
- [Understanding the config.yaml](./docs/UnderstandingConfigFile.md)
- [Adding Models](./docs/AddingModels.md)
- [Adding Modalities](./docs/AddingModalities.md)

---
## Contributor and Maintainer
- [Adithya Thonse](https://github.com/Adithya-Thonse)
