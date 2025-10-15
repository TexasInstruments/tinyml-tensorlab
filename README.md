# tinyml-tensorlab - TI's MCU AI Toolchain

<hr>

##### Table of Contents

- [Introduction](#introduction)
- [Which Repo do I need to fiddle with?](#which-repo-do-i-need-to-fiddle-with)
- [Using this repository](#using-this-repository)
    - [I'm a user](#im-a-user)
    - [I'm a developer](#im-a-developer)
- [Productized Applications](#productized-applications)
- [Release History](#what-is-new)

<hr>

## Introduction

The Tiny ML Tensorlab repository is meant to be as a starting point to install and explore TI's AI offering for MCUs.
It helps to install all the required repositories to get started. Currently, it can handle Time series Classification, Regression and Anomaly Detection tasks. 


Once you clone this repository, you will find the following repositories present within the `tinyml-tensorlab` directory:
* `tinyml-tensorlab`: This repo, serves as a blank wrapper for customers to clone all the tinyml repos at one shot. 
![image](./assets/mcu_ai_repo_connection.png)

The other repositories are here for a purpose:
* `tinyml-modelmaker`: Based on user configuration (yaml files), stitches a flow with relevant scripts to call from tinyverse. This stitches the scripts into a flow of data loading/training/compilation
  * **_This is your home repo. Most of your work will be taking place from this directory._**
* `tinyml-tinyverse` : Individual scripts to load data, do preprocessing, AI training, compilation(using NNC/TVM)
* `tinyml-modeloptimization`: Model optimization toolkit that is necessary for quantization for 2bit/4bit/8bit weights in QAT(Quantization Aware Training)/PTQ(Post Training Quantization) flows for TI devices with or without NPU.
  * As a customer developing models/flows, it is highly likely that you would not have to edit files in this repo
* `tinyml-mlbackend` (TI internal only): Serves as a wrapper on modelmaker to suit the needs of Edge AI Studio Model Composer only. Docker image is generated using this repo.

<hr>

## Which Repo do I need to fiddle with

| User Intent | Criteria 1                                                                                                                                                             | Criteria 2                                                                           | tinyml-modelmaker                                                                                                                                | tinyml-tinyverse | tinyml-modeloptimization                                                                                                                                                                                                                                                                                                                                                                                         |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BYOD        | <ul><li> I will get my own data (format it according to Modelmaker Expectations)</li><li> I expect a compiled binary model as my final output for a device TI supports | <ul><li> I will use TI provided Models as it is                                      | :white_check_mark:  - edit config_*.yaml files. refer [this](./tinyml-modelmaker/docs/UnderstandingConfigFile.md) to understand the config file. | :x:             | :white_check_mark:                                                                                                                                                                                                                                                                                                                                                                                                             |
| BYOD        | <ul><li> I will get my own data (format it according to Modelmaker Expectations)</li><li> I expect a compiled binary model as my final output for a device TI supports | <ul><li> I will use TI provided Models but change the parameters (like channels etc) | :white_check_mark:  - Refer [this](./tinyml-modelmaker/docs/Configuring_Model_layer_params.md) doc                                               | :x:             | :x:                                                                                                                                                                                                                                                                                                                                                                                                             |
| BYOM        | <ul><li> I will get my own data (format it according to Modelmaker Expectations)</li><li> I expect a compiled binary model as my final output for a device TI supports | <ul><li> I want to design my own models / Modify TI provided models.                 | :white_check_mark:  - Refer [this](./tinyml-modelmaker/docs/AddingModels.md) doc                                                                 | :white_check_mark:            | :x:                                                                                                                                                                                                                                                                                                                                                                                                             |
| BYOM        | <ul><li> I have trained an (NPU compatible, if applicable) model and created an onnx model. </li>                                                                      | <ul><li>I need help to compile for a TI supported device.                            | :white_check_mark:  - Refer [this](./tinyml-modelmaker/docs/BYOM_for_Compilation.md) doc to understand editing the config file                                     | :x:             | :x:                                                                                                                                                                                                                                                                                                                                                                                                             |
| BYOM        | <ul><li> I have my own AI Training Framework, I have created a floating point model. </li>                                                                             | <ul><li>I need help to create a NPU Aware Quantized model                            | :x:                                                                                                                                              | :x:             | <input type="checkbox" disabled checked/> - Refer [this](https://github.com/TexasInstruments/tinyml-tensorlab/blob/main/tinyml-modeloptimization/torchmodelopt/examples/motor_fault_time_series_classification) example.<br/> <ul><li> Follow it up  using [TIs Neural Network Compiler](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/) which can help you get your AI model compatible with TI MCUs. |

* A lot more [READMEs](./tinyml-modelmaker/docs) are present under the Tiny ML Modelmaker Repo

<hr>

# Using this repository

To begin with, you can use the repo as a `developer` or `user`.

  ### Prerequisite:

* <details> 
  <summary> Python Environment  </summary>
  
  * **Note**: Irrespective of being a `Linux` or a `Windows` user, it is ideal to use virtual environments on Python rather than operating without one. 
    * For `Linux` we are using `Pyenv` as a Python version management system.
    * For `Windows` we show below using pyenv-win and also using Python's native `venv`
    
  * <details> 
    <summary> Linux OS </summary>
  
      #### Using Pyenv-Linux (Recommended)
      * Follow https://github.com/pyenv/pyenv?tab=readme-ov-file#a-getting-pyenv to install pyenv 
      * Use Python 3.10.xx
      * `pyenv local <python_version>` is recommended. The version given will be used whenever python is called from within this folder. 

      </details>

  * <details> 
    <summary> Windows OS </summary>
    
      #### Using Pyenv-Win (Recommended)
      * Follow steps 1-5 from here using any Python3.10.xx: https://github.com/pyenv-win/pyenv-win?tab=readme-ov-file#quick-start
      * Instead of step 6, `pyenv local <python_version>` is recommended. The version given will be used whenever python is called from within this folder.
      
      #### Using Python venv
      
      * Install Python3.10 from https://www.python.org/downloads/
      
      ```commandline
      python -m venv py310
      .\py310\Scripts\activate
      ```
      </details>

</details>

* **NOTE: C2000 Customers:**
  * Please download and install [TI C2000 Codegen Tools (TI C2000 CGT)](https://www.ti.com/tool/download/C2000-CGT)
    * Please set the installed path in your terminal:
      * Linux: `export C2000_CG_ROOT="/path/to/ti-cgt-c2000_22.6.1.LTS"`
      * Windows: `$env:C2000_CG_ROOT="C:\path\to\wherever\present\ti-cgt-c2000_22.6.1.LTS"`
  * Please download and install [C2000Ware](https://www.ti.com/tool/C2000WARE)
    * Please set the installed path in your terminal:
      * Linux: `export C2000WARE_ROOT="/path/to/C2000Ware_6_00_00_00"`
      * Windows: `$env:C2000WARE_ROOT="C:\path\to\wherever\present\C2000Ware_6_00_00_00\"`
        * Default: `$env:C2000WARE_ROOT="C:\ti\c2000\C2000Ware_6_00_00_00"`
  
* **NOTE: MSPM0 Customers:**
  * Please download and install [TI Arm Codegen Tools (TI Arm CGT Clang)](https://www.ti.com/tool/download/ARM-CGT-CLANG)
    * Please set the installed path in your terminal:
      * Linux: `export MSPM0_C2000_CG_ROOT="/path/to/ti-cgt-armllvm_4.0.3.LTS"`
      * Windows: `$env:MSPM0_C2000_CG_ROOT="C:\path\to\wherever\present\ti-cgt-armllvm_4.0.3.LTS"`
  * Please download and install [MSPM0 SDK](https://www.ti.com/tool/MSPM0-SDK)
    * Please set the installed path in your terminal:
      * Linux: `export M0SDK_PATH="/path/to/mspm0_sdk_2_05_00_05"`
      * Windows: `$env:M0SDK_PATH="C:\path\to\wherever\present\mspm0_sdk_2_05_00_05\"`


* ## I'm a User:
  * As a `user` - The installation and usage is very simple. It is just a `pip install`. But beware that you will not be able to modify any of the features or customize AI models/transforms for your use case
  * <details>

    Install this repository as a Python package:
    ```
    pip install git+https://github.com/TexasInstruments/tinyml-tensorlab.git@main#subdirectory=tinyml-modelmaker
    ```
    It is as simple as:
    ```python
    import tinyml_modelmaker
    tinyml_modelmaker.get_set_go(config)
    ```
    ### Note
    * Several examples of configs are present (Check *.yaml files at `tinyml-modelmaker` repository)
      * You can load one like this:
    * ```python
      import yaml  
      with open('examples/dc_arc_fault/config_dsk.yaml') as fp:
          config = yaml.safe_load(fp)
      ```
    ### Important
    * This method still expects the C2000Ware/MSPM0 SDK to be installed by the user separately and is not automatically installed.
    * This method still expects the TI-CGT/ TI Arm-Clang to be installed by the user separately and is not automatically installed.
    * __Proceeding without installing these SDKs will result in a trained model for the dataset, but will not compile the ONNX model to an compiled artifact.__
        
    </details>

* ## I'm a developer
  * As a `developer` - The installation will use your brain power (although a tiny bit), but allows you to customize with unimaginable power!
  * <details>
    <summary> Linux OS</summary>

    ### 1. Set up TI tinyml-tensorlab
    #### Steps to set up the repositories

    1. Clone this repository
    2. `cd tinyml-tensorlab/tinyml-modelmaker`
    3. Execute: ``` ./setup_all.sh ```
    4. Run the following (to install local repositories, ideal for developers): 
        ```bash
        cd ../tinyml-tinyverse
        pip install -e .
        cd tinyml-modeloptimization/torchmodelopt
        pip install -e .
        cd ../tinyml-modelmaker
        ```
    5. Now you're ready to go!
    ```
    run_tinyml_modelmaker.sh examples/dc_arc_fault/config_dsk.yaml
    ```
    </details>
    
    <details>
    <summary> Windows OS</summary>
  
    #### This repository can be used from native Windows terminal directly.
    * Although we use Pyenv for Python version management on Linux, the same offering for Windows isn't so stable. So even the native venv is good enough.
      * **It is highly recommended to use PowerShell instead of cmd.exe/Command Terminal**
      * If you prefer to use Windows Subsystem for Linux, then a [user guide](./tinyml-modelmaker/docs/Windows_Subsytem_for_Linux.md) to use this toolchain on Windows Subsystem for Linux has been provided.
    
    *  Step 1.1: Clone this repository from GitHub
    *  Step 1.2: Let us ready up the depedencies
    ```powershell
    cd tinyml-tensorlab
    python -m ensurepip --upgrade
    python -m pip install --no-input --upgrade pip setuptools wheel
       ```
    *  Step 1.3: Install Tiny ML Modelmaker
    ```powershell
    git config "--global" core.longpaths true
    cd .\tinyml-modelmaker
    python -m pip install --editable . # --use-pep517
    ```
    * Tiny ML Modelmaker, by default installs Tiny ML Tinyverse and Tiny ML ModelOptimization repositories as a python package.
      * If you intend to use this repository as is, then it is enough.
      * However, if you intend to create models and play with the quantization varieties, then it is better to separately clone
      * Step 1.4: Installing tinyverse
        ```powershell
        cd ..\tinyml-tinyverse
        python -m pip install --editable .
        ```
      * Step 1.5: Installing model optimization toolkit
        ```powershell
        cd ..\tinyml-modeloptimization\torchmodelopt
        python -m pip install --editable .
        ```
    * We can run it now!
    ```powershell
    cd ..\..\tinyml-modelmaker
    python .\tinyml_modelmaker\run_tinyml_modelmaker.py .\examples\dc_arc_fault\config_dsk.yaml
    ```
    
</details>
  
  

* #### Keeping up to date
    
  Since these repositories are undergoing a massive feature addition stage, it is recommended to keep your codes up to date by running the following command:
  ```commandline
  git_pull_all.sh
  ```
    

<hr>

## Productized applications:
<details>

| Application (Title)           | Select sector (Industrial, automotive, personal electronics) | Technology  | Application Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Features / advantages                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|-------------------------------|--------------------------------------------------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Arc fault detection           | Industrial                                                   | Time series | An arc fault is an electrical discharge that occurs when an electrical current flows through an unintended path, often due to damaged, frayed, or improperly installed wiring. This can produce heat and sparks, which can ignite surrounding materials, leading to fires.<br><br>Due to the ability of AI to analyze complex patterns, continuously learn and improve from new data, address a wide range of faults, it is advantageous to use AI. Using AI at the edge empowers the customer with reduced latency, enhanced privacy and scalability while saving bandwidth.<br><br>TI provides UL-1699B tested AI models which have impeccable accuracy and ultra low latency.                                                     | By utilising benefits of AI such as its ability to analyze patterns in signals and ability to handle larger volumes of data, TI's solution allows for immediate detection response to arc faults.<br><br>Coupled with an NPU that provides enhanced AI performance, TI's� brings additional benefits in terms of speed, reliability, and scalability, making it a powerful approach for enhancing electrical safety.<br><br>With TI's complete solution, AFD will never be a showstopper for you.                               |
| Motor Bearing Fault Detection | Industrial                                                   | Time series | Motor bearing faults are often seen in HVAC systems with rotating parts. It occurs due to the wear and tear of moving parts, lack of lubrication, and due to overloading of equipment. It adversely affects the motor lifespan and increases energy consumption, potentially even can cause a failure of the system.<br><br>By using AI, these faults can be detected early by monitoring signs such as subtle changes in vibration patterns. Processing such data locally at the HVAC system can provide real time fault detection and immediate response, which is crucial for preventing damage and ensuring continuous operation.<br><br>TI provides handcrafted AI models which have impeccable accuracy and ultra low latency. | TI's AI solution addresses these by monitoring the vibration and temperature of the motor through sensors and provides a reliable solution by combining the strengths of advanced analytics and real-time processing, leading to more reliable, efficient, and cost-effective maintenance and operation.<br><br>Put together with an NPU for advanced AI performance capabilities, this prevents unexpected failures as the algorithms can detect early signs of faults that might not be noticeable through manual inspections |
| Blower Imbalance Detection    | Industrial                                                   | Time series | Often seen in blowers over time, the blades accumulate dust and particulate matter which clogs and causes some of the blades to be heavier than the other and ultimately resulting in increased energy consumption that is undesirable.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | TI's AI solution addresses these by monitoring the current passing to the blower.                                                                                                                                                                                                                                                                                                                                                                                                                                               |

* To empower your solution with TIs AI, you can use the Model Composer GUI to quickly train an AI model or use the Tiny ML Modelmaker for an advanced set of capabilities. To customers who rely on their own AI training framework, TIs Neural Network Compiler can help you get your AI model compatible with MCUs (P55x,P66x or any other F28 device).

</details>

* To empower your solution with TIs AI, you can use the **[Tiny ML Modelmaker](./tinyml-modelmaker)** for an advanced set of capabilities.
  * Supports any Time series Classification tasks (including Arc Fault and Motor Bearing Fault Classification)
* You can also use the [Edge AI Studio Model Composer GUI](https://dev.ti.com/modelcomposer/) to quickly train an AI model (No Code Platform)
  * This supports only Arc Fault and Motor Bearing Fault Classification applications currently.
* To customers who rely on their own AI training framework, [TIs Neural Network Compiler](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/) can help you get your AI model compatible with MCUs (P55x,P66x or any other F28 device).
* For a full-fledged reference solution on Arc Fault and Motor Bearing Fault, find the comprehensive project in [Digital Power SDK](https://www.ti.com/tool/C2000WARE-DIGITALPOWER-SDK) and [Motor Control SDK](https://www.ti.com/tool/C2000WARE-MOTORCONTROL-SDK).

<hr>

## What is New

- [2025-Oct] Release version 1.2.0 of the software

    <details>

    - Device Support:
      - Added MSPM0 based MCUs: MSPM0G3507, MSPM0G5187
      - Added Connectivity device: CC2745R10-Q1, CC2755R10
    - General:
        - Supports simple gain augmentation for classification tasks
        - Prints dataset file level confusion matrix for classification tasks       
        - Golden Test Vectors for Regression tasks
        - Run modelmaker with only the config, no more target device required in the input. 
    - Flows:
        - Timeseries Forecasting flows supported
        - L1, L2 normalization can be enabled in regression using lambda_reg param
    - Model Optimization:
        - How to use: Documentation updated.
        - Example code for performing regression in modeloptimization
        - Fixing clipping of input data to int8 or uint8 based on dataset (zero_point) (only the input zero point is fixed and not the intermediate layers)
        - BatchNorm is supported by GENERIC quantization
        - Experimental features like additional QDQ at input of model and floating bias can be enabled individually
        - Residual Add supported for different scales, zero points, but not optimised for TINPU
      
  </details>
    
- [2025-Aug] Release version 1.1.0 of the software
    
  <details>

  - General:
    - Generic Timeseries Classification is available with fixed point reference dataset.
    - Compatible with C2000Ware 6.0.0
  - Model Optimization:
    - Aggressive Quantization Modes for Weights & Activation: 2W8A, 4W4A, 4W8A --> massive speedup and memory saved
    - Neural network Architecture Search for generating a TINPU compatible model directly based on user's dataset
  - Dataset:
    - Dataset can be split into train-test-val on a file-by-file basis or within-a-file basis
  - Device Support:
    - Full Support for F280013x
    - Preliminary Support for F29H85x and MSPM0G3507x
  - Compilation:
    - Upgraded TI MCU Neural Network Compiler for MCUs to 2.0.0
  - Windows Platform Specific:
    - Major quantization accuracy improvements  
  - Miscellaneous:
    - Fixed model performance data that appears on the terminal when a training is initiated
    - Added Model Descriptions for all models
    - Setup of the repos is now smoother and cleaner

  </details>
- [2025-Apr] Major feature updates (version 1.0.0) of the software
  <details>

  - General:
    - Tiny ML Modelmaker is now a pip installable package!
    - Existing models can be modified on the fly through a config file (check Tiny ML Modelmaker docs)
    - MPS (Metal Performance Shaders) backend support for Mac host devices!
  - Technology:
    - PTQ and QAT flows supported in tinyml-modelmaker, tinyml-modeloptimization
    - Ternary, 4 bit Quantization support in tinyml-modelmaker
  - Flows:
    - Regression ML tasks supported
    - Autoencoder based Anomaly Detection task supported
  - Feature Extraction:
    - Feature Extraction transforms are now modular and compatible with C2000Ware 5.05 only
    - Supports Haar and Hadamard Transform
    - Golden test vectors file has one set uncommented by default to work OOB
  - Data Visualisation:
    - Multiclass ROC-AUC graphs are autogenerated for better explainability of reports and help select thresholds based on false alarm/ sensitivity preference
    - PCA graphs are auto plotted for feature extracted data � Helps in identifying if the feature extraction actually helped
    - Run now begins with displaying inference time, sram usage and flash usage for all the devices for any model.
  - Dataset
    - Goodness of Fit of dataset now enabled.
  - Extensive Documentation & Know-How Examples to use Modelmaker
  
  </details>
- [2024-November] Updated (version 0.9.0) of the software
- [2024-August] Release version 0.8.0 of the software
- [2024-July] Release version 0.7.0 of the software
- [2024-June] Release version 0.6.0 of the software
- [2024-May] First release (version 0.5.0) of the software

