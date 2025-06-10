
## Overview
This is a tool for collecting data, training and compiling AI models for use on TI's embedded microcontrollers. The compiled models can be deployed on a local development board. A live preview/demo will also be provided to inspect the quality of the developed model while it runs on the development board.

## Development flow
Bring your own data (BYOD): Retrain models from TI Model Zoo to fine-tune with your own data.

## Tasks supported
* Generic Time Series Classification
* ARC Fault
* Motor Fault
* Fan Blower Imbalance Fault

## Supported target devices
These are the devices that are supported currently. As additional devices are supported, this section will be updated.

### F28P55
* Product information: https://www.ti.com/product/TMS320F28P550SJ
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F28P55X
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: 05_04_00

### F28P65
* Product information: https://www.ti.com/product/TMS320F28P650DK
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F28P65X
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: 05_04_00

### F2837
* Product information: https://www.ti.com/product/TMS320F28377D
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F28379D
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: 05_04_00

### F28004
* Product information: https://www.ti.com/product/TMS320F280049C
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F280049C
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: 05_04_00

### F28003
* Product information: https://www.ti.com/product/TMS320F280039C
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F280039C
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: 05_04_00

### F280015
* Product information: https://www.ti.com/product/TMS320F2800157
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F2800157
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: 05_04_00

### MSPM0G3507
* Product information: https://www.ti.com/product/MSPM0G3507
* Launchpad: https://www.ti.com/tool/LP-MSPM0G3507
* MSPM0 SDK: https://www.ti.com/tool/MSPM0-SDK
* SDK release: 2_05_00_05

## Additional information



## Dataset format
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
                                   |--file_list.txt
                                   |--instances_train_list.txt
                                   |--instances_val_list.txt
                                   |--instances_test_list.txt
</pre>

- Use a suitable dataset name instead of dataset_name
- Look at the example dataset [Arc Fault Classification](http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/arc_fault_classification_dsk.zip) to understand further.
- In the config file, provide the name of the dataset (dataset_name in this example) in the field dataset_name and provide the path or URL in the field input_data_path.
- Then the ModelMaker tool can be invoked with the config file.


#### Notes
If the dataset has already been split into train and validation set already, it is possible to provide those paths separately as a tuple in input_data_path.
After the model compilation, the compiled models will be available in a folder inside [./data/projects](./data/projects)
The config file can be in .yaml or in .json format

## Model deployment
- The deploy page provides a button to download the compiled model artifacts to the development board. 
- The downloaded model artifacts are located in a folder inside /opt/projects. It can be used with the SDK to run inference. 
- Please see "C2000Ware Reference Design" in the SDK documentation for more information.

## Glossary of terms

### TRAINING
#### Epochs
Epoch is a term that is used to indicate a pass over the entire training dataset. It is a hyper parameter that can be tuned to get best accuracy. Eg. A model trained for 30 Epochs may give better accuracy than a model trained for 15 Epochs.
#### Learning rate
Learning Rate determines the step size used by the optimization algorithm at each iteration while moving towards the optimal solution. It is a hyper parameter that can be tuned to get best accuracy. Eg. A small Learning Rate typically gives good accuracy while fine tuning a model for a different task.
#### Batch size
Batch size specifies the number of inputs that are propagated through the neural network in one iteration. Several such iterations make up one Epoch.Higher batch size require higher memory and too low batch size can typically impact the accuracy.
#### Weight decay
Weight decay is a regularization technique that can improve stability and generalization of a machine learning algorithm. It is typically done using L2 regularization that penalizes parameters (weights, biases) according to their L2 norm.
### COMPILATION
#### Preset Name
Two presets exist: "default_preset"(Recommended Option), "forced_soft_npu_preset"(Only available on HW-NPU devices to disable HW NPU), 
### DEPLOY
#### Download trained model
Trained model can be downloaded to the PC for inspection.
#### Download compiled model artifacts to PC
Compiled model can be downloaded to the PC for inspection.
#### Download compiled model artifacts to EVM
Compiled model can be downloaded into the EVM for running model inference in SDK. Instructions are given in the help section.
