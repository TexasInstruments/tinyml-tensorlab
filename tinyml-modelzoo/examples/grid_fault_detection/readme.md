# Single phase grid fault detection for on-board chargers
### - Bhanu Vankayalapati, Adithya Thonse
<hr>

## Overview

Nearly all EVs/PHEvs come with an on-board charger(OBC) to charge the high voltage battery. The on-board charger is expected to be robust to any fault or abnormal conditions of the AC grid. Due to the nature of the AC grid, faults may have highly variable signature and therefore are not easy to detect with traditional threshold based heuristic criteria. Using edge-AI model running on the same MCU that controls the OBC power-stage, it is possible to protect the OBC against adverse grid events, log abnormal events, and potentially safe the OBC.

<p align="center">  
    <img src="assets/grid_fault_variation.png" width="500" alt="Simulink Model">
</p>

TI's innovative approach leverages a Convolutional Neural Network (CNN) edge-AI model trained on a propietary grid-fault dataset, running seamlessly on the F29x MCU. This technology enables more accurate and reliable grid fault detection in on-board charging applications.

## Feature Extraction

Given that there are no well defined fault categories for AC grid faults, it is important to define fault categories that have meanigful feature separation while also being physically meaningful in their adverse impact on the OBC operation. For this, we use a hybrid dataset annotation technique that leverages a combination of human annotation and unsupervised annotation using heirarchichal denisty based clustering algorithm. The goodness of annotation is evaluated using dimensionality reduction techniques with maunal QC. At this time, the feature extraction capabilities of model maker are bypassed for this example with plans to evetually enable it.

## Downloading dataset

The zipped example dataset is provided in this project folder `grid_fault_dataset.zip`. 
The path of this zipped dataset file is already mentioned in [configuration](config.yaml) yaml, make sure it is same.

```yaml
dataset:
    input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/grid_fault_dataset.zip
```

## Usage in Tiny ML ModelZoo

This zipped dataset is designed to work with Tiny ML ModelMaker. Run through ModelZoo with the yaml [configuration](config.yaml) using the below code.

```bash
run_tinyml_modelzoo.sh examples/grid_fault_detection/config.yaml  # Use .\run_tinyml_modelzoo.bat for Windows Terminal/Powershell
```

1. `run_tinyml_modelzoo.sh` is the script to run ModelZoo. "run_tinyml_modelzoo.bat" for Windows Terminal/Powershell
2. `examples/grid_fault_detection/config.yaml` path of the yaml configuration to run

The users can configure the yaml [configuration](config.yaml) to change parameters related to training, testing, model and model compilation. In this example, we will configure the parameters of feature extraction. Since the feature extraction step is handled outside the Tiny ML model maker tool, we set feature extraction to

```yaml
data_processing_feature_extraction:
  data_proc_transforms: ['SimpleWindow']
  frame_size: 16  # The dataset here has only 16 values per file, so do not provide a number more than 16
  stride_size: 1
  variables: 1  # Indicates 1 channel (column) of data, i.e: current
```

After doing the above changes in yaml [configuration](config.yaml) file. Run the ModelZoo again for this dataset.

```bash
run_tinyml_modelzoo.sh examples/electrical_fault/config.yaml  # Use .\run_tinyml_modelzoo.bat for Windows Terminal/Powershell
```

Steps to run this example on-device can be found by following this guide: [Deploying Classification Models from ModelMaker to Device](../../docs/deploying_classification_models_from_modelmaker_to_device/readme.md).
<hr>
