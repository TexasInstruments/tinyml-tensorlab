#################################################################################
# Copyright (c) 2023-2026, Texas Instruments
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
#################################################################################

from ... import utils, version
from . import constants, training


def get_model_descriptions(params):
    # populate a good pretrained model for the given task
    model_descriptions = training.get_model_descriptions(task_type=params.common.task_type,
                                                         target_device=params.common.target_device,
                                                         training_device=params.training.training_device)

    #
    model_descriptions = utils.ConfigDict(model_descriptions)

    return model_descriptions


def get_model_description(model_name):
    if not model_name:
        raise ValueError(
            'model_name must be specified for get_model_description(). '
            'If model_name is not known, use get_model_descriptions() that returns supported models.')
    model_description = training.get_model_description(model_name)
    return model_description


def set_model_description(params, model_description):
    if model_description is None:
        raise ValueError(f'could not find pretrained model for {params.training.model_name}')
    if params.common.task_type != model_description['common']['task_type']:
        raise ValueError(
            f'task_type: {params.common.task_type} does not match the pretrained model')
    # get pretrained model checkpoint and other details
    params.update(model_description)
    return params


def get_preset_descriptions(params):
    return constants.PRESET_DESCRIPTIONS


def get_feature_extraction_preset_descriptions(params):
    return constants.FEATURE_EXTRACTION_PRESET_DESCRIPTIONS


def get_dataset_preset_descriptions(params):
    return constants.DATASET_EXAMPLES


def get_preset_compilations(params):
    return constants.PRESET_COMPILATIONS


def get_target_device_descriptions(params):
    return constants.TARGET_DEVICES


def get_sample_dataset_descriptions(params):
    return constants.SAMPLE_DATASET_DESCRIPTIONS


def get_task_descriptions(params):
    return constants.TASK_DESCRIPTIONS


def get_version_descriptions(params):
    version_descriptions = {
        'version': version.get_version(),
    }
    return version_descriptions


def get_tooltip_descriptions(params):
    return {
        'common': {
        },
        'dataset': {
        },
        'training': {
            'training_epochs': {
                'name': 'Epochs',
                'description': 'Epoch is a term that is used to indicate a pass over the entire training dataset. '
                               'It is a hyper parameter that can be tuned to get best accuracy. '
                               'Eg. A model trained for 30 Epochs may give better accuracy than a model trained for 15 Epochs.'
            },
            'learning_rate': {
                'name': 'Learning rate',
                'description': 'Learning Rate determines the step size used by the optimization algorithm '
                               'at each iteration while moving towards the optimal solution. '
                               'It is a hyper parameter that can be tuned to get best accuracy. '
                               'Eg. A small Learning Rate typically gives good accuracy while fine tuning a model for a different task.'
            },
            'batch_size': {
                'name': 'Batch size',
                'description': 'Batch size specifies the number of inputs that are propagated through the '
                               'neural network in one iteration. Several such iterations make up one Epoch.'
                               'Higher batch size require higher memory and too low batch size can '
                               'typically impact the accuracy.'
            },
            'weight_decay': {
                'name': 'Weight decay',
                'description': 'Weight decay is a regularization technique that can improve '
                               'stability and generalization of a machine learning algorithm. '
                               'It is typically done using L2 regularization that penalizes parameters '
                               '(weights, biases) according to their L2 norm.'
            },
            'early_stopping': {
                'name': 'Early Stopping',
                'description': 'Stop training automatically once the validation metric stops improving. '
                               'This helps avoid overfitting and can reduce unnecessary training time.'
            },
            'early_stopping_patience': {
                'name': 'Early Stopping Patience',
                'description': 'Number of epochs with no improvement before stopping early. '
                               'It is a hyper parameter that can be tuned along with Early Stopping.'
            },
        },
        'compilation': {
            'preset_name': {
                'name': 'Preset Name',
                'description': 'Two presets exist: "default_preset"(Recommended Option), '
                               '"forced_soft_npu_preset"(Only available on HW-NPU devices to disable HW NPU), '
            },
        },
        'deploy': {
            'download_trained_model_to_pc': {
                'name': 'Download trained model',
                'description': 'Trained model can be downloaded to the PC for inspection.'
            },
            'download_compiled_model_to_pc': {
                'name': 'Download compiled model artifacts to PC',
                'description': 'Compiled model can be downloaded to the PC for inspection.'
            },
            'download_compiled_model_to_evm': {
                'name': 'Download compiled model artifacts to EVM',
                'description': 'Compiled model can be downloaded into the EVM for running model inference in SDK. Instructions are given in the help section.'
            }
        }
    }


def get_help_descriptions(params):
    tooltip_descriptions = get_tooltip_descriptions(params)

    tooltip_string = ''
    for tooltip_section_key, tooltip_section_dict in tooltip_descriptions.items():
        if tooltip_section_dict:
            tooltip_string += f'\n### {tooltip_section_key.upper()}'
            for tooltip_key, tooltip_dict in tooltip_section_dict.items():
                tooltip_string += f'\n#### {tooltip_dict["name"]}'
                tooltip_string += f'\n{tooltip_dict["description"]}'
            #
        #
    #

    # removed_from_help_string_under_tasks_supported  "* {constants.TASK_DESCRIPTIONS[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION]['task_name']}"

    help_string = f'''
## Overview
This is a tool for collecting data, training and compiling AI models for use on TI's embedded microcontrollers. The compiled models can be deployed on a local development board. A live preview/demo will also be provided to inspect the quality of the developed model while it runs on the development board.

## Development flow
Bring your own data (BYOD): Retrain models from TI Model Zoo to fine-tune with your own data.

## Tasks supported
* {constants.TASK_DESCRIPTIONS[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION]['task_name']}
* {constants.TASK_DESCRIPTIONS[constants.TASK_TYPE_ARC_FAULT]['task_name']}
* {constants.TASK_DESCRIPTIONS[constants.TASK_TYPE_ECG_CLASSIFICATION]['task_name']}
* {constants.TASK_DESCRIPTIONS[constants.TASK_TYPE_MOTOR_FAULT]['task_name']}
* {constants.TASK_DESCRIPTIONS[constants.TASK_TYPE_BLOWER_IMBALANCE]['task_name']}
* {constants.TASK_DESCRIPTIONS[constants.TASK_TYPE_PIR_DETECTION]['task_name']}

## Supported target devices
These are the devices that are supported currently. As additional devices are supported, this section will be updated.

Supported devices: {', '.join(constants.TARGET_DEVICES)}


## Additional information
{constants.TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}

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
- Look at the example dataset [Arc Fault Classification](https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/arc_fault_classification_dsk.zip) to understand further.
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
{tooltip_string}
'''
    return help_string
