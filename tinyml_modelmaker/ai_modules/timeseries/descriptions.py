#################################################################################
# Copyright (c) 2023-2024, Texas Instruments
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

import numbers

from ... import utils, version
from . import constants, training


def _get_paretto_front_best(xy_list, x_index=0, y_index=1, inverse_relaionship=False):
    xy_list = sorted(
        xy_list, key=lambda x: x[x_index], reverse=inverse_relaionship)
    paretto_front = [xy_list[0]]
    for xy in xy_list[1:]:
        if xy[y_index] >= paretto_front[-1][y_index]:
            paretto_front.append(xy)
        #
    #
    # sort based on first index - reverse order in inference time is ascending order in FPS (faster performance)
    paretto_front = sorted(
        paretto_front, key=lambda x: x[x_index], reverse=True)
    return paretto_front


def _get_paretto_front_approx(xy_list, x_index=0, y_index=1, inverse_relaionship=False):
    # normalize the values
    min_x = min(xy[0] for xy in xy_list)
    max_x = max(xy[0] for xy in xy_list)
    min_y = min(xy[1] for xy in xy_list)
    max_y = max(xy[1] for xy in xy_list)
    norm_xy_list = [[(xy[0] - min_x + 1) / (max_x - min_x + 1), (xy[1] - min_y + 1) / (max_y - min_y + 1), xy[2]] for xy
                    in xy_list]
    if inverse_relaionship:
        efficiency_list = [list(xy) + [xy[y_index] * xy[x_index]]
                           for xy in norm_xy_list]
    else:
        efficiency_list = [list(xy) + [xy[y_index] / xy[x_index]]
                           for xy in norm_xy_list]
    #
    efficiency_list = sorted(
        efficiency_list, key=lambda x: x[-1], reverse=True)
    # take the good models
    num_models_selected = max(len(efficiency_list) * 2 // 3, 1)
    efficiency_list = efficiency_list[:num_models_selected]
    selected_indices = [xy[2] for xy in efficiency_list]
    selected_entries = [xy for xy in xy_list if xy[2] in selected_indices]
    # sort based on first index - reverse order in inference time is ascending order in FPS (faster performance)
    paretto_front = sorted(
        selected_entries, key=lambda x: x[x_index], reverse=True)
    return paretto_front


def get_paretto_front_combined(xy_list, x_index=0, y_index=1, inverse_relaionship=False):
    paretto_front_best = _get_paretto_front_best(xy_list, x_index=x_index, y_index=y_index,
                                                 inverse_relaionship=inverse_relaionship)
    paretto_front_approx = _get_paretto_front_approx(xy_list, x_index=x_index, y_index=y_index,
                                                     inverse_relaionship=inverse_relaionship)
    paretto_front_combined = paretto_front_best + paretto_front_approx
    # de-duplicate
    selected_indices = [xy[2] for xy in paretto_front_combined]
    selected_indices = set(selected_indices)
    paretto_front = [xy for xy in xy_list if xy[2] in selected_indices]
    # sort based on first index - reverse order in inference time is ascending order in FPS (faster performance)
    paretto_front = sorted(
        paretto_front, key=lambda x: x[x_index], reverse=True)
    return paretto_front


def set_default_inference_time_us(model_descriptions):
    for m in model_descriptions.values():
        for target_device in m.training.target_devices.keys():
            if not m.training.target_devices[target_device].get('inference_time_us'):
                m.training.target_devices[target_device].inference_time_us = "TBD"


def set_default_sram(model_descriptions):
    for m in model_descriptions.values():
        for target_device in m.training.target_devices.keys():
            if not m.training.target_devices[target_device].get('sram'):
                m.training.target_devices[target_device].sram = "TBD"


def set_default_flash(model_descriptions):
    for m in model_descriptions.values():
        for target_device in m.training.target_devices.keys():
            if not m.training.target_devices[target_device].get('flash'):
                m.training.target_devices[target_device].flash = "TBD"


def set_model_selection_factor(model_descriptions):
    # for m in model_descriptions.values():
    #     for target_device in m.training.target_devices.keys():
    #         m.training.target_devices[target_device].model_selection_factor = None
    task_types = set()
    for m in model_descriptions.values():
        if isinstance(m.common.task_type, list):
            task_types.update(m.common.task_type)
        else:
            task_types.add(m.common.task_type)
    target_devices = [list(m.training.target_devices.keys())
                      for m in model_descriptions.values()]
    target_devices = set([t for t_list in target_devices for t in t_list])
    for target_device in target_devices:
        for task_type in task_types:
            model_desc_list = [
                m for m in model_descriptions.values() if task_type in m.common.task_type]
            model_desc_list = [m for m in model_desc_list if target_device in list(
                m.training.target_devices.keys())]
            inference_time_us = [m.training.target_devices[target_device].inference_time_us for m in
                                 model_desc_list]
            # accuracy_factor = [m.training.target_devices[target_device].accuracy_factor for m in model_desc_list]
            accuracy_factor = ['TBD' for _ in model_desc_list]
            xy_list = [(inference_time_us[i], accuracy_factor[i], i) for i in
                       range(len(inference_time_us))]
            xy_list_shortlisted = [(xy[0], xy[1], xy[2]) for xy in xy_list if
                                   isinstance(xy[0], numbers.Real) and isinstance(xy[1], numbers.Real)]
            # if no models have performance data for this device, then use some dummy data
            if not xy_list_shortlisted:
                xy_list_shortlisted = [(1, 1, xy[2]) for xy in xy_list]
            #
            if len(xy_list_shortlisted) > 0:
                xy_list_shortlisted = get_paretto_front_combined(
                    xy_list_shortlisted)
                for paretto_id, xy in enumerate(xy_list_shortlisted):
                    xy_id = xy[2]
                    m = model_desc_list[xy_id]
                    if m.training.target_devices[target_device].model_selection_factor is None:
                        m.training.target_devices[target_device].model_selection_factor = paretto_id
                    #
                #
            #
        #
    #


def get_training_module_descriptions(params):
    # populate a good pretrained model for the given task
    training_module_descriptions = training.get_training_module_descriptions(target_device=params.common.target_device,
                                                                             training_device=params.training.training_device)
    #
    training_module_descriptions = utils.ConfigDict(
        training_module_descriptions)
    return training_module_descriptions


def get_model_descriptions(params):
    # populate a good pretrained model for the given task
    model_descriptions = training.get_model_descriptions(task_type=params.common.task_type,
                                                         target_device=params.common.target_device,
                                                         training_device=params.training.training_device)

    #
    model_descriptions = utils.ConfigDict(model_descriptions)
    set_default_inference_time_us(model_descriptions)
    set_default_sram(model_descriptions)
    set_default_flash(model_descriptions)
    set_model_selection_factor(model_descriptions)

    return model_descriptions


def get_model_description(model_name):
    assert model_name, 'model_name must be specified for get_model_description().' \
                       'if model_name is not known, use the method get_model_descriptions() that returns supported models.'
    model_description = training.get_model_description(model_name)
    return model_description


def set_model_description(params, model_description):
    assert model_description is not None, f'could not find pretrained model for {params.training.model_name}'
    assert params.common.task_type == model_description['common']['task_type'], \
        f'task_type: {params.common.task_type} does not match the pretrained model'
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
    return constants.TARGET_DEVICE_DESCRIPTIONS


def get_sample_dataset_descriptions(params):
    return constants.SAMPLE_DATASET_DESCRIPTIONS


def get_task_descriptions(params):
    return constants.TASK_DESCRIPTIONS


def get_version_descriptions(params):
    version_descriptions = {
        'version': version.get_version(),
        # 'sdk_version': constants.TARGET_SDK_VERSION_C2000,
        # 'sdk_release': constants.TARGET_SDK_RELEASE_C2000,
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

### {constants.TARGET_DEVICE_F28P55}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_F28P55}

### {constants.TARGET_DEVICE_F28P65}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_F28P65}

### {constants.TARGET_DEVICE_F2837}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_F2837}

### {constants.TARGET_DEVICE_F28004}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_F28004}

### {constants.TARGET_DEVICE_F28003}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_F28003}

### {constants.TARGET_DEVICE_F280013}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_F280013}

### {constants.TARGET_DEVICE_F280015}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_F280015}

### {constants.TARGET_DEVICE_MSPM0G3507}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_MSPM0G3507}

### {constants.TARGET_DEVICE_MSPM0G3519}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_MSPM0G3519}

### {constants.TARGET_DEVICE_MSPM0G5187}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_MSPM0G5187}

### {constants.TARGET_DEVICE_CC2755}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_CC2755}

### {constants.TARGET_DEVICE_CC1352}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_CC1352}


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
- Look at the example dataset [Arc Fault Classification](https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/arc_fault_classification_dsk.zip) to understand further.
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


def get_live_capture_descriptions(params):
    live_capture_descriptions = {
        'a': {
            'context': {
                'task_type': ['arc_fault']
            },
            'defaultValues': {
                'acqMode': 0,
                'device': 0,
                'samples': 1024,
                'samplingFrequency': 200000
            },
         'propInfo': [
             {
                 'caption': 'device',
                 'id': 'device',
                 'infoText': 'Current support is F28P55. You can try other device using correct application example and baud rate.',
                 'options': ['F28P55', 'MSPM0G5187'],
                 'widgetType': 'select'
             },
             {
                 'caption': 'sensor',
                 'id': 'sensor',
                 'infoText': 'Connect hardware to detect sensors',
                 'widgetType': 'select'
             },
             {
                 'caption': 'capture signal mode',
                 'id': 'acqMode',
                 'infoText': 'Select Time Domain Current to capture time domain analog signal. Select Time Domain Current and Label to capture both time domain analog and arc labeling input signals.',
                 'options': [
                     'Time Domain Current',
                     'Time Domain Current and Label'],
                 'widgetType': 'select'
             },
             {
                 'caption': 'sampling frequency',
                 'format': 'dec',
                 'id': 'samplingFrequency',
                 'minValue': 1,
                 'precision': 0,
                 'widgetType': 'input'
             },
             {
                 'caption': 'samples',
                 'format': 'dec',
                 'id': 'samples',
                 'infoText': 'This number will be rounded up to a multiple of 1024.',
                 'minValue': 1024,
                 'multiplesOf': 1024,
                 'precision': 0,
                 'widgetType': 'input'
             }
         ]
        },
        'b': {
            'context': {
                'task_type': ['motor_fault', 'blower_imbalance']},
            'defaultValues': { 'device': 0,
                            'labelName': '',
                            'labelNumber': 0,
                            'motorName': 'motor',
                            'motorNumber': 0,
                            'samplingTime': 4},
            'propInfo': [
                {
                    'caption': 'device',
                    'id': 'device',
                    'infoText': 'Current support is F28P55. You can try other device using correct application example and baud rate.',
                    'options': ['F28P55', 'MSPM0G5187'],
                    'widgetType': 'select'
                },
                {
                    'caption': 'sensor',
                    'id': 'sensor',
                    'infoText': 'Connect hardware to detect sensors',
                    'widgetType': 'select'
                },
                {
                    'caption': 'continuous measurement time (s)',
                    'format': 'dec',
                    'id': 'samplingTime',
                    'minValue': 1,
                    'precision': 0,
                    'widgetType': 'input'
                },
                {
                    'as': {'kind': 'filename', 'order': 1},
                    'caption': 'device under test name',
                    'format': 'text',
                    'id': 'motorName',
                    'infoText': 'This name is used as part of a filename for saving captured data',
                    'placeholder': 'Enter name',
                    'widgetType': 'input'
                },
                {
                    'as': {'kind': 'filename', 'order': 2, 'prefix': 'm'},
                    'caption': 'device under test number',
                    'format': 'dec',
                    'id': 'motorNumber',
                    'infoText': 'This number is used as part of a filename for saving captured data',
                    'placeholder': 'Enter number',
                    'widgetType': 'input'
                },
                {
                    'as': {'kind': 'filename', 'order': 4},
                    'caption': 'label description',
                    'format': 'text',
                    'id': 'labelName',
                    'infoText': 'This name is used as part of a filename for saving captured data',
                    'optional': True,
                    'placeholder': 'Enter description',
                    'widgetType': 'input'},
                {
                    'as': {'kind': 'filename', 'order': 3, 'prefix': 'label'},
                    'caption': 'label number',
                    'format': 'dec',
                    'id': 'labelNumber',
                    'infoText': 'This number is used as part of a filename for saving captured data',
                    'placeholder': 'Enter label number',
                    'widgetType': 'input'}
            ]
        },
        'c': {
            'context': {'task_type': ['generic_timeseries_classification']},
            'defaultValues': {
                'device': 0,
                'labelName': 'Sine',
                'samples': 256,
                'samplingFrequency': 5000,
                },
            'propInfo': [
                {
                    'caption': 'device',
                    'id': 'device',
                    'infoText': 'Current support is MSPM0G3507. You can try other device using correct application example and baud rate.',
                    'options': ['MSPM0G3507', 'MSPM0G3519', 'MSPM0G5187'],
                    'widgetType': 'select'},
                {
                    'caption': 'sensor',
                    'id': 'sensor',
                    'infoText': 'Connect hardware to detect sensors',
                    'widgetType': 'select'},
                {
                    'as': {'kind': 'filename', 'order': 1},
                    'caption': 'label description',
                    'format': 'text',
                    'id': 'labelName',
                    'infoText': 'This name is used as part of a filename for saving captured data',
                    'optional': True,
                    'placeholder': 'Enter description',
                    'widgetType': 'input'},
                {
                    'caption': 'samples',
                    'format': 'dec',
                    'id': 'samples',
                    'infoText': 'This number will be rounded up to a multiple of 256.',
                    'minValue': 256,
                    'multiplesOf': 256,
                    'precision': 0,
                    'widgetType': 'input'}
            ]
        },
        'd': {
            'context': {'task_type': ['pir_detection']},
            'defaultValues': {
                'device': 0,
                'samples': 125,
                'samplingFrequency': 33,
                },
            'propInfo': [
                {
                    'caption': 'device',
                    'id': 'device',
                    'infoText': 'Current support is CC2755R10 and CC1352R1. You can try other device using correct application example and baud rate.',
                    'options': ['CC2755', 'CC1352'],
                    'widgetType': 'select'},
                {
                    'caption': 'sensor',
                    'id': 'sensor',
                    'infoText': 'Connect hardware to detect sensors',
                    'widgetType': 'select'},
                {
                    'caption': 'sampling frequency',
                    'format': 'dec',
                    'id': 'samplingFrequency',
                    'minValue': 25,
                    'precision': 0,
                    'widgetType': 'input'},
                {
                    'caption': 'samples',
                    'format': 'dec',
                    'id': 'samples',
                    'infoText': 'This number will be rounded up to a multiple of 125.',
                    'minValue': 125,
                    'multiplesOf': 125,
                    'precision': 0,
                    'widgetType': 'input'}
            ]
        },

    }
    return live_capture_descriptions


def get_live_capture_example_descriptions(params):
    live_capture_example_descriptions = {
        'arc_fault': {
            'F28P55': {
                'ccsProj': 'ml_arc_detection_F28P55x',
                'deviceName': 'TMS320F28P550SJ9',
                'files': [],
                'from': 'solutions/tida_010955/f28p55x/ccs/arc_detection_f28p55x.projectspec',
                'pkgId': 'digital_power_c2000ware_sdk_software_package',
                'targetCfg': 'targetConfigs/TMS320F28P550SJ9.ccxml',
                'transport': {'baudRate': 4687500}
            },
            'MSPM0G5187': {
                'ccsProj': 'ac_arc_fault_data_capture_LP_MSPM0G5187_nortos_ticlang',
                'deviceName': 'MSPM0G5187',
                'files': [],
                'from': 'examples/nortos/LP_MSPM0G5187/edgeAI/ac_arc_fault_data_capture/ticlang/ac_arc_fault_data_capture_LP_MSPM0G5187_nortos_ticlang.projectspec',
                'pkgId': 'MSPM0-SDK',
                'targetCfg': 'targetConfigs/MSPM0G5187.ccxml',
                'transport': {'baudRate': 5820000}
            }
        },
        'generic_timeseries_classification': {
            'MSPM0G5187': {
                'ccsProj': 'timeseries_data_capture_LP_MSPM0G5187_nortos_ticlang',
                'deviceName': 'MSPM0G5187',
                'files': [],
                'from': 'examples/nortos/LP_MSPM0G5187/edgeAI/timeseries_data_capture/ticlang/timeseries_data_capture_LP_MSPM0G5187_nortos_ticlang.projectspec',
                'pkgId': 'MSPM0-SDK',
                'targetCfg': 'targetConfigs/MSPM0G5187.ccxml',
                'transport': {'baudRate': 115200}
            }
        },
        'ecg_classification': {
            'MSPM0G5187': {
                'ccsProj': 'ecg_anomaly_detection_data_capture_LP_MSPM0G5187_nortos_ticlang',
                'deviceName': 'MSPM0G5187',
                'files': [],
                'from': 'examples/nortos/LP_MSPM0G5187/edgeAI/ecg_anomaly_detection_data_capture/ticlang/ecg_anomaly_detection_data_capture_LP_MSPM0G5187_nortos_ticlang.projectspec',
                'pkgId': 'MSPM0-SDK',
                'targetCfg': 'targetConfigs/MSPM0G5187.ccxml',
                'transport': {'baudRate': 115200}
            }
        },
        'motor_fault': {
            'F28P55': {
                'ccsProj': 'eAI_data_acq_dap_f28p55x',
                'deviceName': 'TMS320F28P550SJ9',
                'files': [],
                'from': 'solutions/edge_ai_fault_detection_with_mc/data_collection_preparation/ccs/eAI_data_acq_dap_f28p55x.projectspec',
                'pkgId': 'motor_control_c2000ware_sdk_software_package',
                'targetCfg': 'targetConfigs/TMS320F28P550SJ9_LaunchPad.ccxml',
                'transport': {'baudRate': 2343750}
            },
            'MSPM0G5187': {
                'ccsProj': 'motor_fault_data_capture_LP_MSPM0G5187_nortos_ticlang',
                'deviceName': 'MSPM0G5187',
                'files': [],
                'from': 'examples/nortos/LP_MSPM0G5187/edgeAI/motor_fault_data_capture/ticlang/motor_fault_data_capture_LP_MSPM0G5187_nortos_ticlang.projectspec',
                'pkgId': 'MSPM0-SDK',
                'targetCfg': 'targetConfigs/MSPM0G5187.ccxml',
                'transport': {'baudRate': 115200}
            }
        },
        'pir_detection': {
            'CC2755': {
                'ccsProj': 'edgeai_smart_pir_detection_LP_EM_CC2745R10_Q1_freertos_ticlang',
                'deviceName': 'CC2745R10',
                'files': [],
                'from': 'examples/rtos/LP_EM_CC2745R10_Q1/edgeai/edgeai_smart_pir_detection/freertos/ticlang/edgeai_smart_pir_detection_LP_EM_CC2745R10_Q1_freertos_ticlang.projectspec',
                'pkgId': 'SIMPLELINK-SDK-EDGEAI-PLUGIN',
                'targetCfg': 'targetConfigs/CC2745R10.ccxml',
                'transport': {'baudRate': 115200}
            },
            'CC1352': {
                'ccsProj': 'edgeai_smart_pir_detection_cc1352_CC1352R1_LAUNCHXL_freertos_ticlang',
                'deviceName': 'CC1352R1F3',
                'files': [],
                'from': 'examples/rtos/CC1352R1_LAUNCHXL/edgeai/edgeai_smart_pir_detection/freertos/ticlang/edgeai_smart_pir_detection_cc1352_CC1352R1_LAUNCHXL_freertos_ticlang.projectspec',
                'pkgId': 'SIMPLELINK-SDK-EDGEAI-PLUGIN',
                'targetCfg': 'targetConfigs/CC1352R1F3.ccxml',
                'transport': {'baudRate': 115200}
            },
               'MSPM0G5187': {
                'ccsProj': 'pir_detection_data_capture_LP_MSPM0G5187_nortos_ticlang',
                'deviceName': 'MSPM0G5187',
                'files': [],
                'from': 'examples/nortos/LP_MSPM0G5187/edgeAI/pir_detection_data_capture/ticlang/pir_detection_data_capture_LP_MSPM0G5187_nortos_ticlang.projectspec',
                'pkgId': 'MSPM0-SDK',
                'targetCfg': 'targetConfigs/MSPM0G5187.ccxml',
                'transport': {'baudRate': 115200}
            }
        }
        
    }
    return live_capture_example_descriptions


def get_live_preview_descriptions(params):
    live_preview_descriptions = {
        'a': {
            'context': {
                'task_type': ['arc_fault']},
            'defaultValues': {'samples': 1024, 'samplingFrequency': 200000},
            'propInfo': [
                {
                    'caption': 'sensor',
                    'id': 'sensor',
                    'infoText': 'Connect hardware to detect sensors',
                    'widgetType': 'select'
                },
                {
                    'caption': 'sampling frequency',
                    'format': 'dec',
                    'id': 'samplingFrequency',
                    'minValue': 1,
                    'precision': 0,
                    'widgetType': 'input'
                },
                {
                    'caption': 'samples',
                    'format': 'dec',
                    'id': 'samples',
                    'infoText': 'This number will be rounded up to a multiple of 1024.',
                    'minValue': 1024,
                    'multiplesOf': 1024,
                    'precision': 0,
                    'widgetType': 'input'
                }
            ]
        },
        'b': {
            'context': {
                'task_type': [ 'motor_fault', 'blower_imbalance', 'generic_timeseries_classification', 'pir_detection']},
            'defaultValues': {},
            'propInfo': [
                {
                    'caption': 'sensor',
                    'id': 'sensor',
                    'infoText': 'Connect hardware to detect sensors',
                    'widgetType': 'select'
                }
            ]
        }
    }
    return live_preview_descriptions


def get_live_preview_example_descriptions(params):
    live_preview_example_descriptions = {
        'arc_fault': {
            'F28P55': {
                'ccsProj': 'ml_arc_detection_F28P55x',
                'deviceName': 'TMS320F28P550SJ9',
                'files': [{'from': 'artifacts/', 'to': 'arc_model'},
                          {'from': 'golden_vectors/user_input_config.h', 'to': ''},
                          {'from': 'model_aux.h', 'to': ''}],
                'from': 'solutions/tida_010955/f28p55x/ccs/arc_detection_f28p55x.projectspec',
                'pkgId': 'digital_power_c2000ware_sdk_software_package',
                'targetCfg': 'targetConfigs/TMS320F28P550SJ9.ccxml',
                'transport': {'baudRate': 4687500}
            },
            'MSPM0G5187': {
                'ccsProj': 'ac_arc_fault_detection_live_preview_LP_MSPM0G5187_nortos_ticlang',
                'deviceName': 'MSPM0G5187',
                'files': [{'from': 'artifacts/mod.a',
                'to': 'model/model.a'},
                {'from': 'artifacts/tvmgen_default.h', 'to': 'model/tvmgen_default.h'},
                {'from': 'golden_vectors/user_input_config.h', 'to': 'model/user_input_config.h'}],
                'from': 'examples/nortos/LP_MSPM0G5187/edgeAI/ac_arc_fault_detection_live_preview/ticlang/ac_arc_fault_detection_live_preview_LP_MSPM0G5187_nortos_ticlang.projectspec',
                'pkgId': 'MSPM0-SDK',
                'targetCfg': 'targetConfigs/MSPM0G5187.ccxml',
                'transport': {'baudRate': 115200}
            }
        },
        'generic_timeseries_classification': {
            'MSPM0G5187': {
                'ccsProj': 'timeseries_live_preview_LP_MSPM0G5187_nortos_ticlang',
                'deviceName': 'MSPM0G5187',
                'files': [{'from': 'artifacts/mod.a',
                'to': 'model/model.a'},
                {'from': 'artifacts/tvmgen_default.h', 'to': 'model/tvmgen_default.h'},
                {'from': 'golden_vectors/user_input_config.h', 'to': 'model/user_input_config.h'}],
                'from': 'examples/nortos/LP_MSPM0G5187/edgeAI/timeseries_live_preview/ticlang/timeseries_live_preview_LP_MSPM0G5187_nortos_ticlang.projectspec',
                'pkgId': 'MSPM0-SDK',
                'targetCfg': 'targetConfigs/MSPM0G5187.ccxml',
                'transport': {'baudRate': 115200}
            }
        },
        'ecg_classification': {
            'MSPM0G5187': {
                'ccsProj': 'ecg_anomaly_detection_live_preview_LP_MSPM0G5187_nortos_ticlang',
                'deviceName': 'MSPM0G5187',
                'files': [{'from': 'artifacts/mod.a',
                'to': 'model/model.a'},
                {'from': 'artifacts/tvmgen_default.h', 'to': 'model/tvmgen_default.h'},
                {'from': 'golden_vectors/user_input_config.h', 'to': 'model/user_input_config.h'}],
                'from': 'examples/nortos/LP_MSPM0G5187/edgeAI/ecg_anomaly_detection_live_preview/ticlang/ecg_anomaly_detection_live_preview_LP_MSPM0G5187_nortos_ticlang.projectspec',
                'pkgId': 'MSPM0-SDK',
                'targetCfg': 'targetConfigs/MSPM0G5187.ccxml',
                'transport': {'baudRate': 115200}
            }
        },
        'motor_fault': {
            'F28P55': {
                'ccsProj': 'eAI_mfd_eval_f28p55x',
                'deviceName': 'TMS320F28P550SJ9',
                'files': [{'from': 'artifacts/', 'to': 'AI_artifacts'},
                       {'from': 'golden_vectors/user_input_config.h', 'to': ''},
                       {'from': 'model_aux.h', 'to': ''}],
                'from': 'solutions/edge_ai_fault_detection_with_mc/motor_fault_livepreview_validation_f28p55x/ccs/eAI_mfd_eval_f28p55x.projectspec',
                'pkgId': 'motor_control_c2000ware_sdk_software_package',
                'targetCfg': 'targetConfigs/TMS320F28P550SJ9_LaunchPad.ccxml',
                'transport': {'baudRate': 2343750}
            },
            'MSPM0G5187': {
                'ccsProj': 'motor_fault_live_preview_LP_MSPM0G5187_nortos_ticlang',
                'deviceName': 'MSPM0G5187',
                'files': [{'from': 'artifacts/mod.a',
                'to': 'model/model.a'},
                {'from': 'artifacts/tvmgen_default.h', 'to': 'model/tvmgen_default.h'},
                {'from': 'golden_vectors/user_input_config.h', 'to': 'model/user_input_config.h'}],
                'from': 'examples/nortos/LP_MSPM0G5187/edgeAI/motor_fault_live_preview/ticlang/motor_fault_live_preview_LP_MSPM0G5187_nortos_ticlang.projectspec',
                'pkgId': 'MSPM0-SDK',
                'targetCfg': 'targetConfigs/MSPM0G5187.ccxml', 
                'transport': {'baudRate': 115200}
            }
        },
        'pir_detection': {
            'CC2755': {
                'ccsProj': 'edgeai_smart_pir_detection_LP_EM_CC2745R10_Q1_freertos_ticlang',
                'deviceName': 'CC2745R10',
                'files': [{'from': 'artifacts/', 'to': 'arc_model'}],
                'from': 'examples/rtos/LP_EM_CC2745R10_Q1/edgeai/edgeai_smart_pir_detection/freertos/ticlang/edgeai_smart_pir_detection_LP_EM_CC2745R10_Q1_freertos_ticlang.projectspec',
                'pkgId': 'SIMPLELINK-SDK-EDGEAI-PLUGIN',
                'targetCfg': 'targetConfigs/CC2745R10.ccxml',
                'transport': {'baudRate': 115200}
            },
            'CC1352': {
                'ccsProj': 'edgeai_smart_pir_detection_cc1352_CC1352R1_LAUNCHXL_freertos_ticlang',
                'deviceName': 'CC1352R1F3',
                'files': [{'from': 'artifacts/', 'to': 'arc_model'}],
                'from': 'examples/rtos/CC1352R1_LAUNCHXL/edgeai/edgeai_smart_pir_detection/freertos/ticlang/edgeai_smart_pir_detection_cc1352_CC1352R1_LAUNCHXL_freertos_ticlang.projectspec',
                'pkgId': 'SIMPLELINK-SDK-EDGEAI-PLUGIN',
                'targetCfg': 'targetConfigs/CC1352R1F3.ccxml',
                'transport': {'baudRate': 115200}
            },
            'MSPM0G5187': {
                'ccsProj': 'pir_detection_live_preview_LP_MSPM0G5187_nortos_ticlang',
                'deviceName': 'MSPM0G5187',
                'files': [{'from': 'artifacts/mod.a',
                'to': 'model/model.a'},
                {'from': 'artifacts/tvmgen_default.h', 'to': 'model/tvmgen_default.h'},
                {'from': 'golden_vectors/user_input_config.h', 'to': 'model/user_input_config.h'}],
                'from': 'examples/nortos/LP_MSPM0G5187/edgeAI/pir_detection_live_preview/ticlang/pir_detection_live_preview_LP_MSPM0G5187_nortos_ticlang.projectspec',
                'pkgId': 'MSPM0-SDK',
                'targetCfg': 'targetConfigs/MSPM0G5187.ccxml', 
                'transport': {'baudRate': 115200}
            }
        }
    }
    return live_preview_example_descriptions


def get_context_help_descriptions(params):
    context_help_descriptions = {
        'annotate': {
            'mce_demo_task_1_default_annotate': {
                'context': {'task_type': ['mce_demo_task_1']},
                'help_url': 'file://gettingStarted/annotate.md'}
        },
        'capture': {
            'mce_demo_task_1_default_capture': {
                'context': {'task_type': ['mce_demo_task_1']},
                'help_url': 'file://gettingStarted/capture.md'}
        },
        'capture_visualization': {
            'mce_demo_task_1_default_visualization': {
                'context': {'task_type': ['mce_demo_task_1']},
                'help_url': 'https://dev.ti.com/tirex/explore'}
        },
        'compile': {
            'default_compile_all': {
                'context': {'task_type': ['mce_demo_task_1', 'mce_demo_task_2', 'mce_demo_task_3']},
                'help_url': 'file://gettingStarted/compile_all.md'},
            'mce_demo_task_1_MCEDemo_Device_2_compile_1': {
                'context': {'devices': ['F28P55', 'MCEDemo_Device_2'],
                            'task_type': ['arc_fault', 'generic_timeseries', 'mce_demo_task_1']},
                'help_url': 'https://dev.ti.com/'},
            'mce_demo_task_1_MCEDemo_Device_2_compile_2': {
                'context': {'devices': ['F28P55', 'MCEDemo_Device_2'],
                            'task_type': ['mce_demo_task_1']},
                'help_url': 'file://gettingStarted/compile_device2.md'},
            'unsupported_device_compile': {
                'context': {'devices': ['F28P55'],
                            'task_type': ['arc_fault', 'generic_timeseries', 'mce_demo_task_1']},
                'help_url': 'file://gettingStarted/compile_f28p55.md'},
            'unsupported_task_compile': {
                'context': {'task_type': ['motor_fault']},
                'help_url': 'file://gettingStarted/should_not_be_included.md'}
        },
        'deploy': {
            'mce_demo_task_1_default_deploy': {
                'context': {'task_type': ['mce_demo_task_1']},
                'help_url': 'file://gettingStarted/deploy.md'}
        },
        'livedemo': {
            'mce_demo_task_1_default_livedemo': {
                'context': {'task_type': ['mce_demo_task_1']},
                'help_url': 'file://gettingStarted/livedemo.md'}
        },
        'train': {
            'mce_demo_task_1_default_train': {
                'context': {'task_type': ['mce_demo_task_1']},
                'help_url': 'file://gettingStarted/train.md'}
        }
    }
    return context_help_descriptions


def get_help_url_descriptions(params):
    help_url_descriptions = "file://help.md"
    return help_url_descriptions