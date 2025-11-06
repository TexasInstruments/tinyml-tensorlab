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
import os

# plugins/additional models
# see the setup_all.sh file to understand how to set this
PLUGINS_ENABLE_GPL = False
PLUGINS_ENABLE_EXTRA = False

# task_type
TASK_TYPE_MOTOR_FAULT = 'motor_fault'
TASK_TYPE_ARC_FAULT = 'arc_fault'
TASK_TYPE_BLOWER_IMBALANCE = 'blower_imbalance'
TASK_TYPE_PIR_DETECTION = 'pir_detection'
TASK_TYPE_GENERIC_TS_CLASSIFICATION = 'generic_timeseries_classification'
TASK_TYPE_GENERIC_TS_REGRESSION = 'generic_timeseries_regression'
TASK_TYPE_GENERIC_TS_ANOMALYDETECTION = 'generic_timeseries_anomalydetection'
TASK_TYPE_GENERIC_TS_FORECASTING= 'generic_timeseries_forecasting'

TASK_TYPES = [
    TASK_TYPE_ARC_FAULT,
    TASK_TYPE_MOTOR_FAULT,
    TASK_TYPE_BLOWER_IMBALANCE,
    TASK_TYPE_GENERIC_TS_CLASSIFICATION,
    TASK_TYPE_GENERIC_TS_REGRESSION,
    TASK_TYPE_GENERIC_TS_ANOMALYDETECTION,
    TASK_TYPE_GENERIC_TS_FORECASTING,
    TASK_TYPE_PIR_DETECTION,
]

# task_category
TASK_CATEGORY_TS_CLASSIFICATION = 'timeseries_classification'
TASK_CATEGORY_TS_REGRESSION = 'timeseries_regression'
TASK_CATEGORY_TS_ANOMALYDETECTION = 'timeseries_anomalydetection'
TASK_CATEGORY_TS_FORECASTING = 'timeseries_forecasting'


TASK_CATEGORIES = [
    TASK_CATEGORY_TS_CLASSIFICATION, TASK_CATEGORY_TS_REGRESSION,TASK_CATEGORY_TS_FORECASTING,
]

# target_device
TARGET_DEVICE_AM263 = 'AM263'
TARGET_DEVICE_F280013 = 'F280013'
TARGET_DEVICE_F280015 = 'F280015'
TARGET_DEVICE_F28003 = 'F28003'
TARGET_DEVICE_F28004 = 'F28004'
TARGET_DEVICE_F2837 = 'F2837'
TARGET_DEVICE_F28P55 = 'F28P55'
TARGET_DEVICE_F28P65 = 'F28P65'
TARGET_DEVICE_F29H85 = 'F29H85'
TARGET_DEVICE_MSPM0G3507 = 'MSPM0G3507'
TARGET_DEVICE_MSPM0G5187 = 'MSPM0G5187'
TARGET_DEVICE_CC2755 = 'CC2755'

TARGET_DEVICES = [
    TARGET_DEVICE_F280013,
    TARGET_DEVICE_F280015,
    TARGET_DEVICE_F28003,
    TARGET_DEVICE_F28004,
    TARGET_DEVICE_F2837,
    TARGET_DEVICE_F28P55,
    TARGET_DEVICE_F28P65,
    TARGET_DEVICE_MSPM0G3507,
    TARGET_DEVICE_MSPM0G5187,
    TARGET_DEVICE_CC2755,
]

# will not be listed in the GUI, but can be used in command line
TARGET_DEVICES_ADDITIONAL = [
    TARGET_DEVICE_AM263,
    TARGET_DEVICE_F29H85,
]

# include additional devices that are not currently supported in release.
TARGET_DEVICES_ALL = TARGET_DEVICES + TARGET_DEVICES_ADDITIONAL

TARGET_DEVICE_TYPE_MCU = 'MCU'

TARGET_DEVICE_TYPES = [
    TARGET_DEVICE_TYPE_MCU
]

# training_device
TRAINING_DEVICE_CPU = 'cpu'
TRAINING_DEVICE_CUDA = 'cuda'
TRAINING_DEVICE_MPS = 'mps'
TRAINING_DEVICE_GPU = TRAINING_DEVICE_CUDA

TRAINING_DEVICES = [
    TRAINING_DEVICE_CPU,
    TRAINING_DEVICE_CUDA,
    TRAINING_DEVICE_MPS
]

TRAINING_BATCH_SIZE_DEFAULT = {
    TASK_TYPE_ARC_FAULT: 32,
    TASK_TYPE_MOTOR_FAULT: 256,
    TASK_TYPE_BLOWER_IMBALANCE: 256,
    TASK_TYPE_GENERIC_TS_CLASSIFICATION: 64,
    TASK_TYPE_GENERIC_TS_REGRESSION: 64,
    TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: 64,
    TASK_TYPE_GENERIC_TS_FORECASTING: 64,
    TASK_TYPE_PIR_DETECTION: 64,
}

TARGET_SDK_VERSION_C2000 = '6.0'
TARGET_SDK_RELEASE_C2000 = '06_00_00'

TARGET_SDK_VERSION_F29H85 = '1.00'
TARGET_SDK_RELEASE_F29H85 = '01_00_00'

TARGET_SDK_VERSION_MSPM0 = "2.05.00.05"
TARGET_SDK_RELEASE_MSPM0 = '2_05_00_05'

TARGET_SDK_VERSION_CC2755 = '9.12.00.00'
TARGET_SDK_RELEASE_CC2755 = '09_12_00_00'


TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION = '\n'
# TODO: Once the links are up add this
'''
* Tiny ML model development information: https://github.com/TexasInstruments/tinyml-tensorlab
'''

##### AM263 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_AM263 = \
    f'''* Product information: https://www.ti.com/product/AM2634
* Development board: https://www.ti.com/tool/LP-AM263
* SDK: https://www.ti.com/tool/MCU-PLUS-SDK-AM263X
* SDK release: {TARGET_SDK_RELEASE_C2000}'''

TARGET_DEVICE_DETAILS_AM263 = \
    f'''Quad-core Arm® Cortex®-R5F MCU up to 400 MHz with real-time control and security
* More details : https://www.ti.com/product/AM2634

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_AM263}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

##### F280015 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_F280015 = \
    f'''* Product information: https://www.ti.com/product/TMS320F2800157
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F2800157
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: {TARGET_SDK_RELEASE_C2000}'''

TARGET_DEVICE_DETAILS_F280015 = \
    f'''C2000™ 32-bit MCU 120-MHz 384-KB flash, FPU, TMU with CLA, CLB, AES and CAN-FD
* More details : https://www.ti.com/tool/LAUNCHXL-F2800157

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_F280015}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

##### F280013 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_F280013 = \
    f'''* Product information: https://www.ti.com/product/TMS320F2800137
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F2800137
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: {TARGET_SDK_RELEASE_C2000}'''

TARGET_DEVICE_DETAILS_F280013 = \
    f'''C2000™ 120-MHz C28x CPU with FPU and TMU, 256-KB Flash, two 12-bit ADCs, 14 PWM channels, CAN (DCAN), one encoder module (eQEP), UART, and more
* More details : https://www.ti.com/tool/LAUNCHXL-F2800137

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_F280013}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

##### F28003 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_F28003 = \
    f'''* Product information: https://www.ti.com/product/TMS320F280039C
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F280039C
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: {TARGET_SDK_RELEASE_C2000}'''

TARGET_DEVICE_DETAILS_F28003 = \
    f'''C2000™ 32-bit MCU 120-MHz 384-KB flash, FPU, TMU with CLA, CLB, AES and CAN-FD
* More details : https://www.ti.com/product/TMS320F280039C

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_F28003}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

##### F28004 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_F28004 = \
    f'''* Product information: https://www.ti.com/product/TMS320F280049C
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F280049C
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: {TARGET_SDK_RELEASE_C2000}'''

TARGET_DEVICE_DETAILS_F28004 = \
    f'''C2000™ 32-bit MCU 120-MHz 384-KB flash, FPU, TMU with CLA, CLB, AES and CAN-FD
* More details : https://www.ti.com/product/TMS320F280049C

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_F28004}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

##### F2837 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_F2837 = \
    f'''* Product information: https://www.ti.com/product/TMS320F28377D
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F28379D
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: {TARGET_SDK_RELEASE_C2000}'''

TARGET_DEVICE_DETAILS_F2837 = \
    f'''C2000™ 32-bit MCU with 800 MIPS, 2xCPU, 2xCLA, FPU, TMU, 1024 KB flash, EMIF, 16b ADC
* More details : https://www.ti.com/product/TMS320F28377D

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_F2837}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

##### F28P65 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_F28P65 = \
    f'''* Product information: https://www.ti.com/product/TMS320F28P650DK
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F28P65X
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: {TARGET_SDK_RELEASE_C2000}'''

TARGET_DEVICE_DETAILS_F28P65 = \
    f'''C2000™ 32-bit MCU, 2x C28x+CLA CPU, Lock Step, 1.28-MB flash, 16-b ADC, HRPWM, EtherCAT, CAN-FD, AES
* More details : https://www.ti.com/product/TMS320F28P650DK

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_F28P65}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

##### F28P55 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_F28P55 = \
    f'''* Product information: https://www.ti.com/product/TMS320F28P550SJ
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F28P55X
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: {TARGET_SDK_RELEASE_C2000}'''

TARGET_DEVICE_DETAILS_F28P55 = \
    f'''C2000™ 32-bit MCU, 1x C28x + 1x CLA, 150-MHz, 1.1-MB flash, 5x ADCs, CLB, AES and NNPU
* More details : https://www.ti.com/product/TMS320F28P550SJ

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_F28P55}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

##### F29H85 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_F29H85 = \
    f'''* Product information: https://www.ti.com/product/F29H850TU
* SOM EVM: https://www.ti.com/tool/F29H85X-SOM-EVM
* C2000 SDK: https://www.ti.com/tool/download/F29H85X-SDK/
* SDK release: {TARGET_SDK_RELEASE_F29H85}'''

TARGET_DEVICE_DETAILS_F29H85 = \
    f'''C2000™ 64-bit MCU with C29x 200MHz tri-core, lockstep, functional safety compliance, 4MB
* More details : https://www.ti.com/product/F29H850TU

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_F29H85}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

##### M0G3507 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_MSPM0G3507 = \
    f'''* Product information: https://www.ti.com/product/MSPM0G3507
* Launchpad: https://www.ti.com/tool/LP-MSPM0G3507
* MSPM0 SDK: https://www.ti.com/tool/MSPM0-SDK
* SDK release: {TARGET_SDK_RELEASE_MSPM0}'''

TARGET_DEVICE_DETAILS_MSPM0G3507= \
    f'''80MHz Arm® Cortex®-M0+ MCU with 128KB flash 32KB SRAM 2x4Msps ADC, DAC, 3xCOMP, 2xOPA, CAN-FD, MATHA
* More details : https://www.ti.com/product/MSPM0G3507

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_MSPM0G3507}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

##### M0G5187 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_MSPM0G5187 = \
    f'''* Product information: https://www.ti.com/product/MSPM0G5187
* Launchpad: https://www.ti.com/tool/LP-MSPM0G5187
* MSPM0 SDK: https://www.ti.com/tool/MSPM0-SDK
* SDK release: {TARGET_SDK_RELEASE_MSPM0}'''

TARGET_DEVICE_DETAILS_MSPM0G5187= \
    f'''80MHz Arm® Cortex®-M0+ MCU with 128KB flash 32KB SRAM 2x4Msps ADC, DAC, 3xCOMP, 2xOPA, CAN-FD, MATHA
* More details : https://www.ti.com/product/MSPM0G5187

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_MSPM0G5187}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

##### CC2755 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_CC2755 = \
    f'''* Product information: https://www.ti.com/product/CC2755R10
* Launchpad: https://www.ti.com/tool/LP-EM-CC2745R10-Q1
* CC2755 SDK: http://tgrex10.toro.design.ti.com/tirex4-handoff/explore/node?node=A__AEIJm0rwIeU.2P1OBWwlaA__SIMPLELINK-SDK-EDGEAI-PLUGIN__Nz0hc8j__LATEST&placeholder=true
* SDK release: {TARGET_SDK_RELEASE_CC2755}'''

TARGET_DEVICE_DETAILS_CC2755= \
    f'''96MHz SimpleLink™ 32-bit Arm® Cortex®-M33 multiprotocol wireless MCU with 1MB flash, HSM, APU, NPU-CDE
* More details : https://www.ti.com/product/CC2755R10

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_CC2755}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''


# higher device_selection_factor indicates higher performance device.
TARGET_DEVICE_DESCRIPTIONS = {
    TARGET_DEVICE_F280013: {
        'device_name': TARGET_DEVICE_F280013,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 0,
        'device_details': TARGET_DEVICE_DETAILS_F280013,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_F280015: {
        'device_name': TARGET_DEVICE_F280015,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 1,
        'device_details': TARGET_DEVICE_DETAILS_F280015,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_F28003: {
        'device_name': TARGET_DEVICE_F28003,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 2,
        'device_details': TARGET_DEVICE_DETAILS_F28003,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_F28004: {
        'device_name': TARGET_DEVICE_F28004,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 3,
        'device_details': TARGET_DEVICE_DETAILS_F28004,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_F2837: {
        'device_name': TARGET_DEVICE_F2837,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 4,
        'device_details': TARGET_DEVICE_DETAILS_F2837,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_F28P65: {
        'device_name': TARGET_DEVICE_F28P65,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 5,
        'device_details': TARGET_DEVICE_DETAILS_F28P65,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_F28P55: {
        'device_name': TARGET_DEVICE_F28P55,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 6,
        'device_details': TARGET_DEVICE_DETAILS_F28P55,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_AM263: {
        'device_name': TARGET_DEVICE_AM263,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 7,
        'device_details': TARGET_DEVICE_DETAILS_AM263,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_F29H85: {
        'device_name': TARGET_DEVICE_F29H85,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 8,
        'device_details': TARGET_DEVICE_DETAILS_F29H85,
        'sdk_version': TARGET_SDK_VERSION_F29H85,
        'sdk_release': TARGET_SDK_RELEASE_F29H85,
    },
    TARGET_DEVICE_MSPM0G3507: {
        'device_name': TARGET_DEVICE_MSPM0G3507,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 9,
        'device_details': TARGET_DEVICE_DETAILS_MSPM0G3507,
        'sdk_version': TARGET_SDK_VERSION_MSPM0,
        'sdk_release': TARGET_SDK_RELEASE_MSPM0,
    },
    TARGET_DEVICE_MSPM0G5187: {
        'device_name': TARGET_DEVICE_MSPM0G5187,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 10,
        'device_details': TARGET_DEVICE_DETAILS_MSPM0G5187,
        'sdk_version': TARGET_SDK_VERSION_MSPM0,
        'sdk_release': TARGET_SDK_RELEASE_MSPM0,
    },
    TARGET_DEVICE_CC2755: {
        'device_name': TARGET_DEVICE_CC2755,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 11,
        'device_details': TARGET_DEVICE_DETAILS_CC2755,
        'sdk_version': TARGET_SDK_VERSION_CC2755,
        'sdk_release': TARGET_SDK_RELEASE_CC2755,
    },
}

TASK_DESCRIPTIONS = {
    TASK_TYPE_ARC_FAULT: {
        'task_name': 'ARC Fault',
        'target_module': 'timeseries',
        'target_devices': TARGET_DEVICES,
        'stages': ['dataset', 'data_processing_feature_extraction', 'training', 'compilation'],
    },

    TASK_TYPE_MOTOR_FAULT: {
        'task_name': 'Motor Fault',
        'target_module': 'timeseries',
        'target_devices': TARGET_DEVICES,
        'stages': ['dataset', 'data_processing_feature_extraction', 'training', 'compilation'],
    },
    TASK_TYPE_BLOWER_IMBALANCE: {
        'task_name': 'Fan Blower Imbalance Fault',
        'target_module': 'timeseries',
        'target_devices': TARGET_DEVICES,
        'stages': ['dataset', 'data_processing_feature_extraction', 'training', 'compilation'],
    },
    TASK_TYPE_GENERIC_TS_CLASSIFICATION: {
        'task_name': 'Generic Time Series Classification',
        'target_module': 'timeseries',
        'target_devices': TARGET_DEVICES,
        'stages': ['dataset', 'data_processing_feature_extraction', 'training', 'compilation'],
    },
    TASK_TYPE_GENERIC_TS_FORECASTING: {
        'task_name': 'Generic Time Series Forecasting',
        'target_module': 'timeseries',
        'target_devices': TARGET_DEVICES,
        'stages': ['dataset', 'data_processing_feature_extraction', 'training', 'compilation'],
    },
    TASK_TYPE_PIR_DETECTION: {
        'task_name': 'PIR Detection',
        'target_module': 'timeseries',
        'target_devices': TARGET_DEVICES,
        'stages': ['dataset', 'data_processing_feature_extraction', 'training', 'compilation'],
    },
}

DATA_PREPROCESSING_DEFAULT = 'default'
DATA_PREPROCESSING_PRESET_DESCRIPTIONS = dict(
    default=dict(downsampling_factor=1), )
FEATURE_EXTRACTION_DEFAULT = 'default'
FEATURE_EXTRACTION_PRESET_DESCRIPTIONS = dict(
    Custom_ArcFault=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], analysis_bandwidth=1, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    Custom_ArcFault_MSPM0=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT'], q15_scale_factor=4, analysis_bandwidth=1, frame_skip=8, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1, normalize_bin=True,),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    Custom_MotorFault_MSPM0=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT'], normalize_bin=True, frame_skip=1, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=3, q15_scale_factor=5 ),
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),
    Custom_MotorFault=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=[], normalize_bin=True, frame_skip=1, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=3, ),
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),
    Custom_Default=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=[], normalize_bin=True, frame_skip=1, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Custom_Default_MSPM0=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=[ 'FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT'], normalize_bin=True, frame_skip=1, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1, q15_scale_factor=5),
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    FFT256Input_FE_RFFT_16Feature_8Frame_removeDC_Full_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=[ 'FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT'], frame_size=256, feature_size_per_frame=16, num_frame_concat=8, min_bin=1, frame_skip=1, offset=0, scale=1, variables=1, q15_scale_factor=5, normalize_bin=True, data_proc_transforms=[],),
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    FFT1024Input_256Feature_1Frame_Full_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=256, num_frame_concat=1, min_bin=1, analysis_bandwidth=1, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1, ),
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    FFT1024Input_256Feature_1Frame_Half_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=256, num_frame_concat=1, min_bin=122, analysis_bandwidth=2, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    FFT1024Input_64Feature_4Frame_Half_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=64, num_frame_concat=4, min_bin=1, analysis_bandwidth=2, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1, ),
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    FFT1024Input_32Feature_8Frame_Quarter_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=32, num_frame_concat=8, min_bin=1, analysis_bandwidth=4, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    ArcFault_1024Input_256Feature_1Frame_Full_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=256, num_frame_concat=1, min_bin=1, analysis_bandwidth=1, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1, ),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    ArcFault_1024Input_256Feature_1Frame_Half_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=256, num_frame_concat=1, min_bin=122, analysis_bandwidth=2, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    ArcFault_1024Input_64Feature_4Frame_Half_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=64, num_frame_concat=4, min_bin=1, analysis_bandwidth=2, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1, ),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    ArcFault_1024Input_32Feature_8Frame_Quarter_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=32, num_frame_concat=8, min_bin=1, analysis_bandwidth=4, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    # ArcFault_512Input_FFT=dict(
    #     data_processing_feature_extraction=dict(transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=512, feature_size_per_frame=256, num_frame_concat=1, min_bin=1, analysis_bandwidth=1, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
    #     common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    ArcFault_1024Input_FE_RFFT_128Feature_8Frame_1InputChannel_removeDC_Full_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT'], frame_size=1024, feature_size_per_frame=128, num_frame_concat=8, min_bin=1, frame_skip=8, scale=1, offset=0, normalize_bin=True, variables=1, q15_scale_factor=4, data_proc_transforms=[],),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    MotorFault_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_1D=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT'], frame_size=256, feature_size_per_frame=16, num_frame_concat=8, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='1D', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3,),  # ch=1,
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),
    MotorFault_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT'], frame_size=256, feature_size_per_frame=16, num_frame_concat=8, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),
    MotorFault_256Input_FFT_128Feature_1Frame_3InputChannel_removeDC_2D1=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'ABS', 'DC_REMOVE', 'LOG_DB', 'CONCAT'], frame_size=256, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),
    # MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_1D=dict(
    #     data_processing_feature_extraction=dict(transform=['RAW_FE', 'CONCAT'], frame_size=128, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, ch=1, offset=0, scale=1, stacking='1D', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3),
    #     common=dict(task_type=TASK_TYPE_MOTOR_FAULT),),
    MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_2D1=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['RAW_FE', 'CONCAT'], frame_size=128, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),
    MotorFault_256Input_FE_RFFT_16Feature_8Frame_3InputChannel_removeDC_2D1=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT'], frame_size=256, feature_size_per_frame=16, num_frame_concat=8, min_bin=1, normalize_bin=True, offset=0, scale=1, frame_skip=1, variables=3, q15_scale_factor=5, data_proc_transforms=[], dc_remove=True, stacking='2D1', ),  # ch=3,
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),

    Generic_1024Input_FFTBIN_64Feature_8Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=64, num_frame_concat=8, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_512Input_FFTBIN_32Feature_8Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT'], frame_size=512, feature_size_per_frame=32, num_frame_concat=8, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_256Input_FFTBIN_16Feature_8Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT'], frame_size=256, feature_size_per_frame=16, num_frame_concat=8, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_1024Input_FFT_512Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'ABS', 'DC_REMOVE', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=512, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_512Input_FFT_256Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'ABS', 'DC_REMOVE', 'LOG_DB', 'CONCAT'], frame_size=512, feature_size_per_frame=256, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_256Input_FFT_128Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'ABS', 'DC_REMOVE', 'LOG_DB', 'CONCAT'], frame_size=256, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_512Input_RAW_512Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['RAW_FE', 'CONCAT'], frame_size=512, feature_size_per_frame=512, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_256Input_RAW_256Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['RAW_FE', 'CONCAT'], frame_size=256, feature_size_per_frame=256, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_128Input_RAW_128Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['RAW_FE', 'CONCAT'], frame_size=128, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    PIRDetection_125Input_25Feature_25Frame_1InputChannel_2D=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['PIR_FE'], frame_size=125, window_count=25, chunk_size=8, fft_size=64, sampling_rate=31.25, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_PIR_DETECTION), ),  
)

DATASET_EXAMPLES = dict(
    default=dict(),
    arc_fault_example_dsi=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/arc_fault_classification_dsi.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=None, data_proc_transforms=['Downsample', 'SimpleWindow'], sampling_rate=313000, frame_size=3130, stride_size=0.01),
    ),
    arc_fault_example_dsk=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/arc_fault_classification_dsk.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('FFT1024Input_256Feature_1Frame_Full_Bandwidth'), data_proc_transforms=[], sampling_rate=1),
    ),
    motor_fault_example_dsk=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/motor_fault_classification_dsk.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('MotorFault_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'), data_proc_transforms=[], sampling_rate=1, variables=3),
    ),
    fan_blower_imbalance_dsh=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/fan_blower_imbalance_dsh.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('MotorFault_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'), data_proc_transforms=[], sampling_rate=1, variables=3),
    ),
    hello_world_example_dsg=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/hello_world_dsg.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('Generic_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'), data_proc_transforms=[], sampling_rate=1, variables=3),
    ),
    ac_arc_fault_msp_example_dsk=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/ac_arc_fault.zip'), 
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('ArcFault_1024Input_FE_RFFT_128Feature_8Frame_1InputChannel_removeDC_Full_Bandwidth'), data_proc_transforms=[], sampling_rate=1),
    ),
    pir_detection_example_dsk=dict(
        dataset=dict(input_data_path='/home/a0500425/ti_edge_ai_studio/pir_detection_classification_dsk.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('PIRDetection_125Input_25Feature_25Frame_1InputChannel_2D'), data_proc_transforms=[], sampling_rate=31.25, variables=1),
    ),
)
DATASET_DEFAULT = 'default'
# compilation settings for various speed and accuracy tradeoffs:
# detection_threshold & detection_top_k are written to the prototxt - inside edgeai-benchmark.
# prototxt is not used in AM62 - so those values does not have effect in AM62 - they are given just for completeness.
# if we really wan't to change the detections settings in AM62, we will have to modify the onnx file, but that's not easy.
COMPILATION_FORCED_SOFT_NPU = 'forced_soft_npu_preset'
COMPILATION_NPU_OPT_FOR_SPACE = 'compress_npu_layer_data'
COMPILATION_DEFAULT = 'default_preset'

HOME_DIR = os.getenv('HOME', os.path.expanduser("~"))

TOOLS_PATH = os.path.abspath(os.getenv('TOOLS_PATH', os.path.join(f'{HOME_DIR}', 'bin')))
# C2000 F28 Compiler
C2000_CGT_VERSION = 'ti-cgt-c2000_22.6.3.LTS'
C2000_CG_ROOT = os.path.abspath(os.getenv('C2000_CG_ROOT', os.path.join(TOOLS_PATH, C2000_CGT_VERSION)))
CL2000_CROSS_COMPILER = os.path.join(C2000_CG_ROOT, 'bin', 'cl2000')
C2000_CGT_INCLUDE = os.path.join(C2000_CG_ROOT, 'include')
# C2000 F28 SDK
C2000WARE_VERSION = 'C2000Ware_6_00_01_00'
C2000WARE_ROOT = os.path.abspath(os.getenv('C2000WARE_ROOT', os.path.join(TOOLS_PATH, C2000WARE_VERSION)))
C2000WARE_INCLUDE = os.path.join(C2000WARE_ROOT, 'device_support', '{DEVICE_NAME}', 'common', 'include')
C2000_DRIVERLIB_INCLUDE = os.path.join(C2000WARE_ROOT, 'driverlib', '{DEVICE_NAME}', 'driverlib')
# C2000 F29 Compiler
C29_CGT_VERSION = 'ti-cgt-c29_2.0.0.STS'
C29_CG_ROOT = os.path.abspath(os.getenv('C29_CG_ROOT', os.path.join(TOOLS_PATH, C29_CGT_VERSION)))
C29CLANG_CROSS_COMPILER = os.path.join(C29_CG_ROOT, 'bin', 'c29clang')
C29_CGT_INCLUDE = os.path.join(C29_CG_ROOT, 'include')
# C2000 F29H85 SDK --> For F29 there is device wise SDK. --> SDK is no longer required for TVM from ti-mcu-nnc-2.0.0
# F29H85_SDK_VERSION = 'f29h85x-sdk_1_01_00_00'
# F29H85_SDK_ROOT = os.path.abspath(os.getenv('F29H85_SDK_ROOT', os.path.join(TOOLS_PATH, F29H85_SDK_VERSION)))
# F29H85_SDK_INCLUDE = os.path.join(F29H85_SDK_ROOT, 'device_support', '{DEVICE_NAME}', 'common', 'include')
# F29H85_DRIVERLIB_INCLUDE = os.path.join(F29H85_SDK_ROOT, 'driverlib', '{DEVICE_NAME}', 'driverlib')

# MSPM0 Compiler
MSPM0_CGT_VERSION= 'ti-cgt-armllvm_4.0.3.LTS'
ARM_LLVM_CGT_PATH = os.path.abspath(os.getenv('ARM_LLVM_CGT_PATH', os.path.join(TOOLS_PATH, MSPM0_CGT_VERSION)))
MSPM0_CROSS_COMPILER = os.path.join(ARM_LLVM_CGT_PATH, 'bin', 'tiarmclang')
# MSPM0 SDK --> SDK is no longer required for TVM from ti-mcu-nnc-2.0.0
# M0SDK_VERSION='mspm0_sdk_2_05_00_05'
# M0SDK_PATH = os.path.abspath(os.getenv('M0SDK_PATH', os.path.join(TOOLS_PATH, M0SDK_VERSION)))
# M0SDK_INCLUDE = os.path.join(M0SDK_PATH, 'source')
# MSPM0_SOURCE_INCLUDE = os.path.join(M0SDK_PATH, 'source', 'third_party', 'CMSIS', 'Core', 'Include')

# CC2755 Compiler
CC2755_CGT_VERSION= 'ti-cgt-armllvm_4.0.3.LTS'
# CC2755_CGT_PATH = os.path.abspath(os.getenv('CC2755_CGT_PATH', os.path.join(TOOLS_PATH, CC2755_CGT_VERSION)))
CC2755_CROSS_COMPILER = os.path.join(ARM_LLVM_CGT_PATH, 'bin', 'tiarmclang')


CROSS_COMPILER_OPTIONS_C28 = f"--abi=eabi -O3 --opt_for_speed=5 --c99 -v28 -ml -mt --gen_func_subsections --float_support={{FLOAT_SUPPORT}} -I{C2000_CGT_INCLUDE} -I{C2000_DRIVERLIB_INCLUDE} -I{C2000WARE_INCLUDE} -I. -Iartifacts --obj_directory=."
CROSS_COMPILER_OPTIONS_F29H85 = f"-O3 -ffast-math -I{C29_CGT_INCLUDE} -I."
CROSS_COMPILER_OPTIONS_MSPM0 = f"-Os -mcpu=cortex-m0plus -march=thumbv6m -mtune=cortex-m0plus -mthumb -mfloat-abi=soft -I. -Wno-return-type"
CROSS_COMPILER_OPTIONS_CC2755 = f"-O3 -mcpu=cortex-m33 -march=thumbv6m -mfpu=fpv5-sp-d16 -DARM_CPU_INTRINSICS_EXIST -mlittle-endian -mfloat-abi=hard -I. -Wno-return-type"

CROSS_COMPILER_OPTIONS_F280013 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32', DEVICE_NAME=TARGET_DEVICE_F280013.lower() + 'x')
CROSS_COMPILER_OPTIONS_F280015 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32', DEVICE_NAME=TARGET_DEVICE_F280015.lower() + 'x')
CROSS_COMPILER_OPTIONS_F28003 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32', DEVICE_NAME=TARGET_DEVICE_F28003.lower() + 'x')
CROSS_COMPILER_OPTIONS_F28004 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32', DEVICE_NAME=TARGET_DEVICE_F28004.lower() + 'x')
CROSS_COMPILER_OPTIONS_F2837 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32', DEVICE_NAME=TARGET_DEVICE_F2837.lower() + 'xd')
CROSS_COMPILER_OPTIONS_F28P65 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu64', DEVICE_NAME=TARGET_DEVICE_F28P65.lower() + 'x')
CROSS_COMPILER_OPTIONS_F28P55 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32', DEVICE_NAME=TARGET_DEVICE_F28P55.lower() + 'x')
COMPILATION_C28_SOFT_TINPU = dict(target="c, ti-npu type=soft skip_normalize=true output_int=true", target_c_mcpu='c28', cross_compiler=CL2000_CROSS_COMPILER, )
COMPILATION_C28_HARD_TINPU = dict(target="c, ti-npu type=hard skip_normalize=true output_int=true", target_c_mcpu='c28', cross_compiler=CL2000_CROSS_COMPILER, )
COMPILATION_C28_SOFT_TINPU_REG = dict(target="c, ti-npu type=soft skip_normalize=false output_int=false", target_c_mcpu='c28', cross_compiler=CL2000_CROSS_COMPILER, )
COMPILATION_C28_SOFT_TINPU_AD = dict(target="c, ti-npu type=soft", target_c_mcpu='c28', cross_compiler=CL2000_CROSS_COMPILER, )
COMPILATION_C28_HARD_TINPU_AD = dict(target="c, ti-npu", target_c_mcpu='c28', cross_compiler=CL2000_CROSS_COMPILER, )
COMPILATION_C28_HARD_TINPU_OPT_SPACE = dict(target="c, ti-npu type=hard skip_normalize=true output_int=true opt_for_space=true", target_c_mcpu='c28', cross_compiler=CL2000_CROSS_COMPILER, )
COMPILATION_F29H85_SOFT_TINPU = dict(target="c, ti-npu type=soft", target_c_mcpu='c29', cross_compiler=C29CLANG_CROSS_COMPILER, )
COMPILATION_MSPM0_SOFT_TINPU = dict(target="c, ti-npu type=soft skip_normalize=true output_int=true", target_c_mcpu='cortex-m0plus', cross_compiler=MSPM0_CROSS_COMPILER, )
COMPILATION_MSPM0_HARD_TINPU = dict(target="c, ti-npu skip_normalize=true output_int=true", target_c_mcpu='cortex-m0plus', cross_compiler=MSPM0_CROSS_COMPILER, )
COMPILATION_MSPM0_HARD_TINPU_OPT_SPACE = dict(target="c, ti-npu skip_normalize=true output_int=true opt_for_space=true", target_c_mcpu='cortex-m0plus', cross_compiler=MSPM0_CROSS_COMPILER, )
COMPILATION_CC2755_SOFT_TINPU = dict(target="c, ti-npu type=soft", target_c_mcpu='cortex-m33', cross_compiler=CC2755_CROSS_COMPILER, )

PRESET_DESCRIPTIONS = {
    TARGET_DEVICE_AM263: {
        TASK_TYPE_ARC_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(target="c", target_c_mcpu='cortex_r5', cross_compiler="tiarmclang",
                                 cross_compiler_options="-O3 -mcpu=cortex-r5 -march=armv7-r -mthumb -mfloat-abi=hard -mfpu=vfpv3-d16 -mlittle-endian -Iartifacts -Wno-return-type", )
            ),
        },
        TASK_TYPE_MOTOR_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(target="c", target_c_mcpu='cortex_r5', cross_compiler="tiarmclang",
                                 cross_compiler_options="-O3 -mcpu=cortex-r5 -march=armv7-r -mthumb -mfloat-abi=hard -mfpu=vfpv3-d16 -mlittle-endian -Iartifacts -Wno-return-type", )
            ),
        },
        TASK_TYPE_BLOWER_IMBALANCE: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(target="c", target_c_mcpu='cortex_r5', cross_compiler="tiarmclang",
                                 cross_compiler_options="-O3 -mcpu=cortex-r5 -march=armv7-r -mthumb -mfloat-abi=hard -mfpu=vfpv3-d16 -mlittle-endian -Iartifacts -Wno-return-type", )
            ),
        },
        TASK_CATEGORY_TS_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(target="c", target_c_mcpu='cortex_r5', cross_compiler="tiarmclang",
                                 cross_compiler_options="-O3 -mcpu=cortex-r5 -march=armv7-r -mthumb -mfloat-abi=hard -mfpu=vfpv3-d16 -mlittle-endian -Iartifacts -Wno-return-type", )
            ),
        },
        TASK_CATEGORY_TS_REGRESSION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(target="c", target_c_mcpu='cortex_r5', cross_compiler="tiarmclang",
                                 cross_compiler_options="-O3 -mcpu=cortex-r5 -march=armv7-r -mthumb -mfloat-abi=hard -mfpu=vfpv3-d16 -mlittle-endian -Iartifacts -Wno-return-type", )
            ),
        },
        TASK_CATEGORY_TS_ANOMALYDETECTION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(target="c", target_c_mcpu='cortex_r5', cross_compiler="tiarmclang",
                                 cross_compiler_options="-O3 -mcpu=cortex-r5 -march=armv7-r -mthumb -mfloat-abi=hard -mfpu=vfpv3-d16 -mlittle-endian -Iartifacts -Wno-return-type", )
            ),
        },
        TASK_CATEGORY_TS_FORECASTING: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(target="c", target_c_mcpu='cortex_r5', cross_compiler="tiarmclang",
                                 cross_compiler_options="-O3 -mcpu=cortex-r5 -march=armv7-r -mthumb -mfloat-abi=hard -mfpu=vfpv3-d16 -mlittle-endian -Iartifacts -Wno-return-type", )
            ),
        },
    },
    TARGET_DEVICE_F280013: {
        TASK_TYPE_ARC_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280013, )
            ),
        },
        TASK_TYPE_MOTOR_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280013, )
            ),
        },
        TASK_TYPE_BLOWER_IMBALANCE: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280013, )
            ),
        },
        TASK_TYPE_GENERIC_TS_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280013, )
            ),
        },
        TASK_TYPE_GENERIC_TS_REGRESSION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU_REG, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280013, )
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280013, )
            ),
        },
        TASK_TYPE_GENERIC_TS_FORECASTING: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280013, )
            ),
        },
    },
    TARGET_DEVICE_F280015: {
        TASK_TYPE_ARC_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280015, )
            ),
        },
        TASK_TYPE_MOTOR_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280015, )
            ),
        },
        TASK_TYPE_BLOWER_IMBALANCE: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280015, )
            ),
        },
        TASK_TYPE_GENERIC_TS_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280015, )
            ),
        },
        TASK_TYPE_GENERIC_TS_REGRESSION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU_REG, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280015, )
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280015, )
            ),
        },
        TASK_TYPE_GENERIC_TS_FORECASTING: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280015, )
            ),
        },
    },
    TARGET_DEVICE_F28003: {
        TASK_TYPE_ARC_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28003, )
            ),
        },
        TASK_TYPE_MOTOR_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28003, )
            ),
        },
        TASK_TYPE_BLOWER_IMBALANCE: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28003, )
            ),
        },
        TASK_TYPE_GENERIC_TS_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28003, )
            ),
        },
        TASK_TYPE_GENERIC_TS_REGRESSION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU_REG, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28003, )
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28003, )
            ),
        },
        TASK_TYPE_GENERIC_TS_FORECASTING: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28003, )
            ),
        },
    },
    TARGET_DEVICE_F28004: {
        TASK_TYPE_ARC_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28004, )
            ),
        },
        TASK_TYPE_MOTOR_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28004, )
            ),
        },
        TASK_TYPE_BLOWER_IMBALANCE: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28004, )
            ),
        },
        TASK_TYPE_GENERIC_TS_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28004, )
            ),
        },
        TASK_TYPE_GENERIC_TS_REGRESSION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU_REG, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28004, )
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28004, )
            ),
        },
        TASK_TYPE_GENERIC_TS_FORECASTING: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28004, )
            ),
        },
    },
    TARGET_DEVICE_F28P65: {
        TASK_TYPE_ARC_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P65, )
            ),
        },
        TASK_TYPE_MOTOR_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P65, )
            ),
        },
        TASK_TYPE_BLOWER_IMBALANCE: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P65, )
            ),
        },
        TASK_TYPE_GENERIC_TS_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P65, )
            ),
        },
        TASK_TYPE_GENERIC_TS_REGRESSION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU_REG, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P65, )
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P65, )
            ),
        },
        TASK_TYPE_GENERIC_TS_FORECASTING: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P65, )
            ),
        },
    },
    TARGET_DEVICE_F28P55: {
        TASK_TYPE_ARC_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_NPU_OPT_FOR_SPACE: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU_OPT_SPACE, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
        },
        TASK_TYPE_MOTOR_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_NPU_OPT_FOR_SPACE: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU_OPT_SPACE, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
        },
        TASK_TYPE_BLOWER_IMBALANCE: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_NPU_OPT_FOR_SPACE: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU_OPT_SPACE, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
        },
        TASK_TYPE_GENERIC_TS_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_NPU_OPT_FOR_SPACE: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU_OPT_SPACE, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
        },
        TASK_TYPE_GENERIC_TS_REGRESSION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU_REG, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_NPU_OPT_FOR_SPACE: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU_OPT_SPACE, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU_AD, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_NPU_OPT_FOR_SPACE: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU_OPT_SPACE, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU_AD, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
        },
        TASK_TYPE_GENERIC_TS_FORECASTING: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_NPU_OPT_FOR_SPACE: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU_OPT_SPACE, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
        },
    },
    TARGET_DEVICE_F2837: {
        TASK_TYPE_ARC_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F2837)
            ),
        },
        TASK_TYPE_MOTOR_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F2837)
            ),
        },
        TASK_TYPE_BLOWER_IMBALANCE: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F2837)
            ),
        },
        TASK_TYPE_GENERIC_TS_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F2837)
            ),
        },
        TASK_TYPE_GENERIC_TS_REGRESSION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU_REG, cross_compiler_options=CROSS_COMPILER_OPTIONS_F2837)
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F2837)
            ),
        },
        TASK_TYPE_GENERIC_TS_FORECASTING: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F2837)
            ),
        },

    },

    TARGET_DEVICE_F29H85: {
        TASK_TYPE_ARC_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_F29H85_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F29H85)
            ),
        },
        TASK_TYPE_MOTOR_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_F29H85_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F29H85)
            ),
        },
        TASK_TYPE_BLOWER_IMBALANCE: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_F29H85_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F29H85)
            ),
        },
        TASK_TYPE_GENERIC_TS_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_F29H85_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F29H85)
            ),
        },
        TASK_TYPE_GENERIC_TS_REGRESSION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_F29H85_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F29H85)
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_F29H85_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F29H85)
            ),
        },
        TASK_TYPE_GENERIC_TS_FORECASTING: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_F29H85_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F29H85)
            ),
        },

        
    },
    TARGET_DEVICE_MSPM0G3507: {
        TASK_TYPE_ARC_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_MSPM0_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM0, )
            ),
        },
        TASK_TYPE_GENERIC_TS_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_MSPM0_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM0, )
            ),
        },
        TASK_TYPE_MOTOR_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_MSPM0_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM0, )
            ),
        },
    },
    TARGET_DEVICE_MSPM0G5187: {
        TASK_TYPE_ARC_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_MSPM0_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM0, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_MSPM0_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM0, )
            ),
            COMPILATION_NPU_OPT_FOR_SPACE: dict(
                compilation=dict(**COMPILATION_MSPM0_HARD_TINPU_OPT_SPACE, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM0, )
            ),
        },
        TASK_TYPE_GENERIC_TS_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_MSPM0_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM0, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_MSPM0_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM0, )
            ),
              COMPILATION_NPU_OPT_FOR_SPACE: dict(
                compilation=dict(**COMPILATION_MSPM0_HARD_TINPU_OPT_SPACE, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM0, )
            ),
        },
        TASK_TYPE_MOTOR_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_MSPM0_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM0, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_MSPM0_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM0, )
            ),
              COMPILATION_NPU_OPT_FOR_SPACE: dict(
                compilation=dict(**COMPILATION_MSPM0_HARD_TINPU_OPT_SPACE, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM0, )
            ),
        },
    },
    
    TARGET_DEVICE_CC2755: {
        TASK_TYPE_ARC_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_CC2755_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_CC2755, )
            ),
        },
        TASK_TYPE_GENERIC_TS_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_CC2755_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_CC2755)
            ),
        },
        TASK_TYPE_GENERIC_TS_REGRESSION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_CC2755_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_CC2755)
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_CC2755_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_CC2755)
            ),
        },
        TASK_TYPE_GENERIC_TS_FORECASTING: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_CC2755_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_CC2755)
            ),
        },
        TASK_TYPE_PIR_DETECTION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_CC2755_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_CC2755)
            ),
        },
    },
}

SAMPLE_DATASET_DESCRIPTIONS = {
    # 'arc_fault_example_dsi': {
    #     'common': {
    #         'task_type': TASK_TYPE_ARC_FAULT,
    #         'task_category': TASK_CATEGORY_TS_CLASSIFICATION,
    #     },
    #     'dataset': {
    #         'dataset_name': 'arc_fault_classification_dsi',
    #         'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/arc_fault_classification_dsi.zip',
    #     },
    #     'info': {
    #         'dataset_url': 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/arc_fault_classification_dsi.zip',
    #         'dataset_detailed_name': 'Arc Fault Classification Example1',
    #         'dataset_description': 'Example arc-fault timeseries classification dataset with 2 categories - arc, normal',
    #         'dataset_size': None,
    #         'dataset_source': 'Generated by Texas Instruments at a specialised test bed',
    #         'dataset_license': 'TI Internal License'
    #     }
    # },
    'arc_fault_example_dsk': {
        'common': {
            'task_type': TASK_TYPE_ARC_FAULT,
            'task_category': TASK_CATEGORY_TS_CLASSIFICATION,
        },
        'dataset': {
            'dataset_name': 'arc_fault_classification_dsk',
            'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/arc_fault_classification_dsk.zip',
        },
        'info': {
            'dataset_url': 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/arc_fault_classification_dsk.zip',
            'dataset_detailed_name': 'Arc Fault Classification Example2',
            'dataset_description': 'Example arc-fault timeseries classification dataset with 2 categories - arc, normal',
            'dataset_size': None,
            'dataset_source': 'Generated by Texas Instruments at a specialised test bed',
            'dataset_license': 'TI Internal License'
        }
    },
     'ac_arc_fault_msp_example_dsk': {
        'common': {
            'task_type': TASK_TYPE_ARC_FAULT,
            'task_category': TASK_CATEGORY_TS_CLASSIFICATION,
        },
        'dataset': {
            'dataset_name': 'ac_arc_fault',
            'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/ac_arc_fault.zip',
        },
        'info': {
            'dataset_url': 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/ac_arc_fault.zip',
            'dataset_detailed_name': 'Ac Arc Fault Classification Example for MSPM0',
            'dataset_description': 'Example arc-fault timeseries classification dataset with 2 categories - arc, normal',
            'dataset_size': None,
            'dataset_source': 'Generated by Texas Instruments at a specialised test bed',
            'dataset_license': 'TI Internal License'
        }
    },
    'motor_fault_example_dsk': {
        'common': {
            'task_type': TASK_TYPE_MOTOR_FAULT,
            'task_category': TASK_CATEGORY_TS_CLASSIFICATION,
        },
        'dataset': {
            'dataset_name': 'motor_fault_example_dsk',
            'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/motor_fault_classification_dsk.zip',
        },
        'info': {
            'dataset_url': 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/motor_fault_classification_dsk.zip',
            'dataset_detailed_name': 'Motor Bearing Fault Classification Example',
            'dataset_description': 'Example motor-fault timeseries classification dataset with 4 categories - normal, localized, erosion, flaking',
            'dataset_size': None,
            'dataset_source': 'Generated by Texas Instruments at a specialised test bed',
            'dataset_license': 'TI Internal License'
        }
    },
    'blower_imbalance_example_dsh': {
        'common': {
            'task_type': TASK_TYPE_BLOWER_IMBALANCE,
            'task_category': TASK_CATEGORY_TS_CLASSIFICATION,
        },
        'dataset': {
            'dataset_name': 'blower_imbalance_example_dsh',
            'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/fan_blower_imbalance_dsh.zip',
        },
        'info': {
            'dataset_url': 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/fan_blower_imbalance_dsh.zip',
            'dataset_detailed_name': 'Blower Imbalance Classification Example',
            'dataset_description': 'Example blower imbalance timeseries classification dataset with 2 categories - 0 Clips, 1 Clip',
            'dataset_size': None,
            'dataset_source': 'Generated by Texas Instruments at a specialised test bed',
            'dataset_license': 'TI Internal License'
        }
    },
    'hello_world_example_dsg': {
        'common': {
            'task_type': TASK_TYPE_GENERIC_TS_CLASSIFICATION,
            'task_category': TASK_CATEGORY_TS_CLASSIFICATION,
        },
        'dataset': {
            'dataset_name': 'hello_world_example_dsg',
            'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/hello_world_dsg.zip',
        },
        'info': {
            'dataset_url': 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/hello_world_dsg.zip',
            'dataset_detailed_name': 'Generic Timeseries Classification Example',
            'dataset_description': 'Example timeseries classification dataset with 3 categories - Sine, Square, Sawtooth',
            'dataset_size': None,
            'dataset_source': 'Generated by Texas Instruments at a specialised test bed',
            'dataset_license': 'TI Internal License'
        }
    },
    'pir_detection_example_dsk': {
        'common': {
            'task_type': TASK_TYPE_PIR_DETECTION,
            'task_category': TASK_CATEGORY_TS_CLASSIFICATION,
        },
        'dataset': {
            'dataset_name': 'pir_detection_classification_dsk',
            'input_data_path': '/home/a0500425/ti_edge_ai_studio/pir_detection_classification_dsk.zip',
        },
        'info': {
            'dataset_url': '/home/a0500425/ti_edge_ai_studio/pir_detection_classification_dsk.zip',
            'dataset_detailed_name': 'PIR Detection Classification Example',
            'dataset_description': 'Example PIR sensor based motion detection timeseries classification dataset with 3 categories - human, dog, background motion',
            'dataset_size': None,
            'dataset_source': 'Generated by Texas Instruments',
            'dataset_license': 'TI Internal License'
        }
    },
}
