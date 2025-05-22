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
TASK_TYPE_GENERIC_TS_CLASSIFICATION = 'generic_timeseries_classification'
TASK_TYPE_GENERIC_TS_REGRESSION = 'generic_timeseries_regression'
TASK_TYPE_GENERIC_TS_ANOMALYDETECTION = 'generic_timeseries_anomalydetection'

TASK_TYPES = [
    TASK_TYPE_ARC_FAULT,
    TASK_TYPE_MOTOR_FAULT,
    TASK_TYPE_BLOWER_IMBALANCE,
    TASK_TYPE_GENERIC_TS_CLASSIFICATION,
    TASK_TYPE_GENERIC_TS_REGRESSION,
    TASK_TYPE_GENERIC_TS_ANOMALYDETECTION
]

# task_category
TASK_CATEGORY_TS_CLASSIFICATION = 'timeseries_classification'
TASK_CATEGORY_TS_REGRESSION = 'timeseries_regression'
TASK_CATEGORY_TS_ANOMALYDETECTION = 'timeseries_anomalydetection'

TASK_CATEGORIES = [
    TASK_CATEGORY_TS_CLASSIFICATION, TASK_CATEGORY_TS_REGRESSION
]

# target_device
TARGET_DEVICE_AM263 = 'AM263'
TARGET_DEVICE_F280015 = 'F280015'
TARGET_DEVICE_F28003 = 'F28003'
TARGET_DEVICE_F28004 = 'F28004'
TARGET_DEVICE_F2837 = 'F2837'
TARGET_DEVICE_F28P55 = 'F28P55'
TARGET_DEVICE_F28P65 = 'F28P65'

TARGET_DEVICES = [
    TARGET_DEVICE_F280015,
    TARGET_DEVICE_F28003,
    TARGET_DEVICE_F28004,
    TARGET_DEVICE_F2837,
    TARGET_DEVICE_F28P55,
    TARGET_DEVICE_F28P65,
]

# will not be listed in the GUI, but can be used in command line
TARGET_DEVICES_ADDITIONAL = [
    TARGET_DEVICE_AM263,
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
}

TARGET_SDK_VERSION = '5.4'
TARGET_SDK_RELEASE = '05_04_00'

TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION = '\n'
# TODO: Once the links are up add this
'''* Tiny ML introduction: https://ti.com/tinyml
* Tiny ML model development information: https://github.com/TexasInstruments/tinyml-tensorlab
* Tiny ML tools introduction: https://dev.ti.com/tinyml/
'''

##### AM263 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_AM263 = \
    f'''* Product information: https://www.ti.com/product/AM2634
* Development board: https://www.ti.com/tool/LP-AM263
* SDK: https://www.ti.com/tool/MCU-PLUS-SDK-AM263X
* SDK release: {TARGET_SDK_RELEASE}'''

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
* SDK release: {TARGET_SDK_RELEASE}'''

TARGET_DEVICE_DETAILS_F280015 = \
    f'''C2000™ 32-bit MCU 120-MHz 384-KB flash, FPU, TMU with CLA, CLB, AES and CAN-FD
* More details : https://www.ti.com/tool/LAUNCHXL-F2800157

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_F280015}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

##### F28003 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_F28003 = \
    f'''* Product information: https://www.ti.com/product/TMS320F280039C
* Launchpad: https://www.ti.com/tool/LAUNCHXL-F280039C
* C2000 SDK: https://www.ti.com/tool/C2000WARE
* SDK release: {TARGET_SDK_RELEASE}'''

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
* SDK release: {TARGET_SDK_RELEASE}'''

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
* SDK release: {TARGET_SDK_RELEASE}'''

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
* SDK release: {TARGET_SDK_RELEASE}'''

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
* SDK release: {TARGET_SDK_RELEASE}'''

TARGET_DEVICE_DETAILS_F28P55 = \
    f'''C2000™ 32-bit MCU, 1x C28x + 1x CLA, 150-MHz, 1.1-MB flash, 5x ADCs, CLB, AES and NNPU
* More details : https://www.ti.com/product/TMS320F28P550SJ

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_F28P55}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

# higher device_selection_factor indicates higher performance device.
TARGET_DEVICE_DESCRIPTIONS = {
    TARGET_DEVICE_F280015: {
        'device_name': TARGET_DEVICE_F280015,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 0,
        'device_details': TARGET_DEVICE_DETAILS_F280015,
        'sdk_version': TARGET_SDK_VERSION,
        'sdk_release': TARGET_SDK_RELEASE,
    },
    TARGET_DEVICE_F28003: {
        'device_name': TARGET_DEVICE_F28003,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 1,
        'device_details': TARGET_DEVICE_DETAILS_F28003,
        'sdk_version': TARGET_SDK_VERSION,
        'sdk_release': TARGET_SDK_RELEASE,
    },
    TARGET_DEVICE_F28004: {
        'device_name': TARGET_DEVICE_F28004,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 2,
        'device_details': TARGET_DEVICE_DETAILS_F28004,
        'sdk_version': TARGET_SDK_VERSION,
        'sdk_release': TARGET_SDK_RELEASE,
    },
    TARGET_DEVICE_F2837: {
        'device_name': TARGET_DEVICE_F2837,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 3,
        'device_details': TARGET_DEVICE_DETAILS_F2837,
        'sdk_version': TARGET_SDK_VERSION,
        'sdk_release': TARGET_SDK_RELEASE,
    },
    TARGET_DEVICE_F28P65: {
        'device_name': TARGET_DEVICE_F28P65,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 4,
        'device_details': TARGET_DEVICE_DETAILS_F28P65,
        'sdk_version': TARGET_SDK_VERSION,
        'sdk_release': TARGET_SDK_RELEASE,
    },
    TARGET_DEVICE_F28P55: {
        'device_name': TARGET_DEVICE_F28P55,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 5,
        'device_details': TARGET_DEVICE_DETAILS_F28P55,
        'sdk_version': TARGET_SDK_VERSION,
        'sdk_release': TARGET_SDK_RELEASE,
    },
    TARGET_DEVICE_AM263: {
        'device_name': TARGET_DEVICE_AM263,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 6,
        'device_details': TARGET_DEVICE_DETAILS_AM263,
        'sdk_version': TARGET_SDK_VERSION,
        'sdk_release': TARGET_SDK_RELEASE,
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
    # TASK_TYPE_AUDIO_CLASSIFICATION: {
    #     'task_name': 'Image Classification',
    #     'target_module': 'vision',
    #     'target_devices': TARGET_DEVICES,
    #     'stages': ['dataset', 'training', 'compilation'],
    # },
}

DATA_PREPROCESSING_DEFAULT = 'default'
DATA_PREPROCESSING_PRESET_DESCRIPTIONS = dict(
    default=dict(downsampling_factor=1), )
FEATURE_EXTRACTION_DEFAULT = 'default'
FEATURE_EXTRACTION_PRESET_DESCRIPTIONS = dict(
    Custom_ArcFault=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], analysis_bandwidth=1, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    Custom_MotorFault=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=[], normalize_bin=True, frame_skip=1, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=3, ),
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),
    Custom_Default=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=[], normalize_bin=True, frame_skip=1, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
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
    #     feature_extraction=dict(transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=512, feature_size_per_frame=256, num_frame_concat=1, min_bin=1, analysis_bandwidth=1, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12),
    #     data_processing=dict(transforms=[], sampling_rate=1, new_sr=1, variables=1, ),
    #     common=dict(task_type=TASK_TYPE_ARC_FAULT), ),

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
    #     feature_extraction=dict(transform=['RAW_FE', 'CONCAT'], frame_size=128, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, ch=1, offset=0, scale=1, stacking='1D', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100),
    #     data_processing=dict(transforms=[], sampling_rate=1, variables=3),
    #     common=dict(task_type=TASK_TYPE_MOTOR_FAULT),),
    MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_2D1=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['RAW_FE', 'CONCAT'], frame_size=128, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),

    Generic_1024Input_FFTBIN_64Feature_8Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=64, num_frame_concat=8, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_512Input_FFTBIN_32Feature_8Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT'], frame_size=512, feature_size_per_frame=32, num_frame_concat=8, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_256Input_FFTBIN_16Feature_8Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT'], frame_size=256, feature_size_per_frame=16, num_frame_concat=8, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_1024Input_FFT_512Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'ABS', 'DC_REMOVE', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=512, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_512Input_FFT_256Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'ABS', 'DC_REMOVE', 'LOG_DB', 'CONCAT'], frame_size=512, feature_size_per_frame=256, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_256Input_FFT_128Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'ABS', 'DC_REMOVE', 'LOG_DB', 'CONCAT'], frame_size=256, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_512Input_RAW_512Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['RAW_FE', 'CONCAT'], frame_size=512, feature_size_per_frame=512, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_256Input_RAW_256Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['RAW_FE', 'CONCAT'], frame_size=256, feature_size_per_frame=256, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_128Input_RAW_128Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['RAW_FE', 'CONCAT'], frame_size=128, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
)

DATASET_EXAMPLES = dict(
    default=dict(),
    arc_fault_example_dsi=dict(
        dataset=dict(input_data_path='http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/arc_fault_classification_dsi.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=None, data_proc_transforms=['Downsample', 'SimpleWindow'], sampling_rate=313000, frame_size=3130, stride_size=0.01),
    ),
    arc_fault_example_dsk=dict(
        dataset=dict(input_data_path='http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/arc_fault_classification_dsk.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('FFT1024Input_256Feature_1Frame_Full_Bandwidth'), data_proc_transforms=[], sampling_rate=1),
    ),
    motor_fault_example_dsk=dict(
        dataset=dict(input_data_path='http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/motor_fault_classification_dsk.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('MotorFault_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'), data_proc_transforms=[], sampling_rate=1, variables=3),
    ),
    fan_blower_imbalance_dsh=dict(
        dataset=dict(input_data_path='http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/fan_blower_imbalance_dsh.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('MotorFault_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'), data_proc_transforms=[], sampling_rate=1, variables=3),
    ),
    hello_world_example_dsg=dict(
        dataset=dict(input_data_path='http://uda0484689.dhcp.ti.com:8100/tinyml_docker_images/datasets/HelloWorldExample.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('Generic_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'), data_proc_transforms=[], sampling_rate=1, variables=3),
    ),
)
DATASET_DEFAULT = 'default'
# compilation settings for various speed and accuracy tradeoffs:
# detection_threshold & detection_top_k are written to the prototxt - inside edgeai-benchmark.
# prototxt is not used in AM62 - so those values does not have effect in AM62 - they are given just for completeness.
# if we really wan't to change the detections settings in AM62, we will have to modify the onnx file, but that's not easy.
COMPILATION_FORCED_SOFT_NPU = 'forced_soft_npu_preset'
COMPILATION_DEFAULT = 'default_preset'

HOME_DIR = os.getenv('HOME', os.path.expanduser("~"))

TOOLS_PATH = os.path.abspath(os.getenv('TOOLS_PATH', os.path.join(f'{HOME_DIR}', 'bin')))
C2000_CGT_VERSION = 'ti-cgt-c2000_22.6.1.LTS'
C2000WARE_VERSION = 'C2000Ware_5_04_00_00'

C2000_CGT_PATH = os.path.abspath(os.getenv('C2000_CGT_PATH', os.path.join(TOOLS_PATH, C2000_CGT_VERSION)))
C2000WARE_PATH = os.path.abspath(os.getenv('C2000WARE_PATH', os.path.join(TOOLS_PATH, C2000WARE_VERSION)))

CROSS_COMPILER_CL2000 = os.path.join(C2000_CGT_PATH, 'bin', 'cl2000')
CGT_INCLUDE = os.path.join(C2000_CGT_PATH, 'include')
C2000WARE_INCLUDE = os.path.join(C2000WARE_PATH, 'device_support', '{DEVICE_NAME}', 'common', 'include')
DRIVERLIB_INCLUDE = os.path.join(C2000WARE_PATH, 'driverlib', '{DEVICE_NAME}', 'driverlib')
CROSS_COMPILER_OPTIONS_C28 = (
    f"--abi=eabi -O3 --opt_for_speed=5 --c99 -v28 -ml -mt --gen_func_subsections --float_support={{FLOAT_SUPPORT}} -I{CGT_INCLUDE} -I{DRIVERLIB_INCLUDE} -I{C2000WARE_INCLUDE} -I. -Iartifacts --obj_directory=.")

CROSS_COMPILER_OPTIONS_F280015 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32',
                                                                   DEVICE_NAME=TARGET_DEVICE_F280015.lower() + 'x')
CROSS_COMPILER_OPTIONS_F28003 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32',
                                                                  DEVICE_NAME=TARGET_DEVICE_F28003.lower() + 'x')
CROSS_COMPILER_OPTIONS_F28004 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32',
                                                                  DEVICE_NAME=TARGET_DEVICE_F28004.lower() + 'x')
CROSS_COMPILER_OPTIONS_F2837 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32',
                                                                 DEVICE_NAME=TARGET_DEVICE_F2837.lower() + 'xd')
CROSS_COMPILER_OPTIONS_F28P65 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu64',
                                                                  DEVICE_NAME=TARGET_DEVICE_F28P65.lower() + 'x')
CROSS_COMPILER_OPTIONS_F28P55 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32',
                                                                  DEVICE_NAME=TARGET_DEVICE_F28P55.lower() + 'x')
COMPILATION_C28_SOFT_TINPU = dict(target="c, ti-npu type=soft skip_normalize=true output_int=true", target_c_mcpu='c28',
                                  cross_compiler=CROSS_COMPILER_CL2000, )
COMPILATION_C28_HARD_TINPU = dict(target="c, ti-npu type=hard skip_normalize=true output_int=true", target_c_mcpu='c28',
                                  cross_compiler=CROSS_COMPILER_CL2000, )

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
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F280015, )
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
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
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28003, )
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
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
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28004, )
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
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
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P65, )
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
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
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
        },
        TASK_TYPE_MOTOR_FAULT: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
        },
        TASK_TYPE_BLOWER_IMBALANCE: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
        },
        TASK_TYPE_GENERIC_TS_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
        },
        TASK_TYPE_GENERIC_TS_REGRESSION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F28P55, )
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
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F2837)
            ),
        },
        TASK_TYPE_GENERIC_TS_ANOMALYDETECTION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_C28_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_F2837)
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
    #         'input_data_path': 'http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/arc_fault_classification_dsi.zip',
    #     },
    #     'info': {
    #         'dataset_url': 'http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/arc_fault_classification_dsi.zip',
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
            'input_data_path': 'http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/arc_fault_classification_dsk.zip',
        },
        'info': {
            'dataset_url': 'http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/arc_fault_classification_dsk.zip',
            'dataset_detailed_name': 'Arc Fault Classification Example2',
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
            'input_data_path': 'http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/motor_fault_classification_dsk.zip',
        },
        'info': {
            'dataset_url': 'http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/motor_fault_classification_dsk.zip',
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
            'input_data_path': 'http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/fan_blower_imbalance_dsh.zip',
        },
        'info': {
            'dataset_url': 'http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/fan_blower_imbalance_dsh.zip',
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
            'input_data_path': 'http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/hello_world_dsg.zip',
        },
        'info': {
            'dataset_url': 'http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/hello_world_dsg.zip',
            'dataset_detailed_name': 'Generic Timeseries Classification Example',
            'dataset_description': 'Example timeseries classification dataset with 3 categories - Sine, Square, Sawtooth',
            'dataset_size': None,
            'dataset_source': 'Generated by Texas Instruments at a specialised test bed',
            'dataset_license': 'TI Internal License'
        }
    },
}
