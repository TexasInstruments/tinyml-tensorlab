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
import os

# plugins/additional models
# see the setup_all.sh file to understand how to set this
PLUGINS_ENABLE_GPL = False
PLUGINS_ENABLE_EXTRA = False

# task_type

TASK_TYPE_RADAR_CLASSIFICATION = 'radar_classification'

TASK_TYPES = [
    TASK_TYPE_RADAR_CLASSIFICATION,

]

# task_category
TASK_CATEGORY_RADAR_CLASSIFICATION = 'radar_classification'

TASK_CATEGORIES = [
   TASK_CATEGORY_RADAR_CLASSIFICATION,
]

TASK_TYPE_TO_CATEGORY = {
    TASK_TYPE_RADAR_CLASSIFICATION: TASK_CATEGORY_RADAR_CLASSIFICATION,
}

def get_default_data_dir_for_task(task_category):
    """
    Determine the default data_dir based on task category.

    Currently vision module only supports classification which uses 'classes'.

    Args:
        task_category: Task category string

    Returns:
        str: 'classes' for radar classification
    """
    return 'classes'  # Vision currently only supports classification


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
TARGET_DEVICE_MSPM0G3519 = 'MSPM0G3519'
TARGET_DEVICE_MSPM33C32 = 'MSPM33C32'
TARGET_DEVICE_MSPM33C34 = 'MSPM33C34'
TARGET_DEVICE_CC2755 = 'CC2755'
TARGET_DEVICE_CC1352 = 'CC1352'
TARGET_DEVICE_CC1354 = 'CC1354'
TARGET_DEVICE_CC35X1 = 'CC35X1'
TARGET_DEVICE_IWRL6432 = 'IWRL6432'

TARGET_DEVICES = [
    TARGET_DEVICE_IWRL6432,
]

# will not be listed in the GUI, but can be used in command line
TARGET_DEVICES_ADDITIONAL = []

# include additional devices that are not currently supported in release.
TARGET_DEVICES_ALL = TARGET_DEVICES + TARGET_DEVICES_ADDITIONAL



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

TRAINING_BATCH_SIZE_DEFAULT = {TASK_TYPE_RADAR_CLASSIFICATION: 64,

}






TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION = '\n * Tiny ML model development information: https://github.com/TexasInstruments/tinyml-tensorlab \n'


TASK_DESCRIPTIONS = {TASK_TYPE_RADAR_CLASSIFICATION: {
        'task_name': 'Radar Point Cloud Classification',
        'target_module': 'radar',
        'target_devices': [TARGET_DEVICE_IWRL6432],
        'stages': ['dataset', 'data_processing_feature_extraction', 'training', 'compilation'],
    },

}
DATA_PREPROCESSING_DEFAULT = 'default'
DATA_PREPROCESSING_PRESET_DESCRIPTIONS = dict(
    default=dict(downsampling_factor=1), )
FEATURE_EXTRACTION_DEFAULT = 'default'
FEATURE_EXTRACTION_PRESET_DESCRIPTIONS = dict( 
    Mnist_Default=dict(
        data_processing_feature_extraction=dict(image_height = 28, image_width = 28, image_num_channel= 1, image_mean= 0.1307, image_scale= 0.3081, variables=1),  
        common=dict(task_type=TASK_TYPE_RADAR_CLASSIFICATION), ),
)

DATASET_EXAMPLES = dict(
    default=dict(),
     mnist_image_classification=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/mnist_classes.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('Mnist_Default'), variables=1),
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

TOOLS_PATH = os.path.abspath(os.getenv('TOOLS_PATH', os.path.join(f'{HOME_DIR}', 'ti')))
# C2000 F28 Compiler
C2000_CGT_VERSION = 'ti-cgt-c2000_22.6.1.LTS'
C2000_CG_ROOT = os.path.abspath(os.getenv('C2000_CG_ROOT', os.path.join(TOOLS_PATH, C2000_CGT_VERSION)))
CL2000_CROSS_COMPILER = os.path.join(C2000_CG_ROOT, 'bin', 'cl2000')
C2000_CGT_INCLUDE = os.path.join(C2000_CG_ROOT, 'include')
# C2000 F28 SDK
C2000WARE_VERSION = 'C2000Ware_6_00_00_00'
C2000WARE_ROOT = os.path.abspath(os.getenv('C2000WARE_ROOT', os.path.join(TOOLS_PATH, C2000WARE_VERSION)))
C2000WARE_INCLUDE = os.path.join(C2000WARE_ROOT, 'device_support', '{DEVICE_NAME}', 'common', 'include')
C2000_DRIVERLIB_INCLUDE = os.path.join(C2000WARE_ROOT, 'driverlib', '{DEVICE_NAME}', 'driverlib')
# C2000 F29 Compiler
C29_CGT_VERSION = 'ti-cgt-c29_2.0.0.STS'
CG_TOOL_ROOT = os.path.abspath(os.getenv('CG_TOOL_ROOT', os.path.join(TOOLS_PATH, C29_CGT_VERSION)))
C29CLANG_CROSS_COMPILER = os.path.join(CG_TOOL_ROOT, 'bin', 'c29clang')
C29_CGT_INCLUDE = os.path.join(CG_TOOL_ROOT, 'include')
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
# M0SDK_VERSION='mspm0_sdk_2_10_00_04'
# M0SDK_PATH = os.path.abspath(os.getenv('M0SDK_PATH', os.path.join(TOOLS_PATH, M0SDK_VERSION)))
# M0SDK_INCLUDE = os.path.join(M0SDK_PATH, 'source')
# MSPM0_SOURCE_INCLUDE = os.path.join(M0SDK_PATH, 'source', 'third_party', 'CMSIS', 'Core', 'Include')

# MSPM33C Compiler
MSPM33C_CGT_VERSION= 'ti-cgt-armllvm_4.0.3.LTS'
MSPM33C_CROSS_COMPILER = os.path.join(ARM_LLVM_CGT_PATH, 'bin', 'tiarmclang')

# IWRL6432 Compiler (Cortex-M4F, tiarmclang toolchain)
IWRL6432_CGT_VERSION = 'ti-cgt-armllvm_5.1.1.LTS'
IWRL6432_ARM_LLVM_CGT_PATH = os.path.abspath(os.getenv('IWRL6432_ARM_LLVM_CGT_PATH', os.path.join(TOOLS_PATH, IWRL6432_CGT_VERSION)))
IWRL6432_CROSS_COMPILER = os.path.join(IWRL6432_ARM_LLVM_CGT_PATH, 'bin', 'tiarmclang')


CROSS_COMPILER_OPTIONS_C28 = (f"--abi=eabi -O3 --opt_for_speed=5 --c99 -v28 -ml -mt --gen_func_subsections --float_support={{FLOAT_SUPPORT}} -I{C2000_CGT_INCLUDE} -I{C2000_DRIVERLIB_INCLUDE} -I{C2000WARE_INCLUDE} -I. -Iartifacts --obj_directory=.")
CROSS_COMPILER_OPTIONS_F29H85 = (f"-O3 -ffast-math -I{C29_CGT_INCLUDE} -I.")
CROSS_COMPILER_OPTIONS_MSPM0 = (f"-Os -mcpu=cortex-m0plus -march=thumbv6m -mtune=cortex-m0plus -mthumb -mfloat-abi=soft -I. -Wno-return-type")
CROSS_COMPILER_OPTIONS_MSPM33C = (f"-O3 -mcpu=cortex-m33 -march=thumbv6m -mfpu=fpv5-sp-d16 -DARM_CPU_INTRINSICS_EXIST -mlittle-endian -mfloat-abi=hard -I. -Wno-return-type")
CROSS_COMPILER_OPTIONS_IWRL6432 = ("-DARM_CPU_INTRINSICS_EXIST -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O3 -Wl,-u,_c_int00 -Wno-return-type -march=armv7e-m -mthumb")

CROSS_COMPILER_OPTIONS_F280013 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32', DEVICE_NAME=TARGET_DEVICE_F280013.lower() + 'x')
CROSS_COMPILER_OPTIONS_F280015 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32', DEVICE_NAME=TARGET_DEVICE_F280015.lower() + 'x')
CROSS_COMPILER_OPTIONS_F28003 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32', DEVICE_NAME=TARGET_DEVICE_F28003.lower() + 'x')
CROSS_COMPILER_OPTIONS_F28004 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32', DEVICE_NAME=TARGET_DEVICE_F28004.lower() + 'x')
CROSS_COMPILER_OPTIONS_F2837 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32', DEVICE_NAME=TARGET_DEVICE_F2837.lower() + 'xd')
CROSS_COMPILER_OPTIONS_F28P65 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu64', DEVICE_NAME=TARGET_DEVICE_F28P65.lower() + 'x')
CROSS_COMPILER_OPTIONS_F28P55 = CROSS_COMPILER_OPTIONS_C28.format(FLOAT_SUPPORT='fpu32', DEVICE_NAME=TARGET_DEVICE_F28P55.lower() + 'x')
COMPILATION_C28_SOFT_TINPU = dict(target="c, ti-npu type=soft skip_normalize=true output_int=true", target_c_mcpu='c28', cross_compiler=CL2000_CROSS_COMPILER, )
COMPILATION_C28_HARD_TINPU = dict(target="c, ti-npu type=hard skip_normalize=true output_int=true", target_c_mcpu='c28', cross_compiler=CL2000_CROSS_COMPILER, )
COMPILATION_C28_HARD_TINPU_OPT_SPACE = dict(target="c, ti-npu type=hard skip_normalize=true output_int=true opt_for_space=true", target_c_mcpu='c28', cross_compiler=CL2000_CROSS_COMPILER, )
COMPILATION_F29H85_SOFT_TINPU = dict(target="c, ti-npu type=soft", target_c_mcpu='c29', cross_compiler=C29CLANG_CROSS_COMPILER, )
COMPILATION_MSPM0_SOFT_TINPU = dict(target="c, ti-npu type=soft skip_normalize=true output_int=true", target_c_mcpu='cortex-m0plus', cross_compiler=MSPM0_CROSS_COMPILER, )
COMPILATION_MSPM0_HARD_TINPU = dict(target="c, ti-npu skip_normalize=true output_int=true", target_c_mcpu='cortex-m0plus', cross_compiler=MSPM0_CROSS_COMPILER, )
COMPILATION_MSPM0_HARD_TINPU_OPT_SPACE = dict(target="c, ti-npu skip_normalize=true output_int=true opt_for_space=true", target_c_mcpu='cortex-m0plus', cross_compiler=MSPM0_CROSS_COMPILER, )
COMPILATION_MSPM33C_SOFT_TINPU = dict(target="c, ti-npu type=soft skip_normalize=true output_int=true", target_c_mcpu='cortex-m33', cross_compiler=MSPM33C_CROSS_COMPILER, )
COMPILATION_MSPM33C_HARD_TINPU = dict(target="c, ti-npu skip_normalize=true output_int=true", target_c_mcpu='cortex-m33', cross_compiler=MSPM33C_CROSS_COMPILER, )
COMPILATION_MSPM33C_HARD_TINPU_OPT_SPACE = dict(target="c, ti-npu skip_normalize=true output_int=true opt_for_space=true", target_c_mcpu='cortex-m33', cross_compiler=MSPM33C_CROSS_COMPILER, )
COMPILATION_IWRL6432_SOFT_TINPU = dict(target="c, ti-npu type=soft skip_normalize=true output_int=true", target_c_mcpu='cortex-m4', cross_compiler=IWRL6432_CROSS_COMPILER, )

PRESET_DESCRIPTIONS = {
    TARGET_DEVICE_IWRL6432: {TASK_TYPE_RADAR_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_IWRL6432_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_IWRL6432, )
            ),
        },

    },

}

SAMPLE_DATASET_DESCRIPTIONS = {
'Pose_and_Fall_Radar_Classification': {
    'common': {
        'task_type': TASK_TYPE_RADAR_CLASSIFICATION,
        'task_category': TASK_CATEGORY_RADAR_CLASSIFICATION,
    },
    'dataset': {
        'dataset_name': 'radar_human_pose_detection',
        'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/radar_human_pose_detection.zip',
    },
    'info': {
        'dataset_url': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/radar_human_pose_detection.zip',
        'dataset_detailed_name': 'Radar Human Pose and Fall Detection Example (IWRL6432)',
        'dataset_description': 'Example mmWave radar point-cloud classification dataset with 5 categories - standing, sitting, lying, falling, walking. Collected using the IWRL6432 radar sensor for human pose and fall detection use cases.',
        'dataset_size': None,
        'dataset_source': 'Generated by Texas Instruments at a specialised test bed',
        'dataset_license': 'TI Internal License',
    }
},
}