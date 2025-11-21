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

TASK_TYPE_IMAGE_CLASSIFICATION = 'image_classification'

TASK_TYPES = [
    TASK_TYPE_IMAGE_CLASSIFICATION,

]

# task_category
TASK_CATEGORY_IMAGE_CLASSIFICATION = 'image_classification'

TASK_CATEGORIES = [
   TASK_CATEGORY_IMAGE_CLASSIFICATION
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
    TARGET_DEVICE_F29H85,
    TARGET_DEVICE_MSPM0G3507,
    TARGET_DEVICE_MSPM0G5187,
    TARGET_DEVICE_CC2755
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
 
    TASK_TYPE_IMAGE_CLASSIFICATION: 64,

}

TARGET_SDK_VERSION_C2000 = '6.0'
TARGET_SDK_RELEASE_C2000 = '06_00_00'

TARGET_SDK_VERSION_F29H85 = '1.00'
TARGET_SDK_RELEASE_F29H85 = '01_00_00'

TARGET_SDK_VERSION_MSPM0 = "2.08.00.03"
TARGET_SDK_RELEASE_MSPM0 = '2_08_00_03'

TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION = '\n * Tiny ML model development information: https://github.com/TexasInstruments/tinyml-tensorlab \n'

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
    f'''80MHz Arm® Cortex®-M0+ MCU with 128KB flash 32KB SRAM 2x4Msps ADC, DAC, USB, TI-NPU
* More details : https://www.ti.com/product/MSPM0G5187

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_MSPM0G5187}

Additional information:
{TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

# higher device_selection_factor indicates higher performance device.
TARGET_DEVICE_DESCRIPTIONS = {
    TARGET_DEVICE_MSPM0G3507: {
        'device_name': TARGET_DEVICE_MSPM0G3507,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 0,
        'device_details': TARGET_DEVICE_DETAILS_MSPM0G3507,
        'sdk_version': TARGET_SDK_VERSION_MSPM0,
        'sdk_release': TARGET_SDK_RELEASE_MSPM0,
    },
    TARGET_DEVICE_F280013: {
        'device_name': TARGET_DEVICE_F280013,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 1,
        'device_details': TARGET_DEVICE_DETAILS_F280013,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_F280015: {
        'device_name': TARGET_DEVICE_F280015,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 2,
        'device_details': TARGET_DEVICE_DETAILS_F280015,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_F28003: {
        'device_name': TARGET_DEVICE_F28003,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 3,
        'device_details': TARGET_DEVICE_DETAILS_F28003,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_F28004: {
        'device_name': TARGET_DEVICE_F28004,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 4,
        'device_details': TARGET_DEVICE_DETAILS_F28004,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_F2837: {
        'device_name': TARGET_DEVICE_F2837,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 5,
        'device_details': TARGET_DEVICE_DETAILS_F2837,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_F28P65: {
        'device_name': TARGET_DEVICE_F28P65,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 6,
        'device_details': TARGET_DEVICE_DETAILS_F28P65,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_F28P55: {
        'device_name': TARGET_DEVICE_F28P55,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 7,
        'device_details': TARGET_DEVICE_DETAILS_F28P55,
        'sdk_version': TARGET_SDK_VERSION_C2000,
        'sdk_release': TARGET_SDK_RELEASE_C2000,
    },
    TARGET_DEVICE_AM263: {
        'device_name': TARGET_DEVICE_AM263,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 8,
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
    TARGET_DEVICE_MSPM0G5187: {
        'device_name': TARGET_DEVICE_MSPM0G5187,
        'device_type': TARGET_DEVICE_TYPE_MCU,
        'device_selection_factor': 9,
        'device_details': TARGET_DEVICE_DETAILS_MSPM0G5187,
        'sdk_version': TARGET_SDK_VERSION_MSPM0,
        'sdk_release': TARGET_SDK_RELEASE_MSPM0,
    },
}

TASK_DESCRIPTIONS = {
  
    TASK_TYPE_IMAGE_CLASSIFICATION: {
        'task_name': 'MNIST Classification',
        'target_module': 'vision',
        'target_devices': TARGET_DEVICES,
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
        common=dict(task_type=TASK_TYPE_IMAGE_CLASSIFICATION), ),
)

DATASET_EXAMPLES = dict(
    default=dict(),
     mnist_image_classification=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/mnist_classes.zip'),
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

TOOLS_PATH = os.path.abspath(os.getenv('TOOLS_PATH', os.path.join(f'{HOME_DIR}', 'bin')))
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
# M0SDK_VERSION='mspm0_sdk_2_08_00_03'
# M0SDK_PATH = os.path.abspath(os.getenv('M0SDK_PATH', os.path.join(TOOLS_PATH, M0SDK_VERSION)))
# M0SDK_INCLUDE = os.path.join(M0SDK_PATH, 'source')
# MSPM0_SOURCE_INCLUDE = os.path.join(M0SDK_PATH, 'source', 'third_party', 'CMSIS', 'Core', 'Include')


CROSS_COMPILER_OPTIONS_C28 = (f"--abi=eabi -O3 --opt_for_speed=5 --c99 -v28 -ml -mt --gen_func_subsections --float_support={{FLOAT_SUPPORT}} -I{C2000_CGT_INCLUDE} -I{C2000_DRIVERLIB_INCLUDE} -I{C2000WARE_INCLUDE} -I. -Iartifacts --obj_directory=.")
CROSS_COMPILER_OPTIONS_F29H85 = (f"-O3 -ffast-math -I{C29_CGT_INCLUDE} -I.")
CROSS_COMPILER_OPTIONS_MSPM0 = (f"-Os -mcpu=cortex-m0plus -march=thumbv6m -mtune=cortex-m0plus -mthumb -mfloat-abi=soft -I. -Wno-return-type")

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

PRESET_DESCRIPTIONS = {
    TARGET_DEVICE_AM263: {
       
    },
    TARGET_DEVICE_F280015: {
    
    },
    TARGET_DEVICE_F28004: {
    
    },
    TARGET_DEVICE_F28P65: {
    
    },
    TARGET_DEVICE_F28P55: {
      
    },
    TARGET_DEVICE_F2837: {
    
    },
    TARGET_DEVICE_F29H85: {
       
    },
    TARGET_DEVICE_MSPM0G3507: {

           TASK_TYPE_IMAGE_CLASSIFICATION: {
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
    TARGET_DEVICE_MSPM0G5187: {
       
        TASK_TYPE_IMAGE_CLASSIFICATION: {
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
    
}

SAMPLE_DATASET_DESCRIPTIONS = {
'mnist_image_classification': {
    'common': {
        'task_type': TASK_TYPE_IMAGE_CLASSIFICATION,
        'task_category': TASK_CATEGORY_IMAGE_CLASSIFICATION,
    },
    'dataset': {
        'dataset_name': 'mnist_image_classification',
        'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/mnist_classes.zip',
    },
    'info': {
        'dataset_url': 'http://yann.lecun.com/exdb/mnist/',
        'dataset_detailed_name': 'Modified National Institute of Standards and Technology (MNIST) Database',
        'dataset_description': 'The MNIST dataset is a large database of handwritten digits (0–9) commonly used for training and testing in the field of machine learning. It consists of 60,000 training images and 10,000 test images, each 28x28 grayscale. MNIST was created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges as a benchmark for image classification research.',
        'dataset_size': '60,000 training images, 10,000 test images (28x28 grayscale)',
        'dataset_source': 'Created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges from NIST data',
        'dataset_license': 'Freely available for research and educational purposes',
        'dataset_citation': 'Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. "The MNIST Database of Handwritten Digits." 1998. http://yann.lecun.com/exdb/mnist/',
    }
},
}