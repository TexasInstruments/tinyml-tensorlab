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

TASK_TYPE_IMAGE_CLASSIFICATION = 'image_classification'

TASK_TYPES = [
    TASK_TYPE_IMAGE_CLASSIFICATION,

]

# task_category
TASK_CATEGORY_IMAGE_CLASSIFICATION = 'image_classification'

TASK_CATEGORIES = [
   TASK_CATEGORY_IMAGE_CLASSIFICATION
]


def get_default_data_dir_for_task(task_category):
    """
    Determine the default data_dir based on task category.

    Currently vision module only supports classification which uses 'classes'.

    Args:
        task_category: Task category string

    Returns:
        str: 'classes' for image classification
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
TARGET_DEVICE_AM13E2 = 'AM13E2'
TARGET_DEVICE_CC2755 = 'CC2755'
TARGET_DEVICE_CC1352 = 'CC1352'
TARGET_DEVICE_CC1354 = 'CC1354'
TARGET_DEVICE_CC35X1 = 'CC35X1'

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
    TARGET_DEVICE_MSPM0G3519,
    TARGET_DEVICE_MSPM0G5187,
    TARGET_DEVICE_MSPM33C32,
    TARGET_DEVICE_MSPM33C34,
    TARGET_DEVICE_CC2755,
    TARGET_DEVICE_CC1352,
    TARGET_DEVICE_CC35X1,
    TARGET_DEVICE_CC1354,
    TARGET_DEVICE_AM13E2
]

# will not be listed in the GUI, but can be used in command line
TARGET_DEVICES_ADDITIONAL = [
    TARGET_DEVICE_AM263,
]

# include additional devices that are not currently supported in release.
TARGET_DEVICES_ALL = TARGET_DEVICES + TARGET_DEVICES_ADDITIONAL



# training backend
TRAINING_BACKEND_TINYML_TINYVERSE = 'tinyml_tinyverse'

# data directory names
DATA_DIR_IMAGES = 'images'

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





TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION = '\n * Tiny ML model development information: https://github.com/TexasInstruments/tinyml-tensorlab \n'


def _task_target_devices(task_type):
    """
    Derive target_devices for task_type from the vision model registry
    (vision has no tinyml_modelzoo backing, so models live under
    ai_modules/vision/training/tinyml_tinyverse/*.py). Deferred import
    avoids the constants <-> training import cycle.
    """
    from .training import get_model_descriptions as _get_model_descriptions
    devices = set()
    for model_desc in _get_model_descriptions(task_type=task_type).values():
        devices.update(model_desc['training'].get('target_devices', []))
    return sorted(devices)


TASK_DESCRIPTIONS = {

    TASK_TYPE_IMAGE_CLASSIFICATION: {
        'task_name': 'MNIST Classification',
        'target_module': 'vision',
        'target_devices': _task_target_devices(TASK_TYPE_IMAGE_CLASSIFICATION),
        'stages': ['dataset', 'data_processing_feature_extraction', 'training', 'compilation'],
    },

}
DATA_PREPROCESSING_DEFAULT = 'default'
DATA_PREPROCESSING_PRESET_DESCRIPTIONS = dict(
    default=dict(downsampling_factor=1), )
FEATURE_EXTRACTION_DEFAULT = 'default'
FEATURE_EXTRACTION_PRESET_DESCRIPTIONS = dict( 
    Mnist_Default=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['GRAYSCALE', 'RESIZE'], image_height = 28, image_width = 28, image_num_channel= 1, image_mean= 0.1307, image_scale= 0.3081, variables=1, data_proc_transforms=[]),  
        common=dict(task_type=TASK_TYPE_IMAGE_CLASSIFICATION), ),
    CoffeeBean_Default=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=["RGB","RESIZE"], 
        image_height = 128, image_width = 128, image_num_channel= 3, image_mean= (0.485,0.456,0.406), image_scale= (0.229, 0.224, 0.225), variables=3, data_proc_transforms=[]),  
        common=dict(task_type=TASK_TYPE_IMAGE_CLASSIFICATION), ),
    CoffeeBean_Augmentation_Default=dict(data_processing_feature_extraction=dict(feat_ext_transform=["RGB","RESIZE"], augmentation_transform=["RANDOM_HORIZONTAL_FLIP","RANDOM_VERTICAL_FLIP","RANDOM_ROTATION","COLOR_JITTER"], horizontal_flip_prob=0.5, vertical_flip_prob=0.5, random_rotation_deg=10, color_jitter_brightness=0.05, color_jitter_contrast=0.05, color_jitter_saturation=0.02, color_jitter_hue=0.005, image_height=128, image_width=128, image_num_channel=3, image_mean=(0.485,0.456,0.406), image_scale=(0.229,0.224,0.225), variables=3, data_proc_transforms=[]), common=dict(task_type=TASK_TYPE_IMAGE_CLASSIFICATION), ),
    MachineReadable_Default=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=["GRAYSCALE","RESIZE"], variables=1, image_height=28, image_width=28, image_num_channel=1, image_mean=0.5, image_scale=0.5, data_proc_transforms=[]),
        common=dict(task_type=TASK_TYPE_IMAGE_CLASSIFICATION), ),
)

DATASET_EXAMPLES = dict(
    default=dict(),
     mnist_image_classification=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/mnist_classes.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('Mnist_Default'), variables=1),
    ),
    coffee_bean_classification=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/coffee_bean_classification.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('CoffeeBean_Default'), variables=1),
    ),
    machine_readable_code_classification=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/machine_readable_code_classification.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('MachineReadable_Default'), variables=1),
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
C2000_CGT_VERSION = 'ti-cgt-c2000_25.11.0.LTS'
C2000_CG_ROOT = os.path.abspath(os.getenv('C2000_CG_ROOT', os.path.join(TOOLS_PATH, C2000_CGT_VERSION)))
CL2000_CROSS_COMPILER = os.path.join(C2000_CG_ROOT, 'bin', 'cl2000')
C2000_CGT_INCLUDE = os.path.join(C2000_CG_ROOT, 'include')
# C2000 F29 Compiler
C29_CGT_VERSION = 'ti-cgt-c29_2.2.1.LTS'
CG_TOOL_ROOT = os.path.abspath(os.getenv('CG_TOOL_ROOT', os.path.join(TOOLS_PATH, C29_CGT_VERSION)))
C29CLANG_CROSS_COMPILER = os.path.join(CG_TOOL_ROOT, 'bin', 'c29clang')
C29_CGT_INCLUDE = os.path.join(CG_TOOL_ROOT, 'include')
# C2000 F29H85 SDK --> For F29 there is device wise SDK. --> SDK is no longer required for TVM from ti-mcu-nnc-2.0.0
# F29H85_SDK_VERSION = 'f29h85x-sdk_1_01_00_00'
# F29H85_SDK_ROOT = os.path.abspath(os.getenv('F29H85_SDK_ROOT', os.path.join(TOOLS_PATH, F29H85_SDK_VERSION)))
# F29H85_SDK_INCLUDE = os.path.join(F29H85_SDK_ROOT, 'device_support', '{DEVICE_NAME}', 'common', 'include')
# F29H85_DRIVERLIB_INCLUDE = os.path.join(F29H85_SDK_ROOT, 'driverlib', '{DEVICE_NAME}', 'driverlib')

# MSPM0 Compiler
MSPM0_CGT_VERSION= 'ti-cgt-armllvm_5.1.1.LTS'
ARM_LLVM_CGT_PATH = os.path.abspath(os.getenv('ARM_LLVM_CGT_PATH', os.path.join(TOOLS_PATH, MSPM0_CGT_VERSION)))
MSPM0_CROSS_COMPILER = os.path.join(ARM_LLVM_CGT_PATH, 'bin', 'tiarmclang')
# MSPM0 SDK --> SDK is no longer required for TVM from ti-mcu-nnc-2.0.0
# M0SDK_VERSION='mspm0_sdk_2_10_00_04'
# M0SDK_PATH = os.path.abspath(os.getenv('M0SDK_PATH', os.path.join(TOOLS_PATH, M0SDK_VERSION)))
# M0SDK_INCLUDE = os.path.join(M0SDK_PATH, 'source')
# MSPM0_SOURCE_INCLUDE = os.path.join(M0SDK_PATH, 'source', 'third_party', 'CMSIS', 'Core', 'Include')

# MSPM33C Compiler
MSPM33C_CGT_VERSION= 'ti-cgt-armllvm_5.1.1.LTS'
MSPM33C_CROSS_COMPILER = os.path.join(ARM_LLVM_CGT_PATH, 'bin', 'tiarmclang')
AM13E2_CROSS_COMPILER = os.path.join(ARM_LLVM_CGT_PATH, 'bin', 'tiarmclang')

CROSS_COMPILER_OPTIONS_C28 = (f"--abi=eabi -O3 --opt_for_speed=5 --c99 -v28 -ml -mt --gen_func_subsections --float_support={{FLOAT_SUPPORT}} -I{C2000_CGT_INCLUDE} -I.")
CROSS_COMPILER_OPTIONS_F29H85 = (f"-O3 -ffast-math -I{C29_CGT_INCLUDE} -I.")
CROSS_COMPILER_OPTIONS_MSPM0 = (f"-Os -mcpu=cortex-m0plus -march=thumbv6m -mtune=cortex-m0plus -mthumb -mfloat-abi=soft -I. -Wno-return-type")
CROSS_COMPILER_OPTIONS_MSPM33C = (f"-O3 -mcpu=cortex-m33 -march=thumbv6m -mfpu=fpv5-sp-d16 -DARM_CPU_INTRINSICS_EXIST -mlittle-endian -mfloat-abi=hard -I. -Wno-return-type")
CROSS_COMPILER_OPTIONS_AM13E2 = f"-DARM_CPU_INTRINSICS_EXIST -mcpu=cortex-m33 -mfloat-abi=hard -mfpu=fpv5-sp-d16 -mlittle-endian -O3 -I. -Wno-return-type -march=thumbv8.1-m.main+cdecp0"

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

# AM13E2
COMPILATION_AM13E2_SOFT_TINPU = dict(target="c, ti-npu type=soft skip_normalize=true output_int=true", target_c_mcpu='cortex-m33', cross_compiler=AM13E2_CROSS_COMPILER, )
COMPILATION_AM13E2_HARD_TINPU = dict(target="c, ti-npu type=hard skip_normalize=true output_int=true", target_c_mcpu='cortex-m33', cross_compiler=AM13E2_CROSS_COMPILER, )
COMPILATION_AM13E2_SOFT_TINPU_REG = dict(target="c, ti-npu type=soft skip_normalize=true output_int=false", target_c_mcpu='cortex-m33', cross_compiler=AM13E2_CROSS_COMPILER, )
COMPILATION_AM13E2_SOFT_TINPU_AD = dict(target="c, ti-npu type=soft skip_normalize=true output_int=false", target_c_mcpu='cortex-m33', cross_compiler=AM13E2_CROSS_COMPILER, )
COMPILATION_AM13E2_SOFT_TINPU_FORECASTING = dict(target="c, ti-npu type=soft", target_c_mcpu='cortex-m33', cross_compiler=AM13E2_CROSS_COMPILER, )
COMPILATION_AM13E2_HARD_TINPU_OPT_SPACE = dict(target="c, ti-npu skip_normalize=true output_int=true opt_for_space=true", target_c_mcpu='cortex-m33', cross_compiler=AM13E2_CROSS_COMPILER, )

PRESET_DESCRIPTIONS = {
    TARGET_DEVICE_AM13E2: {
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
        }
    },
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
    TARGET_DEVICE_MSPM0G3519: {

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
    TARGET_DEVICE_MSPM33C32: {

        TASK_TYPE_IMAGE_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_MSPM33C_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM33C, )
            ),
        },

    },
    TARGET_DEVICE_MSPM33C34: {

        TASK_TYPE_IMAGE_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_MSPM33C_HARD_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM33C, )
            ),
            COMPILATION_FORCED_SOFT_NPU: dict(
                compilation=dict(**COMPILATION_MSPM33C_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM33C, )
            ),
              COMPILATION_NPU_OPT_FOR_SPACE: dict(
                compilation=dict(**COMPILATION_MSPM33C_HARD_TINPU_OPT_SPACE, cross_compiler_options=CROSS_COMPILER_OPTIONS_MSPM33C, )
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
        'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/mnist_classes.zip',
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
'coffee_bean_classification': {
    'common': {
        'task_type': TASK_TYPE_IMAGE_CLASSIFICATION,
        'task_category': TASK_CATEGORY_IMAGE_CLASSIFICATION,
    },
    'dataset': {
        'dataset_name': 'coffee_bean_classification',
        'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/coffee_bean_classification.zip',
    },
    'info': {
        'dataset_url': 'https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224',
        'dataset_detailed_name': 'Coffee Bean Dataset Resized 224x224',
        'dataset_description': 'The Coffee Bean dataset is an image classification dataset containing coffee bean images grouped into roast/bean categories such as Dark, Green, Light, and Medium. It can be used to train and evaluate image classification models that distinguish coffee bean roast levels using visual cues such as color, texture, and surface appearance.',
        'dataset_size': 'Four classes: Dark, Green, Light, and Medium. Images are resized to 224x224 in the original dataset source.',
        'dataset_source': 'Kaggle dataset published as Coffee Bean Dataset Resized (224 X 224)',
        'dataset_license': 'Refer to the dataset page for the latest license/usage terms.',
        'dataset_citation': 'Coffee Bean Dataset Resized (224 X 224), Kaggle. https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224',
    }
},
'machine_readable_code_classification': {
    'common': {
        'task_type': TASK_TYPE_IMAGE_CLASSIFICATION,
        'task_category': TASK_CATEGORY_IMAGE_CLASSIFICATION,
    },
    'dataset': {
        'dataset_name': 'machine_readable_code_classification',
        'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/machine_readable_code_classification.zip',
    },
    'info': {
        'dataset_url': 'Generated locally using the repository dataset generation script.',
        'dataset_detailed_name': 'Synthetic Machine Readable Code 28x28 Dataset',
        'dataset_description': 'The Machine Readable Code dataset is a synthetic low-resolution image classification dataset created for tiny image classification experiments. It contains three classes: QR code, barcode, and other/non-code images. QR images are generated using random alphanumeric payloads, barcode images are generated as Code128 barcodes without printed text, and the other class contains blank, noise, line, and block-pattern images. All images are converted to 28x28 grayscale binary PNGs.',
        'dataset_size': '3,000 images per class, 9,000 images total. Each image is 28x28 grayscale binary.',
        'dataset_source': 'Generated synthetically using Python packages qrcode, python-barcode, Pillow, and NumPy.',
    }
},
'mnist_classes': {
    'common': {
        'task_type': TASK_TYPE_IMAGE_CLASSIFICATION,
        'task_category': TASK_CATEGORY_IMAGE_CLASSIFICATION,
    },
    'dataset': {
        'dataset_name': 'mnist_classes',
        'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/mnist_classes.zip',
    },
    'info': {
        'dataset_url': 'http://yann.lecun.com/exdb/mnist/',
        'dataset_detailed_name': 'MNIST Handwritten Digits Classification',
        'dataset_description': 'MNIST dataset with handwritten digit images (0-9) organized as class folders for image classification workflows. Standard 28x28 grayscale images used for image classification training and evaluation.',
        'dataset_size': 'Multiple classes of handwritten digit images (28x28 grayscale)',
        'dataset_source': 'Created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges from NIST data',
        'dataset_license': 'Freely available for research and educational purposes',
        'dataset_citation': 'Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. "The MNIST Database of Handwritten Digits." 1998. http://yann.lecun.com/exdb/mnist/',
    }
},
}