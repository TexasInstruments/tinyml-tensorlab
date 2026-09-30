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

TASK_TYPE_AUDIO_CLASSIFICATION = 'audio_classification'

TASK_TYPES = [
    TASK_TYPE_AUDIO_CLASSIFICATION,

]

# task_category
TASK_CATEGORY_AUDIO_CLASSIFICATION = 'audio_classification'

TASK_CATEGORIES = [
   TASK_CATEGORY_AUDIO_CLASSIFICATION
]

# Mapping from task_type to task_category
TASK_TYPE_TO_CATEGORY = {
    TASK_TYPE_AUDIO_CLASSIFICATION: TASK_CATEGORY_AUDIO_CLASSIFICATION,
}


def get_task_category(task_type):
    """
    Get the task category for a given task type.

    Args:
        task_type: One of TASK_TYPE_* constants or task_category string

    Returns:
        str: The task category (one of TASK_CATEGORY_* constants)
    """
    # If it's already a task_category, return it
    if task_type in TASK_CATEGORIES:
        return task_type

    # Otherwise look it up in the mapping
    return TASK_TYPE_TO_CATEGORY.get(task_type, TASK_CATEGORY_AUDIO_CLASSIFICATION)


def get_default_data_dir_for_task(task_category):
    """
    Determine the default data_dir based on task category.

    Audio currently only supports classification, which organizes data by class folders.

    Args:
        task_category: One of TASK_CATEGORY_* constants

    Returns:
        str: 'classes' for audio classification
    """
    return DATA_DIR_CLASSES


# target_device
TARGET_DEVICE_MSPM0G5187 = 'MSPM0G5187'

TARGET_DEVICES = [
    TARGET_DEVICE_MSPM0G5187,
]

# will not be listed in the GUI, but can be used in command line
TARGET_DEVICES_ADDITIONAL = []

# include additional devices that are not currently supported in release.
TARGET_DEVICES_ALL = TARGET_DEVICES + TARGET_DEVICES_ADDITIONAL

# data directory names
DATA_DIR_CLASSES = 'classes'

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
 
    TASK_TYPE_AUDIO_CLASSIFICATION: 64,

}


TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION = '\n * Tiny ML model development information: https://github.com/TexasInstruments/tinyml-tensorlab \n'


def _task_target_devices(task_type):
    """
    Derive target_devices for task_type from the audio model registry
    (audio has no tinyml_modelzoo backing, so models live under
    ai_modules/audio/training/tinyml_tinyverse/*.py). Deferred import
    avoids the constants <-> training import cycle.
    """
    from .training import get_model_descriptions as _get_model_descriptions
    devices = set()
    for model_desc in _get_model_descriptions(task_type=task_type).values():
        devices.update(model_desc['training'].get('target_devices', []))
    return sorted(devices)


TASK_DESCRIPTIONS = {

    TASK_TYPE_AUDIO_CLASSIFICATION: {
        'task_name': 'Audio Classification',
        'task_group': 'timeseries',
        'target_module': 'audio',
        'target_devices': _task_target_devices(TASK_TYPE_AUDIO_CLASSIFICATION),
        'stages': ['dataset', 'data_processing_feature_extraction', 'training', 'compilation'],
        'application_specific': False,
        'checkDataEnough': False,
        'task_category': TASK_CATEGORY_AUDIO_CLASSIFICATION
    },

}
DATA_PREPROCESSING_DEFAULT = 'default'
DATA_PREPROCESSING_PRESET_DESCRIPTIONS = dict(
    default=dict(downsampling_factor=1), )
FEATURE_EXTRACTION_DEFAULT = 'default'
FEATURE_EXTRACTION_PRESET_DESCRIPTIONS = dict(
    GoogleSpeechCommands_MFCC_Default=dict(data_processing_feature_extraction=dict(audio_feature="MFCC",n_mfcc=10,n_mels=40,frame_length_ms=30,frame_step_ms=20,normalize_audio=True,mono=True,variables=1,feat_ext_transform=["MFCC"],data_proc_transforms=[]),common=dict(task_type=TASK_TYPE_AUDIO_CLASSIFICATION),),
    # Matches MSPM0G5187 firmware LPC pipeline exactly (lpc_defs.h):
    #   FS=8000, WINDOW_LEN=240 (30ms), FRONTEND_FRAME_LEN=160 (20ms),
    #   LPC_ORDER=10, NUM_FREQ=70, 2-second context window (100 frames x 20ms)
    MSPM0_LPC_8k=dict(data_processing_feature_extraction=dict(sampling_rate=8000,audio_duration_ms=2000,audio_feature="LPC",nlpc=70,lpc_order=10,frame_length_ms=30,frame_step_ms=20,normalize_audio=True,mono=True,variables=1,feat_ext_transform=["LPC"],data_proc_transforms=[]),common=dict(task_type=TASK_TYPE_AUDIO_CLASSIFICATION),),
    # Matches MSPM0G5187 firmware I2S pipeline: 44.1kHz stereo decimated by 10 → 8820 Hz
    MSPM0_LPC_8820=dict(data_processing_feature_extraction=dict(sampling_rate=8820,audio_duration_ms=2000,audio_feature="LPC",nlpc=70,lpc_order=10,frame_length_ms=30,frame_step_ms=20,normalize_audio=True,mono=True,variables=1,feat_ext_transform=["LPC"],data_proc_transforms=[]),common=dict(task_type=TASK_TYPE_AUDIO_CLASSIFICATION),),
    WakeWordDetection_Filterbank_Default=dict(data_processing_feature_extraction=dict(audio_feature="FB",normalize_audio=False,mono=True,variables=1,feat_ext_transform=["FB"],data_proc_transforms=[],fb_conv_kernel=64,fb_output_channel=64,fb_bitwidth=2,fb_context_ms=20,fb_conv_stride=4,input_bit_depth=16),common=dict(task_type=TASK_TYPE_AUDIO_CLASSIFICATION),),
    GlassBreakDetection_512FFT_Default=dict(data_processing_feature_extraction=dict(audio_feature="FFT",normalize_audio=False,mono=True,variables=1,feat_ext_transform=["FFT_Q15", "Q15_SCALE", "Q15_MAG", "BINNING", "CONCAT"],data_proc_transforms=[],frame_size=512,feature_size_per_frame=32,num_frame_concat=86,q15_scale_factor=8),common=dict(task_type=TASK_TYPE_AUDIO_CLASSIFICATION),),
)

DATASET_EXAMPLES = dict(
    default=dict(),
     google_speech_commands_12class=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/google_speech_commands_12class.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('GoogleSpeechCommands_MFCC_Default'), variables=1),
    ),
    cough_detection=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/cough_detection.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('MSPM0_LPC_8820'), variables=1),
    ),
    wake_word_detection=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/wake_word_detection.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('WakeWordDetection_Filterbank_Default'), variables=1),
    ),
    glass_break_detection=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/glass_break_detection.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('GlassBreakDetection_512FFT_Default'), variables=1),
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

# MSPM0 Compiler
MSPM0_CGT_VERSION= 'ti-cgt-armllvm_5.1.1.LTS'
ARM_LLVM_CGT_PATH = os.path.abspath(os.getenv('ARM_LLVM_CGT_PATH', os.path.join(TOOLS_PATH, MSPM0_CGT_VERSION)))
MSPM0_CROSS_COMPILER = os.path.join(ARM_LLVM_CGT_PATH, 'bin', 'tiarmclang')
# MSPM0 SDK --> SDK is no longer required for TVM from ti-mcu-nnc-2.0.0
# M0SDK_VERSION='mspm0_sdk_2_10_00_04'
# M0SDK_PATH = os.path.abspath(os.getenv('M0SDK_PATH', os.path.join(TOOLS_PATH, M0SDK_VERSION)))
# M0SDK_INCLUDE = os.path.join(M0SDK_PATH, 'source')
# MSPM0_SOURCE_INCLUDE = os.path.join(M0SDK_PATH, 'source', 'third_party', 'CMSIS', 'Core', 'Include')

CROSS_COMPILER_OPTIONS_MSPM0 = ("-Os -mcpu=cortex-m0plus -march=thumbv6m -mtune=cortex-m0plus -mthumb -mfloat-abi=soft -I. -Wno-return-type")

COMPILATION_MSPM0_SOFT_TINPU = dict(target="c, ti-npu type=soft skip_normalize=true output_int=true", target_c_mcpu='cortex-m0plus', cross_compiler=MSPM0_CROSS_COMPILER, )
COMPILATION_MSPM0_HARD_TINPU = dict(target="c, ti-npu skip_normalize=true output_int=true", target_c_mcpu='cortex-m0plus', cross_compiler=MSPM0_CROSS_COMPILER, )
COMPILATION_MSPM0_HARD_TINPU_OPT_SPACE = dict(target="c, ti-npu skip_normalize=true output_int=true opt_for_space=true", target_c_mcpu='cortex-m0plus', cross_compiler=MSPM0_CROSS_COMPILER, )

# =============================================================================
# Device Compilation Profiles - Defines compilation characteristics per device
# =============================================================================

# Cross-compiler options lookup
_CROSS_COMPILER_OPTIONS = {
    TARGET_DEVICE_MSPM0G5187: CROSS_COMPILER_OPTIONS_MSPM0,
}

# Device profiles: base compilation config, soft/opt_for_space variants, whether it has hardware NPU.
# For this release, only MSPM0G5187 is supported for audio_classification.
_DEVICE_PROFILES = {
    TARGET_DEVICE_MSPM0G5187: {
        'compilation_base': COMPILATION_MSPM0_HARD_TINPU,
        'compilation_soft': COMPILATION_MSPM0_SOFT_TINPU,
        'compilation_opt_space': COMPILATION_MSPM0_HARD_TINPU_OPT_SPACE,
        'has_hard_npu': True,
    },
}


def _build_preset_descriptions():
    """
    Generate PRESET_DESCRIPTIONS from device profiles.

    Every device with a profile gets a default compilation preset. Devices with a
    hardware NPU additionally get a "forced soft NPU" preset, and an "optimized for
    space" preset if the profile defines one. This mirrors the timeseries module's
    device-profile-driven preset generation so device support doesn't have to be
    hand-duplicated per task_type.
    """
    result = {}

    for device, profile in _DEVICE_PROFILES.items():
        cross_opts = _CROSS_COMPILER_OPTIONS[device]
        base_config = profile['compilation_base']

        result[device] = {
            TASK_TYPE_AUDIO_CLASSIFICATION: {
                COMPILATION_DEFAULT: dict(
                    compilation=dict(**base_config, cross_compiler_options=cross_opts)
                )
            }
        }

        if profile.get('has_hard_npu'):
            soft_config = profile.get('compilation_soft', profile['compilation_base'])
            result[device][TASK_TYPE_AUDIO_CLASSIFICATION][COMPILATION_FORCED_SOFT_NPU] = dict(
                compilation=dict(**soft_config, cross_compiler_options=cross_opts)
            )

            if 'compilation_opt_space' in profile:
                result[device][TASK_TYPE_AUDIO_CLASSIFICATION][COMPILATION_NPU_OPT_FOR_SPACE] = dict(
                    compilation=dict(**profile['compilation_opt_space'], cross_compiler_options=cross_opts)
                )

    return result


# Generate PRESET_DESCRIPTIONS from device profiles
PRESET_DESCRIPTIONS = _build_preset_descriptions()

SAMPLE_DATASET_DESCRIPTIONS = {
'google_speech_commands_12class': {
    'common': {
        'task_type': TASK_TYPE_AUDIO_CLASSIFICATION,
        'task_category': TASK_CATEGORY_AUDIO_CLASSIFICATION,
    },
    'dataset': {
        'dataset_name': 'google_speech_commands_12class',
        'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/google_speech_commands_12class.zip',
    },
    'info': {
        'dataset_url': 'https://pytorch.org/audio/stable/generated/torchaudio.datasets.SPEECHCOMMANDS.html',
        'dataset_detailed_name': 'Google Speech Commands Classification',
        'dataset_description': 'The Google Speech Commands 12-class dataset is a TensorLab-compatible keyword spotting dataset generated from Google Speech Commands v0.02 using TorchAudio. It contains 10 selected command classes: down, go, left, no, off, on, right, stop, up, and yes. All remaining spoken-word classes are mapped into _unknown_, and _silence_ samples are generated from the original _background_noise_ audio files. The dataset is arranged in class-folder format for audio classification workflows.',
        'dataset_size': '12 classes: 10 keyword classes, 1 _unknown_ class, and 1 _silence_ class. Each sample is a 1-second 16 kHz WAV audio clip.',
        'dataset_source': 'Derived from Google Speech Commands v0.02 downloaded using torchaudio.datasets.SPEECHCOMMANDS.',
        'dataset_license': 'Refer to the original Google Speech Commands dataset license and usage terms.',
        'dataset_citation': 'Warden, P. Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition. arXiv:1804.03209, 2018.',
    }
},
'cough_detection': {
    'common': {
        'task_type': TASK_TYPE_AUDIO_CLASSIFICATION,
        'task_category': TASK_CATEGORY_AUDIO_CLASSIFICATION,
    },
    'dataset': {
        'dataset_name': 'cough_detection',
        'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/cough_detection.zip',
    },
    'info': {
        'dataset_url': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/cough_detection.zip',
        'dataset_detailed_name': 'Cough Detection',
        'dataset_description': 'Binary audio classification dataset (cough vs other) captured with TI EdgeAI Studio via the LP-MSPM0G5187 I2S peripheral at 8820 Hz. Cough clips are augmented with attenuated, noise-mixed copies so the model discriminates on spectral shape rather than loudness.',
        'dataset_size': '2 classes: cough, other. 2-second WAV clips at 8820 Hz.',
        'dataset_source': 'Generated by Texas Instruments using TI EdgeAI Studio',
        'dataset_license': 'TI Internal License',
    }
},
'wake_word_detection': {
    'common': {
        'task_type': TASK_TYPE_AUDIO_CLASSIFICATION,
        'task_category': TASK_CATEGORY_AUDIO_CLASSIFICATION,
    },
    'dataset': {
        'dataset_name': 'wake_word_detection',
        'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/wake_word_detection.zip',
    },
    'info': {
        'dataset_url': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/wake_word_detection.zip',
        'dataset_detailed_name': 'Wake Word Detection',
        'dataset_description': 'Binary audio classification dataset (ok_kilby vs other). Wake word used for training is "Ok Kilby"',
        'dataset_size': None,
        'dataset_source': 'Generated by Texas Instruments',
        'dataset_license': 'TI Internal License',
    }
},
'glass_break_detection': {
    'common': {
        'task_type': TASK_TYPE_AUDIO_CLASSIFICATION,
        'task_category': TASK_CATEGORY_AUDIO_CLASSIFICATION,
    },
    'dataset': {
        'dataset_name': 'glass_break_detection',
        'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/glass_break_detection.zip',
    },
    'info': {
        'dataset_url': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/glass_break_detection.zip',
        'dataset_detailed_name': 'Glass Break Detection',
        'dataset_description': 'Binary audio classification dataset (glass_break vs other) for detecting the sound of breaking glass.',
        'dataset_size': None,
        'dataset_source': 'Generated by Texas Instruments',
        'dataset_license': 'TI Internal License',
    }
},
}