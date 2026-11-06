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

TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION = 'radar_point_cloud_classification'
TASK_TYPE_RADAR_SURFACE_CLASSIFICATION = 'radar_surface_classification'

TASK_TYPES = [
    TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION,
    TASK_TYPE_RADAR_SURFACE_CLASSIFICATION,
]

# task_category
TASK_CATEGORY_RADAR_CLASSIFICATION = 'radar_classification'

TASK_CATEGORIES = [
   TASK_CATEGORY_RADAR_CLASSIFICATION,
]

TASK_TYPE_TO_CATEGORY = {
    TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION: TASK_CATEGORY_RADAR_CLASSIFICATION,
    TASK_TYPE_RADAR_SURFACE_CLASSIFICATION: TASK_CATEGORY_RADAR_CLASSIFICATION
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
TARGET_DEVICE_xWRL6432 = 'xWRL6432'

TARGET_DEVICES = [
    TARGET_DEVICE_xWRL6432,
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

TRAINING_BATCH_SIZE_DEFAULT = {
    TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION: 64,
    TASK_TYPE_RADAR_SURFACE_CLASSIFICATION: 64
}

TINYML_TARGET_DEVICE_ADDITIONAL_INFORMATION = '\n * Tiny ML model development information: https://github.com/TexasInstruments/tinyml-tensorlab \n'

TASK_DESCRIPTIONS = {
    TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION: {
        'task_name': 'Radar Posture and Fall Detection',
        'target_module': 'radar',
        'target_devices': [TARGET_DEVICE_xWRL6432],
        'stages': ['dataset', 'data_processing_feature_extraction', 'training', 'compilation'],
    },
    TASK_TYPE_RADAR_SURFACE_CLASSIFICATION: {
            'task_name': 'Radar Surface Classification',
            'target_module': 'radar',
            'target_devices': [TARGET_DEVICE_xWRL6432],
            'stages': ['dataset', 'data_processing_feature_extraction', 'training', 'compilation'],
        },
}

DATA_PREPROCESSING_DEFAULT = 'default'
DATA_PREPROCESSING_PRESET_DESCRIPTIONS = dict()
FEATURE_EXTRACTION_DEFAULT = 'default'

FEATURE_EXTRACTION_PRESET_DESCRIPTIONS = dict(
    # Full pipeline with Point Cloud + Track Data + Simple Window quality filtering
    # Default: 5 PC features * 8 max_points + 9 track features = 49 per frame * 8 frames = 392
    Generic_392Input_PointCloudWithTrack_SimpleWindow=dict(
        data_processing_feature_extraction=dict(
            data_proc_transforms=['Radar Point Cloud Data', 'Radar Track Data', 'Simple Window'],
            frame_size=8,
            stride_size=0.125,
            radar_point_cloud_data_selector=['x','y', 'z','doppler','snr'],
            radar_point_cloud_coordinate_system='relative',
            radar_point_cloud_relative_coordinate_filter=['x','y''z'],
            radar_point_cloud_spatial_filter_bounds=[None, None, None, None, None, None],
            radar_track_data_selector=['x', 'y', 'z', 'velocity', 'acceleration'],
            radar_track_velocity_bounds=[None, None],
            radar_track_acceleration_bounds=[None, None],
            radar_track_spatial_filter_bounds=[None, None, None, None, None, None],
            radar_max_points_per_frame=8,
            radar_min_points_per_window=12,
            radar_point_cloud_structure_strategy='highest only',
            radar_point_cloud_sort_by='z',
            radar_frames_quality_index='per_window',
            radar_feature_ordering='frame_major',
        ),
        common=dict(task_type=TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION), ),

    # Simplified preset: Only Simple Window (for pre-refined datasets)
    # User provides already processed data, just needs windowing with frame_size and stride_size
    # feature_size_per_frame = whatever the pre-processed data provides per frame
    Generic_Radar_SimpleWindowOnly=dict(
        data_processing_feature_extraction=dict(
            data_proc_transforms=['Simple Window'],
            frame_size=8,
            stride_size=0.125,
        ),
        common=dict(task_type=TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION), ),

    # Range Classification preset: Radar Range Data + Simple Window
    # Uses FFT range bins for surface/material classification
    # 26 bins (6-31 inclusive) * 5 frames = 130
    Generic_130Input_Radar_SurfaceClassification=dict(
        data_processing_feature_extraction=dict(
            data_proc_transforms=['Radar Range Data', 'Simple Window'],
            radar_range_bins_bounds=[6, 31],
            frame_size=5,
            stride_size=0.2,
            radar_frames_quality_index=None,
            variables=1,
        ),
        common=dict(task_type=TASK_TYPE_RADAR_SURFACE_CLASSIFICATION), ),

     Generic_248Input_Posture_and_Fall_Detection=dict(
            data_processing_feature_extraction=dict(
                data_proc_transforms=['Radar Point Cloud Data', 'Radar Track Data', 'Simple Window'],
                frame_size=8,
                stride_size=0.125,
                radar_point_cloud_data_selector=['y', 'z','snr'],
                radar_point_cloud_coordinate_system='relative',
                radar_point_cloud_relative_coordinate_filter=['y'],
                radar_point_cloud_spatial_filter_bounds=[None, None, -10.0, 10.0, -10.0, 10.0],
                radar_track_data_selector=['z', 'velocity', 'acceleration'],
                radar_track_velocity_bounds=[-10.0, 10.0],
                radar_track_acceleration_bounds=[-10.0, 10.0],
                radar_track_spatial_filter_bounds=[None, None, None, None, -10.0, 10.0],
                radar_max_points_per_frame=8,
                radar_min_points_per_window=12,
                radar_point_cloud_structure_strategy='highest and lowest',
                radar_point_cloud_sort_by='z',
                radar_frames_quality_index='per_window',
                radar_feature_ordering='frame_major',
            ),
            common=dict(task_type=TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION), ),
    
)

DATASET_EXAMPLES = dict(
    default=dict(),
     radar_point_cloud_classification=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/posture_and_fall_detection_dataset.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('Generic_248Input_Posture_and_Fall_Detection'), variables=1),
    ),
    radar_surface_classification=dict(
        dataset=dict(input_data_path='https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/radar_grass_surface_dataset.zip'),
        data_processing_feature_extraction=dict(feature_extraction_name=FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.get('Generic_130Input_Radar_SurfaceClassification'), variables=1),
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

# Radar Devices Cross compiler settings
xWRL6432_CGT_VERSION = 'ti-cgt-armllvm_5.1.1.LTS'
xWRL6432_ARM_LLVM_CGT_PATH = os.path.abspath(os.getenv('xWRL6432_ARM_LLVM_CGT_PATH', os.path.join(TOOLS_PATH, xWRL6432_CGT_VERSION)))
xWRL6432_CROSS_COMPILER = os.path.join(xWRL6432_ARM_LLVM_CGT_PATH, 'bin', 'tiarmclang')

CROSS_COMPILER_OPTIONS_xWRL6432 = ("-Os -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -march=armv7e-m -mthumb -Wno-return-type -DARM_CPU_INTRINSICS_EXIST -Wl,-u,_c_int00")

COMPILATION_xWRL6432_SOFT_TINPU = dict(target="c, ti-npu type=soft skip_normalize=true output_int=true", target_c_mcpu='cortex-m4', cross_compiler=xWRL6432_CROSS_COMPILER, )

PRESET_DESCRIPTIONS = {
    TARGET_DEVICE_xWRL6432: {TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_xWRL6432_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_xWRL6432, )
            ),
        },
        TASK_TYPE_RADAR_SURFACE_CLASSIFICATION: {
            COMPILATION_DEFAULT: dict(
                compilation=dict(**COMPILATION_xWRL6432_SOFT_TINPU, cross_compiler_options=CROSS_COMPILER_OPTIONS_xWRL6432, )
            ),
        },
    },

}

SAMPLE_DATASET_DESCRIPTIONS = {
'radar_point_cloud_classification': {
    'common': {
        'task_type': TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION,
        'task_category': TASK_CATEGORY_RADAR_CLASSIFICATION,
    },
    'dataset': {
        'dataset_name': 'radar_human_pose_detection',
        'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/posture_and_fall_detection_dataset.zip',
    },
    'info': {
        'dataset_url': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/posture_and_fall_detection_dataset.zip',
        'dataset_detailed_name': 'Radar Human Pose and Fall Detection Example (xWRL6432)',
        'dataset_description': 'Example mmWave radar point-cloud classification dataset with 5 categories - standing, sitting, lying, falling, walking. Collected using the xWRL6432 radar sensor for human pose and fall detection use cases.',
        'dataset_size': None,
        'dataset_source': 'Generated by Texas Instruments at a specialised test bed',
        'dataset_license': 'TI Internal License',
    }
},

'radar_surface_classification': {
    'common': {
        'task_type': TASK_TYPE_RADAR_SURFACE_CLASSIFICATION,
        'task_category': TASK_CATEGORY_RADAR_CLASSIFICATION,
    },
    'dataset': {
        'dataset_name': 'radar_surface_classification',
        'input_data_path': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/radar_grass_surface_dataset.zip',
    },
    'info': {
        'dataset_url': 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/radar_grass_surface_dataset.zip',
        'dataset_detailed_name': 'Radar Surface Classification - Grass/Non Grass Surface Example (xWRL6432)',
        'dataset_description': 'Example mmWave radar Surface Classification Example',
        'dataset_size': None,
        'dataset_source': 'Generated by Texas Instruments at a specialised test bed',
        'dataset_license': 'TI Internal License',
    }
},
}