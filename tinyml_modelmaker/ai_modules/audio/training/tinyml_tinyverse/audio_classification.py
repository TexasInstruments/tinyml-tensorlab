#################################################################################
# Copyright (c) 2023-2026, Texas Instruments
# All Rights Reserved.
#################################################################################

import os
from copy import deepcopy

from tinyml_tinyverse.references.audio_classification import test_onnx as test
from tinyml_tinyverse.references.audio_classification import train
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

from ..... import utils
from ... import constants

from .audio_base import (
    BaseAudioModelTraining,
    create_template_model_description,
    get_model_descriptions_filtered,
    get_model_description_by_name,
)


model_info_str = "Inference time numbers are for comparison purposes only. (Input Size: {})"

template_model_description = create_template_model_description(
    task_category=constants.TASK_TYPE_AUDIO_CLASSIFICATION,
    task_type=constants.TASK_TYPE_AUDIO_CLASSIFICATION,
    dataset_loader='GenericAudioDataset',
    batch_size_key=constants.TASK_TYPE_AUDIO_CLASSIFICATION,
)

_model_descriptions = {
    'DSCNN_NPU': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Lenet5.\nCNN for audio classification.\n'
                          '2 Conv+BatchNorm+Relu+MaxPool layers + 3 Linear layers.'
        ),
        'training': dict(
            model_training_id='CNN_AUDIO_DSCNN',
            model_name='DSCNN_NPU',
            learning_rate=0.1,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_AUDIO_CLASSIFICATION],
            target_devices=[
                constants.TARGET_DEVICE_MSPM0G5187,
            ],
            properties=[dict(type="group", dynamic=True, script="audio.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="audio.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'DSCNN_32K_NPU': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='SRAM/Flash-optimized DSCNN for audio classification.\n'
                          '48 filters, DW 3x1 kernels, stride-2 downsample in DS block 1.\n'
                          'Target: SRAM < 32 KB, Flash < 128 KB, W8 quantization.'
        ),
        'training': dict(
            model_training_id='CNN_AUDIO_DSCNN_32K_NPU',
            model_name='DSCNN_32K_NPU',
            learning_rate=0.001,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_AUDIO_CLASSIFICATION],
            target_devices=[
                constants.TARGET_DEVICE_MSPM0G5187,
            ],
            properties=[dict(type="group", dynamic=True, script="audio.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="audio.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'DSCNN_GB_NPU': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Glass-break detection DSCNN with 32 filters and depthwise-separable convolutions.\n'
                          'Input: (86, 32) FFT features.\n'
                          'Optimized for real-time edge inference.'
        ),
        'training': dict(
            model_training_id='CNN_AUDIO_DSCNN_GB_NPU',
            model_name='DSCNN_GB_NPU',
            learning_rate=0.001,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_AUDIO_CLASSIFICATION],
            target_devices=[
                constants.TARGET_DEVICE_MSPM0G5187,
            ],
            properties=[dict(type="group", dynamic=True, script="audio.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="audio.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'TCDS_ResNet_NPU': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Temporal Channel-Decoupled Separable ResNet for LPC audio features.\n'
                          'Permutes input (N,1,T,F) → (N,F,T,1): LPC coefficients as channels,\n'
                          '1D temporal convolutions only. Peak SRAM ~3-4 KB. Fits MSPM0G5187.'
        ),
        'training': dict(
            model_training_id='CNN_AUDIO_TCDS_ResNet_NPU',
            model_name='TCDS_ResNet_NPU',
            learning_rate=0.001,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_AUDIO_CLASSIFICATION],
            target_devices=[
                constants.TARGET_DEVICE_MSPM0G5187,
            ],
            properties=[dict(type="group", dynamic=True, script="audio.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="audio.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'TCDS_ResNet_FB_NPU': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Temporal CNN with depthwise-separable residual blocks for filterbank audio features.\n'
                          'Flexible channel sizing via model specification with mixed-precision quantization support.'
        ),
        'training': dict(
            model_training_id='CNN_AUDIO_TCDS_ResNet_FB_NPU',
            model_name='TCDS_ResNet_FB_NPU',
            learning_rate=0.01,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_AUDIO_CLASSIFICATION],
            target_devices=[
                constants.TARGET_DEVICE_MSPM0G5187,
            ],
            properties=[dict(type="group", dynamic=True, script="audio.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="audio.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
}

enabled_models_list = ['DSCNN_NPU', 'DSCNN_32K_NPU', 'DSCNN_GB_NPU', 'TCDS_ResNet_NPU', 'TCDS_ResNet_FB_NPU']


def get_model_descriptions(task_type=None):
    return get_model_descriptions_filtered(
        _model_descriptions,
        enabled_models_list,
        task_type=task_type,
    )


def get_model_description(model_name):
    return get_model_description_by_name(
        _model_descriptions,
        enabled_models_list,
        model_name,
    )


class ModelTraining(BaseAudioModelTraining):
    """
    audio classification-specific model training class.

    Common audio args are handled by BaseaudioModelTraining:
    - audio height / width / channels
    - audio mean / scale
    - data_proc_transforms
    - feat_ext_transform
    - train/test paths
    - quantization train flow
    - ONNX test flow
    """

    train_module = train
    test_module = test

    def _init_task_specific_params(self):
        self.params.update(
            training=utils.ConfigDict(
                file_level_classification_log_path=os.path.join(
                    self.params.training.train_output_path
                    if self.params.training.train_output_path
                    else self.params.training.training_path,
                    'file_level_classification_summary.log'
                ),
            )
        )

    def _get_task_specific_train_argv(self):
        return [
            '--gof-test', f'{self.params.data_processing_feature_extraction.gof_test}',

            # NAS parameters
            '--nas_enabled', f'{self.params.training.nas_enabled}',
            '--nas_optimization_mode', f'{self.params.training.nas_optimization_mode}',
            '--nas_model_size', f'{self.params.training.nas_model_size}',
            '--nas_epochs', f'{self.params.training.nas_epochs}',
            '--nas_nodes_per_layer', f'{self.params.training.nas_nodes_per_layer}',
            '--nas_layers', f'{self.params.training.nas_layers}',
            '--nas_init_channels', f'{self.params.training.nas_init_channels}',
            '--nas_init_channel_multiplier', f'{self.params.training.nas_init_channel_multiplier}',
            '--nas_fanout_concat', f'{self.params.training.nas_fanout_concat}',
            '--load_saved_model', f'{self.params.training.load_saved_model}',

            # Feature extraction with neural network
            '--nn-for-feature-extraction',f'{self.params.data_processing_feature_extraction.nn_for_feature_extraction}',

            # Classification-specific
            '--file-level-classification-log', f'{self.params.training.file_level_classification_log_path}',
        ]

    def _get_task_specific_test_argv(self):
        return [
            '--nn-for-feature-extraction',
            f'{self.params.data_processing_feature_extraction.nn_for_feature_extraction}',
            '--file-level-classification-log', f'{self.params.training.file_level_classification_log_path}',
        ]
