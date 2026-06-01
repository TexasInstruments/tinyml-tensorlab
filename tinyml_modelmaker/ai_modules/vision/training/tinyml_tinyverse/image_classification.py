#################################################################################
# Copyright (c) 2023-2026, Texas Instruments
# All Rights Reserved.
#################################################################################

import os
from copy import deepcopy

from tinyml_tinyverse.references.image_classification import test_onnx as test
from tinyml_tinyverse.references.image_classification import train
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

from ..... import utils
from ... import constants
from tinyml_modelzoo.device_info import DEVICE_RUN_INFO

from .image_base import (
    BaseImageModelTraining,
    create_template_model_description,
    get_model_descriptions_filtered,
    get_model_description_by_name,
)


model_info_str = "Inference time numbers are for comparison purposes only. (Input Size: {})"

template_model_description = create_template_model_description(
    task_category=constants.TASK_CATEGORY_IMAGE_CLASSIFICATION,
    task_type=constants.TASK_TYPE_IMAGE_CLASSIFICATION,
    dataset_loader='GenericImageDataset',
    batch_size_key=constants.TASK_TYPE_IMAGE_CLASSIFICATION,
)

_model_descriptions = {
    'Lenet5': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Lenet5.\nCNN for image classification.\n'
                          '2 Conv+BatchNorm+Relu+MaxPool layers + 3 Linear layers.'
        ),
        'training': dict(
            model_training_id='CNN_LENET5',
            model_name='Lenet5',
            learning_rate=0.04,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_IMAGE_CLASSIFICATION],
            target_devices={
                constants.TARGET_DEVICE_MSPM0G3507:
                    dict(model_selection_factor=None) |
                    DEVICE_RUN_INFO['Lenet5'][constants.TARGET_DEVICE_MSPM0G3507],
                constants.TARGET_DEVICE_MSPM0G3519:
                    dict(model_selection_factor=None) |
                    DEVICE_RUN_INFO['Lenet5'][constants.TARGET_DEVICE_MSPM0G3519],
                constants.TARGET_DEVICE_MSPM0G5187:
                    dict(model_selection_factor=None) |
                    DEVICE_RUN_INFO['Lenet5'][constants.TARGET_DEVICE_MSPM0G5187],
            },
        ),
    }),
    'MobileNetV1_58k_NPU': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='MobileNetV1_58k_NPU.\n NPU-compliant MobileNetV1-inspired tiny model.\n'
                          '2 Conv+BatchNorm+Relu+MaxPool layers + 3 Linear layers.'
        ),
        'training': dict(
            model_training_id='CNN_IMG_MOBILENETV1_58K_NPU',
            model_name='MobileNetV1_58k_NPU',
            learning_rate=0.04,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_IMAGE_CLASSIFICATION],
            target_devices={
                constants.TARGET_DEVICE_MSPM0G3507:
                    dict(model_selection_factor=None) |
                    DEVICE_RUN_INFO['MobileNetV1_58k_NPU'][constants.TARGET_DEVICE_MSPM0G3507],
                constants.TARGET_DEVICE_MSPM0G3519:
                    dict(model_selection_factor=None) |
                    DEVICE_RUN_INFO['MobileNetV1_58k_NPU'][constants.TARGET_DEVICE_MSPM0G3519],
                constants.TARGET_DEVICE_MSPM0G5187:
                    dict(model_selection_factor=None) |
                    DEVICE_RUN_INFO['MobileNetV1_58k_NPU'][constants.TARGET_DEVICE_MSPM0G5187],
            },
        ),
    }),
}

enabled_models_list = ['Lenet5', 'MobileNetV1_58k_NPU']


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


class ModelTraining(BaseImageModelTraining):
    """
    Image classification-specific model training class.

    Common image args are handled by BaseImageModelTraining:
    - image height / width / channels
    - image mean / scale
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
