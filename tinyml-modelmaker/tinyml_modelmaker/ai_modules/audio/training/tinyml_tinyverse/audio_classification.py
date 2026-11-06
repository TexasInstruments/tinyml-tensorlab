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

from .audio_base import BaseAudioModelTraining

# Import model descriptions from modelzoo
from tinyml_modelzoo.model_descriptions.audio import (
    _model_descriptions,
    enabled_models_list,
    get_model_descriptions,
    get_model_description,
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
