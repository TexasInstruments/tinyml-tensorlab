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

from tinyml_tinyverse.references.timeseries_classification import test_onnx as test
from tinyml_tinyverse.references.timeseries_classification import train

from ..... import utils
from .timeseries_base import BaseModelTraining

# Import model descriptions from modelzoo
from tinyml_modelzoo.model_descriptions.classification import (
    _model_descriptions,
    enabled_models_list,
    get_model_descriptions,
    get_model_description,
)


class ModelTraining(BaseModelTraining):
    """Classification-specific model training class."""

    train_module = train
    test_module = test

    def _init_task_specific_params(self):
        """Initialize classification-specific parameters."""
        self.params.update(
            training=utils.ConfigDict(
                file_level_classification_log_path=os.path.join(
                    self.params.training.train_output_path if self.params.training.train_output_path else self.params.training.training_path,
                    'file_level_classification_summary.log'),
            )
        )

    def _get_task_specific_train_argv(self):
        """Get classification-specific training arguments."""
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
            # Feature Extraction based on Neural Networks
            '--nn-for-feature-extraction', f'{self.params.data_processing_feature_extraction.nn_for_feature_extraction}',
            # MSP-specific
            '--gain-variations', f'{self.params.data_processing_feature_extraction.gain_variations}',
            '--q15-scale-factor', f'{self.params.data_processing_feature_extraction.q15_scale_factor}',
            # PIR Detection related params
            '--window-count', f'{self.params.data_processing_feature_extraction.window_count}',
            '--chunk-size', f'{self.params.data_processing_feature_extraction.chunk_size}',
            '--fft-size', f'{self.params.data_processing_feature_extraction.fft_size}',
            # Classification Task Specific Params
            '--file-level-classification-log', f'{self.params.training.file_level_classification_log_path}',
        ]

    def _get_task_specific_test_argv(self):
        """Get classification-specific test arguments."""
        return [
            # Feature Extraction based on Neural Networks
            '--nn-for-feature-extraction', f'{self.params.data_processing_feature_extraction.nn_for_feature_extraction}',
            # MSP-specific
            '--gain-variations', f'{self.params.data_processing_feature_extraction.gain_variations}',
            '--q15-scale-factor', f'{self.params.data_processing_feature_extraction.q15_scale_factor}',
            # PIR Detection related params
            '--window-count', f'{self.params.data_processing_feature_extraction.window_count}',
            '--chunk-size', f'{self.params.data_processing_feature_extraction.chunk_size}',
            '--fft-size', f'{self.params.data_processing_feature_extraction.fft_size}',
            # Classification Task Specific Params
            '--file-level-classification-log', f'{self.params.training.file_level_classification_log_path}',
        ]
