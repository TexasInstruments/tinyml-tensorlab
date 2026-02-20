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

from tinyml_tinyverse.references.timeseries_anomalydetection import test_onnx as test
from tinyml_tinyverse.references.timeseries_anomalydetection import train

from .timeseries_base import BaseModelTraining, get_anomaly_detection_log_summary_regex

# Import model descriptions from modelzoo
from tinyml_modelzoo.model_descriptions.anomalydetection import (
    _model_descriptions,
    enabled_models_list,
    get_model_descriptions,
    get_model_description,
)


class ModelTraining(BaseModelTraining):
    """Anomaly Detection-specific model training class."""

    train_module = train
    test_module = test

    def _run_testing(self, device):
        """Run model testing/evaluation with anomaly detection-specific args."""
        if self.params.testing.test_data and os.path.exists(self.params.testing.test_data):
            data_path = self.params.testing.test_data
        else:
            data_path = os.path.join(self.params.dataset.dataset_path, self.params.dataset.data_dir)

        if self.params.testing.model_path and os.path.exists(self.params.testing.model_path):
            model_path = self.params.testing.model_path
            output_dir = self.params.training.training_path
        else:
            from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion
            if self.params.training.quantization == TinyMLQuantizationVersion.NO_QUANTIZATION:
                model_path = os.path.join(self.params.training.training_path, 'model.onnx')
                output_dir = self.params.training.training_path
            else:
                model_path = os.path.join(self.params.training.training_path_quantization, 'model.onnx')
                output_dir = self.params.training.training_path_quantization

        argv = self._build_common_test_argv(device, data_path, model_path, output_dir)
        argv.extend(self._get_task_specific_test_argv())

        args = self.test_module.get_args_parser().parse_args(argv)
        args.quit_event = self.quit_event
        # Anomaly detection specific args
        args.cache_dataset = None
        args.gpu = self.params.training.num_gpus
        self.test_module.run(args)

    def _get_log_summary_regex(self):
        """Get anomaly detection-specific log summary regex."""
        return get_anomaly_detection_log_summary_regex()
