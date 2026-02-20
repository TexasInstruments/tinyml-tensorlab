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

from tinyml_tinyverse.references.timeseries_forecasting import test_onnx as test
from tinyml_tinyverse.references.timeseries_forecasting import train

from .timeseries_base import BaseModelTraining, get_forecasting_log_summary_regex

# Import model descriptions from modelzoo
from tinyml_modelzoo.model_descriptions.forecasting import (
    _model_descriptions,
    enabled_models_list,
    get_model_descriptions,
    get_model_description,
)


class ModelTraining(BaseModelTraining):
    """Forecasting-specific model training class."""

    train_module = train
    test_module = test

    def _get_task_specific_train_argv(self):
        """Get forecasting-specific training arguments."""
        return [
            '--forecast-horizon', f'{self.params.data_processing_feature_extraction.forecast_horizon}',
            '--target-variables', f'{self.params.data_processing_feature_extraction.target_variables}',
        ]

    def _get_task_specific_test_argv(self):
        """Get forecasting-specific test arguments."""
        return [
            '--forecast-horizon', f'{self.params.data_processing_feature_extraction.forecast_horizon}',
            '--target-variables', f'{self.params.data_processing_feature_extraction.target_variables}',
        ]

    def _get_log_summary_regex(self):
        """Get forecasting-specific log summary regex."""
        return get_forecasting_log_summary_regex()
