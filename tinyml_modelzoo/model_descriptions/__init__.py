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

"""
Model descriptions for TinyML models.
"""

from ._base import get_model_descriptions_filtered, get_model_description_by_name

from .classification import get_model_descriptions as get_classification_model_descriptions
from .classification import get_model_description as get_classification_model_description

from .regression import get_model_descriptions as get_regression_model_descriptions
from .regression import get_model_description as get_regression_model_description

from .anomalydetection import get_model_descriptions as get_anomalydetection_model_descriptions
from .anomalydetection import get_model_description as get_anomalydetection_model_description

from .forecasting import get_model_descriptions as get_forecasting_model_descriptions
from .forecasting import get_model_description as get_forecasting_model_description

__all__ = [
    "get_model_descriptions_filtered",
    "get_model_description_by_name",
    "get_classification_model_descriptions",
    "get_classification_model_description",
    "get_regression_model_descriptions",
    "get_regression_model_description",
    "get_anomalydetection_model_descriptions",
    "get_anomalydetection_model_description",
    "get_forecasting_model_descriptions",
    "get_forecasting_model_description",
]
