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
Base helper functions for model descriptions.
"""


def get_model_descriptions_filtered(model_descriptions, enabled_models_list, task_type=None):
    """
    Filter model descriptions based on enabled models list.

    Args:
        model_descriptions: Dictionary of all model descriptions
        enabled_models_list: List of enabled model names
        task_type: Optional task type filter (not currently used but available for future)

    Returns:
        dict: Filtered model descriptions
    """
    return {k: v for k, v in model_descriptions.items() if k in enabled_models_list}


def get_model_description_by_name(model_descriptions, enabled_models_list, model_name):
    """
    Get a specific model description by name.

    Args:
        model_descriptions: Dictionary of all model descriptions
        enabled_models_list: List of enabled model names
        model_name: Name of the model to retrieve

    Returns:
        dict or None: Model description if found, None otherwise
    """
    filtered = get_model_descriptions_filtered(model_descriptions, enabled_models_list)
    return filtered.get(model_name, None)
