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

from copy import deepcopy

from tinyml_modelzoo import constants
from tinyml_modelzoo.utils import deep_update_dict
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

from ._base import get_model_descriptions_filtered, get_model_description_by_name

template_model_description = dict(
    common=dict(
        task_category=constants.TASK_CATEGORY_AUDIO_CLASSIFICATION,
        task_type=constants.TASK_TYPE_AUDIO_CLASSIFICATION,
        generic_model=True,
        model_details='',
    ),
    download=dict(download_url='', download_path=''),
    training=dict(
        quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
        training_backend='tinyml_tinyverse',
        model_training_id='',
        dataset_loader='GenericAudioDataset',
        model_name='',
        learning_rate=2e-3,
        model_spec=None,
        batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_AUDIO_CLASSIFICATION],
        target_devices=[],
        training_devices={constants.TRAINING_DEVICE_CPU: True, constants.TRAINING_DEVICE_CUDA: True, constants.TRAINING_DEVICE_MPS: True,},
    ),
    compilation=dict()
)

AUDIO_PROPERTIES = [dict(type="group", dynamic=True, script="audio.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                    dict(type="group", dynamic=True, script="audio.py", name="train_group", label="Training Parameters", default=[])]

_model_descriptions = {
    'DSCNN_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Lenet5.\nCNN for audio classification.\n'
                          '2 Conv+BatchNorm+Relu+MaxPool layers + 3 Linear layers.'
        ),
        'training': dict(
            model_training_id='CNN_AUDIO_DSCNN',
            model_name='DSCNN_NPU',
            learning_rate=0.1,
            target_devices=[
                constants.TARGET_DEVICE_MSPM0G5187,
            ],
            properties=AUDIO_PROPERTIES
        ),
    }),
    'DSCNN_32K_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='SRAM/Flash-optimized DSCNN for audio classification.\n'
                          '48 filters, DW 3x1 kernels, stride-2 downsample in DS block 1.\n'
                          'Target: SRAM < 32 KB, Flash < 128 KB, W8 quantization.'
        ),
        'training': dict(
            model_training_id='CNN_AUDIO_DSCNN_32K_NPU',
            model_name='DSCNN_32K_NPU',
            learning_rate=0.001,
            target_devices=[
                constants.TARGET_DEVICE_MSPM0G5187,
            ],
            properties=AUDIO_PROPERTIES
        ),
    }),
    'DSCNN_GB_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Glass-break detection DSCNN with 32 filters and depthwise-separable convolutions.\n'
                          'Input: (86, 32) FFT features.\n'
                          'Optimized for real-time edge inference.'
        ),
        'training': dict(
            model_training_id='CNN_AUDIO_DSCNN_GB_NPU',
            model_name='DSCNN_GB_NPU',
            learning_rate=0.001,
            target_devices=[
                constants.TARGET_DEVICE_MSPM0G5187,
            ],
            properties=AUDIO_PROPERTIES
        ),
    }),
    'TCDS_ResNet_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Temporal Channel-Decoupled Separable ResNet for LPC audio features.\n'
                          'Permutes input (N,1,T,F) → (N,F,T,1): LPC coefficients as channels,\n'
                          '1D temporal convolutions only. Peak SRAM ~3-4 KB. Fits MSPM0G5187.'
        ),
        'training': dict(
            model_training_id='CNN_AUDIO_TCDS_ResNet_NPU',
            model_name='TCDS_ResNet_NPU',
            learning_rate=0.001,
            target_devices=[
                constants.TARGET_DEVICE_MSPM0G5187,
            ],
            properties=AUDIO_PROPERTIES
        ),
    }),
    'TCDS_ResNet_FB_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Temporal CNN with depthwise-separable residual blocks for filterbank audio features.\n'
                          'Flexible channel sizing via model specification with mixed-precision quantization support.'
        ),
        'training': dict(
            model_training_id='CNN_AUDIO_TCDS_ResNet_FB_NPU',
            model_name='TCDS_ResNet_FB_NPU',
            learning_rate=0.01,
            target_devices=[
                constants.TARGET_DEVICE_MSPM0G5187,
            ],
            properties=AUDIO_PROPERTIES
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
