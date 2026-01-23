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
from copy import deepcopy

from tinyml_modelzoo import constants
from tinyml_modelzoo.utils import deep_update_dict
from tinyml_modelzoo.device_info import DEVICE_RUN_INFO
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

from ._base import get_model_descriptions_filtered, get_model_description_by_name

this_dir_path = os.path.dirname(os.path.abspath(__file__))
repo_parent_path = os.path.abspath(os.path.join(this_dir_path, '..', '..', '..', '..'))

model_info_str = "Inference time numbers are for comparison purposes only. (Input Size: {})"
template_gui_model_properties = [
    dict(type="group", dynamic=False, name="train_group", label="Training Parameters", default=["training_epochs", "learning_rate"]),
    dict(label="Epochs", name="training_epochs", type="integer", default=50, min=1, max=1000),
    dict(label="Learning Rate", name="learning_rate", type="float", default=0.04, min=0.001, max=0.1, decimal_places=3, increment=0.001)]
template_model_description = dict(
    common=dict(
        task_category=constants.TASK_CATEGORY_TS_FORECASTING,
        task_type=constants.TASK_TYPE_GENERIC_TS_FORECASTING,
        generic_model=True,
        model_details='',
    ),
    download=dict(download_url='', download_path=''),
    training=dict(
        quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
        dataset_loader='GenericTSDatasetForecasting',
        training_backend='tinyml_tinyverse',
        model_training_id='',
        model_name='',
        learning_rate=2e-3,
        model_spec=None,
        batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_FORECASTING],
        target_devices={},
        training_devices={constants.TRAINING_DEVICE_CPU: True, constants.TRAINING_DEVICE_CUDA: True, constants.TRAINING_DEVICE_MPS: True,},
    ),
    compilation=dict()
)
_model_descriptions = {
    'FCST_13k': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Forecasting Model with 13k params.\n 2 Conv+BatchNorm+Relu layers \n Linear Layer.\n',
        ),
        'training': dict(
            model_training_id='FC_CNN_TS_GEN_BASE_13K',
            model_name='FCST_13k',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_13k'][constants.TARGET_DEVICE_AM261]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesforecasting.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'FCST_LSTM10': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Forecasting Model which uses a single LSTM layer with hidden size=10 followed by a linear dense layer',
        ),
        'training': dict(
            model_training_id='LSTM10_TS_GEN_BASE',
            model_name='FCST_LSTM10',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FCST_LSTM10'][constants.TARGET_DEVICE_AM261]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesforecasting.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'FCST_LSTM8': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Forecasting Model which uses a single LSTM layer with hidden size=8 followed by a linear dense layer',
        ),
        'training': dict(
            model_training_id='LSTM8_TS_GEN_BASE',
            model_name='FCST_LSTM8',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FCST_LSTM8'][constants.TARGET_DEVICE_AM261]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesforecasting.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'FCST_3k': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Forecasting Model with 3k params. 4 layers of MLPs (Fully Connected Layers)\n',
        ),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_3K',
            model_name='FCST_3k',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['FCST_3k'][constants.TARGET_DEVICE_AM261]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesforecasting.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    # NPU-Optimized Forecasting Models
    'FCST_500_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Forecasting Model with ~500 params.\nBN -> Conv(4ch, k7) -> Conv(8ch, k5) -> AvgPool -> FC.\nAll layers NPU-compliant (m4 channels, kH≤7).',
        ),
        'training': dict(
            model_training_id='FC_CNN_TS_GEN_BASE_500_NPU',
            model_name='FCST_500_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_500_NPU'][constants.TARGET_DEVICE_AM261]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesforecasting.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'FCST_1k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Forecasting Model with ~1K params.\nBN -> Conv(8ch, k7) -> Conv(8ch, k5) -> AvgPool -> FC.\nAll layers NPU-compliant (m4 channels, kH≤7).',
        ),
        'training': dict(
            model_training_id='FC_CNN_TS_GEN_BASE_1K_NPU',
            model_name='FCST_1k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_1k_NPU'][constants.TARGET_DEVICE_AM261]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesforecasting.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'FCST_2k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Forecasting Model with ~2K params.\nBN -> Conv(8ch, k7) -> Conv(16ch, k5) -> AvgPool -> FC.\nAll layers NPU-compliant (m4 channels, kH≤7).',
        ),
        'training': dict(
            model_training_id='FC_CNN_TS_GEN_BASE_2K_NPU',
            model_name='FCST_2k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_2k_NPU'][constants.TARGET_DEVICE_AM261]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesforecasting.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'FCST_4k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Forecasting Model with ~4K params.\nBN -> Conv(8ch, k7) -> Conv(16ch, k5) -> Conv(16ch, k3) -> AvgPool -> FC.\nAll layers NPU-compliant (m4 channels, kH≤7).',
        ),
        'training': dict(
            model_training_id='FC_CNN_TS_GEN_BASE_4K_NPU',
            model_name='FCST_4k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_4k_NPU'][constants.TARGET_DEVICE_AM261]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesforecasting.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'FCST_6k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Forecasting Model with ~6K params using depthwise separable convolutions.\nBN -> Conv(8ch) -> DWConv(8ch) -> PWConv(16ch) -> DWConv(16ch) -> PWConv(24ch) -> AvgPool -> FC.\nAll layers NPU-compliant (m4 channels, kH≤7).',
        ),
        'training': dict(
            model_training_id='FC_CNN_TS_GEN_BASE_6K_NPU',
            model_name='FCST_6k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_6k_NPU'][constants.TARGET_DEVICE_AM261]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesforecasting.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'FCST_8k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Forecasting Model with ~8K params.\nBN -> Conv(16ch, k7) -> Conv(24ch, k5) -> Conv(24ch, k3) -> AvgPool -> FC.\nAll layers NPU-compliant (m4 channels, kH≤7).',
        ),
        'training': dict(
            model_training_id='FC_CNN_TS_GEN_BASE_8K_NPU',
            model_name='FCST_8k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_8k_NPU'][constants.TARGET_DEVICE_AM261]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesforecasting.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'FCST_10k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Forecasting Model with ~10K params.\nBN -> Conv(16ch, k7) -> Conv(24ch, k5) -> Conv(32ch, k3) -> AvgPool -> FC.\nAll layers NPU-compliant (m4 channels, kH≤7).',
        ),
        'training': dict(
            model_training_id='FC_CNN_TS_GEN_BASE_10K_NPU',
            model_name='FCST_10k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_10k_NPU'][constants.TARGET_DEVICE_AM261]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesforecasting.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'FCST_20k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Forecasting Model with ~20K params.\nBN -> Conv(16ch, k7) -> MaxPool(4) -> Conv(32ch, k5) -> Conv(48ch, k3) -> Conv(48ch, k3) -> AvgPool -> FC.\nAll layers NPU-compliant (m4 channels, kH≤7, MaxPool≤4).',
        ),
        'training': dict(
            model_training_id='FC_CNN_TS_GEN_BASE_20K_NPU',
            model_name='FCST_20k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FCST_20k_NPU'][constants.TARGET_DEVICE_AM261]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesforecasting.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
}

enabled_models_list = [
    'FCST_13k',
    'FCST_LSTM10',
    'FCST_LSTM8',
    'FCST_3k',
    # NPU-Optimized Forecasting Models
    'FCST_500_NPU',
    'FCST_1k_NPU',
    'FCST_2k_NPU',
    'FCST_4k_NPU',
    'FCST_6k_NPU',
    'FCST_8k_NPU',
    'FCST_10k_NPU',
    'FCST_20k_NPU',
]


def get_model_descriptions(task_type=None):
    return get_model_descriptions_filtered(_model_descriptions, enabled_models_list, task_type)


def get_model_description(model_name):
    return get_model_description_by_name(_model_descriptions, enabled_models_list, model_name)
