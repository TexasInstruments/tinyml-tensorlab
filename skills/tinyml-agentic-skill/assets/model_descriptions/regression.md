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
from copy import deepcopy

from tinyml_modelzoo import constants
from tinyml_modelzoo.utils import deep_update_dict
from tinyml_modelzoo.device_info import DEVICE_RUN_INFO
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

from ._base import get_model_descriptions_filtered, get_model_description_by_name

this_dir_path = os.path.dirname(os.path.abspath(__file__))
repo_parent_path = os.path.abspath(os.path.join(this_dir_path, '..', '..', '..', '..'))

model_info_str = "Inference time numbers are for comparison purposes only. (Input Size: {})"
template_gui_model_properties = []
template_model_description = dict(
    common=dict(
        task_category=constants.TASK_CATEGORY_TS_REGRESSION,
        task_type=constants.TASK_TYPE_GENERIC_TS_REGRESSION,
        generic_model=True,
        model_details='',
    ),
    download=dict(download_url='', download_path=''),
    training=dict(
        quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
        dataset_loader='GenericTSDatasetReg',
        training_backend='tinyml_tinyverse',
        model_training_id='',
        model_name='',
        learning_rate=2e-3,
        model_spec=None,
        batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_REGRESSION],
        target_devices={},
        training_devices={constants.TRAINING_DEVICE_CPU: True, constants.TRAINING_DEVICE_CUDA: True, constants.TRAINING_DEVICE_MPS: True,},
    ),
    compilation=dict()
)
_model_descriptions = {
    'REGR_10k': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Regression Model with 10k params. 3 Conv+BatchNorm+Relu layers + 2 Linear Layer.',
        ),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_10K',
            model_name='REGR_10k',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['REGR_10k'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'REGR_1k': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Regression Model with 1k params. 2 Conv+BatchNorm+Relu layers + 2 Linear Layer.',
        ),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_1K',
            model_name='REGR_1k',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_1k'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'REGR_2k': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Regression Model with 2k params. 3 Conv+BatchNorm+Relu layers + 2 Linear Layer.',
        ),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_2K',
            model_name='REGR_2k',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['REGR_2k'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'REGR_3k': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Regression Model with 3k params. 4 layers of MLPs (Fully Connected Layers)',
        ),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_3K',
            model_name='REGR_3k',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['REGR_3k'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'REGR_13k': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Regression Model with 13k params. 3 Conv+BatchNorm+Relu layers + Linear Layer.',
        ),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_13K',
            model_name='REGR_13k',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['REGR_13k'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'REGR_4k': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Regression Model with 4k params. 2 Conv+BatchNorm+Relu layers + Linear Layer.',
        ),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_4K',
            model_name='REGR_4k',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['REGR_4k'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    # NPU-Optimized Regression Models
    'REGR_500_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-Optimized Regression Model with ~500 params.\n2 Conv+BatchNorm+Relu layers + Linear Layer.\nUltra-compact model. Fully NPU compliant with m4 channels and FC input>=16.',
            # help_url="file://docs/NPU_CONFIGURATION_GUIDELINES.md"
        ),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_500_NPU',
            model_name='REGR_500_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['REGR_500_NPU'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'REGR_2k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-Optimized Regression Model with ~2K params.\n3 Conv+BatchNorm+Relu layers + 2 Linear Layers.\nFills gap between 1K and 3K. Fully NPU compliant with m4 channels.',
            # help_url="file://docs/NPU_CONFIGURATION_GUIDELINES.md"
        ),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_2K_NPU',
            model_name='REGR_2k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['REGR_2k_NPU'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'REGR_6k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-Optimized Regression Model with ~6K params.\nDepthwise separable convolutions + Linear Layers.\nFills gap between 4K and 10K. Fully NPU compliant with DWCONV+PWCONV pattern.',
            # help_url="file://docs/NPU_CONFIGURATION_GUIDELINES.md"
        ),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_6K_NPU',
            model_name='REGR_6k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['REGR_6k_NPU'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'REGR_8k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-Optimized Regression Model with ~8K params.\n4 Conv+BatchNorm+Relu layers + 2 Linear Layers.\nFills gap between 6K and 10K. Fully NPU compliant with m4 channels.',
            # help_url="file://docs/NPU_CONFIGURATION_GUIDELINES.md"
        ),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_8K_NPU',
            model_name='REGR_8k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['REGR_8k_NPU'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'REGR_20k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-Optimized Regression Model with ~20K params.\n4 Conv+BatchNorm+Relu layers + MaxPool + 2 Linear Layers.\nHigh capacity model. Fully NPU compliant with m4 channels and MaxPool<=4.',
            # help_url="file://docs/NPU_CONFIGURATION_GUIDELINES.md"
        ),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_20K_NPU',
            model_name='REGR_20k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['REGR_20k_NPU'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesregression.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
}

enabled_models_list = [
    # Existing models
    'REGR_10k',
    'REGR_1k',
    'REGR_2k',
    'REGR_3k',
    'REGR_13k',
    'REGR_4k',
    # NPU-Optimized gap-filling models
    'REGR_500_NPU',
    'REGR_2k_NPU',
    'REGR_6k_NPU',
    'REGR_8k_NPU',
    'REGR_20k_NPU',
]


def get_model_descriptions(task_type=None):
    return get_model_descriptions_filtered(_model_descriptions, enabled_models_list, task_type)


def get_model_description(model_name):
    return get_model_description_by_name(_model_descriptions, enabled_models_list, model_name)
