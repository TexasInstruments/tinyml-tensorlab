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
        task_category=constants.TASK_CATEGORY_TS_ANOMALYDETECTION,
        task_type=constants.TASK_TYPE_GENERIC_TS_ANOMALYDETECTION,
        generic_model=True,
        model_details='',
    ),
    download=dict(download_url='', download_path=''),
    training=dict(
        quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
        dataset_loader='GenericTSDatasetAD',
        training_backend='tinyml_tinyverse',
        model_training_id='',
        model_name='',
        learning_rate=2e-3,
        model_spec=None,
        batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_ANOMALYDETECTION],
        target_devices={},
        training_devices={constants.TRAINING_DEVICE_CPU: True, constants.TRAINING_DEVICE_CUDA: True, constants.TRAINING_DEVICE_MPS: True,},
    ),
    compilation=dict()
)
_model_descriptions = {
    'AD_17k': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Fan blade Anomaly Detection Model with 17k params. 4 Conv+BatchNorm+Relu layers and then inversion of the same',
        ),
        'training': dict(
            model_training_id='AD_CNN_TS_17K',
            model_name='AD_17k',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['AD_17k'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'Ondevice_Trainable_AD_Linear': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Anomaly Detection Model with 3 encoder layers and 3 decoder layers. Each layer in enocder and decoder is a Linear layer and the last layer can be trained on device',
        ),
        'training': dict(
            model_training_id='AD_3_LAYER_DEEP_ONDEVICE_TRAINABLE_MODEL_TS',
            model_name='Ondevice_Trainable_AD_Linear',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'AD_Linear': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Anomaly Detection Model with 3 encoder layers and 3 decoder layers. Each layer in enocder and decoder is a Linear layer',
        ),
        'training': dict(
            model_training_id='AD_3_LAYER_DEEP_LINEAR_MODEL_TS',
            model_name='AD_Linear',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'AD_16k': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Anomaly Detection Model with 16k params. 4 Conv+BatchNorm+Relu layers and then inversion of the same',
        ),
        'training': dict(
            model_training_id='AE_CNN_TS_GEN_BASE_16K',
            model_name='AD_16k',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['AD_16k'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'AD_4k': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Anomaly Detection Model with 4k params. 3 Conv+BatchNorm+Relu layers and then inversion of the same',
        ),
        'training': dict(
            model_training_id='AE_CNN_TS_GEN_BASE_4K',
            model_name='AD_4k',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['AD_4k'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'AD_1k': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='Anomaly Detection Model with 1k params. 3 Conv+BatchNorm+Relu layers and then inversion of the same. Small Channel width.',
        ),
        'training': dict(
            model_training_id='AE_CNN_TS_GEN_BASE_1K',
            model_name='AD_1k',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['AD_1k'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    # NPU-Optimized Anomaly Detection Models
    'AD_500_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Anomaly Detection Model with ~500 params. 2-layer CNN autoencoder with m4 channels.',
        ),
        'training': dict(
            model_training_id='AE_CNN_TS_GEN_BASE_500_NPU',
            model_name='AD_500_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['AD_500_NPU'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'AD_2k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Anomaly Detection Model with ~2K params. 3-layer CNN autoencoder with m4 channels.',
        ),
        'training': dict(
            model_training_id='AE_CNN_TS_GEN_BASE_2K_NPU',
            model_name='AD_2k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=7) | (DEVICE_RUN_INFO['AD_2k_NPU'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'AD_6k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Anomaly Detection Model with ~6K params using depthwise separable convolutions.',
        ),
        'training': dict(
            model_training_id='AE_CNN_TS_GEN_BASE_6K_NPU',
            model_name='AD_6k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=8) | (DEVICE_RUN_INFO['AD_6k_NPU'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'AD_8k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Anomaly Detection Model with ~8K params. 4-layer CNN autoencoder with m4 channels.',
        ),
        'training': dict(
            model_training_id='AE_CNN_TS_GEN_BASE_8K_NPU',
            model_name='AD_8k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=9) | (DEVICE_RUN_INFO['AD_8k_NPU'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'AD_10k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Anomaly Detection Model with ~10K params. 3-layer CNN autoencoder with higher channel width.',
        ),
        'training': dict(
            model_training_id='AE_CNN_TS_GEN_BASE_10K_NPU',
            model_name='AD_10k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=10) | (DEVICE_RUN_INFO['AD_10k_NPU'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
    'AD_20k_NPU': deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            model_details='NPU-optimized Anomaly Detection Model with ~20K params. 4-layer CNN autoencoder for high capacity anomaly detection.',
        ),
        'training': dict(
            model_training_id='AE_CNN_TS_GEN_BASE_20K_NPU',
            model_name='AD_20k_NPU',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_F29H85: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_F29H85]),
                constants.TARGET_DEVICE_F29P58: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_F29P58]),
                constants.TARGET_DEVICE_F29P32: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_F29P32]),
                constants.TARGET_DEVICE_MSPM33C32: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_MSPM33C32]),
                constants.TARGET_DEVICE_AM13E2: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_AM13E2]),
                constants.TARGET_DEVICE_AM263: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_AM263]),
                constants.TARGET_DEVICE_AM263P: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_AM263P]),
                constants.TARGET_DEVICE_AM261: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_AM261]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_MSPM0G3519]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_CC2755]),
                constants.TARGET_DEVICE_CC1352: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_CC1352]),
                constants.TARGET_DEVICE_CC1354: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_CC1354]),
                constants.TARGET_DEVICE_CC35X1: dict(model_selection_factor=11) | (DEVICE_RUN_INFO['AD_20k_NPU'][constants.TARGET_DEVICE_CC35X1]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                        dict(type="group", dynamic=True, script="generictimeseriesanomalydetection.py", name="train_group", label="Training Parameters", default=[])]
        ),
    }),
}

enabled_models_list = [
    # Existing models
    'AD_17k',
    'Ondevice_Trainable_AD_Linear',
    'AD_Linear',
    'AD_16k',
    'AD_4k',
    'AD_1k',
    # NPU-Optimized gap-filling models
    'AD_500_NPU',
    'AD_2k_NPU',
    'AD_6k_NPU',
    'AD_8k_NPU',
    'AD_10k_NPU',
    'AD_20k_NPU',
]


def get_model_descriptions(task_type=None):
    return get_model_descriptions_filtered(_model_descriptions, enabled_models_list, task_type)


def get_model_description(model_name):
    return get_model_description_by_name(_model_descriptions, enabled_models_list, model_name)
