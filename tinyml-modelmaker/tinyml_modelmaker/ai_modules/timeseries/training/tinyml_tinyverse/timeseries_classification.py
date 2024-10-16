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
import shutil

from tinyml_tinyverse.references.timeseries_classification import test_onnx as test
from tinyml_tinyverse.references.timeseries_classification import train
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

import tinyml_modelmaker

from ..... import utils
from ... import constants

this_dir_path = os.path.dirname(os.path.abspath(__file__))
repo_parent_path = os.path.abspath(os.path.join(this_dir_path, '..', '..', '..', '..', '..', '..'))

# tinyml_tinyverse_path = os.path.join(repo_parent_path, 'tinyml-tinyverse')
# www_modelzoo_path = 'https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01'


model_urls = {
    'TimeSeries_Generic_7k': {'download_url': '', 'download_path': '', },
    'TimeSeries_Generic_AF_7k': {'download_url': '', 'download_path': '', },
    'TimeSeries_Generic_MF_7k': {'download_url': '', 'download_path': '', },
    'TimeSeries_Generic_3k': {'download_url': '', 'download_path': '', },
    'TimeSeries_Generic_AF_3k': {'download_url': '', 'download_path': '', },
    'TimeSeries_Generic_MF_3k': {'download_url': '', 'download_path': '', },
    'ArcFault_model_1400': {'download_url': '', 'download_path': '', },
    'ArcFault_model_200': {'download_url': '', 'download_path': '', },
    'ArcFault_model_300': {'download_url': '', 'download_path': '', },
    'ArcFault_model_700': {'download_url': '', 'download_path': '', },
    'MotorFault_model_1': {'download_url': '', 'download_path': '', },
    'MotorFault_model_2': {'download_url': '', 'download_path': '', },
    'MotorFault_model_3': {'download_url': '', 'download_path': '', },
    'TimeSeries_Generic_7k_t': {'download_url': '', 'download_path': '', },
    'TimeSeries_Generic_AF_7k_t': {'download_url': '', 'download_path': '', },
    'TimeSeries_Generic_MF_7k_t': {'download_url': '', 'download_path': '', },
    'TimeSeries_Generic_3k_t': {'download_url': '', 'download_path': '', },
    'TimeSeries_Generic_AF_3k_t': {'download_url': '', 'download_path': '', },
    'TimeSeries_Generic_MF_3k_t': {'download_url': '', 'download_path': '', },
    'ArcFault_model_1400_t': {'download_url': '', 'download_path': '', },
    'ArcFault_model_200_t': {'download_url': '', 'download_path': '', },
    'ArcFault_model_300_t': {'download_url': '', 'download_path': '', },
    'ArcFault_model_700_t': {'download_url': '', 'download_path': '', },
    'MotorFault_model_1_t': {'download_url': '', 'download_path': '', },
    'MotorFault_model_2_t': {'download_url': '', 'download_path': '', },
    'MotorFault_model_3_t': {'download_url': '', 'download_path': '', },
}

disbanded_model_descriptions = {
    'TimeSeries_Generic_AF_7k': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_ARC_FAULT,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_AF_7k'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
            with_input_batchnorm=False,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_TS_GEN_BASE_7K',
            model_name='TimeSeries_Generic_AF_7k',
            model_architecture='backbone',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION],
            target_devices={
                # constants.TARGET_DEVICE_AM263: dict(performance_infer_time_ms='TBD',
                #                                     accuracy_factor='TBD',),
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },
    'TimeSeries_Generic_AF_7k_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_ARC_FAULT,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_AF_7k_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_TS_GEN_BASE_7K',
            model_name='TimeSeries_Generic_AF_7k_t',
            model_architecture='backbone',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD',
                                                      accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P55: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },

    'TimeSeries_Generic_MF_7k': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_MOTOR_FAULT,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_MF_7k'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
            with_input_batchnorm=False,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_TS_GEN_BASE_7K',
            model_name='TimeSeries_Generic_MF_7k',
            model_architecture='backbone',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION],
            target_devices={
                # constants.TARGET_DEVICE_AM263: dict(performance_infer_time_ms='TBD',
                #                                     accuracy_factor='TBD',),
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },
    'TimeSeries_Generic_MF_7k_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_MOTOR_FAULT,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_MF_7k_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_TS_GEN_BASE_7K',
            model_name='TimeSeries_Generic_MF_7k_t',
            model_architecture='backbone',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P55: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },

    'TimeSeries_Generic_AF_3k': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_ARC_FAULT,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_AF_3k'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
            with_input_batchnorm=False,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_TS_GEN_BASE_3K',
            model_name='TimeSeries_Generic_AF_3k',
            model_architecture='backbone',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION],
            target_devices={
                # constants.TARGET_DEVICE_AM263: dict(performance_infer_time_ms='TBD',
                #                                     accuracy_factor='TBD',),
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },
    'TimeSeries_Generic_AF_3k_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_ARC_FAULT,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_AF_3k_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_TS_GEN_BASE_3K',
            model_name='TimeSeries_Generic_AF_3k_t',
            model_architecture='backbone',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P55: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },

    'TimeSeries_Generic_MF_3k': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_MOTOR_FAULT,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_MF_3k'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
            with_input_batchnorm=False,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_TS_GEN_BASE_3K',
            model_name='TimeSeries_Generic_MF_3k',
            model_architecture='backbone',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION],
            target_devices={
                # constants.TARGET_DEVICE_AM263: dict(performance_infer_time_ms='TBD',
                #                                     accuracy_factor='TBD',),
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },
    'TimeSeries_Generic_MF_3k_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_MOTOR_FAULT,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_MF_3k_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_TS_GEN_BASE_3K',
            model_name='TimeSeries_Generic_MF_3k_t',
            model_architecture='backbone',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P55: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },

    'ArcFault_model_1400': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_ARC_FAULT,
        ),
        'download': model_urls['ArcFault_model_1400'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
            with_input_batchnorm=False,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_AF_3L_1400',
            model_name='ArcFault_model_1400',
            model_architecture='backbone',
            learning_rate=0.04,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_af_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_ARC_FAULT],
            target_devices={
                # constants.TARGET_DEVICE_AM263: dict(performance_infer_time_ms='TBD',
                #                                     accuracy_factor='TBD',),
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },
    'ArcFault_model_1400_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_ARC_FAULT,
        ),
        'download': model_urls['ArcFault_model_1400_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_AF_3L_1400',
            model_name='ArcFault_model_1400_t',
            model_architecture='backbone',
            learning_rate=0.04,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_af_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_ARC_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P55: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict(
            # model_compilation_id='afd-b1-0002',
            
        )
    },

    'ArcFault_model_200': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_ARC_FAULT,
        ),
        'download': model_urls['ArcFault_model_200'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
            with_input_batchnorm=False,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_AF_3L_200',
            model_name='ArcFault_model_200',
            model_architecture='backbone',
            learning_rate=0.04,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_af_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_ARC_FAULT],
            target_devices={
                # constants.TARGET_DEVICE_AM263: dict(performance_infer_time_ms='TBD',
                #                                     accuracy_factor='TBD',),
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },
    'ArcFault_model_300': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_ARC_FAULT,
        ),
        'download': model_urls['ArcFault_model_300'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
            with_input_batchnorm=False,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_AF_3L_300',
            model_name='ArcFault_model_300',
            model_architecture='backbone',
            learning_rate=0.04,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_af_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_ARC_FAULT],
            target_devices={
                # constants.TARGET_DEVICE_AM263: dict(performance_infer_time_ms='TBD',
                #                                     accuracy_factor='TBD',),
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },
    'ArcFault_model_700': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_ARC_FAULT,
        ),
        'download': model_urls['ArcFault_model_700'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
            with_input_batchnorm=False,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_AF_3L_700',
            model_name='ArcFault_model_700',
            model_architecture='backbone',
            learning_rate=0.04,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_af_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_ARC_FAULT],
            target_devices={
                # constants.TARGET_DEVICE_AM263: dict(performance_infer_time_ms='TBD',
                #                                     accuracy_factor='TBD',),
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },
    'MotorFault_model_1': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_MOTOR_FAULT,
        ),
        'download': model_urls['MotorFault_model_1'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
            with_input_batchnorm=False,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_MF_1L',
            model_name='MotorFault_model_1',
            model_architecture='backbone',
            learning_rate=0.01,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_mf_1l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_MOTOR_FAULT],
            target_devices={
                # constants.TARGET_DEVICE_AM263: dict(performance_infer_time_ms='TBD',
                #                                     accuracy_factor='TBD',),
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict(
            # model_compilation_id='mfd-b1-0001',
            
        )
    },
    'MotorFault_model_2': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_MOTOR_FAULT,
        ),
        'download': model_urls['MotorFault_model_2'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
            with_input_batchnorm=False,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_MF_2L',
            model_name='MotorFault_model_2',
            model_architecture='backbone',
            learning_rate=0.01,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_mf_2l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_MOTOR_FAULT],
            target_devices={
                # constants.TARGET_DEVICE_AM263: dict(performance_infer_time_ms='TBD',
                #                                     accuracy_factor='TBD',),
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict(
            # model_compilation_id='mfd-b2-0003',
            
        )
    },
    'MotorFault_model_3': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_MOTOR_FAULT,
        ),
        'download': model_urls['MotorFault_model_3'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
            with_input_batchnorm=False,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_MF_3L',
            model_name='MotorFault_model_3',
            model_architecture='backbone',
            learning_rate=0.01,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_mf_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_MOTOR_FAULT],
            target_devices={
                # constants.TARGET_DEVICE_AM263: dict(performance_infer_time_ms='TBD',
                #                                     accuracy_factor='TBD',),
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict(
            # model_compilation_id='mfd-b2-0003',
            
        )
    },

}

_model_descriptions = {
    'TimeSeries_Generic_7k': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_7k'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
            with_input_batchnorm=False,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_TS_GEN_BASE_7K',
            model_name='TimeSeries_Generic_7k',
            model_architecture='backbone',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION],
            target_devices={
                # constants.TARGET_DEVICE_AM263: dict(performance_infer_time_ms='TBD',
                #                                     accuracy_factor='TBD',),
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },
    'TimeSeries_Generic_7k_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_7k_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_TS_GEN_BASE_7K',
            model_name='TimeSeries_Generic_7k_t',
            model_architecture='backbone',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', ),
                constants.TARGET_DEVICE_F28P55: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },

    'TimeSeries_Generic_3k': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_3k'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
            with_input_batchnorm=False,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_TS_GEN_BASE_3K',
            model_name='TimeSeries_Generic_3k',
            model_architecture='backbone',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION],
            target_devices={
                # constants.TARGET_DEVICE_AM263: dict(performance_infer_time_ms='TBD',
                #                                     accuracy_factor='TBD',),
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },
    'TimeSeries_Generic_3k_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_3k_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_TS_GEN_BASE_3K',
            model_name='TimeSeries_Generic_3k_t',
            model_architecture='backbone',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms='TBD', accuracy_factor='TBD',
                                                      model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms='TBD',
                                                    accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
                constants.TARGET_DEVICE_F28P55: dict(performance_infer_time_ms='TBD',
                                                     accuracy_factor='TBD', model_selection_factor=None, ),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },

    'ArcFault_model_700_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_ARC_FAULT,
        ),
        'download': model_urls['ArcFault_model_700_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_AF_3L_700',
            model_name='ArcFault_model_700_t',
            model_architecture='backbone',
            dataset_loader='ArcFaultDataset',
            learning_rate=0.04,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_af_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_ARC_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms=1.299, accuracy_factor='TBD',
                                                      model_selection_factor=1, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms=1.299,
                                                     accuracy_factor='TBD', model_selection_factor=1),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms=1.559,
                                                     accuracy_factor='TBD', model_selection_factor=1),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms=1.299,
                                                    accuracy_factor='TBD', model_selection_factor=1),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms=0.779,
                                                     accuracy_factor='TBD', model_selection_factor=1),
                constants.TARGET_DEVICE_F28P55: dict(performance_infer_time_ms=0.307,
                                                     accuracy_factor='TBD', model_selection_factor=1),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict(
            # model_compilation_id='afd-b1-0002',
            
        )
    },
    'ArcFault_model_300_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_ARC_FAULT,
        ),
        'download': model_urls['ArcFault_model_300_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_AF_3L_300',
            model_name='ArcFault_model_300_t',
            model_architecture='backbone',
            dataset_loader='ArcFaultDataset',
            learning_rate=0.04,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_af_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_ARC_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms=0.973, accuracy_factor='TBD',
                                                      model_selection_factor=2, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms=0.973,
                                                     accuracy_factor='TBD', model_selection_factor=2),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms=1.167,
                                                     accuracy_factor='TBD', model_selection_factor=2),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms=0.973,
                                                    accuracy_factor='TBD', model_selection_factor=2),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms=0.584,
                                                     accuracy_factor='TBD', model_selection_factor=2),
                constants.TARGET_DEVICE_F28P55: dict(performance_infer_time_ms=0.254,
                                                     accuracy_factor='TBD', model_selection_factor=2),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict(
            # model_compilation_id='afd-b1-0002',
            
        )
    },
    'ArcFault_model_200_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_ARC_FAULT,
        ),
        'download': model_urls['ArcFault_model_200_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_AF_3L_200',
            model_name='ArcFault_model_200_t',
            model_architecture='backbone',
            dataset_loader='ArcFaultDataset',
            learning_rate=0.04,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_af_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_ARC_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms=0.529, accuracy_factor='TBD',
                                                      model_selection_factor=3, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms=0.529,
                                                     accuracy_factor='TBD', model_selection_factor=3),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms=0.635,
                                                     accuracy_factor='TBD', model_selection_factor=3),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms=0.529,
                                                    accuracy_factor='TBD', model_selection_factor=3),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms=0.318,
                                                     accuracy_factor='TBD', model_selection_factor=3),
                constants.TARGET_DEVICE_F28P55: dict(performance_infer_time_ms=0.193,
                                                     accuracy_factor='TBD', model_selection_factor=3),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict(
            # model_compilation_id='afd-b1-0002',
            
        )
    },

    'MotorFault_model_1_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_MOTOR_FAULT,
        ),
        'download': model_urls['MotorFault_model_1_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_MF_1L',
            model_name='MotorFault_model_1_t',
            model_architecture='backbone',
            dataset_loader='MotorFaultDataset',
            learning_rate=0.01,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_mf_1l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_MOTOR_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms=0.629, accuracy_factor='TBD',
                                                      model_selection_factor=2, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms=0.629,
                                                     accuracy_factor='TBD', model_selection_factor=2),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms=0.754,
                                                     accuracy_factor='TBD', model_selection_factor=2),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms=0.629,
                                                    accuracy_factor='TBD', model_selection_factor=2),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms=0.377,
                                                     accuracy_factor='TBD', model_selection_factor=2),
                constants.TARGET_DEVICE_F28P55: dict(performance_infer_time_ms=0.197,
                                                     accuracy_factor='TBD', model_selection_factor=2),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict(
            # model_compilation_id='mfd-b1-0002',
            
        )
    },
    'MotorFault_model_2_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_MOTOR_FAULT,
        ),
        'download': model_urls['MotorFault_model_2_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_MF_2L',
            model_name='MotorFault_model_2_t',
            model_architecture='backbone',
            dataset_loader='MotorFaultDataset',
            learning_rate=0.01,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_mf_2l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_MOTOR_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms=18.79, accuracy_factor='TBD',
                                                      model_selection_factor=1, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms=18.79,
                                                     accuracy_factor='TBD', model_selection_factor=1),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms=22.548,
                                                     accuracy_factor='TBD', model_selection_factor=1),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms=18.79,
                                                    accuracy_factor='TBD', model_selection_factor=1),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms=11.274,
                                                     accuracy_factor='TBD', model_selection_factor=1),
                constants.TARGET_DEVICE_F28P55: dict(performance_infer_time_ms=2.208,
                                                     accuracy_factor='TBD', model_selection_factor=1),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },
    'MotorFault_model_3_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
            task_type=constants.TASK_TYPE_MOTOR_FAULT,
        ),
        'download': model_urls['MotorFault_model_3_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_MF_3L',
            model_name='MotorFault_model_3_t',
            model_architecture='backbone',
            learning_rate=0.01,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_mf_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_MOTOR_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(performance_infer_time_ms=18.79, accuracy_factor='TBD',
                                                      model_selection_factor=1, ),
                constants.TARGET_DEVICE_F28003: dict(performance_infer_time_ms=18.79,
                                                     accuracy_factor='TBD', model_selection_factor=1),
                constants.TARGET_DEVICE_F28004: dict(performance_infer_time_ms=22.548,
                                                     accuracy_factor='TBD', model_selection_factor=1),
                constants.TARGET_DEVICE_F2837: dict(performance_infer_time_ms=18.79,
                                                    accuracy_factor='TBD', model_selection_factor=1),
                constants.TARGET_DEVICE_F28P65: dict(performance_infer_time_ms=11.274,
                                                     accuracy_factor='TBD', model_selection_factor=1),
                constants.TARGET_DEVICE_F28P55: dict(performance_infer_time_ms=2.208,
                                                     accuracy_factor='TBD', model_selection_factor=1),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        'compilation': dict()
    },
}

enabled_models_list = [
    'TimeSeries_Generic_7k', 'TimeSeries_Generic_3k',
    # 'TimeSeries_Generic_AF_7k', 'TimeSeries_Generic_AF_3k',
    # 'TimeSeries_Generic_MF_7k', 'TimeSeries_Generic_MF_3k',
    'TimeSeries_Generic_7k_t', 'TimeSeries_Generic_3k_t',
    # 'TimeSeries_Generic_AF_7k_t', 'TimeSeries_Generic_AF_3k_t',
    # 'TimeSeries_Generic_MF_7k_t', 'TimeSeries_Generic_MF_3k_t',
    # 'ArcFault_model_200', 'ArcFault_model_300', 'ArcFault_model_700',
    'ArcFault_model_200_t', 'ArcFault_model_300_t', 'ArcFault_model_700_t',
    # 'MotorFault_model_1', 'MotorFault_model_2', 'MotorFault_model_3',
    'MotorFault_model_1_t', 'MotorFault_model_2_t', 'MotorFault_model_3_t',
]


def get_model_descriptions(task_type=None):
    model_descriptions_selected = {k: v for k, v in _model_descriptions.items() if k in enabled_models_list}
    return model_descriptions_selected


def get_model_description(model_name):
    model_descriptions = get_model_descriptions()
    return model_descriptions[model_name] if model_name in model_descriptions else None


class ModelTraining:
    @classmethod
    def init_params(self, *args, **kwargs):
        params = dict(training=dict())
        params = utils.ConfigDict(params, *args, **kwargs)
        return params

    def __init__(self, *args, quit_event=None, **kwargs):
        self.params = self.init_params(*args, **kwargs)
        self.quit_event = quit_event

        # num classes
        # TODO: Is the below required?
        '''
        self.train_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_train.json'
        self.val_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_val.json'
        with open(self.train_ann_file) as train_ann_fp:
            train_anno = json.load(train_ann_fp)
            categories = train_anno['categories']
            self.object_categories = [cat['name'] for cat in categories]
        #
        '''

        log_summary_regex = {
            'js': [
                # Floating Point Training Metrics
                {'type': 'Epoch (FloatTrain)', 'name': 'Epoch (FloatTrain)', 'description': 'Epochs (FloatTrain)',
                 'unit': 'Epoch', 'value': None,
                 'regex': [
                     {'op': 'search', 'pattern': r'FloatTrain: Epoch:\s+\[(?<eid>\d+)\]\s+Total', 'groupId': 'eid'}],
                 },
                {'type': 'Training Loss (FloatTrain)', 'name': 'Loss (FloatTrain)',
                 'description': 'Training Loss (FloatTrain)', 'unit': 'Loss', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'FloatTrain: Epoch:\s\[\d+\]\sTotal\stime:[\s\S]*?loss\:\s+(?<loss>\d+\.\d+)',
                            'groupId': 'loss'}],
                 },
                {'type': 'Validation Accuracy (FloatTrain)', 'name': 'Accuracy (FloatTrain)',
                 'description': 'Validation Accuracy (FloatTrain)', 'unit': 'Accuracy Top-1%', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'FloatTrain: Test:\s+\s+Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                            'groupId': 'accuracy', 'scale_factor': 1}],
                 },
                {'type': 'F1-Score (FloatTrain)', 'name': 'F1-Score (FloatTrain)',
                 'description': 'F1-Score (FloatTrain)', 'unit': 'F1-Score', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'FloatTrain: Test:\s+\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                            'groupId': 'f1score', 'scale_factor': 1}],
                 },
                {'type': 'Confusion Matrix (FloatTrain)', 'name': 'Confusion Matrix (FloatTrain)',
                 'description': 'Confusion Matrix (FloatTrain)', 'unit': 'Confusion Matrix', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'FloatTrain:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\w\s\:\=\+\-\|]+)\s*INFO',
                            'groupId': 'cm', 'scale_factor': 1}],
                 },
                # Quantized Training
                {'type': 'Epoch (QuantTrain)', 'name': 'Epoch (QuantTrain)', 'description': 'Epochs (QuantTrain)',
                 'unit': 'Epoch', 'value': None,
                 'regex': [
                     {'op': 'search', 'pattern': r'QuantTrain: Epoch:\s+\[(?<eid>\d+)\]\s+Total', 'groupId': 'eid'}],
                 },
                {'type': 'Training Loss (QuantTrain)', 'name': 'Loss (QuantTrain)',
                 'description': 'Training Loss (QuantTrain)', 'unit': 'Loss', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'QuantTrain: Epoch:\s\[\d+\]\sTotal\stime:[\s\S]*?loss\:\s+(?<loss>\d+\.\d+)',
                            'groupId': 'loss'}],
                 },
                {'type': 'F1-Score (QuantTrain)', 'name': 'F1-Score (QuantTrain)',
                 'description': 'F1-Score (QuantTrain)', 'unit': 'F1-Score', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'QuantTrain: Test:\s+\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                            'groupId': 'f1score', 'scale_factor': 1}],
                 },
                {'type': 'Confusion Matrix (QuantTrain)', 'name': 'Confusion Matrix (QuantTrain)',
                 'description': 'Confusion Matrix (QuantTrain)', 'unit': 'Confusion Matrix', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'QuantTrain:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\w\s\:\=\+\-\|]+)\s*INFO',
                            'groupId': 'cm', 'scale_factor': 1}],
                 },
                {'type': 'Validation Accuracy (QuantTrain)', 'name': 'Accuracy (QuantTrain)',
                 'description': 'Validation Accuracy (QuantTrain)', 'unit': 'Accuracy Top-1%', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'QuantTrain: Test:\s+\s+Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                            'groupId': 'accuracy', 'scale_factor': 1}],
                 },
                # Best Epoch QuantTrain Metrics
                {'type': 'Epoch (QuantTrain, BestEpoch)', 'name': 'Epoch (QuantTrain, BestEpoch)', 'description': 'Epochs (QuantTrain, BestEpoch)',
                 'unit': 'Epoch', 'value': None,
                 'regex': [
                     {'op': 'search', 'pattern': r'QuantTrain.BestEpoch: Best Epoch:\s+(?<eid>\d+)\s+', 'groupId': 'eid'}],
                 },
                # {'type': 'Training Loss (QuantTrain, BestEpoch)', 'name': 'Loss (QuantTrain, BestEpoch)',
                #  'description': 'Training Loss (QuantTrain, BestEpoch)', 'unit': 'Loss', 'value': None,
                #  'regex': [{'op': 'search',
                #             'pattern': r'QuantTrain.BestEpoch: Epoch:\s\[\d+\]\sTotal\stime:[\s\S]*?loss\:\s+(?<loss>\d+\.\d+)',
                #             'groupId': 'loss'}],
                #  },
                {'type': 'F1-Score (QuantTrain, BestEpoch)', 'name': 'F1-Score (QuantTrain, BestEpoch)',
                 'description': 'F1-Score (QuantTrain, BestEpoch)', 'unit': 'F1-Score', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'QuantTrain.BestEpoch:\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                            'groupId': 'f1score', 'scale_factor': 1}],
                 },
                {'type': 'Confusion Matrix (QuantTrain, BestEpoch)', 'name': 'Confusion Matrix (QuantTrain, BestEpoch)',
                 'description': 'Confusion Matrix (QuantTrain, BestEpoch)', 'unit': 'Confusion Matrix', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'QuantTrain.BestEpoch:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\w\s\:\=\+\-\|]+)\s*INFO',
                            'groupId': 'cm', 'scale_factor': 1}],
                 },
                {'type': 'Validation Accuracy (QuantTrain, BestEpoch)', 'name': 'Accuracy (QuantTrain, BestEpoch)',
                 'description': 'Validation Accuracy (QuantTrain, BestEpoch)', 'unit': 'Accuracy Top-1%', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'QuantTrain.BestEpoch: Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                            'groupId': 'accuracy', 'scale_factor': 1}],
                 },
                # Best Epoch FloatTrain Metrics
                {'type': 'Epoch (FloatTrain, BestEpoch)', 'name': 'Epoch (FloatTrain, BestEpoch)',
                 'description': 'Epochs (FloatTrain, BestEpoch)',
                 'unit': 'Epoch', 'value': None,
                 'regex': [
                     {'op': 'search', 'pattern': r'FloatTrain.BestEpoch\s*: Best Epoch:\s+(?<eid>\d+)\s+',
                      'groupId': 'eid'}],
                 },
                # {'type': 'Training Loss (FloatTrain, BestEpoch)', 'name': 'Loss (FloatTrain, BestEpoch)',
                #  'description': 'Training Loss (FloatTrain, BestEpoch)', 'unit': 'Loss', 'value': None,
                #  'regex': [{'op': 'search',
                #             'pattern': r'FloatTrain.BestEpoch: Epoch:\s\[\d+\]\sTotal\stime:[\s\S]*?loss\:\s+(?<loss>\d+\.\d+)',
                #             'groupId': 'loss'}],
                #  },
                {'type': 'F1-Score (FloatTrain, BestEpoch)', 'name': 'F1-Score (FloatTrain, BestEpoch)',
                 'description': 'F1-Score (FloatTrain, BestEpoch)', 'unit': 'F1-Score', 'value': None,
                 'regex': [
                     {'op': 'search', 'pattern': r'FloatTrain.BestEpoch\s*:\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                      'groupId': 'f1score', 'scale_factor': 1}],
                 },
                {'type': 'Confusion Matrix (FloatTrain, BestEpoch)', 'name': 'Confusion Matrix (FloatTrain, BestEpoch)',
                 'description': 'Confusion Matrix (FloatTrain, BestEpoch)', 'unit': 'Confusion Matrix', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'FloatTrain.BestEpoch\s*:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\w\s\:\=\+\-\|]+)\s*INFO',
                            'groupId': 'cm', 'scale_factor': 1}],
                 },
                {'type': 'Validation Accuracy (FloatTrain, BestEpoch)', 'name': 'Accuracy (FloatTrain, BestEpoch)',
                 'description': 'Validation Accuracy (FloatTrain, BestEpoch)', 'unit': 'Accuracy Top-1%', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'FloatTrain.BestEpoch\s*: Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                            'groupId': 'accuracy', 'scale_factor': 1}],
                 },
                # Test data
                {'type': 'Test Accuracy (Test Data)', 'name': 'Accuracy (Test Data)',
                 'description': 'Test Accuracy (Test Data)', 'unit': 'Accuracy Top-1%', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'test_data\s*:\s*Test Data Evaluation Accuracy:\s+(?<accuracy>[-+e\d+\.\d+]+)%',
                            'groupId': 'accuracy', 'scale_factor': 1}],
                 },
                {'type': 'Confusion Matrix (Test Data)', 'name': 'Confusion Matrix',
                 'description': 'Confusion Matrix (Test Data)', 'unit': 'Confusion Matrix', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'test_data\s*:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\w\s\:\=\+\-\|]+)(\r\n|\r|\n)^$',
                            'groupId': 'cm', 'scale_factor': 1}],
                 },
                {'type': 'Matrix Label', 'name': 'Matrix Label', 'description': 'Matrix Label',
                 'unit': 'Matrix Label', 'value': None,
                    "regex": [{'op': 'search', 'pattern': r'Ground Truth:\s*(?<label>\w+)\s*\|\s*',
                               'scale_factor': 1, 'groupId': 'label'}],
                },
                {'type': 'Matrix Cell', 'name': 'Matrix Cell', 'description': 'Matrix Cell',
                 'unit': 'Matrix Cell', 'value': None,
                 "regex": [{'op': 'search', 'pattern': r'\|\s*(?<cell>\d+)',
                            'scale_factor': 1, 'groupId': 'cell'}],
                 },

            ]
        }
        # update params that are specific to this backend and model
        self.params.update(
            training=utils.ConfigDict(
                log_file_path=os.path.join(
                    self.params.training.train_output_path if self.params.training.train_output_path else self.params.training.training_path,
                    'run.log'),
                log_summary_regex=log_summary_regex,
                summary_file_path=os.path.join(self.params.training.training_path, 'summary.yaml'),
                model_checkpoint_path=os.path.join(self.params.training.training_path, 'checkpoint.pth'),
                model_checkpoint_path_quantization=os.path.join(self.params.training.training_path_quantization,
                                                                'checkpoint.pth'),
                model_export_path=os.path.join(self.params.training.training_path, 'model.onnx'),
                model_export_path_quantization=os.path.join(self.params.training.training_path_quantization,
                                                            'model.onnx'),
                model_proto_path=None,
                tspa_license_path=os.path.abspath(os.path.join(os.path.dirname(tinyml_modelmaker.ai_modules.timeseries.training.tinyml_tinyverse.__file__), 'LICENSE.txt'))
                # num_classes=self.params.training.num_classes,  # len(self.object_categories)
            )
        )

    def clear(self):
        # clear the training folder
        shutil.rmtree(self.params.training.training_path, ignore_errors=True)

    def run(self, **kwargs):
        ''''
        The actual training function. Move this to a worker process, if this function is called from a GUI.
        '''
        os.makedirs(self.params.training.training_path, exist_ok=True)

        distributed = 1 if self.params.training.num_gpus > 1 else 0
        device = 'cuda' if self.params.training.num_gpus > 0 else 'cpu'
        # training params
        argv = ['--model', f'{self.params.training.model_training_id}',
                '--dual-op', f'{self.params.training.dual_op}',
                '--model-config', f'{self.params.training.model_config}',
                '--model-spec', f'{self.params.training.model_spec}',
                # '--weights', f'{self.params.training.pretrained_checkpoint_path}',
                '--dataset', 'modelmaker',
                '--dataset-loader', f'{self.params.training.dataset_loader}',
                '--annotation-prefix', f'{self.params.dataset.annotation_prefix}',
                #'--num-classes', f'{self.params.training.num_classes}',
                '--gpus', f'{self.params.training.num_gpus}',
                '--batch-size', f'{self.params.training.batch_size}',
                '--opt', f'{self.params.training.optimizer}',
                '--weight-decay', f'{self.params.training.weight_decay}',
                '--lr-scheduler', f'{self.params.training.lr_scheduler}',
                '--lr-warmup-epochs', '1',
                '--distributed', f'{distributed}',
                '--device', f'{device}',
                # '--out_dir', f'{self.params.training.}',
                '--sampling-rate', f'{self.params.data_processing.sampling_rate}',
                '--resampling-factor', f'{self.params.data_processing.resampling_factor}',
                '--new-sr', f'{self.params.data_processing.new_sr}',
                '--stride_window', f'{self.params.data_processing.stride_window}',
                '--sequence_window', f'{self.params.data_processing.sequence_window}',
                '--data-proc-transforms', self.params.data_processing.transforms,
                '--feat-ext-transform', f'{self.params.feature_extraction.transform}',

                '--generic-model', f'{self.params.common.generic_model}',

                '--feat-ext-store-dir', f'{self.params.feature_extraction.feat_ext_store_dir}',
                '--dont-train-just-feat-ext', f'{self.params.feature_extraction.dont_train_just_feat_ext}',

                '--frame-size', f'{self.params.feature_extraction.frame_size}',
                '--feature-size-per-frame', f'{self.params.feature_extraction.feature_size_per_frame}',
                '--num-frame-concat', f'{self.params.feature_extraction.num_frame_concat}',
                '--frame-skip', f'{self.params.feature_extraction.frame_skip}',
                '--min-fft-bin', f'{self.params.feature_extraction.min_fft_bin}',
                '--fft-bin-size', f'{self.params.feature_extraction.fft_bin_size}',
                '--dc-remove', f'{self.params.feature_extraction.dc_remove}',
                # '--num-channel', f'{self.params.feature_extraction.num_channel}',
                '--stacking', f'{self.params.feature_extraction.stacking}',
                '--offset', f'{self.params.feature_extraction.offset}',
                '--scale', f'{self.params.feature_extraction.scale}',

                #'--tensorboard-logger', 'True',
                '--variables', f'{self.params.data_processing.variables}',
                '--with-input-batchnorm', f'{self.params.training.with_input_batchnorm}',
                '--lis', f'{self.params.training.log_file_path}',
                # Do not add newer arguments after this line, it will change the behaviour of the code.
                '--data-path', os.path.join(self.params.dataset.dataset_path, self.params.dataset.data_dir),
                '--store-feat-ext-data', f'{self.params.feature_extraction.store_feat_ext_data}',
                '--epochs', f'{self.params.training.training_epochs}',
                '--lr', f'{self.params.training.learning_rate}',
                '--output-dir', f'{self.params.training.training_path}',
                ]

        # import dynamically - force_import every time to avoid clashes with scripts in other repositories
        # train = utils.import_file_or_folder(
        #     os.path.join(tinyml_tinyverse_path, 'references', 'timeseries_classification', 'train.py'),
        #     __name__, force_import=True)
        args = train.get_args_parser().parse_args(argv)
        args.quit_event = self.quit_event
        if self.params.testing.skip_train not in [True, 'True', 'true', 1, '1']:  # Is user wants to only test their model
            if self.params.training.run_quant_train_only in [True, 'True', 'true', 1, '1']:
                if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
                    argv = argv[:-2]  # Remove --output-dir <output-dir>
                    argv.extend([
                        '--output-dir', f'{self.params.training.training_path_quantization}',
                        '--quantization', f'{self.params.training.quantization}', ]),

                    args = train.get_args_parser().parse_args(argv)
                    args.quit_event = self.quit_event
                    # launch the training
                    # TODO: originally train.main(args)
                    train.run(args)
                else:
                    raise f"quantization cannot be {TinyMLQuantizationVersion.NO_QUANTIZATION} if run_quant_train_only argument is chosen"
            else:
                train.run(args)

                if (self.params.feature_extraction.store_feat_ext_data in [True, 'True', 'true', 1, '1']
                        and
                        self.params.feature_extraction.dont_train_just_feat_ext in [True, 'True', 'true', 1, '1']):
                    return self.params

                if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
                    argv = argv[
                           :-8]  # remove --store-feat-ext-data <True/False> --epochs <epochs> --lr <lr> --output-dir <output-dir>
                    argv.extend([
                        '--output-dir', f'{self.params.training.training_path_quantization}',
                        '--epochs', f'{max(5, self.params.training.training_epochs // 10)}',
                        '--lr', f'{self.params.training.learning_rate / 100}',
                        '--weights', f'{self.params.training.model_checkpoint_path}',
                        '--quantization', f'{self.params.training.quantization}',
                        '--lr-warmup-epochs', '0',
                        '--store-feat-ext-data', 'False']),

                    args = train.get_args_parser().parse_args(argv)
                    args.quit_event = self.quit_event
                    # launch the training
                    # TODO: originally train.main(args)
                    train.run(args)

        if self.params.testing.enable in [True, 'true', 1, '1']:
            if self.params.testing.test_data and (os.path.exists(self.params.testing.test_data)):
                data_path = self.params.testing.test_data
            else:
                data_path = os.path.join(self.params.dataset.dataset_path, self.params.dataset.data_dir)

            if self.params.testing.model_path and (os.path.exists(self.params.testing.model_path)):
                model_path = self.params.testing.model_path
            else:
                model_path = os.path.join(self.params.training.training_path_quantization, 'model.onnx')
            argv = [
                '--dataset', 'modelmaker',
                '--dataset-loader', f'{self.params.training.dataset_loader}',
                '--annotation-prefix', f'{self.params.dataset.annotation_prefix}',
                '--gpus', f'{self.params.training.num_gpus}',
                '--batch-size', f'{self.params.training.batch_size}',
                '--distributed', '0',
                '--device', f'{device}',

                '--variables', f'{self.params.data_processing.variables}',
                '--sampling-rate', f'{self.params.data_processing.sampling_rate}',
                '--resampling-factor', f'{self.params.data_processing.resampling_factor}',
                '--new-sr', f'{self.params.data_processing.new_sr}',
                '--stride_window', f'{self.params.data_processing.stride_window}',
                '--sequence_window', f'{self.params.data_processing.sequence_window}',

                '--data-proc-transforms', self.params.data_processing.transforms,
                '--feat-ext-transform', f'{self.params.feature_extraction.transform}',
                # Arc Fault and Motor Fault Related Params
                '--frame-size', f'{self.params.feature_extraction.frame_size}',
                '--feature-size-per-frame', f'{self.params.feature_extraction.feature_size_per_frame}',
                '--num-frame-concat', f'{self.params.feature_extraction.num_frame_concat}',
                '--frame-skip', f'{self.params.feature_extraction.frame_skip}',
                '--min-fft-bin', f'{self.params.feature_extraction.min_fft_bin}',
                '--fft-bin-size', f'{self.params.feature_extraction.fft_bin_size}',
                '--dc-remove', f'{self.params.feature_extraction.dc_remove}',
                # '--num-channel', f'{self.params.feature_extraction.num_channel}',
                '--stacking', f'{self.params.feature_extraction.stacking}',
                '--offset', f'{self.params.feature_extraction.offset}',
                '--scale', f'{self.params.feature_extraction.scale}',

                # '--tensorboard-logger', 'True',
                '--lis', f'{self.params.training.log_file_path}',
                '--data-path', f'{data_path}',
                '--output-dir', f'{self.params.training.training_path_quantization}',
                '--model-path', f'{model_path}',
                '--generic-model', f'{self.params.common.generic_model}',
                ]
            # test = utils.import_file_or_folder(os.path.join(tinyml_tinyverse_path, 'references', 'timeseries_classification', 'test_onnx.py'), __name__, force_import=True)

            args = test.get_args_parser().parse_args(argv)
            args.quit_event = self.quit_event
            # launch the training
            test.run(args)

        return self.params

    def stop(self):
        if self.quit_event is not None:
            self.quit_event.set()
            return True
        #
        return False

    def get_params(self):
        return self.params
