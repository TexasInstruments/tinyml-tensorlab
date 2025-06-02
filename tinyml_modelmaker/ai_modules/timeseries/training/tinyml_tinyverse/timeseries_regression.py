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

import torch.backends.mps

from tinyml_tinyverse.references.timeseries_regression import test_onnx as test
from tinyml_tinyverse.references.timeseries_regression import train
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

import tinyml_modelmaker

from ..... import utils
from ... import constants
from .device_run_info import DEVICE_RUN_INFO
# Let the default model device run info dict be blank
MOD_DEV_RUN = dict(inference_time_us='TBD', flash='TBD', sram='TBD')

this_dir_path = os.path.dirname(os.path.abspath(__file__))
repo_parent_path = os.path.abspath(os.path.join(this_dir_path, '..', '..', '..', '..', '..', '..'))

# tinyml_tinyverse_path = os.path.join(repo_parent_path, 'tinyml-tinyverse')

model_urls = {
    'TimeSeries_Generic_Regr_13k_t': {'download_url': '', 'download_path': '', },
    'TimeSeries_Generic_Regr_3k_t': {'download_url': '', 'download_path': '', },
}


model_info_str = "Inference time numbers are for comparison purposes only. (Input Size: {})"
_model_descriptions = {
    # Regression Models
    'TimeSeries_Generic_Regr_13k_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_REGRESSION,
            task_type=constants.TASK_TYPE_GENERIC_TS_REGRESSION,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_Regr_13k_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            dataset_loader='GenericTSDatasetReg',
            training_backend='tinyml_tinyverse',
            model_training_id='CNN_TS_GEN_BASE_13K',
            model_name='TimeSeries_Generic_Regr_13k_t',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_REGRESSION],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P55]),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
				constants.TRAINING_DEVICE_MPS: True,
            }
        ),
        'compilation': dict()
    },
    'TimeSeries_Generic_Regr_3k_t': {
        'common': dict(
            task_category=constants.TASK_CATEGORY_TS_REGRESSION,
            task_type=constants.TASK_TYPE_GENERIC_TS_REGRESSION,
            generic_model=True,
        ),
        'download': model_urls['TimeSeries_Generic_Regr_3k_t'],
        'training': dict(
            quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
            with_input_batchnorm=True,
            dataset_loader='GenericTSDatasetReg',
            training_backend='tinyml_tinyverse',
            model_training_id='REG_TS_GEN_BASE_3K',
            model_name='TimeSeries_Generic_Regr_3k_t',
            learning_rate=2e-3,
            model_spec=None,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_REGRESSION],
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P55]),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
				constants.TRAINING_DEVICE_MPS: True,
            }
        ),
        'compilation': dict()
    },
}

enabled_models_list = [
    # Regression Models
    'TimeSeries_Generic_Regr_13k_t',
    'TimeSeries_Generic_Regr_3k_t',
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
                model_export_path=os.path.join(self.params.training.training_path, 'model.onnx'),
                model_proto_path=None,
                tspa_license_path=os.path.abspath(os.path.join(os.path.dirname(tinyml_modelmaker.ai_modules.timeseries.training.tinyml_tinyverse.__file__), 'LICENSE.txt'))
                # num_classes=self.params.training.num_classes,  # len(self.object_categories)
            )
        )
        if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
            self.params.update(
                training=utils.ConfigDict(
                    model_checkpoint_path_quantization=os.path.join(self.params.training.training_path_quantization,
                                                                    'checkpoint.pth'),
                    model_export_path_quantization=os.path.join(self.params.training.training_path_quantization,
                                                                'model.onnx'),
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
        device = 'cpu'
        if self.params.training.num_gpus > 0:
            if torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cuda'
        # training params
        argv = ['--model', f'{self.params.training.model_training_id}',
                '--dual-op', f'{self.params.training.dual_op}',
                '--model-config', f'{self.params.training.model_config}',
                '--augment-config', f'{self.params.training.augment_config}',
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
                '--sampling-rate', f'{self.params.data_processing_feature_extraction.sampling_rate}',
                '--resampling-factor', f'{self.params.data_processing_feature_extraction.resampling_factor}',
                '--new-sr', f'{self.params.data_processing_feature_extraction.new_sr}',
                #'--sequence-window', f'{self.params.data_processing_feature_extraction.sequence_window}',
                '--stride-size', f'{self.params.data_processing_feature_extraction.stride_size}',
                '--data-proc-transforms', self.params.data_processing_feature_extraction.data_proc_transforms,
                '--feat-ext-transform', self.params.data_processing_feature_extraction.feat_ext_transform,

                '--generic-model', f'{self.params.common.generic_model}',

                '--feat-ext-store-dir', f'{self.params.data_processing_feature_extraction.feat_ext_store_dir}',
                '--dont-train-just-feat-ext', f'{self.params.data_processing_feature_extraction.dont_train_just_feat_ext}',

                '--frame-size', f'{self.params.data_processing_feature_extraction.frame_size}',
                '--feature-size-per-frame', f'{self.params.data_processing_feature_extraction.feature_size_per_frame}',
                '--num-frame-concat', f'{self.params.data_processing_feature_extraction.num_frame_concat}',
                '--frame-skip', f'{self.params.data_processing_feature_extraction.frame_skip}',
                '--min-bin', f'{self.params.data_processing_feature_extraction.min_bin}',
                '--normalize-bin', f'{self.params.data_processing_feature_extraction.normalize_bin}',
                '--dc-remove', f'{self.params.data_processing_feature_extraction.dc_remove}',
                '--analysis-bandwidth', f'{self.params.data_processing_feature_extraction.analysis_bandwidth}',
                # '--num-channel', f'{self.params.data_processing_feature_extraction.num_channel}',
                '--log-base', f'{self.params.data_processing_feature_extraction.log_base}',
                '--log-mul', f'{self.params.data_processing_feature_extraction.log_mul}',
                '--log-threshold', f'{self.params.data_processing_feature_extraction.log_threshold}',
                '--stacking', f'{self.params.data_processing_feature_extraction.stacking}',
                '--offset', f'{self.params.data_processing_feature_extraction.offset}',
                '--scale', f'{self.params.data_processing_feature_extraction.scale}',
                '--nn-for-feature-extraction', f'{self.params.data_processing_feature_extraction.nn_for_feature_extraction}',
                '--output-dequantize', f'{self.params.training.output_dequantize}',

                #'--tensorboard-logger', 'True',
                '--variables', f'{self.params.data_processing_feature_extraction.variables}',
                '--with-input-batchnorm', f'{self.params.training.with_input_batchnorm}',
                '--lis', f'{self.params.training.log_file_path}',
                # Do not add newer arguments after this line, it will change the behavior of the code.
                '--data-path', os.path.join(self.params.dataset.dataset_path, self.params.dataset.data_dir),
                '--store-feat-ext-data', f'{self.params.data_processing_feature_extraction.store_feat_ext_data}',
                '--epochs', f'{self.params.training.training_epochs}',
                '--lr', f'{self.params.training.learning_rate}',
                '--output-dir', f'{self.params.training.training_path}',
                ]

        # import dynamically - force_import every time to avoid clashes with scripts in other repositories
        # train = utils.import_file_or_folder(
        #     os.path.join(tinyml_tinyverse_path, 'references', 'timeseries_regression', 'train.py'),
        #     __name__, force_import=True)
        args = train.get_args_parser().parse_args(argv)
        args.quit_event = self.quit_event
        if not utils.misc_utils.str2bool(self.params.testing.skip_train):  # Is user wants to only test their model
            if utils.misc_utils.str2bool(self.params.training.run_quant_train_only):
                if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
                    argv = argv[:-2]  # Remove --output-dir <output-dir>
                    argv.extend([
                        '--output-dir', f'{self.params.training.training_path_quantization}',
                        '--quantization', f'{self.params.training.quantization}',
                        '--quantization-method', f'{self.params.training.quantization_method}',
                        '--weight-bitwidth', f'{self.params.training.quantization_weight_bitwidth}',
                        '--activation-bitwidth', f'{self.params.training.quantization_activation_bitwidth}',
                    ]),

                    args = train.get_args_parser().parse_args(argv)
                    args.quit_event = self.quit_event
                    # launch the training
                    # TODO: originally train.main(args)
                    train.run(args)
                else:
                    raise f"quantization cannot be {TinyMLQuantizationVersion.NO_QUANTIZATION} if run_quant_train_only argument is chosen"
            else:
                train.run(args)

                if utils.misc_utils.str2bool(self.params.data_processing_feature_extraction.store_feat_ext_data) and utils.misc_utils.str2bool(self.params.data_processing_feature_extraction.dont_train_just_feat_ext):
                    return self.params

                if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
                    argv = argv[:-8]
                    # remove --store-feat-ext-data <True/False> --epochs <epochs> --lr <lr> --output-dir <output-dir>
                    argv.extend([
                        '--output-dir', f'{self.params.training.training_path_quantization}',
                        '--epochs', f'{max(50, self.params.training.training_epochs // 5)}',
                        '--lr', f'{self.params.training.learning_rate / 100}',
                        '--weights', f'{self.params.training.model_checkpoint_path}',
                        '--quantization', f'{self.params.training.quantization}',
                        '--quantization-method', f'{self.params.training.quantization_method}',
                        '--weight-bitwidth', f'{self.params.training.quantization_weight_bitwidth}',
                        '--activation-bitwidth', f'{self.params.training.quantization_activation_bitwidth}',
                        '--lr-warmup-epochs', '0',
                        '--store-feat-ext-data', 'False']),

                    args = train.get_args_parser().parse_args(argv)
                    args.quit_event = self.quit_event
                    # launch the training
                    # TODO: originally train.main(args)
                    train.run(args)

        if utils.misc_utils.str2bool(self.params.testing.enable):
            if self.params.testing.test_data and (os.path.exists(self.params.testing.test_data)):
                data_path = self.params.testing.test_data
            else:
                data_path = os.path.join(self.params.dataset.dataset_path, self.params.dataset.data_dir)

            if self.params.testing.model_path and (os.path.exists(self.params.testing.model_path)):
                model_path = self.params.testing.model_path
            else:
                if self.params.training.quantization == TinyMLQuantizationVersion.NO_QUANTIZATION:
                    model_path = os.path.join(self.params.training.training_path, 'model.onnx')
                    output_dir = self.params.training.training_path
                else:
                    model_path = os.path.join(self.params.training.training_path_quantization, 'model.onnx')
                    output_dir = self.params.training.training_path_quantization
            argv = [
                '--dataset', 'modelmaker',
                '--dataset-loader', f'{self.params.training.dataset_loader}',
                '--annotation-prefix', f'{self.params.dataset.annotation_prefix}',
                '--gpus', f'{self.params.training.num_gpus}',
                '--batch-size', f'{self.params.training.batch_size}',
                '--distributed', '0',
                '--device', f'{device}',

                '--variables', f'{self.params.data_processing_feature_extraction.variables}',
                '--sampling-rate', f'{self.params.data_processing_feature_extraction.sampling_rate}',
                '--resampling-factor', f'{self.params.data_processing_feature_extraction.resampling_factor}',
                '--new-sr', f'{self.params.data_processing_feature_extraction.new_sr}',
                #'--sequence-window', f'{self.params.data_processing_feature_extraction.sequence_window}',
                '--stride-size', f'{self.params.data_processing_feature_extraction.stride_size}',

                '--data-proc-transforms', self.params.data_processing_feature_extraction.data_proc_transforms,
                '--feat-ext-transform', self.params.data_processing_feature_extraction.feat_ext_transform,
                # Arc Fault and Motor Fault Related Params
                '--frame-size', f'{self.params.data_processing_feature_extraction.frame_size}',
                '--feature-size-per-frame', f'{self.params.data_processing_feature_extraction.feature_size_per_frame}',
                '--num-frame-concat', f'{self.params.data_processing_feature_extraction.num_frame_concat}',
                '--frame-skip', f'{self.params.data_processing_feature_extraction.frame_skip}',
                '--min-bin', f'{self.params.data_processing_feature_extraction.min_bin}',
                '--normalize-bin', f'{self.params.data_processing_feature_extraction.normalize_bin}',
                '--dc-remove', f'{self.params.data_processing_feature_extraction.dc_remove}',
                '--analysis-bandwidth', f'{self.params.data_processing_feature_extraction.analysis_bandwidth}',
                # '--num-channel', f'{self.params.data_processing_feature_extraction.num_channel}',
                '--log-base', f'{self.params.data_processing_feature_extraction.log_base}',
                '--log-mul', f'{self.params.data_processing_feature_extraction.log_mul}',
                '--log-threshold', f'{self.params.data_processing_feature_extraction.log_threshold}',
                '--stacking', f'{self.params.data_processing_feature_extraction.stacking}',
                '--offset', f'{self.params.data_processing_feature_extraction.offset}',
                '--scale', f'{self.params.data_processing_feature_extraction.scale}',
                '--nn-for-feature-extraction', f'{self.params.data_processing_feature_extraction.nn_for_feature_extraction}',
                '--output-dequantize', f'{self.params.training.output_dequantize}',

                # '--tensorboard-logger', 'True',
                '--lis', f'{self.params.training.log_file_path}',
                '--data-path', f'{data_path}',
                '--output-dir', output_dir,
                '--model-path', f'{model_path}',
                '--generic-model', f'{self.params.common.generic_model}',
                ]
            # test = utils.import_file_or_folder(os.path.join(tinyml_tinyverse_path, 'references', 'timeseries_regression', 'test_onnx.py'), __name__, force_import=True)

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
