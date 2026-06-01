#################################################################################
# Copyright (c) 2023-2026, Texas Instruments
# All Rights Reserved.
#################################################################################

import os
import shutil
from copy import deepcopy

import torch.backends.mps

from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

import tinyml_modelmaker

from ..... import utils
from ... import constants


def get_image_classification_log_summary_regex():
    return {
        'js': [
            # Floating Point Training
            {'type': 'Epoch (FloatTrain)', 'name': 'Epoch (FloatTrain)', 'description': 'Epochs (FloatTrain)',
             'unit': 'Epoch', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'FloatTrain: Epoch:\s+\[(?<eid>\d+)\]\s+Total',
                         'groupId': 'eid'}],
             },
            {'type': 'Training Loss (FloatTrain)', 'name': 'Loss (FloatTrain)',
             'description': 'Training Loss (FloatTrain)', 'unit': 'Loss', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'FloatTrain: Epoch:.*?loss:\s+(?<loss>[0-9\.]+)',
                         'groupId': 'loss'}],
             },
            {'type': 'Validation Accuracy (FloatTrain)', 'name': 'Accuracy (FloatTrain)',
             'description': 'Validation Accuracy (FloatTrain)', 'unit': 'Accuracy Top-1%', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'FloatTrain: Test:\s+\s+Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                         'groupId': 'accuracy',
                         'scale_factor': 1}],
             },
            {'type': 'F1-Score (FloatTrain)', 'name': 'F1-Score (FloatTrain)',
             'description': 'F1-Score (FloatTrain)', 'unit': 'F1-Score', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'FloatTrain: Test:\s+\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                         'groupId': 'f1score',
                         'scale_factor': 1}],
             },
            {'type': 'Confusion Matrix (FloatTrain)', 'name': 'Confusion Matrix (FloatTrain)',
             'description': 'Confusion Matrix (FloatTrain)', 'unit': 'Confusion Matrix', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'FloatTrain:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\s\S]+?)INFO',
                         'groupId': 'cm',
                         'scale_factor': 1}],
             },

            # Quantized Training
            {'type': 'Epoch (QuantTrain)', 'name': 'Epoch (QuantTrain)', 'description': 'Epochs (QuantTrain)',
             'unit': 'Epoch', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'QuantTrain: Epoch:\s+\[(?<eid>\d+)\]\s+Total',
                         'groupId': 'eid'}],
             },
            {'type': 'Training Loss (QuantTrain)', 'name': 'Loss (QuantTrain)',
             'description': 'Training Loss (QuantTrain)', 'unit': 'Loss', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'QuantTrain: Epoch:.*?loss:\s+(?<loss>[0-9\.]+)',
                         'groupId': 'loss'}],
             },
            {'type': 'Validation Accuracy (QuantTrain)', 'name': 'Accuracy (QuantTrain)',
             'description': 'Validation Accuracy (QuantTrain)', 'unit': 'Accuracy Top-1%', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'QuantTrain: Test:\s+\s+Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                         'groupId': 'accuracy',
                         'scale_factor': 1}],
             },
            {'type': 'F1-Score (QuantTrain)', 'name': 'F1-Score (QuantTrain)',
             'description': 'F1-Score (QuantTrain)', 'unit': 'F1-Score', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'QuantTrain: Test:\s+\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                         'groupId': 'f1score',
                         'scale_factor': 1}],
             },
            {'type': 'Confusion Matrix (QuantTrain)', 'name': 'Confusion Matrix (QuantTrain)',
             'description': 'Confusion Matrix (QuantTrain)', 'unit': 'Confusion Matrix', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'QuantTrain:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\s\S]+?)INFO',
                         'groupId': 'cm',
                         'scale_factor': 1}],
             },

            # Best Epoch FloatTrain
            {'type': 'Epoch (FloatTrain, BestEpoch)', 'name': 'Epoch (FloatTrain, BestEpoch)',
             'description': 'Epochs (FloatTrain, BestEpoch)', 'unit': 'Epoch', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'FloatTrain.BestEpoch\s*: Best Epoch:\s+(?<eid>\d+)\s*',
                         'groupId': 'eid'}],
             },
            {'type': 'Validation Accuracy (FloatTrain, BestEpoch)', 'name': 'Accuracy (FloatTrain, BestEpoch)',
             'description': 'Accuracy (FloatTrain, BestEpoch)', 'unit': 'Accuracy Top-1%', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'FloatTrain.BestEpoch\s*: Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                         'groupId': 'accuracy',
                         'scale_factor': 1}],
             },
            {'type': 'F1-Score (FloatTrain, BestEpoch)', 'name': 'F1-Score (FloatTrain, BestEpoch)',
             'description': 'F1-Score (FloatTrain, BestEpoch)', 'unit': 'F1-Score', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'FloatTrain.BestEpoch\s*:\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                         'groupId': 'f1score',
                         'scale_factor': 1}],
             },

            # Best Epoch QuantTrain
            {'type': 'Epoch (QuantTrain, BestEpoch)', 'name': 'Epoch (QuantTrain, BestEpoch)',
             'description': 'Epochs (QuantTrain, BestEpoch)', 'unit': 'Epoch', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'QuantTrain.BestEpoch: Best Epoch:\s+(?<eid>\d+)\s*',
                         'groupId': 'eid'}],
             },
            {'type': 'Validation Accuracy (QuantTrain, BestEpoch)', 'name': 'Accuracy (QuantTrain, BestEpoch)',
             'description': 'Accuracy (QuantTrain, BestEpoch)', 'unit': 'Accuracy Top-1%', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'QuantTrain.BestEpoch: Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                         'groupId': 'accuracy',
                         'scale_factor': 1}],
             },
            {'type': 'F1-Score (QuantTrain, BestEpoch)', 'name': 'F1-Score (QuantTrain, BestEpoch)',
             'description': 'F1-Score (QuantTrain, BestEpoch)', 'unit': 'F1-Score', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'QuantTrain.BestEpoch:\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                         'groupId': 'f1score',
                         'scale_factor': 1}],
             },

            # Test data
            {'type': 'Test Accuracy (Test Data)', 'name': 'Accuracy (Test Data)',
             'description': 'Test Accuracy (Test Data)', 'unit': 'Accuracy Top-1%', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'test_data\s*:\s*Test Data Evaluation Accuracy:\s+(?<accuracy>[-+e\d+\.\d+]+)%',
                         'groupId': 'accuracy',
                         'scale_factor': 1}],
             },
            {'type': 'Confusion Matrix (Test Data)', 'name': 'Confusion Matrix',
             'description': 'Confusion Matrix (Test Data)', 'unit': 'Confusion Matrix', 'value': None,
             'regex': [{'op': 'search',
                         'pattern': r'test_data\s*:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\s\S]+?)INFO',
                         'groupId': 'cm',
                         'scale_factor': 1}],
             },
        ]
    }
def create_template_model_description(task_category, task_type, dataset_loader=None, batch_size_key=None):
    training_dict = dict(
        quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
        training_backend='tinyml_tinyverse',
        model_training_id='',
        model_name='',
        learning_rate=2e-3,
        model_spec=None,
        batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT.get(batch_size_key or task_type, 32),
        target_devices={
            constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=None),
            constants.TARGET_DEVICE_MSPM0G3519: dict(model_selection_factor=None),
            constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=None),
        },
        training_devices={
            constants.TRAINING_DEVICE_CPU: True,
            constants.TRAINING_DEVICE_CUDA: True,
            constants.TRAINING_DEVICE_MPS: True,
        },
    )

    if dataset_loader:
        training_dict['dataset_loader'] = dataset_loader

    return dict(
        common=dict(
            task_category=task_category,
            task_type=task_type,
            generic_model=True,
            model_details='',
        ),
        download=dict(download_url='', download_path=''),
        training=training_dict,
        compilation=dict()
    )


def get_model_descriptions_filtered(model_descriptions, enabled_models_list, task_type=None):
    return {k: v for k, v in model_descriptions.items() if k in enabled_models_list}


def get_model_description_by_name(model_descriptions, enabled_models_list, model_name):
    filtered = get_model_descriptions_filtered(model_descriptions, enabled_models_list)
    return filtered.get(model_name, None)
class BaseImageModelTraining:
    """
    Base class for image training modules.
    """

    train_module = None
    test_module = None

    @classmethod
    def init_params(cls, *args, **kwargs):
        params = dict(training=dict())
        params = utils.ConfigDict(params, *args, **kwargs)
        return params
    
    # @staticmethod
    # def _argv_value(value):
    #     if value is None:
    #         return "None"

        if isinstance(value, (list, tuple)):
            return repr(list(value))

        return str(value)

    def __init__(self, *args, quit_event=None, **kwargs):
        self.params = self.init_params(*args, **kwargs)
        self.quit_event = quit_event

        log_summary_regex = self._get_log_summary_regex()

        self.params.update(
            training=utils.ConfigDict(
                log_file_path=os.path.join(
                    self.params.training.train_output_path
                    if self.params.training.train_output_path
                    else self.params.training.training_path,
                    'run.log'
                ),
                log_summary_regex=log_summary_regex,
                summary_file_path=os.path.join(self.params.training.training_path, 'summary.yaml'),
                model_checkpoint_path=os.path.join(self.params.training.training_path, 'checkpoint.pth'),
                model_export_path=os.path.join(self.params.training.training_path, 'model.onnx'),
                model_proto_path=None,
                tspa_license_path=os.path.abspath(os.path.join(
                    os.path.dirname(tinyml_modelmaker.ai_modules.vision.training.tinyml_tinyverse.__file__),
                    'LICENSE.txt'
                )),
            )
        )

        self._init_task_specific_params()

        if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
            self.params.update(
                training=utils.ConfigDict(
                    model_checkpoint_path_quantization=os.path.join(
                        self.params.training.training_path_quantization, 'checkpoint.pth'),
                    model_export_path_quantization=os.path.join(
                        self.params.training.training_path_quantization, 'model.onnx'),
                )
            )

    def _get_log_summary_regex(self):
        return get_image_classification_log_summary_regex()

    def _init_task_specific_params(self):
        pass

    def _get_device(self):
        distributed = 1 if self.params.training.num_gpus > 1 else 0

        device = 'cpu'
        if self.params.training.num_gpus > 0:
            if torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cuda'

        return device, distributed

    def _build_common_train_argv(self, device, distributed):
        """
        Common image training args.
        """
        return [
            '--model', f'{self.params.training.model_training_id}',
            '--dual-op', f'{self.params.training.dual_op}',
            '--model-config', f'{self.params.training.model_config}',
            '--augment-config', f'{self.params.training.augment_config}',
            '--model-spec', f'{self.params.training.model_spec}',

            '--dataset', 'modelmaker',
            '--dataset-loader', f'{self.params.training.dataset_loader}',
            '--annotation-prefix', f'{self.params.dataset.annotation_prefix}',

            '--gpus', f'{self.params.training.num_gpus}',
            '--batch-size', f'{self.params.training.batch_size}',
            '--opt', f'{self.params.training.optimizer}',
            '--weight-decay', f'{self.params.training.weight_decay}',
            '--lr-scheduler', f'{self.params.training.lr_scheduler}',
            '--lr-warmup-epochs', '1',
            '--distributed', f'{distributed}',
            '--device', f'{device}',

            '--generic-model', f'{self.params.common.generic_model}',

            # Transform
            '--data-proc-transforms', self.params.data_processing_feature_extraction.data_proc_transforms,
            '--feat-ext-transform', self.params.data_processing_feature_extraction.feat_ext_transform,
            '--augmentation-transform', self.params.data_processing_feature_extraction.augmentation_transform,
            '--feat-ext-store-dir', f'{self.params.data_processing_feature_extraction.feat_ext_store_dir}',
            '--dont-train-just-feat-ext', f'{self.params.data_processing_feature_extraction.dont_train_just_feat_ext}',
            '--store-feat-ext-data', f'{self.params.data_processing_feature_extraction.store_feat_ext_data}',
            
            # Image preprocessing params.
            '--image-height', f'{self.params.data_processing_feature_extraction.image_height}',
            '--image-width', f'{self.params.data_processing_feature_extraction.image_width}',
            '--image-num-channel', f'{self.params.data_processing_feature_extraction.image_num_channel}',
            '--image-mean', f'{self.params.data_processing_feature_extraction.image_mean}',
            '--image-scale', f'{self.params.data_processing_feature_extraction.image_scale}',

            # # Optional image transform params.
            '--pad-value', f'{self.params.data_processing_feature_extraction.pad_value}',
            '--binary-threshold', f'{self.params.data_processing_feature_extraction.binary_threshold}',
            '--random-rotation-deg', f'{self.params.data_processing_feature_extraction.random_rotation_deg}',
            '--horizontal-flip-prob', f'{self.params.data_processing_feature_extraction.horizontal_flip_prob}',
            '--vertical-flip-prob', f'{self.params.data_processing_feature_extraction.vertical_flip_prob}',
            '--color-jitter-brightness', f'{self.params.data_processing_feature_extraction.color_jitter_brightness}',
            '--color-jitter-contrast', f'{self.params.data_processing_feature_extraction.color_jitter_contrast}',
            '--color-jitter-saturation', f'{self.params.data_processing_feature_extraction.color_jitter_saturation}',
            '--color-jitter-hue', f'{self.params.data_processing_feature_extraction.color_jitter_hue}',

            '--output-int', f'{self.params.training.output_int}',
            '--variables', f'{self.params.data_processing_feature_extraction.variables}',
            '--lis', f'{self.params.training.log_file_path}',
            '--ondevice-training', f'{self.params.training.ondevice_training}',
            '--data-path', os.path.join(self.params.dataset.dataset_path, self.params.dataset.data_dir),
            '--epochs', f'{self.params.training.training_epochs}',
            '--lr', f'{self.params.training.learning_rate}',
            '--output-dir', f'{self.params.training.training_path}',
        ]

    def _get_task_specific_train_argv(self):
        return []

    def _build_common_test_argv(self, device, data_path, model_path, output_dir):
        """
        Common image test args.
        """
        return [
            '--dataset', 'modelmaker',
            '--dataset-loader', f'{self.params.training.dataset_loader}',
            '--annotation-prefix', f'{self.params.dataset.annotation_prefix}',

            '--gpus', f'{self.params.training.num_gpus}',
            '--batch-size', f'{self.params.training.batch_size}',
            '--distributed', '0',
            '--device', f'{device}',

            '--variables', f'{self.params.data_processing_feature_extraction.variables}',

            '--feat-ext-transform', self.params.data_processing_feature_extraction.feat_ext_transform,
            '--data-proc-transforms', self.params.data_processing_feature_extraction.data_proc_transforms,
            '--augmentation-transform', self.params.data_processing_feature_extraction.augmentation_transform,
            '--image-height', f'{self.params.data_processing_feature_extraction.image_height}',
            '--image-width', f'{self.params.data_processing_feature_extraction.image_width}',
            '--image-num-channel', f'{self.params.data_processing_feature_extraction.image_num_channel}',
            '--image-mean', f'{self.params.data_processing_feature_extraction.image_mean}',
            '--image-scale', f'{self.params.data_processing_feature_extraction.image_scale}',

            '--pad-value', f'{self.params.data_processing_feature_extraction.pad_value}',
            '--binary-threshold', f'{self.params.data_processing_feature_extraction.binary_threshold}',
            '--random-rotation-deg', f'{self.params.data_processing_feature_extraction.random_rotation_deg}',
            '--horizontal-flip-prob', f'{self.params.data_processing_feature_extraction.horizontal_flip_prob}',
            '--vertical-flip-prob', f'{self.params.data_processing_feature_extraction.vertical_flip_prob}',
            '--color-jitter-brightness', f'{self.params.data_processing_feature_extraction.color_jitter_brightness}',
            '--color-jitter-contrast', f'{self.params.data_processing_feature_extraction.color_jitter_contrast}',
            '--color-jitter-saturation', f'{self.params.data_processing_feature_extraction.color_jitter_saturation}',
            '--color-jitter-hue', f'{self.params.data_processing_feature_extraction.color_jitter_hue}',

            '--nn-for-feature-extraction', f'{self.params.data_processing_feature_extraction.nn_for_feature_extraction}',
            '--output-int', f'{self.params.training.output_int}',

            '--lis', f'{self.params.training.log_file_path}',
            '--data-path', f'{data_path}',
            '--output-dir', output_dir,
            '--model-path', f'{model_path}',
            '--generic-model', f'{self.params.common.generic_model}',
        ]

    def _get_task_specific_test_argv(self):
        return []

    def _get_quant_train_epochs_divisor(self):
        return 5

    def _get_min_quant_epochs(self):
        return 10
    def clear(self):
        shutil.rmtree(self.params.training.training_path, ignore_errors=True)

    def run(self, **kwargs):
        os.makedirs(self.params.training.training_path, exist_ok=True)

        device, distributed = self._get_device()

        # Float training argv.
        argv = self._build_common_train_argv(device, distributed)
        argv.extend(self._get_task_specific_train_argv())

        args = self.train_module.get_args_parser().parse_args(argv)
        args.quit_event = self.quit_event

        if not utils.misc_utils.str2bool(self.params.testing.skip_train):
            if utils.misc_utils.str2bool(self.params.training.run_quant_train_only):
                if self.params.training.quantization == TinyMLQuantizationVersion.NO_QUANTIZATION:
                    raise ValueError(
                        f"quantization cannot be {TinyMLQuantizationVersion.NO_QUANTIZATION} "
                        f"if run_quant_train_only is chosen"
                    )

                quant_argv = deepcopy(argv)
                quant_argv = self._replace_arg(quant_argv, '--output-dir', self.params.training.training_path_quantization)
                quant_argv = self._replace_arg(quant_argv, '--epochs',
                                                str(max(self._get_min_quant_epochs(),
                                                        self.params.training.training_epochs // self._get_quant_train_epochs_divisor())))
                quant_argv = self._replace_arg(quant_argv, '--lr',
                                                str(float(self.params.training.learning_rate) / 100.0))
                quant_argv.extend([
                    '--quantization', f'{self.params.training.quantization}',
                    '--quantization-method', f'{self.params.training.quantization_method}',
                    '--weight-bitwidth', f'{self.params.training.quantization_weight_bitwidth}',
                    '--activation-bitwidth', f'{self.params.training.quantization_activation_bitwidth}',
                    '--lr-warmup-epochs', '0',
                    '--store-feat-ext-data', 'False',
                ])

                args = self.train_module.get_args_parser().parse_args(quant_argv)
                args.quit_event = self.quit_event
                self.train_module.run(args)

            else:
                self.train_module.run(args)

                if (utils.misc_utils.str2bool(self.params.data_processing_feature_extraction.store_feat_ext_data)
                        and utils.misc_utils.str2bool(self.params.data_processing_feature_extraction.dont_train_just_feat_ext)):
                    return self.params

                if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
                    quant_argv = deepcopy(argv)
                    quant_argv = self._replace_arg(quant_argv, '--output-dir',
                                                    self.params.training.training_path_quantization)
                    quant_argv = self._replace_arg(quant_argv, '--epochs',
                                                    str(max(self._get_min_quant_epochs(),
                                                            self.params.training.training_epochs // self._get_quant_train_epochs_divisor())))
                    quant_argv = self._replace_arg(quant_argv, '--lr',
                                                    str(float(self.params.training.learning_rate) / 100.0))
                    quant_argv.extend([
                        '--weights', f'{self.params.training.model_checkpoint_path}',
                        '--quantization', f'{self.params.training.quantization}',
                        '--quantization-method', f'{self.params.training.quantization_method}',
                        '--weight-bitwidth', f'{self.params.training.quantization_weight_bitwidth}',
                        '--activation-bitwidth', f'{self.params.training.quantization_activation_bitwidth}',
                        '--lr-warmup-epochs', '0',
                        '--store-feat-ext-data', 'False',
                    ])

                    args = self.train_module.get_args_parser().parse_args(quant_argv)
                    args.quit_event = self.quit_event
                    self.train_module.run(args)

        if utils.misc_utils.str2bool(self.params.testing.enable):
            
            if self.params.testing.test_data and os.path.exists(self.params.testing.test_data):
                data_path = self.params.testing.test_data
            else:
                data_path = os.path.join(
                    self.params.dataset.dataset_path,
                    self.params.dataset.data_dir
                )

            if self.params.testing.model_path and os.path.exists(self.params.testing.model_path):
                model_path = self.params.testing.model_path
                output_dir = os.path.dirname(model_path)
            else:
                if self.params.training.quantization == TinyMLQuantizationVersion.NO_QUANTIZATION:
                    model_path = os.path.join(
                        self.params.training.training_path,
                        'model.onnx'
                    )
                    output_dir = self.params.training.training_path
                else:
                    model_path = os.path.join(
                        self.params.training.training_path_quantization,
                        'model.onnx'
                    )
                    output_dir = self.params.training.training_path_quantization

            test_argv = self._build_common_test_argv(
                device,
                data_path,
                model_path,
                output_dir
            )

            test_argv.extend(self._get_task_specific_test_argv())

            args = self.test_module.get_args_parser().parse_args(test_argv)
            args.quit_event = self.quit_event

            self.test_module.run(args)

        return self.params

    @staticmethod
    def _replace_arg(argv, key, value):
        argv = list(argv)
        if key not in argv:
            argv.extend([key, str(value)])
            return argv

        idx = argv.index(key)
        if idx + 1 >= len(argv):
            raise ValueError(f"Argument {key} exists but has no value")
        argv[idx + 1] = str(value)
        return argv

    def stop(self):
        if self.quit_event is not None:
            self.quit_event.set()
            return True
        return False

    def get_params(self):
        return self.params

