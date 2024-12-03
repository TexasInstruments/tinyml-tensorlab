#################################################################################
# Copyright (c) 2018-2022, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

import os

from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

from ... import utils
from . import constants


def init_params(*args, **kwargs):
    default_params = dict(
        common=dict(
            verbose_mode=True,
            download_path=os.path.join('.', 'data', 'downloads'),
            projects_path=os.path.join('.', 'data', 'projects'),
            project_path=None,
            project_run_path=None,
            task_type=None,
            task_category=None,
            target_machine='evm',
            target_device=None,
            # run_name can be any string, but there are some special cases:
            # {date-time} will be replaced with datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # {model_name} will be replaced with the name of the model
            run_name=os.path.join('{date-time}, {model_name}'),
            generic_model=False,
        ),
        download=None,
        dataset=dict(
            enable=True,
            dataset_name=None,
            dataset_path=None,  # dataset split will be created here
            extract_path=None,
            split_factor=(0.6, 0.3, 0.1),
            split_names=('train', 'val', 'test'),
            max_num_files=10000,
            input_data_path=None,  # input images
            input_annotation_path=None,  # annotation file
            data_path_splits=None,
            data_dir='classes',
            annotation_path_splits=None,
            annotation_dir='annotations',
            annotation_prefix='instances',  # change this if your dataset has a different annotation prefix
            annotation_format='univ_ts_json',
            dataset_download=False,
            dataset_reload=False,
        ),
        training=dict(
            enable=True,
            model_name=None,
            model_config=None,
            model_spec=None,
            dataset_loader='GenericTSDataset',
            model_training_id=None,
            training_backend=None,
            pretrained_checkpoint_path=None,
            target_devices={},
            project_path=None,
            dataset_path=None,
            training_path=None,
            training_path_quantization=None,
            log_file_path=None,
            log_summary_regex=None,
            training_epochs=10,
            warmup_epochs=1,
            num_last_epochs=5,
            batch_size=8,
            learning_rate=2e-3,
            optimizer='sgd',
            weight_decay=1e-4,
            lr_scheduler='cosineannealinglr',
            training_device='cuda',  # 'cpu', 'cuda'
            num_gpus=1,  # 0,1
            distributed=True,
            training_master_port=29500,
            train_output_path=None,
            run_quant_train_only=False,
            # out_dir=os.getcwd())
            quantization=TinyMLQuantizationVersion.NO_QUANTIZATION,
            with_input_batchnorm=False,
            dual_op=False,
            properties=[
                dict(type="group", dynamic=True, name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                dict(type="group", dynamic=False, name="train_group", label="Training Parameters", default=["training_epochs", "learning_rate"]),
                dict(label="Epochs", name="training_epochs", type="integer", default=50, min=1, max=300),
                dict(label="Learning Rate", name="learning_rate", type="float", default=0.04, min=0.001, max=0.1,
                     decimal_places=3, increment=0.001),
                ]),

        testing=dict(
            enable=True,
            skip_train=False,
            # test_quant_model=False,
            # quant_model_path=None,
            test_data=None,
            model_path=None,
        ),
        data_processing=dict(
            sampling_rate=313000,
            new_sr=3009,
            sequence_window=3009,
            stride_size=0.01,
            transforms=[],  # 'DownSample SimpleWindow',
            variables=1,
            resampling_factor=1,
        ),
        feature_extraction=dict(
            feature_extraction_name=None,
            transform=[],  # 'FFT512 FFT256',
            frame_size=None,
            feature_size_per_frame=None,
            num_frame_concat=None,
            frame_skip=None,
            min_bin=None,
            normalize_bin=0,
            store_feat_ext_data=False,
            feat_ext_store_dir=None,
            dont_train_just_feat_ext=False,
            dc_remove=True,
            analysis_bandwidth=1,
            log_base=None,
            log_mul=None,
            log_threshold=None,
            stacking=None,
            offset=None,
            scale=None,
        ),
        compilation=dict(
            enable=True,
            model_path = None,
            compile_output_path=None,
            compile_preset_name=None,
            keep_libc_files=False,
            properties=[dict(
                label="Compilation Preset", name="compile_preset_name", type="enum",
                default=constants.COMPILATION_DEFAULT,
                enum=[
                    # {"value": constants.COMPILATION_FORCED_SOFT_NPU, "label": "Forced Software NPU", "tooltip": "Only for F28P55, to disable HW NPU"},
                      {"value": constants.COMPILATION_DEFAULT, "label": "Default best", "tooltip": "Default Inference Mode"},
                      ])
            ],
        ),
    )

    params = utils.ConfigDict(default_params, *args, **kwargs)
    return params
