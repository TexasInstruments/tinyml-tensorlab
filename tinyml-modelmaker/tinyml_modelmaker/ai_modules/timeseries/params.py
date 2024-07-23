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

import datetime
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

from ... import utils
from . import constants


def init_params(*args, **kwargs):
    default_params = dict(
        common=dict(
            verbose_mode=True,
            download_path='./data/downloads',
            projects_path='./data/projects',
            project_path=None,
            project_run_path=None,
            task_type=None,
            task_category=None,
            target_machine='evm',
            target_device=None,
            # run_name can be any string, but there are some special cases:
            # {date-time} will be replaced with datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # {model_name} will be replaced with the name of the model
            run_name='{date-time}/{model_name}',
        ),
        download=None,
        dataset=dict(
            enable=True,
            dataset_name=None,
            dataset_path=None,  # dataset split will be created here
            extract_path=None,
            split_factor=(0.6, 0.3),
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
            dataset_reload=False
        ),
        training=dict(
            enable=True,
            model_name=None,
            model_config=None,
            model_spec=None,
            dataset_loader='SimpleTSDataset',
            model_training_id=None,
            training_backend=None,
            pretrained_checkpoint_path=None,
            target_devices={},
            generic_model=False,
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
            weight_decay=1e-4,
            training_device='cuda',  # 'cpu', 'cuda'
            num_gpus=1,  # 0,1
            distributed=True,
            training_master_port=29500,
            train_output_path=None,
            run_quant_train_only=False,
            # out_dir=os.getcwd())
            quantization=TinyMLQuantizationVersion.NO_QUANTIZATION,
            with_input_batchnorm=False,
            properties=[
                # dict(type="group", dynamic=False, name="train", label="Train", default=["training_epochs", "learning_rate"]),
                # dict(type="group", dynamic=True, name="preprocessing", label="Preprocessing",),
                dict(label="Epochs", name="training_epochs", type="integer", default=50, min=1, max=300),
                dict(label="Learning Rate", name="learning_rate", type="float", default=0.04, min=0.001, max=0.1,
                     decimal_places=3, increment=0.001),
                # dict(label="Resampling Factor", name="resampling_factor", type="integer", default=15, min=1, max=100),

                dict(label="Feature Extraction Name", name="feature_extraction_name", type="enum",
                     default='FFT1024Input_256Feature_1Frame_Full_Bandwidth', enum=[
                        {"value": "FFT1024Input_256Feature_1Frame_Full_Bandwidth", "label": "FFT1024Input_256Feature_1Frame_Full_Bandwidth", "tooltip": "Windowing and FFT"},
                        {"value": "MotorFault_256Input_FFT_16Feature_8Frame_3InputChannel_removeDC_1D",
                         "label": "MotorFault_256Input_FFT_16Feature_8Frame_3InputChannel_removeDC_1D",
                         "tooltip": "Windowing and FFT"},
                        {"value": "MotorFault_256Input_FFT_16Feature_8Frame_3InputChannel_removeDC_2D1",
                         "label": "MotorFault_256Input_FFT_16Feature_8Frame_3InputChannel_removeDC_2D1",
                         "tooltip": "Windowing and FFT"},
                        # {"value": "MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_1D",
                        #  "label": "MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_1D",
                        #  "tooltip": "Windowing and FFT"},
                        {"value": "MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_2D1",
                         "label": "MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_2D1",
                         "tooltip": "Windowing and FFT"},

                    ]),

                # dict(label="Number of Input Channels", name="variables", type="integer", default=1, min=1, max=100),

                dict(label="Feature Size per frame", name="feature_size_per_frame", type="enum", default=256, enum=[
                    {"value": 16, "label": "16 features extracted per frame", "tooltip": "Extract 16 features per frame"},
                    {"value": 32, "label": "32 features extracted per frame", "tooltip": "Extract 32 features per frame"},
                    {"value": 64, "label": "64 features extracted per frame", "tooltip": "Extract 64 features per frame"},
                    {"value": 128, "label": "128 features extracted per frame", "tooltip": "Extract 128 features per frame"},
                    {"value": 256, "label": "256 features extracted per frame", "tooltip": "Extract 256 features per frame"},
                    {"value": 512, "label": "512 features extracted per frame", "tooltip": "Extract 512 features per frame"}]),

                dict(label="Frames to Concatenate", name="num_frame_concat", type="enum", default=1,
                     enum=[{"value": 1, "label": "1 frame", "tooltip": "1 frame concatenated for feature extraction"},
                           {"value": 4, "label": "4 frames", "tooltip": "4 frames concatenated for feature extraction"},
                           {"value": 8, "label": "8 frames", "tooltip": "8 frames concatenated for feature extraction"},
                           {"value": 16, "label": "16 frames",
                            "tooltip": "16 frames concatenated for feature extraction"}, ]),

                dict(label="Minimum FFT Bin Number", name="min_fft_bin", type="integer", default=1, min=0, max=256),
                dict(label="FFT Bins used per Feature", name="fft_bin_size", type="integer", default=1, min=1, max=8),

                # dict(label="Directly run Quantized Training", name="run_quant_train_only", type="enum", default='False',
                #      enum=[{"value": "True", "label": "True", "tooltip": "Quant Training Only"},
                #            {"value": "False", "label": "False", "tooltip": "Float + Quant Training"}]),
            ]
        ),
        testing=dict(
            enable=True,
            skip_train=False,
            # test_quant_model=False,
            # quant_model_path=None,
            test_data=None,
            model_path=None,
        ),
        data_processing=dict(
            org_sr=313000,
            new_sr=3009,
            stride_window=0.001,
            sequence_window=0.25,
            transforms=[],  # 'DownSample SimpleWindow',
            variables=1,
            resampling_factor=1,
        ),
        feature_extraction=dict(
            feature_extraction_name=None,
            transform='',  # 'FFT512 FFT256',
            frame_size=1024,
            feature_size_per_frame=512,
            num_frame_concat=1,
            frame_skip=1,
            min_fft_bin=1,
            fft_bin_size=2,
            store_feat_ext_data=False,
            feat_ext_store_dir=None,
            dont_train_just_feat_ext=False,
            dc_remove=True,
            stacking=None,
            offset=0,
            scale=1,
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
