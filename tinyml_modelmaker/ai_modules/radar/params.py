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

from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion, TinyMLQuantizationMethod

from ... import utils
from . import constants


def init_params(*args, **kwargs):
    default_params = dict(
        common=dict(
            download_path=os.path.join('.', 'data', 'downloads'),
            projects_path=os.path.join('.', 'data', 'projects'),
            project_path=None,
            project_run_path=None,
            task_type=None,
            task_category='radar_classification',
            target_device=None,
            target_module='radar',
            # run_name can be any string, but there are some special cases:
            # {date-time} will be replaced with datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # {model_name} will be replaced with the name of the model
            run_name=os.path.join('{date-time}','{model_name}'),
            generic_model=False,
        ),
        download=None,
        dataset=dict(
            enable=True,
            dataset_name=None,
            dataset_path=None,  # dataset split will be created here
            extract_path=None,
            split_factor=(0.7, 0.2, 0.1),
            split_names=('train', 'val', 'test'),
            max_num_files=10000,
            input_data_path=None,  # url/zip folder
            input_annotation_path=None,  # annotation file
            data_path_splits=None,
            data_dir=None,  # None = auto-detect based on task_category - classes (mostly)
            annotation_path_splits=None,
            annotation_dir='annotations',
            annotation_prefix='instances',  # change this if your dataset has a different annotation prefix
            annotation_format='univ_ts_json',
            split_type='amongst_files',
        ),
        training=dict(
            enable=True,
            model_name=None,
            augment_config=None,
            model_config=None,
            model_spec=None,
            dataset_loader='GenericRadarDataset',
            model_training_id=None,
            training_backend=None,
            pretrained_checkpoint_path=None,
            target_devices={},
            project_path=None,
            dataset_path=None,
            training_path=None,
            training_path_quantization=None,
            log_file_path=None,
            file_level_classification_log_path=None,
            log_summary_regex=None,
            training_epochs=100,
            early_stopping=False,
            early_stopping_patience=5,
            early_stopping_min_delta=0.0,
            batch_size=64,
            learning_rate=1e-4,
            lambda_reg=0,
            optimizer='sgd',
            weight_decay=0,
            lr_scheduler=None,
            momentum=0,
            training_device=constants.TRAINING_DEVICE_CPU,  # 'cpu', 'cuda'
            num_gpus=0,  # 0,1
            distributed=True,
            train_output_path=None,
            run_quant_train_only=False,
            label_smoothing=0,
            # out_dir=os.getcwd())
            auto_quantization = False,
            quantization=TinyMLQuantizationVersion.NO_QUANTIZATION,
            quantization_method=TinyMLQuantizationMethod.QAT,
            quantization_weight_bitwidth=8,
            quantization_activation_bitwidth=8,
            output_int=False,  # Dynamic default based on task_category and quantization
            dual_op=False,
            properties=[
                dict(type="group", dynamic=True, name="preprocessing_group", label="Preprocessing Parameters", default=[]),
                dict(type="group", dynamic=True, name="train_group", label="Training Parameters", default=[]),
            ],
            
            #######################################
            # nas params
            #######################################
            nas_enabled=False,
            nas_optimization_mode='Memory', # 'Compute' 'Memory'
            nas_epochs=10,   # num epochs for which search to be executed

            # Preset mode
            nas_model_size=None,  # 's' 'm' 'l' 'xl'

            # Customization mode
            nas_nodes_per_layer=4,  # Each layer consists of few nodes for DAG construction
            nas_layers=3,    # Minimum should be 3
            nas_init_channels=1,    # Initial feature map channel for conv layer
            nas_init_channel_multiplier=3,  # First layer channel multiplier
            nas_fanout_concat=4,   # num nodes_per_layer to concat for output of current layer, it should always be less than equal to nodes_per_layer
            
            load_saved_model=None,
            ondevice_training = False,
            trainable_layers_from_last = 1,
        ),
        testing=dict(
            enable=True,
            skip_train=False,
            device_inference=False,
            # test_quant_model=False,
            # quant_model_path=None,
            test_data=None,
            model_path=None,
        ),
        data_processing_feature_extraction=dict(
            sampling_rate=1,
            new_sr=None,
            frame_size=1,
            stride_size=0.125,
            variables=1, #Not considered separately in radar datasets
            target_variables=[],

            feat_ext_transform=[],  # Getting used for custom presets
            data_proc_transforms=[], # Array consisting 5 stage Radar Data Processing Pipeline 
            feature_size_per_frame=None,
            num_frame_concat=None,

            store_feat_ext_data=False,
            feat_ext_store_dir=None,
            dont_train_just_feat_ext=False,

            offset=0,
            scale=None,
            nn_for_feature_extraction=False,
            gof_test=False,

            # [min_bin, max_bin] array index range to keep; None = keep all 'bin*' columns.
            radar_range_bins_bounds=None,
            radar_point_cloud_data_selector=None,  # sub-array ['x','y','z','doppler','snr']; 
            radar_point_cloud_coordinate_system='absolute',  # 'absolute' or 'relative'
            radar_point_cloud_relative_coordinate_filter=None,  # sub-array ['x','y','z']; 
            # Array [min_x, max_x, min_y, max_y, min_z, max_z].
            # None = no spatial filtering (equivalent to leaving every bound at its
            # sentinel of +-1e9/None).
            radar_point_cloud_spatial_filter_bounds=None,
            # Array [min_doppler, max_doppler, min_snr, max_snr]. None = no filtering.
            radar_point_cloud_non_spatial_filter_bounds=None,

            radar_track_data_selector=None,  # from array ['velocity','x','y','z','acceleration']; 
            # array [min_velocity, max_velocity] applied to all track_velocity_* columns. None = no filtering.
            radar_track_velocity_bounds=None,
            # array [min_acceleration, max_acceleration] applied to all track_acc_* columns. None = no filtering.
            radar_track_acceleration_bounds=None,
            # array [min_x, max_x, min_y, max_y, min_z, max_z]
            # applied to track position columns. None = no filtering.
            radar_track_spatial_filter_bounds=None,

            # Based on Applications in future : Can be made dynamic
            radar_data_scaling_track_velocity=None,

            radar_max_points_per_frame=100,
            radar_min_points_per_frame=0,
            radar_min_points_per_window=12, 
            radar_point_cloud_structure_strategy='highest and lowest',
            radar_point_cloud_sort_by=None,  # which selected point feature to sort/rank points by (x, y, z)
            radar_frames_quality_index=None,  # None, 'per_radar_frame', or 'per_window'
            radar_feature_ordering='frame_major' #'frame_major'/'feature_major'
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
                    {"value": constants.COMPILATION_DEFAULT, "label": "Default best", "tooltip": "Default Inference Mode"},
                ]
            )
            ],
        ),
    )

    params = utils.ConfigDict(default_params, *args, **kwargs)
    return params