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
from os.path import join as opj
import logging
import sys
import warnings
from glob import glob
import shutil

from .... import utils
from . import dataset_utils
from ...timeseries.constants import TASK_CATEGORY_TS_ANOMALYDETECTION


def get_datasets_list(task_type=None):
    if task_type == 'timeseries_classification':
        return ['arc_fault_example_dsi', 'dc_arc_fault_example_dsk']  # ['oxford_flowers102']
    elif task_type == 'audio_classification':
        return ['SpeechCommands']  # ['oxford_flowers102']
    else:
        assert False, 'unknown task type for get_datasets_list'


def get_target_module(backend_name):
    this_module = sys.modules[__name__]
    target_module = getattr(this_module, backend_name)
    return target_module


class DatasetHandling:
    @classmethod
    def init_params(self, *args, **kwargs):
        params = dict(
            dataset=dict(
            )
        )
        params = utils.ConfigDict(params, *args, **kwargs)
        return params

    def __init__(self, *args, quit_event=None, **kwargs):
        self.params = self.init_params(*args, **kwargs)
        self.quit_event = quit_event
        self.params.dataset.data_path_splits = []
        self.params.dataset.annotation_path_splits = []
        self.logger = logging.getLogger("root.DatasetHandling")
        '''
        for split_idx, split_name in enumerate(self.params.dataset.split_names):
            self.params.dataset.data_path_splits.append(opj(self.params.dataset.dataset_path, split_name))
            self.params.dataset.annotation_path_splits.append(opj(self.params.dataset.dataset_path, self.params.dataset.annotation_dir,
                             f'{self.params.dataset.annotation_prefix}_{split_name}_list.txt'))
        '''
        #

    def clear(self):
        pass

    def run(self):
        # max_num_files = self.get_max_num_files()
        # dataset reading/splitting
        need_to_create_splits = False
        # self.out_files = []
        if self.params.dataset.input_data_path and os.path.exists(self.params.dataset.input_data_path):
            if not os.path.isdir(self.params.dataset.input_data_path):
                extract_root = os.path.dirname(self.params.dataset.input_data_path)
                extract_success = utils.extract_files(self.params.dataset.input_data_path, extract_root)
                if not extract_success:
                    raise "Dataset could not be extracted"
                self.params.dataset.input_data_path = os.path.dirname(self.params.dataset.input_data_path)

            for split_name in self.params.dataset.split_names:
                self.params.dataset.data_path_splits.append(opj(self.params.dataset.dataset_path, split_name))
                # If training_list.txt and validation_list.txt files are already present, then

                if os.path.exists(opj(self.params.dataset.input_data_path, split_name + '_list.txt')):
                    # Everything is as per expectations
                    split_list_file = opj(self.params.dataset.input_data_path, split_name + '_list.txt')
                    # assert False, 'File needs to be present: {}'.format(split_list_file)
                elif os.path.exists(opj(self.params.dataset.input_data_path, self.params.dataset.annotation_dir, split_name + '_list.txt')):
                    split_list_file = opj(self.params.dataset.input_data_path, self.params.dataset.annotation_dir, split_name + '_list.txt')
                elif os.path.exists(opj(self.params.dataset.input_data_path, self.params.dataset.annotation_dir, f'{self.params.dataset.annotation_prefix}_{split_name}_list.txt')):
                    split_list_file = opj(self.params.dataset.input_data_path, self.params.dataset.annotation_dir, f'{self.params.dataset.annotation_prefix}_{split_name}_list.txt')
                else:
                    need_to_create_splits = True
                    self.logger.info(f'Fresh splits will be created as {split_name} split list file is not present')

            os.makedirs(self.params.dataset.dataset_path, exist_ok=True)
            for directory in glob(opj(self.params.dataset.input_data_path, '*')):
                utils.misc_utils.make_symlink(os.path.abspath(directory), opj(self.params.dataset.dataset_path, os.path.basename(directory)))  # self.params.dataset.data_dir
            if need_to_create_splits:
                annotations_dir = opj(self.params.dataset.dataset_path,  self.params.dataset.annotation_dir)
                utils.misc_utils.remove_if_exists(annotations_dir)
                utils.misc_utils.remove_if_exists(opj(self.params.dataset.dataset_path, self.params.dataset.data_dir + '_edited'))
                utils.misc_utils.remove_if_exists(opj(self.params.dataset.dataset_path, self.params.dataset.data_dir + '_original'))
                os.makedirs(annotations_dir, exist_ok=True)
                
                for split_name in self.params.dataset.split_names:
                    split_list_file = opj(self.params.dataset.dataset_path, self.params.dataset.annotation_dir,f'{self.params.dataset.annotation_prefix}_{split_name}_list.txt')
                    self.params.dataset.annotation_path_splits.append(split_list_file)

                #We need to handle the datasplitting differently for anomaly detection compared to other categerios 
                #In Anomaly detection, we don't use anomaly data to train the model, so we need to add all anomaly data to test list
                #Splitting will be applied only on Normal data and then model will be trained on normal training data only
                if self.params.common.task_category == TASK_CATEGORY_TS_ANOMALYDETECTION:
                    self.normal_files_path =  opj(self.params.dataset.dataset_path, self.params.dataset.data_dir, "Normal")
                    self.anomaly_files_path = opj(self.params.dataset.dataset_path, self.params.dataset.data_dir, "Anomaly")
                    normal_file_list = dataset_utils.get_file_list(self.normal_files_path)
                    anomaly_file_list = dataset_utils.get_file_list(self.anomaly_files_path)
                    normal_file_list = [os.path.join(os.path.basename(os.path.dirname(file)), os.path.basename(file)) for file in normal_file_list]
                    anomaly_file_list = [os.path.join(os.path.basename(os.path.dirname(file)), os.path.basename(file)) for file in anomaly_file_list]
                    
                    #Store the file paths in txt files for processing purpose
                    normal_paths_file = os.path.join(annotations_dir, 'normal_list.txt')
                    anomaly_paths_file = os.path.join(annotations_dir, 'anomlay_list.txt')
                    with open(normal_paths_file, 'w') as file:
                        file.write('\n'.join(normal_file_list))
                    with open(anomaly_paths_file, 'w') as file:
                        file.write('\n'.join(anomaly_file_list))
                    
                    if self.params.dataset.split_type == 'amongst_files':
                        dataset_utils.create_inter_file_split(normal_paths_file, self.params.dataset.annotation_path_splits, self.params.dataset.split_factor, shuffle_items=True, random_seed=42)
                    elif self.params.dataset.split_type == 'within_files':
                        
                        out_dir = opj(self.params.dataset.dataset_path, self.params.dataset.data_dir + '_edited')
                        os.makedirs(out_dir, exist_ok=True)
                        dataset_utils.create_intra_file_split(normal_paths_file, self.params.dataset.annotation_path_splits, self.params.dataset.split_factor,
                                                            self.params.dataset.data_dir, out_dir, self.params.dataset.split_names, shuffle_items=True, random_seed=42)
                        shutil.move(opj(self.params.dataset.dataset_path, self.params.dataset.data_dir), opj(self.params.dataset.dataset_path, self.params.dataset.data_dir + '_original'))
                        shutil.move(out_dir, opj(self.params.dataset.dataset_path, self.params.dataset.data_dir))
                        shutil.move(opj(self.params.dataset.dataset_path, self.params.dataset.data_dir + '_original',"Anomaly"), opj(self.params.dataset.dataset_path, self.params.dataset.data_dir))
                        
                    #add all the anomaly file paths to test list as we will use the anomaly data during test only. 
                    annotation_test_list_path =  opj(self.params.dataset.dataset_path, self.params.dataset.annotation_dir,f'{self.params.dataset.annotation_prefix}_test_list.txt')
                    with open(anomaly_paths_file, 'r') as source:
                            content = source.read()
                    with open(annotation_test_list_path, 'a') as destination:
                        destination.write('\n'+content)
                    os.remove(normal_paths_file)
                    os.remove(anomaly_paths_file)
                    
                else:
                    self.file_list = dataset_utils.create_filelist(
                        opj(self.params.dataset.dataset_path, self.params.dataset.data_dir), annotations_dir,
                        ignore_str_list=['_list.txt', '.md', 'LICENSE', '.DS_Store', '_background_noise_'])
                    self.logger.info(f'File list is written to: {self.file_list}')
                    if self.params.dataset.split_type == 'amongst_files':
                        dataset_utils.create_inter_file_split(self.file_list, self.params.dataset.annotation_path_splits, self.params.dataset.split_factor, shuffle_items=True, random_seed=42)
                    elif self.params.dataset.split_type == 'within_files':
                        out_dir = opj(self.params.dataset.dataset_path, self.params.dataset.data_dir + '_edited')
                        os.makedirs(out_dir, exist_ok=True)
                        dataset_utils.create_intra_file_split(self.file_list, self.params.dataset.annotation_path_splits, self.params.dataset.split_factor,
                                                            self.params.dataset.data_dir, out_dir, self.params.dataset.split_names, shuffle_items=True, random_seed=42)
                        shutil.move(opj(self.params.dataset.dataset_path, self.params.dataset.data_dir), opj(self.params.dataset.dataset_path, self.params.dataset.data_dir + '_original'))
                        shutil.move(out_dir, opj(self.params.dataset.dataset_path, self.params.dataset.data_dir))

                        # self.out_files = dataset_utils.create_simple_split(self.file_list, self.params.common.project_run_path + '/dataset', self.params.dataset.split_names, self.params.dataset.split_factor, shuffle_items=True, random_seed=42)
                    self.logger.info('Splits of the dataset can be found at: {}'.format(self.params.dataset.annotation_path_splits))
        else:
            assert False, f'invalid dataset provided at {self.params.dataset.input_data_path}'

    def get_max_num_files(self):
        if isinstance(self.params.dataset.max_num_files, (list, tuple)):
            max_num_files = self.params.dataset.max_num_files
        elif isinstance(self.params.dataset.max_num_files, int):
            assert (0.0 < self.params.dataset.split_factor < 1.0), 'split_factor must be between 0 and 1.0'
            assert len(self.params.dataset.split_names) > 1, 'split_names must have at least two entries'
            max_num_files = [None] * len(self.params.dataset.split_names)
            for split_id, split_name in enumerate(self.params.dataset.split_names):
                if split_id == 0:
                    max_num_files[split_id] = round(self.params.dataset.max_num_files * self.params.dataset.split_factor)
                else:
                    max_num_files[split_id] = self.params.dataset.max_num_files - max_num_files[0]
                #
            #
        else:
            warnings.warn('unrecognized value for max_num_files - must be int, list or tuple')
            assert len(self.params.dataset.split_names) > 1, 'split_names must have at least two entries'
            max_num_files = [None] * len(self.params.dataset.split_names)
        #
        return max_num_files

    def get_params(self):
        return self.params
