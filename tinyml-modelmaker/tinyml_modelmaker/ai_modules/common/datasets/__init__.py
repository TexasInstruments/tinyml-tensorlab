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
from os.path import join as opj
import logging
# import PIL
import sys
import warnings
from glob import glob
import shutil

from .... import utils
from . import dataset_utils


def get_datasets_list(task_type=None):
    if task_type == 'timeseries_classification':
        return ['arc_fault_example_dsi', 'arc_fault_example_dsk']  # ['oxford_flowers102']
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
                # self.file_list = dataset_utils.create_filelist(self.params.dataset.dataset_path, self.params.common.project_run_path, ignore_str='_list.txt')
                annotations_dir = opj(self.params.dataset.dataset_path,  self.params.dataset.annotation_dir)
                utils.misc_utils.remove_if_exists(annotations_dir)
                os.makedirs(annotations_dir, exist_ok=True)

                for split_name in self.params.dataset.split_names:
                    split_list_file = opj(self.params.dataset.dataset_path, self.params.dataset.annotation_dir,
                                                   f'{self.params.dataset.annotation_prefix}_{split_name}_list.txt')
                    self.params.dataset.annotation_path_splits.append(split_list_file)

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
