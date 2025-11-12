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

import copy
import datetime
import os

# import tarfile
from zipfile import ZipFile

# import torch
import yaml

from ... import utils
from . import constants, datasets, descriptions
from .params import init_params
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion



class ModelRunner():
    @classmethod
    def init_params(self, *args, **kwargs):
        params = init_params(*args, **kwargs)
        # set the checkpoint download folder
        # (for the models that are downloaded using torch.hub eg. mmdetection uses that)
        # torch.hub.set_dir(os.path.join(params.common.download_path, 'pretrained', 'torch', 'hub'))
        return params

    def __init__(self, *args, verbose=True, **kwargs):
        self.params = self.init_params(*args, **kwargs)

        # print the runner params
        if verbose:
            [print(key, ':', value) for key, value in vars(self.params).items()]
        #
        # normalize the paths
        if not self.params.dataset.dataset_name:
            self.params.dataset.dataset_name = os.path.splitext(os.path.basename(self.params.dataset.input_data_path))[0]
        self.params.dataset.input_data_path = utils.absolute_path(self.params.dataset.input_data_path)
        self.params.dataset.input_annotation_path = utils.absolute_path(self.params.dataset.input_annotation_path)

        self.params.common.run_name = self.resolve_run_name(self.params.common.run_name, self.params.training.model_name)
        self.params.dataset.extract_path = self.params.dataset.dataset_path

        if self.params.training.train_output_path:
            self.params.common.projects_path = utils.absolute_path(self.params.training.train_output_path)
            self.params.common.project_path = os.path.join(self.params.common.projects_path)# , self.params.dataset.dataset_name)
            self.params.dataset.dataset_path = os.path.join(self.params.common.project_path, 'dataset')
            self.params.common.project_run_path = self.params.common.projects_path
            self.params.training.training_path = utils.absolute_path(os.path.join(self.params.training.train_output_path, 'training_base'))
            if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
                self.params.training.training_path_quantization = utils.absolute_path(os.path.join(self.params.training.train_output_path, 'training_quantization'))
            self.params.training.model_packaged_path = os.path.join(self.params.training.train_output_path,
                                    '_'.join(os.path.split(self.params.common.run_name))+'.zip')
        else:
            self.params.common.projects_path = utils.absolute_path(self.params.common.projects_path)
            self.params.common.project_path = os.path.join(self.params.common.projects_path, self.params.dataset.dataset_name)
            self.params.common.project_run_path = os.path.join(self.params.common.project_path, 'run', self.params.common.run_name)
            self.params.dataset.dataset_path = os.path.join(self.params.common.project_path, 'dataset')
            self.params.training.training_path = utils.absolute_path(os.path.join(self.params.common.project_run_path, 'training', 'base'))
            if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
                self.params.training.training_path_quantization = utils.absolute_path(os.path.join(self.params.common.project_run_path, 'training', 'quantization'))
            self.params.training.model_packaged_path = os.path.join(self.params.training.training_path,
                                    '_'.join(os.path.split(self.params.common.run_name))+'.zip')

        assert self.params.common.target_device in constants.TARGET_DEVICES_ALL, f'common.target_device must be set to one of: {constants.TARGET_DEVICES_ALL}'
        # target_device_compilation_folder = self.params.common.target_device

        if self.params.compilation.compile_output_path:
            if self.params.training.enable == False and self.params.compilation.enable == True:
                self.params.common.projects_path = utils.absolute_path(self.params.compilation.compile_output_path)
                self.params.common.project_run_path = self.params.common.projects_path
            self.params.compilation.compilation_path = utils.absolute_path(self.params.compilation.compile_output_path)
            self.params.compilation.model_packaged_path = os.path.join(self.params.compilation.compile_output_path,
                                                                    '_'.join(os.path.split(
                                                                        self.params.common.run_name)) + f'_{self.params.common.target_device}.zip')
        else:
            # self.params.compilation.compilation_path = utils.absolute_path(os.path.join(self.params.common.project_run_path, 'compilation', target_device_compilation_folder))
            self.params.compilation.compilation_path = utils.absolute_path(os.path.join(self.params.common.project_run_path, 'compilation'))
            self.params.compilation.model_packaged_path = os.path.join(self.params.compilation.compilation_path,
                                                                    '_'.join(os.path.split(
                                                                        self.params.common.run_name)) + f'_{self.params.common.target_device}.zip')

        if self.params.common.target_device in self.params.training.target_devices:
            inference_time_us_list = {k:v.get('inference_time_us') for k,v in self.params.training.target_devices.items()}
            sram_usage_list = {k: v.get('sram') for k, v in self.params.training.target_devices.items()}
            flash_usage_list = {k: v.get('flash') for k, v in self.params.training.target_devices.items()}
            print('---------------------------------------------------------------------')
            print(f'Run Name: {self.params.common.run_name}')
            print(f'- Model: {self.params.training.model_name}')
            print(f'- TargetDevices & Estimated Inference Times (us): {inference_time_us_list}')
            print(f'- TargetDevices & Estimated SRAM Usage (bytes): {sram_usage_list}')
            print(f'- TargetDevices & Estimated Flash Usage (bytes): {flash_usage_list}')
            print('- This model can be compiled for the above device(s).')
            print('---------------------------------------------------------------------')
        #

    def resolve_run_name(self, run_name, model_name):
        if not run_name:
            return ''
        #
        # modify or set any parameters here as required.
        if '{date-time}' in run_name:
            run_name = run_name.replace('{date-time}', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        #
        if '{model_name}' in run_name:
            run_name = run_name.replace('{model_name}', model_name)
        #
        return run_name

    def clear(self):
        pass

    def prepare(self):
        # create folders
        os.makedirs(self.params.common.project_path, exist_ok=True)
        os.makedirs(self.params.common.project_run_path, exist_ok=True)

        #####################################################################
        # handle all downloads here
        utils.download_all(self.params)

        #####################################################################
        # prepare for dataset handling (loading, splitting, limiting files etc).
        self.dataset_handling = datasets.DatasetHandling(self.params)
        self.params.update(self.dataset_handling.get_params())
        # actual dataset handling
        if self.params.dataset.enable:
            self.dataset_handling.clear()
            self.dataset_handling.run()
        #

        #####################################################################
        # prepare model training
        if self.params.training.enable:
            from . import training
            self.training_target_module = training.get_target_module(self.params.training.training_backend,
                                                                  self.params.common.task_category)
            self.model_training = self.training_target_module.ModelTraining(self.params)
            self.params.update(self.model_training.get_params())

        #####################################################################
        # prepare for model compilation
        # TODO : Uncomment below lines after adding compilation/tinyml_benchmark.py
        # self.model_compilation = tinyml_modelmaker.ai_modules.common.compilation.tinyml_benchmark.ModelCompilation(self.params)
        if self.params.compilation.enable:
            from . import compilation
            self.model_compilation = compilation.tinyml_benchmark.ModelCompilation(self.params)
            self.params.update(self.model_compilation.get_params())

        # write out the description of the current run
        run_params_file = self.write_status_file()
        return run_params_file

    def run(self):

        # actual model training
        if self.params.training.enable:
            # from ...version import __version__
            # with open(self.params.training.log_file_path, 'a') as lfp:
            #     lfp.write(f'\nModelmaker version: {__version__}\n')
            self.model_training.clear()
            self.model_training.run()
            # remove special characters
            utils.cleanup_special_chars(self.params.training.log_file_path)
            # training frameworks don't create a compact package after training. do it here.
            model_training_package_files = [
                self.params.dataset.annotation_path_splits,
                self.params.training.model_proto_path,
                self.params.training.log_file_path,
                self.params.training.tspa_license_path,
                self.params.training.file_level_classification_log_path
            ]

            if self.params.training.quantization == TinyMLQuantizationVersion.NO_QUANTIZATION:
                model_training_package_files.extend([
                    os.path.join(self.params.training.training_path, 'golden_vectors'),
                    os.path.join(self.params.training.training_path, 'post_training_analysis'),
                    os.path.join(self.params.training.training_path, 'model_aux.h'),]
                )
                if utils.misc_utils.str2bool(self.params.common.generic_model):
                    model_training_package_files.extend([self.params.training.model_checkpoint_path,
                                                         self.params.training.model_export_path,])
            else:
                model_training_package_files.extend([
                    os.path.join(self.params.training.training_path_quantization, 'golden_vectors'),
                    os.path.join(self.params.training.training_path_quantization, 'post_training_analysis'),
                    os.path.join(self.params.training.training_path_quantization, 'model_aux.h'),]

                )
                if utils.misc_utils.str2bool(self.params.common.generic_model):
                    model_training_package_files.extend([
                        self.params.training.model_checkpoint_path_quantization,
                        self.params.training.model_export_path_quantization, ])


            self.package_trained_model(model_training_package_files, self.params.training.model_packaged_path)
            if not utils.misc_utils.str2bool(self.params.testing.skip_train):
                if self.params.training.training_path_quantization:
                    print(f'\nTrained model is at: {self.params.training.training_path_quantization}\n')
                else:
                    print(f'\nTrained model is at: {self.params.training.training_path}\n')
            # we are done with training
            with open(self.params.training.log_file_path, 'a') as lfp:
                lfp.write('\nSUCCESS: ModelMaker - Training completed.')
            #
        #

        #####################################################################
        # actual model compilation
        if self.params.compilation.enable:
            # from ...version import __version__
            # with open(self.params.compilation.log_file_path, 'a') as lfp:
            #     lfp.write(f'\nModelmaker version: {__version__}\n')
            self.model_compilation.clear()
            exit_flag = self.model_compilation.run()
            if exit_flag:
                print(f'Compilation failed')
                with open(self.params.compilation.log_file_path, 'a') as lfp:
                    lfp.write('FAILURE: ModelMaker - Compilation failed.')
                return self.params
            
            os.makedirs(self.params.compilation.model_compiled_path, exist_ok=True)
            model_compilation_package_files = [
                os.path.join(self.params.compilation.compilation_path, 'artifacts'),
                self.params.compilation.tspa_license_path,
            ]
            if self.params.training.quantization == TinyMLQuantizationVersion.NO_QUANTIZATION:
                model_compilation_package_files.extend([
                    os.path.join(self.params.training.training_path, 'golden_vectors'),
                    os.path.join(self.params.training.training_path, 'post_training_analysis'),
                    os.path.join(self.params.training.training_path, 'model_aux.h'),
                ])
            else:
                model_compilation_package_files.extend([
                    os.path.join(self.params.training.training_path_quantization, 'golden_vectors'),
                    os.path.join(self.params.training.training_path_quantization, 'post_training_analysis'),
                    os.path.join(self.params.training.training_path_quantization, 'model_aux.h'),
                ])


            self.package_trained_model(model_compilation_package_files, self.params.compilation.model_packaged_path)
            print(f'Compiled model is at: {self.params.compilation.compilation_path}')
            with open(self.params.compilation.log_file_path, 'a') as lfp:
                lfp.write('\nSUCCESS: ModelMaker - Compilation completed.')
            if self.params.testing.device_inference:
                try:
                    from tinyml_testsuite import test_golden_vector
                    run_params_file = os.path.join(self.params.common.project_run_path, 'run.yaml')
                    test_golden_vector(run_params_file, True)
                except ImportError as e:
                    print(f"Device Inference cannot be done due to an exception: {e}")


        return self.params

    def get_params(self):
        return self.params

    def write_status_file(self):
        run_params_file = os.path.join(self.params.common.project_run_path, 'run.yaml')
        utils.write_dict(self.params, run_params_file)
        # create or update the status file
        status_params_file = os.path.join(self.params.common.project_run_path, 'status.yaml')
        status_params = dict()
        if os.path.exists(status_params_file):
            with open(status_params_file) as fp:
                status_params = yaml.safe_load(fp)

        status_params = utils.ConfigDict(status_params)
        # format the run_params to create status_params
        run_params_formatted = copy.deepcopy(self.params)
        # run_params_formatted.compilation = {self.params.common.target_device:run_params_formatted.compilation}
        run_params_formatted.compilation = run_params_formatted.compilation
        run_params_formatted = utils.ConfigDict(run_params_formatted)
        status_params.update(run_params_formatted)
        utils.write_dict(status_params, status_params_file)
        # Extra features requested as per MCE Spec
        if self.params.compilation.enable:
            os.makedirs(os.path.join(self.params.compilation.compilation_path), exist_ok=True)
            utils.write_dict(status_params, os.path.join(self.params.compilation.compilation_path, 'status.yaml'), write_yaml=False)
        if self.params.training.enable:
            os.makedirs(os.path.join(self.params.common.project_run_path, 'training'), exist_ok=True)
            utils.write_dict(status_params, os.path.join(self.params.common.project_run_path, 'training', 'status.yaml'), write_yaml=False)

        return run_params_file

    def package_trained_model(self, input_files, compressed_file_name):
        # tfp = tarfile.open(tarfile_name, 'w:gz', dereference=True)
        with ZipFile(compressed_file_name, 'w') as tfp:
            for inpf in input_files:
                inpf_list = inpf if isinstance(inpf, (list,tuple)) else [inpf]
                for inpf_entry in inpf_list:
                    if inpf_entry is not None and os.path.exists(inpf_entry):
                        if os.path.isdir(inpf_entry):
                            for root, dirs, files in os.walk(inpf_entry):
                                for file in files:
                                    tfp.write(
                                        os.path.join(root, file),
                                        arcname=os.path.relpath(os.path.join(root, file), os.path.dirname(inpf_entry)))
                        else:
                            tfp.write(inpf_entry, arcname=os.path.basename(inpf_entry))
                #
            #
        #
        # tfp.close()
        tarfile_size = os.path.getsize(compressed_file_name)
        return tarfile_size


    @staticmethod
    def get_training_module_descriptions(*args, **kwargs):
        return descriptions.get_training_module_descriptions(*args, **kwargs)

    @staticmethod
    def get_model_descriptions(*args, **kwargs):
        return descriptions.get_model_descriptions(*args, **kwargs)

    @staticmethod
    def get_model_description(*args, **kwargs):
        return descriptions.get_model_description(*args, **kwargs)

    @staticmethod
    def set_model_description(*args, **kwargs):
        return descriptions.set_model_description(*args, **kwargs)

    @staticmethod
    def get_preset_descriptions(*args, **kwargs):
        return descriptions.get_preset_descriptions(*args, **kwargs)

    @staticmethod
    def get_feature_extraction_preset_descriptions(*args, **kwargs):
        return descriptions.get_feature_extraction_preset_descriptions(*args, **kwargs)

    @staticmethod
    def get_dataset_preset_descriptions(*args, **kwargs):
        return descriptions.get_dataset_preset_descriptions(*args, **kwargs)

    @staticmethod
    def get_preset_compilations(*args, **kwargs):
        return descriptions.get_preset_compilations(*args, **kwargs)

    @staticmethod
    def get_target_device_descriptions(*args, **kwargs):
        return descriptions.get_target_device_descriptions(*args, **kwargs)

    @staticmethod
    def get_task_descriptions(*args, **kwargs):
        return descriptions.get_task_descriptions(*args, **kwargs)

    @staticmethod
    def get_sample_dataset_descriptions(*args, **kwargs):
        return descriptions.get_sample_dataset_descriptions(*args, **kwargs)

    @staticmethod
    def get_version_descriptions(*args, **kwargs):
        return descriptions.get_version_descriptions(*args, **kwargs)

    @staticmethod
    def get_tooltip_descriptions(*args, **kwargs):
        return descriptions.get_tooltip_descriptions(*args, **kwargs)

    @staticmethod
    def get_help_descriptions(*args, **kwargs):
        return descriptions.get_help_descriptions(*args, **kwargs)

    @staticmethod
    def get_live_capture_descriptions(*args, **kwargs):
        return descriptions.get_live_capture_descriptions(*args, **kwargs)

    @staticmethod
    def get_live_capture_example_descriptions(*args, **kwargs):
        return descriptions.get_live_capture_example_descriptions(*args, **kwargs)

    @staticmethod
    def get_live_preview_descriptions(*args, **kwargs):
        return descriptions.get_live_preview_descriptions(*args, **kwargs)

    @staticmethod
    def get_live_preview_example_descriptions(*args, **kwargs):
        return descriptions.get_live_preview_example_descriptions(*args, **kwargs)

    @staticmethod
    def get_context_help_descriptions(*args, **kwargs):
        return descriptions.get_context_help_descriptions(*args, **kwargs)

    @staticmethod
    def get_help_url_descriptions(*args, **kwargs):
        return descriptions.get_help_url_descriptions(*args, **kwargs)