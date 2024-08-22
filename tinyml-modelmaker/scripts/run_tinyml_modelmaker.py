# #################################################################################
# # Copyright (c) 2023-2024, Texas Instruments
# # All Rights Reserved.
# #
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # * Redistributions of source code must retain the above copyright notice, this
# #   list of conditions and the following disclaimer.
# #
# # * Redistributions in binary form must reproduce the above copyright notice,
# #   this list of conditions and the following disclaimer in the documentation
# #   and/or other materials provided with the distribution.
# #
# # * Neither the name of the copyright holder nor the names of its
# #   contributors may be used to endorse or promote products derived from
# #   this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# #################################################################################
#
# import os
# import datetime
# import sys
# import argparse
# import yaml
# import json
#
#
# def main(config):
#     import tinyml_modelmaker
#
#     # get the ai backend module
#     ai_target_module = tinyml_modelmaker.ai_modules.get_target_module(config['common']['target_module'])
#
#     # get default params
#     params = ai_target_module.runner.ModelRunner.init_params()
#
#     # get pretrained model for the given model_name
#
#     model_name = config['training']['model_name']
#     model_description = ai_target_module.runner.ModelRunner.get_model_description(model_name)
#     if config['training']['enable']:
#         if model_description is None:
#             print(f"please check if the given model_name is a supported one: {model_name}")
#             return False
#     #
#     dataset_preset_descriptions = ai_target_module.runner.ModelRunner.get_dataset_preset_descriptions(params)
#     dataset_preset_name = ai_target_module.constants.DATASET_DEFAULT
#     if config['dataset']['enable']:
#         if 'dataset_name' in config['dataset']:
#             dataset_preset_name = config['dataset']['dataset_name']
#     dataset_preset_description = dataset_preset_descriptions.get(dataset_preset_name) or dict()
#
#     feature_extraction_preset_descriptions = ai_target_module.runner.ModelRunner.get_feature_extraction_preset_descriptions(params)
#     feature_extraction_preset_name = ai_target_module.constants.FEATURE_EXTRACTION_DEFAULT
#     if 'feature_extraction_name' in config['feature_extraction']:
#         feature_extraction_preset_name = config['feature_extraction']['feature_extraction_name']
#     feature_extraction_preset_description = feature_extraction_preset_descriptions.get(feature_extraction_preset_name) or dict()
#
#     # get the presets for this device and task
#     # applying the default_preset. The values can be changed from config file if needed
#     preset_descriptions = ai_target_module.runner.ModelRunner.get_preset_descriptions(params)
#     target_device = config['common']['target_device']
#     task_type = config['common']['task_type']  # Because SDTO-CCS was unwilling to change task_type to application_type
#     config['common']['task_type'] = task_type
#     task_category = tinyml_modelmaker.get_task_category_type_from_task_type(task_type)
#     config['common']['task_category'] = task_category
#
#     compilation_preset_name = ai_target_module.constants.COMPILATION_DEFAULT  # 'default_preset'
#     if 'compile_preset_name' in config['compilation']:
#         compilation_preset_name = config['compilation']['compile_preset_name']
#     compilation_preset_description = preset_descriptions[target_device][task_type][compilation_preset_name]
#
#     # update the params with model_description, preset and config
#     params = params.update(model_description).update(dataset_preset_description).update(feature_extraction_preset_description).update(compilation_preset_description).update(config)
#
#     # create the runner
#     model_runner = ai_target_module.runner.ModelRunner(
#         params
#     )
#
#     # prepare
#     run_params_file = model_runner.prepare()
#     print(f'Run params is at: {run_params_file}')
#
#     # run
#     model_runner.run()
#
#     # finish up
#     if model_runner.get_params().training.enable:
#         if not (model_runner.get_params().testing.skip_train in [True, 'True', 'true', 1, '1']):
#             if model_runner.get_params().training.training_path_quantization:
#                 print(f'Trained model is at: {model_runner.get_params().training.training_path_quantization}')
#             else:
#                 print(f'Trained model is at: {model_runner.get_params().training.training_path}')
#     if model_runner.get_params().compilation.enable:
#         print(f'Compiled model is at: {model_runner.get_params().compilation.compilation_path}')
#     return True
#
#
# if __name__ == '__main__':
#     print(f'argv: {sys.argv}')
#     # the cwd must be the root of the repository
#     if os.path.split(os.getcwd())[-1] == 'scripts':
#         os.chdir('..')
#     #
#
#     parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
#     parser.add_argument('config_file', type=str, default=None)
#     parser.add_argument('--run_name', type=str)
#     parser.add_argument('--task_type', type=str)
#     parser.add_argument('--model_name', type=str)
#     parser.add_argument('--input_data_path', type=str)
#     parser.add_argument('--target_device', type=str)
#     parser.add_argument('--num_gpus', type=int)
#     parser.add_argument('--batch_size', type=int)
#     args = parser.parse_args()
#
#     # read the config
#     with open(args.config_file) as fp:
#         if args.config_file.endswith('.yaml'):
#             config = yaml.safe_load(fp)
#         elif args.config_file.endswith('.json'):
#             config = json.load(fp)
#         else:
#             assert False, f'unrecognized config file extension for {args.config_file}'
#         #
#     #
#
#     # override with supported commandline args
#     kwargs = vars(args)
#     if 'run_name' in kwargs:
#         config['common']['run_name'] = kwargs['run_name']
#     #
#     if 'task_type' in kwargs:
#         config['common']['task_type'] = kwargs['task_type']
#     #
#     if 'target_device' in kwargs:
#         config['common']['target_device'] = kwargs['target_device']
#     #
#     if 'model_name' in kwargs:
#         config['training']['model_name'] = kwargs['model_name']
#     #
#     if 'input_data_path' in kwargs:
#         config['dataset']['input_data_path'] = kwargs['input_data_path']
#     #
#     if 'num_gpus' in kwargs:
#         config['training']['num_gpus'] = kwargs['num_gpus']
#     #
#     if 'batch_size' in kwargs:
#         config['training']['batch_size'] = kwargs['batch_size']
#     #
#     if 'learning_rate' in kwargs:
#         config['training']['learning_rate'] = kwargs['learning_rate']
#     #
#
#     main(config)
