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

import argparse
import copy
import getpass
import os
import re
import sys


def run(config):
    import tinyml_modelmaker

    # get the ai backend module
    ai_target_module = tinyml_modelmaker.ai_modules.get_target_module(config['common']['target_module'])

    # get params for the given config
    params = ai_target_module.runner.ModelRunner.init_params()

    # get_training_module_descriptions
    training_module_descriptions = ai_target_module.runner.ModelRunner.get_training_module_descriptions(params)

    # get supported pretrained models for the given params
    model_descriptions = ai_target_module.runner.ModelRunner.get_model_descriptions(params)
    feature_extraction_preset_descriptions = ai_target_module.runner.ModelRunner.get_feature_extraction_preset_descriptions(params)
    # update descriptions
    model_descriptions_desc = dict()
    for k, v in model_descriptions.items():
        s = copy.deepcopy(params)
        s.update(copy.deepcopy(v)).update(config)
        # if 'feature_extraction' in v.keys():  # Modify the feature_extraction_name choices as per model
        # Only populate feature extraction names whose task_type is same as model's task type
        feature_extraction_choices = [fe_name for fe_name, fe_dict in feature_extraction_preset_descriptions.items()
                                      if s.get('common').get('task_type') == fe_dict.get('common').get('task_type')]
        if feature_extraction_choices:
            for property_dict in s.get('training').get('properties'):
                # s.get('training').get('properties') is a list (of dicts)
                # property_dict is a dict
                if property_dict['name'] == 'feature_extraction_name':
                    new_enum = []
                    for feature_extraction_enum in property_dict['enum']:
                        if feature_extraction_enum['value'] in feature_extraction_choices:
                            new_enum.append(feature_extraction_enum)
                    property_dict['enum'] = new_enum
                    property_dict['default'] = new_enum[0]['value']  # Random to be set as default


        model_descriptions_desc[k] = s
    #

    # get presets
    preset_descriptions = ai_target_module.runner.ModelRunner.get_preset_descriptions(params)

    # get target device descriptions
    target_device_descriptions = ai_target_module.runner.ModelRunner.get_target_device_descriptions(params)

    # task descriptions
    task_descriptions = ai_target_module.runner.ModelRunner.get_task_descriptions(params)

    # sample dataset descriptions
    sample_dataset_descriptions = ai_target_module.runner.ModelRunner.get_sample_dataset_descriptions(params)

    # version info
    version_descriptions = ai_target_module.runner.ModelRunner.get_version_descriptions(params)

    # tooltip descriptions
    tooltip_descriptions = ai_target_module.runner.ModelRunner.get_tooltip_descriptions(params)

    # help descriptions - to be written to markdown (.md) file
    help_descriptions = ai_target_module.runner.ModelRunner.get_help_descriptions(params)

    description = dict(training_module_descriptions=training_module_descriptions,
                       model_descriptions=model_descriptions_desc,
                       preset_descriptions=preset_descriptions,
                       target_device_descriptions=target_device_descriptions,
                       task_descriptions=task_descriptions,
                       sample_dataset_descriptions=sample_dataset_descriptions,
                       version_descriptions=version_descriptions,
                       tooltip_descriptions=tooltip_descriptions,
                       help_descriptions=help_descriptions)
    return description, help_descriptions


def main(args):
    import tinyml_modelmaker

    # prepare input config
    kwargs = vars(args)
    config = dict(common=dict(), dataset=dict())
    if 'target_module' in kwargs:
        config['common']['target_module'] = kwargs['target_module']
    #
    if 'download_path' in kwargs:
        config['common']['download_path'] = kwargs['download_path']
    #

    # get description
    description, help = run(config)

    # write description
    description_file = os.path.join(args.description_path, f'description_{args.target_module}' + '.yaml')
    tinyml_modelmaker.utils.write_dict(description, description_file)
    with open(description_file) as df_yaml_fh:
        df_yaml_txt = df_yaml_fh.readlines()
    with open(description_file, 'w') as df_yaml_fh:
        for line in df_yaml_txt:
            df_yaml_fh.write(re.sub(os.path.join('home', getpass.getuser(), '.*/'), os.path.join('opt', 'tinyml', 'code', 'tinyml-mlbackend', 'tinyml_proprietary_models', ''), line))
    with open(os.path.splitext(description_file)[0]+'.json') as df_json_fh:
        df_json_txt = df_json_fh.readlines()
    with open(os.path.splitext(description_file)[0]+'.json', 'w') as df_json_fh:
        for line in df_json_txt:
            df_json_fh.write(re.sub(os.path.join('home', getpass.getuser(), '.*/'), os.path.join('opt', 'tinyml', 'tinyml-mlbackend', 'tinyml_proprietary_models', ''), line))


    help_file = os.path.join(args.description_path, f'help_{args.target_module}' + '.md')
    with open(help_file, 'w') as fp:
        fp.write(help)
    #

    print(f'description is written at: {description_file} and {help_file}')


if __name__ == '__main__':
    print(f'argv: {sys.argv}')
    # the cwd must be the root of the repository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('..')
    #

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--target_module', type=str, default='timeseries')
    parser.add_argument('--download_path', type=str, default=os.path.join('.', 'data', 'downloads'))
    parser.add_argument('--description_path', type=str, default=os.path.join('.', 'data', 'descriptions'))
    args = parser.parse_args()

    main(args)
