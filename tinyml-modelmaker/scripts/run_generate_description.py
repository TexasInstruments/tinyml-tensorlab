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

        # GUI expects an object (dict) here, matching description_timeseries.json's
        # shape - per-device model_selection_factor/flash/inference_time_us/sram
        # metadata doesn't exist in the current codebase, so empty per-device
        # objects are the best available; keeps the shape stable for GUI parsing.
        target_devices = s.get('training', {}).get('target_devices')
        if isinstance(target_devices, list):
            s['training']['target_devices'] = {d: {} for d in target_devices}
        #

        model_descriptions_desc[k] = s
    #

    # get presets
    preset_descriptions = ai_target_module.runner.ModelRunner.get_preset_descriptions(params)

    # get target device descriptions
    target_device_descriptions = ai_target_module.runner.ModelRunner.get_target_device_descriptions(params)
    # GUI expects an object (dict) here too, matching description_timeseries.json's
    # shape - per-device metadata (device_type/device_details/sdk_version/etc) no
    # longer exists in the current codebase (see overrides.json's
    # target_device_metadata for the last known values), so empty objects it is.
    if isinstance(target_device_descriptions, list):
        target_device_descriptions = {d: {} for d in target_device_descriptions}
    #

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

    # rex dependencies
    # rex_dependencies = ai_target_module.runner.ModelRunner.get_rex_dependencies(params)

    description = dict(model_descriptions=model_descriptions_desc,
                       preset_descriptions=preset_descriptions,
                       target_device_descriptions=target_device_descriptions,
                       task_descriptions=task_descriptions,
                       sample_dataset_descriptions=sample_dataset_descriptions,
                       version_descriptions=version_descriptions,
                       tooltip_descriptions=tooltip_descriptions,
                       help_descriptions=help_descriptions,
                    )
    return description, help_descriptions


def _sanitize_paths(written_file):
    # written_file is the .yaml path written by tinyml_modelmaker.utils.write_dict
    # (which also writes the sibling .json). Strip the local user's home dir out
    # of any absolute paths baked into the description so it is portable to the
    # mlbackend container.
    yaml_file = os.path.splitext(written_file)[0] + '.yaml'
    with open(yaml_file) as df_yaml_fh:
        df_yaml_txt = df_yaml_fh.readlines()
    with open(yaml_file, 'w') as df_yaml_fh:
        for line in df_yaml_txt:
            df_yaml_fh.write(re.sub(os.path.join('home', getpass.getuser(), '.*/'), os.path.join('opt', 'tinyml', 'code', 'tinyml-mlbackend', 'tinyml_proprietary_models', ''), line))

    json_file = os.path.splitext(written_file)[0] + '.json'
    with open(json_file) as df_json_fh:
        df_json_txt = df_json_fh.readlines()
    with open(json_file, 'w') as df_json_fh:
        for line in df_json_txt:
            df_json_fh.write(re.sub(os.path.join('home', getpass.getuser(), '.*/'), os.path.join('opt', 'tinyml', 'tinyml-mlbackend', 'tinyml_proprietary_models', ''), line))


def _flatten_descriptions(combined_description):
    # combined_description: {module_name: description_dict}. GUI wants the old
    # single-namespace shape back (no top-level module key). model_descriptions/
    # sample_dataset_descriptions/task_descriptions have zero name collisions across
    # modules (verified) - straight merge, but assert loudly instead of silently
    # overwriting if that ever stops being true (e.g. a future module reintroduces
    # the registry-leak bug fixed for timeseries/radar).
    modules = list(combined_description.keys())
    flat = dict()

    for section in ('model_descriptions', 'sample_dataset_descriptions', 'task_descriptions'):
        merged = dict()
        for m in modules:
            for k, v in combined_description[m][section].items():
                if k in merged:
                    raise ValueError(f"_flatten_descriptions: key '{k}' in '{section}' collides "
                                     f"across modules (module '{m}') - cannot flatten safely")
                #
                merged[k] = v
            #
        #
        flat[section] = merged
    #

    # preset_descriptions: keyed by device name, second level keyed by task_type.
    # Device names collide across modules (same physical device, different modules'
    # task-type support) but task_type keys never collide across modules - deep
    # merge at the device level instead of overwriting the whole device entry.
    merged_presets = dict()
    for m in modules:
        for device, task_type_dict in combined_description[m]['preset_descriptions'].items():
            existing = merged_presets.setdefault(device, dict())
            for task_type, v in task_type_dict.items():
                if task_type in existing:
                    raise ValueError(f"_flatten_descriptions: task_type '{task_type}' for device "
                                     f"'{device}' collides across modules (module '{m}')")
                #
                existing[task_type] = v
            #
        #
    #
    flat['preset_descriptions'] = merged_presets

    # tooltip_descriptions: identical across modules in practice - deep-merge
    # subcategories, assert no conflicting value for a key seen in >1 module.
    merged_tooltips = dict()
    for m in modules:
        for category, entries in combined_description[m]['tooltip_descriptions'].items():
            existing = merged_tooltips.setdefault(category, dict())
            for k, v in entries.items():
                if k in existing and existing[k] != v:
                    raise ValueError(f"_flatten_descriptions: tooltip '{category}.{k}' conflicts "
                                     f"across modules (module '{m}')")
                #
                existing[k] = v
            #
        #
    #
    flat['tooltip_descriptions'] = merged_tooltips

    versions = {combined_description[m]['version_descriptions']['version'] for m in modules}
    if len(versions) > 1:
        raise ValueError(f"_flatten_descriptions: version_descriptions differ across modules: {versions}")
    #
    flat['version_descriptions'] = dict(version=next(iter(versions)))

    # target_device_descriptions: object keyed by device name per module - union,
    # first-appearance order preserved (dict insertion order).
    merged_devices = dict()
    for m in modules:
        for device, meta in combined_description[m]['target_device_descriptions'].items():
            merged_devices.setdefault(device, meta)
        #
    #
    flat['target_device_descriptions'] = merged_devices

    # help_descriptions: per-module prose - can't merge into one string without
    # losing meaning, concatenate with module headers instead.
    flat['help_descriptions'] = '\n\n'.join(
        f'# {m}\n\n{combined_description[m]["help_descriptions"]}' for m in modules)

    return flat


def main(args):
    import tinyml_modelmaker

    kwargs = vars(args)
    target_modules = kwargs['target_module']
    target_modules = [target_modules] if isinstance(target_modules, str) else target_modules

    combined_description = dict()
    combined_help = dict()
    for target_module in target_modules:
        config = dict(common=dict(target_module=target_module), dataset=dict())
        if 'download_path' in kwargs:
            config['common']['download_path'] = kwargs['download_path']
        #

        description, help = run(config)

        # write per-module description (kept for backward compatibility - e.g.
        # tinyml-mlbackend/model_composer_extensions/config.json points at description_timeseries.json)
        description_file = os.path.join(args.description_path, f'description_{target_module}' + '.yaml')
        tinyml_modelmaker.utils.write_dict(description, description_file)
        _sanitize_paths(description_file)

        help_file = os.path.join(args.description_path, f'help_{target_module}' + '.md')
        with open(help_file, 'w') as fp:
            fp.write(help)
        #

        combined_description[target_module] = description
        combined_help[target_module] = help

        print(f'description is written at: {description_file} and {help_file}')
    #

    # flatten every target module's description into a single descriptions.json/.yaml
    # with no top-level module key (matches the old description_timeseries.json shape)
    flat_description = _flatten_descriptions(combined_description)
    combined_file = os.path.join(args.description_path, 'descriptions.yaml')
    tinyml_modelmaker.utils.write_dict(flat_description, combined_file)
    _sanitize_paths(combined_file)

    combined_help_file = os.path.join(args.description_path, 'descriptions_help.md')
    with open(combined_help_file, 'w') as fp:
        for target_module, help_text in combined_help.items():
            fp.write(f'# {target_module}\n\n{help_text}\n\n')
        #
    #

    print(f'combined description is written at: {combined_file}')


if __name__ == '__main__':
    print(f'argv: {sys.argv}')
    # the cwd must be the root of the repository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('..')
    #

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--target_module', type=str, nargs='+', default=['timeseries', 'audio', 'vision', 'radar'])
    parser.add_argument('--download_path', type=str, default=os.path.join('.', 'data', 'downloads'))
    parser.add_argument('--description_path', type=str, default=os.path.join('.', 'data', 'descriptions'))
    args = parser.parse_args()

    main(args)
