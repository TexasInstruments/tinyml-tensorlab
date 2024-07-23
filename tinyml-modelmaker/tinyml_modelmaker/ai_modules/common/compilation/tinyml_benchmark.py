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
import shutil
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

from .... import utils
from ...timeseries import constants

this_dir_path = os.path.dirname(os.path.abspath(__file__))
repo_parent_path = os.path.abspath(os.path.join(this_dir_path, '../../../../../'))

tinyml_tinyverse_path = os.path.join(repo_parent_path, 'tinyml-tinyverse')

class ModelCompilation():
    @classmethod
    def init_params(self, *args, **kwargs):
        params = dict(
            compilation=dict(
            )
        )
        params = utils.ConfigDict(params, *args, **kwargs)
        return params

    def __init__(self, *args, quit_event=None, **kwargs):
        self.params = self.init_params(*args, **kwargs)
        self.quit_event = quit_event
        self.artifact_ext = '.zip'
        # prepare for model compilation
        # TODO: self._prepare_pipeline_config()
        self.work_dir = os.path.join(self.params.get('compilation').get('compilation_path'), 'artifacts')
        self.package_dir = os.path.join(self.params.get('compilation').get('compilation_path'), 'pkg')
        # TODO: Can we have something for progress regex?
        progress_regex = \
            {'type':'Progress', 'name':'Progress', 'description':'Progress of Compilation', 'unit':'Frame', 'value':None,
             'regex':[{'op':'search', 'pattern':r'infer\s+\:\s+.*?\s+(?<infer>\d+)', 'groupId':'infer'}],
             }

        if self.params.common.task_category == constants.TASK_CATEGORY_TS_CLASSIFICATION:
            log_summary_regex = {
                'js': [
                    progress_regex,
                    # {'type':'Validation Accuracy', 'name':'Accuracy', 'description':'Accuracy of Compilation', 'unit':'Accuracy Top-1%', 'value':None,
                    #  'regex':[{'op':'search', 'pattern':r'benchmark results.*?accuracy_top1.*?\:\s+(?<accuracy>\d+\.\d+)', 'group':1, 'dtype':'float', 'scale_factor':1}],
                    #  },
                    {'type': 'Completed',
                     'name': 'Completed', 'description': 'Completion of Compilation', 'unit': None,
                     'value': None,
                     'regex': [
                         {'op': 'search', 'pattern': r'success\:.*compilation\s+completed', 'groupId': '', 'dtype': 'str',
                          'case_sensitive': False}],
                     },
                ]
            }
        else:
            log_summary_regex = None
        #

        model_compiled_path = self._get_compiled_artifact_dir()
        packaged_artifact_path = self._get_packaged_artifact_path() # actual, internal, short path
        # model_packaged_path = self.work_dir # TODO: This has been changed from self._get_final_artifact_path() # a more descriptive symlink

        self.params.update(
            compilation=utils.ConfigDict(
                model_compiled_path=model_compiled_path,
                log_file_path=os.path.join(self.params.compilation.compile_output_path if self.params.compilation.compile_output_path else self.params.compilation.compilation_path, 'run.log'),
                log_summary_regex=log_summary_regex,
                summary_file_path=os.path.join(model_compiled_path, 'summary.yaml'),
                output_tensors_path=os.path.join(model_compiled_path, 'outputs'),
                # model_packaged_path=model_packaged_path, # final compiled package
                model_visualization_path=os.path.join(model_compiled_path, 'artifacts', 'tempDir', 'runtimes_visualization.svg'),
            )
        )

    def clear(self):
        # clear the dirs
        # shutil.rmtree(self.params.compilation.compilation_path, ignore_errors=True)
        return

    def run(self, **kwargs):
        ''''
        The actual compilation function. Move this to a worker process, if this function is called from a GUI.
        '''
        os.makedirs(self.params.compilation.compilation_path, exist_ok=True)

        if self.params.compilation.model_path and os.path.exists(self.params.compilation.model_path):
            model_file = self.params.compilation.model_path
        else:
            model_file = self.params.training.model_export_path_quantization if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION else self.params.training.model_export_path
        argv = [
            '--FILE', f'{model_file}',
            '--output_dir', f'{self.params.compilation.compilation_path}',
            '--config', f'{self.params.compilation.compilation_path}',
            '--cross_compiler', f'{self.params.compilation.cross_compiler}',
            '--cross_compiler_options', f'{self.params.compilation.cross_compiler_options}',
            '--target', f'{self.params.compilation.target}',
            '--target_c_mcpu', f'{self.params.compilation.target_c_mcpu}',
            '--keep_libc_files' if self.params.compilation.keep_libc_files else '--no-keep_libc_files',
            '--lis', f'{self.params.compilation.log_file_path}',
        ]
        compile_scr = utils.import_file_or_folder(os.path.join(tinyml_tinyverse_path, 'references', 'common', 'compilation.py'), __name__, force_import=True)
        args = compile_scr.get_args_parser().parse_args(argv)
        compile_scr.run(args)
        args.quit_event = self.quit_event

    def _get_compiled_artifact_dir(self):
        # compiled_artifact_dir = os.path.join(self.work_dir, self.params.compilation.model_compilation_id)
        compiled_artifact_dir = self.work_dir
        return compiled_artifact_dir

    def _get_packaged_artifact_path(self):
        # packaged_artifact_path = os.path.join(self.package_dir, self.params.compilation.model_compilation_id) + self.artifact_ext
        packaged_artifact_path = os.path.join(self.package_dir) + self.artifact_ext
        return packaged_artifact_path

    # final_artifact_name is a more descriptive name with the actual name of the model
    # this will be used to create a symlink to the packaged_artifact_path
    def _get_final_artifact_path(self):
        pipeline_config = list(self.pipeline_configs.values())[0]
        session_name = pipeline_config['session'].get_param('session_name')
        target_device_suffix = self.params.common.target_device
        run_name_splits = list(os.path.split(self.params.common.run_name))
        final_artifact_name = '_'.join(run_name_splits + [session_name, target_device_suffix])
        final_artifact_path = os.path.join(self.package_dir, final_artifact_name) + self.artifact_ext
        return final_artifact_path

    def _has_logs(self):
        log_dir = self._get_compiled_artifact_dir()
        if (log_dir is None) or (not os.path.exists(log_dir)):
            return False
        #
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        if len(log_files) == 0:
            return False
        #
        return True

    def get_params(self):
        return self.params
