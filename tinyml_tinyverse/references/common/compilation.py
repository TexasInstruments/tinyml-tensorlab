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
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from glob import glob
from logging import getLogger

import numpy as np
from tvm.driver.tvmc.compiler import drive_compile

from tinyml_tinyverse.common.compilation.tvm_input_config import default_tvm_args
from tinyml_tinyverse.common.utils import misc_utils, utils
from tinyml_tinyverse.common.utils.mdcl_utils import Logger

np.set_printoptions(threshold=np.inf)


def get_args_parser():
    DESCRIPTION = "Given model, this script generates artifacts and a wrapper app.c for running on MCU"
    parser = ArgumentParser(description=DESCRIPTION)
    # parser.add_argument('--model_path', help="The model to compile (tflite/onnx)")
    parser.add_argument('--output_dir', default=os.getcwd(), help="Output directory to dump artifacts")
    parser.add_argument('--keep_intermittent_files', action='store_true', help='Keep intermittent .bin files')
    parser.add_argument('--config', help='', type=str, default='default', )
    parser.add_argument('--verbose', help="increase verbosity.", type=int, default=0, )
    parser.add_argument('--version', help='', type=bool, default=False, )
    parser.add_argument('--cross_compiler',
                        help="the cross compiler to generate target libraries, e.g. 'tiarmclang/aarch64-linux-gnu-gcc'.",
                        type=str, default='tiarmclang', )
    parser.add_argument('--cross_compiler_options',
                        help="the cross compiler options to generate target libraries, e.g. '-mfpu=neon-vfpv4'.",
                        type=str,
                        default='-O3 -mcpu=cortex-r5 -march=armv7-r -mthumb -mfloat-abi=hard -mfpu=vfpv3-d16 -mlittle-endian -Iartifacts_r5_c -Wno-return-type', )
    parser.add_argument('--dump_code', help='', type=str, default='', )
    parser.add_argument('--dump_offloads', help='', type=str, default='', )
    parser.add_argument('--output', help="output the compiled module to a specified archive. Defaults to 'module.tar'.",
                        type=str, default='mod.a', )

    parser.add_argument('--target', help='', type=str, default='c', )
    parser.add_argument('--target_c_mcpu', help='', type=str, default='cortex-r5', )
    parser.add_argument('--target_cmsis_nn_mcpu', help='', type=str, default=None, )
    parser.add_argument('--target_cmsis_nn_mattr', help='', type=str, default=None, )
    parser.add_argument('--target_cmsis_nn_debug_last_error', help='', type=str, default=None, )
    parser.add_argument('--tuning_records', help='', type=str, default='', )

    parser.add_argument('--FILE', help='The model to compile (tflite/onnx)', type=str, default='model.onnx', )
    parser.add_argument('--opt_level', help="specify which optimization level to use. Defaults to '3'.", type=int,
                        default=3, )
    parser.add_argument('--module_name', help="The output module name. Defaults to 'default'.", type=str,
                        default='default', )
    parser.add_argument('--lis', help='Log File', type=str,)  # default=ops(opb(__file__))[0] + ".lis")
    parser.add_argument('--DEBUG', action='store_true', help='Log mode set to DEBUG')

    # Following arguments were used with default values for all tvm runs.
    # Hence the arguments were taken out from tinyml_modelmaker/tinyml_benchmark.py
    parser.add_argument('--output_format',
                        help="output format. Use 'so' for shared object or 'mlf' for Model Library Format "
                             "(only for microTVM targets), or 'a' for static library. Defaults to 'so'.", type=str,
                        default='a', )
    parser.add_argument('--pass_config',
                        help="configurations to be used at compile time. This option can be provided multiple "
                             "times, each one to set one configuration value, "
                             "e.g. '--pass-config relay.backend.use_auto_scheduler=0', "
                             "e.g. '--pass-config tir.add_lower_pass=opt_level1,pass1,opt_level2,pass2'.", type=str,
                        nargs='+', default=['tir.disable_vectorize=1', 'tir.usmp.algorithm=hill_climb'], )
    parser.add_argument('--executor', help="The graph executor to build the model ", type=str, default='aot', )
    parser.add_argument('--executor_aot_unpacked_api', help='', type=int, default=1, )
    parser.add_argument('--executor_aot_interface_api', help='', type=str, default='c', )
    parser.add_argument('--runtime', help="The runtime configuration.", type=str, default='crt', )
    parser.add_argument('--keep_libc_files', help='Keep lib0.c, lib1.c, lib2.c... files', action=BooleanOptionalAction)
    parser.add_argument('--generic-model', help="Open Source models", type=misc_utils.str_or_bool, default=False)

    return parser


def gen_artifacts(args):
    logger = getLogger("root.gen_artifacts")
    input_args = dict()
    input_args.update(default_tvm_args)
    logger.debug("Input Arguments before updating args: {}".format(input_args))
    input_args.update(args.__dict__)
    logger.debug("Input Arguments before updating output: {}".format(input_args))
    artifacts_dir = os.path.join(args.output_dir, "artifacts")
    input_args.update({'output': os.path.join(artifacts_dir, args.output)})
    logger.debug("Input Arguments after updating output: {}".format(input_args))

    curr_dir = os.getcwd()
    logger.info("Changing directory to: {}".format(args.output_dir))
    os.chdir(args.output_dir)
    try:
        logger.info("Calling TVM to generate artifacts: ")
        drive_compile(Namespace(**input_args))
    except Exception:
        raise
    logger.info("Changing directory back to: {}".format(curr_dir))
    os.chdir(curr_dir)
    # We also need to delete the devc.o file as it is not required
    try:
        os.remove(os.path.join(artifacts_dir, 'devc.o'))
    except FileNotFoundError:
        pass
    if not(args.keep_libc_files):
        libc_files = glob(os.path.join(artifacts_dir, 'lib*.c'))
        for filename in libc_files:
            logger.debug("Removing {}".format(filename))
            os.remove(filename)
    return


def remove_intermittent_files(dir):
    logger = getLogger("root.remove_intermittent_files")
    logger.info(f"Removing intermittent files from: {dir}")
    for file in glob(os.path.join(dir, '*bin')):
        logger.debug(f"Deleting {file}")
        os.remove(file)
    for file in glob(os.path.join(dir, '*txt')):
        logger.debug(f"Deleting {file}")
        os.remove(file)
    return


def main(args):
    logger = Logger(log_file=args.lis or os.path.join(args.output_dir, 'compilation.lis'),
                    DEBUG=args.DEBUG, name="root", append_log=True, console_log=True)
    from ..version import get_version_str
    logger.info(f"TinyVerse Toolchain Version: {get_version_str()}")
    logger.info("Script: {}".format(os.path.relpath(__file__)))
    logger.debug(args)
    # Often we hear of compilation breaking as the compiler/sdk paths provided are invalid
    exit_flag = 0
    if not os.path.exists(args.cross_compiler):
        logger.error(f'Cross Compiler path is invalid: {args.cross_compiler}')
        exit_flag = 1
    for arg in args.cross_compiler_options.split():
        if arg.startswith('-I') and (arg not in ['-I.', '-Iartifacts']):
            if not os.path.exists(arg[2:]):  # [2:] to remove '-I'
                logger.error(f"Compilation will fail as path is invalid: {arg[2:]}")
                exit_flag = 1
    if exit_flag:
        logger.info("By default, compiler and SDK are searched in ~/bin/, unless set explicitly by user using TOOLS_PATH or C2000WARE_PATH or CGT_PATH")
        logger.error("Exiting due to previous errors. Compiled model directory will be empty.")
        return

    args.model_format = None
    args.input_shapes = None
    args.dump_code = None
    args.dump_offloads = None
    args.target_example_target_hook_device = None
    args.target_example_target_hook_from_device = None
    args.target_example_target_hook_libs = None
    args.target_example_target_hook_target_device_type = None
    args.target_cmsis_nn_mcpu = None if args.target_cmsis_nn_mcpu=='None' else args.target_cmsis_nn_mcpu
    args.target_cmsis_nn_mattr = None if args.target_cmsis_nn_mattr == 'None' else args.target_cmsis_nn_mattr
    if not args.generic_model:
        try:
            utils.decrypt(args.FILE, utils.get_crypt_key())
        except Exception:
            pass
    try:
        gen_artifacts(args)
    except Exception:
        if not args.generic_model:
            utils.encrypt(args.FILE, utils.get_crypt_key())
        raise
    if not args.generic_model:
        utils.encrypt(args.FILE, utils.get_crypt_key())

    if not args.keep_intermittent_files:
        remove_intermittent_files(args.output_dir)


def run(args):
    # This code was majorly cleaned up in TINYML_ALGO-212
    main(args)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run(args)
