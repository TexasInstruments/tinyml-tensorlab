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

from argparse import ArgumentParser, Namespace
from glob import glob
from jinja2 import Environment, FileSystemLoader
from logging import getLogger
import numpy as np
import os
import onnxruntime
from shutil import copytree, rmtree
from tinyml_tinyverse.common.compilation import default_tvm_args
from tinyml_tinyverse.common.utils.mdcl_utils import command_display, Logger
from tvm.driver.tvmc.compiler import drive_compile
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
    parser.add_argument('--keep_libc_files', help='Keep lib0.c, lib1.c, lib2.c... files', action='store_true')

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
    # We need to copy a few tvm files (like dlpack, c_backend_api.h and c_runtime_api.h)
    # logger.info("Copying other files from TVM library (like dlpack, c_backend_api.h and c_runtime_api.h)")
    # copytree(src=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tinyml_tinyverse/common/compilation/tvm_required_files/"), dst=artifacts_dir, dirs_exist_ok=True)
    # We also need to delete the devc.o file as it is not required
    os.remove(os.path.join(artifacts_dir, 'devc.o'))
    if not(args.keep_libc_files):
        libc_files = glob(os.path.join(artifacts_dir, 'lib*.c'))
        for filename in libc_files:
            logger.debug("Removing {}".format(filename))
            os.remove(filename)
    return


"""
def get_model_ip_op_details(model_path):
    logger = getLogger("root.get_model_ip_op_details")
    np2c_dtype_dict = {np.uint8: 'uint8_t', np.uint16: 'uint16_t', np.uint32: 'uint32_t',
                       np.int8: 'int8_t', np.int16: 'int16_t', np.int32: 'int32_t',
                       np.float32: 'float', }
    # The above dict is used to convert from numpy data types to float data types
    logger.info("Loading model to infer data types and input/output shapes")
    model = onnx.load(model_path)
    # Now we shall get the shape of the inputs and outputs
    ip_dim, op_dim = [], []
    for inp in model.graph.input:
        shape = str(inp.type.tensor_type.shape.dim)
        ip_dim = [int(s) for s in shape.split() if s.isdigit()]
    for outp in model.graph.output:
        shape = str(outp.type.tensor_type.shape.dim)
        op_dim = [int(s) for s in shape.split() if s.isdigit()]
    # For TFLite and ONNX models separate statements
    # TODO: Get input_data_type and output_data_type
    input_data_type = np.float32
    output_data_type = np.uint8
    # Assuming only 1 input, 1 output for now
    return np2c_dtype_dict[input_data_type], np2c_dtype_dict[output_data_type], ip_dim, op_dim
"""


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


def get_model_ip_op_details(model_path):
    logger = getLogger("root.get_model_ip_op_details")
    # The below dict is used to convert from numpy data types to float data types
    np2c_dtype_dict = {np.uint8: 'uint8_t', np.uint16: 'uint16_t', np.uint32: 'uint32_t',
                       np.int8: 'int8_t', np.int16: 'int16_t', np.int32: 'int32_t',
                       np.float32: 'float', 'tensor(float)': 'float',}
    logger.info("Loading model to infer data types and input/output shapes")
    onnx_session = onnxruntime.InferenceSession(model_path)
    input_details = onnx_session.get_inputs()
    output_details = onnx_session.get_outputs()
    # Now we shall get the shape of the inputs and outputs
    # TODO: Assuming only 1 input, 1 output for now, require support for multiple datatypes
    # ip_dim = input_details[0].shape
    # ip_dtype = input_details[0].type
    # op_dim = output_details[0].shape
    # op_dtype = output_details[0].type
    # For TFLite and ONNX models separate statements
    return [(np2c_dtype_dict[ip.type], ip.shape) for ip in input_details], [(np2c_dtype_dict[op.type], op.shape) for op in output_details]
    # return np2c_dtype_dict[ip_dtype], np2c_dtype_dict[op_dtype], ip_dim, op_dim


def gen_wrapper_code(model_path, output_dir):
    logger = getLogger("root.gen_wrapper_code")
    input_data_type_dims, output_data_type_dims = get_model_ip_op_details(model_path)
    environment = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   "../../tinyml_tinyverse/common/compilation/templates")))
    template = environment.get_template("c_wrapper_template.txt")
    # Output data is only required if we're matching against golden. For now just keep it
    context = {}
    context['input_related_data'] = []
    context['output_related_data'] = []

    context['inputs_base_addresses'] = []
    context['outputs_base_addresses'] = []

    for i, (ip_dtype, ip_dim) in enumerate(input_data_type_dims):
        # context[f'input_data_type{i}'] = ip_dtype
        # context[f'ip_dim_str{i}'] = ''.join(["[{}]".format(x) for x in ip_dim])  # Formatting from (1,752,1)-> [1][752][1]
        # context[f'input_data{i}'] = np.array2string(np.random.randint(0, 255, size=np.prod(ip_dim)), separator=', ')[1:-1]
        # context[f'dim_str_ip_base{i}'] = ''.join(["[0]" for _ in ip_dim])  # Formatting from  [1][752][1] -> [0][0][0]
        '''
        The following string format: 
        {{input_data_type}} input_1{{ip_dim_str}} = { {{input_data1}} }; 
        '''
        context['input_related_data'].append('{input_data_type} input_{i}{ip_dim_str} = {{{input_data}}};'.format(
            input_data_type = ip_dtype, i=i+1,
            ip_dim_str= ''.join(["[{}]".format(x) for x in ip_dim]),  # Formatting from (1,752,1)-> [1][752][1]
            input_data=np.array2string(np.random.randint(0, 255, size=np.prod(ip_dim)), separator=', ')[1:-1],
            # ''.join(["[0]" for _ in ip_dim])
        ))
        '''
        The following string format:
        struct tvmgen_default_inputs  inputs = {&input_1{{dim_str_ip_base}}};
        '''
        context['inputs_base_addresses'].append('&input_{i}{dim_str_ip_base}'.format(
            i=i+1, dim_str_ip_base=''.join(["[0]" for _ in ip_dim]),
        ))

    for i, (op_dtype, op_dim) in enumerate(output_data_type_dims):
        # context[f'output_data_type{i}'] = op_dtype
        # context[f'op_dim_str{i}'] = ''.join(["[{}]".format(x) for x in op_dim])  # Formatting from (1)-> [1]
        # context[f'output_data{i}'] = np.array2string(np.random.randint(0, 255, size=np.prod(op_dim)), separator=', ')[1:-1],
        # context[f'dim_str_op_base{i}'] = ''.join(["[0]" for _ in op_dim]),
        '''
        The following string format:
        {{output_data_type}} output_1{{op_dim_str}} = { {{output_data1}} }; 
        '''
        context['output_related_data'].append('{output_data_type} output_{i}{op_dim_str} = {{{output_data}}};'.format(
            output_data_type=op_dtype, i=i+1,
            op_dim_str=''.join(["[{}]".format(x) for x in op_dim]),  # Formatting from (1,752,1)-> [1][752][1]
            output_data=np.array2string(np.random.randint(0, 255, size=np.prod(op_dim)), separator=', ')[1:-1],
            # ''.join(["[0]" for _ in op_dim])
        ))
        '''
        The following string format:
        struct tvmgen_default_outputs outputs = {&output_1{{dim_str_op_base}}};
        '''
        context['outputs_base_addresses'].append(
            '&output_{i}{dim_str_op_base}'.format(
                i=i+1, dim_str_op_base=''.join(["[0]" for _ in op_dim]),
            ))

    context['input_related_data'] = '\n'.join(context['input_related_data'])
    context['output_related_data'] = '\n'.join(context['output_related_data'])

    context['inputs_base_addresses'] = ', '.join(context['inputs_base_addresses'])
    context['outputs_base_addresses'] = ', '.join(context['outputs_base_addresses'])

    wrapper_filename = os.path.join(output_dir, "app.c")
    with open(wrapper_filename, mode="w", encoding="utf-8") as fh:
        fh.write(template.render(context))
    logger.info("Creating wrapper script using template at: {}".format(wrapper_filename))

    return


def main(args):
    # ret_code = gen_artifacts(args)
    # if ret_code:
    #     raise Exception(f"Process failed with return code {ret_code}")
    # logger = command_display(args.lis or os.path.join(args.output_dir, 'compilation.lis'), args.DEBUG)
    logger = Logger(log_file=args.lis or os.path.join(args.output_dir, 'compilation.lis'), DEBUG=args.DEBUG, name="root", append_log=False, console_log=True)
    logger.info("Script: {}".format(os.path.relpath(__file__)))
    logger.info(args)
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
    gen_artifacts(args)
    if not args.keep_intermittent_files:
        remove_intermittent_files(args.output_dir)
    # gen_wrapper_code(args.FILE, os.path.join(args.output_dir, 'artifacts'))  # TODO: Enable this once a better app.c can be done


def run(args):
    main(args)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run(args)
