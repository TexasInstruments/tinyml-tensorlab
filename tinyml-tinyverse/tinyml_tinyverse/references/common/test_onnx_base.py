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

"""
Base module for ONNX model testing scripts.
Contains common functionality shared across classification, regression,
anomaly detection, and forecasting tasks.
"""

import datetime
import os
import platform
from argparse import ArgumentParser

import onnxruntime as ort
import torch

from tinyml_tinyverse.common.utils import misc_utils, utils
from tinyml_tinyverse.common.utils.mdcl_utils import Logger


def get_base_test_args_parser(description="This script loads time series dataset and tests it against a onnx model using ONNX RT"):
    """
    Create argument parser with common arguments shared across all task types for testing.
    Task-specific arguments should be added by calling add_task_specific_args().
    """
    parser = ArgumentParser(description=description)
    parser.add_argument('--dataset', default='folder', help='dataset')
    parser.add_argument('--dataset-loader', default='SimpleTSDataset', help='dataset loader')
    parser.add_argument("--loader-type", default="classification", type=str,
                        help="Dataset Loader Type: classification/regression/forecasting")
    parser.add_argument('--annotation-prefix', default='instances', help='annotation-prefix')
    parser.add_argument('--data-path', default=os.path.join('.', 'data', 'datasets'), help='dataset')
    parser.add_argument('--output-dir', default=None, help='path where to save')
    parser.add_argument('--model-path', default=None, help='ONNX model Path')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
    parser.add_argument('-j', '--workers', default=0 if platform.system() in ['Windows'] else 16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--date', default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), help='current date')
    parser.add_argument('--seed', default=42, help="Seed for all randomness", type=int)
    parser.add_argument('--lis', help='Log File', type=str)
    parser.add_argument('--DEBUG', action='store_true', help='Log mode set to DEBUG')

    # Training parameters (used for distributed inference)
    parser.add_argument("--distributed", default=None, type=misc_utils.str2bool_or_none,
                        help="use distributed training even if this script is not launched using torch.distributed.launch or run")
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=1024, type=int)

    # Feature Extraction Params
    parser.add_argument('--data-proc-transforms', help="Data Preprocessing transforms ", default=[])
    parser.add_argument('--feat-ext-transform', help="Feature Extraction transforms ", default=[])

    # Simple TimeSeries Params
    parser.add_argument('--variables', help="1- if Univariate, 2/3/.. if multivariate")
    parser.add_argument('--resampling-factor', help="Resampling ratio")
    parser.add_argument('--sampling-rate', help="Sampled frequency ")
    parser.add_argument('--new-sr', help="Required to subsample every nth value from the dataset")
    parser.add_argument('--stride-size', help="Fraction (0-1) that will be multiplied by frame-size to get the actual stride", type=float)

    # FFT/Feature extraction arguments
    parser.add_argument('--frame-size', help="Frame Size")
    parser.add_argument('--feature-size-per-frame', help="FFT feature size per frame")
    parser.add_argument('--num-frame-concat', help="Number of FFT frames to concat")
    parser.add_argument('--frame-skip', help="Skip frames while computing FFT")
    parser.add_argument('--min-bin', help="Remove DC Component from FFT")
    parser.add_argument('--normalize-bin', help="Normalize Binning")
    parser.add_argument('--dc-remove', help="Remove DC Component from FFT")
    parser.add_argument('--analysis-bandwidth', help="Spectrum of FFT used for binning")
    parser.add_argument('--log-base', help="base value for logarithm")
    parser.add_argument('--log-mul', help="multiplier for logarithm")
    parser.add_argument('--log-threshold', help="offset added to values for logarithmic calculation")
    parser.add_argument('--stacking', help="1D/2D1/None")
    parser.add_argument('--offset', help="Index for data overlap; 0: no overlap, n: start index for overlap")
    parser.add_argument('--scale', help="Scaling factor to input data")
    parser.add_argument('--generic-model', help="Open Source models", type=misc_utils.str_or_bool, default=False)
    parser.add_argument("--output-int", default=None, type=misc_utils.str_or_bool,
                        help="Get quantized int8 output from model (False for dequantized float output). If not specified, determined automatically based on task type and quantization level.")

    return parser


def setup_test_environment(args, task_name='test'):
    """
    Common setup for ONNX testing.

    Args:
        args: Parsed arguments
        task_name: Name of the task (for output directory naming)

    Returns:
        tuple: (logger, device)
    """
    if not args.output_dir:
        output_folder = os.path.basename(os.path.split(args.data_path)[0])
        args.output_dir = os.path.join('.', 'data', 'checkpoints', task_name, output_folder, args.model, args.date)
    utils.mkdir(args.output_dir)
    log_file = os.path.join(args.output_dir, 'run.log')
    logger = Logger(log_file=args.lis or log_file, DEBUG=args.DEBUG, name="root", append_log=True, console_log=True)
    utils.seed_everything(args.seed)

    from ..version import get_version_str
    logger.info(f"TinyVerse Toolchain Version: {get_version_str()}")
    logger.info("Script: {}".format(os.path.relpath(__file__)))

    utils.init_distributed_mode(args)
    logger.debug("Args: {}".format(args))

    device = torch.device(args.device)
    return logger, device


def prepare_transforms(args):
    """Prepare data transforms from args."""
    if isinstance(args.data_proc_transforms, list):
        if len(args.data_proc_transforms) and isinstance(args.data_proc_transforms[0], list):
            args.transforms = args.data_proc_transforms[0] + args.feat_ext_transform
        else:
            args.transforms = args.data_proc_transforms + args.feat_ext_transform


def load_onnx_model(model_path, generic_model=False):
    """
    Load ONNX model with optional decryption.

    Args:
        model_path: Path to ONNX model file
        generic_model: Whether this is a generic (unencrypted) model

    Returns:
        tuple: (ort_session, input_name, output_name)
    """
    if not generic_model:
        utils.decrypt(model_path, utils.get_crypt_key())
    ort_sess = ort.InferenceSession(model_path)
    if not generic_model:
        utils.encrypt(model_path, utils.get_crypt_key())

    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name
    return ort_sess, input_name, output_name


def run_distributed_test(main_fn, args):
    """
    Run testing with optional distributed mode.

    Args:
        main_fn: The main testing function to call
        args: Parsed arguments
    """
    if args.device != 'cpu' and args.distributed is True:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = str(args.gpus)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        torch.multiprocessing.spawn(main_fn, nprocs=args.gpus, args=(args,))
    else:
        main_fn(0, args)
