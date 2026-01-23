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
#
# Few lines are from: https://github.com/pytorch/vision
#
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
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
Base module for timeseries training scripts.
Contains common functionality shared across classification, regression,
anomaly detection, and forecasting tasks.
"""

import datetime
import os
import platform
import random
import sys
import timeit
from argparse import ArgumentParser
from logging import getLogger

import numpy as np
import onnxruntime as ort
import torchinfo
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion, TinyMLQuantizationMethod

import torch

from tinyml_tinyverse.common import models
from tinyml_tinyverse.common.utils import misc_utils, utils, load_weights
from tinyml_tinyverse.common.utils.mdcl_utils import create_dir, Logger


def split_weights(weights_name):
    """Split weights string into URLs and enum names."""
    weights_list = weights_name.split(',')
    weights_urls = []
    weights_enums = []
    for w in weights_list:
        w = w.lstrip()
        if misc_utils.is_url_or_file(w):
            weights_urls.append(w)
        else:
            weights_enums.append(w)
    return (weights_urls[0] if len(weights_urls) > 0 else None), (weights_enums[0] if len(weights_enums) > 0 else None)


def get_base_args_parser(description="This script loads time series data and trains it generating a model"):
    """
    Create argument parser with common arguments shared across all task types.
    Task-specific arguments should be added by calling add_task_specific_args().
    """
    parser = ArgumentParser(description=description)

    # Data processing arguments
    parser.add_argument('--resampling-factor', help="Resampling ratio")
    parser.add_argument('--sampling-rate', help="Sampled frequency ", type=float, required=True)
    parser.add_argument('--new-sr', help="Required to subsample every nth value from the dataset")
    parser.add_argument('--stride-size', help="Fraction (0-1) that will be multiplied by frame-size to get the actual stride", type=float)
    parser.add_argument('--data-proc-transforms', help="Data Preprocessing transforms ", default=[])
    parser.add_argument('--feat-ext-transform', help="Feature Extraction transforms ", default=[])
    parser.add_argument('--store-feat-ext-data', help='Store Data post Feature extractions')
    parser.add_argument('--feat-ext-store-dir', help='Store Data post Feature extractions in this directory')
    parser.add_argument('--dont-train-just-feat-ext', help='Quit after Feature Extraction without Training.')

    # FFT/Feature extraction arguments
    parser.add_argument('--frame-size', help="Frame Size")
    parser.add_argument('--feature-size-per-frame', help="FFT feature size per frame")
    parser.add_argument('--num-frame-concat', help="Number of FFT frames to concat")
    parser.add_argument('--frame-skip', help="Skip frames while computing FFT")
    parser.add_argument('--min-bin', help="Remove DC Component from FFT")
    parser.add_argument('--normalize-bin', help="Normalize FFT Binning")
    parser.add_argument('--dc-remove', help="Remove DC Component from FFT")
    parser.add_argument('--analysis-bandwidth', help="Spectrum of FFT used for binning")
    parser.add_argument('--log-base', help="base value for logarithm")
    parser.add_argument('--log-mul', help="multiplier for logarithm")
    parser.add_argument('--log-threshold', help="offset added to values for logarithmic calculation")
    parser.add_argument('--stacking', help="1D/2D1/None")
    parser.add_argument('--offset', help="Index for data overlap; 0: no overlap, n: start index for overlap")
    parser.add_argument('--scale', help="Scaling factor to input data")

    # Model arguments
    parser.add_argument('--generic-model', help="Open Source models", type=misc_utils.str_or_bool, default=False)
    parser.add_argument('--gen_golden_vectors', help="Generate golden vectors to feed into the model", type=misc_utils.str_or_bool, default=True)
    parser.add_argument('--variables', help="Column selection (0-indexed after time removal): int (first n), list of ints [0,2,4], or column names ['x','y']")

    # Dataset arguments
    parser.add_argument('--data-path', default=os.path.join('.', 'data', 'datasets'), help='dataset')
    parser.add_argument('--dataset', default='folder', help='dataset')
    parser.add_argument('--dataset-loader', default='SimpleTSDataset', help='dataset loader')
    parser.add_argument("--loader-type", default="classification", type=str,
                        help="Dataset Loader Type: classification/regression/forecasting")
    parser.add_argument('--annotation-prefix', default='instances', help='annotation-prefix')
    parser.add_argument('--model', default='ArcDet4x16', help='model')
    parser.add_argument('--dual-op', default=False, help='True if you need model to have FC layer input as secondary output', type=misc_utils.str_or_bool)
    parser.add_argument('--augment-config', default=None, help='yaml file indicating augment configurations')
    parser.add_argument('--model-config', default=None, help='yaml file indicating model configurations')
    parser.add_argument('--model-spec', default=None, help='Model Specification. (Used for models not defined in repo)')

    # Training arguments
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
    parser.add_argument('-b', '--batch-size', default=1024, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=0 if platform.system() in ['Windows'] else 8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float, metavar='W',
                        help='weight decay (default: 4e-5)', dest='weight_decay')
    parser.add_argument('--label-smoothing', default=0.0, type=float, help='label smoothing (default: 0.0)',
                        dest='label_smoothing')
    parser.add_argument('--mixup-alpha', default=0.0, type=float, help='mixup alpha (default: 0.0)')
    parser.add_argument('--cutmix-alpha', default=0.0, type=float, help='cutmix alpha (default: 0.0)')

    # Learning rate scheduler arguments
    parser.add_argument('--lr-scheduler', default="cosineannealinglr", help='the lr scheduler (default: cosineannealinglr)')
    parser.add_argument('--lr-warmup-epochs', default=5, type=int, help='the number of epochs to warmup (default: 5)')
    parser.add_argument('--lr-warmup-method', default="constant", type=str, help='the warmup method (default: constant)')
    parser.add_argument('--lr-warmup-decay', default=0.01, type=float, help='the decay for lr')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')

    # Output and logging arguments
    parser.add_argument('--print-freq', default=None, type=int, help='print frequency')
    parser.add_argument('--output-dir', default=None, help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument("--cache-dataset", dest="cache_dataset",
                        help="Cache the datasets for quicker initialization.", action="store_true")
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo",
                        default=None, type=misc_utils.str_or_bool)
    parser.add_argument("--export-only", dest="export_only", help="Export onnx", action="store_true")
    parser.add_argument('--auto-augment', default=None, help='auto augment policy (default: None)')
    parser.add_argument('--random-erase', default=0.0, type=float, help='random erasing probability (default: 0.0)')

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true', help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training O0 for FP32 training, O1 for mixed precision training.')

    # Distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--distributed", default=None, type=misc_utils.str2bool_or_none,
                        help="use distributed training even if this script is not launched using torch.distributed.launch or run")

    # EMA and misc arguments
    parser.add_argument('--model-ema', action='store_true',
                        help='enable tracking Exponential Moving Average of model parameters')
    parser.add_argument('--model-ema-decay', type=float, default=0.9,
                        help='decay factor for Exponential Moving Average of model parameters(default: 0.9)')
    parser.add_argument('--date', default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), help='current date')
    parser.add_argument('--seed', default=42, help="Seed for all randomness", type=int)
    parser.add_argument('--lis', help='Log File', type=str)
    parser.add_argument('--DEBUG', action='store_true', help='Log mode set to DEBUG')

    # Model compilation and export arguments
    parser.add_argument("--compile-model", default=0, type=int, help="Compile the model using PyTorch2.0 functionality")
    parser.add_argument("--opset-version", default=17, type=int, help="ONNX Opset version")

    # Quantization arguments
    parser.add_argument("--quantization", "--quantize", dest="quantization", default=0, type=int,
                        choices=TinyMLQuantizationVersion.get_choices(), help="Quantization Aware Training (QAT)")
    parser.add_argument("--quantization-method", default="QAT", choices=["PTQ", "QAT"],
                        help="Actual Quantization Flavour - applies only if quantization is enabled")
    parser.add_argument("--weight-bitwidth", default=8, type=int, choices=[16, 8, 4, 2],
                        help="Weight Bitwidth - applies only if quantization is enabled")
    parser.add_argument("--activation-bitwidth", default=8, type=int,
                        help="Activation Bitwidth- applies only if quantization is enabled")
    parser.add_argument("--quantization-error-logging", default=True, type=misc_utils.str_or_bool,
                        help="log the quantization error")

    # Weights loading arguments
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-state-dict-name", default="model", type=str,
                        help="the weights member name to load from the checkpoint")
    parser.add_argument("--output-int", default=None, type=misc_utils.str_or_bool,
                        help="Get quantized int8 output from model (False for dequantized float output). If not specified, determined automatically based on task type and quantization level.")
    parser.add_argument("--ondevice-training", default=False, type=misc_utils.str2bool,
                        help="Specified whether the current model can be trained on device or not")
    parser.add_argument('--trainable_layers_from_last', default=1, type=int,
                        help='Number of trainable layers from end for on-device training (k)')
    parser.add_argument("--partial-quantization", default=False, type=misc_utils.str2bool,
                        help="Specified whether the current model can use partial quantization or not")
    return parser


def generate_golden_vector_dir(output_dir):
    """Create golden vectors directory."""
    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    create_dir(golden_vectors_dir)
    return golden_vectors_dir


def generate_user_input_config(output_dir, dataset, extra_defines=None):
    """
    Generate user_input_config.h file with preprocessing flags.

    Args:
        output_dir: Output directory path
        dataset: Dataset object with preprocessing_flags and feature_extraction_params
        extra_defines: Optional dict of additional #define statements to add
    """
    logger = getLogger("root.generate_user_input_config")
    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    user_input_config_h = os.path.join(golden_vectors_dir, 'user_input_config.h')
    logger.info("Creating user_input_config.h at: {}".format(user_input_config_h))

    with open(user_input_config_h, 'w') as fp:
        fp.write("#ifndef INPUT_CONFIG_H_\n")
        fp.write("#define INPUT_CONFIG_H_\n\n")
        fp.write(''.join([f'#define {flag}\n' for flag in dataset.preprocessing_flags]))
        fp.write('\n'.join([f'#define {k} {v}' for k, v in dataset.feature_extraction_params.items()]))
        if extra_defines:
            for key, value in extra_defines.items():
                fp.write(f'\n#define {key} {value}\n')
        fp.write("\n\n#endif /* INPUT_CONFIG_H_ */\n")
    return


def generate_test_vector(output_dir, test_vector_data):
    """Generate test_vector.c file with golden vector data."""
    logger = getLogger("root.generate_test_vector")
    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    test_vector_c = os.path.join(golden_vectors_dir, 'test_vector.c')
    logger.info("Creating test_vector.c at: {}".format(test_vector_c))
    with open(test_vector_c, 'w') as fp:
        fp.write(test_vector_data)
    return


def generate_model_aux(output_dir, dataset):
    """Generate model_aux.h file with class information."""
    logger = getLogger("root.generate_model_aux")
    model_aux_h = os.path.join(output_dir, 'model_aux.h')
    class_list_ordered = ', '.join([f'"{dataset.inverse_label_map.get(label_index)}"'
                                     for label_index in sorted(dataset.inverse_label_map.keys())])
    logger.info("Creating model_aux.h at: {}".format(model_aux_h))
    with open(model_aux_h, 'w') as fp:
        fp.write(f'const NUMBER_OF_CLASSES = {len(dataset.classes)};\n')
        fp.write('const char *classIdToName[NUMBER_OF_CLASSES] = {' + class_list_ordered + '};')
    return


def load_datasets(data_path, args, dataset_loader_dict):
    """Load train and test datasets."""
    dataset, dataset_test, train_sampler, test_sampler = utils.load_data(data_path, args, dataset_loader_dict)
    return dataset, dataset_test, train_sampler, test_sampler


def run_distributed(main_fn, args):
    """
    Run training with optional distributed mode.

    Args:
        main_fn: The main training function to call
        args: Parsed arguments
    """
    if args.device != 'cpu' and args.distributed is True:
        # for explanation of what is happening here, please see this:
        # https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
        # this assignment of RANK assumes a single machine, but with multiple gpus
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = str(args.gpus)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        torch.multiprocessing.spawn(main_fn, nprocs=args.gpus, args=(args,))
    else:
        main_fn(0, args)
    return


def create_golden_vectors_base(output_dir, dataset, output_int, generic_model, ort_sess,
                                sample_indices, files_per_set=3, include_features=True):
    """
    Base function to create golden vectors.

    Args:
        output_dir: Output directory
        dataset: Dataset object
        output_int: Whether output should be int format
        generic_model: Whether using generic model
        ort_sess: ONNX runtime session
        sample_indices: Indices to sample from dataset
        files_per_set: Number of files per set (2 or 3)
        include_features: Whether to include feature vectors

    Returns:
        List of generated vector file paths
    """
    logger = getLogger("root.generate_golden_vectors")
    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name
    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')

    logger.info(f"Creating Golden data for reference at {golden_vectors_dir}")
    vector_files = []

    for index in sample_indices:
        np_raw = dataset.X_raw[index]
        np_feat = dataset.X[index]
        pred = ort_sess.run([output_name], {input_name: np.expand_dims(np_feat, 0).astype(np.float32)})[0]
        half_path = os.path.join(golden_vectors_dir)

        # ADC/Raw data
        np.savetxt(half_path + f'adc_{index}.txt', np_raw.flatten(),
                   fmt='%f,' if np_raw.dtype.kind == 'f' else '%d,',
                   header=f'//(Index: {index}): ADC Data\nfloat raw_input_test[{len(np_raw.flatten())}]= {{',
                   footer='}', comments='', newline=' ')
        vector_files.append(half_path + f'adc_{index}.txt')

        # Features (optional)
        if include_features:
            np.savetxt(half_path + f'features_{index}.txt', np_feat.flatten(),
                       fmt='%.5f,',
                       header=f'//(Index: {index}): Extracted Features\nfloat model_test_input[{len(np_feat.flatten())}] = {{',
                       footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'features_{index}.txt')

        # Output
        np.savetxt(half_path + f'output_{index}.txt', pred.flatten(),
                   fmt='%d,' if output_int else '%f,',
                   header=f'//(Index: {index}): Expected Model Output\n{"int8_t" if output_int else "float"} golden_output[{len(pred.flatten())}] = {{',
                   footer='}', comments='', newline=' ')
        vector_files.append(half_path + f'output_{index}.txt')

    return vector_files


def assemble_golden_vectors_header(vector_files, files_per_set=3):
    """
    Assemble golden vector files into a single header file.

    Args:
        vector_files: List of vector file paths
        files_per_set: Number of files per set (2 or 3)

    Returns:
        Header file content as string
    """
    header_file_info = """#if defined(__C29__) || defined(__TMS320C2000__)
    #include "device.h"
#else
    #include <stdint.h>
#endif

// //////////////////////////////////////////////////////////////////////////////////////////////////////
// 1. Please uncomment one (and only one) of the below sets at a time. (Remove /* and */ only)
// 2. Do not uncomment random lines from random sets. It will not serve your purpose
// //////////////////////////////////////////////////////////////////////////////////////////////////////"""

    for i, file_path in enumerate(vector_files):
        if i % files_per_set == 0:
            if i // files_per_set == 0:
                # Set0 will not be commented so that the generated code can run automatically without build errors
                header_file_info += f'\n\n// SET {i // files_per_set}'
            else:
                header_file_info += f'\n/*\n// SET {i // files_per_set}'
        with open(file_path) as fp:
            file_array = fp.read()
            header_file_info += f'\n{file_array};\n'
        if i % files_per_set == (files_per_set - 1):
            if i // files_per_set == 0:
                header_file_info += '\n'
            else:
                header_file_info += '*/\n'
        os.remove(file_path)

    return header_file_info


def setup_training_environment(args, gpu, task_name, script_file):
    """
    Set up the training environment: output directory, logger, distributed mode, etc.

    Args:
        args: Parsed arguments
        gpu: GPU index
        task_name: Name of the task (classification, regression, etc.)
        script_file: Path to the script file (__file__ from calling module)

    Returns:
        tuple: (logger, device)
    """
    if not args.output_dir:
        output_folder = os.path.basename(os.path.split(args.data_path)[0])
        args.output_dir = os.path.join('.', 'data', 'checkpoints', task_name, output_folder, args.model, args.date)
    utils.mkdir(args.output_dir)
    log_file = os.path.join(args.output_dir, 'run.log')
    Logger(log_file=args.lis or log_file, DEBUG=args.DEBUG, name="root",
           append_log=True if args.quantization else False, console_log=True)
    logger = getLogger("root.main")
    utils.seed_everything(args.seed)

    from ..version import get_version_str
    logger.info(f"TinyVerse Toolchain Version: {get_version_str()}")
    logger.info("Script: {}".format(os.path.relpath(script_file)))

    if args.weights:
        (args.weights_url, args.weights_enum) = split_weights(args.weights)

    if args.device != 'cpu' and args.distributed is True:
        os.environ['RANK'] = str(int(os.environ['RANK']) * args.gpus + gpu) if 'RANK' in os.environ else str(gpu)
        os.environ['LOCAL_RANK'] = str(gpu)

    if args.lr_warmup_epochs > 0 and args.epochs <= args.lr_warmup_epochs:
        logger.info('Note: too less number of epochs - disabling warmup')
        args.lr_warmup_epochs = 0

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


def create_model(args, variables, num_classes, input_features, logger):
    """
    Create and configure the model.

    Args:
        args: Parsed arguments
        variables: Number of input variables
        num_classes: Number of output classes
        input_features: Number of input features
        logger: Logger instance

    Returns:
        model: The created model
    """
    model = models.get_model(
        args.model, variables, num_classes, input_features=input_features, model_config=args.model_config,
        model_spec=args.model_spec,
        dual_op=args.dual_op)
    return model


def log_model_summary(model, args, variables, input_features, logger):
    """Log model summary using torchinfo."""
    if args.generic_model:
        try:
            if not args.quantization:
                logger.info(f"{torchinfo.summary(model, (1, variables, input_features, 1))}")
        except UnicodeEncodeError as e:
            logger.warning(f"Model Information/summary could not be provided because of {e}")


def load_pretrained_weights(model, args, logger):
    """Load pretrained weights if specified."""
    if args.weights and args.weights_url:
        logger.info(f"loading pretrained checkpoint for training: {args.weights_url}")
        model = load_weights.load_weights(model, args.weights_url, state_dict_name=args.weights_state_dict_name)
    return model


def setup_optimizer_and_scheduler(model, args):
    """
    Set up optimizer and learning rate scheduler.

    Args:
        model: The model to optimize
        args: Parsed arguments

    Returns:
        tuple: (optimizer, lr_scheduler)
    """
    optimizer = utils.init_optimizer(model, args.opt, args.lr, args.momentum, args.weight_decay)
    lr_scheduler = utils.init_lr_scheduler(
        args.lr_scheduler, optimizer, args.epochs, args.lr_warmup_epochs, args.lr_step_size, args.lr_gamma,
        args.lr_warmup_method, args.lr_warmup_decay)
    return optimizer, lr_scheduler


def setup_distributed_model(model, args, device):
    """
    Set up model for distributed training and EMA.

    Args:
        model: The model
        args: Parsed arguments
        device: Torch device

    Returns:
        tuple: (model, model_without_ddp, model_ema)
    """
    model_without_ddp = model

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=args.model_ema_decay)

    return model, model_without_ddp, model_ema


def resume_from_checkpoint(model_without_ddp, optimizer, lr_scheduler, model_ema, args):
    """
    Resume training from checkpoint.

    Args:
        model_without_ddp: Model without DDP wrapper
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        model_ema: EMA model (or None)
        args: Parsed arguments

    Returns:
        Updated args with start_epoch
    """
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint['model_ema'])
    return args


def handle_export_only(model, args, variables, input_features, logger):
    """
    Handle export_only flag - export model and exit if set.

    Args:
        model: The model to export
        args: Parsed arguments
        variables: Number of input variables
        input_features: Number of input features
        logger: Logger instance

    Returns:
        bool: True if export_only was handled (should return from main), False otherwise
    """
    if args.export_only:
        if args.distributed is False or (args.distributed is True and int(os.environ['LOCAL_RANK']) == 0):
            utils.export_model(model, input_shape=(1, variables, input_features), output_dir=args.output_dir,
                               opset_version=args.opset_version, quantization=args.quantization,
                               generic_model=args.generic_model)
        return True
    return False


def move_model_to_device(model, device, logger):
    """
    Move model to device with error handling.

    Args:
        model: The model
        device: Torch device
        logger: Logger instance

    Returns:
        bool: True if successful, exits otherwise
    """
    try:
        model.to(device)
        return True
    except AssertionError as e:
        logger.error(f"Input options have asked to run on GPU, but no GPU was found. "
                     f"Either change num_gpus to 0 or verify that your GPU works. Error raised: {e}")
        sys.exit(1)


def save_checkpoint(model_without_ddp, optimizer, lr_scheduler, epoch, args, model_ema=None, extra_data=None):
    """
    Save training checkpoint.

    Args:
        model_without_ddp: Model without DDP wrapper
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        epoch: Current epoch
        args: Parsed arguments
        model_ema: EMA model (or None)
        extra_data: Additional data to save in checkpoint

    Returns:
        dict: The checkpoint dictionary
    """
    checkpoint = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args
    }
    if model_ema:
        checkpoint['model_ema'] = model_ema.state_dict()
    if extra_data:
        checkpoint.update(extra_data)
    return checkpoint


def export_trained_model(model, args, dataset, example_input=None, input_shape=None):
    """
    Export model after training.

    Args:
        model: The trained model
        args: Parsed arguments
        dataset: Dataset (for shape info)
        example_input: Optional example input tensor
        input_shape: Optional input shape tuple (overrides dataset shape)
    """
    logger = getLogger("root.main")
    logger.info('Exporting model after training.')
    if args.distributed is False or (args.distributed is True and int(os.environ['LOCAL_RANK']) == 0):
        if input_shape is None:
            input_shape = (1,) + dataset.X.shape[1:]
        utils.export_model(
            model, input_shape=input_shape, output_dir=args.output_dir, opset_version=args.opset_version,
            quantization=args.quantization, example_input=example_input, generic_model=args.generic_model,
            remove_hooks_for_jit=True if (args.quantization_method == TinyMLQuantizationMethod.PTQ and args.quantization) else False)


def log_training_time(start_time):
    """Log total training time."""
    logger = getLogger("root.main")
    total_time = timeit.default_timer() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    return total_time


def compute_default_output_int(task_category, quantization):
    """
    Compute the default output_int value based on task category and quantization level.

    This implements the same logic as get_skip_normalize_and_output_int() from constants.py:
    - For quantization=0 (float models): output_int=False
    - For quantization=1,2 (quantized models):
        - output_int=True for classification tasks (timeseries and image)
        - output_int=False for regression, forecasting, and anomaly detection

    Args:
        task_category: Task category string (e.g., 'timeseries_classification', 'image_classification')
        quantization: Quantization level (0, 1, or 2)

    Returns:
        bool: The default output_int value
    """
    # Default for float models (quantization = 0)
    if quantization == 0:
        return False

    # For quantized models (quantization = 1 or 2)
    if quantization in [1, 2]:
        # output_int is True only for classification tasks
        return task_category in ['timeseries_classification', 'image_classification']

    # Fallback for unexpected quantization values
    return False


def apply_output_int_default(args, task_category):
    """
    Apply default output_int value if not specified by user.

    Args:
        args: Parsed command line arguments
        task_category: Task category string
    """
    if args.output_int is None:
        args.output_int = compute_default_output_int(task_category, args.quantization)


def get_output_int_flag(args):
    """Determine if output should be int format for golden vectors."""
    return args.quantization == TinyMLQuantizationVersion.QUANTIZATION_TINPU and args.output_int


def load_onnx_for_inference(output_dir, generic_model=False):
    """
    Load ONNX model for inference (e.g., golden vector generation).

    Args:
        output_dir: Output directory containing model.onnx
        generic_model: Whether this is a generic (unencrypted) model

    Returns:
        tuple: (ort_session, input_name, output_name)
    """
    model_path = os.path.join(output_dir, 'model.onnx')
    if not generic_model:
        utils.decrypt(model_path, utils.get_crypt_key())
    ort_sess = ort.InferenceSession(model_path)
    if not generic_model:
        utils.encrypt(model_path, utils.get_crypt_key())

    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name
    return ort_sess, input_name, output_name


def create_data_loaders(dataset, dataset_test, train_sampler, test_sampler, args, gpu=0):
    """
    Create train and test data loaders.

    Args:
        dataset: Training dataset
        dataset_test: Test dataset
        train_sampler: Training sampler
        test_sampler: Test sampler
        args: Parsed arguments
        gpu: GPU index (for pin_memory)

    Returns:
        tuple: (data_loader, data_loader_test)
    """
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers,
        pin_memory=True if gpu > 0 else False, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers,
        pin_memory=True if gpu > 0 else False, collate_fn=utils.collate_fn)
    return data_loader, data_loader_test
