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


import datetime
import os
import platform
import random
import sys
import timeit
from argparse import ArgumentParser
from logging import getLogger

import numpy as np
import pandas as pd

from tinyml_tinyverse.common.models import NeuralNetworkWithPreprocess
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion, TinyMLQuantizationMethod

# Torch Modules
import torch
import torch.nn as nn
import torchinfo
from tabulate import tabulate

from tinyml_tinyverse.common import models
from tinyml_tinyverse.common.datasets import GenericTSDataset

# Tiny ML TinyVerse Modules
from tinyml_tinyverse.common.utils import misc_utils, utils, load_weights,gof_utils
from tinyml_tinyverse.common.utils.mdcl_utils import Logger, create_dir

dataset_loader_dict = {'GenericTSDataset' : GenericTSDataset}


def split_weights(weights_name):
    weights_list = weights_name.split(',')
    weights_urls = []
    weights_enums = []
    for w in weights_list:
        w = w.lstrip()
        if misc_utils.is_url_or_file(w):
            weights_urls.append(w)
        else:
            weights_enums.append(w)
    return ((weights_urls[0] if len(weights_urls)>0 else None), (weights_enums[0] if len(weights_enums)>0 else None))


def get_args_parser():
    """
    This function is used to process inputs given to the program
    """
    DESCRIPTION = "This script loads time series data and trains it generating a model"
    parser = ArgumentParser(description=DESCRIPTION)
    # parser.add_argument('--out_dir', help='Run directory', default=os.getcwd())
    parser.add_argument('--resampling-factor', help="Resampling ratio")
    parser.add_argument('--sampling-rate', help="Sampled frequency ", type=float, required=True)
    parser.add_argument('--new-sr', help="Required to subsample every nth value from the dataset")  # default=3009)
    #parser.add_argument('--sequence-window', help="Window length (s) to stride by")  # default=0.001)
    parser.add_argument('--stride-size', help="Window length per sequence in sec", type=float)
    parser.add_argument('--data-proc-transforms', help="Data Preprocessing transforms ", default=[])  # default=['DownSample', 'SimpleWindow'])

    parser.add_argument('--feat-ext-transform', help="Feature Extraction transforms ", default=[])
    parser.add_argument('--store-feat-ext-data', help='Store Data post Feature extractions')
    parser.add_argument('--feat-ext-store-dir', help='Store Data post Feature extractions in this directory')
    parser.add_argument('--dont-train-just-feat-ext', help='Quit after Feature Extraction without Training. Does not have any effect if --store-feat-ext-data is not used')
    parser.add_argument('--frame-size', help="Frame Size")
    parser.add_argument('--feature-size-per-frame', help="FFT feature size per frame")
    parser.add_argument('--num-frame-concat', help="Number of FFT frames to concat")
    parser.add_argument('--frame-skip', help="Skip frames while computing FFT")
    parser.add_argument('--min-bin', help="Remove DC Component from FFT")
    parser.add_argument('--normalize-bin', help="Normalize FFT Binning")
    parser.add_argument('--dc-remove', help="Remove DC Component from FFT")
    parser.add_argument('--analysis-bandwidth', help="Spectrum of FFT used for binning")
    # parser.add_argument('--num-channel', help="Number of input channels (ex.axis, phase)", default=16, type=int)
    parser.add_argument('--log-base', help="base value for logarithm")
    parser.add_argument('--log-mul', help="multiplier for logarithm")
    parser.add_argument('--log-threshold', help="offset added to values for logarithmic calculation")
    parser.add_argument('--stacking', help="1D/2D1/None")
    parser.add_argument('--offset', help="Index for data overlap; 0: no overlap, n: start index for overlap")
    parser.add_argument('--scale', help="Scaling factor to input data")
    parser.add_argument('--generic-model', help="Open Source models", type=misc_utils.str_or_bool, default=False)

    parser.add_argument('--gen_golden_vectors', help="Generate golden vectors to feed into the model", type=misc_utils.str_or_bool, default=True)

    parser.add_argument('--variables', help="1- if Univariate, 2/3/.. if multivariate")

    parser.add_argument('--data-path', default=os.path.join('.', 'data', 'datasets'), help='dataset')
    parser.add_argument('--dataset', default='folder', help='dataset')
    parser.add_argument('--gof-test', type=misc_utils.str2bool, default=False, help='Enable goodness-of-fit test') 
    parser.add_argument('--dataset-loader', default='SimpleTSDataset', help='dataset loader')
    parser.add_argument("--loader-type", default="classification", type=str,
                        help="Dataset Loader Type: classification/regression")
    parser.add_argument('--annotation-prefix', default='instances', help='annotation-prefix')
    parser.add_argument('--model', default='ArcDet4x16', help='model')
    parser.add_argument('--dual-op', default=False, help='True if you need model to have FC layer input as secondary output', type=misc_utils.str_or_bool)
    parser.add_argument('--augment-config', default=None, help='yaml file indicating augment configurations',)
    parser.add_argument('--model-config', default=None, help='yaml file indicating model configurations',)
    parser.add_argument('--model-spec', default=None, help='Model Specification. (Used for models not defined in repo)')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
    parser.add_argument('-b', '--batch-size', default=1024, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=0 if platform.system() in ['Windows'] else 16, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--label-smoothing', default=0.0, type=float, help='label smoothing (default: 0.0)', dest='label_smoothing')
    parser.add_argument('--mixup-alpha', default=0.0, type=float, help='mixup alpha (default: 0.0)')
    parser.add_argument('--cutmix-alpha', default=0.0, type=float, help='cutmix alpha (default: 0.0)')
    parser.add_argument('--lr-scheduler', default="cosineannealinglr", help='the lr scheduler (default: cosineannealinglr)')
    parser.add_argument('--lr-warmup-epochs', default=5, type=int, help='the number of epochs to warmup (default: 5)')
    parser.add_argument('--lr-warmup-method', default="constant", type=str, help='the warmup method (default: constant)')
    parser.add_argument('--lr-warmup-decay', default=0.01, type=float, help='the decay for lr')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default=None, help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument("--cache-dataset", dest="cache_dataset", help="Cache the datasets for quicker initialization. It also serializes the transforms", action="store_true",)
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true",)
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", default=None, type=misc_utils.str_or_bool,)
    parser.add_argument("--export-only", dest="export_only", help="Export onnx", action="store_true",)
    parser.add_argument('--auto-augment', default=None, help='auto augment policy (default: None)')
    parser.add_argument('--random-erase', default=0.0, type=float, help='random erasing probability (default: 0.0)')

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true', help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str, help='For apex mixed precision training O0 for FP32 training, O1 for mixed precision training.For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--distributed", default=None, type=misc_utils.str2bool_or_none, help="use dstributed training even if this script is not launched using torch.disctibuted.launch or run")

    parser.add_argument('--model-ema', action='store_true', help='enable tracking Exponential Moving Average of model parameters')
    parser.add_argument('--model-ema-decay', type=float, default=0.9, help='decay factor for Exponential Moving Average of model parameters(default: 0.9)')
    parser.add_argument('--date', default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), help='current date')
    parser.add_argument('--seed', default=42, help="Seed for all randomness", type=int)
    parser.add_argument('--lis', help='Log File', type=str,)# default=ops(opb(__file__))[0] + ".lis")
    parser.add_argument('--DEBUG', action='store_true', help='Log mode set to DEBUG')

    parser.add_argument("--compile-model", default=0, type=int, help="Compile the model using PyTorch2.0 functionality")
    parser.add_argument("--opset-version", default=17, type=int, help="ONNX Opset version")
    
    parser.add_argument("--quantization", "--quantize", dest="quantization", default=0, type=int, choices=TinyMLQuantizationVersion.get_choices(), help="Quantization Aware Training (QAT)")
    # parser.add_argument("--quantization-type", default="DEFAULT", help="Actual Quantization Flavour - applies only if quantization is enabled")
    parser.add_argument("--quantization-method", default="QAT", choices=["PTQ", "QAT"], help="Actual Quantization Flavour - applies only if quantization is enabled")
    parser.add_argument("--weight-bitwidth", default=8, type=int, choices=[8, 4, 2], help="Weight Bitwidth - applies only if quantization is enabled")
    parser.add_argument("--activation-bitwidth", default=8, type=int, help="Activation Bitwidth- applies only if quantization is enabled")

    parser.add_argument("--quantization-error-logging", default=True, type=misc_utils.str_or_bool, help="log the quantization error")

    parser.add_argument("--with-input-batchnorm", default=True, help="onnx opset 18 doesn't export input batchnorm, use this if using TINPU style QAT only")

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-state-dict-name", default="model", type=str, help="the weights member name to load from the checkpoint")
    parser.add_argument("--nn-for-feature-extraction", default=False, type=misc_utils.str2bool, help="Use an AI model for preprocessing")

    return parser

def generate_golden_vector_dir(output_dir):
    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    create_dir(golden_vectors_dir)
    return

def generate_user_input_config(output_dir, dataset):
    logger = getLogger("root.generate_user_input_config")
    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    user_input_config_h = os.path.join(golden_vectors_dir, 'user_input_config.h')
    logger.info("Creating user_input_config.h at: {}".format(user_input_config_h))

    with open(user_input_config_h, 'w') as fp:
        fp.write("#ifndef INPUT_CONFIG_H_\n")
        fp.write("#define INPUT_CONFIG_H_\n\n")
        fp.write(''.join([f'#define {flag}\n' for flag in dataset.preprocessing_flags]))
        fp.write('\n'.join([f'#define {k} {v}' for k, v in dataset.feature_extraction_params.items()]))
        fp.write("\n\n#endif /* INPUT_CONFIG_H_ */\n")
    return

def generate_test_vector(output_dir, test_vector_data):
    logger = getLogger("root.generate_test_vector")
    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    test_vector_c = os.path.join(golden_vectors_dir, 'test_vector.c')
    logger.info("Creating test_vector.c at: {}".format(test_vector_c))
    with open(test_vector_c, 'w') as fp:
        fp.write(test_vector_data)
    return

def generate_model_aux(output_dir, dataset):
    logger = getLogger("root.generate_model_aux")
    model_aux_h = os.path.join(output_dir, 'model_aux.h')
    class_list_ordered = ', '.join([f'"{dataset.inverse_label_map.get(label_index)}"' for label_index in sorted(dataset.inverse_label_map.keys())])
    logger.info("Creating model_aux.h at: {}".format(model_aux_h))
    with open(model_aux_h, 'w') as fp:
        fp.write(f'const NUMBER_OF_CLASSES = {len(dataset.classes)};\n')
        fp.write('const char *classIdToName[NUMBER_OF_CLASSES] = {' + class_list_ordered + '};')
    return

def generate_golden_vectors(output_dir, dataset, generic_model=False, nn_for_feature_extraction=False):
    logger = getLogger("root.generate_golden_vectors")
    import onnxruntime as ort
    vector_files = []
    if not generic_model:
        utils.decrypt(os.path.join(output_dir, 'model.onnx'), utils.get_crypt_key())
    ort_sess = ort.InferenceSession(os.path.join(output_dir, 'model.onnx'))
    if not generic_model:
        utils.encrypt(os.path.join(output_dir, 'model.onnx'), utils.get_crypt_key())
    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name

    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    
    logger.info(f"Creating Golden data for reference at {golden_vectors_dir}")
    label_index_dict = {dataset.inverse_label_map.get(label): np.where(dataset.Y == label)[0] for label in np.unique(dataset.Y)}

    for label, indices in label_index_dict.items():
        # For each label, 4 random golden test vectors will be selected and printed out
        for index in random.sample(list(indices), k=2):  # Originally k=4
            np_raw = dataset.X_raw[index]
            if nn_for_feature_extraction:
                np_feat = np_raw
                pred = ort_sess.run([output_name], {input_name: np.expand_dims(np_raw, 0).astype(np.float32)})[0]
            else:
                np_feat = dataset.X[index]
                pred = ort_sess.run([output_name], {input_name: np.expand_dims(np_feat, 0).astype(np.float32)})[0]

            half_path = os.path.join(golden_vectors_dir)

            # Saving as .txt
            np.savetxt(half_path + f'adc_{label}_{index}.txt', np_raw.flatten(), fmt='%.0f,', header=f'//Class: {label} (Index: {index}): ADC Data\nfloat raw_input_test[{len(np_raw.flatten())}]= {{', footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'adc_{label}_{index}.txt')
            if not nn_for_feature_extraction:
                np.savetxt(half_path + f'features_{label}_{index}.txt', np_feat.flatten(), fmt='%.5f,', header=f'//Class: {label} (Index: {index}): Extracted Features\nfloat32_t model_test_input[{len(np_feat.flatten())}] = {{', footer='}', comments='', newline=' ')
                vector_files.append(half_path + f'features_{label}_{index}.txt')
            np.savetxt(half_path + f'output_{label}_{index}.txt', pred.flatten(), fmt='%.0f,', header=f'//Class: {label} (Index: {index}): Expected Model Output\nint8_t golden_output[{len(pred.flatten())}] = {{', footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'output_{label}_{index}.txt')

    header_file_info = """#include "device.h"
// //////////////////////////////////////////////////////////////////////////////////////////////////////
// 1. Please uncomment one (and only one) of the below sets at a time. (Remove /* and */ only)
// 2. Do not uncomment random lines from random sets. It will not serve your purpose
// //////////////////////////////////////////////////////////////////////////////////////////////////////"""
    
    for i, file_path in enumerate(vector_files):
        # There are 3 vector files for each set. So First (index 0) and Third (index 2) need to have the /* and */ respectively
        if i % 3 == 0:
            if i // 3 == 0:
                # Set0 will not be commented so that the generated code can run automatically without build errors
                header_file_info += f'\n\n// SET {i // 3}'
            else:
                header_file_info += f'\n/*\n// SET {i // 3}'
        with open(file_path) as fp:
            file_array = fp.read()
            header_file_info += f'\n{file_array};\n'
        if i % 3 == 2:
            if i // 3 == 0:
                # Set0 will not be commented so that the generated code can run automatically without build errors
                header_file_info += '\n'
            else:
                header_file_info += '*/\n'
        os.remove(file_path)
    generate_user_input_config(output_dir, dataset)
    generate_test_vector(output_dir, header_file_info)
    generate_model_aux(output_dir, dataset)
    return


def main(gpu, args):
    transform = None
    if not args.output_dir:
        output_folder = os.path.basename(os.path.split(args.data_path)[0])
        args.output_dir = os.path.join('.', 'data', 'checkpoints', 'classification', output_folder, args.model, args.date)
    utils.mkdir(args.output_dir)
    log_file = os.path.join(args.output_dir, 'run.log')
    logger = Logger(log_file=args.lis or log_file, DEBUG=args.DEBUG, name="root", append_log=True if args.quantization else False, console_log=True)
    # logger = command_display(args.lis or log_file, args.DEBUG)
    utils.seed_everything(args.seed)
    logger = getLogger("root.main")
    from ..version import get_version_str
    logger.info(f"TinyVerse Toolchain Version: {get_version_str()}")
    logger.info("Script: {}".format(os.path.relpath(__file__)))

    # if args.quantization and args.store_feat_ext_data:
    #     logger.info("Avoiding storage of feature extracted data again during QAT")
    #     args.store_feat_ext_data = False

    if args.weights:
        (args.weights_url, args.weights_enum) = split_weights(args.weights)

    if args.device != 'cpu' and args.distributed is True:
        os.environ['RANK'] = str(int(os.environ['RANK'])*args.gpus + gpu) if 'RANK' in os.environ else str(gpu)
        os.environ['LOCAL_RANK'] = str(gpu)

    if args.lr_warmup_epochs > 0 and args.epochs <= args.lr_warmup_epochs:
        logger.info('Note: too less number of epochs - disabling warmup')
        args.lr_warmup_epochs = 0

    utils.init_distributed_mode(args)
    logger.debug("Args: {}".format(args))

    device = torch.device(args.device)

    # torch.backends.cudnn.benchmark = True
    if isinstance(args.data_proc_transforms, list):
        if len(args.data_proc_transforms) and isinstance(args.data_proc_transforms[0], list):
            args.transforms = args.data_proc_transforms[0] + args.feat_ext_transform  # args.data_proc_transforms is a list of lists
        else:
            args.transforms = args.data_proc_transforms + args.feat_ext_transform
    # if misc_utils.str2bool_or_none(args.store_feat_ext_data):
    dataset, dataset_test, train_sampler, test_sampler = utils.load_data(args.data_path, args, dataset_loader_dict)  # (126073, 1, 152), 126073

    # if misc_utils.str2bool_or_none(args.store_feat_ext_data):
    try:
        utils.plot_feature_components_graph(dataset, graph_type='pca', instance_type='train', output_dir=args.output_dir)
        utils.plot_feature_components_graph(dataset_test, graph_type='pca', instance_type='validation', output_dir=args.output_dir)
        # The below two lines work fine, but slows down a lot
        # plot_graph(dataset, graph_type='tsne', instance_type='train')
        # plot_graph(dataset_test, graph_type='tsne', instance_type='validation')
        if args.gof_test:
            if args.frame_size !='None':
                gof_utils.goodness_of_fit_test(frame_size=int(args.frame_size), classes_dir=args.data_path, output_dir=args.output_dir,class_names=dataset.classes)
            else:
                logger.warning(f"Goodness of Fit plots will not be generated because frame_size was not given in the YAML file.")
    
    except Exception as e:
        logger.warning(f"Feature Extraction plots will not be generated because: {e}")

    if misc_utils.str2bool(args.dont_train_just_feat_ext):
        logger.info('Exiting execution without training')
        sys.exit(0)

    # collate_fn = None
    num_classes = len(dataset.classes)

    logger.info("Loading data:")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True,
        collate_fn=utils.collate_fn, )

    logger.info("Creating model")
    # Model Creation
    variables = dataset.X.shape[1]  # n,c,h,w --> c is 1 (args.variables. However, after motor fault was supported, it concatenates 3x128 to 1x384 hence channels have been changed
    input_features = dataset.X.shape[2]
    # TODO: One solution is to see where exactly variables get used in timeseries_dataset and see if it can be made redundant there
    model = models.get_model(
        args.model, variables, num_classes, input_features=input_features, model_config=args.model_config,
        model_spec=args.model_spec, with_input_batchnorm=False if args.nn_for_feature_extraction else misc_utils.str2bool(args.with_input_batchnorm),
        dual_op=args.dual_op)
    if args.generic_model:
        # logger.info("\nModel:\n{}\n".format(model))
        logger.info(f"{torchinfo.summary(model, (1, variables, input_features, 1))}")

    if args.weights:
        if args.weights_url:
            logger.info(f"loading pretrained checkpoint for training: {args.weights_url}")
            model = load_weights.load_weights(model, args.weights_url, state_dict_name=args.weights_state_dict_name)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.export_only:
        if args.distributed is False or (args.distributed is True and int(os.environ['LOCAL_RANK']) == 0):
            utils.export_model(model, input_shape=(1, variables, input_features), output_dir=args.output_dir, opset_version=args.opset_version, quantization=args.quantization, generic_model=args.generic_model)
            return

    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=args.model_ema_decay)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint['model_ema'])
    phase = 'QuantTrain' if args.quantization else 'FloatTrain'

    logger.info("Start training")
    start_time = timeit.default_timer()
    best = dict(accuracy=0.0, f1=0, conf_matrix=dict(), epoch=None)

    # model = NeuralNetworkWithPreprocess
    if args.nn_for_feature_extraction:
        fe_model = models.FEModelLinear(dataset.X.shape[1], dataset.X_raw.shape[2], dataset.X.shape[2], with_input_batchnorm=False if args.quantization==TinyMLQuantizationVersion.QUANTIZATION_GENERIC else True).to(device)  # dataset.X_raw.shape[2],
        fe_model = NeuralNetworkWithPreprocess(fe_model, None)
        # Feature Extraction Model
        optimizer = utils.init_optimizer(fe_model, args.opt, args.lr, args.momentum, args.weight_decay)
        lr_scheduler = utils.init_lr_scheduler(
            args.lr_scheduler, optimizer, args.epochs, args.lr_warmup_epochs, args.lr_step_size, args.lr_gamma,
            args.lr_warmup_method, args.lr_warmup_decay)
        fe_model = utils.get_trained_feature_extraction_model(
            fe_model, args, data_loader, data_loader_test, device, lr_scheduler, optimizer)

        # Train this model
        model = NeuralNetworkWithPreprocess(fe_model, model)
    else:
        model = NeuralNetworkWithPreprocess(None, model)
    # Does nothing in Floating Point Training
    model = utils.quantization_wrapped_model(
        model, args.quantization, args.quantization_method, args.weight_bitwidth, args.activation_bitwidth,
        args.epochs, args.quantization_error_logging)
    
    optimizer = utils.init_optimizer(model, args.opt, args.lr, args.momentum, args.weight_decay)
    lr_scheduler = utils.init_lr_scheduler(
        args.lr_scheduler, optimizer, args.epochs, args.lr_warmup_epochs, args.lr_step_size, args.lr_gamma,
        args.lr_warmup_method, args.lr_warmup_decay)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        utils.train_one_epoch_classification(
            model, criterion, optimizer, data_loader, device, epoch, transform, args.apex, model_ema,
            print_freq=args.print_freq, phase=phase, num_classes=num_classes, dual_op=args.dual_op,
            is_ptq=True if (args.quantization_method in ['PTQ'] and args.quantization) else False,
            nn_for_feature_extraction=args.nn_for_feature_extraction)
        if not (args.quantization_method in ['PTQ'] and args.quantization):
            lr_scheduler.step()
        avg_accuracy, avg_f1, auc, avg_conf_matrix = utils.evaluate_classification(model, criterion, data_loader_test, device=device, transform=transform, phase=phase, num_classes=num_classes, dual_op=args.dual_op, nn_for_feature_extraction=args.nn_for_feature_extraction)
        if model_ema:
            avg_accuracy, avg_f1, auc, avg_conf_matrix = utils.evaluate_classification(model_ema, criterion, data_loader_test, device=device, transform=transform,
                                          log_suffix='EMA', print_freq=args.print_freq, phase=phase, dual_op=args.dual_op, nn_for_feature_extraction=args.nn_for_feature_extraction)
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            if model_ema:
                checkpoint['model_ema'] = model_ema.state_dict()
            # utils.save_on_master(
            #     checkpoint,
            #     os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            if avg_accuracy > best['accuracy']:
                logger.info(f"Epoch {epoch}: {avg_accuracy:.2f} (Val accuracy) > {best['accuracy']:.2f} (So far best accuracy). Hence updating checkpoint.pth")
                best['accuracy'] = avg_accuracy
                best['f1'] = avg_f1
                best['auc'] = auc
                best['conf_matrix'] = avg_conf_matrix
                best['epoch'] = epoch
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint.pth'))

    logger = getLogger(f"root.main.{phase}.BestEpoch")
    logger.info("")
    logger.info("Printing statistics of best epoch:")
    logger.info(f"Best Epoch: {best['epoch']}")
    logger.info(f"Acc@1 {best['accuracy']:.3f}")
    logger.info(f"F1-Score {best['f1']:.3f}")
    logger.info(f"AUC ROC Score {best['f1']:.3f}")
    logger.info("")
    # logger.info(f"Confusion Matrix:\n" + '\n'.join(
    #     [f"Ground Truth:(Class {dataset.inverse_label_map[i]}), Predicted:(Class {dataset.inverse_label_map[j]}): {int(best['conf_matrix'][i][j])}" for j in
    #      range(num_classes) for i in range(num_classes)]))
    logger.info('Confusion Matrix:\n {}'.format(tabulate(pd.DataFrame(best['conf_matrix'],
                  columns=[f"Predicted as: {x}" for x in dataset.inverse_label_map.values()],
                  index=[f"Ground Truth: {x}" for x in dataset.inverse_label_map.values()]),
                                                         headers="keys", tablefmt='grid')))
    # if args.nn_for_feature_extraction:
    #     logger.info("Feature extraction AI model is being combined with AI model")
    #     model = models.CombinedModel(fe_model, model)
    logger.info('Exporting model after training.')
    if args.distributed is False or (args.distributed is True and int(os.environ['LOCAL_RANK']) == 0):
        if args.nn_for_feature_extraction:
            example_input = next(iter(data_loader_test))[0]
            input_shape = (1,) + dataset.X_raw.shape[1:]
        else:
            example_input = next(iter(data_loader_test))[1]
            input_shape = (1,) + dataset.X.shape[1:]
        utils.export_model(
            model, input_shape=input_shape, output_dir=args.output_dir, opset_version=args.opset_version,
            quantization=args.quantization, example_input=example_input, generic_model=args.generic_model,
            remove_hooks_for_jit= True if (args.quantization_method==TinyMLQuantizationMethod.PTQ and args.quantization) else False)
    total_time = timeit.default_timer() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    if args.gen_golden_vectors:
        generate_golden_vector_dir(args.output_dir)
        generate_golden_vectors(args.output_dir, dataset, args.generic_model, args.nn_for_feature_extraction)
        
    return


def run(args):
    if args.device != 'cpu' and args.distributed is True:
        # for explanation of what is happening here, please see this:
        # https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
        # this assignment of RANK assumes a single machine, but with multiple gpus
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = str(args.gpus)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args,))
    else:
        main(0, args)
    return


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()

    # run the training.
    # if args.distributed is True is set, then this will launch distributed training
    # depending on args.gpus
    run(arguments)
