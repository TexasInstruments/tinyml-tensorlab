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


from argparse import ArgumentParser
import datetime
from glob import glob
from logging import getLogger
import os
import time
import random
import sys
import pandas as pd
import onnxruntime as ort
import torcheval
import numpy as np
from tabulate import tabulate
# Torch Modules
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchinfo
# Tiny ML TinyVerse Modules
from tinyml_tinyverse.common.utils import utils, misc_utils
from tinyml_tinyverse.common.utils.mdcl_utils import command_display, Logger, create_dir
from tinyml_tinyverse.common.datasets import *
from tinyml_tinyverse.common import models
from edgeai_torchmodelopt.xnn.utils import is_url_or_file, load_weights
import tinyml_torchmodelopt

from tinyml_tinyverse.common.utils.utils import get_confusion_matrix
import pdb
dataset_loader_dict = {'SimpleTSDataset': SimpleTSDataset, 'ArcFaultDataset': ArcFaultDataset, 'MotorFaultDataset': MotorFaultDataset}


def split_weights(weights_name):
    weights_list = weights_name.split(',')
    weights_urls = []
    weights_enums = []
    for w in weights_list:
        w = w.lstrip()
        if is_url_or_file(w):
            weights_urls.append(w)
        else:
            weights_enums.append(w)
        #
    #
    return ((weights_urls[0] if len(weights_urls)>0 else None), (weights_enums[0] if len(weights_enums)>0 else None))


def get_args_parser():
    """
    This function is used to process inputs given to the program
    """
    DESCRIPTION = "This script loads time series data and trains it generating a model"
    parser = ArgumentParser(description=DESCRIPTION)
    # parser.add_argument('--out_dir', help='Run directory', default=os.getcwd())
    parser.add_argument('--resampling-factor', help="Resampling ratio")
    parser.add_argument('--org-sr', help="Sampled frequency ", type=float, required=True)
    parser.add_argument('--new-sr', help="Required to subsample every nth value from the dataset")  # default=3009)
    parser.add_argument('--stride_window', help="Window length (s) to stride by", type=float)  # default=0.001)
    parser.add_argument('--sequence_window', help="Window length per sequence in sec", type=float, required=True)
    parser.add_argument('--data-proc-transforms', help="Data Preprocessing transforms ", default=[],
                        nargs='+')  # default=['DownSample', 'SimpleWindow'])
    parser.add_argument('--generic-model', help="Open Source models", type=misc_utils.str_or_bool, default=False)

    parser.add_argument('--feat-ext-transform', help="Feature Extraction transforms ", default='', )
    parser.add_argument('--store-feat-ext-data', help='Store Data post Feature extractions', type=misc_utils.str_or_bool, default=False)
    parser.add_argument('--feat-ext-store-dir', help='Store Data post Feature extractions in this directory', type=str)
    parser.add_argument('--dont-train-just-feat-ext', help='Quit after Feature Extraction without Training. Does not have any effect if --store-feat-ext-data is not used', type=misc_utils.str_or_bool, default=False)
    parser.add_argument('--frame-size', help="Frame Size", default=1024, type=int)
    parser.add_argument('--feature-size-per-frame', help="FFT feature size per frame", default=512,  type=int)
    parser.add_argument('--num-frame-concat', help="Number of FFT frames to concat", default=1, type=int)
    parser.add_argument('--frame-skip', help="Skip frames while computing FFT", default=1, type=int)
    parser.add_argument('--min-fft-bin', help="Remove DC Component from FFT", default=1, type=int)
    parser.add_argument('--fft-bin-size', help="FFT Bin Size", default=2, type=int)
    parser.add_argument('--dc-remove', help="Remove DC Component from FFT", default=True, type=bool )
    parser.add_argument('--num-channel', help="Number of input channels (ex.axis, phase)", default=16, type=int)
    parser.add_argument('--stacking', help="1D/2D1/None", default=None, type=str)
    parser.add_argument('--offset', help="Index for data overlap; 0: no overlap, n: start index for overlap", default=0, type=int)
    parser.add_argument('--scale', help="Scaling factor to input data", default=1, type=float)

    parser.add_argument('--gen_golden_vectors', help="Generate golden vectors to feed into the model", type=misc_utils.str_or_bool, default=True)

    parser.add_argument('--variables', help="1- if Univariate, 2/3/.. if multivariate", type=int, default=1)

    parser.add_argument('--data-path', default='./data/datasets/', help='dataset')
    parser.add_argument('--dataset', default='folder', help='dataset')
    parser.add_argument('--dataset-loader', default='SimpleTSDataset', help='dataset loader')
    parser.add_argument('--annotation-prefix', default='instances', help='annotation-prefix')
    parser.add_argument('--model', default='ArcDet4x16', help='model')
    parser.add_argument('--model-config', default=None, help='yaml file indicating model configurations',)
    parser.add_argument('--model-spec', default=None, help='Model Specification. (Used for models not defined in repo)')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
    parser.add_argument('-b', '--batch-size', default=1024, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--label-smoothing', default=0.0, type=float,
                        help='label smoothing (default: 0.0)',
                        dest='label_smoothing')
    parser.add_argument('--mixup-alpha', default=0.0, type=float, help='mixup alpha (default: 0.0)')
    parser.add_argument('--cutmix-alpha', default=0.0, type=float, help='cutmix alpha (default: 0.0)')
    parser.add_argument('--lr-scheduler', default="cosineannealinglr", help='the lr scheduler (default: cosineannealinglr)')
    parser.add_argument('--lr-warmup-epochs', default=5, type=int, help='the number of epochs to warmup (default: 5)')
    parser.add_argument('--lr-warmup-method', default="constant", type=str,
                        help='the warmup method (default: constant)')
    parser.add_argument('--lr-warmup-decay', default=0.01, type=float, help='the decay for lr')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default=None, help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_onnx_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        default=None,
        type=misc_utils.str_or_bool,
    )
    parser.add_argument(
        "--export-only",
        dest="export_only",
        help="Export onnx",
        action="store_true",
    )
    parser.add_argument('--auto-augment', default=None, help='auto augment policy (default: None)')
    parser.add_argument('--random-erase', default=0.0, type=float, help='random erasing probability (default: 0.0)')

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--distributed", default=None, type=misc_utils.str2bool_or_none,
                        help="use dstributed training even if this script is not launched using torch.disctibuted.launch or run")

    parser.add_argument(
        '--model-ema', action='store_true',
        help='enable tracking Exponential Moving Average of model parameters')
    parser.add_argument(
        '--model-ema-decay', type=float, default=0.9,
        help='decay factor for Exponential Moving Average of model parameters(default: 0.9)')
    parser.add_argument('--date', default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), help='current date')
    parser.add_argument('--seed', default=42, help="Seed for all randomness", type=int)
    parser.add_argument('--lis', help='Log File', type=str,)# default=ops(opb(__file__))[0] + ".lis")
    parser.add_argument('--DEBUG', action='store_true', help='Log mode set to DEBUG')

    parser.add_argument("--compile-model", default=0, type=int, help="Compile the model using PyTorch2.0 functionality")
    parser.add_argument("--opset-version", default=17, type=int, help="ONNX Opset version")
    
    parser.add_argument("--quantization", "--quantize", dest="quantization", default=0, type=int, choices=tinyml_torchmodelopt.quantization.TinyMLQuantizationVersion.get_choices(), help="Quantization Aware Training (QAT)")
    parser.add_argument("--quantization-type", default="DEFAULT", help="Actual Quantization Flavour - applies only if quantization is enabled") 
    parser.add_argument("--quantization-error-logging", default=1, type=bool, help="log the quantization error")
    
    parser.add_argument("--with-input-batchnorm", default=True, help="onnx opset 18 doesn't export input batchnorm, use this if using TINPU style QAT only")

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-state-dict-name", default="model", type=str, help="the weights member name to load from the checkpoint")
    
    return parser


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    if batch.ndim == 3:
        return batch.permute(0, 2, 1)
    else:
        return batch


def all_tensors_have_same_dimensions(tensors):
    """Checks if all tensors in a list are of the same dimensions.

    Args:
    tensors: A list of tensors.

    Returns:
    True if all tensors in the list are of the same dimensions, False otherwise.
    """
    # Check if the list is empty.
    if not tensors:
        return True
    # Get the dimensions of the first tensor.
    first_tensor_dimensions = tensors[0].shape
    # Check if the dimensions of all other tensors match the dimensions of the first tensor.
    for tensor in tensors[1:]:
        if tensor.shape != first_tensor_dimensions:
            return False
    # If all tensors have the same dimensions, return True.
    return True


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    # for waveform, _, label, *_ in batch:
    for sequence, label in batch:
        tensors += [sequence]
        targets += [torch.tensor(label)]
    # Group the list of tensors into a batched tensor
    if all_tensors_have_same_dimensions(tensors):
        tensors = torch.stack((tensors))  # TODO: Is this correct
    else:
        tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "audio_classification", "datasets", "audiofolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(datadir, args):
    # Data loading code
    logger = getLogger("root.load_data")
    logger.info("Loading data")
    dataset_loader = dataset_loader_dict.get(args.dataset_loader)

    st = time.time()
    if args.test_onnx_only:
        # datadir is supposed to be test dir
        if args.dataset == 'modelmaker':
            test_folders = os.path.normpath(datadir).split(os.sep)
            test_anno = glob(
                os.path.join(os.sep.join(test_folders[:-1]), 'annotations', f'{args.annotation_prefix}_test*_list.txt'))
            test_list = test_anno[0] if len(test_anno) == 1 and os.path.exists(test_anno[0]) else None
            dataset_test = dataset_loader("test", dataset_dir=args.data_path, validation_list=test_list, **vars(args)).prepare(**vars(args))
        else:
            # dataset_test = torchvision.datasets.ImageFolder(datadir, val_transform)
            dataset_test = dataset_loader("test", dataset_dir=args.data_path, **vars(args)).prepare(**vars(args))
        logger.info("Loading Test/Evaluation data")
        logger.info('Test Data: target count: {} : Split Up: {}'.format(len(dataset_test.Y), ';\t'.join([
            f"{[f'{label_name}({label_index})' for label_name, label_index in dataset_test.label_map.items() if label_index == i][0]}:"
            f" {len(np.where(dataset_test.Y == i)[0])} " for i in np.unique(dataset_test.Y)])))
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        logger.info("Took {0:.2f} seconds".format(time.time() - st))

        return dataset_test, dataset_test, test_sampler, test_sampler

    logger.info("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(datadir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        logger.info("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)

        if args.dataset == 'modelmaker':
            train_folders = os.path.normpath(datadir).split(os.sep)
            train_anno = glob(os.path.join(os.sep.join(train_folders[:-1]), 'annotations', f'{args.annotation_prefix}_train*_list.txt'))
            training_list = train_anno[0] if len(train_anno)==1 and os.path.exists(train_anno[0]) else None
            dataset = dataset_loader("training", dataset_dir=args.data_path, training_list=training_list, **vars(args)).prepare(**vars(args))
        else:
            dataset = dataset_loader("training", dataset_dir=args.data_path, **vars(args)).prepare(**vars(args))
        if args.cache_dataset:
            logger.info("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, datadir), cache_path)
    logger.info("Took {0:.2f} seconds".format(time.time() - st))

    logger.info("Loading validation data")
    st = time.time()
    cache_path = _get_cache_path(datadir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        logger.info("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        # val_transform = presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size,
        #                                      interpolation=interpolation,
        #                                      image_mean=args.image_mean, image_scale=args.image_scale)
        if args.dataset == 'modelmaker':
            val_folders = os.path.normpath(datadir).split(os.sep)
            val_anno = glob(os.path.join(os.sep.join(val_folders[:-1]), 'annotations', f'{args.annotation_prefix}_val*_list.txt'))
            val_list = val_anno[0] if len(val_anno)==1 and os.path.exists(val_anno[0]) else None
            dataset_test = dataset_loader("val", dataset_dir=args.data_path, validation_list=val_list, **vars(args)).prepare(**vars(args))
        else:
            dataset_test = dataset_loader("val", dataset_dir=args.data_path, **vars(args)).prepare(**vars(args))
        # TODO: Add utils and uncomment the if block
        # if args.cache_dataset:
        #     logger.info("Saving dataset_test to {}".format(cache_path))
        #     utils.mkdir(os.path.dirname(cache_path))
        #     utils.save_on_master((dataset_test, datadir), cache_path)
    logger.info("Took {:.2f} seconds".format(time.time() - st))
    logger.info("\nCreating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        logger.info('Train Data: target count: {} : Split Up: {}'.format(len(dataset.Y), ';\t'.join(
            [f"{[f'{label_name}({label_index})' for label_name, label_index in dataset.label_map.items() if label_index == i][0]}:"
             f" {len(np.where(dataset.Y == i)[0])} " for i in np.unique(dataset.Y)])))
        logger.info('Val Data: target count: {} : Split Up: {}'.format(len(dataset_test.Y), ';\t'.join(
            [f"{[f'{label_name}({label_index})' for label_name, label_index in dataset_test.label_map.items() if label_index == i][0]}:"
             f" {len(np.where(dataset_test.Y == i)[0])} " for i in np.unique(dataset_test.Y)])))
        # logger.critical('target train 0/1: {}/{} {}'.format(len(np.where(dataset.Y == np.unique(dataset.Y)[0])[0]), len(np.where(dataset.Y == np.unique(dataset.Y)[1])[0]), len(dataset.Y)))
        class_sample_count = np.array([len(np.where(dataset.Y == t)[0]) for t in np.unique(dataset.Y)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in np.array(dataset.Y).astype(int)])
        samples_weight = torch.from_numpy(samples_weight)
        # samples_weight = samples_weight.double()
        train_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        # train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def generate_golden_vectors(output_dir, dataset):
    logger = getLogger("root.generate_golden_vectors")
    import onnxruntime as ort
    headerfile_info = {}
    vector_files = []
    ort_sess = ort.InferenceSession(os.path.join(output_dir, 'model.onnx'))
    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name

    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    create_dir(golden_vectors_dir)
    logger.info(f"Creating Golden data for reference at {golden_vectors_dir}")
    label_index_dict = {label: np.where(dataset.Y == label)[0] for label in np.unique(dataset.Y)}

    for label, indices in label_index_dict.items():
        # For each label, 4 random golden test vectors will be selected and printed out
        for index in random.sample(list(indices), k=2):  # Originally k=4
            np_raw = dataset.X_raw[index]
            np_feat = dataset.X[index]
            pred = ort_sess.run([output_name], {input_name: np.expand_dims(np_feat, 0).astype(np.float32)})[0]

            half_path = os.path.join(golden_vectors_dir, f'test_vector_class{label}')

            # Saving as .txt
            np.savetxt(half_path + f'_X_adc_{index}.txt', np_raw.flatten(), fmt='%.0f,', header=f'uint16_t test_vector_class{label}_X_adc_{index}[{len(np_raw.flatten())}]= {{', footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'_X_adc_{index}.txt')
            np.savetxt(half_path + f'_X_features_{index}.txt', np_feat.flatten(), fmt='%.5f,', header=f'int8_t test_vector_class{label}_X_features_{index}[{len(np_feat.flatten())}] = {{', footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'_X_features_{index}.txt')
            np.savetxt(half_path + f'_Y_{index}.txt', pred.flatten(), fmt='%.0f,', header=f'int8_t test_vector_class{label}_Y_{index}[{len(pred.flatten())}] = {{', footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'_Y_{index}.txt')

    headerfile_info = '\n'.join([f'#define {k} {v}' for k, v in dataset.feature_extraction_params.items()])
    for file_path in vector_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path) as fp:
            file_array = fp.read()
        headerfile_info += f'\n{file_array}\n'

    global_var_h = os.path.join(golden_vectors_dir, 'global.h')
    with open(global_var_h, 'w') as fp:
        fp.write(headerfile_info)
    logger.info("Creating C header file for variables at: {}".format(global_var_h))


def main(gpu, args):
    transform = None
    if not args.output_dir:
        output_folder = os.path.basename(os.path.split(args.data_path)[0])
        args.output_dir = os.path.join('./data/checkpoints/classification', output_folder, f'{args.model}', args.date)
    utils.mkdir(args.output_dir)
    log_file = os.path.join(args.output_dir, f'run.log')
    logger = Logger(log_file=args.lis or log_file, DEBUG=args.DEBUG, name="root", append_log=True if args.quantization else False, console_log=True)
    # logger = command_display(args.lis or log_file, args.DEBUG)
    utils.seed_everything(args.seed)
    logger = getLogger("root.main")
    logger.info("Script: {}".format(os.path.relpath(__file__)))

    # if args.quantization and args.store_feat_ext_data:
    #     logger.info("Avoiding storage of feature extracted data again during QAT")
    #     args.store_feat_ext_data = False
    if args.store_feat_ext_data:
        if args.feat_ext_store_dir in [None, 'None']:
            args.feat_ext_store_dir = os.path.join(args.output_dir, 'feat_ext_data')
            logger.info(f"feat_ext_store_dir has been defaulted to: {args.feat_ext_store_dir}")
        create_dir(args.feat_ext_store_dir)

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
            args.transforms = [args.data_proc_transforms[0] + [args.feat_ext_transform]]  # args.data_proc_transforms is a list of lists
        else:
            args.transforms = [args.data_proc_transforms + [args.feat_ext_transform]]
    dataset, dataset_test, train_sampler, test_sampler = load_data(args.data_path, args)  # (126073, 1, 152), 126073
    if args.store_feat_ext_data and args.dont_train_just_feat_ext:
        logger.info("Exiting execution without training")
        sys.exit(0)


    # collate_fn = None
    num_classes = len(dataset.classes)

    logger.info("Loading data:")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True,
        collate_fn=collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True,
        collate_fn=collate_fn, )

    if args.test_onnx_only:
        ort_sess = ort.InferenceSession(os.path.join(args.output_dir, 'model.onnx'))
        input_name = ort_sess.get_inputs()[0].name
        output_name = ort_sess.get_outputs()[0].name
        predicted = []
        ground_truth = torch.tensor([])
        for batched_data, batched_target in data_loader:
            batched_data = batched_data.to(device, non_blocking=True).float()
            batched_target = batched_target.to(device, non_blocking=True).long()
            if transform:
                batched_data = transform(batched_data)
            for data in batched_data:
                predicted.append(np.argmax(ort_sess.run([output_name], {input_name: np.array(data.unsqueeze(0)).astype(np.float32)})[0]))
            ground_truth = torch.cat((ground_truth, batched_target))

        metric = torcheval.metrics.MulticlassAccuracy()
        metric.update(torch.Tensor(predicted), ground_truth)

        logger.info(f"Test Data Evaluation Accuracy: {metric.compute()*100:.2f}%")

        confusion_matrix = get_confusion_matrix(torch.Tensor(predicted).type(torch.int64), ground_truth.type(torch.int64), num_classes).cpu().numpy()
        #
        # logger.info('\n' + '\n'.join(
        #     [f"Ground Truth:(Class {dataset.inverse_label_map[i]}), Predicted:(Class {dataset.inverse_label_map[j]}): {int(confusion_matrix[i][j])}" for j in
        #      range(num_classes) for i in range(num_classes)]))
        logger.info('Confusion Matrix:\n {}'.format(tabulate(pd.DataFrame(confusion_matrix,
                      columns=[f"Predicted as: {x}" for x in dataset.inverse_label_map.values()],
                      index=[f"Ground Truth: {x}" for x in dataset.inverse_label_map.values()]),
                                                             headers="keys", tablefmt='grid')))

        return

    logger.info("Creating model")
    # Model Creation
    variables = dataset.X.shape[1]  # n,c,h,w --> c is 1 (args.variables. However, after motor fault was supported, it concatenates 3x128 to 1x384 hence channels have been changed
    input_features = dataset.X.shape[2]
    # TODO: One solution is to see where exactly variables get used in timeseries_dataset and see if it can be made redundant there
    model = models.get_model(
        args.model, variables, num_classes, input_features=input_features, model_config=args.model_config,
        model_spec=args.model_spec, with_input_batchnorm=True if args.with_input_batchnorm in ['True', True] else False) # args.model is a string, how to make it a callable
    if args.generic_model:
        # logger.info("\nModel:\n{}\n".format(model))
        logger.info(f"{torchinfo.summary(model, (1, variables, input_features, 1))}")

    if args.weights:
        if args.weights_url:
            # if args.test_onnx_only:
            #     logger.info(f"loading pretrained checkpoint for test: {args.weights_url}")
            # else:
            logger.info(f"loading pretrained checkpoint for training: {args.weights_url}")
            model = load_weights(model, args.weights_url, state_dict_name=args.weights_state_dict_name)

    if args.quantization == tinyml_torchmodelopt.quantization.TinyMLQuantizationVersion.QUANTIZATION_GENERIC:
        model = tinyml_torchmodelopt.quantization.GenericTinyMLQATFxModule(model, total_epochs=args.epochs)
    elif args.quantization == tinyml_torchmodelopt.quantization.TinyMLQuantizationVersion.QUANTIZATION_TINPU:
        model = tinyml_torchmodelopt.quantization.TINPUTinyMLQATFxModule(model, total_epochs=args.epochs)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.export_only:
        if args.distributed is False or (args.distributed is True and int(os.environ['LOCAL_RANK']) == 0):
            utils.export_model(model, input_shape=(1, variables, input_features), output_dir=args.output_dir, opset_version=args.opset_version, quantization=args.quantization)
            return

    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name)
    elif opt_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, eps=0.0316, alpha=0.9)
    else:
        raise RuntimeError("Invalid optimizer {}. Only SGD and RMSprop are supported.".format(args.opt))

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == 'steplr':
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                       T_max=args.epochs - args.lr_warmup_epochs)
    elif args.lr_scheduler == 'exponentiallr':
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
                           "are supported.".format(args.lr_scheduler))

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == 'linear':
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_warmup_decay,
                                                                    total_iters=args.lr_warmup_epochs)
        elif args.lr_warmup_method == 'constant':
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=args.lr_warmup_decay,
                                                                      total_iters=args.lr_warmup_epochs)
        else:
            raise RuntimeError(f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant "
                               "are supported.")
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

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
    start_time = time.time()
    best = dict(accuracy=0.0, f1=0, conf_matrix=dict(), epoch=None)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        utils.train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, transform, args.apex, model_ema,
                              print_freq=args.print_freq, phase=phase, num_classes=num_classes)
        lr_scheduler.step()
        avg_accuracy, avg_f1, avg_conf_matrix = utils.evaluate(model, criterion, data_loader_test, device=device, transform=transform, phase=phase, num_classes=num_classes)
        if model_ema:
            avg_accuracy, avg_f1, avg_conf_matrix = utils.evaluate(model_ema, criterion, data_loader_test, device=device, transform=transform,
                                          log_suffix='EMA', print_freq=args.print_freq, phase=phase)
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
                best['conf_matrix'] = avg_conf_matrix
                best['epoch'] = epoch
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint.pth'))

    logger = getLogger("root.main.BestEpoch")
    logger.info("")
    logger.info("Printing statistics of best epoch:")
    logger.info(f"Best Epoch: {best['epoch']}")
    logger.info(f"Acc@1 {best['accuracy']:.3f}")
    logger.info(f"F1-Score {best['f1']:.3f}")
    # logger.info(f"Confusion Matrix:\n" + '\n'.join(
    #     [f"Ground Truth:(Class {dataset.inverse_label_map[i]}), Predicted:(Class {dataset.inverse_label_map[j]}): {int(best['conf_matrix'][i][j])}" for j in
    #      range(num_classes) for i in range(num_classes)]))
    logger.info('Confusion Matrix:\n {}'.format(tabulate(pd.DataFrame(best['conf_matrix'],
                  columns=[f"Predicted as: {x}" for x in dataset.inverse_label_map.values()],
                  index=[f"Ground Truth: {x}" for x in dataset.inverse_label_map.values()]),
                                                         headers="keys", tablefmt='grid')))

    logger = getLogger("root.main")
    logger.info('Exporting model after training.')
    if args.distributed is False or (args.distributed is True and int(os.environ['LOCAL_RANK']) == 0):       
        example_input = next(iter(data_loader_test))[0]
        utils.export_model(model, input_shape=(1,) + dataset.X.shape[1:], output_dir=args.output_dir, opset_version=args.opset_version, 
                           quantization=args.quantization, quantization_error_logging=args.quantization_error_logging, example_input=example_input)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    if args.gen_golden_vectors:
        generate_golden_vectors(args.output_dir, dataset)


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


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()

    # run the training.
    # if args.distributed is True is set, then this will launch distributed training
    # depending on args.gpus
    run(arguments)
