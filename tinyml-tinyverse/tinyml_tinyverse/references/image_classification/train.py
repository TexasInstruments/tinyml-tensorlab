#################################################################################
# Copyright (c) 2023-2026, Texas Instruments
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
from argparse import ArgumentParser, Namespace
from logging import getLogger

import numpy as np
import pandas as pd
from tinyml_tinyverse.common.models import NeuralNetworkWithPreprocess
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion, TinyMLQuantizationMethod
from tinyml_torchmodelopt.nas.train_cnn_search import search_and_get_model

# Torch Modules
import torch
import torch.nn as nn
import torchinfo
from tabulate import tabulate

from tinyml_tinyverse.common import models
from tinyml_tinyverse.common.datasets import GenericImageDataset
from tinyml_tinyverse.common.utils import misc_utils, utils, gof_utils
from tinyml_tinyverse.common.utils.mdcl_utils import Logger

# Import common functions from base module
from ..common.train_base import (
    get_base_args_parser,
    generate_golden_vector_dir,
    generate_user_input_config,
    generate_test_vector,
    generate_model_aux,
    load_datasets,
    run_distributed,
    assemble_golden_vectors_header,
    setup_training_environment,
    prepare_transforms,
    create_data_loaders,
    log_model_summary,
    load_pretrained_weights,
    setup_optimizer_and_scheduler,
    setup_distributed_model,
    resume_from_checkpoint,
    save_checkpoint,
    handle_export_only,
    move_model_to_device,
    log_training_time,
    apply_output_int_default,
    get_output_int_flag,
    load_onnx_for_inference,
)

dataset_loader_dict = {'GenericImageDataset':GenericImageDataset}
dataset_load_state = {'dataset': None, 'dataset_test': None, 'train_sampler': None, 'test_sampler': None}


def get_args_parser():
    """
    This function is used to process inputs given to the program
    """
    DESCRIPTION = "This script loads image data and trains it generating a model"
    parser = ArgumentParser(description=DESCRIPTION)
    # parser.add_argument('--out_dir', help='Run directory', default=os.getcwd())

    parser.add_argument('--data-proc-transforms', help="Data Preprocessing transforms ", default=[])  # default=['DownSample', 'SimpleWindow'])
    parser.add_argument('--augmentation-transform', help="Training-only image augmentation transforms", default=[])
    parser.add_argument('--feat-ext-transform', help="Feature Extraction transforms ", default=[])
    parser.add_argument('--store-feat-ext-data', help='Store Data post Feature extractions')
    parser.add_argument('--feat-ext-store-dir', help='Store Data post Feature extractions in this directory')
    parser.add_argument('--dont-train-just-feat-ext', help='Quit after Feature Extraction without Training. Does not have any effect if --store-feat-ext-data is not used')
    parser.add_argument('--image-height', help="Image dimension(Height)")
    parser.add_argument('--image-width', help="Image dimension(Width)")
    parser.add_argument('--image-mean', help="Average pixel intensity of dataset computed per channel")
    parser.add_argument('--image-scale', help="Standard deviation of pixel intensities per channel")
    parser.add_argument('--image-num-channel', help="Number of channels( RGB=3, Greyscale=1) present in the image")
    parser.add_argument('--generic-model', help="Open Source models", type=misc_utils.str_or_bool, default=False)
    parser.add_argument('--file-level-classification-log', help='File level classification log file', type=str)
    parser.add_argument('--gen_golden_vectors', help="Generate golden vectors to feed into the model", type=misc_utils.str_or_bool, default=True)
    # Optional image preprocessing params
    parser.add_argument('--pad-value', help="Padding pixel value used for RESIZE_PAD transform", default=0)
    parser.add_argument('--binary-threshold', help="Threshold value used for BINARIZE transform", default=128)

    # CLAHE params
    parser.add_argument('--clahe-clip-limit', help="Clip limit for CLAHE contrast enhancement", default=2.0)
    parser.add_argument('--clahe-tile-grid-size', help="Tile grid size for CLAHE transform, for example '(8, 8)'", default=(8, 8))

    # Sobel params
    parser.add_argument('--sobel-mode', help="Sobel mode: x, y, or magnitude", default="magnitude")
    parser.add_argument('--sobel-ksize', help="Kernel size for Sobel filter", default=3)

    # Laplacian params
    parser.add_argument('--laplacian-ksize', help="Kernel size for Laplacian filter", default=3)

    # Random augmentation params
    parser.add_argument('--horizontal-flip-prob', help="Probability for RANDOM_HORIZONTAL_FLIP transform during training", default=0.5)
    parser.add_argument('--vertical-flip-prob', help="Probability for RANDOM_VERTICAL_FLIP transform during training", default=0.5)
    parser.add_argument('--random-rotation-deg', help="Maximum rotation angle in degrees for RANDOM_ROTATION transform during training", default=15)

    # Color jitter params
    parser.add_argument('--color-jitter-brightness', help="Brightness factor for COLOR_JITTER transform during training", default=0.10)
    parser.add_argument('--color-jitter-contrast', help="Contrast factor for COLOR_JITTER transform during training", default=0.10)
    parser.add_argument('--color-jitter-saturation', help="Saturation factor for COLOR_JITTER transform during training", default=0.05)
    parser.add_argument('--color-jitter-hue', help="Hue factor for COLOR_JITTER transform during training", default=0.01)
    parser.add_argument('--variables', help="1- if Univariate, 2/3/.. if multivariate")

    parser.add_argument('--data-path', default=os.path.join('.', 'data', 'datasets'), help='dataset')
    parser.add_argument('--dataset', default='folder', help='dataset')
    parser.add_argument('--gof-test', type=misc_utils.str2bool, default=False, help='Enable goodness-of-fit test') 
    parser.add_argument('--dataset-loader', default='GenericImageDataset', help='dataset loader')
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
    parser.add_argument('-j', '--workers', default=0 if platform.system() in ['Windows'] else 8, type=int, metavar='N', help='number of data loading workers (default: 16)')
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
    parser.add_argument('--print-freq', default=None, type=int, help='print frequency')
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

    #######################################
    # nas args
    #######################################
    parser.add_argument("--nas_enabled", default=False, help="Enable/ Disable NAS")
    parser.add_argument("--nas_optimization_mode", default="Memory", type=str,  help="Optimize model for compute or storage efficiency")
    parser.add_argument("--nas_model_size", default='None', choices=['s', 'm', 'l', 'xl', 'None'], help="Proxy for model size")
    parser.add_argument("--nas_epochs", default=10, type=int, help="Iterations for search")

    parser.add_argument("--nas_nodes_per_layer", default=4, type=int, help="Number of nodes per layer")
    parser.add_argument("--nas_layers", default=3, type=int, help="Shoulde be minimum 3")
    parser.add_argument("--nas_init_channels", default=1, type=int, help="Initial channel size of the first feature map")
    parser.add_argument("--nas_init_channel_multiplier", default=3, type=int, help="Channel size of after first preprocess")
    parser.add_argument("--nas_fanout_concat", default=4, type=int, help="Number of nodes to concat for output after each layer")

    parser.add_argument("--load_saved_model", type=str, default='None', help="Model path for pre-searched nas model")

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-state-dict-name", default="model", type=str, help="the weights member name to load from the checkpoint")
    parser.add_argument("--nn-for-feature-extraction", default=False, type=misc_utils.str2bool, help="Use an AI model for preprocessing")
    parser.add_argument("--output-int", default=None, type=misc_utils.str_or_bool, help="Get quantized int8 output from model (False for dequantized float output). If not specified, determined automatically based on task type and quantization level.")
    parser.add_argument("--ondevice-training", default=False, type=misc_utils.str2bool, help="Specified whether the current model can be trained on device or not")

    return parser


def get_nas_args(args, data_loader, data_loader_test, num_classes, variables):
    """Configure NAS arguments based on model size preset."""
    if args.nas_model_size != "None":
        model_size = args.nas_model_size
        if model_size == 's':
            args.nas_nodes_per_layer, args.nas_layers = 4, 3
            args.nas_init_channels, args.nas_init_channel_multiplier, args.nas_fanout_concat = 1, 3, 4
        elif model_size == 'm':
            args.nas_nodes_per_layer, args.nas_layers = 4, 10
            args.nas_init_channels, args.nas_init_channel_multiplier, args.nas_fanout_concat = 1, 3, 4
        elif model_size == 'l':
            args.nas_nodes_per_layer, args.nas_layers = 4, 12
            args.nas_init_channels, args.nas_init_channel_multiplier, args.nas_fanout_concat = 4, 3, 4
        elif model_size == 'xl':
            args.nas_nodes_per_layer, args.nas_layers = 4, 20
            args.nas_init_channels, args.nas_init_channel_multiplier, args.nas_fanout_concat = 4, 3, 4
        elif model_size == 'xxl':
            args.nas_nodes_per_layer, args.nas_layers = 6, 20
            args.nas_init_channels, args.nas_init_channel_multiplier, args.nas_fanout_concat = 8, 3, 4

    nas_args_dict = {
        'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay, 'gpu': 0,
        'nas_budget': args.nas_epochs, 'nas_init_channels': args.nas_init_channels,
        'nas_nodes_per_layer': args.nas_nodes_per_layer, 'nas_layers': args.nas_layers,
        'nas_multiplier': args.nas_fanout_concat, 'nas_stem_multiplier': args.nas_init_channel_multiplier,
        'nas_optimization_mode': args.nas_optimization_mode, 'in_channels': variables, 'grad_clip': 5,
        'mode': 'cnn', 'arch_learning_rate': 1e-2, 'arch_weight_decay': 1e-3, 'unrolled': True,
        'num_classes': num_classes, 'train_loader': data_loader, 'valid_loader': data_loader_test,
    }
    return Namespace(**nas_args_dict)


def generate_golden_vectors(output_dir, dataset, output_int, generic_model=False, nn_for_feature_extraction=False):
    logger = getLogger("root.generate_golden_vectors")
    ort_sess, input_name, output_name = load_onnx_for_inference(output_dir, generic_model)
    vector_files = []

    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    
    logger.info(f"Creating Golden data for reference at {golden_vectors_dir}")
    label_index_dict = {dataset.inverse_label_map.get(label): np.where(dataset.Y == label)[0] for label in np.unique(dataset.Y)}

    for label, indices in label_index_dict.items():
        for index in random.sample(list(indices), k=2):
            np_raw = dataset.X_raw[index]
            if nn_for_feature_extraction:
                np_feat = np_raw
                pred = ort_sess.run([output_name], {input_name: np.expand_dims(np_raw, 0).astype(np.float32)})[0]
            else:
                np_feat = dataset.X[index]
                pred = ort_sess.run([output_name], {input_name: np.expand_dims(np_feat, 0).astype(np.float32)})[0]

            half_path = os.path.join(golden_vectors_dir)

            # Saving as .txt
            arr = np_raw.detach().cpu().numpy().flatten()
            # np.savetxt(half_path + f'image_{label}_{index}.txt', np_raw.flatten(), fmt='%f,' if np_raw.dtype.kind == 'f' else '%d,', header=f'//Class: {label} (Index: {index}): Image Data\nfloat raw_input_test[{len(np_raw.flatten())}]= {{', footer='}', comments='', newline=' ')
            np.savetxt(half_path + f'image_{label}_{index}.txt', arr, fmt='%f,' if arr.dtype.kind == 'f' else '%d,',header=f'//Class: {label} (Index: {index}): Image Data\nfloat raw_input_test[{len(arr)}]= {{',footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'image_{label}_{index}.txt')
            if not nn_for_feature_extraction:
                np.savetxt(half_path + f'features_{label}_{index}.txt', np_feat.flatten(), fmt='%.5f,', header=f'//Class: {label} (Index: {index}): Extracted Features\nfloat model_test_input[{len(np_feat.flatten())}] = {{', footer='}', comments='', newline=' ')
                vector_files.append(half_path + f'features_{label}_{index}.txt')
            np.savetxt(half_path + f'output_{label}_{index}.txt', pred.flatten(), fmt='%d,' if output_int else '%f,', header=f'//Class: {label} (Index: {index}): Expected Model Output\n{"int8_t" if output_int else "float"} golden_output[{len(pred.flatten())}] = {{', footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'output_{label}_{index}.txt')

    header_file_info = assemble_golden_vectors_header(vector_files, files_per_set=3)
    generate_user_input_config(output_dir, dataset)
    generate_test_vector(output_dir, header_file_info)
    generate_model_aux(output_dir, dataset)

def set_dataset_augmentation_enabled(dataset, enabled):
    if hasattr(dataset, "set_augmentation_enabled"):
        dataset.set_augmentation_enabled(enabled)
        
def main(gpu, args):
    """Main training function for classification."""
    logger, device = setup_training_environment(args, gpu, 'classification', __file__)
    prepare_transforms(args)
   
    # Load or reuse datasets
    if args.quantization:
        dataset, dataset_test, train_sampler, test_sampler = (dataset_load_state['dataset'], dataset_load_state['dataset_test'],
                                                               dataset_load_state['train_sampler'], dataset_load_state['test_sampler'])
    else:
        dataset, dataset_test, train_sampler, test_sampler = load_datasets(args.data_path, args, dataset_loader_dict)
        dataset_load_state['dataset'], dataset_load_state['dataset_test'] = dataset, dataset_test
        dataset_load_state['train_sampler'], dataset_load_state['test_sampler'] = train_sampler, test_sampler

        try:
            utils.plot_feature_components_graph(dataset, graph_type='pca', instance_type='train', output_dir=args.output_dir)
            utils.plot_feature_components_graph(dataset_test, graph_type='pca', instance_type='validation', output_dir=args.output_dir)
            if args.gof_test:
                if args.frame_size != 'None':
                    gof_utils.goodness_of_fit_test(frame_size=int(args.frame_size), classes_dir=args.data_path,
                                                   output_dir=args.output_dir, class_names=dataset.classes)
                else:
                    logger.warning(f"Goodness of Fit plots will not be generated because frame_size was not given in the YAML file.")
        except Exception as e:
            logger.warning(f"Feature Extraction plots will not be generated because: {e}")

    if misc_utils.str2bool(args.dont_train_just_feat_ext):
        logger.info('Exiting execution without training')
        sys.exit(0)

    # collate_fn = None
    num_classes = len(dataset.classes)
    variables = dataset.X.shape[1]
    input_features = dataset.X.shape[2]

    logger.info("Loading data:")
    data_loader, data_loader_test = create_data_loaders(dataset, dataset_test, train_sampler, test_sampler, args, gpu)

    logger.info("Creating model")

    if args.load_saved_model == 'None':
        if args.nas_enabled == 'True':
            if args.quantization:
                model = torch.load(os.path.join(os.path.dirname(args.output_dir), os.path.join('base', 'nas_model.pt')), weights_only=False)
            else:
                nas_args = get_nas_args(args, data_loader, data_loader_test, num_classes, variables)
                model = search_and_get_model(nas_args)
                if not model:
                    logger.error("Please check on prior errors. NAS wasn't able to create a model")
                    sys.exit(1)
                torch.save(model, os.path.join(args.output_dir, 'nas_model.pt'))
        else:
            model = models.get_model(
                args.model, variables, num_classes, input_features=input_features, model_config=args.model_config,
                model_spec=args.model_spec,
                dual_op=args.dual_op)
    else:
        model = torch.load(args.load_saved_model, weights_only=False)

    if args.generic_model or args.nas_enabled:
        summary_input_shape = (1,) + tuple(dataset.X.shape[1:])
        logger.info(f"Model summary input shape: {summary_input_shape}")
        logger.info(f"{torchinfo.summary(model, summary_input_shape)}")

    model = load_pretrained_weights(model, args, logger)

    if handle_export_only(model, args, variables, input_features, logger):
        return

    move_model_to_device(model, device, logger)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
# logger.info(f"args.transforms = {args.transforms}"
    model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, args)
    resume_from_checkpoint(model_without_ddp, optimizer, lr_scheduler, model_ema, args)

    phase = 'QuantTrain' if args.quantization else 'FloatTrain'
    logger.info("Start training")
    start_time = timeit.default_timer()
    best = dict(accuracy=0.0, f1=0, conf_matrix=dict(), epoch=None)

    # model = NeuralNetworkWithPreprocess
    if args.nn_for_feature_extraction:
        fe_model = models.FEModelLinear(dataset.X.shape[1], dataset.X_raw.shape[2], dataset.X.shape[2]).to(device)
        fe_model = NeuralNetworkWithPreprocess(fe_model, None)
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(fe_model, args)
        fe_model = utils.get_trained_feature_extraction_model(
            fe_model, args, data_loader, data_loader_test, device, lr_scheduler, optimizer)
        model = NeuralNetworkWithPreprocess(fe_model, model)
    else:
        model = NeuralNetworkWithPreprocess(None, model)

    # if output_int not set by user, then set it to default of task_type
    if args.output_int == None:
        args.output_int = True
    model = utils.quantization_wrapped_model(
        model, args.quantization, args.quantization_method, args.weight_bitwidth, args.activation_bitwidth,
        args.epochs, args.output_int)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
         train_sampler.set_epoch(epoch)

        set_dataset_augmentation_enabled(dataset, True)

        utils.train_one_epoch_classification(
            model, criterion, optimizer, data_loader, device, epoch, None, args.apex, model_ema,
            print_freq=args.print_freq, phase=phase, num_classes=num_classes, dual_op=args.dual_op,
            is_ptq=True if (args.quantization_method in ['PTQ'] and args.quantization) else False,
            nn_for_feature_extraction=args.nn_for_feature_extraction)

        set_dataset_augmentation_enabled(dataset, False)
        if not (args.quantization_method in ['PTQ'] and args.quantization):
            lr_scheduler.step()
        set_dataset_augmentation_enabled(dataset, False)
        set_dataset_augmentation_enabled(dataset_test, False)
        avg_accuracy, avg_f1, auc, avg_conf_matrix, predictions, ground_truth = utils.evaluate_classification(
            model, criterion, data_loader_test, device=device, transform=None, phase=phase,
            num_classes=num_classes, dual_op=args.dual_op, nn_for_feature_extraction=args.nn_for_feature_extraction)
        if model_ema:
            avg_accuracy, avg_f1, auc, avg_conf_matrix, predictions, ground_truth = utils.evaluate_classification(
                model_ema, criterion, data_loader_test, device=device, transform=None,
                log_suffix='EMA', print_freq=args.print_freq, phase=phase, dual_op=args.dual_op,
                nn_for_feature_extraction=args.nn_for_feature_extraction)
        if args.output_dir and avg_accuracy >= best['accuracy']:
            logger.info(f"Epoch {epoch}: {avg_accuracy:.2f} (Val accuracy) >= {best['accuracy']:.2f} (So far best accuracy). Hence updating checkpoint.pth")
            best['accuracy'], best['f1'], best['auc'], best['conf_matrix'], best['epoch'] = avg_accuracy, avg_f1, auc, avg_conf_matrix, epoch
            best['predictions'], best['ground_truth'] = predictions, ground_truth
            checkpoint = save_checkpoint(model_without_ddp, optimizer, lr_scheduler, epoch, args, model_ema)
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

    # Log best epoch results
    set_dataset_augmentation_enabled(dataset, False)
    set_dataset_augmentation_enabled(dataset_test, False)
    logger = getLogger(f"root.main.{phase}.BestEpoch")
    logger.info("")
    logger.info("Printing statistics of best epoch:")
    logger.info(f"Best Epoch: {best['epoch']}")
    logger.info(f"Acc@1 {best['accuracy']:.3f}")
    logger.info(f"F1-Score {best['f1']:.3f}")
    logger.info(f"AUC ROC Score {best['f1']:.3f}")
    logger.info("")
    logger.info('Confusion Matrix:\n {}'.format(tabulate(pd.DataFrame(best['conf_matrix'],
                  columns=[f"Predicted as: {x}" for x in dataset.inverse_label_map.values()],
                  index=[f"Ground Truth: {x}" for x in dataset.inverse_label_map.values()]),
                                                         headers="keys", tablefmt='grid')))

    Logger(log_file=args.file_level_classification_log, DEBUG=args.DEBUG,
           name="root.utils.print_file_level_classification_summary",
           append_log=True if args.quantization else False, console_log=False)
    getLogger("root.utils.print_file_level_classification_summary").propagate = False
    utils.print_file_level_classification_summary(dataset_test, best['predictions'], best['ground_truth'], phase)
    logger.info(f"Generated file-level classification summary in: {args.file_level_classification_log}")

    # Export model
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
            remove_hooks_for_jit=True if (args.quantization_method == TinyMLQuantizationMethod.PTQ and args.quantization) else False)

    log_training_time(start_time)
    
    if args.gen_golden_vectors:
        
        set_dataset_augmentation_enabled(dataset, False)
        set_dataset_augmentation_enabled(dataset_test, False)
        generate_golden_vector_dir(args.output_dir)
        output_int = get_output_int_flag(args)
        generate_golden_vectors(args.output_dir, dataset, output_int, args.generic_model, args.nn_for_feature_extraction)


def run(args):
    """Run training with optional distributed mode."""
    run_distributed(main, args)


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()
    # Apply default output_int if not specified by user
    apply_output_int_default(arguments, 'image_classification')

    # run the training.
    # if args.distributed is True is set, then this will launch distributed training
    # depending on args.gpus
    run(arguments)
