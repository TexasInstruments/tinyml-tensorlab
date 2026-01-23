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
# BSD 3-Clause License - Copyright (c) Soumith Chintala 2016
#################################################################################

"""
Time series classification training script.
"""

import os
import random
import sys
import timeit
from argparse import Namespace
from logging import getLogger

import numpy as np
import pandas as pd
from tabulate import tabulate

from tinyml_tinyverse.common.models import NeuralNetworkWithPreprocess
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion, TinyMLQuantizationMethod
from tinyml_torchmodelopt.nas.train_cnn_search import search_and_get_model

import torch
import torch.nn as nn
import torchinfo

from tinyml_tinyverse.common import models
from tinyml_tinyverse.common.datasets import GenericTSDataset
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

dataset_loader_dict = {'GenericTSDataset': GenericTSDataset}
dataset_load_state = {'dataset': None, 'dataset_test': None, 'train_sampler': None, 'test_sampler': None}


def get_args_parser():
    """Create argument parser with classification-specific arguments."""
    parser = get_base_args_parser("This script loads time series data and trains a classification model")

    # Classification-specific arguments
    parser.add_argument('--gain-variations', help='Gain Variation Dictionary to be applied to each of the classes')
    parser.add_argument('--gof-test', type=misc_utils.str2bool, default=False, help='Enable goodness-of-fit test')
    parser.add_argument('--q15-scale-factor', help="q15 scaling factor")
    parser.add_argument('--file-level-classification-log', help='File level classification log file', type=str)

    # PIR Detection related params
    parser.add_argument('--window-count', help="Number of windows in each input frame ", type=int, default=25)
    parser.add_argument('--chunk-size', help="length of kurtosis section size within a window in samples ", type=int, default=8)
    parser.add_argument('--fft-size', help="dimension of a FFT operation on input frame ", type=int, default=64)

    # NAS arguments
    parser.add_argument("--nas_enabled", default=False, help="Enable/ Disable NAS", type=misc_utils.str2bool)
    parser.add_argument("--nas_optimization_mode", default="Memory", type=str, help="Optimize model for compute or storage efficiency")
    parser.add_argument("--nas_model_size", default='None', choices=['s', 'm', 'l', 'xl', 'None'], help="Proxy for model size")
    parser.add_argument("--nas_epochs", default=10, type=int, help="Iterations for search")
    parser.add_argument("--nas_nodes_per_layer", default=4, type=int, help="Number of nodes per layer")
    parser.add_argument("--nas_layers", default=3, type=int, help="Should be minimum 3")
    parser.add_argument("--nas_init_channels", default=1, type=int, help="Initial channel size of the first feature map")
    parser.add_argument("--nas_init_channel_multiplier", default=3, type=int, help="Channel size of after first preprocess")
    parser.add_argument("--nas_fanout_concat", default=4, type=int, help="Number of nodes to concat for output after each layer")
    parser.add_argument("--load_saved_model", type=str, default='None', help="Model path for pre-searched nas model")

    # Feature extraction with NN
    parser.add_argument("--nn-for-feature-extraction", default=False, type=misc_utils.str2bool, help="Use an AI model for preprocessing")

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
    """Generate golden vectors for classification."""
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

            np.savetxt(half_path + f'adc_{label}_{index}.txt', np_raw.flatten(),
                       fmt='%f,' if np_raw.dtype.kind == 'f' else '%d,',
                       header=f'//Class: {label} (Index: {index}): ADC Data\nfloat raw_input_test[{len(np_raw.flatten())}]= {{',
                       footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'adc_{label}_{index}.txt')
            if not nn_for_feature_extraction:
                np.savetxt(half_path + f'features_{label}_{index}.txt', np_feat.flatten(), fmt='%.5f,',
                           header=f'//Class: {label} (Index: {index}): Extracted Features\nfloat model_test_input[{len(np_feat.flatten())}] = {{',
                           footer='}', comments='', newline=' ')
                vector_files.append(half_path + f'features_{label}_{index}.txt')
            np.savetxt(half_path + f'output_{label}_{index}.txt', pred.flatten(),
                       fmt='%d,' if output_int else '%f,',
                       header=f'//Class: {label} (Index: {index}): Expected Model Output\n{"int8_t" if output_int else "float"} golden_output[{len(pred.flatten())}] = {{',
                       footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'output_{label}_{index}.txt')

    header_file_info = assemble_golden_vectors_header(vector_files, files_per_set=3)
    generate_user_input_config(output_dir, dataset)
    generate_test_vector(output_dir, header_file_info)
    generate_model_aux(output_dir, dataset)


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
        log_model_summary(model, args, variables, input_features, logger)

    model = load_pretrained_weights(model, args, logger)

    if handle_export_only(model, args, variables, input_features, logger):
        return

    move_model_to_device(model, device, logger)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, args)
    resume_from_checkpoint(model_without_ddp, optimizer, lr_scheduler, model_ema, args)

    phase = 'QuantTrain' if args.quantization else 'FloatTrain'
    logger.info("Start training")
    start_time = timeit.default_timer()
    best = dict(accuracy=0.0, f1=0, conf_matrix=dict(), epoch=None)

    # Handle nn_for_feature_extraction
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
        utils.train_one_epoch_classification(
            model, criterion, optimizer, data_loader, device, epoch, None, args.apex, model_ema,
            print_freq=args.print_freq, phase=phase, num_classes=num_classes, dual_op=args.dual_op,
            is_ptq=True if (args.quantization_method in ['PTQ'] and args.quantization) else False,
            nn_for_feature_extraction=args.nn_for_feature_extraction)
        if not (args.quantization_method in ['PTQ'] and args.quantization):
            lr_scheduler.step()
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
        generate_golden_vector_dir(args.output_dir)
        output_int = get_output_int_flag(args)
        generate_golden_vectors(args.output_dir, dataset, output_int, args.generic_model, args.nn_for_feature_extraction)


def run(args):
    """Run training with optional distributed mode."""
    run_distributed(main, args)


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()
    # Apply default output_int if not specified by user
    apply_output_int_default(arguments, 'timeseries_classification')
    run(arguments)
