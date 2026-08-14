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
# BSD 3-Clause License - Copyright (c) Soumith Chintala 2016
#################################################################################

"""
Radar Classification Training Pipeline.
"""

import os
import onnx
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
from tinyml_tinyverse.common.datasets import GenericRadarDataset
from tinyml_tinyverse.common.utils import misc_utils, utils, gof_utils
from tinyml_tinyverse.common.utils.mdcl_utils import Logger

def log_model_graph(model, args, input_size, logger):
    """
    Log model graph using torchinfo.
    Shows: model architecture, parameter counts, MACs, memory.
    """
    logger.info("=" * 80)
    logger.info("=== MODEL GRAPH (FloatTrain) ===")
    logger.info("=" * 80)

    logger.info(f"Model: {args.model} | Input Size: {input_size}")
    logger.info("")

    try:
        model_summary = torchinfo.summary(
            model,
            input_size=input_size,
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4,
        )
        logger.info(str(model_summary))
    except Exception as e:
        logger.info(f"torchinfo summary failed ({e}); falling back to model repr")
        logger.info(str(model))
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total:,} | Trainable: {trainable:,}")

    logger.info("=" * 80)
    logger.info("=== END MODEL GRAPH ===")
    logger.info("=" * 80)
    logger.info("")


# Quantization Configuration Logging - prints during QuantTrain phase
def log_quant_config(args, logger):
    """
    Log quantization configuration - called during QuantTrain phase.
    Only prints quantization settings
    """
    logger.info("=" * 80)
    logger.info("=== QUANTIZATION CONFIGURATION (QuantTrain) ===")
    logger.info("=" * 80)

    logger.info(f"Quantization Version: {args.quantization}")
    logger.info(f"Quantization Method:  {args.quantization_method}")
    logger.info(f"Weight Bitwidth:      {args.weight_bitwidth}")
    logger.info(f"Activation Bitwidth:  {args.activation_bitwidth}")
    logger.info(f"Output Integer:       {args.output_int}")

    # Check for quantization stubs in model if available
    logger.info("")

    logger.info("=" * 80)
    logger.info("=== END QUANTIZATION CONFIG ===")
    logger.info("=" * 80)
    logger.info("")

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
    run_memory_preflight,
    load_pretrained_weights,
    setup_optimizer_and_scheduler,
    setup_distributed_model,
    resume_from_checkpoint,
    save_checkpoint,
    handle_export_only,
    move_model_to_device,
    compile_model_if_enabled,
    log_training_time,
    apply_output_int_default,
    get_output_int_flag,
    load_onnx_for_inference,
)

dataset_loader_dict = {'GenericRadarDataset': GenericRadarDataset}
dataset_load_state = {'dataset': None, 'dataset_test': None, 'train_sampler': None, 'test_sampler': None}


def get_args_parser():
    """Create argument parser with classification-specific arguments."""
    DESCRIPTION = "This script loads radar data and trains it generating a model"
    parser = get_base_args_parser("This script loads radar data and trains a radar classification model")

    # Classification-specific arguments
    parser.add_argument('--gof-test', type=misc_utils.str2bool, default=False, help='Enable goodness-of-fit test')
    parser.add_argument('--file-level-classification-log', help='File level classification log file', type=str)

    # Radar Processing Pipeline Arguments
    parser.add_argument('--radar-range-bins-bounds',
                         help="[min_bin, max_bin] index range of 'bin*' columns to keep. Unset/None = keep all.")
    parser.add_argument('--radar-point-cloud-data-selector',
                         help="List of point cloud features to keep, from ['distance','height','azimuth','doppler','snr']")
    parser.add_argument('--radar-point-cloud-coordinate-system', help="'absolute' or 'relative' point cloud coordinates")
    parser.add_argument('--radar-point-cloud-relative-coordinate-filter',
                         help="List from ['distance','height'] - which axes to make relative to track position")
    parser.add_argument('--radar-point-cloud-spatial-filter-bounds',
                         help="[min_distance, max_distance, min_height, max_height, min_azimuth, max_azimuth]. "
                              "Unset/None = no filtering (equivalent to +-1e9 on every bound).")
    parser.add_argument('--radar-point-cloud-non-spatial-filter-bounds',
                         help="[min_doppler, max_doppler, min_snr, max_snr]. Unset/None = no filtering.")
    parser.add_argument('--radar-track-data-selector',
                         help="List of track features to keep, from ['velocity','distance','height','azimuth','acceleration']")
    parser.add_argument('--radar-track-velocity-bounds',
                         help="[min_velocity, max_velocity] applied to all track_velocity_* columns. Unset/None = no filtering.")
    parser.add_argument('--radar-track-acceleration-bounds',
                         help="[min_acceleration, max_acceleration] applied to all track_acc_* columns. Unset/None = no filtering.")
    parser.add_argument('--radar-track-spatial-filter-bounds',
                         help="[min_distance, max_distance, min_height, max_height, min_azimuth, max_azimuth] "
                              "applied to track position columns. Unset/None = no filtering.")
    parser.add_argument('--radar-data-scaling-track-velocity', help="Multiplier applied to track_velocity_* columns")
    parser.add_argument('--radar-max-points-per-frame', help="Max point cloud points kept per frame (zero-padded if fewer)")
    parser.add_argument('--radar-min-points-per-frame', help="Min real (pre-padding) points required per frame before it's dropped (for per_radar_frame mode)")
    parser.add_argument('--radar-min-points-per-window', help="Min real points required across entire window (sum of per-frame points) before window is dropped (for per_window mode)")
    parser.add_argument('--radar-point-cloud-structure-strategy',
                         help="'highest only' / 'lowest only' / 'highest and lowest' - which points to keep, per frame")
    parser.add_argument('--radar-point-cloud-sort-by', default='height',
                         help="Which selected point feature ('distance'/'height'/'azimuth'/'doppler'/'snr') to rank points by "
                              "before applying --radar-point-cloud-structure-strategy")
    parser.add_argument('--radar-frames-quality-index',
                         help="Quality filtering mode: 'per_radar_frame' (drop sparse frames), 'per_window' (drop sparse windows), or unset (no filtering)")
    parser.add_argument('--radar-feature-ordering',
                         help="Choose the method for window concatenation and flattening - Either 'frame_major' or 'feature_major'")

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


def generate_golden_vectors(output_dir, dataset, output_int, generic_model=False):
    """Generate golden vectors for radar classification."""
    logger = getLogger("root.generate_golden_vectors")
    ort_sess, input_name, output_name = load_onnx_for_inference(output_dir, generic_model)
    vector_files = []

    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    logger.info(f"Creating Golden data for reference at {golden_vectors_dir}")
    label_index_dict = {dataset.inverse_label_map.get(label): np.where(dataset.Y == label)[0] for label in np.unique(dataset.Y)}

    for label, indices in label_index_dict.items():
        for index in random.sample(list(indices), k=2):
            np_feat = np.array(dataset.X[index], dtype=np.float32)
            pred = ort_sess.run([output_name], {input_name: np.expand_dims(np_feat, 0)})[0]

            half_path = os.path.join(golden_vectors_dir)

            np.savetxt(half_path + f'features_{label}_{index}.txt', np_feat.flatten(), fmt='%.5f,',
                       header=f'//Class: {label} (Index: {index}): Feature Data\nfloat model_test_input[{len(np_feat.flatten())}] = {{',
                       footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'features_{label}_{index}.txt')
            np.savetxt(half_path + f'output_{label}_{index}.txt', pred.flatten(),
                       fmt='%d,' if output_int else '%f,',
                       header=f'//Class: {label} (Index: {index}): Expected Model Output\n{"int8_t" if output_int else "float"} golden_output[{len(pred.flatten())}] = {{',
                       footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'output_{label}_{index}.txt')

    header_file_info = assemble_golden_vectors_header(vector_files, files_per_set=2)
    generate_user_input_config(output_dir, dataset)
    generate_test_vector(output_dir, header_file_info)
    generate_model_aux(output_dir, dataset)


def main(gpu, args):
    """Main training function for classification."""
    logger, device = setup_training_environment(args, gpu, 'classification', __file__)
    prepare_transforms(args)

    # Load or reuse datasets
    if args.quantization and dataset_load_state['dataset'] is not None:
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
    variables = 1
    input_features = dataset.X.shape[1]

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
            # Prepare model_config with frame_size for models that need it - Internal features splitting required
            # in Dual Stream Architecture models 
            model_config_dict = {}
            if args.model_config:
                import os
                if isinstance(args.model_config, dict):
                    model_config_dict = args.model_config.copy()
                elif isinstance(args.model_config, str) and os.path.exists(args.model_config):
                    # Load existing config from file
                    if args.model_config.endswith(('.yaml', '.yml')):
                        import yaml
                        with open(args.model_config) as fp:
                            model_config_dict = yaml.safe_load(fp) or {}
                    elif args.model_config.endswith('.json'):
                        import json
                        with open(args.model_config) as fp:
                            model_config_dict = json.load(fp) or {}
                    else:
                        model_config_dict = {}
                else:
                    model_config_dict = {}
            

            # Only pass frame_size to model if Simple Window transform is active (windowing mode)
            # If Simple Window is not active, dataset runs in pass-through mode (per-frame), so frame_size should be 1/None
            simple_window_active = False
            if hasattr(dataset, '_transform_active'):
                simple_window_active = dataset._transform_active(dataset.TRANSFORM_SIMPLE_WINDOW)

            if simple_window_active and args.frame_size not in (None, 'None'):
                model_config_dict['frame_size'] = int(args.frame_size)
            elif not simple_window_active:
                model_config_dict['frame_size'] = 1

            model = models.get_model(
                args.model, variables, num_classes, input_features=input_features,
                model_config=model_config_dict,
                model_spec=args.model_spec,
                dual_op=args.dual_op)
    else:
        model = torch.load(args.load_saved_model, weights_only=False)

    run_memory_preflight(model, args, (1,) + dataset.X.shape[1:], logger)

    phase = 'QuantTrain' if args.quantization else 'FloatTrain'

    if phase == 'FloatTrain':
        log_model_graph(model, args, (1,) + dataset.X.shape[1:], logger)

    model = load_pretrained_weights(model, args, logger)

    if handle_export_only(model, args, variables, input_features, logger):
        return

    move_model_to_device(model, device, logger)
    model = compile_model_if_enabled(model, args, logger, input_shape=(1,) + dataset.X.shape[1:])
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, args)
    resume_from_checkpoint(model_without_ddp, optimizer, lr_scheduler, model_ema, args)

    logger.info("Start training")
    start_time = timeit.default_timer()
    best = dict(accuracy=0.0, f1=0, conf_matrix=dict(), epoch=None)

    model = NeuralNetworkWithPreprocess(None, model)

    # if output_int not set by user, then set it to default of task_type
    if args.output_int == None:
        args.output_int = True
    model = utils.quantization_wrapped_model(
        model, args.quantization, args.quantization_method, args.weight_bitwidth, args.activation_bitwidth,
        args.epochs, args.output_int)

    # Log quantization configuration (QuantTrain only)
    if args.quantization and args.quantization != 0:
        log_quant_config(args, logger)



    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        utils.train_one_epoch_classification(
            model, criterion, optimizer, data_loader, device, epoch, None, args.apex, model_ema,
            print_freq=args.print_freq, phase=phase, num_classes=num_classes, dual_op=args.dual_op,
            is_ptq=True if (args.quantization_method in ['PTQ'] and args.quantization) else False)
        if not (args.quantization_method in ['PTQ'] and args.quantization):
            lr_scheduler.step()
        avg_accuracy, avg_f1, auc, avg_conf_matrix, predictions, ground_truth = utils.evaluate_classification(
            model, criterion, data_loader_test, device=device, transform=None, phase=phase,
            num_classes=num_classes, dual_op=args.dual_op)
        if model_ema:
            avg_accuracy, avg_f1, auc, avg_conf_matrix, predictions, ground_truth = utils.evaluate_classification(
                model_ema, criterion, data_loader_test, device=device, transform=None,
                log_suffix='EMA', print_freq=args.print_freq, phase=phase, dual_op=args.dual_op)
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
    logger.info(f"AUC ROC Score {best['auc']:.3f}")
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
        example_input = next(iter(data_loader_test))[1] 
        input_shape = (1,) + dataset.X.shape[1:]
        utils.export_model(
            model, input_shape=input_shape, output_dir=args.output_dir, opset_version=args.opset_version,
            quantization=args.quantization, example_input=example_input, generic_model=args.generic_model,
            remove_hooks_for_jit=True if (args.quantization_method == TinyMLQuantizationMethod.PTQ and args.quantization) else False,
            input_names=["input"])

    log_training_time(start_time)

    if args.gen_golden_vectors:
        generate_golden_vector_dir(args.output_dir)
        output_int = get_output_int_flag(args)
        generate_golden_vectors(args.output_dir, dataset, output_int, args.generic_model)

def run(args):
    """Run training with optional distributed mode."""
    run_distributed(main, args)


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()
    # Apply default output_int if not specified by user
    apply_output_int_default(arguments, 'radar_classification')
    run(arguments)
