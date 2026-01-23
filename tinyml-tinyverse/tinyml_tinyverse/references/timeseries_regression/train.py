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
Time series regression training script.
"""

import os
import random
import sys
import timeit
from logging import getLogger

import numpy as np

import torch
import torch.nn as nn

from tinyml_tinyverse.common.datasets import GenericTSDatasetReg
from tinyml_tinyverse.common.utils import misc_utils, utils

# Import common functions from base module
from ..common.train_base import (
    get_base_args_parser,
    generate_golden_vector_dir,
    generate_user_input_config,
    generate_test_vector,
    load_datasets,
    run_distributed,
    assemble_golden_vectors_header,
    setup_training_environment,
    prepare_transforms,
    create_model,
    log_model_summary,
    load_pretrained_weights,
    setup_optimizer_and_scheduler,
    setup_distributed_model,
    resume_from_checkpoint,
    save_checkpoint,
    handle_export_only,
    move_model_to_device,
    export_trained_model,
    log_training_time,
    apply_output_int_default,
    get_output_int_flag,
    load_onnx_for_inference,
    create_data_loaders,
)

dataset_loader_dict = {'GenericTSDatasetReg': GenericTSDatasetReg}
dataset_load_state = {'dataset': None, 'dataset_test': None, 'train_sampler': None, 'test_sampler': None}


def get_args_parser():
    """Create argument parser with regression-specific arguments."""
    parser = get_base_args_parser("This script loads time series data and trains a regression model")

    # Override default loader-type for regression
    for action in parser._actions:
        if action.dest == 'loader_type':
            action.default = 'regression'
            break

    # Regression-specific arguments
    parser.add_argument('--lambda-reg', default=0.01, type=float, help='lambda for L1 & L2 normalization in regression')
    return parser


def generate_golden_vectors(output_dir, output_int, dataset, generic_model=False):
    """Generate golden vectors for regression."""
    logger = getLogger("root.generate_golden_vectors")
    ort_sess, input_name, output_name = load_onnx_for_inference(output_dir, generic_model)
    vector_files = []

    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    logger.info(f"Creating Golden data for reference at {golden_vectors_dir}")

    for index in random.sample(range(len(dataset)), k=8):
        np_raw = dataset.X_raw[index]
        np_feat = dataset.X[index]
        pred = ort_sess.run([output_name], {input_name: np.expand_dims(np_feat, 0).astype(np.float32)})[0]
        half_path = os.path.join(golden_vectors_dir)

        np.savetxt(half_path + f'adc_{index}.txt', np_raw.flatten(),
                   fmt='%f,' if np_raw.dtype.kind == 'f' else '%d,',
                   header=f'//(Index: {index}): ADC Data\nfloat raw_input_test[{len(np_raw.flatten())}]= {{',
                   footer='}', comments='', newline=' ')
        vector_files.append(half_path + f'adc_{index}.txt')
        np.savetxt(half_path + f'features_{index}.txt', np_feat.flatten(), fmt='%.5f,',
                   header=f'//(Index: {index}): Extracted Features\nfloat model_test_input[{len(np_feat.flatten())}] = {{',
                   footer='}', comments='', newline=' ')
        vector_files.append(half_path + f'features_{index}.txt')
        np.savetxt(half_path + f'output_{index}.txt', pred.flatten(),
                   fmt='%d,' if output_int else '%f,',
                   header=f'//(Index: {index}): Expected Model Output\n{"int8_t" if output_int else "float"} golden_output[{len(pred.flatten())}] = {{',
                   footer='}', comments='', newline=' ')
        vector_files.append(half_path + f'output_{index}.txt')

    header_file_info = assemble_golden_vectors_header(vector_files, files_per_set=3)
    generate_test_vector(output_dir, header_file_info)


def main(gpu, args):
    """Main training function for regression."""
    logger, device = setup_training_environment(args, gpu, 'classification', __file__)  # Uses 'classification' path like original
    prepare_transforms(args)

    # Load or reuse datasets
    if not args.quantization:
        dataset, dataset_test, train_sampler, test_sampler = load_datasets(args.data_path, args, dataset_loader_dict)
        dataset_load_state['dataset'], dataset_load_state['dataset_test'] = dataset, dataset_test
        dataset_load_state['train_sampler'], dataset_load_state['test_sampler'] = train_sampler, test_sampler
    else:
        dataset, dataset_test, train_sampler, test_sampler = (dataset_load_state['dataset'], dataset_load_state['dataset_test'],
                                                               dataset_load_state['train_sampler'], dataset_load_state['test_sampler'])

    try:
        utils.plot_feature_components_graph(dataset, graph_type='pca', instance_type='train', output_dir=args.output_dir)
        utils.plot_feature_components_graph(dataset_test, graph_type='pca', instance_type='validation', output_dir=args.output_dir)
    except Exception as e:
        logger.warning(f"Feature Extraction plots will not be generated because: {e}")

    if misc_utils.str2bool(args.dont_train_just_feat_ext):
        logger.info("Exiting execution without training")
        sys.exit(0)

    generate_golden_vector_dir(args.output_dir)
    if misc_utils.str2bool(args.gen_golden_vectors):
        generate_user_input_config(args.output_dir, dataset)
        if misc_utils.str2bool(args.dont_train_just_feat_ext):
            logger.info('ModelMaker completed for test bench. Exiting.')
            sys.exit()

    num_classes = len(dataset.classes)
    variables = dataset.X.shape[1]
    input_features = dataset.X.shape[2]

    logger.info("Loading data:")
    data_loader, data_loader_test = create_data_loaders(dataset, dataset_test, train_sampler, test_sampler, args, gpu)

    logger.info("Creating model")
    model = create_model(args, variables, num_classes, input_features, logger)
    log_model_summary(model, args, variables, input_features, logger)
    model = load_pretrained_weights(model, args, logger)

    # if output_int not set by user, then set it to default of task_type
    if args.output_int == None:
        args.output_int = False
    model = utils.quantization_wrapped_model(
        model, args.quantization, args.quantization_method, args.weight_bitwidth, args.activation_bitwidth,
        args.epochs, args.output_int, args.partial_quantization)
    
    if handle_export_only(model, args, variables, input_features, logger):
        return

    move_model_to_device(model, device, logger)
    criterion = nn.MSELoss()

    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, args)
    model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)
    resume_from_checkpoint(model_without_ddp, optimizer, lr_scheduler, model_ema, args)

    phase = 'QuantTrain' if args.quantization else 'FloatTrain'
    logger.info("Start training")
    start_time = timeit.default_timer()
    best = dict(mse=np.inf, r2=0, epoch=None)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        utils.train_one_epoch_regression(
            model, criterion, optimizer, data_loader, device, epoch, None, args.lambda_reg, args.apex, model_ema,
            print_freq=args.print_freq, phase=phase, num_classes=num_classes, dual_op=args.dual_op,
            is_ptq=True if (args.quantization_method in ['PTQ'] and args.quantization) else False)
        if not (args.quantization_method in ['PTQ'] and args.quantization):
            lr_scheduler.step()
        logger.info(f"Epoch : {epoch}")
        avg_mse, avg_r2_score = utils.evaluate_regression(model, criterion, data_loader_test, device=device,
                                                           transform=None, phase=phase, num_classes=num_classes, dual_op=args.dual_op)
        if model_ema:
            avg_mse, avg_r2_score = utils.evaluate_regression(
                model_ema, criterion, data_loader_test, device=device, transform=None,
                log_suffix='EMA', print_freq=args.print_freq, phase=phase, dual_op=args.dual_op)
        if args.output_dir and avg_mse <= best['mse']:
            logger.info(f"Epoch {epoch}: {avg_mse:.2f} (Val MSE) <= {best['mse']:.2f} (So far least error). Hence updating checkpoint.pth")
            best['mse'], best['r2'], best['epoch'] = avg_mse, avg_r2_score, epoch
            checkpoint = save_checkpoint(model_without_ddp, optimizer, lr_scheduler, epoch, args, model_ema)
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

    # Log best epoch results
    logger = getLogger(f"root.main.{phase}.BestEpoch")
    logger.info("")
    logger.info("Printing statistics of best epoch:")
    logger.info(f"Best Epoch: {best['epoch']}")
    logger.info(f"MSE {best['mse']:.3f}")
    logger.info(f"R2-Score {best['r2']:.3f}")
    logger.info("")

    export_trained_model(model, args, dataset)
    log_training_time(start_time)

    if args.gen_golden_vectors:
        generate_golden_vector_dir(args.output_dir)
        output_int = get_output_int_flag(args)
        generate_golden_vectors(args.output_dir, output_int, dataset, args.generic_model)


def run(args):
    """Run training with optional distributed mode."""
    run_distributed(main, args)


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()
    # Apply default output_int if not specified by user
    apply_output_int_default(arguments, 'timeseries_regression')
    run(arguments)
