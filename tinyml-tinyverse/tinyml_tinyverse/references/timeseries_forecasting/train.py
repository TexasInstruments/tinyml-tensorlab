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
Time series forecasting training script.
"""

import os
import random
import sys
import timeit
from logging import getLogger

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from tinyml_tinyverse.common import models
from tinyml_tinyverse.common.datasets import GenericTSDatasetForecasting
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
    log_model_summary,
    load_pretrained_weights,
    setup_optimizer_and_scheduler,
    setup_distributed_model,
    resume_from_checkpoint,
    save_checkpoint,
    handle_export_only,
    export_trained_model,
    log_training_time,
    apply_output_int_default,
    get_output_int_flag,
    load_onnx_for_inference,
    create_data_loaders,
)

dataset_loader_dict = {'GenericTSDatasetForecasting': GenericTSDatasetForecasting}
dataset_load_state = {'dataset': None, 'dataset_test': None, 'train_sampler': None, 'test_sampler': None}


def get_args_parser():
    """Create argument parser with forecasting-specific arguments."""
    parser = get_base_args_parser("This script loads time series data and trains a forecasting model")

    # Override default loader-type for forecasting
    for action in parser._actions:
        if action.dest == 'loader_type':
            action.default = 'forecasting'
            break

    # Forecasting-specific arguments
    parser.add_argument('--forecast-horizon', help="Number of future timesteps to be predicted", type=int)
    parser.add_argument('--target-variables', help='Target variables to be predicted', default=[])
    return parser


def generate_golden_vectors(output_dir, output_int, dataset, generic_model=False):
    """Generate golden vectors for forecasting."""
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
    """Main training function for forecasting."""
    logger, device = setup_training_environment(args, gpu, 'forecasting', __file__)
    prepare_transforms(args)

    # Load or reuse datasets
    if args.quantization:
        dataset, dataset_test, train_sampler, test_sampler = (dataset_load_state['dataset'], dataset_load_state['dataset_test'],
                                                               dataset_load_state['train_sampler'], dataset_load_state['test_sampler'])
    else:
        dataset, dataset_test, train_sampler, test_sampler = load_datasets(args.data_path, args, dataset_loader_dict)
        dataset_load_state['dataset'], dataset_load_state['dataset_test'] = dataset, dataset_test
        dataset_load_state['train_sampler'], dataset_load_state['test_sampler'] = train_sampler, test_sampler

    if misc_utils.str2bool(args.dont_train_just_feat_ext):
        logger.info("Exiting execution without training")
        sys.exit(0)

    generate_golden_vector_dir(args.output_dir)
    if misc_utils.str2bool(args.gen_golden_vectors):
        generate_user_input_config(args.output_dir, dataset)
        if misc_utils.str2bool(args.dont_train_just_feat_ext):
            logger.info('ModelMaker completed for test bench. Exiting.')
            sys.exit()

    num_target_variables = dataset.Y.shape[-1] if dataset.Y is not None else 1
    total_forecast_outputs = args.forecast_horizon * num_target_variables
    variables = dataset.X.shape[1]
    input_features = dataset.X.shape[2]

    logger.info("Loading data:")
    data_loader, data_loader_test = create_data_loaders(dataset, dataset_test, train_sampler, test_sampler, args, gpu)

    logger.info("Creating model")
    model = models.get_model(
        args.model, variables, total_forecast_outputs, input_features=input_features, model_config=args.model_config,
        model_spec=args.model_spec,
        dual_op=args.dual_op)

    log_model_summary(model, args, variables, input_features, logger)
    model = load_pretrained_weights(model, args, logger)

    # if output_int not set by user, then set it to default of task_type
    if args.output_int == None:
        args.output_int = False
    model = utils.quantization_wrapped_model(
        model, args.quantization, args.quantization_method, args.weight_bitwidth, args.activation_bitwidth,
        args.epochs, args.output_int)

    if handle_export_only(model, args, variables, input_features, logger):
        return

    model.to(device)
    criterion = nn.HuberLoss()

    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, args)
    model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)
    resume_from_checkpoint(model_without_ddp, optimizer, lr_scheduler, model_ema, args)

    phase = 'QuantTrain' if args.quantization else 'FloatTrain'
    logger.info("Start training")
    start_time = timeit.default_timer()

    best_epoch_values = {
        'epoch': -1,
        'true_values': None,
        'predictions': None,
        'overall_smape': float('inf'),
    }

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        utils.train_one_epoch_forecasting(
            model, criterion, optimizer, data_loader, device, epoch, None, args.apex, model_ema,
            print_freq=args.print_freq, phase=phase, num_classes=total_forecast_outputs, dual_op=args.dual_op,
            is_ptq=True if (args.quantization_method in ['PTQ'] and args.quantization) else False)

        if not (args.quantization_method in ['PTQ'] and args.quantization):
            lr_scheduler.step()

        target_tensor, prediction_tensor, overall_smape = utils.evaluate_forecasting(
            model, criterion, data_loader_test, device=device, transform=None, phase=phase,
            num_classes=total_forecast_outputs, dual_op=args.dual_op)

        if model_ema:
            target_tensor, prediction_tensor, overall_smape = utils.evaluate_forecasting(
                model_ema, criterion, data_loader_test, device=device, transform=None,
                log_suffix='EMA', print_freq=args.print_freq, phase=phase, dual_op=args.dual_op)

        if overall_smape < best_epoch_values['overall_smape']:
            best_epoch_values['overall_smape'] = overall_smape
            best_epoch_values['epoch'] = epoch
            best_epoch_values['true_values'] = target_tensor.clone()
            best_epoch_values['predictions'] = prediction_tensor.clone()

            if args.output_dir:
                checkpoint = save_checkpoint(model_without_ddp, optimizer, lr_scheduler, epoch, args, model_ema,
                                             extra_data={'metrics': {'overall_smape': overall_smape}})
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

        logger.info(f"Epoch {epoch}: Best Overall SMAPE across all variables across all predicted timesteps so far: {best_epoch_values['overall_smape']:.2f}% (Epoch {best_epoch_values['epoch']})")

    # Log best epoch metrics
    logger = getLogger(f"root.main.{phase}.BestEpoch")
    logger.info("Printing statistics of best epoch:")
    logger.info(f"Best epoch:{best_epoch_values['epoch'] + 1}")
    logger.info(f"Overall SMAPE across all variables: {best_epoch_values['overall_smape']:.2f}%")
    logger.info("Per-Variable Metrics:")

    for idx, item in enumerate(dataset.header_row):
        for target_variable_name in item:
            logger.info(f"  Variable {target_variable_name}:")
            logger.info(f"      SMAPE of {target_variable_name} across all predicted timesteps: {utils.smape(best_epoch_values['true_values'][:, :, idx], best_epoch_values['predictions'][:, :, idx]):.2f}%")
            logger.info(f"      R² of {target_variable_name} across all predicted timesteps: {utils.get_r2_score(best_epoch_values['predictions'][:, :, idx], best_epoch_values['true_values'][:, :, idx]):.4f}")

            for step in range(args.forecast_horizon):
                logger.info(f"      Timestep {step + 1}:")
                logger.info(f"          SMAPE: {utils.smape(best_epoch_values['true_values'][:, step, idx], best_epoch_values['predictions'][:, step, idx]):.2f}%")
                logger.info(f"          R²: {utils.get_r2_score(best_epoch_values['predictions'][:, step, idx], best_epoch_values['true_values'][:, step, idx]):.4f}")

    # Save final predictions and create visualizations for best epoch
    if args.output_dir and best_epoch_values['true_values'] is not None:
        results_dir = os.path.join(args.output_dir, f'best_epoch_{best_epoch_values["epoch"]}_results')
        os.makedirs(results_dir, exist_ok=True)

        utils.save_forecasting_predictions_csv(
            best_epoch_values['true_values'],
            best_epoch_values['predictions'],
            results_dir,
            dataset.header_row,
            args.forecast_horizon,
        )

        plots_dir = os.path.join(results_dir, 'prediction_plots')
        os.makedirs(plots_dir, exist_ok=True)

        for idx, item in enumerate(dataset.header_row):
            for target_variable_name in item:
                fig, axes = plt.subplots(int(np.ceil(args.forecast_horizon / 2)), 2, figsize=(12, 5))
                axes = axes.flatten()
                for step in range(args.forecast_horizon):
                    step_targets = best_epoch_values['true_values'][:, step, idx]
                    step_outputs = best_epoch_values['predictions'][:, step, idx]
                    step_smape = utils.smape(best_epoch_values['true_values'][:, step, idx], best_epoch_values['predictions'][:, step, idx])
                    step_r2 = utils.get_r2_score(best_epoch_values['predictions'][:, step, idx], best_epoch_values['true_values'][:, step, idx])

                    # Convert to numpy for matplotlib plotting
                    step_targets_np = step_targets.detach().cpu().numpy() if isinstance(step_targets, torch.Tensor) else step_targets
                    step_outputs_np = step_outputs.detach().cpu().numpy() if isinstance(step_outputs, torch.Tensor) else step_outputs

                    ax = axes[step]
                    ax.scatter(step_targets_np, step_outputs_np, alpha=0.5, label='Predictions')
                    min_val = min(step_targets_np.min(), step_outputs_np.min())
                    max_val = max(step_targets_np.max(), step_outputs_np.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
                    ax.set_xlabel(f"Actual Variable {target_variable_name}")
                    ax.set_ylabel(f"Predicted Variable {target_variable_name}")
                    ax.set_title(f"{step + 1}-step ahead\nR² = {step_r2:.4f},SMAPE = {step_smape:.2f}%")
                    ax.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{target_variable_name}_predictions.png'))
                plt.close()

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
    apply_output_int_default(arguments, 'timeseries_forecasting')
    run(arguments)
