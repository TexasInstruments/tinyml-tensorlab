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
Time series forecasting ONNX model testing script.
"""

import os
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import torch

from tinyml_tinyverse.common.datasets import GenericTSDatasetForecasting
from tinyml_tinyverse.common.utils import utils
from tinyml_tinyverse.common.utils.mdcl_utils import Logger

# Import common functions from base module
from ..common.test_onnx_base import (
    get_base_test_args_parser,
    prepare_transforms,
    load_onnx_model,
    run_distributed_test,
)

dataset_loader_dict = {'GenericTSDatasetForecasting': GenericTSDatasetForecasting}


def get_args_parser():
    """Create argument parser with forecasting-specific arguments."""
    parser = get_base_test_args_parser("This script loads time series dataset and tests a forecasting model using ONNX RT")

    # Override default loader-type for forecasting
    for action in parser._actions:
        if action.dest == 'loader_type':
            action.default = 'forecasting'
            break

    # Forecasting-specific arguments
    parser.add_argument('--forecast-horizon', help="Number of future timesteps to be predicted", type=int)
    parser.add_argument('--target-variables', help='Target variables to be predicted', default=[])

    return parser


def main(gpu, args):
    """Main testing function for forecasting."""
    transform = None
    if not args.output_dir:
        output_folder = os.path.basename(os.path.split(args.data_path)[0])
        args.output_dir = os.path.join('.', 'data', 'checkpoints', 'forecasting', output_folder, args.model, args.date)
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
    prepare_transforms(args)

    dataset, dataset_test, train_sampler, test_sampler = utils.load_data(
        args.data_path, args, dataset_loader_dict, test_only=True)

    logger.info("Loading data:")
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)

    logger.info(f"Loading ONNX model: {args.model_path}")
    ort_sess, input_name, output_name = load_onnx_model(args.model_path, args.generic_model)

    predicted = torch.tensor([]).to(device, non_blocking=True)
    ground_truth = torch.tensor([]).to(device, non_blocking=True)

    for _, batched_data, batched_target in data_loader_test:
        batched_data = batched_data.to(device, non_blocking=True).float()
        batched_target = batched_target.to(device, non_blocking=True).float()
        if transform:
            batched_data = transform(batched_data)
        for data in batched_data:
            predicted = torch.cat((predicted, torch.tensor(
                ort_sess.run([output_name], {input_name: data.unsqueeze(0).cpu().numpy()})[0]
            ).to(device)))
        ground_truth = torch.cat((ground_truth, batched_target))

    predicted = predicted.view_as(ground_truth)

    logger = getLogger("root.main.test_data")
    for idx, item in enumerate(dataset_test.header_row):
        for target_variable_name in item:
            logger.info(f"Variable {target_variable_name}:")
            logger.info(f"  SMAPE of {target_variable_name} across all predicted timesteps: {utils.smape(ground_truth[:, :, idx], predicted[:, :, idx]):.2f}%")
            logger.info(f"  R² of {target_variable_name} across all predicted timesteps: {utils.get_r2_score(predicted[:, :, idx], ground_truth[:, :, idx]):.4f}")

            # Log timestep specific metrics
            for step in range(args.forecast_horizon):
                logger.info(f"  Timestep {step + 1}:")
                logger.info(f"      SMAPE: {utils.smape(ground_truth[:, step, idx], predicted[:, step, idx]):.2f}%")
                logger.info(f"      R²: {utils.get_r2_score(predicted[:, step, idx], ground_truth[:, step, idx]):.4f}")

    # Save final predictions and create visualizations
    if args.output_dir and ground_truth is not None:
        results_dir = os.path.join(args.output_dir, 'test_results')
        os.makedirs(results_dir, exist_ok=True)

        # Save predictions in CSV format
        utils.save_forecasting_predictions_csv(
            ground_truth,
            predicted,
            results_dir,
            dataset_test.header_row,
            args.forecast_horizon,
        )

        plots_dir = os.path.join(results_dir, 'prediction_plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Create scatter plots for each variable
        for idx, item in enumerate(dataset_test.header_row):
            for target_variable_name in item:
                fig, axes = plt.subplots(int(np.ceil(args.forecast_horizon / 2)), 2, figsize=(12, 5))
                axes = axes.flatten()
                for step in range(args.forecast_horizon):
                    step_targets = ground_truth[:, step, idx]
                    step_outputs = predicted[:, step, idx]

                    step_smape = utils.smape(ground_truth[:, step, idx], predicted[:, step, idx])
                    step_r2 = utils.get_r2_score(predicted[:, step, idx], ground_truth[:, step, idx])

                    # Scatter plot
                    ax = axes[step]
                    ax.scatter(step_targets, step_outputs, alpha=0.5, label='Predictions')

                    # Add perfect prediction line
                    min_val = min(step_targets.min(), step_outputs.min())
                    max_val = max(step_targets.max(), step_outputs.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

                    ax.set_xlabel(f"Actual Variable {target_variable_name}")
                    ax.set_ylabel(f"Predicted Variable {target_variable_name}")
                    ax.set_title(f"{step + 1}-step ahead\nR² = {step_r2:.4f}, SMAPE = {step_smape:.2f}%")
                    ax.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{target_variable_name}_predictions.png'))
                plt.close()


def run(args):
    """Run testing with optional distributed mode."""
    run_distributed_test(main, args)


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()
    run(arguments)
