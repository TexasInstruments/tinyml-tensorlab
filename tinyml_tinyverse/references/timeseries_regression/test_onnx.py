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
Time series regression ONNX model testing script.
"""

import os
from logging import getLogger

import pandas as pd
import torch
import torcheval

from tinyml_tinyverse.common.datasets import GenericTSDataset, GenericTSDatasetReg
from tinyml_tinyverse.common.utils import misc_utils, utils, mdcl_utils
from tinyml_tinyverse.common.utils.mdcl_utils import Logger

# Import common functions from base module
from ..common.test_onnx_base import (
    get_base_test_args_parser,
    prepare_transforms,
    load_onnx_model,
    run_distributed_test,
)

dataset_loader_dict = {'GenericTSDataset': GenericTSDataset, 'GenericTSDatasetReg': GenericTSDatasetReg}


def get_args_parser():
    """Create argument parser with regression-specific arguments."""
    parser = get_base_test_args_parser("This script loads time series dataset and tests a regression model using ONNX RT")

    # Override default loader-type for regression
    for action in parser._actions:
        if action.dest == 'loader_type':
            action.default = 'regression'
            break

    return parser


def main(gpu, args):
    """Main testing function for regression."""
    transform = None
    if not args.output_dir:
        output_folder = os.path.basename(os.path.split(args.data_path)[0])
        args.output_dir = os.path.join('.', 'data', 'checkpoints', 'classification', output_folder, args.model, args.date)
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

    num_classes = len(dataset.classes)

    logger.info("Loading data:")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)

    logger.info(f"Loading ONNX model: {args.model_path}")
    ort_sess, input_name, output_name = load_onnx_model(args.model_path, args.generic_model)

    predicted = torch.tensor([]).to(device, non_blocking=True)
    ground_truth = torch.tensor([]).to(device, non_blocking=True)

    for _, batched_data, batched_target in data_loader:
        batched_data = batched_data.to(device, non_blocking=True).float()
        batched_target = batched_target.to(device, non_blocking=True).float()
        if transform:
            batched_data = transform(batched_data)
        for data in batched_data:
            predicted = torch.cat((predicted, torch.tensor(
                ort_sess.run([output_name], {input_name: data.unsqueeze(0).cpu().numpy()})[0]
            ).to(device)))
        ground_truth = torch.cat((ground_truth, batched_target))

    mdcl_utils.create_dir(os.path.join(args.output_dir, 'post_training_analysis'))
    logger.info("Plotting Regressions on dataset")

    metric = torcheval.metrics.MeanSquaredError()
    r2_score = torcheval.metrics.R2Score()

    df = pd.DataFrame({
        "predicted": predicted.to('cpu').numpy().flatten(),
        "ground_truth": ground_truth.to('cpu').numpy().flatten()
    })
    df.to_csv(os.path.join(args.output_dir, 'post_training_analysis', "results_on_test_set.csv"), index=False)
    logger.info(f"Outputs on the test set saved at : {os.path.join(args.output_dir, 'post_training_analysis', 'results_on_test_set.csv')}")

    utils.plot_actual_vs_predicted_regression(ground_truth.to('cpu'), predicted.to('cpu'),
                                               os.path.join(args.output_dir, 'post_training_analysis'), phase='test')
    utils.plot_residual_error_regression(ground_truth.to('cpu'), predicted.to('cpu'),
                                          os.path.join(args.output_dir, 'post_training_analysis'), phase='test')

    metric.update(predicted.to('cpu'), ground_truth.to('cpu'))
    r2_score.update(predicted.to('cpu'), ground_truth.to('cpu'))

    logger = getLogger("root.main.test_data")
    logger.info(f"Test Data Evaluation RMSE: {torch.sqrt(metric.compute()):.2f}")
    logger.info(f"Test Data Evaluation R2-Score: {r2_score.compute():.2f}")
    return


def run(args):
    """Run testing with optional distributed mode."""
    run_distributed_test(main, args)


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()
    run(arguments)
