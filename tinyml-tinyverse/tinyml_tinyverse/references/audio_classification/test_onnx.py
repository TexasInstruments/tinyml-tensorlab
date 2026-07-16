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


import datetime
import os
import platform
from argparse import ArgumentParser
from logging import getLogger

import onnxruntime as ort
import pandas as pd
import torch
import torcheval
from tabulate import tabulate

from tinyml_tinyverse.common.datasets import GoogleSpeechCommandsDataset

# Tiny ML TinyVerse Modules
from tinyml_tinyverse.common.utils import misc_utils, utils, mdcl_utils
from tinyml_tinyverse.common.utils.mdcl_utils import Logger
from tinyml_tinyverse.common.utils.utils import get_confusion_matrix

# Import common functions from base module
from ..common.test_onnx_base import (
    get_base_test_args_parser,
    prepare_transforms,
    load_onnx_model,
    run_distributed_test,
)

dataset_loader_dict = {'GoogleSpeechCommandsDataset': GoogleSpeechCommandsDataset}


def get_args_parser():
    DESCRIPTION = "This script loads time series dataset and tests it against a onnx model using ONNX RT"
    parser = get_base_test_args_parser("This script loads an audio wav dataset and tests a classification model")
   
    # Audio preprocessing / feature extraction params
    parser.add_argument('--sample-rate', help='Audio sample rate in Hz', default=16000, type=int)
    parser.add_argument('--audio-duration-ms', help='Audio clip duration in milliseconds', default=1000, type=int)
    parser.add_argument('--audio-feature', help='Audio feature type: MFCC, LPC, or RAW', default='MFCC', type=str)
    # MFCC params
    parser.add_argument('--n-mfcc', help='Number of MFCC coefficients', default=10, type=int)
    parser.add_argument('--n-mels', help='Number of Mel filterbank bins', default=40, type=int)
    parser.add_argument('--frame-length-ms', help='Frame/window length in milliseconds', default=30, type=int)
    parser.add_argument('--frame-step-ms', help='Frame step/hop length in milliseconds', default=20, type=int)

    # LPC params
    parser.add_argument('--nlpc', help='Number of LPC output features/filterbank energies', default=14, type=int)
    parser.add_argument('--lpc-order', help='LPC analysis order', default=14, type=int)

    # Audio loading params
    parser.add_argument('--normalize-audio', help='Normalize waveform by max absolute value', default=True, type=misc_utils.str2bool)
    parser.add_argument('--mono', help='Convert multi-channel audio to mono', default=True, type=misc_utils.str2bool)

    # Training parameters
    parser.add_argument('--file-level-classification-log', help='File-level classification Log File', type=str)
    parser.add_argument("--nn-for-feature-extraction", default=False, type=misc_utils.str2bool,
                        help="Use an AI model for preprocessing")
   
   
    return parser


def main(gpu, args):
    transform = None
    if not args.output_dir:
        output_folder = os.path.basename(os.path.split(args.data_path)[0])
        args.output_dir = os.path.join('.', 'data', 'checkpoints', 'classification', output_folder, args.model, args.date)
    utils.mkdir(args.output_dir)
    log_file = os.path.join(args.output_dir, 'run.log')
    logger = Logger(log_file=args.lis or log_file, DEBUG=args.DEBUG, name="root", append_log=True, console_log=True)
    # logger = command_display(args.lis or log_file, args.DEBUG)
    utils.seed_everything(args.seed)
    from tinyml_tinyverse.version import get_version_str
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

    predicted = torch.tensor([], dtype=torch.float32).to(device, non_blocking=True)
    ground_truth = torch.tensor([], dtype=torch.float32).to(device, non_blocking=True)
    for batched_raw_data, batched_data, batched_target in data_loader:
        batched_raw_data = batched_raw_data.long().to(device, non_blocking=True)
        batched_data = batched_data.float().to(device, non_blocking=True)
        batched_target = batched_target.long().to(device, non_blocking=True)
        if transform:
            batched_data = transform(batched_data)
        if args.nn_for_feature_extraction:
            for data in batched_raw_data:
                predicted = torch.cat((predicted, torch.tensor(
                    ort_sess.run([output_name], {input_name: data.unsqueeze(0).cpu().numpy().astype(np.float32)})[0]
                ).to(device)))
        else:
            for data in batched_data:
                predicted = torch.cat((predicted, torch.tensor(
                    ort_sess.run([output_name], {input_name: data.unsqueeze(0).cpu().numpy()})[0]
                ).to(device)))
        ground_truth = torch.cat((ground_truth, batched_target))

    try:
        mdcl_utils.create_dir(os.path.join(args.output_dir, 'post_training_analysis'))
        logger.info("Plotting OvR Multiclass ROC score")
        utils.plot_multiclass_roc(ground_truth, predicted, os.path.join(args.output_dir, 'post_training_analysis'),
                                  label_map=dataset.inverse_label_map, phase='test')
        logger.info("Plotting Class difference scores")
        utils.plot_pairwise_differenced_class_scores(ground_truth, predicted,
                                                      os.path.join(args.output_dir, 'post_training_analysis'),
                                                      label_map=dataset.inverse_label_map, phase='test')
    except Exception as e:
        logger.warning(f"Post Training Analysis plots will not be generated because: {e}")

    metric = torcheval.metrics.MulticlassAccuracy()
    # predicted = torch.argmax(predicted, dim=1)
    metric.update(predicted, ground_truth)
    logger = getLogger("root.main.test_data")
    logger.info(f"Test Data Evaluation Accuracy: {metric.compute() * 100:.2f}%")
    try:
        logger.info(
            f"Test Data Evaluation AUC ROC Score: {utils.get_au_roc(predicted, ground_truth, num_classes):.3f}")
    except ValueError as e:
        logger.warning("Not able to compute AUC ROC. Error: " + str(e))
    if len(torch.unique(ground_truth)) == 1:
        logger.warning("Confusion Matrix can not be printed because only items of 1 class was present in test data")
    else:
        try:
            confusion_matrix = get_confusion_matrix(predicted, ground_truth.type(torch.int64),
                                                    num_classes).cpu().numpy()
            logger.info('Confusion Matrix:\n {}'.format(tabulate(pd.DataFrame(
                confusion_matrix, columns=[f"Predicted as: {x}" for x in dataset.inverse_label_map.values()],
                index=[f"Ground Truth: {x}" for x in dataset.inverse_label_map.values()]), headers="keys", tablefmt='grid')))
        except ValueError as e:
            logger.warning("Not able to compute Confusion Matrix. Error: " + str(e))

        try:
            Logger(log_file=args.file_level_classification_log, DEBUG=args.DEBUG,
                   name="root.utils.print_file_level_classification_summary", append_log=True, console_log=False)
            getLogger("root.utils.print_file_level_classification_summary").propagate = False
            utils.print_file_level_classification_summary(dataset_test, predicted, ground_truth, "TestData")
            logger.info(f"Generated File-level classification summary of test data in: {args.file_level_classification_log}")
        except Exception as e:
            logger.error(f"Failed to generate file-level classification summary: {str(e)}")

    return

def run(args):
    """Run testing with optional distributed mode."""
    run_distributed_test(main, args)


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()
    run(arguments)