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

"""
Radar Classification ONNX model Testing Pipeline.
"""
import os

from logging import getLogger

import numpy as np
import pandas as pd

import platform

import torch
import torcheval
from tabulate import tabulate


from tinyml_tinyverse.common.datasets import GenericRadarDataset


# Tiny ML TinyVerse Modules
from tinyml_tinyverse.common.utils import misc_utils, utils, mdcl_utils
from tinyml_tinyverse.common.utils.mdcl_utils import Logger
from tinyml_tinyverse.common.utils.utils import get_confusion_matrix

# Import common functions from base module
from ..common.test_onnx_base import (
    get_base_test_args_parser,
    load_onnx_model,
    run_distributed_test,
)
from ..common.train_base import shutdown_data_loaders

dataset_loader_dict = {'GenericRadarDataset': GenericRadarDataset}

def get_args_parser():
    """Create argument parser with classification-specific arguments."""
    
    parser = get_base_test_args_parser("This script loads a radar point cloud dataset and tests a classification model using ONNX RT")
    # Classification-specific arguments

    parser.add_argument('--gain-variations', help='Gain Variation Dictionary to be applied to each of the classes')
    parser.add_argument('--q15-scale-factor', help="q15 scaling factor")
    parser.add_argument('--file-level-classification-log', help='File-level classification Log File', type=str)
    parser.add_argument("--nn-for-feature-extraction", default=False, type=misc_utils.str2bool,
                            help="Use an AI model for preprocessing")

    # Radar Pre Processing Pipeline Arguments
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
    parser.add_argument('--radar-min-points-per-frame', help="Min real (pre-padding) points required per frame before it's dropped")
    parser.add_argument('--radar-point-cloud-structure-strategy',
                            help="'highest only' / 'lowest only' / 'highest and lowest' - which points to keep, per frame")
    parser.add_argument('--radar-point-cloud-sort-by', default='height',
                            help="Which selected point feature ('distance'/'height'/'azimuth'/'doppler'/'snr') to rank points by "
                                "before applying --radar-point-cloud-structure-strategy")
    parser.add_argument('--radar-min-points-per-window', help="Min real points required across entire window (sum of per-frame points) before window is dropped (for per_window mode)")
    parser.add_argument('--radar-frames-quality-index',
                            help="Set to 'per_radar_frame' to enable per-frame point-count quality filtering during windowing")
    parser.add_argument('--radar-feature-ordering',
                             help="Choose the method for window concatenation and flattening - Either 'frame_major' or 'feature_major'")

    return parser

def main(gpu, args):
    """Main testing function for classification."""
    transform = None
    if not args.output_dir:
        output_folder = os.path.basename(os.path.split(args.data_path)[0])
        args.output_dir = os.path.join('.', 'data', 'checkpoints', 'classification', output_folder, args.model, args.date)
    utils.mkdir(args.output_dir)
    log_file = os.path.join(args.output_dir, 'run.log')
    logger = Logger(log_file=args.lis or log_file, DEBUG=args.DEBUG, name="root", append_log=True, console_log=True)
   
    utils.seed_everything(args.seed)
    from tinyml_tinyverse.version import get_version_str
    logger.info(f"TinyVerse Toolchain Version: {get_version_str()}")
    logger.info("Script: {}".format(os.path.relpath(__file__)))

    utils.init_distributed_mode(args)
    logger.debug("Args: {}".format(args))

    device = torch.device(args.device)

    dataset, dataset_test, train_sampler, test_sampler = utils.load_data(
        args.data_path, args, dataset_loader_dict, test_only=True)

    if isinstance(args.data_proc_transforms, list):
        if len(args.data_proc_transforms) and isinstance(args.data_proc_transforms[0], list):
            args.transforms = args.data_proc_transforms[0] + args.feat_ext_transform
        else:
            args.transforms = args.data_proc_transforms + args.feat_ext_transform

    num_classes = len(dataset.classes)

    logger.info("Loading data:")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True,
        multiprocessing_context='fork' if platform.system() == 'Linux' and args.workers > 0 else None,
        collate_fn=utils.collate_fn)
    
    try:
        logger.info(f"Loading ONNX model: {args.model_path}")
        ort_sess, input_name, output_name = load_onnx_model(args.model_path, args.generic_model)

        # See evaluate_classification (common/utils/utils.py) for why non_blocking must be gated on CUDA.
        non_blocking = (device.type == 'cuda')
        predicted = torch.tensor([]).to(device, non_blocking=non_blocking)
        ground_truth = torch.tensor([]).to(device, non_blocking=non_blocking)
        for _, batched_data, batched_target in data_loader:
            batched_data = batched_data.float().to(device, non_blocking=non_blocking)
            batched_target = batched_target.long().to(device, non_blocking=non_blocking)
            if transform:
                batched_data = transform(batched_data)
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

        metric.update(predicted, ground_truth)
        logger = getLogger("root.main.test_data")
        logger.info(f"Test Data Evaluation Accuracy: {metric.compute() * 100:.2f}%")
        try:
            logger.info(f"Test Data Evaluation AUC ROC Score: {utils.get_au_roc(predicted, ground_truth, num_classes):.3f}")
        except ValueError as e:
            logger.warning("Not able to compute AUC ROC. Error: " + str(e))
        if len(torch.unique(ground_truth)) == 1:
            logger.warning("Confusion Matrix can not be printed because only items of 1 class was present in test data")
        else:
            try:
                confusion_matrix = get_confusion_matrix(predicted, ground_truth.type(torch.int64), num_classes).cpu().numpy()
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
    finally:
        shutdown_data_loaders(data_loader)
    return

def run(args):
    """Run testing with optional distributed mode."""
    run_distributed_test(main, args)


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()
    run(arguments)