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
Time series anomaly detection ONNX model testing script.
"""

import csv
import os
from logging import getLogger

import numpy as np
import torch
import pandas as pd
from tabulate import tabulate

from tinyml_tinyverse.common.datasets import GenericTSDataset, GenericTSDatasetReg, GenericTSDatasetAD
from tinyml_tinyverse.common.utils import utils, mdcl_utils
from tinyml_tinyverse.common.utils.mdcl_utils import Logger

# Import common functions from base module
from ..common.test_onnx_base import (
    get_base_test_args_parser,
    prepare_transforms,
    load_onnx_model,
    run_distributed_test,
)

dataset_loader_dict = {
    'GenericTSDataset': GenericTSDataset,
    'GenericTSDatasetReg': GenericTSDatasetReg,
    'GenericTSDatasetAD': GenericTSDatasetAD,
}


def get_args_parser():
    """Create argument parser with anomaly detection-specific arguments."""
    parser = get_base_test_args_parser("This script loads time series dataset and tests an anomaly detection model using ONNX RT")

    # Override defaults for anomaly detection
    for action in parser._actions:
        if action.dest == 'loader_type':
            action.default = 'anomalydetection'
        elif action.dest == 'opt':
            action.default = 'Adam'
        elif action.dest == 'lr':
            action.default = 0.001

    return parser


def get_model_performance(threshold, normal_errors, anomaly_errors):
    """Calculate model performance metrics for a given threshold."""
    normal_detected_as_anomaly = sum(1 for x in normal_errors if x > threshold)
    anomaly_detected_as_anomaly = sum(1 for x in anomaly_errors if x > threshold)
    normal_detected_as_normal = len(normal_errors) - normal_detected_as_anomaly
    anomaly_detected_as_normal = len(anomaly_errors) - anomaly_detected_as_anomaly

    true_positives = anomaly_detected_as_anomaly
    true_negatives = normal_detected_as_normal
    false_positives = normal_detected_as_anomaly
    false_negatives = anomaly_detected_as_normal

    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_negatives + false_positives) * 100
    precision = true_positives / (true_positives + false_positives) * 100 if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) * 100 if true_positives + false_negatives > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    false_positive_rate = false_positives / (true_negatives + false_positives) * 100 if true_negatives + false_positives > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'false_positive_rate': false_positive_rate,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def get_reconstruction_errors_stats(args):
    """Calculate reconstruction error statistics from training data."""
    log_file = os.path.join(args.output_dir, 'run.log')
    logger = Logger(log_file=args.lis or log_file, DEBUG=args.DEBUG, name="root", append_log=True, console_log=True)
    utils.seed_everything(args.seed)
    device = torch.device(args.device)

    prepare_transforms(args)
    dataset, dataset_test, train_sampler, test_sampler = utils.load_data(args.data_path, args, dataset_loader_dict)

    logger.info("Loading data:")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True if args.gpu > 0 else False, collate_fn=utils.collate_fn)

    logger.info(f"Loading ONNX model: {args.model_path}")
    ort_sess, input_name, output_name = load_onnx_model(args.model_path, args.generic_model)

    errors = torch.tensor([]).to(device, non_blocking=True)
    for _, data, targets in data_loader:
        data = data.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).long()
        batch_reconstruction_errors = torch.tensor([]).to(device, non_blocking=True)
        for input, target_label in zip(data, targets):
            input = input.unsqueeze(0).cpu().numpy()
            output = torch.tensor(ort_sess.run([output_name], {input_name: input})[0]).to(device)
            current_output_error = torch.mean((torch.from_numpy(input).to(device) - output) ** 2, dim=(1, 2, 3))
            batch_reconstruction_errors = torch.cat((batch_reconstruction_errors, current_output_error))
        errors = torch.cat((errors, batch_reconstruction_errors))

    normal_error_mean = torch.mean(errors)
    normal_error_std = torch.std(errors)
    return normal_error_mean.cpu(), normal_error_std.cpu()


def main(gpu, args):
    """Main testing function for anomaly detection."""
    transform = None
    if not args.output_dir:
        output_folder = os.path.basename(os.path.split(args.data_path)[0])
        args.output_dir = os.path.join('.', 'data', 'checkpoints', 'anomalydetection', output_folder, args.model, args.date)
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
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True if gpu > 0 else False, collate_fn=utils.collate_fn)

    logger.info(f"Loading ONNX model: {args.model_path}")
    ort_sess, input_name, output_name = load_onnx_model(args.model_path, args.generic_model)

    errors = torch.tensor([]).to(device, non_blocking=True)
    ground_truth = torch.tensor([]).to(device, non_blocking=True)

    for _, data, targets in data_loader:
        data = data.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).long()
        if transform:
            data = transform(data)
        batch_reconstruction_errors = torch.tensor([]).to(device, non_blocking=True)
        batch_target_labels = torch.tensor([]).to(device, non_blocking=True)
        for input, target_label in zip(data, targets):
            input = input.unsqueeze(0).cpu().numpy()
            output = torch.tensor(ort_sess.run([output_name], {input_name: input})[0]).to(device)
            current_output_errors = torch.mean((torch.from_numpy(input).to(device) - output) ** 2, dim=(1, 2, 3))
            batch_reconstruction_errors = torch.cat((batch_reconstruction_errors, current_output_errors))
            batch_target_labels = torch.cat((batch_target_labels, target_label))
        errors = torch.cat((errors, batch_reconstruction_errors))
        ground_truth = torch.cat((ground_truth, batch_target_labels))

    post_training_analysis_path = os.path.join(args.output_dir, 'post_training_analysis')
    mdcl_utils.create_dir(post_training_analysis_path)

    # The classes folder in dataset should have two folders named Anomaly and Normal
    anomaly_errors = errors[ground_truth == 0].cpu().numpy()
    normal_errors = errors[ground_truth == 1].cpu().numpy()
    logger.info("Plotting reconstructions errors")

    normal_train_mean, normal_train_std = get_reconstruction_errors_stats(args)
    anomaly_test_mean = np.mean(anomaly_errors)
    anomaly_test_std = np.std(anomaly_errors)
    normal_test_mean = np.mean(normal_errors)
    normal_test_std = np.std(normal_errors)

    # Results
    logger.info(f"Reconstruction Error Statistics:")
    logger.info(f"Normal training data - Mean: {normal_train_mean:.6f}, Std: {normal_train_std:.6f}")
    logger.info(f"Anomaly test data - Mean: {anomaly_test_mean:.6f}, Std: {anomaly_test_std:.6f}")
    logger.info(f"Normal test data - Mean: {normal_test_mean:.6f}, Std: {normal_test_std:.6f}")

    # Threshold - K is the number of standard deviations from the mean
    all_k_values = [i * 0.5 for i in range(0, 10)]
    results_data = []
    best_f1_score = 0
    best_f1_score_index = 0

    for i, k in enumerate(all_k_values):
        threshold = normal_train_mean + k * normal_train_std
        results = get_model_performance(threshold, normal_errors, anomaly_errors)
        results["k_value"] = k
        results["threshold"] = float(threshold)
        results_data.append(results)
        if results["f1_score"] > best_f1_score:
            best_f1_score_index = i
            best_f1_score = results["f1_score"]

    csv_path = os.path.join(post_training_analysis_path, 'threshold_performance.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['k_value', 'threshold', 'accuracy', 'precision', 'recall',
                      'f1_score', 'false_positive_rate', 'true_positives',
                      'true_negatives', 'false_positives', 'false_negatives']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results_data:
            for key in ['accuracy', 'precision', 'recall', 'f1_score', 'false_positive_rate']:
                if key in result:
                    result[key] = round(result[key], 2)
            writer.writerow(result)

    logger.info(f"Threshold performance data saved to {csv_path}")

    best_results = results_data[best_f1_score_index]
    best_threshold = best_results["threshold"]
    logger.info(f"Threshold for K = {best_results['k_value']} : {best_threshold:.6f}")

    utils.plot_reconstruction_errors(anomaly_errors, normal_errors, normal_train_mean, best_threshold, post_training_analysis_path)
    utils.plot_reconstruction_errors(anomaly_errors, normal_errors, normal_train_mean, best_threshold, post_training_analysis_path, log_scale=True)

    logger.info(f"False positive rate: {best_results['false_positive_rate']:.2f}%")
    logger.info(f"Anomaly detection rate (recall): {best_results['recall']:.2f}%")
    logger.info(f"Accuracy: {best_results['accuracy']:.2f}%")
    logger.info(f"Precision: {best_results['precision']:.2f}%")
    logger.info(f"F1 Score: {best_results['f1_score']:.2f}%")

    
    # Create a 2x2 confusion matrix
    confusion_matrix = np.array([
        [best_results['true_negatives'], best_results['false_positives']],
        [best_results['false_negatives'], best_results['true_positives']]
    ])

    # Format using pandas and tabulate
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix, 
        columns=["Predicted as: Normal", "Predicted as: Anomaly"],
        index=["Ground Truth: Normal", "Ground Truth: Anomaly"]
    )

    logger.info('Confusion Matrix:\n {}'.format(tabulate(confusion_matrix_df, headers="keys", tablefmt='grid')))


def run(args):
    """Run testing with optional distributed mode."""
    run_distributed_test(main, args)


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()
    run(arguments)
