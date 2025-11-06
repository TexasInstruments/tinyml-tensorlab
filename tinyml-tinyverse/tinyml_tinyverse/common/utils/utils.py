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

import base64
import copy
import datetime
import errno
import hashlib
import numbers
import os
import random
import re
import timeit
from itertools import combinations
from collections import OrderedDict, defaultdict, deque
from glob import glob
from logging import getLogger
from os.path import basename as opb

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np
import onnx
import pandas as pd
import torch
import torch.distributed as dist
from cryptography.fernet import Fernet
from tabulate import tabulate
from torcheval.metrics.functional import multiclass_confusion_matrix, multiclass_f1_score, multiclass_auroc, r2_score, mean_squared_error

from ..models.generic_models import FEModel  # , FEModel2
from tinyml_torchmodelopt.quantization import (
    TinyMLQuantizationVersion, TinyMLQuantizationMethod, TinyMLQConfigType,
    GenericTinyMLQATFxModule, TINPUTinyMLQATFxModule, GenericTinyMLPTQFxModule, TINPUTinyMLPTQFxModule)

try:
    from apex import amp
except ImportError:
    amp = None


def all_tensors_have_same_dimensions(tensors):
    """Checks if all tensors in a list are of the same dimensions.

    Args:
    tensors: A list of tensors.

    Returns:
    True if all tensors in the list are of the same dimensions, False otherwise.
    """
    # Check if the list is empty.
    if not tensors:
        return True
    # Get the dimensions of the first tensor.
    first_tensor_dimensions = tensors[0].shape
    # Check if the dimensions of all other tensors match the dimensions of the first tensor.
    for tensor in tensors[1:]:
        if tensor.shape != first_tensor_dimensions:
            return False
    # If all tensors have the same dimensions, return True.
    return True

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    if batch.ndim == 3:
        return batch.permute(0, 2, 1)
    else:
        return batch


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    raw_tensors, tensors, targets = [], [], []
    # Gather in lists, and encode labels as indices
    # for waveform, _, label, *_ in batch:
    for raw_sequence, sequence, label in batch:
        raw_tensors += [raw_sequence]
        tensors += [sequence]
        if isinstance(label, int):
            targets += [torch.tensor(label)]
        else:
            targets += [label]
    # Group the list of tensors into a batched tensor
    if all_tensors_have_same_dimensions(tensors):
        tensors = torch.stack(tensors)  # TODO: Is this correct
    else:
        tensors = pad_sequence(tensors)
    if all_tensors_have_same_dimensions(raw_tensors):
        raw_tensors = torch.stack(raw_tensors)  # TODO: Is this correct
    else:
        raw_tensors = pad_sequence(raw_tensors)
    targets = torch.stack(targets)
    return raw_tensors, tensors, targets


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "audio_classification", "datasets", "audiofolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(datadir, args, dataset_loader_dict, test_only=False):
    # Data loading code
    logger = getLogger("root.load_data")
    logger.info("Loading data")
    dataset_loader = dataset_loader_dict.get(args.dataset_loader)
    
    st = timeit.default_timer()
    if test_only:
        # datadir is supposed to be test dir
        if args.dataset == 'modelmaker':
            test_folders = os.path.normpath(datadir).split(os.sep)
            test_anno = glob(
                os.path.join(os.sep.join(test_folders[:-1]), 'annotations', f'{args.annotation_prefix}_test*_list.txt'))
            test_list = test_anno[0] if len(test_anno) == 1 and os.path.exists(test_anno[0]) else None
            dataset_test = dataset_loader("test", dataset_dir=args.data_path, validation_list=test_list, **vars(args)).prepare(**vars(args))
        else:
            dataset_test = dataset_loader("test", dataset_dir=args.data_path, **vars(args)).prepare(**vars(args))
        logger.info("Loading Test/Evaluation data")
        if args.loader_type in ['classification']:
            logger.info('Test Data: target count: {} : Split Up: {}'.format(len(dataset_test.Y), ';\t'.join([
                f"{[f'{label_name}({label_index})' for label_name, label_index in dataset_test.label_map.items() if label_index == i][0]}:"
                f" {len(np.where(dataset_test.Y == i)[0])} " for i in np.unique(dataset_test.Y)])))
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        logger.info("Took {0:.2f} seconds".format(timeit.default_timer() - st))

        return dataset_test, dataset_test, test_sampler, test_sampler

    logger.info("Loading training data")
    st = timeit.default_timer()
    cache_path = _get_cache_path(datadir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        logger.info("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        if args.dataset == 'modelmaker':
            train_folders = os.path.normpath(datadir).split(os.sep)
            train_anno = glob(os.path.join(os.sep.join(train_folders[:-1]), 'annotations', f'{args.annotation_prefix}_train*_list.txt'))
            training_list = train_anno[0] if len(train_anno)==1 and os.path.exists(train_anno[0]) else None
            dataset = dataset_loader("training", dataset_dir=args.data_path, training_list=training_list, **vars(args)).prepare(**vars(args))
        else:
            dataset = dataset_loader("training", dataset_dir=args.data_path, **vars(args)).prepare(**vars(args))
        if args.cache_dataset:
            logger.info("Saving dataset_train to {}".format(cache_path))
            mkdir(os.path.dirname(cache_path))
            save_on_master((dataset, datadir), cache_path)
    logger.info("Took {0:.2f} seconds".format(timeit.default_timer() - st))

    logger.info("Loading validation data")
    st = timeit.default_timer()
    cache_path = _get_cache_path(datadir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        logger.info("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        # val_transform = presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size,
        #                                      interpolation=interpolation,
        #                                      image_mean=args.image_mean, image_scale=args.image_scale)
        if args.dataset == 'modelmaker':
            val_folders = os.path.normpath(datadir).split(os.sep)
            val_anno = glob(os.path.join(os.sep.join(val_folders[:-1]), 'annotations', f'{args.annotation_prefix}_val*_list.txt'))
            val_list = val_anno[0] if len(val_anno)==1 and os.path.exists(val_anno[0]) else None
            dataset_test = dataset_loader("val", dataset_dir=args.data_path, validation_list=val_list, **vars(args)).prepare(**vars(args))
        else:
            dataset_test = dataset_loader("val", dataset_dir=args.data_path, **vars(args)).prepare(**vars(args))
        # TODO: Add utils and uncomment the if block
        # if args.cache_dataset:
        #     logger.info("Saving dataset_test to {}".format(cache_path))
        #     utils.mkdir(os.path.dirname(cache_path))
        #     utils.save_on_master((dataset_test, datadir), cache_path)
    logger.info("Took {:.2f} seconds".format(timeit.default_timer() - st))
    logger.info("\nCreating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        if args.loader_type in ['classification']:
            logger.info('Train Data: target count: {} : Split Up: {}'.format(len(dataset.Y), ';\t'.join(
                [f"{[f'{label_name}({label_index})' for label_name, label_index in dataset.label_map.items() if label_index == i][0]}:"
                 f" {len(np.where(dataset.Y == i)[0])} " for i in np.unique(dataset.Y)])))
            logger.info('Val Data: target count: {} : Split Up: {}'.format(len(dataset_test.Y), ';\t'.join(
                [f"{[f'{label_name}({label_index})' for label_name, label_index in dataset_test.label_map.items() if label_index == i][0]}:"
                 f" {len(np.where(dataset_test.Y == i)[0])} " for i in np.unique(dataset_test.Y)])))
            # logger.critical('target train 0/1: {}/{} {}'.format(len(np.where(dataset.Y == np.unique(dataset.Y)[0])[0]), len(np.where(dataset.Y == np.unique(dataset.Y)[1])[0]), len(dataset.Y)))
            class_sample_count = np.array([len(np.where(dataset.Y == t)[0]) for t in np.unique(dataset.Y)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in np.array(dataset.Y).astype(int)])
            samples_weight = torch.from_numpy(samples_weight)
            # samples_weight = samples_weight.double()
            train_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def plot_feature_components_graph(dataset_instance, graph_type, instance_type, output_dir):
    logger = getLogger("root.utils.plot_feature_components_graph")
    graph_fn_dict = {'pca': PCA, 'tsne': TSNE}
    time_series_data = graph_fn_dict.get(graph_type)(n_components=3).fit_transform(
        dataset_instance.X.reshape([dataset_instance.X.shape[0], -1]))

    n_clusters = len(dataset_instance.classes)
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    colors = plt.cm.get_cmap("tab10", n_clusters)
    for i in range(n_clusters):
        xdata = time_series_data[np.where(np.array(dataset_instance.Y) == i)][:, 0]
        ydata = time_series_data[np.where(np.array(dataset_instance.Y) == i)][:, 1]
        zdata = time_series_data[np.where(np.array(dataset_instance.Y) == i)][:, 2]
        # plt.scatter(xdata, ydata, zdata, c='aquamarine', label=f'Cluster {i}')
        ax.scatter3D(xdata, ydata, zdata, label=f'{dataset_instance.inverse_label_map[i]}', color=colors(i))  # c=zdata, cmap='viridis'

    plt.title(f"{graph_type.upper()} Visualization of Feature Extracted Clusters")
    plt.legend(loc="lower right")
    ax.set_xlabel('Principal Component 1', rotation=150)
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3', rotation=60)
    logger.info(f"Feature Components Graph is at: {os.path.join(output_dir, f'{graph_type}_on_feature_extracted_{instance_type}_data.png')}")
    plt.savefig(os.path.join(output_dir, f'{graph_type}_on_feature_extracted_{instance_type}_data.png'))



def plot_multiclass_roc(ground_truth, predicted, output_dir, label_map=None, phase=''):
    """
    Plots an OvR (One-vs-Rest) Multiclass ROC Curve.

    Parameters:
        ground_truth (1D array): Ground truth labels (n_samples,).
        predicted (2D array): Predicted probabilities (n_samples, num_classes).
        output_dir (str): Output directory.
    """
    logger = getLogger("root.plot_multiclass_roc")
    num_classes = predicted.size(dim=-1)
    if not isinstance(predicted, np.ndarray):
        predicted = predicted.cpu().numpy()
    if not isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.cpu().numpy()

    if num_classes == 2:
        ground_truth_binarized = ground_truth
    else:
        # Binarize the ground truth labels
        ground_truth_binarized = label_binarize(ground_truth, classes=np.arange(num_classes))

    # Initialize plot
    fig = plt.figure(figsize=(10, 8))

    # Colors for each class
    colors = plt.cm.get_cmap("tab10", num_classes)

    # Loop through each class
    fpr_list = []
    tpr_list = []
    thresholds_list = []
    for i in range(num_classes):
        # Compute ROC curve and AUC for the current class
        try:
            fpr, tpr, thresholds = roc_curve(ground_truth_binarized[:, i], predicted[:, i])
        except IndexError:
            # 2 class classification
            fpr, tpr, thresholds = roc_curve(ground_truth_binarized, predicted[:, 1])

        fpr_list.extend(fpr)
        tpr_list.extend(tpr)
        thresholds_list.extend(thresholds)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve for the class
        plt.plot(fpr, tpr, color=colors(i), label=f"{label_map[i]} (AUC = {roc_auc:.2f})")

        # Annotate thresholds
        for j in range(len(thresholds)//3, len(thresholds)*2//3, max(len(thresholds) // 5, 1)):
            # Offset for threshold labels to avoid overlap
            x_offset = 0.02 if j % 2 == 0 else -0.02
            y_offset = 0.02 if j % 2 == 0 else -0.02

            # Markers
            plt.scatter(fpr[j], tpr[j], color=colors(i), s=40, edgecolor='black', alpha=0.8)
            # Threshold text
            plt.text(fpr[j] + x_offset, tpr[j] + y_offset, f"{thresholds[j]:.0f}", fontsize=8, color=colors(i), alpha=1)

    pd.DataFrame(list(zip(fpr_list, tpr_list, thresholds_list)), columns=['fpr', 'tpr', 'thresholds']).to_csv(
        os.path.join(output_dir, 'fpr_tpr_thresholds.csv'))
    # Plot diagonal line for random guessing
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess")

    # Customize plot
    plt.title("One v/s Rest Multi-class ROC Curve (with thresholds)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    logger.info(f"Plot is at: {os.path.join(output_dir, f'One_vs_Rest_MultiClass_ROC_{phase}.png')}")
    plt.savefig(os.path.join(output_dir, f'One_vs_Rest_MultiClass_ROC_{phase}.png'))


def plot_pairwise_differenced_class_scores(ground_truth, predicted, output_dir, label_map=None, phase=''):
    """
      Plots histograms of pairwise differences between class scores for each class combination.

      Parameters:
      - scores (torch.Tensor): Array of shape (n, num_classes) where `n` is the number of sequences,
                                and `num_classes` is the number of classes.

      Returns:
      - None
      """
    logger = getLogger("root.utils.plot_pairwise_differenced_class_scores")
    n, num_classes = predicted.size()

    # Generate all pairs of class combinations
    class_pairs = list(combinations(range(num_classes), 2))

    # Set up the figure for subplots
    num_subplots = len(class_pairs)
    ncols = 3  # Number of columns in the subplot grid
    nrows = (num_subplots + ncols - 1) // ncols  # Compute required rows
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    fig.suptitle("Pairwise class prediction difference ")
    axes = np.ravel(axes)  # Flatten axes for easy iteration

    for idx, (i, j) in enumerate(class_pairs):
        # Compute differences
        indices = torch.where((ground_truth == i) | (ground_truth == j))[0]
        class_scores = predicted[indices, :]
        differences = class_scores[:, i] - class_scores[:, j]
        positive_differences = differences[differences >= 0]
        negative_differences = differences[differences < 0]

        # Create histogram bins
        bins = torch.linspace(differences.min(), differences.max(), 50)

        # Plot histograms
        axes[idx].hist(positive_differences.cpu().numpy(), bins=bins, color='blue', alpha=0.7, label=f'{label_map[i]} >= {label_map[j]}')
        axes[idx].hist(negative_differences.cpu().numpy(), bins=bins, color='orange', alpha=0.7, label=f'{label_map[i]} < {label_map[j]}')

        # Add titles and labels
        axes[idx].set_title(f'Pair: Class{i} v/s Class{j}', fontsize=14)
        axes[idx].set_xlabel(f'Difference: (x{i} - x{j})', fontsize=12)
        axes[idx].set_ylabel('Frequency', fontsize=12)
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

    # Hide unused subplots if the grid is larger than required
    for ax in axes[num_subplots:]:
        ax.set_visible(False)

    # Adjust layout
    plt.tight_layout()
    # plt.legend(loc="lower right")

    logger.info(f"Plot is at: {os.path.join(output_dir, f'Histogram_Class_Score_differences_{phase}.png')}")
    plt.savefig(os.path.join(output_dir, f'Histogram_Class_Score_differences_{phase}.png'))


def plot_regression(ground_truth, predictions, output_dir, phase=''):
    """
    Plots scatter plots of predictions and ground_truth with lines connecting corresponding indices
    for each axis (if the arrays have multiple axes).

    Parameters:
    - predictions (numpy.ndarray): Array of predicted values. Can be 1D or 2D.
    - ground_truth (numpy.ndarray): Array of ground truth values. Must match predictions in shape.

    Returns:
    - None
    """
    logger = getLogger("root.utils.plot_regression")
    if predictions.shape != ground_truth.shape:
        raise ValueError("predictions and ground_truth must have the same shape.")

    # Handle 1D arrays as 2D for uniformity
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
        ground_truth = ground_truth[:, np.newaxis]

    num_axes = predictions.shape[1]  # Number of axes to plot
    indices = np.arange(predictions.shape[0])  # Indices along the first axis

    # Set up the figure for subplots
    fig, axes = plt.subplots(num_axes, 1, figsize=(10, 5 * num_axes), squeeze=False)

    for axis in range(num_axes):
        ax = axes[axis, 0]

        # Scatter plots for ground truth and predictions
        ax.scatter(indices, ground_truth[:, axis], color='green', label=f'Ground Truth (Axis {axis})', s=50, alpha=0.8)
        ax.scatter(indices, predictions[:, axis], color='blue', label=f'Predictions (Axis {axis})', s=50, alpha=0.8)

        # Draw lines connecting corresponding points
        for i in range(len(indices)):
            ax.plot([indices[i], indices[i]],
                    [ground_truth[i, axis], predictions[i, axis]],
                    color='gray', alpha=0.5, linestyle='--')

        # Add labels, title, and legend
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Target', fontsize=12)
        ax.set_title(f'Regression Scatter Plot', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)

    # Adjust layout
    plt.tight_layout()
    logger.info(f"Plot is at: {os.path.join(output_dir, f'Regression_plot_{phase}.png')}")
    plt.savefig(os.path.join(output_dir, f'Regression_plot_{phase}.png'))



def plot_actual_vs_predicted_regression(ground_truth, predictions, output_dir, phase='', max_points=1000):
    logger = getLogger("root.utils.plot_actual_vs_predicted")
    if predictions.shape != ground_truth.shape:
        raise ValueError("predictions and ground_truth must have the same shape.")
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
        ground_truth = ground_truth[:, np.newaxis]
    num_axes = predictions.shape[1]
    num_samples = predictions.shape[0]
    if num_samples > max_points:
        indices = np.random.choice(num_samples, max_points, replace=False)
        predictions = predictions[indices]
        ground_truth = ground_truth[indices]
    fig, axes = plt.subplots(1, num_axes, figsize=(7 * num_axes, 6), squeeze=False)  
    for axis in range(num_axes):
        ax = axes[0, axis]
        ax.scatter(ground_truth[:, axis], predictions[:, axis], 
                  color='blue', alpha=0.6, edgecolor='k', s=50)
        min_val = min(ground_truth[:, axis].min(), predictions[:, axis].min())
        max_val = max(ground_truth[:, axis].max(), predictions[:, axis].max())
        
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Perfect Prediction')
        correlation_matrix = np.corrcoef(ground_truth[:, axis], predictions[:, axis])
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'Actual vs Predicted \n$R^2={r_squared:.4f}$', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'actual_vs_predicted_{phase}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot is saved at: {save_path}")
    plt.close(fig)


def plot_residual_error_regression(ground_truth, predictions, output_dir, phase='', bins=20):
    from scipy import stats
    logger = getLogger("root.utils.plot_residual_error")
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if ground_truth.ndim > 1:
        ground_truth = ground_truth.flatten()
    if predictions.ndim > 1:
        predictions = predictions.flatten()    
    residuals = ground_truth - predictions
    fig, ax = plt.subplots(figsize=(10, 7))
    
    n, bins, patches = ax.hist(residuals, bins=bins, density=True, 
                              hatch='///', edgecolor='black', 
                              alpha=0.7, label='Residual')
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(min(residuals), max(residuals), 100)
    y = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, y, 'r-', linewidth=2, label=f'Normal Fit\nμ={mu:.4f}, σ={sigma:.4f}')
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    median_residual = np.median(residuals)
    ax.axvline(mean_residual, color='blue', linestyle='-', linewidth=2, label=f'Mean: {mean_residual:.4f}')
    ax.axvline(median_residual, color='green', linestyle='--', linewidth=2, label=f'Median: {median_residual:.4f}')
    ax.set_xlabel('Deviation (Actual - Predicted)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    title = f'Residual Error Distribution\nStd Dev: {std_residual:.4f}'
    ax.set_title(title, fontsize=16)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    stats_text = (f'Statistics:\n'
                 f'Mean: {mean_residual:.4f}\n'
                 f'Median: {median_residual:.4f}\n'
                 f'Std Dev: {std_residual:.4f}\n'
                 f'Min: {min(residuals):.4f}\n'
                 f'Max: {max(residuals):.4f}')
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, 
           fontsize=12, va='top', bbox=props)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'residual_error_{phase}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Residual plot saved at: {save_path}")
    plt.close(fig)


def plot_reconstruction_errors(anomaly_errors, normal_test_errors,normal_data_mean, threshold,image_save_folder, bins=100,  log_scale=False):
    plt.figure(figsize=(10, 6))
    
    plt.hist(anomaly_errors, bins, alpha=0.3, label="Anomaly", color='b')
    plt.hist(normal_test_errors, bins, alpha=0.3, label="Normal", color='r')
    

    # Add vertical lines for means
    plt.axvline(np.mean(anomaly_errors), color='b', linestyle='dashed', linewidth=2,
                label=f'Anomaly test mean: {np.mean(anomaly_errors):.4f}')
    plt.axvline(np.mean(normal_test_errors), color='r', linestyle='dashed', linewidth=2,
                label=f'Normal test mean: {np.mean(normal_test_errors):.4f}')
    plt.axvline(normal_data_mean, color='orange',linestyle='dashed', linewidth=2,
                label=f'Normal train mean: {normal_data_mean}')
    plt.axvline(threshold, color='green', linestyle='dashed', linewidth=2,
                label=f'Threshold: {threshold:.4f}')

    if(log_scale):
        plt.xscale('log')
        plt.yscale('log')

    plt.title('Distribution of Reconstruction Errors', fontsize=16)
    plt.xlabel(f'Reconstruction Error{"(Log scale)" if log_scale else ""}', fontsize=14)
    plt.ylabel(f'Error count{"(Log scale)" if log_scale else ""}', fontsize=14)
    plt.legend(loc='upper right')
    
    plt.savefig(os.path.join(image_save_folder, f'reconstruction_error{"_log_scale" if log_scale else ""}.png'))
    plt.close()
    
class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt="{median:.4f} ({global_avg:.4f})"):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        try:
            t = t.tolist()
        except AttributeError:
            pass
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        latest = self.deque[-1]
        if isinstance(latest, numbers.Number):
            d = torch.tensor(list(self.deque))
            return d.median().item()
        elif isinstance(latest, torch.Tensor) and latest.ndim == 0:
            d = torch.tensor(list(self.deque))
            return d.median().item()
        else:
            return latest  # TODO: This isn't a median, it is a placeholder

    @property
    def avg(self):
        latest = self.deque[-1]
        if isinstance(latest, numbers.Number):
            d = torch.tensor(list(self.deque), dtype=torch.float32)
            return d.mean().item()
        elif isinstance(latest, torch.Tensor) and latest.ndim == 0:
            d = torch.tensor(list(self.deque), dtype=torch.float32)
            return d.mean().item()
        else:
            return latest


    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        latest = self.deque[-1]
        if isinstance(latest, numbers.Number):
            return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)
        elif isinstance(latest, torch.Tensor) and latest.ndim == 0:
            return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)
        else:
            return re.sub("        ", " ", str(self.global_avg).replace('\n', ''))


class MetricLogger(object):
    def __init__(self, delimiter="\t", phase=""):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = getLogger(f"root.utils.MetricLogger.{phase}")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if not isinstance(v, (float, int)):
                raise TypeError(
                    f"This method expects the value of the input arguments to be of type float or int, instead  got {type(v)}"
                )
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, **kwargs):
        self.meters[name] = SmoothedValue(**kwargs)

    def log_every(self, iterable, print_freq=5, header=None):
        i = 0
        if not header:
            header = ""
        start_time = timeit.default_timer()
        end = timeit.default_timer()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(timeit.default_timer() - end)
            yield obj
            iter_time.update(timeit.default_timer() - end)
            if print_freq is not None and i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    self.logger.info(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = timeit.default_timer()
        total_time = timeit.default_timer() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(f"{header} Total time: {total_time_str}")


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """
    def __init__(self, model, decay, device='cpu'):
        ema_avg = (lambda avg_model_param, model_param, num_averaged:
                   decay * avg_model_param + (1 - decay) * model_param)
        super().__init__(model, device, ema_avg)

    def update_parameters(self, model):
        for p_swa, p_model in zip(self.module.state_dict().values(), model.state_dict().values()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_,
                                     self.n_averaged.to(device)))
        self.n_averaged += 1


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        try:
            # this can crash if num classes is less than maxk.
            # so put in try except
            _, pred = output.topk(maxk, 1, True, True)
        except Exception:
            maxk = output.size()[-1]
            _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def get_confusion_matrix(output, target, classes):
    """
    Compute multi-class confusion matrix, a matrix of dimension num_classes x num_classes
    where each element at position (i,j) is the number of examples with true class i that were predicted to be class j.
    """
    return multiclass_confusion_matrix(output, target, classes)


def get_f1_score(output, target, classes):
    return multiclass_f1_score(output, target, num_classes=classes)


def get_au_roc(output, target, classes):
    return multiclass_auroc(output, target, num_classes=classes, average='macro')


def get_r2_score(output,target):
    return r2_score(target, output, multioutput='uniform_average')  # variance_weighted, raw_values


def get_mse(output,target):
    return mean_squared_error(target,output,multioutput='uniform_average')  # raw_values

# Calculate Symmetric Mean Absolute Percentage Error
def smape(y_true, y_pred): # y_true and y_pred must be tensors
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    denominator=np.where(denominator==0,1e-8,denominator)  # Avoid division by zero
    return torch.mean(numerator / denominator) * 100  # Multiply by 100 to get percentage
    
def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def accuracy1(output, target, topk=(1,)):
    with torch.no_grad():
        correct = 0
        res = []
        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)
        res.append(100. * correct / target.size(0))
        # print(pred,correct,res)
        return res

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16

    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def store_model_weights(model, checkpoint_path, checkpoint_key='model', strict=True):
    """
    This method can be used to prepare weights files for new models. It receives as
    input a model architecture and a checkpoint from the training script and produces
    a file with the weights ready for release.

    Examples:
        from torchvision import models as M

        # Classification
        model = M.mobilenet_v3_large(pretrained=False)
        print(store_model_weights(model, './class.pth'))

        # Quantized Classification
        model = M.quantization.mobilenet_v3_large(pretrained=False, quantize=False)
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        _ = torch.quantization.prepare_qat(model, inplace=True)
        print(store_model_weights(model, './qat.pth'))

        # Object Detection
        model = M.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, pretrained_backbone=False)
        print(store_model_weights(model, './obj.pth'))

        # Segmentation
        model = M.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, aux_loss=True)
        print(store_model_weights(model, './segm.pth', strict=False))

    Args:
        model (pytorch.nn.Module): The model on which the weights will be loaded for validation purposes.
        checkpoint_path (str): The path of the checkpoint we will load.
        checkpoint_key (str, optional): The key of the checkpoint where the model weights are stored.
            Default: "model".
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        output_path (str): The location where the weights are saved.
    """
    # Store the new model next to the checkpoint_path
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    # Deep copy to avoid side-effects on the model object.
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load the weights to the model to validate that everything works
    # and remove unnecessary weights (such as auxiliaries, etc)
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)

    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)

    return output_path


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def train_one_epoch_regression(model, criterion, optimizer, data_loader, device, epoch, transform, lambda_reg=0.01,
                    apex=False, model_ema=None, print_freq=None, phase="", dual_op=True, is_ptq=False, **kwargs):
    model.train()
    metric_logger = MetricLogger(delimiter="  ", phase=phase)
    metric_logger.add_meter("lr", window_size=1, fmt="{value}")
    metric_logger.add_meter("samples/s", window_size=10, fmt="{value}")

    header = f"Epoch: [{epoch}]"
    # TODO: If transform is required
    if transform:
        transform = transform.to(device)
    # for _, data, target in metric_logger.log_every(data_loader, print_freq, header):
    for _, data, target in data_loader:
        start_time = timeit.default_timer()
        data = data.to(device).float()
        target = target.to(device).float()
        if transform:
            data = transform(data)

        if dual_op:
            output, secondary_output = model(data)  # (n,1,8000) -> (n,35)
        else:
            output = model(data)  # (n,1,8000) -> (n,35)

        loss = criterion(output, target)
        if not is_ptq:
            optimizer.zero_grad()
            if lambda_reg:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

                loss += (lambda_reg*(l1_norm))
                loss += (lambda_reg*(l2_norm))
            if apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
        mse = get_mse(output, target).squeeze()
        batch_size = output.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['mse'].update(mse, n=batch_size)
        metric_logger.meters['samples/s'].update(batch_size / (timeit.default_timer() - start_time))

    if model_ema:
        model_ema.update_parameters(model)

def train_one_epoch_forecasting(model, criterion, optimizer, data_loader, device, epoch, transform,
                    apex=False, model_ema=None, print_freq=None, phase="", dual_op=True, is_ptq=False, **kwargs):
    model.train()
    print_freq = print_freq if print_freq else len(data_loader)
    metric_logger = MetricLogger(delimiter="  ", phase=phase)
    metric_logger.add_meter("lr", window_size=1, fmt="{value}")
    metric_logger.add_meter("samples/s", window_size=10, fmt="{value}")

    header = f"Epoch: [{epoch}]"
    # TODO: If transform is required
    if transform:
        transform = transform.to(device)
    
    for _, data, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = timeit.default_timer()
        data = data.to(device).float()
        target = target.to(device).float()

        # apply transform and model on whole batch directly on device
        # TODO: If transform is required
        if transform:
            data = transform(data)

        if dual_op:
            output, secondary_output = model(data)  # (n,1,8000) -> (n,35)
        else:
            output = model(data)  # (n,1,8000) -> (n,35)"

        output = output.view_as(target)

        loss = criterion(output, target)

        if not is_ptq:
            optimizer.zero_grad()
            if apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        smape_score = smape(target.detach(), output.detach()).item()
        batch_size = output.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['smape'].update(smape_score, n=batch_size)
        metric_logger.meters['samples/s'].update(batch_size / (timeit.default_timer() - start_time))

    if model_ema:
        model_ema.update_parameters(model)
    

def evaluate_forecasting(model, criterion, data_loader, device, transform=None, log_suffix='', print_freq=None, phase='', dual_op=True, **kwargs):
    logger = getLogger(f"root.train_utils.evaluate.{phase}")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", phase=phase)
    print_freq = print_freq if print_freq else len(data_loader)
    header = f'Test: {log_suffix}'

    targets=[]
    outputs=[]

    with torch.no_grad():
        for _, data, target in metric_logger.log_every(data_loader, print_freq, header):
            # Move data and target to the specified device
            data = data.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()

            # Apply transformation if provided
            if transform:
                data = transform(data)

            # Forward pass through the model
            if dual_op:
                output, _ = model(data)  # Ignore secondary_output if not needed
            else:
                output = model(data)

            # Reshape output to match target shape
            output = output.view_as(target)

            # Compute loss
            loss = criterion(output, target)
            metric_logger.update(loss=loss.item())
            batch_size = data.shape[0]
            smape_score = smape(target.detach(), output.detach()).item()
            metric_logger.meters['smape'].update(smape_score, n=batch_size)
            targets.append(target)
            outputs.append(output)
            
    metric_logger.synchronize_between_processes()
    target_tensor=torch.cat(targets, dim=0)
    prediction_tensor=torch.cat(outputs, dim=0)
    overall_smape= smape(target_tensor, prediction_tensor)
    logger.info(f"Current SMAPE across all target variables and across all predicted timesteps: {overall_smape:.2f}%")
    return target_tensor,prediction_tensor,overall_smape

def save_forecasting_predictions_csv(true_values, predictions, output_dir,header_row, forecast_horizon):
    """
    Save predictions in CSV format with alternating ground truth and predicted values.

    Args:
        true_values (np.ndarray): Ground truth values [batch_size,forecast_horizon, n_variables]
        predictions (np.ndarray): Predicted values [batch_size,forecast_horizon, n_variables]
        output_dir (str): Base directory to save CSV files
        header_row (list): List of variable names: column number in key value pairs
        forecast_horizon (int): Number of time steps to forecast
    """
    csv_dir=os.path.join(output_dir, 'predictions_csv')
    os.makedirs(csv_dir, exist_ok=True)

    for idx,item in enumerate(header_row):
        for target_variable_name in item:
            data = {}
            for step in range(forecast_horizon):
                data[f'ground_{step+1}'] = true_values[:, step, idx]
                data[f'predicted_{step+1}'] = predictions[:, step, idx]
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(csv_dir,f'{target_variable_name}_predictions.csv'), index=False)

def evaluate_regression(model, criterion, data_loader, device, transform, log_suffix='', print_freq=None, phase='', dual_op=True, **kwargs):
    logger = getLogger(f"root.train_utils.evaluate.{phase}")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", phase=phase)
    print_freq = print_freq if print_freq else len(data_loader)
    header = f'Test: {log_suffix}'

    target_array = torch.Tensor([]).to(device, non_blocking=True)
    predictions_array = torch.Tensor([]).to(device, non_blocking=True)
    with torch.no_grad():
        # for _, data, target in metric_logger.log_every(data_loader, print_freq, header):
        val_loss = 0
        target_list = []
        predictions_list = []
        for _, data, target in data_loader:


            data = data.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()

            if transform:
                data = transform(data)

            if dual_op:
                output, secondary_output = model(data)
            else:
                output = model(data)

            loss = criterion(output, target)  # .squeeze()
            val_loss += loss.item()
            mse = get_mse(output, target)  # .squeeze()
            r2 = get_r2_score(output, target)  # .squeeze()
            target_list.append(target)
            predictions_list.append(output)
            # FIXME need to take into account that the datasets could have been padded in distributed setup
            batch_size = data.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['mse'].update(mse, n=batch_size)
            metric_logger.meters['r2'].update(r2, n=batch_size)

    metric_logger.synchronize_between_processes()
    target_array = torch.cat(target_list)
    predictions_array = torch.cat(predictions_list)


    logger.info(f'{header} MSE {get_mse(predictions_array, target_array):.3f}')
    logger.info(f'{header} R2-Score {get_r2_score(predictions_array, target_array):.3f}')
    return get_mse(predictions_array, target_array), get_r2_score(predictions_array, target_array)

def train_one_epoch_anomalydetection(
        model, criterion, optimizer, data_loader, device, epoch, transform,
        apex=False, model_ema=None, print_freq=None, phase="", dual_op=True, is_ptq=False, **kwargs):
    model.train()
    print_freq = print_freq if print_freq else len(data_loader)
    metric_logger = MetricLogger(delimiter="  ", phase=phase)
    header = f"Training   - Epoch[{epoch}]:"
    if transform:
        transform = transform.to(device)
    for _,data, labels in metric_logger.log_every(data_loader, print_freq, header):
        # for batch_idx, (data, target) in enumerate(data_loader):
        start_time = timeit.default_timer()
        data = data.to(device).float()
        #In anomlay detection with auto encoder, the target and the input data both are same. 
        target = data.clone()

        # apply transform and model on whole batch directly on device
        # TODO: If transform is required
        if transform:
            data = transform(data)

        if dual_op:
            output, secondary_output = model(data)  # (n,1,8000) -> (n,35)
        else:
            output = model(data)  # (n,1,8000) -> (n,35)

        loss = criterion(output, target)

        if not is_ptq:
            optimizer.zero_grad()
            if apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        metric_logger.update(loss=loss.item())
        
    if model_ema:
        model_ema.update_parameters(model)


def evaluate_anomalydetection(
        model, criterion, data_loader, device, transform, epoch, log_suffix='', print_freq=None, phase='', dual_op=True, **kwargs):
    logger = getLogger(f"root.train_utils.evaluate.{phase}")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", phase=phase)
    print_freq = print_freq if print_freq else len(data_loader)
    header = f'Validation{log_suffix} - Epoch[{epoch}]: '

    with torch.no_grad():
        for _, data, labels in metric_logger.log_every(data_loader, print_freq, header):
            # for data, target in data_loader:
            data = data.to(device, non_blocking=True).float()
            #In anomlay detection with auto encoder, the target and the input data both are same. 
            target = data
            if transform:
                data = transform(data)

            if dual_op:
                output, secondary_output = model(data)
            else:
                output = model(data)

            loss = criterion(output, target) 
            batch_size = data.shape[0]
            metric_logger.update(loss=loss.item())
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg


def train_one_epoch_classification(
        model, criterion, optimizer, data_loader, device, epoch, transform,
        apex=False, model_ema=None, print_freq=None, phase="", dual_op=True, is_ptq=False,
        nn_for_feature_extraction=False, **kwargs):
    model.train()
    print_freq = print_freq if print_freq else len(data_loader)
    metric_logger = MetricLogger(delimiter="  ", phase=phase)
    metric_logger.add_meter("lr", window_size=1, fmt="{value}")
    metric_logger.add_meter("samples/s", window_size=10, fmt="{value}")
    #
    # new_sample_rate = 8000
    # transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

    header = f"Epoch: [{epoch}]"
    # TODO: If transform is required
    if transform:
        transform = transform.to(device)
    # for _, data, target in metric_logger.log_every(data_loader, print_freq, header):
    for data_raw, data_feat_ext, target in metric_logger.log_every(data_loader, print_freq, header):
        # for batch_idx, (data, target) in enumerate(data_loader):
        # logger.info(batch_idx)
        start_time = timeit.default_timer()
        if nn_for_feature_extraction:
            data = data_raw.to(device).float()
        else:
            data = data_feat_ext.to(device).float()
        target = target.to(device).long()

        # apply transform and model on whole batch directly on device
        # TODO: If transform is required
        if transform:
            data = transform(data)

        if dual_op:
            output, secondary_output = model(data)  # (n,1,8000) -> (n,35)
        else:
            output = model(data)  # (n,1,8000) -> (n,35)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = criterion(output, target)

        if not is_ptq:
            optimizer.zero_grad()
            if apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        acc1 = accuracy(output, target, topk=(1,))
        # f1_score = get_f1_score(output, target, kwargs.get('num_classes'))
        batch_size = output.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1[0], n=batch_size)
        metric_logger.meters['samples/s'].update(batch_size / (timeit.default_timer() - start_time))

    if model_ema:
        model_ema.update_parameters(model)


def evaluate_classification(model, criterion, data_loader, device, transform, log_suffix='', print_freq=None, phase='', dual_op=True, nn_for_feature_extraction=False, **kwargs):
    logger = getLogger(f"root.train_utils.evaluate.{phase}")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", phase=phase)
    print_freq = print_freq if print_freq else len(data_loader)
    header = f'Test: {log_suffix}'
    confusion_matrix_total = np.zeros((kwargs.get('num_classes'), kwargs.get('num_classes')))

    target_array = torch.Tensor([]).to(device, non_blocking=True)
    predictions_array = torch.Tensor([]).to(device, non_blocking=True)

    with torch.no_grad():
        # for _, data, target in metric_logger.log_every(data_loader, print_freq, header):
        for data_raw, data_feat_ext, target  in metric_logger.log_every(data_loader, print_freq, header):
            # for data, target in data_loader:
            if nn_for_feature_extraction:
                data = data_raw.to(device, non_blocking=True).float()
            else:
                data = data_feat_ext.to(device).float()

            target = target.to(device, non_blocking=True).long()
            if transform:
                data = transform(data)

            if dual_op:
                output, secondary_output = model(data)
            else:
                output = model(data)

            target_array = torch.cat((target_array, target))
            predictions_array = torch.cat((predictions_array, output))

            loss = criterion(output.squeeze(), target)
            acc1 = accuracy(output.squeeze(), target, topk=(1,))
            f1_score = get_f1_score(output, target, kwargs.get('num_classes'))

            confusion_matrix = get_confusion_matrix(output, target, kwargs.get('num_classes')).cpu().numpy()
            confusion_matrix_total += confusion_matrix

            # au_roc = get_au_roc(output, target, kwargs.get('num_classes')) # .cpu().numpy()
            # au_roc_total += au_roc
            # FIXME need to take into account that the datasets could have been padded in distributed setup
            batch_size = data.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1[0], n=batch_size)
            metric_logger.meters['f1'].update(f1_score, n=batch_size)
            # metric_logger.meters['auroc'].update(au_roc, n=batch_size)
            # metric_logger.meters['cm'].update(confusion_matrix, n=batch_size)
            # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # logger.info(f'{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}')
    logger.info(f'{header} Acc@1 {accuracy(predictions_array.squeeze(), target_array, topk=(1,))[0]:.3f}')
    logger.info(f'{header} F1-Score {get_f1_score(predictions_array.squeeze(), target_array, kwargs.get("num_classes")):.3f}')
    # auc = get_au_roc_from_conf_matrix(confusion_matrix_total)
    # logger.info('AU-ROC Score: {:.3f}'.format(auc))
    auc = get_au_roc(predictions_array, target_array, kwargs.get('num_classes'))
    logger.info("AU-ROC Score: {:.3f}".format(auc))
    logger.info('Confusion Matrix:\n {}'.format(tabulate(pd.DataFrame(get_confusion_matrix(
        predictions_array.cpu(), target_array.type(dtype=torch.int64).cpu(), kwargs.get('num_classes')),
        columns=[f"Predicted as: {x}" for x in range(kwargs.get('num_classes'))],
        index=[f"Ground Truth: {x}" for x in range(kwargs.get('num_classes'))]), headers="keys", tablefmt='grid')))

    # logger.info(f'{header} AUROC {metric_logger.auroc.global_avg:.3f}')
    # logger.info('\n' + '\n'.join([f"Ground Truth:(Class {i}), Predicted:(Class {j}): {int(confusion_matrix_total[i][j])}" for j in range(kwargs.get('num_classes')) for i in range(kwargs.get('num_classes'))]))

    # logger.info('Confusion Matrix:\n {}'.format(tabulate(pd.DataFrame(confusion_matrix_total,
    #               columns=[f"Predicted as: {x}" for x in range(kwargs.get('num_classes'))],
    #               index=[f"Ground Truth: {x}" for x in range(kwargs.get('num_classes'))]),
    #                                                      headers="keys", tablefmt='grid')))

    # logger.info(f'AU-ROC: {au_roc_total}')
    return metric_logger.acc1.global_avg, metric_logger.f1.global_avg, auc, confusion_matrix_total, predictions_array, target_array

def print_file_level_classification_summary(dataset, predicted, ground_truth,phase):
    logger_flcs = getLogger(f"root.utils.print_file_level_classification_summary.{phase}")
    # Convert model output scores into predicted class labels
    predicted_classes = torch.argmax(predicted, dim=1).cpu().numpy()
    ground_truth_np = ground_truth.cpu().numpy()
    
    unique_files = np.unique(dataset.file_names)
    num_classes = len(dataset.inverse_label_map)
    class_names = list(dataset.inverse_label_map.values())
    
    results = []
    
    # For each file, count how many samples were classified into each class
    for file in unique_files:
        # Find indices of samples that belong to this file
        file_indices = np.where(np.array(dataset.file_names) == file)[0]
        # If no samples from this file, skip
        if len(file_indices) == 0:
            continue
        # Get predictions and ground truth for this file's samples
        file_predictions = predicted_classes[file_indices]
        file_ground_truth = ground_truth_np[file_indices]
        true_class = file_ground_truth[0]
        true_class_name = dataset.inverse_label_map[true_class]

        # Count predictions per class for this file
        counts = [np.sum(file_predictions == c) for c in range(num_classes)]
        
        # Add to results
        row = {'True Class': true_class_name, 'File': opb(file), 'Total Samples': len(file_indices)}
        for i, class_name in enumerate(class_names):
            row[f'Predicted as {class_name}'] = counts[i]
        
        results.append(row)
    
    df = pd.DataFrame(results)
    logger_flcs.info(f'File-Level Classification Summary of {phase}:\n {tabulate(df, headers="keys", tablefmt="pretty")}')

def export_model(model, input_shape, output_dir, opset_version=17, quantization=0,
                 example_input=None, generic_model=False, remove_hooks_for_jit=False):
    logger = getLogger("root.export_model")
    device="cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # example_input[0].unsqueeze(0) will create a tensor with batch size 1: n,c,h,w -> 1,c,h,w
    dummy_input = torch.rand(size=input_shape).to(device) if example_input is None else \
        example_input[0].unsqueeze(0).to(dtype=torch.float, device=device)
    onnx_file = os.path.join(output_dir, 'model.onnx')
    logger.debug(f"Quantization Mode: {quantization}, {type(quantization)}")
    logger.info(f'Exporting ONNX model from: {onnx_file}')

    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)
    if quantization:
        if example_input is not None and hasattr(model_copy, 'measure_stats'):
            example_input = example_input.to(dtype=torch.float, device=device)
            model_copy_for_log = copy.deepcopy(model_copy)         
            qdq_model_output = model_copy_for_log(example_input)
            model_copy_for_log = model_copy_for_log.convert()
            int_model_output = model_copy_for_log(example_input)
            try:
                convert_diff_stats = model_copy_for_log.measure_stats(qdq_model_output, int_model_output)
            except TypeError:
                convert_diff_stats = model_copy_for_log.measure_stats(qdq_model_output[0], int_model_output[0])
            logger.info(f"Quantization Convert Diff: {convert_diff_stats}")

        # convert the model
        model_copy = model_copy.convert()
        # export converted onnx model
        model_copy.export(dummy_input, onnx_file, opset_version=opset_version)

        if remove_hooks_for_jit:   
            remove_hooks(model_copy)
        ts_model = torch.jit.trace(model_copy, dummy_input)
        torch.jit.save(ts_model, os.path.splitext(onnx_file)[0]+"_ts.pth")
        if not generic_model:
            encrypt(os.path.splitext(onnx_file)[0]+"_ts.pth", get_crypt_key())
    else:
        torch.onnx.export(model_copy, dummy_input, onnx_file, opset_version=opset_version)

    onnx.shape_inference.infer_shapes_path(onnx_file, onnx_file)
    if not generic_model:
        encrypt(onnx_file, get_crypt_key())


def remove_hooks(model):
    model._backward_hooks = OrderedDict()
    model._forward_hooks = OrderedDict()
    model._forward_pre_hooks = OrderedDict()
    for child in model.children():
        remove_hooks(child)


def encrypt(filename, key):
    if not key:
        return
    f = Fernet(key)
    with open(filename, "rb") as file:
        # read the encrypted data
        file_data = file.read()
    # encrypt data
    encrypted_data = f.encrypt(file_data)
    # write the encrypted file
    with open(filename, "wb") as file:
        file.write(encrypted_data)


def get_crypt_key():
    try:
        from .crypt_key import password
        my_password = password.encode()
        key = hashlib.md5(my_password).hexdigest()
        key_64 = base64.urlsafe_b64encode(key.encode("utf-8"))
    except ImportError:
        key_64 = ''

    return key_64


def decrypt(filename, key):
    if not key:
        return
    f = Fernet(key)
    with open(filename, "rb") as file:
        # read the encrypted data
        encrypted_data = file.read()
    # decrypt data
    decrypted_data = f.decrypt(encrypted_data)
    with open(filename, "wb") as file:
        file.write(decrypted_data)


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        try:
            return torch.tensor(val)
        except ValueError:
            return val

    t = torch.tensor(val, device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dist.barrier()
    dist.all_reduce(t)
    return t


def init_optimizer(model, opt_name="sgd", lr=0.1, momentum=0.9, weight_decay=4e-5):
    logger = getLogger('root.utils.init_optimizer')
    opt_name = opt_name.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
            nesterov="nesterov" in opt_name)
    elif opt_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum,
                                        weight_decay=weight_decay, eps=0.0316, alpha=0.9)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        logger.warning("Invalid optimizer {}. Only SGD and RMSprop, Adam are supported. Defaulting to Adam".format(opt_name))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def init_lr_scheduler(
        lr_scheduler="cosineannealinglr", optimizer="sgd", epochs=90, lr_warmup_epochs=5, lr_step_size=30, lr_gamma=0.1,
        lr_warmup_method="constant", lr_warmup_decay=0.01):
    lr_scheduler = lr_scheduler.lower()
    if lr_scheduler == 'steplr':
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    elif lr_scheduler == 'cosineannealinglr':
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                       T_max=epochs - lr_warmup_epochs)
    elif lr_scheduler == 'exponentiallr':
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
                           "are supported.".format(lr_scheduler))

    if lr_warmup_epochs > 0:
        if lr_warmup_method == 'linear':
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=lr_warmup_decay,
                                                                    total_iters=lr_warmup_epochs)
        elif lr_warmup_method == 'constant':
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=lr_warmup_decay,
                                                                      total_iters=lr_warmup_epochs)
        else:
            raise RuntimeError(f"Invalid warmup lr method '{lr_warmup_method}'. Only linear and constant "
                               "are supported.")
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler
    return lr_scheduler


def quantization_wrapped_model(model, quantization=0, quantization_method='QAT', weight_bitwidth=8, activation_bitwidth=8, epochs=10, output_dequantize=False):
    logger = getLogger('root.utils.quantization_wrapped_model')
    if quantization == TinyMLQuantizationVersion.QUANTIZATION_GENERIC:
        if quantization_method == TinyMLQuantizationMethod.QAT:
            model = GenericTinyMLQATFxModule(model, qconfig_type=TinyMLQConfigType(weight_bitwidth, activation_bitwidth).qconfig_type, total_epochs=epochs)
        if quantization_method == TinyMLQuantizationMethod.PTQ:
            model = GenericTinyMLPTQFxModule(model, qconfig_type=TinyMLQConfigType(weight_bitwidth, activation_bitwidth).qconfig_type, total_epochs=epochs)
    elif quantization == TinyMLQuantizationVersion.QUANTIZATION_TINPU:
        if quantization_method == TinyMLQuantizationMethod.QAT:
            model = TINPUTinyMLQATFxModule(model, qconfig_type=TinyMLQConfigType(weight_bitwidth, activation_bitwidth).qconfig_type, total_epochs=epochs, output_dequantize=output_dequantize)
        if quantization_method == TinyMLQuantizationMethod.PTQ:
            model = TINPUTinyMLPTQFxModule(model, qconfig_type=TinyMLQConfigType(weight_bitwidth, activation_bitwidth).qconfig_type, total_epochs=epochs, output_dequantize=output_dequantize)
    if quantization:
        logger.info(f"Proceeding with {quantization_method} quantization")
    return model


def get_trained_feature_extraction_model(model, args, data_loader, data_loader_test, device, lr_scheduler, optimizer, is_ptq=False):
    logger = getLogger('root.get_trained_feature_extraction_model')
    criterion = torch.nn.MSELoss()

    is_ptq = True if (args.quantization_method in ['PTQ'] and args.quantization) else False
    # model = quantization_wrapped_model(
    #     model, args.quantization, args.quantization_method, args.weight_bitwidth, args.activation_bitwidth, args.epochs, args.)
    for epoch in range(args.start_epoch, args.epochs ): # args.epochs
        model.train()

        header = f"Epoch: [{epoch}]"
        for data_raw, data_fe, _ in data_loader:
            start_time = timeit.default_timer()

            data_raw = data_raw.to(device).float()
            data_fe = data_fe.to(device).float()

            output = model(data_raw)  # (n,1,8000) -> (n,35)

            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = criterion(output, data_fe)

            if not is_ptq:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if not is_ptq:
            lr_scheduler.step()

        # Set the model to evaluation mode
        model.eval()

        # Initialize metrics
        mse = 0
        mae = 0
        total_samples = 0

        # Evaluate the model
        with torch.no_grad():
            for data_raw, data_fe, _ in data_loader_test:
                # Assuming the dataset returns (data, target)
                data_raw = data_raw.to(device).float()
                data_fe = data_fe.to(device).float()
                outputs = model(data_raw)

                # Calculate loss
                loss = criterion(outputs, data_fe)

                # Update metrics
                mse += loss.item() * data_fe.numel()
                mae += torch.sum(torch.abs(outputs - data_fe)).item()
                total_samples += data_fe.numel()

        # Calculate average metrics
        mse /= total_samples
        rmse = mse ** 0.5
        mae /= total_samples

        # Print evaluation results
        logger.info(f'{header}: MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f} lr: {optimizer.param_groups[0]["lr"]}')
    return model