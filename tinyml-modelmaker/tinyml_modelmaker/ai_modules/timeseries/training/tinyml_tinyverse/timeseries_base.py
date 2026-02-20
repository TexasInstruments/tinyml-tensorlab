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
Base class for timeseries training modules.
This module contains common functionality shared across all timeseries task types:
- Classification
- Regression
- Anomaly Detection
- Forecasting
"""

import os
import shutil
from copy import deepcopy

import torch.backends.mps

from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

import tinyml_modelmaker

from ..... import utils
from ... import constants


def get_base_log_summary_regex():
    """
    Returns the base log summary regex patterns common to all timeseries tasks.
    Subclasses can extend this by calling this function and adding task-specific patterns.
    """
    return {
        'js': [
            # Floating Point Training Metrics
            {'type': 'Epoch (FloatTrain)', 'name': 'Epoch (FloatTrain)', 'description': 'Epochs (FloatTrain)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain: Epoch:\s+\[(?<eid>\d+)\]\s+Total', 'groupId': 'eid'}],
             },
            {'type': 'Training Loss (FloatTrain)', 'name': 'Loss (FloatTrain)',
             'description': 'Training Loss (FloatTrain)', 'unit': 'Loss', 'value': None,
             'regex': [{'op': 'search',
                        'pattern': r'FloatTrain: Epoch:.*?loss:\s+(?<loss>[0-9\.]+)',
                        'groupId': 'loss'}],
             },
            {'type': 'Validation Accuracy (FloatTrain)', 'name': 'Accuracy (FloatTrain)',
             'description': 'Validation Accuracy (FloatTrain)', 'unit': 'Accuracy Top-1%', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'FloatTrain: Test:\s+\s+Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                        'groupId': 'accuracy', 'scale_factor': 1}],
             },
            {'type': 'F1-Score (FloatTrain)', 'name': 'F1-Score (FloatTrain)',
             'description': 'F1-Score (FloatTrain)', 'unit': 'F1-Score', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'FloatTrain: Test:\s+\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                        'groupId': 'f1score', 'scale_factor': 1}],
             },
            {'type': 'Confusion Matrix (FloatTrain)', 'name': 'Confusion Matrix (FloatTrain)',
             'description': 'Confusion Matrix (FloatTrain)', 'unit': 'Confusion Matrix', 'value': None,
             'regex': [{'op': 'search',
                        'pattern': r'FloatTrain:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\s\S]+?)INFO',
                        'groupId': 'cm', 'scale_factor': 1}],
             },
            # Quantized Training
            {'type': 'Epoch (QuantTrain)', 'name': 'Epoch (QuantTrain)', 'description': 'Epochs (QuantTrain)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain: Epoch:\s+\[(?<eid>\d+)\]\s+Total', 'groupId': 'eid'}],
             },
            {'type': 'Training Loss (QuantTrain)', 'name': 'Loss (QuantTrain)',
             'description': 'Training Loss (QuantTrain)', 'unit': 'Loss', 'value': None,
             'regex': [{'op': 'search',
                        'pattern': r'QuantTrain: Epoch:.*?loss:\s+(?<loss>[0-9\.]+)',
                        'groupId': 'loss'}],
             },
            {'type': 'F1-Score (QuantTrain)', 'name': 'F1-Score (QuantTrain)',
             'description': 'F1-Score (QuantTrain)', 'unit': 'F1-Score', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'QuantTrain: Test:\s+\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                        'groupId': 'f1score', 'scale_factor': 1}],
             },
            {'type': 'Confusion Matrix (QuantTrain)', 'name': 'Confusion Matrix (QuantTrain)',
             'description': 'Confusion Matrix (QuantTrain)', 'unit': 'Confusion Matrix', 'value': None,
             'regex': [{'op': 'search',
                        'pattern': r'QuantTrain:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\s\S]+?)INFO',
                        'groupId': 'cm', 'scale_factor': 1}],
             },
            {'type': 'Validation Accuracy (QuantTrain)', 'name': 'Accuracy (QuantTrain)',
             'description': 'Validation Accuracy (QuantTrain)', 'unit': 'Accuracy Top-1%', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'QuantTrain: Test:\s+\s+Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                        'groupId': 'accuracy', 'scale_factor': 1}],
             },
            # Best Epoch QuantTrain Metrics
            {'type': 'Epoch (QuantTrain, BestEpoch)', 'name': 'Epoch (QuantTrain, BestEpoch)', 'description': 'Epochs (QuantTrain, BestEpoch)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain.BestEpoch: Best Epoch:\s+(?<eid>\d+)\s*', 'groupId': 'eid'}],
             },
            {'type': 'F1-Score (QuantTrain, BestEpoch)', 'name': 'F1-Score (QuantTrain, BestEpoch)',
             'description': 'F1-Score (QuantTrain, BestEpoch)', 'unit': 'F1-Score', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'QuantTrain.BestEpoch:\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                        'groupId': 'f1score', 'scale_factor': 1}],
             },
            {'type': 'Confusion Matrix (QuantTrain, BestEpoch)', 'name': 'Confusion Matrix (QuantTrain, BestEpoch)',
             'description': 'Confusion Matrix (QuantTrain, BestEpoch)', 'unit': 'Confusion Matrix', 'value': None,
             'regex': [{'op': 'search',
                        'pattern': r'QuantTrain.BestEpoch:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\s\S]+?)INFO',
                        'groupId': 'cm', 'scale_factor': 1}],
             },
            {'type': 'Validation Accuracy (QuantTrain, BestEpoch)', 'name': 'Accuracy (QuantTrain, BestEpoch)',
             'description': 'Validation Accuracy (QuantTrain, BestEpoch)', 'unit': 'Accuracy Top-1%', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'QuantTrain.BestEpoch: Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                        'groupId': 'accuracy', 'scale_factor': 1}],
             },
            # Best Epoch FloatTrain Metrics
            {'type': 'Epoch (FloatTrain, BestEpoch)', 'name': 'Epoch (FloatTrain, BestEpoch)',
             'description': 'Epochs (FloatTrain, BestEpoch)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain.BestEpoch\s*: Best Epoch:\s+(?<eid>\d+)\s*',
                  'groupId': 'eid'}],
             },
            {'type': 'F1-Score (FloatTrain, BestEpoch)', 'name': 'F1-Score (FloatTrain, BestEpoch)',
             'description': 'F1-Score (FloatTrain, BestEpoch)', 'unit': 'F1-Score', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain.BestEpoch\s*:\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                  'groupId': 'f1score', 'scale_factor': 1}],
             },
            {'type': 'Confusion Matrix (FloatTrain, BestEpoch)', 'name': 'Confusion Matrix (FloatTrain, BestEpoch)',
             'description': 'Confusion Matrix (FloatTrain, BestEpoch)', 'unit': 'Confusion Matrix', 'value': None,
             'regex': [{'op': 'search',
                        'pattern': r'FloatTrain.BestEpoch\s*:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\s\S]+?)INFO',
                        'groupId': 'cm', 'scale_factor': 1}],
             },
            {'type': 'Validation Accuracy (FloatTrain, BestEpoch)', 'name': 'Accuracy (FloatTrain, BestEpoch)',
             'description': 'Validation Accuracy (FloatTrain, BestEpoch)', 'unit': 'Accuracy Top-1%', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'FloatTrain.BestEpoch\s*: Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                        'groupId': 'accuracy', 'scale_factor': 1}],
             },
            # Test data
            {'type': 'Test Accuracy (Test Data)', 'name': 'Accuracy (Test Data)',
             'description': 'Test Accuracy (Test Data)', 'unit': 'Accuracy Top-1%', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'test_data\s*:\s*Test Data Evaluation Accuracy:\s+(?<accuracy>[-+e\d+\.\d+]+)%',
                        'groupId': 'accuracy', 'scale_factor': 1}],
             },
            {'type': 'Confusion Matrix (Test Data)', 'name': 'Confusion Matrix',
             'description': 'Confusion Matrix (Test Data)', 'unit': 'Confusion Matrix', 'value': None,
             'regex': [{'op': 'search',
                        'pattern': r'test_data\s*:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\s\S]+?)INFO',
                        'groupId': 'cm', 'scale_factor': 1}],
             },
            {'type': 'Matrix Label', 'name': 'Matrix Label', 'description': 'Matrix Label',
             'unit': 'Matrix Label', 'value': None,
                "regex": [{'op': 'search', 'pattern': r'Ground Truth:\s*(?<label>\w+)\s*\|\s*',
                           'scale_factor': 1, 'groupId': 'label'}],
            },
            {'type': 'Matrix Cell', 'name': 'Matrix Cell', 'description': 'Matrix Cell',
             'unit': 'Matrix Cell', 'value': None,
             "regex": [{'op': 'search', 'pattern': r'\|\s*(?<cell>\d+)',
                        'scale_factor': 1, 'groupId': 'cell'}],
             },
        ]
    }


def get_regression_log_summary_regex():
    """
    Returns the log summary regex patterns for regression tasks.
    Extracts MSE and R2-Score metrics from training logs.
    Log format example:
    - INFO: root.train_utils.evaluate.FloatTrain: Test:  MSE 5306.965
    - INFO: root.train_utils.evaluate.FloatTrain: Test:  R2-Score -21386.584
    - INFO: root.main.FloatTrain.BestEpoch: Best Epoch: 38
    - INFO: root.main.FloatTrain.BestEpoch: MSE 15.475
    - INFO: root.main.FloatTrain.BestEpoch: R2-Score 0.994
    """
    return {
        'js': [
            # Floating Point Training Metrics
            {'type': 'Epoch (FloatTrain)', 'name': 'Epoch (FloatTrain)', 'description': 'Epochs (FloatTrain)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain: Epoch:\s+\[(?<eid>\d+)\]\s+Total', 'groupId': 'eid'}],
             },
            {'type': 'Training Loss (FloatTrain)', 'name': 'Loss (FloatTrain)',
             'description': 'Training Loss (FloatTrain)', 'unit': 'Loss', 'value': None,
             'regex': [{'op': 'search',
                        'pattern': r'FloatTrain: Epoch:.*?loss:\s+(?<loss>[0-9\.]+)',
                        'groupId': 'loss'}],
             },
            # Floating Point Validation Metrics (per epoch)
            {'type': 'Validation MSE (FloatTrain)', 'name': 'MSE (FloatTrain)',
             'description': 'Validation MSE (FloatTrain)', 'unit': 'MSE', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'FloatTrain: Test:\s+MSE\s+(?<mse>[-+e\d+\.\d+]+)',
                        'groupId': 'mse', 'scale_factor': 1}],
             },
            {'type': 'Validation R2-Score (FloatTrain)', 'name': 'R2-Score (FloatTrain)',
             'description': 'Validation R2-Score (FloatTrain)', 'unit': 'R2-Score', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'FloatTrain: Test:\s+R2-Score\s+(?<r2>[-+e\d+\.\d+]+)',
                        'groupId': 'r2', 'scale_factor': 1}],
             },
            # Quantized Training
            {'type': 'Epoch (QuantTrain)', 'name': 'Epoch (QuantTrain)', 'description': 'Epochs (QuantTrain)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain: Epoch:\s+\[(?<eid>\d+)\]\s+Total', 'groupId': 'eid'}],
             },
            {'type': 'Training Loss (QuantTrain)', 'name': 'Loss (QuantTrain)',
             'description': 'Training Loss (QuantTrain)', 'unit': 'Loss', 'value': None,
             'regex': [{'op': 'search',
                        'pattern': r'QuantTrain: Epoch:.*?loss:\s+(?<loss>[0-9\.]+)',
                        'groupId': 'loss'}],
             },
            # Quantized Validation Metrics (per epoch)
            {'type': 'Validation MSE (QuantTrain)', 'name': 'MSE (QuantTrain)',
             'description': 'Validation MSE (QuantTrain)', 'unit': 'MSE', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'QuantTrain: Test:\s+MSE\s+(?<mse>[-+e\d+\.\d+]+)',
                        'groupId': 'mse', 'scale_factor': 1}],
             },
            {'type': 'Validation R2-Score (QuantTrain)', 'name': 'R2-Score (QuantTrain)',
             'description': 'Validation R2-Score (QuantTrain)', 'unit': 'R2-Score', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'QuantTrain: Test:\s+R2-Score\s+(?<r2>[-+e\d+\.\d+]+)',
                        'groupId': 'r2', 'scale_factor': 1}],
             },
            # Best Epoch FloatTrain Metrics
            {'type': 'Epoch (FloatTrain, BestEpoch)', 'name': 'Epoch (FloatTrain, BestEpoch)',
             'description': 'Epochs (FloatTrain, BestEpoch)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain\.BestEpoch: Best Epoch:\s+(?<eid>\d+)',
                  'groupId': 'eid'}],
             },
            {'type': 'MSE (FloatTrain, BestEpoch)', 'name': 'MSE (FloatTrain, BestEpoch)',
             'description': 'MSE (FloatTrain, BestEpoch)', 'unit': 'MSE', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain\.BestEpoch: MSE\s+(?<mse>[-+e\d+\.\d+]+)',
                  'groupId': 'mse', 'scale_factor': 1}],
             },
            {'type': 'R2-Score (FloatTrain, BestEpoch)', 'name': 'R2-Score (FloatTrain, BestEpoch)',
             'description': 'R2-Score (FloatTrain, BestEpoch)', 'unit': 'R2-Score', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain\.BestEpoch: R2-Score\s+(?<r2>[-+e\d+\.\d+]+)',
                  'groupId': 'r2', 'scale_factor': 1}],
             },
            # Best Epoch QuantTrain Metrics
            {'type': 'Epoch (QuantTrain, BestEpoch)', 'name': 'Epoch (QuantTrain, BestEpoch)',
             'description': 'Epochs (QuantTrain, BestEpoch)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain\.BestEpoch: Best Epoch:\s+(?<eid>\d+)',
                  'groupId': 'eid'}],
             },
            {'type': 'MSE (QuantTrain, BestEpoch)', 'name': 'MSE (QuantTrain, BestEpoch)',
             'description': 'MSE (QuantTrain, BestEpoch)', 'unit': 'MSE', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain\.BestEpoch: MSE\s+(?<mse>[-+e\d+\.\d+]+)',
                  'groupId': 'mse', 'scale_factor': 1}],
             },
            {'type': 'R2-Score (QuantTrain, BestEpoch)', 'name': 'R2-Score (QuantTrain, BestEpoch)',
             'description': 'R2-Score (QuantTrain, BestEpoch)', 'unit': 'R2-Score', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain\.BestEpoch: R2-Score\s+(?<r2>[-+e\d+\.\d+]+)',
                  'groupId': 'r2', 'scale_factor': 1}],
             },
             # Test data
            {'type': 'Test RMSE (Test Data)', 'name': 'RMSE (Test Data)',
             'description': 'Test RMSE (Test Data)', 'unit': 'RMSE', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'test_data\s*:\s*Test Data Evaluation RMSE:\s+(?<rmse>[-+e\d+\.\d+]+)',
                        'groupId': 'rmse', 'scale_factor': 1}],
             },
            {'type': 'R2-Score (Test Data)', 'name': 'R2-Score (Test Data)',
             'description': 'R2-Score (Test Data)', 'unit': 'R2-Score', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'test_data\s*:\s*Test Data Evaluation R2-Score:\s+(?<r2>[-+e\d+\.\d+]+)',
                        'groupId': 'r2', 'scale_factor': 1}],
             },
        ]
    }


def get_forecasting_log_summary_regex():
    """
    Returns the log summary regex patterns for forecasting tasks.
    Includes common patterns (epoch tracking, training loss) from the base regex
    plus forecasting-specific metrics (SMAPE, R2).
    Log format example:
    - INFO: root.utils.MetricLogger.FloatTrain: Epoch: [0] Total time: 0:00:03
    - INFO: root.utils.MetricLogger.FloatTrain: Test:   [  0/314]  ... loss: 1.8408 (1.8408)  smape: 9.6656 (9.6656)
    - INFO: root.train_utils.evaluate.FloatTrain: Current SMAPE across all target variables and across all predicted timesteps: 8.54%
    - INFO: root.main.FloatTrain.BestEpoch: Best epoch:10
    - INFO: root.main.FloatTrain.BestEpoch: Overall SMAPE across all variables: 0.36%
    - INFO: root.main.FloatTrain.BestEpoch:       SMAPE of indoorTemperature across all predicted timesteps: 0.36%
    - INFO: root.main.FloatTrain.BestEpoch:       R² of indoorTemperature across all predicted timesteps: 0.9967
    - INFO: root.main.test_data :   SMAPE of indoorTemperature across all predicted timesteps: 0.95%
    - INFO: root.main.test_data :   R² of indoorTemperature across all predicted timesteps: 0.9833
    """
    return {
        'js': [
            # ---- Common patterns (epoch tracking and training loss) ----
            # Floating Point Epoch Tracking
            {'type': 'Epoch (FloatTrain)', 'name': 'Epoch (FloatTrain)', 'description': 'Epochs (FloatTrain)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain: Epoch:\s+\[(?<eid>\d+)\]\s+Total', 'groupId': 'eid'}],
             },
            {'type': 'Training Loss (FloatTrain)', 'name': 'Loss (FloatTrain)',
             'description': 'Training Loss (FloatTrain)', 'unit': 'Loss', 'value': None,
             'regex': [{'op': 'search',
                        'pattern': r'FloatTrain: Epoch:\s\[\d+\]\sTotal\stime:[\s\S]*?loss\:\s+(?<loss>\d+\.\d+)',
                        'groupId': 'loss'}],
             },
            # Quantized Epoch Tracking
            {'type': 'Epoch (QuantTrain)', 'name': 'Epoch (QuantTrain)', 'description': 'Epochs (QuantTrain)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain: Epoch:\s+\[(?<eid>\d+)\]\s+Total', 'groupId': 'eid'}],
             },
            {'type': 'Training Loss (QuantTrain)', 'name': 'Loss (QuantTrain)',
             'description': 'Training Loss (QuantTrain)', 'unit': 'Loss', 'value': None,
             'regex': [{'op': 'search',
                        'pattern': r'QuantTrain: Epoch:\s\[\d+\]\sTotal\stime:[\s\S]*?loss\:\s+(?<loss>\d+\.\d+)',
                        'groupId': 'loss'}],
             },
            # ---- Forecasting-specific patterns ----
            # Floating Point Validation Metrics (per epoch)
            {'type': 'Overall SMAPE (FloatTrain)', 'name': 'Overall SMAPE (FloatTrain)',
             'description': 'Overall SMAPE across all variables and timesteps (FloatTrain)', 'unit': 'SMAPE%', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'FloatTrain: Current SMAPE across all target variables and across all predicted timesteps:\s+(?<smape>[-+e\d+\.\d+]+)%',
                        'groupId': 'smape', 'scale_factor': 1}],
             },
            # Quantized Validation Metrics (per epoch)
            {'type': 'Overall SMAPE (QuantTrain)', 'name': 'Overall SMAPE (QuantTrain)',
             'description': 'Overall SMAPE across all variables and timesteps (QuantTrain)', 'unit': 'SMAPE%', 'value': None,
             'regex': [{'op': 'search', 'pattern': r'QuantTrain: Current SMAPE across all target variables and across all predicted timesteps:\s+(?<smape>[-+e\d+\.\d+]+)%',
                        'groupId': 'smape', 'scale_factor': 1}],
             },
            # Best Epoch FloatTrain Metrics
            {'type': 'Epoch (FloatTrain, BestEpoch)', 'name': 'Epoch (FloatTrain, BestEpoch)',
             'description': 'Epochs (FloatTrain, BestEpoch)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain\.BestEpoch: Best epoch:(?<eid>\d+)',
                  'groupId': 'eid'}],
             },
            {'type': 'Overall SMAPE (FloatTrain, BestEpoch)', 'name': 'Overall SMAPE (FloatTrain, BestEpoch)',
             'description': 'Overall SMAPE across all variables (FloatTrain, BestEpoch)', 'unit': 'SMAPE%', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain\.BestEpoch: Overall SMAPE across all variables:\s+(?<smape>[-+e\d+\.\d+]+)%',
                  'groupId': 'smape', 'scale_factor': 1}],
             },
            {'type': 'Per-Variable SMAPE (FloatTrain, BestEpoch)', 'name': 'Per-Variable SMAPE (FloatTrain, BestEpoch)',
             'description': 'SMAPE for each variable across all timesteps (FloatTrain, BestEpoch)', 'unit': 'SMAPE%', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain\.BestEpoch:.*?SMAPE of [\w\s]+ across all predicted timesteps:\s+(?<smape>[-+e\d+\.\d+]+)%',
                  'groupId': 'smape', 'scale_factor': 1}],
             },
            {'type': 'Per-Variable R2 (FloatTrain, BestEpoch)', 'name': 'Per-Variable R2 (FloatTrain, BestEpoch)',
             'description': 'R² for each variable across all timesteps (FloatTrain, BestEpoch)', 'unit': 'R2-Score', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain\.BestEpoch:.*?R² of [\w\s]+ across all predicted timesteps:\s+(?<r2>[-+e\d+\.\d+]+)',
                  'groupId': 'r2', 'scale_factor': 1}],
             },
            # Best Epoch QuantTrain Metrics
            {'type': 'Epoch (QuantTrain, BestEpoch)', 'name': 'Epoch (QuantTrain, BestEpoch)',
             'description': 'Epochs (QuantTrain, BestEpoch)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain\.BestEpoch: Best epoch:(?<eid>\d+)',
                  'groupId': 'eid'}],
             },
            {'type': 'Overall SMAPE (QuantTrain, BestEpoch)', 'name': 'Overall SMAPE (QuantTrain, BestEpoch)',
             'description': 'Overall SMAPE across all variables (QuantTrain, BestEpoch)', 'unit': 'SMAPE%', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain\.BestEpoch: Overall SMAPE across all variables:\s+(?<smape>[-+e\d+\.\d+]+)%',
                  'groupId': 'smape', 'scale_factor': 1}],
             },
            {'type': 'Per-Variable SMAPE (QuantTrain, BestEpoch)', 'name': 'Per-Variable SMAPE (QuantTrain, BestEpoch)',
             'description': 'SMAPE for each variable across all timesteps (QuantTrain, BestEpoch)', 'unit': 'SMAPE%', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain\.BestEpoch:.*?SMAPE of [\w\s]+ across all predicted timesteps:\s+(?<smape>[-+e\d+\.\d+]+)%',
                  'groupId': 'smape', 'scale_factor': 1}],
             },
            {'type': 'Per-Variable R2 (QuantTrain, BestEpoch)', 'name': 'Per-Variable R2 (QuantTrain, BestEpoch)',
             'description': 'R² for each variable across all timesteps (QuantTrain, BestEpoch)', 'unit': 'R2-Score', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain\.BestEpoch:.*?R² of [\w\s]+ across all predicted timesteps:\s+(?<r2>[-+e\d+\.\d+]+)',
                  'groupId': 'r2', 'scale_factor': 1}],
             },
            # ---- Test Data Evaluation Metrics ----
            {'type': 'Per-Variable SMAPE (Test Data)', 'name': 'Per-Variable SMAPE (Test Data)',
             'description': 'SMAPE for each variable across all timesteps (Test Data)', 'unit': 'SMAPE%', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'test_data\s*:.*?SMAPE of [\w\s]+ across all predicted timesteps:\s+(?<smape>[-+e\d+\.\d+]+)%',
                  'groupId': 'smape', 'scale_factor': 1}],
             },
            {'type': 'Per-Variable R2 (Test Data)', 'name': 'Per-Variable R2 (Test Data)',
             'description': 'R² for each variable across all timesteps (Test Data)', 'unit': 'R2-Score', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'test_data\s*:.*?R² of [\w\s]+ across all predicted timesteps:\s+(?<r2>[-+e\d+\.\d+]+)',
                  'groupId': 'r2', 'scale_factor': 1}],
             },
        ]
    }


def get_anomaly_detection_log_summary_regex():
    """
    Returns the log summary regex patterns for anomaly detection tasks.
    Extracts MSE metrics from training logs (best epoch only, as per-epoch validation logging is not performed).
    """
    return {
        'js': [
            # Floating Point Training Metrics
            {'type': 'Epoch (FloatTrain)', 'name': 'Epoch (FloatTrain)', 'description': 'Epochs (FloatTrain)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain:.*?Epoch:\s+\[(?<eid>\d+)\]\s+Total', 'groupId': 'eid'}],
             },
            {'type': 'Training Loss (FloatTrain)', 'name': 'Loss (FloatTrain)',
             'description': 'Training Loss (FloatTrain)', 'unit': 'Loss', 'value': None,
             'regex': [{'op': 'search',
                        'pattern': r'FloatTrain:.*?Training.*?Epoch\[\d+\].*?loss:\s+(?<loss>\d+\.\d+)',
                        'groupId': 'loss'}],
             },
            # Quantized Training Metrics
            {'type': 'Epoch (QuantTrain)', 'name': 'Epoch (QuantTrain)', 'description': 'Epochs (QuantTrain)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain:.*?Epoch:\s+\[(?<eid>\d+)\]\s+Total', 'groupId': 'eid'}],
             },
            {'type': 'Training Loss (QuantTrain)', 'name': 'Loss (QuantTrain)',
             'description': 'Training Loss (QuantTrain)', 'unit': 'Loss', 'value': None,
             'regex': [{'op': 'search',
                        'pattern': r'QuantTrain:.*?Training.*?Epoch\[\d+\].*?loss:\s+(?<loss>\d+\.\d+)',
                        'groupId': 'loss'}],
             },
            # Best Epoch FloatTrain Metrics
            {'type': 'Epoch (FloatTrain, BestEpoch)', 'name': 'Epoch (FloatTrain, BestEpoch)',
             'description': 'Epochs (FloatTrain, BestEpoch)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain.BestEpoch\s*: Best Epoch:\s+(?<eid>\d+)',
                  'groupId': 'eid'}],
             },
            {'type': 'MSE (FloatTrain, BestEpoch)', 'name': 'MSE (FloatTrain, BestEpoch)',
             'description': 'MSE (FloatTrain, BestEpoch)', 'unit': 'MSE', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'FloatTrain.BestEpoch\s*: MSE\s+(?<mse>[-+e\d+\.\d+]+)',
                  'groupId': 'mse', 'scale_factor': 1}],
             },
            # Best Epoch QuantTrain Metrics
            {'type': 'Epoch (QuantTrain, BestEpoch)', 'name': 'Epoch (QuantTrain, BestEpoch)',
             'description': 'Epochs (QuantTrain, BestEpoch)',
             'unit': 'Epoch', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain.BestEpoch\s*: Best Epoch:\s+(?<eid>\d+)',
                  'groupId': 'eid'}],
             },
            {'type': 'MSE (QuantTrain, BestEpoch)', 'name': 'MSE (QuantTrain, BestEpoch)',
             'description': 'MSE (QuantTrain, BestEpoch)', 'unit': 'MSE', 'value': None,
             'regex': [
                 {'op': 'search', 'pattern': r'QuantTrain.BestEpoch\s*: MSE\s+(?<mse>[-+e\d+\.\d+]+)',
                  'groupId': 'mse', 'scale_factor': 1}],
             },
        ]
    }


def create_template_model_description(task_category, task_type, dataset_loader=None, batch_size_key=None):
    """
    Factory function to create template model descriptions.

    Args:
        task_category: The task category constant (e.g., constants.TASK_CATEGORY_TS_CLASSIFICATION)
        task_type: The task type constant (e.g., constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION)
        dataset_loader: Optional dataset loader name. If None, not included in template.
        batch_size_key: Key for batch size lookup in constants.TRAINING_BATCH_SIZE_DEFAULT

    Returns:
        dict: Template model description dictionary
    """
    training_dict = dict(
        quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
        training_backend='tinyml_tinyverse',
        model_training_id='',
        model_name='',
        learning_rate=2e-3,
        model_spec=None,
        batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT.get(batch_size_key or task_type, 32),
        target_devices={
            constants.TARGET_DEVICE_F280013: dict(model_selection_factor=None),
            constants.TARGET_DEVICE_F280015: dict(model_selection_factor=None),
            constants.TARGET_DEVICE_F28003: dict(model_selection_factor=None),
            constants.TARGET_DEVICE_F28004: dict(model_selection_factor=None),
            constants.TARGET_DEVICE_F2837: dict(model_selection_factor=None),
            constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=None),
            constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=None),
        },
        training_devices={
            constants.TRAINING_DEVICE_CPU: True,
            constants.TRAINING_DEVICE_CUDA: True,
            constants.TRAINING_DEVICE_MPS: True,
        },
    )

    if dataset_loader:
        training_dict['dataset_loader'] = dataset_loader

    return dict(
        common=dict(
            task_category=task_category,
            task_type=task_type,
            generic_model=True,
            model_details='',
        ),
        download=dict(download_url='', download_path=''),
        training=training_dict,
        compilation=dict()
    )


def get_model_descriptions_filtered(model_descriptions, enabled_models_list, task_type=None):
    """
    Filter model descriptions based on enabled models list.

    Args:
        model_descriptions: Dictionary of all model descriptions
        enabled_models_list: List of enabled model names
        task_type: Optional task type filter (not currently used but available for future)

    Returns:
        dict: Filtered model descriptions
    """
    return {k: v for k, v in model_descriptions.items() if k in enabled_models_list}


def get_model_description_by_name(model_descriptions, enabled_models_list, model_name):
    """
    Get a specific model description by name.

    Args:
        model_descriptions: Dictionary of all model descriptions
        enabled_models_list: List of enabled model names
        model_name: Name of the model to retrieve

    Returns:
        dict or None: Model description if found, None otherwise
    """
    filtered = get_model_descriptions_filtered(model_descriptions, enabled_models_list)
    return filtered.get(model_name, None)


class BaseModelTraining:
    """
    Base class for timeseries model training.

    Subclasses should:
    1. Set class attributes: train_module, test_module
    2. Override _get_task_specific_train_argv() for task-specific training arguments
    3. Override _get_task_specific_test_argv() for task-specific test arguments
    4. Optionally override _get_log_summary_regex() to extend/modify regex patterns
    5. Optionally override _get_quant_train_epochs_divisor() for different quant epoch calculation
    """

    # Subclasses must set these
    train_module = None
    test_module = None

    @classmethod
    def init_params(cls, *args, **kwargs):
        params = dict(training=dict())
        params = utils.ConfigDict(params, *args, **kwargs)
        return params

    def __init__(self, *args, quit_event=None, **kwargs):
        self.params = self.init_params(*args, **kwargs)
        self.quit_event = quit_event

        # Get log summary regex (can be overridden by subclasses)
        log_summary_regex = self._get_log_summary_regex()

        # Update params that are specific to this backend and model
        self.params.update(
            training=utils.ConfigDict(
                log_file_path=os.path.join(
                    self.params.training.train_output_path if self.params.training.train_output_path else self.params.training.training_path,
                    'run.log'),
                log_summary_regex=log_summary_regex,
                summary_file_path=os.path.join(self.params.training.training_path, 'summary.yaml'),
                model_checkpoint_path=os.path.join(self.params.training.training_path, 'checkpoint.pth'),
                model_export_path=os.path.join(self.params.training.training_path, 'model.onnx'),
                model_proto_path=None,
                tspa_license_path=os.path.abspath(os.path.join(
                    os.path.dirname(tinyml_modelmaker.ai_modules.timeseries.training.tinyml_tinyverse.__file__),
                    'LICENSE.txt'))
            )
        )

        # Add task-specific init params
        self._init_task_specific_params()

        if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
            self.params.update(
                training=utils.ConfigDict(
                    model_checkpoint_path_quantization=os.path.join(
                        self.params.training.training_path_quantization, 'checkpoint.pth'),
                    model_export_path_quantization=os.path.join(
                        self.params.training.training_path_quantization, 'model.onnx'),
                )
            )

    def _get_log_summary_regex(self):
        """
        Get log summary regex patterns. Override in subclass to extend or modify.

        Returns:
            dict: Log summary regex configuration
        """
        return get_base_log_summary_regex()

    def _init_task_specific_params(self):
        """
        Initialize task-specific parameters. Override in subclass if needed.
        """
        pass

    def _get_device(self):
        """
        Determine the training device based on GPU availability.

        Returns:
            tuple: (device string, distributed flag)
        """
        distributed = 1 if self.params.training.num_gpus > 1 else 0
        device = 'cpu'
        if self.params.training.num_gpus > 0:
            if torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cuda'
        return device, distributed

    def _build_common_train_argv(self, device, distributed):
        """
        Build common training arguments shared across all task types.

        Returns:
            list: Common training arguments
        """
        return [
            '--model', f'{self.params.training.model_training_id}',
            '--dual-op', f'{self.params.training.dual_op}',
            '--model-config', f'{self.params.training.model_config}',
            '--augment-config', f'{self.params.training.augment_config}',
            '--model-spec', f'{self.params.training.model_spec}',
            '--dataset', 'modelmaker',
            '--dataset-loader', f'{self.params.training.dataset_loader}',
            '--annotation-prefix', f'{self.params.dataset.annotation_prefix}',
            '--gpus', f'{self.params.training.num_gpus}',
            '--batch-size', f'{self.params.training.batch_size}',
            '--opt', f'{self.params.training.optimizer}',
            '--weight-decay', f'{self.params.training.weight_decay}',
            '--lr-scheduler', f'{self.params.training.lr_scheduler}',
            '--lr-warmup-epochs', '1',
            '--distributed', f'{distributed}',
            '--device', f'{device}',
            '--sampling-rate', f'{self.params.data_processing_feature_extraction.sampling_rate}',
            '--resampling-factor', f'{self.params.data_processing_feature_extraction.resampling_factor}',
            '--new-sr', f'{self.params.data_processing_feature_extraction.new_sr}',
            '--stride-size', f'{self.params.data_processing_feature_extraction.stride_size}',
            '--data-proc-transforms', self.params.data_processing_feature_extraction.data_proc_transforms,
            '--feat-ext-transform', self.params.data_processing_feature_extraction.feat_ext_transform,
            '--generic-model', f'{self.params.common.generic_model}',
            '--feat-ext-store-dir', f'{self.params.data_processing_feature_extraction.feat_ext_store_dir}',
            '--dont-train-just-feat-ext', f'{self.params.data_processing_feature_extraction.dont_train_just_feat_ext}',
            '--frame-size', f'{self.params.data_processing_feature_extraction.frame_size}',
            '--feature-size-per-frame', f'{self.params.data_processing_feature_extraction.feature_size_per_frame}',
            '--num-frame-concat', f'{self.params.data_processing_feature_extraction.num_frame_concat}',
            '--frame-skip', f'{self.params.data_processing_feature_extraction.frame_skip}',
            '--min-bin', f'{self.params.data_processing_feature_extraction.min_bin}',
            '--normalize-bin', f'{self.params.data_processing_feature_extraction.normalize_bin}',
            '--dc-remove', f'{self.params.data_processing_feature_extraction.dc_remove}',
            '--analysis-bandwidth', f'{self.params.data_processing_feature_extraction.analysis_bandwidth}',
            '--log-base', f'{self.params.data_processing_feature_extraction.log_base}',
            '--log-mul', f'{self.params.data_processing_feature_extraction.log_mul}',
            '--log-threshold', f'{self.params.data_processing_feature_extraction.log_threshold}',
            '--stacking', f'{self.params.data_processing_feature_extraction.stacking}',
            '--offset', f'{self.params.data_processing_feature_extraction.offset}',
            '--scale', f'{self.params.data_processing_feature_extraction.scale}',
            '--output-int', f'{self.params.training.output_int}',
            '--variables', f'{self.params.data_processing_feature_extraction.variables}',
            '--lis', f'{self.params.training.log_file_path}',
            '--ondevice-training', f'{self.params.training.ondevice_training}',
            '--partial-quantization', f'{self.params.training.partial_quantization}',
            '--trainable_layers_from_last', f'{self.params.training.trainable_layers_from_last}',
            '--data-path', os.path.join(self.params.dataset.dataset_path, self.params.dataset.data_dir),
            '--store-feat-ext-data', f'{self.params.data_processing_feature_extraction.store_feat_ext_data}',
            '--epochs', f'{self.params.training.training_epochs}',
            '--lr', f'{self.params.training.learning_rate}',
            '--output-dir', f'{self.params.training.training_path}',
        ]

    def _get_task_specific_train_argv(self):
        """
        Get task-specific training arguments. Override in subclass.

        Returns:
            list: Task-specific training arguments
        """
        return []

    def _build_common_test_argv(self, device, data_path, model_path, output_dir):
        """
        Build common test arguments shared across all task types.

        Returns:
            list: Common test arguments
        """
        return [
            '--dataset', 'modelmaker',
            '--dataset-loader', f'{self.params.training.dataset_loader}',
            '--annotation-prefix', f'{self.params.dataset.annotation_prefix}',
            '--gpus', f'{self.params.training.num_gpus}',
            '--batch-size', f'{self.params.training.batch_size}',
            '--distributed', '0',
            '--device', f'{device}',
            '--variables', f'{self.params.data_processing_feature_extraction.variables}',
            '--sampling-rate', f'{self.params.data_processing_feature_extraction.sampling_rate}',
            '--resampling-factor', f'{self.params.data_processing_feature_extraction.resampling_factor}',
            '--new-sr', f'{self.params.data_processing_feature_extraction.new_sr}',
            '--stride-size', f'{self.params.data_processing_feature_extraction.stride_size}',
            '--data-proc-transforms', self.params.data_processing_feature_extraction.data_proc_transforms,
            '--feat-ext-transform', self.params.data_processing_feature_extraction.feat_ext_transform,
            '--frame-size', f'{self.params.data_processing_feature_extraction.frame_size}',
            '--feature-size-per-frame', f'{self.params.data_processing_feature_extraction.feature_size_per_frame}',
            '--num-frame-concat', f'{self.params.data_processing_feature_extraction.num_frame_concat}',
            '--frame-skip', f'{self.params.data_processing_feature_extraction.frame_skip}',
            '--min-bin', f'{self.params.data_processing_feature_extraction.min_bin}',
            '--normalize-bin', f'{self.params.data_processing_feature_extraction.normalize_bin}',
            '--dc-remove', f'{self.params.data_processing_feature_extraction.dc_remove}',
            '--analysis-bandwidth', f'{self.params.data_processing_feature_extraction.analysis_bandwidth}',
            '--log-base', f'{self.params.data_processing_feature_extraction.log_base}',
            '--log-mul', f'{self.params.data_processing_feature_extraction.log_mul}',
            '--log-threshold', f'{self.params.data_processing_feature_extraction.log_threshold}',
            '--stacking', f'{self.params.data_processing_feature_extraction.stacking}',
            '--offset', f'{self.params.data_processing_feature_extraction.offset}',
            '--scale', f'{self.params.data_processing_feature_extraction.scale}',
            '--output-int', f'{self.params.training.output_int}',
            '--lis', f'{self.params.training.log_file_path}',
            '--data-path', f'{data_path}',
            '--output-dir', output_dir,
            '--model-path', f'{model_path}',
            '--generic-model', f'{self.params.common.generic_model}',
        ]

    def _get_task_specific_test_argv(self):
        """
        Get task-specific test arguments. Override in subclass.

        Returns:
            list: Task-specific test arguments
        """
        return []

    def _get_quant_train_epochs_divisor(self):
        """
        Get the divisor for calculating quantization training epochs.
        Override in subclass if different ratio is needed.

        Returns:
            int: Divisor for epoch calculation (default 5, meaning quant epochs = float epochs / 5)
        """
        return 5

    def _get_min_quant_epochs(self):
        """
        Get minimum quantization training epochs.
        Override in subclass if different minimum is needed.

        Returns:
            int: Minimum quant epochs (default 10)
        """
        return 10

    def clear(self):
        """Clear the training folder."""
        shutil.rmtree(self.params.training.training_path, ignore_errors=True)

    def run(self, **kwargs):
        """
        The actual training function.
        Move this to a worker process if this function is called from a GUI.
        """
        os.makedirs(self.params.training.training_path, exist_ok=True)

        device, distributed = self._get_device()

        # Build training arguments
        argv = self._build_common_train_argv(device, distributed)

        # Insert task-specific arguments before the trailing arguments
        # The trailing arguments are: --data-path, --store-feat-ext-data, --epochs, --lr, --output-dir
        # That's 10 items (5 key-value pairs)
        task_argv = self._get_task_specific_train_argv()
        if task_argv:
            # Insert task-specific args before the last 10 items
            argv = argv[:-10] + task_argv + argv[-10:]

        args = self.train_module.get_args_parser().parse_args(argv)
        args.quit_event = self.quit_event

        if not utils.misc_utils.str2bool(self.params.testing.skip_train):
            if utils.misc_utils.str2bool(self.params.training.run_quant_train_only):
                if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
                    argv = argv[:-2]  # Remove --output-dir <output-dir>
                    argv.extend([
                        '--output-dir', f'{self.params.training.training_path_quantization}',
                        '--quantization', f'{self.params.training.quantization}',
                        '--quantization-method', f'{self.params.training.quantization_method}',
                        '--weight-bitwidth', f'{self.params.training.quantization_weight_bitwidth}',
                        '--activation-bitwidth', f'{self.params.training.quantization_activation_bitwidth}',
                    ])

                    args = self.train_module.get_args_parser().parse_args(argv)
                    args.quit_event = self.quit_event
                    self.train_module.run(args)
                else:
                    raise ValueError(f"quantization cannot be {TinyMLQuantizationVersion.NO_QUANTIZATION} if run_quant_train_only argument is chosen")
            else:
                self.train_module.run(args)

                if utils.misc_utils.str2bool(self.params.data_processing_feature_extraction.store_feat_ext_data) and \
                   utils.misc_utils.str2bool(self.params.data_processing_feature_extraction.dont_train_just_feat_ext):
                    return self.params

                if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
                    # Remove trailing arguments for quant training
                    argv = argv[:-8]  # Remove --store-feat-ext-data, --epochs, --lr, --output-dir pairs

                    quant_epochs = max(
                        self._get_min_quant_epochs(),
                        self.params.training.training_epochs // self._get_quant_train_epochs_divisor()
                    )

                    argv.extend([
                        '--output-dir', f'{self.params.training.training_path_quantization}',
                        '--epochs', f'{quant_epochs}',
                        '--lr', f'{self.params.training.learning_rate / 100}',
                        '--weights', f'{self.params.training.model_checkpoint_path}',
                        '--quantization', f'{self.params.training.quantization}',
                        '--quantization-method', f'{self.params.training.quantization_method}',
                        '--weight-bitwidth', f'{self.params.training.quantization_weight_bitwidth}',
                        '--activation-bitwidth', f'{self.params.training.quantization_activation_bitwidth}',
                        '--lr-warmup-epochs', '0',
                        '--store-feat-ext-data', 'False'
                    ])

                    args = self.train_module.get_args_parser().parse_args(argv)
                    args.quit_event = self.quit_event
                    self.train_module.run(args)

        # Run testing if enabled
        if utils.misc_utils.str2bool(self.params.testing.enable):
            self._run_testing(device)

        return self.params

    def _run_testing(self, device):
        """Run model testing/evaluation."""
        if self.params.testing.test_data and os.path.exists(self.params.testing.test_data):
            data_path = self.params.testing.test_data
        else:
            data_path = os.path.join(self.params.dataset.dataset_path, self.params.dataset.data_dir)

        if self.params.testing.model_path and os.path.exists(self.params.testing.model_path):
            model_path = self.params.testing.model_path
            output_dir = self.params.training.training_path
        else:
            if self.params.training.quantization == TinyMLQuantizationVersion.NO_QUANTIZATION:
                model_path = os.path.join(self.params.training.training_path, 'model.onnx')
                output_dir = self.params.training.training_path
            else:
                model_path = os.path.join(self.params.training.training_path_quantization, 'model.onnx')
                output_dir = self.params.training.training_path_quantization

        argv = self._build_common_test_argv(device, data_path, model_path, output_dir)
        argv.extend(self._get_task_specific_test_argv())

        args = self.test_module.get_args_parser().parse_args(argv)
        args.quit_event = self.quit_event
        self.test_module.run(args)

    def stop(self):
        """Stop the training process."""
        if self.quit_event is not None:
            self.quit_event.set()
            return True
        return False

    def get_params(self):
        """Get the current parameters."""
        return self.params
