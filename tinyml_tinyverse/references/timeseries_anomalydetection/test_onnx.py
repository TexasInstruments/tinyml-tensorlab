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


import datetime
import os
import csv
import platform
from argparse import ArgumentParser
from logging import getLogger

import onnxruntime as ort
import torch
import torcheval
import numpy as np
from tinyml_tinyverse.common.datasets import GenericTSDataset, GenericTSDatasetReg, GenericTSDatasetAD

# Tiny ML TinyVerse Modules
from tinyml_tinyverse.common.utils import misc_utils, utils, mdcl_utils
from tinyml_tinyverse.common.utils.mdcl_utils import Logger
from tinyml_tinyverse.common.utils.utils import get_confusion_matrix

dataset_loader_dict = {'GenericTSDataset': GenericTSDataset, 'GenericTSDatasetReg': GenericTSDatasetReg,
                       'GenericTSDatasetAD' : GenericTSDatasetAD,}


def get_args_parser():
    DESCRIPTION = "This script loads time series dataset and tests it against a onnx model using ONNX RT"
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--dataset', default='folder', help='dataset')
    parser.add_argument('--dataset-loader', default='SimpleTSDataset', help='dataset loader')
    parser.add_argument("--loader-type", default="regression", type=str,
                        help="Dataset Loader Type: classification/regression")
    parser.add_argument('--annotation-prefix', default='instances', help='annotation-prefix')
    parser.add_argument('--data-path', default=os.path.join('.', 'data', 'datasets'), help='dataset')
    parser.add_argument('--output-dir', default=None, help='path where to save')
    parser.add_argument('--model-path', default=None, help='ONNX model Path')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
    parser.add_argument('-j', '--workers', default=0 if platform.system() in ['Windows'] else 16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--date', default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), help='current date')
    parser.add_argument('--seed', default=42, help="Seed for all randomness", type=int)
    parser.add_argument('--lis', help='Log File', type=str,)# default=ops(opb(__file__))[0] + ".lis")
    parser.add_argument('--DEBUG', action='store_true', help='Log mode set to DEBUG')

    # Training parameters
    parser.add_argument("--distributed", default=None, type=misc_utils.str2bool_or_none,
                        help="use dstributed training even if this script is not launched using torch.disctibuted.launch or run")
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=1024, type=int)
    # Feature Extraction Params
    parser.add_argument('--data-proc-transforms', help="Data Preprocessing transforms ", default=[])  # default=['DownSample', 'SimpleWindow'])
    parser.add_argument('--feat-ext-transform', help="Feature Extraction transforms ", default=[])

    # Simple TimeSeries Params
    parser.add_argument('--variables', help="1- if Univariate, 2/3/.. if multivariate")
    parser.add_argument('--resampling-factor', help="Resampling ratio")
    parser.add_argument('--sampling-rate', help="Sampled frequency ")
    parser.add_argument('--new-sr', help="Required to subsample every nth value from the dataset")  # default=3009)
    parser.add_argument('--stride-size', help="Fraction (0-1) that will be multiplied by frame-size to get the actual stride", type=float)
    # Arc Fault and Motor Fault Related Params
    parser.add_argument('--frame-size', help="Frame Size")
    parser.add_argument('--feature-size-per-frame', help="FFT feature size per frame")
    parser.add_argument('--num-frame-concat', help="Number of FFT frames to concat")
    parser.add_argument('--frame-skip', help="Skip frames while computing FFT")
    parser.add_argument('--min-bin', help="Remove DC Component from FFT")
    parser.add_argument('--normalize-bin', help="Normalize Binning")
    parser.add_argument('--dc-remove', help="Remove DC Component from FFT")
    parser.add_argument('--analysis-bandwidth', help="Spectrum of FFT used for binning")
    # parser.add_argument('--num-channel', help="Number of input channels (ex.axis, phase)", default=16, type=int)
    parser.add_argument('--log-base', help="base value for logarithm")
    parser.add_argument('--log-mul', help="multiplier for logarithm")
    parser.add_argument('--log-threshold', help="offset added to values for logarithmic calculation")
    parser.add_argument('--stacking', help="1D/2D1/None")
    parser.add_argument('--offset', help="Index for data overlap; 0: no overlap, n: start index for overlap")
    parser.add_argument('--scale', help="Scaling factor to input data")
    parser.add_argument('--generic-model', help="Open Source models", type=misc_utils.str_or_bool, default=False)
    parser.add_argument("--output-dequantize", default=False, type=misc_utils.str2bool, help="Get dequantized output from model")
    return parser


def main(gpu, args):
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

    # torch.backends.cudnn.benchmark = True
    if isinstance(args.data_proc_transforms, list):
        if len(args.data_proc_transforms) and isinstance(args.data_proc_transforms[0], list):
            args.transforms = args.data_proc_transforms[0] + args.feat_ext_transform  # args.data_proc_transforms is a list of lists
        else:
            args.transforms = args.data_proc_transforms + args.feat_ext_transform
    dataset, dataset_test, train_sampler, test_sampler = utils.load_data(args.data_path, args, dataset_loader_dict, test_only=True)  # (126073, 1, 152), 126073

    logger.info("Loading data:")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True if gpu>0 else False,
        collate_fn=utils.collate_fn)
    logger.info(f"Loading ONNX model: {args.model_path}")
    if not args.generic_model:
        utils.decrypt(args.model_path, utils.get_crypt_key())
    ort_sess = ort.InferenceSession(args.model_path)
    if not args.generic_model:
        utils.encrypt(args.model_path, utils.get_crypt_key())

    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name
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
            current_output_errors = torch.mean((torch.from_numpy(input).to(device) - output)**2, dim=(1,2,3))
            batch_reconstruction_errors = torch.cat((batch_reconstruction_errors, current_output_errors))
            batch_target_labels = torch.cat((batch_target_labels, target_label))
        errors = torch.cat((errors, batch_reconstruction_errors))
        ground_truth = torch.cat((ground_truth, batch_target_labels))
    
    post_training_analysis_path = os.path.join(args.output_dir, 'post_training_analysis')
    mdcl_utils.create_dir(post_training_analysis_path)
    #The classes folder in dataset should have two folders named Anomaly and Normal. Then dataset.classes[0] will be Anomaly and dataset.classes[1] will be Normal 
    anomaly_errors = errors[ground_truth == 0].cpu().numpy()
    normal_errors = errors[ground_truth == 1].cpu().numpy()
    logger.info("Plotting reconstructions errors")
    
    normal_train_mean, normal_train_std = get_reconstruction_errors_stats(args)
    anomaly_test_mean = np.mean(anomaly_errors)
    anomaly_test_std = np.std(anomaly_errors)
    normal_test_mean = np.mean(normal_errors)
    normal_test_std = np.std(normal_errors)
    
    #Results
    logger.info(f"Reconstruction Error Statistics:")
    logger.info(f"Normal training data - Mean: {normal_train_mean:.6f}, Std: {normal_train_std:.6f}")
    logger.info(f"Anomaly test data - Mean: {anomaly_test_mean:.6f}, Std: {anomaly_test_std:.6f}")
    logger.info(f"Normal test data - Mean: {normal_test_mean:.6f}, Std: {normal_test_std:.6f}")
    
    #Threshold
    # K is the number of standard deviation we are going away from the mean. This is used to find appropriate threshold. 
    
    
    all_k_values = [i*0.5 for i in range(0, 10)]
    results_data = []
    best_f1_score = 0
    best_f1_score_index = 0
    for i, k in enumerate(all_k_values):
        threshold = normal_train_mean + k*normal_train_std
        results = get_model_performance(threshold,normal_errors, anomaly_errors)
        results["k_value"] = k
        results["threshold"] = float(threshold)
        results_data.append(results)
        if results["f1_score"]>best_f1_score:
            best_f1_score_index = i
            best_f1_score = results["f1_score"]
            
    csv_path = os.path.join(post_training_analysis_path, 'threshold_performance.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        # Define CSV headers based on your results dictionary keys
        fieldnames = ['k_value', 'threshold', 'accuracy', 'precision', 'recall', 
                    'f1_score', 'false_positive_rate', 'true_positives', 
                    'true_negatives', 'false_positives', 'false_negatives']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write each result row
        for result in results_data:
            # Format percentages for better readability in CSV
            for key in ['accuracy', 'precision', 'recall', 'f1_score', 'false_positive_rate']:
                if key in result:
                    result[key] = round(result[key], 2)  # Keep as numeric value for sorting
            
            writer.writerow(result)

    logger.info(f"Threshold performance data saved to {csv_path}")
    
    best_results = results_data[best_f1_score_index]
    best_threshold = best_results["threshold"]
    logger.info(f"Threshold for K = {best_results['k_value']} : {best_threshold:.6f}")
    utils.plot_reconstruction_errors(anomaly_errors, normal_errors, normal_train_mean, best_threshold, post_training_analysis_path)
    utils.plot_reconstruction_errors(anomaly_errors, normal_errors, normal_train_mean, best_threshold, post_training_analysis_path,log_scale=True)
    
    
    logger.info(f"False positive rate: {best_results['false_positive_rate']:.2f}%")
    logger.info(f"Anomaly detection rate (recall): {best_results['recall']:.2f}%")
    logger.info(f"Accuracy: {best_results['accuracy']:.2f}%")
    logger.info(f"Precision: {best_results['precision']:.2f}%")
    logger.info(f"F1 Score: {best_results['f1_score']:.2f}%")

    logger.info("\nConfusion Matrix:")
    logger.info(f"                         Predicted Normal        Predicted Anomaly")
    logger.info(f"Actual  Normal text     {best_results['true_negatives']:17d}    {best_results['false_positives']:18d}")
    logger.info(f"Actual Anomaly          {best_results['false_negatives']:17d}    {best_results['true_positives']:18d}")


def get_model_performance(threshold, normal_errors, anomaly_errors):
    normal_detected_as_anomaly =sum(1 for x in normal_errors if x > threshold)
    anomaly_detected_as_anomaly =sum(1 for x in anomaly_errors if x > threshold)
    normal_detected_as_normal = len(normal_errors) - normal_detected_as_anomaly
    anomaly_detected_as_normal = len(anomaly_errors) - anomaly_detected_as_anomaly
    
    true_positives = anomaly_detected_as_anomaly
    true_negatives = normal_detected_as_normal
    false_positives = normal_detected_as_anomaly
    false_negatives = anomaly_detected_as_normal
    
    accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_negatives + false_positives) * 100
    precision = true_positives/(true_positives + false_positives) * 100 if true_positives + false_positives > 0 else 0
    recall = true_positives/(true_positives + false_negatives) * 100 if true_positives + false_negatives >0 else 0
    f1_score = 2*precision*recall/(precision+recall) if precision + recall > 0 else 0
    false_positive_rate = false_positives/(true_negatives + false_positives) *100 if true_negatives + false_positives > 0 else 0
    
    return  {
        'accuracy': accuracy,
        'precision':precision,
        'recall':recall,
        'f1_score': f1_score,
        'false_positive_rate':false_positive_rate,
        'true_positives':true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    } 

def get_reconstruction_errors_stats(args):
    log_file = os.path.join(args.output_dir, 'run.log')
    logger = Logger(log_file=args.lis or log_file, DEBUG=args.DEBUG, name="root", append_log=True, console_log=True)
    utils.seed_everything(args.seed)
    device = torch.device(args.device)

    if isinstance(args.data_proc_transforms, list):
        if len(args.data_proc_transforms) and isinstance(args.data_proc_transforms[0], list):
            args.transforms = args.data_proc_transforms[0] + args.feat_ext_transform  # args.data_proc_transforms is a list of lists
        else:
            args.transforms = args.data_proc_transforms + args.feat_ext_transform
    dataset, dataset_test, train_sampler, test_sampler = utils.load_data(args.data_path, args, dataset_loader_dict) 

    logger.info("Loading data:")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True if args.gpu>0 else False,
        collate_fn=utils.collate_fn)
    logger.info(f"Loading ONNX model: {args.model_path}")
    if not args.generic_model:
        utils.decrypt(args.model_path, utils.get_crypt_key())
    ort_sess = ort.InferenceSession(args.model_path)
    if not args.generic_model:
        utils.encrypt(args.model_path, utils.get_crypt_key())

    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name
    errors = torch.tensor([]).to(device, non_blocking=True)
    for _, data, targets in data_loader:
        data = data.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).long()
        batch_reconstruction_errors = torch.tensor([]).to(device, non_blocking=True)
        for input, target_label in zip(data, targets):
            input = input.unsqueeze(0).cpu().numpy()
            output = torch.tensor(ort_sess.run([output_name], {input_name: input})[0]).to(device)
            current_output_error = torch.mean((torch.from_numpy(input).to(device) - output)**2, dim=(1,2,3))
            batch_reconstruction_errors = torch.cat((batch_reconstruction_errors, current_output_error))
        errors = torch.cat((errors, batch_reconstruction_errors))
    
    normal_error_mean = torch.mean(errors)
    normal_error_std = torch.std(errors)
    return normal_error_mean.cpu(), normal_error_std.cpu()
    
def run(args):
    if args.device != 'cpu' and args.distributed is True:
        # for explanation of what is happening here, please see this:
        # https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
        # this assignment of RANK assumes a single machine, but with multiple gpus
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = str(args.gpus)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args,))
    else:
        main(0, args)


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()

    # run the training.
    # if args.distributed is True is set, then this will launch distributed training
    # depending on args.gpus
    run(arguments)
