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
import platform
from argparse import ArgumentParser
from logging import getLogger

import onnxruntime as ort
import pandas as pd
import torch
import torcheval
from tabulate import tabulate

from tinyml_tinyverse.common.datasets import GenericTSDataset

# Tiny ML TinyVerse Modules
from tinyml_tinyverse.common.utils import misc_utils, utils, mdcl_utils
from tinyml_tinyverse.common.utils.mdcl_utils import Logger
from tinyml_tinyverse.common.utils.utils import get_confusion_matrix

dataset_loader_dict = {'GenericTSDataset': GenericTSDataset}


def get_args_parser():
    DESCRIPTION = "This script loads time series dataset and tests it against a onnx model using ONNX RT"
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--dataset', default='folder', help='dataset')
    parser.add_argument('--dataset-loader', default='GenericTSDataset', help='dataset loader')
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
    parser.add_argument("--nn-for-feature-extraction", default=False, type=misc_utils.str2bool, help="Use an AI model for preprocessing")
    parser.add_argument("--output-int", default=None, type=misc_utils.str_or_bool, help="Get quantized int8 output from model (False for dequantized float output). If not specified, determined automatically based on task type and quantization level.")

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

    num_classes = len(dataset.classes)

    logger.info("Loading data:")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True,
        collate_fn=utils.collate_fn)
    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=args.batch_size,
    #     sampler=test_sampler, num_workers=args.workers, pin_memory=True,
    #     collate_fn=utils.collate_fn, )
    logger.info(f"Loading ONNX model: {args.model_path}")
    if not args.generic_model:
        utils.decrypt(args.model_path, utils.get_crypt_key())
    ort_sess = ort.InferenceSession(args.model_path)
    if not args.generic_model:
        utils.encrypt(args.model_path, utils.get_crypt_key())

    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name
    predicted = torch.tensor([]).to(device, non_blocking=True)
    ground_truth = torch.tensor([]).to(device, non_blocking=True)
    for batched_raw_data, batched_data, batched_target in data_loader:
        batched_raw_data = batched_raw_data.to(device, non_blocking=True).long()
        batched_data = batched_data.to(device, non_blocking=True).float()
        batched_target = batched_target.to(device, non_blocking=True).long()
        if transform:
            batched_data = transform(batched_data)
        if args.nn_for_feature_extraction:
            for data in batched_raw_data:
                predicted = torch.cat((predicted, torch.tensor(ort_sess.run([output_name], {input_name: data.unsqueeze(0).cpu().numpy().astype(np.float32)})[0]).to(device)))
        else:
            for data in batched_data:
                predicted = torch.cat((predicted, torch.tensor(ort_sess.run([output_name], {input_name: data.unsqueeze(0).cpu().numpy()})[0]).to(device)))
        ground_truth = torch.cat((ground_truth, batched_target))

    mdcl_utils.create_dir(os.path.join(args.output_dir, 'post_training_analysis'))
    # logger.info("Plotting OvR Multiclass ROC score")
    # utils.plot_multiclass_roc(ground_truth.cpu().numpy(), predicted.cpu().numpy(), os.path.join(args.output_dir, 'post_training_analysis'),
    #                           label_map=dataset.inverse_label_map, phase='test')
    # logger.info("Plotting Class difference scores")
    # utils.plot_pairwise_differenced_class_scores(ground_truth.cpu().numpy(), predicted.cpu().numpy(), os.path.join(args.output_dir, 'post_training_analysis'),
    #                           label_map=dataset.inverse_label_map, phase='test')
    metric = torcheval.metrics.MulticlassAccuracy()
    metric.update(torch.argmax(predicted.squeeze(), dim=1), ground_truth.squeeze())
    logger = getLogger("root.main.test_data")
    logger.info(f"Test Data Evaluation Accuracy: {metric.compute() * 100:.2f}%")
    # logger.info(
    #     f"Test Data Evaluation AUC ROC Score: {utils.get_au_roc(predicted.type(torch.int64), ground_truth, num_classes):.3f}")
    if len(torch.unique(ground_truth)) == 1:
        logger.warning("Confusion Matrix can not be printed because only items of 1 class was present in test data")
    else:
        confusion_matrix = get_confusion_matrix(predicted.squeeze().type(torch.int64), ground_truth.squeeze().type(torch.int64),
                                                num_classes).cpu().numpy()
        logger.info('Confusion Matrix:\n {}'.format(tabulate(pd.DataFrame(
            confusion_matrix, columns=[f"Predicted as: {x}" for x in dataset.inverse_label_map.values()],
            index=[f"Ground Truth: {x}" for x in dataset.inverse_label_map.values()]), headers="keys", tablefmt='grid')))
    return

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
