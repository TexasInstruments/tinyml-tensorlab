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
from collections import OrderedDict, defaultdict, deque
from glob import glob
from logging import getLogger

import numpy as np
import onnx
import pandas as pd
import torch
import torch.distributed as dist
from cryptography.fernet import Fernet
from tabulate import tabulate
from torcheval.metrics.functional import (
    multiclass_confusion_matrix,
    multiclass_f1_score,
)

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
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    # for waveform, _, label, *_ in batch:
    for sequence, label in batch:
        tensors += [sequence]
        targets += [torch.tensor(label)]
    # Group the list of tensors into a batched tensor
    if all_tensors_have_same_dimensions(tensors):
        tensors = torch.stack(tensors)  # TODO: Is this correct
    else:
        tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets


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
        # train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


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
            return latest  # TODO: This isnt a median, it is a placeholder

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
        except:
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


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, transform,
                    apex=False, model_ema=None, print_freq=100, phase="", dual_op=True, **kwargs):
    model.train()
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
    for data, target in metric_logger.log_every(data_loader, print_freq, header):
        # for batch_idx, (data, target) in enumerate(data_loader):
        # logger.info(batch_idx)
        start_time = timeit.default_timer()
        data = data.to(device).float()
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
        # loss = F.nll_loss(output.squeeze(), target)
        # loss = nn.CrossEntropyLoss(label_smoothing=0.0)(output.squeeze(), target)
        loss = criterion(output, target)
        # loss = criterion(output.squeeze().argmax(1).to(torch.float32), target.to(torch.float32))

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # loss.backward()
        optimizer.step()

        # acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        # acc1 = utils.accuracy(output.squeeze().argmax(1), target, topk=(1,))
        acc1 = accuracy(output, target, topk=(1,))
        # confusion_matrix = get_confusion_matrix(output, target, kwargs.get('num_classes'))
        f1_score = get_f1_score(output, target, kwargs.get('num_classes'))
        batch_size = output.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        # metric_logger.meters['acc1'].update(acc1, n=batch_size)
        metric_logger.meters['acc1'].update(acc1[0], n=batch_size)
        metric_logger.meters['f1'].update(f1_score, n=batch_size)
        # metric_logger.meters['cm'].update(confusion_matrix, n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['samples/s'].update(batch_size / (timeit.default_timer() - start_time))
        '''
        # print training stats
        if batch_idx % log_interval == 0:
            logger.info(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())
        '''
    if model_ema:
        model_ema.update_parameters(model)


def evaluate(model, criterion, data_loader, device, transform, log_suffix='', print_freq=100, phase='', dual_op=True, **kwargs):
    logger = getLogger(f"root.train_utils.evaluate.{phase}")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", phase=phase)
    print_freq = min(print_freq, len(data_loader))
    header = f'Test: {log_suffix}'
    confusion_matrix_total = np.zeros((kwargs.get('num_classes'), kwargs.get('num_classes')))
    with torch.no_grad():
        for data, target in metric_logger.log_every(data_loader, print_freq, header):
            # for data, target in data_loader:
            data = data.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).long()
            if transform:
                data = transform(data)

            if dual_op:
                output, secondary_output = model(data)
            else:
                output = model(data)

            loss = criterion(output.squeeze(), target)
            acc1 = accuracy(output.squeeze(), target, topk=(1,))

            confusion_matrix = get_confusion_matrix(output, target, kwargs.get('num_classes')).cpu().numpy()
            confusion_matrix_total += confusion_matrix

            f1_score = get_f1_score(output, target, kwargs.get('num_classes'))
            # FIXME need to take into account that the datasets could have been padded in distributed setup
            batch_size = data.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1[0], n=batch_size)
            metric_logger.meters['f1'].update(f1_score, n=batch_size)
            # metric_logger.meters['cm'].update(confusion_matrix, n=batch_size)
            # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # logger.info(f'{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}')
    logger.info(f'{header} Acc@1 {metric_logger.acc1.global_avg:.3f}')
    logger.info(f'{header} F1-Score {metric_logger.f1.global_avg:.3f}')
    # logger.info('\n' + '\n'.join([f"Ground Truth:(Class {i}), Predicted:(Class {j}): {int(confusion_matrix_total[i][j])}" for j in range(kwargs.get('num_classes')) for i in range(kwargs.get('num_classes'))]))
    logger.info('Confusion Matrix:\n {}'.format(tabulate(pd.DataFrame(confusion_matrix_total,
                  columns=[f"Predicted as: {x}" for x in range(kwargs.get('num_classes'))],
                  index=[f"Ground Truth: {x}" for x in range(kwargs.get('num_classes'))]),
                                                         headers="keys", tablefmt='grid')))
    return metric_logger.acc1.global_avg, metric_logger.f1.global_avg, confusion_matrix_total


def export_model(model, input_shape, output_dir, opset_version=17, quantization=0, quantization_error_logging=False, example_input=None, generic_model=False):
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
        if quantization_error_logging and example_input is not None and hasattr(model_copy, 'measure_stats'):
            example_input = example_input.to(dtype=torch.float, device=device)
            model_copy_for_log = copy.deepcopy(model_copy)         
            qdq_model_output = model_copy_for_log(example_input)
            model_copy_for_log = model_copy_for_log.convert(output_dequantize=quantization_error_logging)
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
        # export a torchscript model as well for visualization
        ts_model = torch.jit.trace(model_copy, dummy_input)
        torch.jit.save(ts_model, os.path.splitext(onnx_file)[0]+"_ts.pth")
        if not generic_model:
            encrypt(os.path.splitext(onnx_file)[0]+"_ts.pth", get_crypt_key())
    else:
        torch.onnx.export(model_copy, dummy_input, onnx_file, opset_version=opset_version)

    onnx.shape_inference.infer_shapes_path(onnx_file, onnx_file)
    if not generic_model:
        encrypt(onnx_file, get_crypt_key())


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

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t