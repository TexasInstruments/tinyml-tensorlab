import numpy as np
import torch
import logging


class AvgrageMeter(object):
    """
    Computes and stores the average and current value.
    Useful for tracking metrics like loss or accuracy during training/validation.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k.
    Args:
        output (Tensor): Model predictions (logits or probabilities).
        target (Tensor): Ground truth labels.
        topk (tuple): Tuple of k values for top-k accuracy.
    Returns:
        list: List of accuracies for each k in topk.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def count_parameters_in_MB(model):
    """
    Count the number of parameters in a model (in millions).
    Args:
        model (nn.Module): The model to count parameters for.
    Returns:
        float: Number of parameters in millions.
    """
    return np.sum(
        [np.prod(v.size()) for name, v in model.named_parameters()
         if "auxiliary" not in name]
    ) / 1e6


def get_device(gpu_index=0):
    """
    Return the best available torch.device for NAS.

    Preference order: CUDA (with specified index) > MPS (Apple Silicon) > CPU.

    Args:
        gpu_index (int): CUDA device index (ignored for MPS/CPU).
    Returns:
        torch.device: The resolved compute device.
    """
    logger = logging.getLogger("root.modelopt.nas")
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_index}')
        logger.info('NAS device: %s (%s)', device, torch.cuda.get_device_name(device))
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info('NAS device: mps (Apple Metal)')
    else:
        device = torch.device('cpu')
        logger.info('NAS device: cpu')
    return device
