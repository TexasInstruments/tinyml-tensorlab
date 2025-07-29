import numpy as np
import torch
import os

class AvgrageMeter(object):
    """
    Computes and stores the average and current value.
    Useful for tracking metrics like loss or accuracy during training/validation.
    """
    def __init__(self):
        self.reset()  # Initialize/reset all statistics

    def reset(self):
        """
        Reset all statistics to zero.
        """
        self.avg = 0  # Average value
        self.sum = 0  # Sum of all values
        self.cnt = 0  # Count of all samples

    def update(self, val, n=1):
        """
        Update the meter with a new value.
        Args:
            val (float): Value to add.
            n (int): Number of samples the value represents (default 1).
        """
        self.sum += val * n  # Update sum
        self.cnt += n        # Update count
        self.avg = self.sum / self.cnt  # Update average


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
    maxk = max(topk)  # Maximum k value
    batch_size = target.size(0)  # Number of samples in batch

    _, pred = output.topk(maxk, 1, True, True)  # Get top-k predictions
    pred = pred.t()  # Transpose for comparison
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # Compare with targets

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)  # Number of correct predictions in top-k
        res.append(correct_k.mul_(100.0/batch_size))        # Convert to percentage
    return res

def count_parameters_in_MB(model):
    """
    Count the number of parameters in a model (in millions).
    Args:
        model (nn.Module): The model to count parameters for.
    Returns:
        float: Number of parameters in millions (MB).
    """
    return np.sum([np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name]) / 1e6

def save(model, model_path):
    """
    Save the model's state dictionary to a file.
    Args:
        model (nn.Module): The model to save.
        model_path (str): Path to save the model.
    """
    torch.save(model.state_dict(), model_path)

def create_exp_dir(path):
    """
    Create a directory for experiment outputs if it doesn't exist.
    Args:
        path (str): Directory path to create.
    """
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))