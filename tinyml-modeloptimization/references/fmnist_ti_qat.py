"""
PyTorch 2.0 Quantization Aware Training using tinyml_torchmodelopt.
Contributed by Ajay Jayaraj (ajayj@ti.com)

© Copyright 2024, PyTorch.
Based on the PyTorch FMNIST tutorial:
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

Refer to readme.md for instruction to set up environment
"""


import os
import timeit

import onnx
import tinyml_torchmodelopt.quantization as tinpu_quantization
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Pass the ``Dataset`` as an argument to ``DataLoader``. This wraps an
# iterable over the dataset, and supports automatic batching, sampling,
# shuffling and multiprocess data loading.

# Here we define a batch size of 64, i.e. each element
# in the dataloader iterable will return a batch of 64 features and labels.
BATCH_SIZE = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# To define a neural network in PyTorch, we create a class that inherits
# from `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_.
# We define the layers of the network in the ``__init__`` function and specify how
# data will pass through the network in the ``forward`` function.

# Train on CPU device
DEVICE = 'cpu'

# https://pytorch.org/docs/stable/quantization.html
# Define a floating point model where some layers could be statically quantized
class NeuralNetwork(nn.Module):
    """FMNIST Network"""
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """Specify how data passes through the network."""
        #x = self.flatten(x)
        x = x.reshape(x.shape[0], -1)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(DEVICE)
print(model)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def timethis(func):
    def timed(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        print('Function', func.__name__, 'time:', round((end - start) * 1000, 1), 'ms')
        return result
    return timed

@timethis
def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                return top1, top5

    return top1, top5

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


# To train a model, we need a `loss function <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# and an `optimizer <https://pytorch.org/docs/stable/optim.html>`_.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# In a single training loop, the model makes predictions on the training dataset
# (fed to it in batches), and backpropagates the prediction error to adjust the model's parameters.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Check the model's performance against the test dataset to ensure it is learning.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def rename_input_node_for_onnx_model(onnx_model, input_node_name):
    """Rename the input of an ONNX model"""

    # Update graph input name.
    onnx_model.graph.input[0].name = input_node_name

    # Update input of the first node also to correspond.
    onnx_model.graph.node[0].input[0] = input_node_name

    # Check and write out the updated model
    onnx.checker.check_model(onnx_model)

    return onnx_model

# The training process is conducted over several iterations (*epochs*).
# During each epoch, the model learns parameters to make better predictions.
# Print the model's accuracy and loss at each epoch and ensure the
# accuracy increases and the loss decreases with every epoch.
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Size of baseline model")
print_size_of_model(model)

num_eval_batches = 100
top1, top5 = evaluate(model, loss_fn, test_dataloader, neval_batches=num_eval_batches)
print(f"Evaluation accuracy on {num_eval_batches * BATCH_SIZE} images, {top1.avg}")

# TorchScript-based ONNX Exporter
# https://pytorch.org/docs/stable/onnx_torchscript.html
# Runs the model once to trace its execution and then exports the traced model
# to the specified file
x = test_data[0][0]
torch.onnx.export(model, x, "fmnist.onnx", input_names=["input"])
x.numpy().tofile('input.dat')

# Quantize model
epochs=10

# Specify the quantization scheme for TI NPU.
qconfig_type = {
    'weight': {
        'bitwidth': 8,
        'qscheme': torch.per_channel_symmetric,
        'power2_scale': True
    },
    'activation': {
        'bitwidth': 8,
        'qscheme': torch.per_tensor_symmetric,
        'power2_scale': True,
        'range_max': None,
        'fixed_range': False
    }
}

'''
#####################################################
Wrap the model in the TI quantization module for NPU.
#####################################################
'''
ti_model = tinpu_quantization.TINPUTinyMLQATFxModule(model, qconfig_type, total_epochs=epochs)
print(ti_model)

# Perform Quantization Aware Training.
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, ti_model, loss_fn, optimizer)
    test(test_dataloader, ti_model, loss_fn)

# Convert PyTorch QDQ layers to TI NPU int8 layers.
ti_model.to(DEVICE)
ti_model = ti_model.convert()

# Export int8 quantized model to onnx.
QUANTIZED_MODEL = 'fmnist_int8.onnx'
ti_model.export(x, QUANTIZED_MODEL)

# Set input name in the ONNX model to 'input' for consistency with float model
updated_model = rename_input_node_for_onnx_model(onnx.load(QUANTIZED_MODEL),
                                                 'input')
onnx.save(updated_model, QUANTIZED_MODEL)
