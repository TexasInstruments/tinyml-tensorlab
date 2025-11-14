# Standard library imports
import os
import re
import random
import argparse
import tarfile
import shutil
from typing import Tuple, List
from collections import Counter

# Third-party scientific libraries
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import confusion_matrix

# PyTorch core
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

# PyTorch vision
from torchvision import datasets, transforms

# PyTorch quantization
import torch.ao.quantization as tq
import torch.ao.quantization.quantize_fx as qfx
from torch.ao.quantization import quantize_fx

# TinyML quantization
from tinyml_torchmodelopt.quantization import (
    TINPUTinyMLQATFxModule, 
    TINPUTinyMLPTQFxModule, 
    GenericTinyMLQATFxModule, 
    GenericTinyMLPTQFxModule
)

# ONNX
import onnx
import onnxruntime as ort

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f" Using device: {DEVICE}")


def train(dataloader, model, loss_fn, optimizer, scheduler):
    """
    Train the model for one epoch and print running accuracy.
    """
    model.train()
    
    first_batch = next(iter(dataloader), None)
    if first_batch is None:
        print(" Error: No data in dataloader! Exiting training loop.")
        return 0, 0, model, loss_fn, optimizer

    total_loss = 0
    running_corrects = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        inputs, targets = data.to(DEVICE), target.to(DEVICE)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Compute accuracy - NO softmax needed
        _, preds = torch.max(outputs, 1)
        
        # Running statistics
        total_loss += loss.item()
        running_corrects += (preds == targets).sum().item()
        total_samples += targets.size(0)
        running_acc = running_corrects / total_samples
        
        # Print running accuracy every 10 batches
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}, Running Train Accuracy: {running_acc:.4f}")
    
    # Compute final epoch loss & accuracy
    avg_loss = total_loss / len(dataloader)
    avg_acc = running_corrects / total_samples  # Overall epoch accuracy
    print(f" Epoch Finished - Avg Loss: {avg_loss:.4f}, Avg Train Accuracy: {avg_acc:.4f}")

    return avg_loss, avg_acc, model, loss_fn, optimizer

def train_model(model, train_loader, total_epochs, learning_rate):
    """
    Train the model for multiple epochs and display accuracy.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    for epoch in range(total_epochs):
        print(f"\n Epoch {epoch + 1}/{total_epochs}")

        loss, acc, model, loss_fn, optimizer = train(train_loader, model, loss_fn, optimizer, scheduler)

        scheduler.step()
        last_lr = scheduler.get_last_lr()[0]

        print(f" Epoch {epoch+1} - Loss: {loss:.5f} - Train Accuracy: {acc:.4f} - LR: {last_lr:.4f}")

    return model
# Validation & Testing Function
def test(model, test_loader, loss_fn):
    """
    Evaluate the model on test or validation data.
    
    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for test/validation dataset.
        loss_fn: Loss function.
        device: Device to evaluate on.
    
    Returns:
        test_loss: Average test loss.
        test_acc: Average test accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0


    with torch.no_grad():
        for batch, (data,target) in test_loader:
            inputs, targets = data.to(DEVICE), target.to(DEVICE)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            _, preds = torch.max(outputs, 1)
            test_loss += loss.item()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    return test_loss / len(test_loader), correct / total


def calibrate(dataloader: DataLoader, model: nn.Module, loss_fn):
    """
    Calibrate the model using the provided dataloader.
    Compute the loss for the purpose of information.
    Returns the average loss.
    """
    model.eval()
    avg_loss = 0.0
    total_batches = len(dataloader)
    
    with torch.no_grad():
     for batch_idx, (data,target) in enumerate(dataloader):
        
            inputs, targets = data.to(DEVICE), target.to(DEVICE)
    
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss = loss_fn(outputs, targets)
            avg_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"  Calibration Batch {batch_idx}: Loss: {loss.item():.4f}")
    
    avg_loss /= total_batches
    print(f" Calibration complete. Average Loss: {avg_loss:.4f}")
    return avg_loss, model, loss_fn, None

def get_quant_model(nn_model: nn.Module, example_input: torch.Tensor, total_epochs: int, weight_bitwidth: int,
        activation_bitwidth: int, quantization_method: str, quantization_device_type: str, bias_calibration_factor: int = 0) -> nn.Module:
    """
    Convert the torch model to quant wrapped torch model. The function requires
    an example input to convert the model.
    """

    '''
    The QAT wrapper module does the preparation like in:
    quant_model = quantize_fx.prepare_qat_fx(nn_model, qconfig_mapping, example_input)
    It also uses an appropriate qconfig that imposes the constraints of the hardware.

    The api being called doesn't actually pass qconfig_type - so it will be defined inside. 
    But if you need to pass, it can be defined.
    '''
    is_qat = (quantization_method == 'QAT')

    if weight_bitwidth is None or activation_bitwidth is None:
        '''
        # 8bit weight / activation is default - no need to specify inside.
        qconfig_type = {
            'weight': {
                'bitwidth': 8,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': True,
                'range_max': None,
                'fixed_range': False
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
        qconfig_type = None
    elif weight_bitwidth == 8:
        qconfig_type = {
            'weight': {
                'bitwidth': weight_bitwidth,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': True,
                'range_max': None,
                'fixed_range': False
            },
            'activation': {
                'bitwidth': activation_bitwidth,
                'qscheme': torch.per_tensor_symmetric,
                'power2_scale': True,
                'range_max': None,
                'fixed_range': False,
                'histogram_range': 1
            },
        }
    elif weight_bitwidth == 4:
     
        qconfig_type = {
            'weight': {
                'bitwidth': weight_bitwidth,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': False,
                'range_max': None,
                'fixed_range': False,
            },
            'activation': {
                'bitwidth': activation_bitwidth,
                'qscheme': torch.per_tensor_symmetric,
                'power2_scale': False,
                'range_max': None,
                'fixed_range': False,
                'histogram_range': 1
            },
        }
    elif weight_bitwidth == 2:
        qconfig_type = {
            'weight': {
                'bitwidth': weight_bitwidth,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': False,
                'range_max': None,
                'fixed_range': False,
                'quant_min': -1,
                'quant_max': 1,
            },
            'activation': {
                'bitwidth': activation_bitwidth,
                'qscheme': torch.per_tensor_symmetric,
                'power2_scale': False,
                'range_max': None,
                'fixed_range': False,
                'histogram_range': 1
            }
        }
    else:
        raise RuntimeError("unsupported quantization parameters")
 
    if quantization_device_type == 'TINPU':
        if quantization_method == 'QAT':
            quant_model = TINPUTinyMLQATFxModule(nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs)
        elif quantization_method == 'PTQ':
            quant_model = TINPUTinyMLPTQFxModule(nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs, bias_calibration_factor=bias_calibration_factor)
        else:
            raise RuntimeError(f"Unknown Quantization method: {quantization_method}")
        #
    elif quantization_device_type == 'GENERIC':
        if quantization_method == 'QAT':
            quant_model = GenericTinyMLQATFxModule(nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs)
        elif quantization_method == 'PTQ':
            quant_model = GenericTinyMLPTQFxModule(nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs)
        else:
            raise RuntimeError(f"Unknown Quantization method: {quantization_method}")
        
    else:
        raise RuntimeError(f"Unknown Quantization device type: {quantization_device_type}")

    
    return quant_model

def calibrate_model(model: nn.Module, dataloader: DataLoader, total_epochs: int) -> nn.Module:
    """
    Calibrate the model for PTQ - (torch model or qat wrapped torch model) with the given train dataloader,
    learning_rate and loss are not needed for PTQ / calibration as backward / back propagation is not performed.
    loss_fn is used here only for the purpose of information - to know how good is the calibration.
    """
    # loss_fn for multi class classification
    loss_fn = torch.nn.CrossEntropyLoss()
    
   # with torch.no_grad():
    for epoch in range(total_epochs):
        # train the model for an epoch
        loss, model, loss_fn, opti = calibrate(dataloader, model, loss_fn)
        last_lr = 0
        print(f"Epoch: {epoch+1}\t Loss: {round(loss, 5)}")

    return model

def export_model(quant_model, example_input: torch.Tensor, model_name: str, with_quant: bool = False):
    """
    Export the quantized model and print its layer-wise quantization parameters.
    """

    quant_model.to(DEVICE)

    # Convert model using FX Graph-based quantization if needed
    if with_quant:
        if hasattr(quant_model, "convert"):
            print(" Running `convert()` on quant_model...")
            quant_model = quant_model.convert()

        else:
            quant_model = quantize_fx.convert_fx(quant_model.module)
   
    #  Export to ONNX
    example_input=torch.randn(1,1,28,28).to(DEVICE)
    if hasattr(quant_model, "export"):
        print(" Exporting to ONNX...")
        quant_model.export(example_input, model_name, input_names=['input'])
    else:
        torch.onnx.export(quant_model, example_input, model_name, input_names=['input'])

    print("Model exported successfully")
    return quant_model

def validate_model(model: nn.Module, test_loader: DataLoader, num_categories: int, categories_name: List[str]) -> float:
    """
    The function takes the model (torch model or qat wrapped torch model), torch dataloader
    and the num_categories to give the confusion matrix and accuracy of the model.
    """
    model.eval()
    y_target = []
    y_pred = []

    for batch_idx, (data,target) in enumerate(test_loader):
  
        X = data.clone().to(DEVICE)
        y = target.clone().to(torch.long).to(DEVICE)

        # make prediction for the current batch
        pred = model(X)

        # take the max probability among the classes predicted
        _, pred = torch.max(pred, 1)
        y_pred.append(pred.cpu().numpy())
        y_target.append(y.cpu().numpy())

    y_pred = np.concatenate(y_pred)
    y_target = np.concatenate(y_target)
    categories_idx = np.arange(0, num_categories, 1)
    # create a confusion matrix
    cf_matrix = confusion_matrix(y_target, y_pred)
    df_cm = pd.DataFrame(cf_matrix, 
                         index=[str(categories_name[i]) for i in categories_idx],
                         columns=[str(categories_name[i]) for i in categories_idx])
    
    print()
    print("Confusion Matrix")
    print(df_cm)
    
    # Accuracy
    accuracy = np.diag(cf_matrix).sum() / np.array(cf_matrix).sum()
    return accuracy

def validate_saved_model(model_name: str, dataloader: DataLoader) -> float:
    """
    Validate the saved ONNX model using the test DataLoader.
    """
    correct_predictions = 0
    total_predictions = 0

    ort_session_options = ort.SessionOptions()
    ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    ort_session = ort.InferenceSession(model_name, ort_session_options)

    for batch, (data,target) in  enumerate(dataloader):
        X = data.numpy()  
        Y = target.numpy()

        
        outputs = ort_session.run(None, {'input': X})  

        # Convert predictions to class indices
        preds = np.argmax(outputs[0], axis=1)

        # Count correct predictions
        correct_predictions += np.sum(preds == Y)
        total_predictions += Y.shape[0]

    accuracy = round(correct_predictions / total_predictions, 5)
    return accuracy

def set_seed(SEED):
        # set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)  
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    #https://pytorch.org/docs/stable/notes/randomness.html
    #these flags are for reproducibility
    cudnn.deterministic = True  
    cudnn.benchmark = False    
    os.environ['PYTHONHASHSEED'] = str(SEED)

class LeNet5(nn.Module):
    def __init__(self): 
        super(LeNet5, self).__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.bn0(x)
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        return y

    
if __name__ == '__main__':

    MODEL_NAME = "mnist.onnx"
    CATEGORIES_NAME = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    NUM_EPOCHS = 14
    LEARNING_RATE = 0.01
    QUANTIZATION_METHOD = 'QAT' #'PTQ' #'QAT' #None
    WEIGHT_BITWIDTH = 8 #2 #4 #8
    ACTIVATION_BITWIDTH = 8 #8 #4 #2
    QUANTIZATION_DEVICE_TYPE = 'TINPU' #'TINPU', 'GENERIC'
    NORMALIZE_INPUT = False #True, #False
    NUM_CATEGORIES = 10 
    BATCH_SIZE = 64
    SEED = 42
    MODEL_TRAINING = True
    LOAD_MODEL_FROM_FILE = False
    LOAD_CHECKPOINT_FROM_FILE = False

    assert not (LOAD_MODEL_FROM_FILE and LOAD_CHECKPOINT_FROM_FILE), 'only one of LOAD_MODEL_FROM_FILE and LOAD_CHECKPOINT_FROM_FILE'

    assert QUANTIZATION_DEVICE_TYPE != 'GENERIC' or (not NORMALIZE_INPUT), \
        'normalizing input with BatchNorm is not supported for the export format used for Generic Quantization. Please set NORMALIZE_INPUT to False.'
    
    set_seed(SEED)
        
    #Downloading and preparing the dataset
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])
    train_ds = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=1)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=1,drop_last=True)

    # build model
    model = LeNet5().to(DEVICE)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f" Total parameters: {total_params:,}")
    
    set_seed(SEED)

    #Define the dataloaders
  
    example_input, _ = next(iter(test_loader))
    example_input = torch.unsqueeze(example_input[0],0).to(DEVICE)  # Add channel dimension

    nn_model = None
    
    #Import model structure
    if LOAD_MODEL_FROM_FILE:
        nn_model = torch.load(os.path.join('trained_models', 'mnist_pb2pth_model.pth'))
    else:
        nn_model = LeNet5().to(DEVICE)

    if LOAD_CHECKPOINT_FROM_FILE:
        checkpoint = torch.load(os.path.join('trained_models', 'mnist_pb2pth_checkpoint.pth'))
        nn_model.load_state_dict(checkpoint)

    #Train and Validate fp32 model
    if MODEL_TRAINING:
        nn_model = train_model(nn_model, train_loader, NUM_EPOCHS, LEARNING_RATE)
    print("Validating FP32 Model")
    accuracy = validate_model(nn_model, test_loader, NUM_CATEGORIES , CATEGORIES_NAME)
    export_model(nn_model, example_input, MODEL_NAME, with_quant=False)
    print("FP32 model accuracy is", accuracy)

    if QUANTIZATION_METHOD in ('QAT', 'PTQ'):

        MODEL_NAME = 'quant_' + MODEL_NAME
        quant_epochs = int(NUM_EPOCHS*2) if ((WEIGHT_BITWIDTH<8) or (ACTIVATION_BITWIDTH<8)) else max(NUM_EPOCHS//2, 5)
        quant_model = get_quant_model(nn_model, example_input=example_input, total_epochs=quant_epochs, 
                                      weight_bitwidth=WEIGHT_BITWIDTH, activation_bitwidth=ACTIVATION_BITWIDTH, 
                                      quantization_method=QUANTIZATION_METHOD, quantization_device_type=QUANTIZATION_DEVICE_TYPE,
                                      bias_calibration_factor=0.0)
    
        if QUANTIZATION_METHOD == 'QAT':
            quant_learning_rate = (LEARNING_RATE/100) #if ((WEIGHT_BITWIDTH<8) or (ACTIVATION_BITWIDTH<8)) else (LEARNING_RATE/10)
        
            quant_model = train_model(quant_model, train_loader, quant_epochs, quant_learning_rate)
    
        elif QUANTIZATION_METHOD == 'PTQ':
            quant_model = calibrate_model(quant_model, train_loader, quant_epochs)
        
        accuracy = validate_model(quant_model, test_loader, NUM_CATEGORIES, CATEGORIES_NAME)
        print(f"{QUANTIZATION_METHOD} Model Accuracy: {round(accuracy, 5)}\n")
        if WEIGHT_BITWIDTH == 8 and False:
            export_model(quant_model.module, example_input, 'qdq_' + MODEL_NAME, with_quant=False)
        quant_model = export_model(quant_model, example_input, MODEL_NAME, with_quant=True)

        
    else:
        print("No Quantization method is specified. Will not do quantization.")
     
    test_loader_onnx  = torch.utils.data.DataLoader(test_ds,  batch_size=1, num_workers=1,drop_last=True)
    accuracy = validate_saved_model(
        "quant_mnist.onnx", test_loader_onnx)
    print(f"Exported ONNX Quant Model Accuracy: {round(accuracy, 5)}")