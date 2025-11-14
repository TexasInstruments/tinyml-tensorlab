# MNIST Digit Classification

## Installation Instructions

To run the MNIST Digit Classification example, execute the following command:

```commandline
python main.py
```

This MNIST Digit Classification example is a PyTorch implementation of the classic LeNet-5 Convolutional Neural Network, extended to include Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ) using the TINPUTinyMLQuantFxModule framework.
This example demonstrates a complete TinyML workflow — from model training to quantization and ONNX export — optimized for deployment on TI NPUs.

Image classification on the MNIST dataset serves as a fundamental benchmark in computer vision. The dataset consists of grayscale handwritten digits (0–9), and the task involves recognizing these digits accurately.
This problem is particularly relevant to TinyML as it tests the feasibility of running efficient, low-power image recognition models on microcontrollers.


## Walkthrough of this Example

1. Download and prepare the MNIST dataset

2. Define the LeNet-5 CNN model

3. Train and test the baseline FP32 model

4. Wrap the trained model with the TINPUTinyMLQuantFxModule

5. Perform Quantization-Aware Training (QAT) or Post-Training Quantization (PTQ)

6. Validate the quantized model on the test dataset

7. Export both FP32 and quantized models to ONNX format

## Let’s Understand Each Step

### Prepare Dataloader

The MNIST dataset is a standard collection of handwritten digit images, each of size 28×28 pixels and labeled from 0 to 9.

It contains:

- 60,000 images for training

- 10,000 images for testing


Each image is normalized using mean = 0.1307 and std = 0.3081:

```python
transforms.Normalize((0.1307,), (0.3081,))
```

The dataset is downloaded and loaded through torchvision.datasets.

### LeNet-5 Convolutional Neural Network

The LeNet-5 architecture, originally proposed by Yann LeCun, is one of the earliest and most influential convolutional neural networks.
It’s particularly effective for small grayscale image classification tasks such as MNIST.

**Network structure:**

1. Input Layer: 1×28×28 grayscale image

2. Convolution 1: 8 filters, kernel size 3×3, Batch Normalization + ReLU + MaxPool(2×2)

3. Convolution 2: 16 filters, kernel size 3×3, Batch Normalization + ReLU + MaxPool(2×2)

4. Fully Connected Layers:

- FC1: 400 → 120

- FC2: 120 → 84

- FC3: 84 → 10 (output classes)

### Train and Test the Model

- Loss function: CrossEntropyLoss()

- Optimizer: torch.optim.SGD()

- Scheduler: CosineAnnealingLR()

The model is trained for 14 epochs and evaluated on the MNIST test set.


### TINPUTinyMLQuantFxModule Wrapper

### TINPUTinyMLQuantFxModule wrapper
Coming to the main part of the example.

- Wrap the trained neural network with **TINPUTinyMLQuantFxModule** and the number of epochs the 'QAT'/'PTQ' will be trained or calibrated for respectively.

### Train and Test ti_model for 'QAT'/'PTQ'
- With the same Loss function and optimizer, train and test the wrapped ti model with respective dataloader.

### Export ti_model
- Convert the ti_model from pytorch qdq layers to TI NPU int8/int4/int2 layers.
- Export the ti_model from TI NPU int8 layers to onnx with name as "quant_mnist.onnx"

