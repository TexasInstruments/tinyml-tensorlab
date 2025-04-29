# FMNIST Image Classification

FMNIST is fashion MNIST dataset comprises of 60000 training and 10000 testing samples. The dataset is classified in 10 classes. In this example we use a classification model. The model comprises of combination of Linear Relu layers. This is a beginner friendly example to get started on using Tiny ML Model Optimization.

## Walkthrough of this Example
1. Create train and test dataloader from torch.datasets
2. Create a classification Neural Network
3. Train and test the model on dataloader
4. Wrap the trained model around TINPUTinyMLQATFxModule 
5. Train and test this ti_model for QAT
6. Export the quantized model
7. Rename the input node of model
8. Save the ONNX model

## Let's understand each step

### Prepare Dataloader
- Download the FMNIST dataset from torch.datasets. For training and testing we need two dataloader, one will be train_dataloader and another will be test_dataloader.
- We define the batch size of these dataloader.

### Neural Network
- Create a simple neural network comprising of linear and relu layers. Lets call this multiple linear and relu layers as linear_relu_stack.
- The linear_relu_stack expects the input to be the flattened image shape and output is the number of classes.

### Train and Test Model
- Loss function: nn.CrossEntropyLoss()
- Optimizer: torch.optim.SGD()
- The model is trained and tested for 5 epochs. The model is then evaluated on test dataloader. After evaluation the model is exported with name as "fmnist.onnx".

### TINPUTinyMLQATFxModule wrapper
Coming to the main part of the example.

- Wrap the trained neural network with **TINPUTinyMLQATFxModule** and the number of epochs the QAT will be trained for.

### Train and Test ti_model
- With the same Loss function and optimizer, train and test the wrapped ti model with respective dataloader.

### Export ti_model
- Convert the ti_model from pytorch qdq layers to TI NPU int8 layers.
- Export the ti_model from TI NPU int8 layers to onnx with name as "fmnist_int8.onnx"

### Renaming Node
- Rename the input node of the converted ti_model to 'input'. This will help in inference.

### Saving ONNX model
- Overwrite the exported ti_model with this updated node name model and save model with same file name "fmnist_int8.onnx"