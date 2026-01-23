# Audio Keyword Spotting

## Installation Instructions

```
To run the Audio Keyword spotting example run the following command:

```commandline
python main.py
```


This  'audio keyword spotting' example is a PyTorch implementation of the Keyword Spotting application, one of the benchmark applications in MLPerf Tiny. MLPerf Tiny is an open-source benchmarking suite specifically designed for Tiny ML systems. Developed collaboratively by over 50 organizations from both academic and industrial sectors, the MLPerf Tiny inference benchmark suite offers four standardized benchmarks. These benchmarks are uniquely tailored to evaluate the key performance metrics crucial in Tiny ML applications: latency, energy consumption, and accuracy. By assessing these three aspects simultaneously, the suite effectively captures the complex trade-offs inherent in Tiny ML systems.

Identifying particular words and short expressions, often referred to as keyword spotting, represents a crucial application of machine learning in low-power environments. This technology plays a significant role in facilitating voice-based interactions between humans and devices. As voice commands become increasingly prevalent in our daily lives, the ability to accurately detect specific keywords while conserving energy has become a key focus in the development of smart, power-efficient systems.

## Walkthrough of this Example
1. Create train and test dataloader from torch.datasets
2. Create a classification Neural Network
3. Train and test the model on dataloader
4. Wrap the trained model around TINPUTinyMLQuantFxModule
5. Train and test this ti_model for 'QAT'/'PTQ'
6. Export the quantized model
8. Save the ONNX model

## Let's understand each step

### Prepare Dataloader
- The MLPerf Tiny benchmark for keyword spotting utilizes a modified version of the Google Speech Commands v2 dataset. While the original dataset contains 30 classes, MLPerf focuses on a subset of 10 specific keywords: "down," "go," "left," "no," "on," "off," "right," "stop," "up," and "yes."

- In addition to these 10 classes, MLPerf introduces two new categories:

- 1."Unknown": This class encompasses samples from the remaining 20 keywords in the original dataset.
- 2."Silence": This class is created using segments extracted from the background noise samples provided in the original dataset.
- The dataset is then divided into three sets: Training set: 80%, Validation set: 10% andTest set: 10%.

- A key aspect of the MLPerf approach is addressing the class imbalance, particularly in the test set. While the training and validation sets maintain a higher proportion of "unknown" samples, reflecting the original distribution, the test set is deliberately balanced. This is achieved by randomly selecting 408 samples from the initial 6,523 "unknown" samples in the test set. The training and validation sets preserve the original class distribution, including a larger number of "unknown" samples.The test set offers a balanced representation across all classes, ensuring a fair evaluation of the model's performance on both the target keywords and the additional "unknown" and "silence" categories.

### Depthwise Separable Convolutional Neural Network
- The keyword spotting system employs a Depthwise Separable Convolutional Neural Network (DSCNN). This architecture consists of an initial 2D convolutional layer followed by four depthwise separable convolutional blocks. Each block includes depthwise and pointwise convolutions, batch normalization, and ReLU activation. The model uses dropout for regularization, global average pooling to reduce spatial dimensions, and a final fully connected layer for classification.

### Train and Test Model

- Loss function: nn.CrossEntropyLoss()
- Optimizer: torch.optim.SGD()
- Scheduler: optim.lr_scheduler.CosineAnnealingLR()
- The model is trained and tested for 10 epochs. The model is then evaluated on test dataloader. After evaluation the model is exported with name as "kws.onnx".

### TINPUTinyMLQuantFxModule wrapper
Coming to the main part of the example.

- Wrap the trained neural network with **TINPUTinyMLQuantFxModule** and the number of epochs the 'QAT'/'PTQ' will be trained or calibrated for respectively.

### Train and Test ti_model for 'QAT'/'PTQ'
- With the same Loss function and optimizer, train and test the wrapped ti model with respective dataloader.

### Export ti_model
- Convert the ti_model from pytorch qdq layers to TI NPU int8/int4/int2 layers.
- Export the ti_model from TI NPU int8 layers to onnx with name as "quant_kws.onnx"
