# Tiny ML Model Optimization

## Overview

Tiny ML Model Optimization is the toolkit for optimizing models. It offers an extensive set of operations to boost your model performance. In life cycle of model, from training, optimizing to deploying and inference, this repository helps in the optimization phase. 

The key features of Model Optimization:

**Basic**
- 8bit/4bit/2bit quantization
- Quantization Aware Training / Post Training Quantization
- Convert modules to leverage TI's NPU power
- Perform Inference on exported ONNX

**Advanced**
- Mixed Precision
- Calibrating Bias with Bias Calibration Factor

Using the tinyml-modeloptimization toolkit can be challenging and overwhelming for users. The example directory is designed to help users get started, guiding them from basic to advanced usage.

| Example     |  Dataset          | Description |
|:-----------------------:|:---------------:| :----: |
|[Image classification](./fmnist_image_classification)|[Fashion MNIST Dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html) | Simple Example, Dataloader from torch.datasets, LinearRelu NN Model, QAT, Supports TI-NPU, Renaming node, ONNX Inference |
|[Time series classification](./motor_fault_time_series_classification)| [Motor Fault Dataset](./motor_fault_time_series_classification/motor_fault_dataset.csv) | Dataloader from CSV, QAT/PTQ, 2b/4b/8b Weights, CNN Model, ONNX Inference |
|[Audio Keyword Spotting](./audio_keyword_spotting)| [SpeechCommands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) | Extensive Example, Dataloader from tensorflow dataset, DSCNN Model, Perform PTQ, Mixed Precision, Bias Calibration |





