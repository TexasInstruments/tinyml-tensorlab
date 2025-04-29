# Motor Fault Time Series Classification

Motor Fault Dataset is a subset of motor fault dataset prepared by TIs Internal Team. The dataset consists of vibrations received from running motors. The motors can be classified as Normal, Localized Fault, Erosion Fault, Flaking Fault. The dataset has 4800030 samples. Each sample has 4 variables and 1 target.

This example will use a Deep Learning model to train and classify the type of motor fault.

## Walkthrough of this Example
1. Create train and test dataloader from csv
2. Configure the training and quantization params
4. Wrap the trained model around `TinyMLFxModule`
5. Train and test this ti_model for `QAT/PTQ`
6. Export the quantized model

## Let's understand each step

### Prepare Dataloader
The MotorFault dataset is derived from a subset of an in-house dataset created by TI. It is structured by segmenting data into windows of data points. The data follows the NCHW format and is divided in a 4:1 ratio for training and testing dataloaders. The samples are shuffled within the dataloader.

### Configure Dataloader
This example comes with various configuration of dataloader
- **BATCH_SIZE** (N): Creates batches of windows
- **HIDDEN_CHANNELS** (C): Provide the hidden channels for convolution layers
- **WINDOW_LENGTH** (H): Decides the length of windows
- **WINDOW_OFFSET**: Offset of datapoint for each window

### Using Quantization
- **QUANTIZATION_METHOD**: QAT / PTQ
- **WEIGHT_BITWIDTH**: Bit width of weights
- **ACTIVATION_BITWIDTH**: Bit width of activations
- **QUANTIZATION_DEVICE_TYPE**: Generic quantization or TI-NPU supported quantization
- **NORMALIZE_INPUT**: Model normalizes the input or not

### Train and Test Quantization on CNN Model
For lower bitwidth in quantization, the epochs of training or calibration is increased, whereas for high bitwidth the epochs are reduced. This is done due to the ease of learning for high bitwidth. The QAT training occurs in `train_model` and calibration for PTQ is done using `calibrate_model`.

For the QAT training, the learning rate is lowered to avoid deviations from trained values. The model is tested on the test dataloader using the `validate_model` function

### Exporting the quantized model
The model is first converted and then exported using the `export_model` function. The accuracy of exported model which is the end-user model is also checked using `validate_saved_model`.