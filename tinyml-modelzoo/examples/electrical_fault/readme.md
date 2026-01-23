# Electrical Fault Detection and Classification: Focus on solving multicollinearity problem
### - Tushar Sharma, Adithya Thonse, Fasna S 
<hr>

## Overview

The Electrical Fault Dataset is a multivariate time series dataset. It is obtained by modelling a 3 phase - transmission line of power system on MATLAB Simulink. Three-phase systems or electrical power systems often use three sinusoidal voltages (Va, Vb, Vc) that are phase-shifted by 120 degrees. Each phase also carries a current (Ia, Ib, Ic). The dataset is designed to find faults in transmission line and classify the type of fault using the line voltage and current. 

The fault can be Line-to-line, Line-to-ground, Line-to-line-to-ground and more. Line-to-line (LL) fault is a fault between two phase conductors (e.g., A-B). This typically appears as a short-duration high-energy event affecting particular frequency bins. Line-to-ground (LG) fault: a fault between a phase conductor and ground. Pattern differs from LL faults and can be identified using combinations of voltage and current measurements. LLG / LLLG: multi-conductor faults involving two or three lines and possibly ground. These create distinct signatures across voltage and current channels.

<p align="center">  
    <img src="assets/simulink.png" width="280" alt="Simulink Model">
</p>

There are 6 measurable parameters/variables - **(Va, Vb, Vc, Ia, Ib, Ic)** i.e the voltage and current of three phases. 
There are two dataset files present in the compressed zip: [electrical_fault_raw.zip](https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/electrical_fault_raw.zip)

- `detect_dataset.xlsx` : which finds if there is a fault or not. Target = [Output (S)]
    - There is one target only i.e. Output (S) which has two unique values (0, 1) denoting fault and no fault.
- `classData.csv` : which classifies the type of fault Target = [G, C, B, A]
    - There are 4 target variables i.e G (Ground), C (Node C), B (Node B), A (NodeA). The value of each target is either 0 or 1.
    - Examples [G, C, B, A]:
        - [0, 0, 0, 0] means No Fault
        - [1, 0, 0, 1] means LG Fault btw Ground and Node A
        - [0, 0, 1, 1] means LL Fault btw Node A and Node B
        - [1, 0, 1, 1] means LLG Fault btw Node A, Node B and Ground
        - [0, 1, 1, 1] means LLL Fault btw all Nodes
        - [1, 1, 1, 1] means LLLG Fault btw all Nodes and Ground

For this example we will be using `detect_dataset.xlsx` to detect whether there is electrical fault or not.

## Downloading dataset

Prepare the zipped dataset by running the electrical_fault python file. The script will create zipped dataset as `electrical_fault_dataset.zip`. 
```bash
cd examples/electrical_fault
python electrical_fault.py
```
The path of this zipped dataset file is already mentioned in [configuration](config.yaml) yaml, make sure it is same.

```yaml
dataset:
    input_data_path: 'examples/electrical_fault/electrical_fault_dataset.zip'
```

## Usage in TinyML ModelMaker

This zipped dataset is designed to work with TinyML ModelMaker. Run the modelmaker with the yaml [configuration](config.yaml) using the below code.

```bash
run_tinyml_modelmaker.sh examples/electrical_fault/config.yaml
```

1. `run_tinyml_modelmaker.sh` is the script to run modelmaker. It take two required arguments.
2. `examples/electrical_fault/config.yaml` path of the yaml configuration to run

The users can configure the yaml [configuration](config.yaml) to change parameters related to **data preprocessing feature extraction**, training, testing, model and model compilation. In this example, we will configure the parameters of feature extraction. 

## What if Multicollinearity in your dataset ?

[Multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity) is a condition when independent variable posses a linear relationship with one or more than one independent variable. Raw voltage and current channels can be strongly correlated (for example, Vc may be nearly a linear combination of Va and Vb in some operating conditions). In the time domain this redundancy can confuse simple models. This gives rise to poor prediction of weights of ML model during training.

Transforming to the frequency domain (FFT) captures how energy is distributed across frequencies, and binning groups similar spectral regions together. Spectral features often decorrelate spatially correlated channels because they focus on frequency content rather than instantaneous amplitude. After binning and optional log-scaling, the model sees compact, lower-dimensional features that are more robust to correlated input channels.

In this electrical fault dataset, the independent variables are current and voltages of Phase A, B, C. If we check the collinearity between these variables. We find that Multicollinearity exists between some independent variables.


| Independent Variable |  Ia  |  Ib  |  Ic  |     Va    |     Vb    |    Vc     |
|      :----:          | :--: | :--: | :--: |    :--:   |    :--:   |   :--:    |
|        Ia            | 1.00 |-0.49 |-0.45 |    0.23   |    0.69   | **-0.94** |
|        Ib            |-0.49 | 1.00 |-0.54 | **-0.95** |    0.26   |   0.72    |
|        Ic            |-0.45 |-0.54 | 1.00 |    0.74   | **-0.94** |   0.17    |
|        Va            | 0.23 |-0.95 | 0.74 |    1.00   |   -0.52   |  -0.51    |
|        Vb            | 0.69 | 0.26 |-0.94 |   -0.52   |    1.00   |  -0.46    |
|        Vc            |-0.94 | 0.72 | 0.17 |   -0.51   |   -0.46   |   1.00    |


To solve the problem of Multicollinearity, we can do one or more of the following:
- Remove the highly correlated features
- Perform feature extraction to do dimensionality reduction of features

## Feature Extraction is the solution

In this example we will explore the method to do dimensionality reduction using `FFT` and `BINNING` of features. The **data preprocessing feature extraction** section of yaml [configuration](config.yaml) can be used to configure it.

When modifying the configuration file to disable feature extraction (as illustrated below) in modelmaker, you'll encounter an error that prevents proper model training with effective hyperparameters. This issue stems from *multicollinearity* in the data.

```yaml
data_processing_feature_extraction:
   feature_extraction_name: Custom_Default
   feat_ext_transform: []
```

Now, with this information lets see how feature extraction will effect this error.

- FFT related options: FFT_FE, FFT_POS_HALF, DC_REMOVE, ABS
    1. `FFT_FE` is used to perform fft on a frame
    2. `FFT_POS_HALF` takes the 1st half of the fft which is symmetrical from middle
    3. `DC_REMOVE` removes the DC component of the FFT
    4. `ABS` takes the magnitude of the real and imaginary values of FFT
- Binning related options: BINNING
    1. `BINNING` performs the binning of magnitude of fft values
- Other options: LOG_DB, CONCAT
    1. `LOG_DB` takes the log of binned values
    2. `CONCAT` does concatenation of current features with features from previous frames

### Example configuration walkthrough (numbers)
- `frame_size = 256`: using 256-sample frames. If the sampling rate is 2048 Hz, this corresponds to a time window of 125 ms.
- After FFT, positive half of spectrum length is `frame_size/2 = 128` bins.
- `feature_size_per_frame = 32` with binning: each feature aggregates `128 / 32 = 4` FFT bins (simple uniform grouping).
- `num_frame_concat = 4`: the final model input has `32 * 4 = 128` values per channel per example.

We have to add FFT and few more transforms in this `transform` variable.
```yaml
data_processing_feature_extraction:
    feat_ext_transform: ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'LOG_DB', 'CONCAT']
```

### Configuration 1: FFT

Next we will define our features shape.
```yaml
data_processing_feature_extraction:
    feat_ext_transform: ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'LOG_DB', 'CONCAT']
    
    frame_size: 256
    feature_size_per_frame: 128
    num_frame_concat: 1
```

1. `frame_size`: slices frame from the dataset of size frame_size
2. `feature_size_per_frame`: size of binned features from one frame
3. `num_frame_concat`: number of frames used for concatenating features

After doing the above changes in yaml [configuration](config.yaml) file. Run the modelmaker again for this dataset.

```bash
run_tinyml_modelmaker.sh examples/electrical_fault/config.yaml
```

You can see that, you don't encounter any error during modelmaker run. This is because the feature extraction was succesfully able to mitigate the mulitcollinearity problem. This will resolve the error of to train the model properly with good hyper parameters.

### Configuration 2: FFT + BIN

Another feature extraction is to perform `FFT` with Binning. For this, we need to add `BINNING` to transforms. The feature size for each frame would become half of the frame size and can be reduced further based on feature_size_per_frame selected. So, yaml configuration would look like.

```yaml
data_processing_feature_extraction:
    feat_ext_transform: ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'BINNING', 'ABS', 'LOG_DB', 'CONCAT']
    
    frame_size: 256
    feature_size_per_frame: 32
    num_frame_concat: 4
```

The first configuration employed individual frames (num_frame_concat: 1), while the second configuration utilized a sequence of 4 frames (num_frame_concat: 4). To decrease the model's size, you can lower the number of frames processed. For instance, setting num_frame_concat to 2 in the second configuration would approximately halve both the model size and processing time on your device.

## Performance on device

We benchmarked the performance of the `CLS_ResCat_3k` model. The device used is F28P55x which comes with a HW accelearator (TINPU) to give low latency performance on ML models. Numbers are provided for running the model on NPU & CPU. Here both the configuration of Feature extraction produces the same architecture of model, so the model performance will be same. We clubbed the two configuration as 'with Feature Extraction'.

|              Configuration             | AI Model Cycles | Inference Time (us) | Flash Usage (B) | SRAM Usage (B) |
|----------------------------------------|-----------------|---------------------|-----------------|----------------|
|  **NPU (without Feature Extraction)**  |  Bad Training   |    Bad Training     |   Bad Training  |  Bad Training  |
|  **NPU (with Feature Extraction)**     |      125509     |       836.73        |      3175       |     3846       |
|  **CPU (with Feature Extraction)**     |      500860     |      3339.07        |      2995       |     4992       |

<hr>
Update history:
[29th Dec 2025]: Compatible with v1.2 of Tiny ML Modelmaker
[12th Mar 2025]: Compatible with v1.0 of Tiny ML Modelmaker
