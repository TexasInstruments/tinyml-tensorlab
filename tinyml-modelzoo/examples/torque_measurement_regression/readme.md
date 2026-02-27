# Torque Measurement on PMSM Dataset: Running regression model on F28P55x
### -Tushar Sharma, Adithya Thonse, Fasna S, Abhijeet Pal, Jaswanth Jadda
<hr>

## Overview

Time series regression is a supervised machine learning approach where models learn to predict continuous numerical values based on time-ordered historical data. The algorithm trains by analyzing the relationship between input features and target variables across sequential timestamps, identifying temporal patterns, trends, and seasonality.

ModelMaker provides comprehensive support for this supervised learning technique, enabling you to:

- Train regression models on labeled time-sequenced data
- Evaluate prediction accuracy using appropriate metrics for continuous variables
- Deploy solutions that leverage learned temporal dependencies

For demonstration purposes, we'll walk through implementing a supervised time series regression model using the **Permanent Magnet Synchronous Motor (PMSM) Dataset**, where historical sensor readings and operational parameters serve as features to predict torque values as the target variable.

## About the dataset

This dataset contains sensor measurements collected from **Permanent Magnet Synchronous Motor (PMSM)** running on a controlled test bench. The motor is a prototype model from a German OEM, and the measurements were collected by the **LEA department at Paderborn University**. All signals are sampled at **2 Hz** (two samples per second).  

You can find the original dataset and its full description here:  [Electric Motor Temperature (Kaggle)](https://www.kaggle.com/wkirgsn/electric-motor-temperature)

**Sensor attributes in the dataset:**

| Column Name | Description |
|------------------|-------------|
| `i_d` | Current d-component (active component) |
| `i_q` | Current q-component (reactive component) |
| `u_d` | Voltage d-component (active component) |
| `u_q` | Voltage q-component (reactive component) |
| `motor_speed` | Motor speed |
| `ambient` | Ambient temperature |
| `coolant` | Coolant temperature |
| `pm` | Permanent magnet surface temperature |
| `stator_winding` | Stator winding temperature |
| `stator_tooth` | Stator tooth temperature |
| `stator_yoke` | Stator yoke temperature |
| `torque` | Torque induced by current |

In our regression example we’ll use `i_d, i_q, u_d, u_q, motor_speed, ambient, coolant, pm, stator_winding, stator_tooth` as **input features** to predict **target variable**:  **Motor torque (`torque`)**

## Usage in Tiny ML ModelZoo

You can run this example directly in **Tiny ML ModelMaker** using the following command:

```bash
./run_tinyml_modelzoo.sh ./examples/torque_measurement_regression/config.yaml
```
The model pipeline is configured using a YAML file, where you can enable or disable different stages such as dataset loading, data processing, feature extraction, training, testing, and compilation depending on your needs.

You can see the output from running modelmaker in the following folder **tinyml-modelmaker/data/projects/torque_measurement/run/{date-time}/{model}**.

## Configuring the YAML File

```yaml
data_processing_feature_extraction:
    data_proc_transforms: [SimpleWindow]
    variables: 10
    frame_size: 256 # 64, 128
    stride_size: 0.25 # 0.5 1
```

Try tweaking the frame size and stride size to see how it affects the model training. For this example, having a smaller frame size results in better r2-score. Tweaking them should produce similar results as shown in results section.

## How to run on device

### CCS Studio

Code Composer Studio (CCS) is a free integrated development environment (IDE) provided by Texas Instruments (TI) for developing and debugging applications for TI's micro-controllers and processors. It offers various examples for users to get started with their problem statement. One of the application is f28p55x_generic_timeseries_regression. We will use this example to run on device.

### Requirements

The CCS example *f28p55x_generic_timeseries_regression* requires 4 files from modelmaker. We will copy the files from modelmaker run to the CCS example project. 

1. C2000Ware 6.01.00.00
2. Location of example: *C:\ti\c2000\C2000Ware_6_01_00_00\libraries\ai\examples\generic_timeseries_regression\f28p55x*

## Running for Target Device

Run the modelmaker from command line. After the run is finished. Copy the 4 files (path present below) from Modelmaker to CCS Project. Build the CCS Project, flash the program and start debugging the application. Check for the variable *test_result* for different sets of test cases preset in test_vector.c.

### Compiled model files

- mod.a: The compiled model is present in this file. 
  - Path Modelmaker: *tinyml-modelmaker/data/projects/torque_measurement/run/{date-time}/{model}/compilation/artifacts/mod.a*
  - Path CCS Project: *f28p55x_generic_timeseries_regression/artifacts/mod.a*
- tvmgen_default.h: Header file to access the model inference APIs from mod.a 
  - Path Modelmaker: *tinyml-modelmaker/data/projects/torque_measurement/run/{date-time}/{model}/compilation/artifacts/tvmgen_default.h*
  - Path CCS Project: *f28p55x_generic_timeseries_regression/artifacts/tvmgen_default.h*

### Test data for device verification

- test_vector.c: Test cases to check if the model works on device currently
  - Path Modelmaker: *tinyml-modelmaker/data/projects/torque_measurement/run/{date-time}/{model}/training/quantization/golden_vectors/test_vector.c*
  - Path CCS Project: *f28p55x_generic_timeseries_regression/test_vector.c*
- user_input_config.h: Configuration of feature extraction library in SDK. 
  - Path Modelmaker: *tinyml-modelmaker/data/projects/torque_measurement/run/{date-time}/{model}/training/quantization/golden_vectors/user_input_config.h*
  - Path CCS Project: *f28p55x_generic_timeseries_regression/user_input_config.h*


Steps to run a regression example on-device can be found in the following guide: [Deploying Regression Models from ModelMaker to Device](../../docs/deploying_regression_models_from_modelmaker_to_device/readme.md)

## Model Metrics

The model successfully completed training on the dataset with notable performance metrics. Analysis of the training logs revealed that the floating-point model achieved an impressive MSE of 12.98 and an R2-Score of 0.994, demonstrating strong predictive accuracy. In comparison, the quantized model showed somewhat diminished but still acceptable performance with an MSE of 83.59 and an R2-Score of 0.963. The model when evalated for test data gave RMSE of 9.40, MSE of 88.36 and R2-Score of 0.98.

We have summarized the results for two frame sizes (128, 256) in the table below.

## Results

|**Frame Size**| **Float R2-Score (MSE)** | **Quant R2-Score (MSE)** | **Cycles** |
|--------------|--------------------------|--------------------------|------------|
|    **128**   |        0.994 (12.98)     |       0.963  (83.59)     |    114102  |
|    **256**   |        0.983 (36.82)     |       0.948 (113.36)     |    187445  |

Regression model with frame size `128` was able to learn the PMSM dataset better than frame size of `256`. On evaluation of test dataset, model with frame size `128` performed better on device.

<hr>
Update history:

[11th Dec 2025]: Compatible with v1.3.0 of Tiny ML Modelmaker
