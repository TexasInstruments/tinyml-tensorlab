# Motor Speed Prediction: Running regression model on F29H85x
### -Tushar Sharma, Adithya Thonse, Fasna S, Abhijeet Pal, Jaswanth Jadda
<hr>

## Overview

This dataset simulates 15,000 observations of three-phase induction motors operating under varied electrical and mechanical conditions. It is designed to capture realistic, physically-informed relationships between inputs (supply, motor, and operating parameters) and the target variable (motor speed in RPM).

The dataset combines clear, physics-based relations (e.g., frequency → synchronous speed) with non-linear interactions (e.g., load, voltage and temperature affecting slip). It is well suited for benchmarking regression models on domain-informed prediction tasks and for Tiny ML experiments where compact models must generalize from physically-structured inputs.

`Key feature groups`

- Supply parameters
  - **Voltage** (V): range 200–460 V; higher voltage typically reduces slip and keeps speed closer to synchronous speed.
  - **Current** (A): proportional to load; higher current indicates greater power draw and heating.
  - **Frequency** (Hz): either 50 or 60 Hz; determines the synchronous speed of the rotating magnetic field.
  - **Power factor**: 0.65–0.95, indicates how effectively electrical power is converted to work.

- Motor characteristics
  - **Poles**: {2, 4, 6, 8}; more poles → lower synchronous speed (but typically higher torque).
  - **Stator resistance** (Ω): affects losses and slip, especially with temperature changes.

- Operating conditions
  - **Load torque** (N·m): higher torque increases slip and reduces speed.
  - **Ambient temperature** (°C): higher temperature increases winding resistance and affects performance.

`Target`

- **Motor speed (RPM)**: the dependent variable we predict. It is derived from synchronous speed and slip.

Electromechanical relationships (summary)

<p align="center">  
    <img src="assets/induction_motor_relationships.png" width="720" alt="Image Alt">
</p>

- Synchronous speed (RPM): Sync = 120 × Frequency / Poles
- Slip (fraction): Slip = (Sync − Actual) / Sync — typical values are a few percent at rated load and can grow under heavy load.
- Actual speed: Actual = Sync × (1 − Slip)

Derived quantities included in the dataset

- Active power: P = V × I × PF
- Apparent power: S = V × I
- Thermal effects and simple resistance scaling are included so models must learn both linear and non-linear interactions.

## Usage in Tiny ML ModelMaker

Prepare the zipped dataset by running the induction_motor_speed_prediction python file. The script will create zipped dataset as `induction_motor_speed_prediction_dataset.zip`. 
```bash
cd examples/induction_motor_speed_prediction
python induction_motor_speed_prediction.py
```
The path of this zipped dataset file is already mentioned in [configuration](config.yaml) yaml, make sure it is same.

```yaml
dataset:
    input_data_path: 'examples/induction_motor_speed_prediction/induction_motor_speed_prediction_dataset.zip'
```

This zipped dataset is designed to work with Tiny ML ModelMaker. Run the modelmaker with the yaml [configuration](config.yaml) using the below code.

```bash
run_tinyml_modelzoo.sh examples/induction_motor_speed_prediction/config.yaml
```

1. `run_tinyml_modelzoo.sh` is the script to run modelmaker. It take two required arguments.
2. `examples/induction_motor_speed_prediction/config.yaml` path of the yaml configuration to run

The users can configure the yaml [configuration](config.yaml) to change parameters related to **data preprocessing, feature extraction**, training, testing, model and model compilation. In this example, we will configure the parameters of feature extraction. In this example we will be using REGR_1k model.

REGR_1k is a regression model designed with keeping memory in consideration. It consists of 2 BatchNorm+Conv+Relu layer and 2 light Linear layers. This model is compatible to run on TINIE HW accelerator.

## Output of Running modelmaker

After running the modelmaker you can find the following results for different size of frames. This dataset is not a timeseries dataset, so using frame size of 1 will produce better result.

|**Frame Size**| **Float R2-Score (MSE)** | **Quant R2-Score (MSE)** | **Exported R2-Score (RMSE)** |
|--------------|--------------------------|--------------------------|------------------------------|
|     **1**    |       1.000   (185.94)   |       0.999   (451.44)   |        0.990  (87.46)        |
|     **4**    |       0.997   (645.68)   |       0.997   (785.67)   |        0.990  (46.36)        |
|     **8**    |       0.978  (2347.63)   |       0.978  (2341.76)   |        0.970  (59.88)        |

## How to run on device

We will need [CCS Studio](https://www.ti.com/tool/CCSTUDIO), [F29H85x](https://www.ti.com/tool/LAUNCHXL-F29H85X) Microcontroller and its corresponding SDK [F29-SDK](https://www.ti.com/tool/F29-SDK).

### CCS Studio

Code Composer Studio (CCS) is a free integrated development environment (IDE) provided by Texas Instruments (TI) for developing and debugging applications for TI's micro-controllers and processors. It offers various examples for users to get started with their problem statement. One of the application is generic_timeseries_regression. We will use this example to run on device.

### Requirements

The CCS example *generic_timeseries_regression* requires 4 files from modelmaker. We will copy the files from modelmaker run to the CCS example project. 

1. f29h85x-sdk 1.03.00.00
2. Location of example: *C:\ti\f29h85x-sdk_1_03_00_00\examples\ai\generic_timeseries_regression\f29h85x*

## Running for Target Device

Run the modelmaker from command line. After the run is finished. Copy the 4 files (path present below) from Modelmaker to CCS Project. Build the CCS Project, flash the program and start debugging the application. Check for the variable *test_result* for different sets of test cases preset in test_vector.c.

### Compiled model files

- mod.a: The compiled model is present in this file. 
  - Path Modelmaker: *tinyml-modelmaker/data/projects/induction_motor_speed_prediction/run/{date-time}/{model}/compilation/artifacts/mod.a*
  - Path CCS Project: *generic_timeseries_regression/artifacts/mod.a*
- tvmgen_default.h: Header file to access the model inference APIs from mod.a 
  - Path Modelmaker: *tinyml-modelmaker/data/projects/induction_motor_speed_prediction/run/{date-time}/{model}/compilation/artifacts/tvmgen_default.h*
  - Path CCS Project: *generic_timeseries_regression/artifacts/tvmgen_default.h*

### Test data for device verification

- test_vector.c: Test cases to check if the model works on device currently
  - Path Modelmaker: *tinyml-modelmaker/data/projects/induction_motor_speed_prediction/run/{date-time}/{model}/training/quantization/golden_vectors/test_vector.c*
  - Path CCS Project: *generic_timeseries_regression/test_vector.c*
- user_input_config.h: Configuration of feature extraction library in SDK. 
  - Path Modelmaker: *tinyml-modelmaker/data/projects/induction_motor_speed_prediction/run/{date-time}/{model}/training/quantization/golden_vectors/user_input_config.h*
  - Path CCS Project: *generic_timeseries_regression/user_input_config.h*

Steps to run a regression example on-device can be found by in the following guide: [Deploying Regression Models from ModelMaker to Device](../../docs/deploying_regression_models_from_modelmaker_to_device/readme.md)

## Results by running on device

From the map file the memory usage is obtained and the cycles are measured using the profiling code. To do the same you can check the 
1. **profiling.h** provides APIs to calculate cycles consumed. It would be present in source of ai library.
2. **.map** file can be generated by building the CCS Project. It will be present in the build configuration folder you selected (by default RAM)

|**Frame Size**| **Flash Usage (B)** | **SRAM Usage (B)** | **Cycles** | **Inference Time (us)** |
|--------------|---------------------|--------------------|------------|-------------------------|
|     **1**    |         4306        |         128        |     1889   |           9.45          |
|     **4**    |         4582        |         130        |     3519   |          17.59          |
|     **8**    |         5292        |         256        |     7288   |          36.44          |


<hr>
Update history:

[6th Jan 2026]: Compatible with v1.3.0 of Tiny ML Modelmaker
