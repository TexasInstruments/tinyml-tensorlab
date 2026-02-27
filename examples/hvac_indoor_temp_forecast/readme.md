# HVAC Indoor Temperature Forecasting using LSTM : Timeseries Forecasting

## Overview  

Indoor temperature forecasting is a critical building block for intelligent HVAC control systems. Traditional HVAC controllers react to temperature deviations after they occur, which often leads to delayed corrections, unnecessary compressor activity, and increased energy consumption.

Machine learning based forecasting enables predictive control, where future indoor temperature trends can be anticipated using historical sensor data and environmental conditions. By accurately forecasting indoor temperature over a short prediction horizon, HVAC systems can plan smoother and more energy-efficient control actions.

**In this example, we demonstrate how a neural network based indoor temperature forecasting model can be trained offline and compiled for deployment using Tiny ML ModelMaker. The trained model is intended to run fully on-device and can serve as an input to higher-level control or optimization algorithms that determine compressor actuation strategies.**

## About Dataset

For this example, we use a synthetic HVAC dataset designed to emulate realistic indoor thermal behavior. You can find the dataset [here.](https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/hvac_indoor_temp_forecast_dataset.zip) 

The dataset captures the relationship between compressor operation, outdoor temperature, and resulting indoor temperature over time.
The dataset consists of time-series samples collected at uniform timesteps and includes the following signals:

1. **Compressor frequency (`compressorFrequency`)**
2. **Outdoor temperature (`outdoorTemperature`)**
3. **Indoor temperature (`indoorTemperature`)**

## Input Features and Prediction Target

**Input features**:<br>
1. **Past 5 `compressorFrequency` values**
2. **Past 5 `outdoorTemperature` values**
3. **Past 5 `indoorTemperature` values**


**Prediction target**:<br>

**`indoorTemperature` at the next timestep** (or over a short future horizon. You can do this by setting forecast_horizon parameter in the YAML file.)


## Usage in Tiny ML ModelZoo

Here is the command to use this dataset with Tiny ML ModelZoo:

```bash
./run_tinyml_modelzoo.sh examples/hvac_indoor_temp_forecast/config.yaml
```

Users can configure the model pipeline using a YAML configuration file (like shown in the command above), where different stages (dataset loading, data processing and feature extraction, training, testing, and compilation) can be enabled or disabled based on requirements.

## Configuring the YAML File

The YAML file is the core configuration file used in Tiny ML ModelMaker to define the pipeline for tasks such as dataset loading, model training, testing, and compilation. 

### common and dataset section

```yaml
common:
    target_module: 'timeseries'
    task_type: 'generic_timeseries_forecasting'
    target_device: 'F28P55'
    run_name: '{date-time}/{model_name}'
    
dataset:
    # enable/disable dataset loading
    enable: True
    dataset_name: hvac_indoor_temp_forecast
    input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/hvac_indoor_temp_forecast_dataset.zip
```

Important parameters configured here are: task_type set to `generic_timeseries_forecasting`, indicating the type of task being performed.`target_device` set to F28P55, ensuring that the compiled output is optimized for this device. Dataset loading is enabled, with the correct path to the dataset zip folder provided (`input_data_path`). The `data_dir` parameter is automatically detected based on the task type (forecasting tasks use 'files').

### data_processing_feature_extraction section

```yaml
data_processing_feature_extraction: # One or more can be cascaded in the list
    # transforms: 'Downsample SimpleWindow'
    data_proc_transforms: ['SimpleWindow'] # 'SimpleWindow' must be specified for forecasting tasks
    # Downsample
    #sampling_rate: 5
    #new_sr: 2
    # SimpleWindow
    frame_size: 5
    stride_size: 0.2
    forecast_horizon: 1 # Number of future timesteps to be predicted
    variables: 3 # takes the first 'variables' columns after the time columns as input to predict the target variables
    target_variables: [2] # Other format for `target_variables` specification: ['pm','torque'] or  [7,11] or ['7','11',]. If numbers which represent indices, are provided in list, assign column number 0 to the first non time column and continue numbering from there
```
For forecasting tasks, the SimpleWindow transform must be selected. Additionally, the important parameters configured here are: `frame_size` set to 5, representing the number of timestamps in a frame.`variables` set to 3, indicating the number of input variables.`target_variables`: Set to [2], as the indoorTemperature variable is located at index 2 (when column numbering starts from 0).

## training, testing and compilation section

```yaml
training:
    # enable/disable training
    enable: True
    # F28x generic timeseries model names: CLS_1k_NPU, CLS_4k_NPU, CLS_6k_NPU, CLS_13k_NPU
    # GUI only model names: ArcFault_model_200_t, ArcFault_model_300_t, ArcFault_model_700_t
    model_name: 'FCST_LSTM10' #'FCST_13k' #
    #model_name: 'FCST_13k'
    # model_spec: '../tinyml-mlbackend/proprietary_models/cnn_af_3l.py'
    model_config: ''
    batch_size: 6
    training_epochs: 10
    num_gpus: 0
    quantization: 2
    optimizer: adamw
    learning_rate: 0.003
    weight_decay: 0.001
    lr_scheduler: none
    output_int: False

testing:
    enable: True # False

compilation:
    # enable/disable compilationp
    enable: True # False
    keep_libc_files: True
```
The data is then trained using `FCST_LSTM10` model which is 611 paramters LSTM based model to learn indoor thermal dynamic. `quantization` is set to 2 which means we use TI NPU quantization. We are compiling this example using the ti-npu soft preset, which means the software emulation of the TI-NPU with some optimized operations.



## Results

On running the YAML file, the following accuracies were observed:

### **I. Float Train**
```
   INFO: root.main.FloatTrain.BestEpoch: Printing statistics of best epoch:
   INFO: root.main.FloatTrain.BestEpoch: Best epoch:9
   INFO: root.main.FloatTrain.BestEpoch: Overall SMAPE across all variables: 0.30%
   INFO: root.main.FloatTrain.BestEpoch: Per-Variable Metrics:
   INFO: root.main.FloatTrain.BestEpoch:   Variable indoorTemperature:
   INFO: root.main.FloatTrain.BestEpoch:       SMAPE of indoorTemperature across all predicted timesteps: 0.30%
   INFO: root.main.FloatTrain.BestEpoch:       R² of indoorTemperature across all predicted timesteps: 0.9971
   INFO: root.main.FloatTrain.BestEpoch:       Timestep 1:
   INFO: root.main.FloatTrain.BestEpoch:           SMAPE: 0.30%
   INFO: root.main.FloatTrain.BestEpoch:           R²: 0.9971
```
---

### **II. Quant Train**
```
   INFO: root.main           : Epoch 9: Best Overall SMAPE across all variables across all predicted timesteps so far: 0.80% (Epoch 3)
   INFO: root.main.QuantTrain.BestEpoch: Printing statistics of best epoch:
   INFO: root.main.QuantTrain.BestEpoch: Best epoch:4
   INFO: root.main.QuantTrain.BestEpoch: Overall SMAPE across all variables: 0.80%
   INFO: root.main.QuantTrain.BestEpoch: Per-Variable Metrics:
   INFO: root.main.QuantTrain.BestEpoch:   Variable indoorTemperature:
   INFO: root.main.QuantTrain.BestEpoch:       SMAPE of indoorTemperature across all predicted timesteps: 0.80%
   INFO: root.main.QuantTrain.BestEpoch:       R² of indoorTemperature across all predicted timesteps: 0.9893
   INFO: root.main.QuantTrain.BestEpoch:       Timestep 1:
   INFO: root.main.QuantTrain.BestEpoch:           SMAPE: 0.80%
   INFO: root.main.QuantTrain.BestEpoch:           R²: 0.9893
```
---

### **III. Test Data**
```
   INFO: root.main.test_data : Variable indoorTemperature:
   INFO: root.main.test_data :   SMAPE of indoorTemperature across all predicted timesteps: 0.73%
   INFO: root.main.test_data :   R² of indoorTemperature across all predicted timesteps: 0.9857
   INFO: root.main.test_data :   Timestep 1:
   INFO: root.main.test_data :       SMAPE: 0.73%
   INFO: root.main.test_data :       R²: 0.9857
```

You can find the compiled model at: tinyml-modelmaker/data/projects/{dataset_name}/run/{date-time}/{model_name}/compilation


## Running on Device

We have compiled this example using the ti-npu soft preset for F28P55x device, which means the software emulation of the TI-NPU with some optimized operations. After successfully running Modelmaker, you will get four main files:

1. **Artifacts**:
   - `mod.a` and `tvmgen_default.h` are generated and stored in:
     ```
     data/projects/{dataset_name}/run/{date-time}/{model_name}/compilation/artifacts
     ```

2. **Golden Vectors**:
   - `user_input_config.h` and `test_vector.c` are stored in:
     ```
     data/projects/{dataset_name}/run/{date-time}/{model_name}/training/quantization/golden_vectors
     ```

These four files will be needed while running on device.

In this example, we will use the following setup:

- **Device**: LAUNCHXL-F28P55X
- **C2000Ware Version**: 6.00
- **Code Composer Studio (CCS)**: Version 20.3.0

Steps to run this example on-device can be found by following this guide: [Deploying Forecasting Models from ModelMaker to Device](../../docs/deploying_forecasting_models_from_modelmaker_to_device/readme.md)

Upon flashing and running the project we can see the model output matches the golden vectors.

<p align="center">
    <img src="ondevice_results.png" style="width:30vw; max-width:400px; height:auto;">
</p>



## Performance Metrics

Here are the key performance metrics for the model running on the device:

| Metric               | Value       |
|----------------------|-------------|
| **Device Name**      | F28P55x     |
| **AI Model Cycles**  | 199432      |
| **Inference Time**   | 1329.55 µs  |
| **Results Match**    | TRUE        |
| **Model Size**       | 9452 bytes  |
| **Model Flash**      | 9174 bytes  |
| **Model SRAM**       | 278 bytes   |
| **Application Size** | 229 bytes   |
| **Application FLASH**| 227 bytes   |
| **Application SRAM** | 2 bytes     |

<table align="center">
  <tr>
    <td align="center">
      <img src="performance_metrics_graphs/flash_usage_graph.png" width="100%">
      <br>Flash Usage
    </td>
    <td align="center">
      <img src="performance_metrics_graphs/sram_usage_graph.png" width="100%">
      <br>SRAM Usage
    </td>
  </tr>
</table>
<hr>
Update history:
[23rd Dec 2025]: Compatible with v1.3 of Tiny ML Modelmaker