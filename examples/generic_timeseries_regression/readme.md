# Generic Timeseries Regression: Hello World Example
### -Abhijeet Pal, Adithya Thonse, Tushar Sharma, Fasna Sharaf, Jaswanth Kumar
<hr>

## Overview

### TimeSeries Regression

Timeseries regression is predicting target variable based on independent variables and previous independent variables for context. It's widely used in industrial, appliance, and automotive sectors such as estimating weight loading of washing machine, motor torque measurement, etc. In **ModelMaker**, we support timeseries regression.

This example serves as a **"Hello World" introduction** to time series regression using the TinyML ModelMaker toolchain. This example demonstrates how to use **any generic timeseries regression task** with our toolchain. 

### The Simulated Dataset

To demonstrate timeseries regression, we use a dummy dataset. This dataset is a independent variable x, where x is randomly generated in range from 0 to 3 and the target variable y = 1.2 sin(x) + 3.2 cos(x)

- x is randomly generated in the range 0 to 3
- target variable y is y = 1.2 sin(x) + 3.2 cos(x)


This example will walk you through:
- How the dataset should be structured
- How to configure the YAML file for regression tasks
- Running the complete pipeline from training to compilation

## About the Dataset

The dataset consists of synthetically generated columns:

| Column Name | Description |
|-------------|-------------|
| `x` | randomly generated in range [0,3]|
| `y` | y = 1.2 sin(x) + 3.2 cos(x) |

**Dataset split:**
- Training: 7 files 
- Validation: 2 files 
- Test: 1 file 

Each file contains 5000 datapoints of generated x corresponding y variable
The dataset can be downloaded from here: [`generic_timeseries_regression_dataset'](https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/generic_timeseries_regression.zip)

## Preparing the Dataset

For regression tasks, **ModelMaker** expects the dataset to be packaged in a specific folder structure inside a **.zip** file like this:

```
{dataset_name}.zip/
│
├── files/
│    ├── {file1}/
│    ├── {file2}/
│    ├── {file3}/
|    ├──  .....
│    └── {fileN}/
│
└──annotations/
     ├──file_list.txt            # List of all the files in the dataset
     ├──instances_train_list.txt # List of all the files in the train set (subset of file_list.txt)
     ├──instances_val_list.txt   # List of all the files in the validation set (subset of file_list.txt)
     └──instances_test_list.txt  # List of all the files in the test set (subset of file_list.txt)
```

**Note:**

Unlike classification tasks, regression **always requires annotation files**. These tell the tool which files belong to training, validation, and testing sets. The data directory is automatically named 'files' for regression and forecasting tasks.

For this example, we have already prepared the dataset in the required format. You can find the zipped dataset at: [`generic_timeseries_regression_dataset`](https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/generic_timeseries_regression.zip)

## Usage in Tiny ML ModelZoo

You can run this example directly in **TinyML ModelMaker** using the following command:

```bash
./run_tinyml_modelzoo.sh ./examples/generic_timeseries_regression/config.yaml 
```

The model pipeline is configured using a YAML file, where you can enable or disable different stages such as dataset loading, data processing, feature extraction, training, testing, and compilation depending on your needs.

## Configuring the YAML file

### `common` section

Set the task type to `generic_timeseries_regression` along with other basic parameters as shown below:

```yaml
common:
    task_type: generic_timeseries_regression
    target_device: F28P55
```

### `dataset` section

Defines dataset details:
- **dataset_name**: Name for your dataset (appears in logs)
- **input_data_path**: Path to the dataset

Here is how we configured `dataset` section for our hello world dataset example:

```yaml
dataset:
    dataset_name: generic_timeseries_regression
    input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/generic_timeseries_regression.zip
```

### `data_processing_feature_extraction` section

<b>Under `data_processing_feature_extraction` section, you have to specify the following parameters mandatorily</b>:

- `variables`: Takes the first `variables` columns of the data files (after the time columns) as input to predict the target variables.

We can use data processing transforms such as SimpleWindow, LOG_DB, ABS, FFT_FE, NORMALIZE, etc. before training the dataset.

Here is an example on how to configure `data_processing_feature_extraction` in YAML file:

```yaml
data_processing_feature_extraction:
    data_proc_transforms:
    - SimpleWindow  
    frame_size: 10
    stride_size: 1
    variables: 1  # Only 1 feature used in this example for predicting target
```

### `training` section

You can configure training parameters here like `model_name`, `training_epochs`, `optimizer` etc. 
Here we are using an simple CNN model with 2k parameters model:

```yaml
training:
  optimizer: adam
  lr_scheduler: cosineannealinglr
  model_name: REGR_2k
  batch_size: 128
  training_epochs: 45 
  lambda_reg: 0.01
  num_gpus: 1 # 1 when using gpu if using cpu use 0
  quantization: 2 # 0 for float model, 2 for 8 bit quantized model
  partial_quantization: True # input batchnorm, first conv/linear layer and last fc layer are not quantized, default is False
```

### `compile` and `test` section

You can enable or disable compilation and testing as needed:

```yaml
testing: {}
compilation: {}
```

## Results

This section explains how we evaluate the regression model.

### Scoring Metrics

We mainly use two metrics for evaluation:

**1. RMSE (Root Mean Square Error)**

Measures the mean of the square root of the sum of squares of errors between predicted and actual values.

    - Range: 0 to inf.
    - Ideal value: 0 (lower is better).


**2. R² Score (Coefficient of Determination)**

Indicates how well predictions match the actual values. It is the explained variance of dataset by the regression model.

    - Range: (-∞, 1].
    - Ideal value: 1 (higher is better).

A score close to 1 means the model explains most of the variation in the data.

### How the Best Epoch is Chosen?

The model computes loss for each of the epochs. The epoch where the Mean squared loss (MSE) is the least is marked as the best running epoch, and the current model is stored. When an epoch with lesser RMSE is encountered, the saved checkpoint is updated with the current parameters of the model.

**1. Prediction Plots**

Can be found at `data/projects/{dataset_name}/run/{date-time}/{model_name}/training/quantization/post_training_analysis/`

We plot the prediction by the model vs the actual value, i.e. predicted vs actual plot for the dataset. Also, we plot histogram of Error (actual - predicted) plot along with metrics such as mean, standard deviation, etc.

<p align='center'>
<img src="./assets/generic_regression_actual_vs_predicted.png"
alt="float_train_prediction_plot" width="60%"/>
</p>

**2. Output vs Input File for Test Data**

They are located at `data/projects/{dataset_name}/run/{date-time}/{model_name}/training/quantization/post_training_analysis/`

 With filename results_on_test_set.csv. The file consists of two columns ground_truth, predicted. where ground truth is the true value and predicted is the predicted value by the model on the test data

Also you can see the compiled model at: `data/projects/{dataset_name}/run/{date-time}/{model_name}/compilation`


For the generic_timeseries_regression dataset the results on the test set are 

| Model | RMSE | R2 Score  |
|----------|----------|----------|
| Float Model   | 0.15  | 0.95  | 
| Partially Quantized Model  | 0.11   | 0.98 |
| Fully Quantized Model  | 0.11   | 0.97 |

For Float Model Configuration is :
```yaml
  quantization: 0 
  partial_quantization: False
```
For Partially Quantized Model configuration is:
```yaml
  quantization: 2
  partial_quantization: True
```

For Fully Quantized Model configuration is:
```yaml
  quantization: 2
  partial_quantization: False
```

## Running on Device

After successfully running ModelMaker, you will get the compiled model artifacts:

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

Steps to run this example on-device can be found by following this guide: [Deploying Regression Models from ModelMaker to Device](../../docs/deploying_regression_models_from_modelmaker_to_device/readme.md)

**Update history:**

[29th Jan 2025]: Compatible with v1.3 of Tiny ML ModelMaker