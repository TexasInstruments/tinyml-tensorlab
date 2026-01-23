
# Washing Machine Weight Loading
## Prediction of weight of clothes in Washing Machine based on Voltage, Current and Speed Features

### - Abhijeet Pal, Adithya Thonse, Tushar Sharma, Fasna S, Jaswanth Jadda

## Overview :
Prediction of weight of clothes in washing machine eliminates the use of weight sensor which is required for automatic water level adjustment, reducing manufacturing costs. We use machine learning models using features such as voltage, current and speed, ensuring water level adjustment without any mechanical failure due to sensor. Provides a cost-effective and reliable sensor-less solution.

## About the Dataset :

The dataset consists of 6 current, voltage and speed based features. The dataset is obtained from TI Kilby Labs. We have dataset of 100g precision from 0g to 900g. Two different models are proposed and we obtain best RMSE of 25.78g using float model, and 31.23g using partially quantized model. In partial quantization, the input batch norm layer, first conv layer and the last linear layer are in float. Other layers of the model are int8.

The variables in the dataset are as follows :  

| Variable Name | Description |
|----------|----------|
| Vd |  Voltage component along d-axis   |
| Id  | Current component along d-axis |
| Vq | Voltage component along q-axis |
| Iq | Current component along q-axis |
| Iqref | Reference Current |
| Speed | Speed of the washing machine motor |

## Preparing the dataset : 

For regression tasks, the dataset structure is expected to be as follows :

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

* We need to have the annotations folder in regression task.
* The data directory is automatically named 'files' for regression tasks.

* The dataset can be directly downloaded from https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/washing_machine_loading_data.zip, it's directly downloaded in the example

## Usage in TinyML ModelMaker

The example can be run directly using the following command

```
./run_tinyml_modelmaker.sh  ./examples/reg_washing_machine/config.yaml

```
wherein 
1.  ``` run_tinyml_modelmaker.sh``` is the script to run modelmaker. It takes input of CONFIG_FILE
2. ``` ./examples/reg_washing_machine/config.yaml``` is the location of the CONFIG_FILE

We can change the training configurations, no. of features or variables for input regression (need to change dataset as well), feature extraction, frame size, etc. in the config file.

## Configuring the YAML file

The YAML file contains the configurations to define pipelines for dataset loading, feature extraction, testing and compilation. 

For regression tasks in the common section 
``` 
task_type: 'generic_timeseries_regression'
target_device: 'F28P55'
``` 

In the dataset section
```
dataset_name: 'washing_machine_load_weighing'
input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/washing_machine_loading_data.zip
```

* The data directory is automatically detected as 'files' for regression tasks

In the data_processing_feature_extraction section

``` 
data_proc_transforms: 'SimpleWindow'
frame_size: 1024
variables: 6
``` 

* In this regression task, feature extraction such as FFT, Binning, ABS, other features extraction methods have not shown to improve the RMSE of prediction. Hence only a SimpleWindow with appropriate frame size is used

* variables is 6 is the count of number of input features for prediction


In the training section

``` 
model_name: 'REGR_13k'
batch_size: 128
training_epochs: 500
num_gpus: 1
quantization: 0
``` 

* 'REGR_13k' is the model_name which consists of 12477 trainable parameters, REGR_4k can also be used, it contains 4093 trainable parameters 
* Batch size can be tuned, 128 is used in this example. 
* Quantization set to 0 is float float, and quantization set to 2 can be used for partially quantized model. 
* In regression currently the first input batch norm layer, first conv/linear layer and the last linear layer are not quantized i.e. kept in float rest all other layers are quantized to int8
* The Loss used for Training is Mean Squared Error (MSE) and metric used is RMSE
* It's recommended to run the model for at least 500 training epochs to obtain good RMSE in weight loading prediction
## Dataset Split
The dataset consists of 391 files, each file consists of 12300 datapoints on average of the washing machine running in constant torque mode (Mode 2)

Train : 276 files
Validation : 51 files
Test : 64 files

Train - Val - Test split of (70% : 14% : 16%)

## Results
RMSE values obtained for both the models and different frame sizes on the test set

| Frame Size | REGR_4k | REGR_13k | Partially Quantized REGR_4k | Partially Quantized REGR_13k |
|----------|-------------|-------------|-------------|-------------|
| 256 | 63.00 | 35.77 | 62.32 | 33.37 |
| 512 | 46.91 | 32.58 | 52.42 | 31.23 |
| 1024 | 59.01 | 25.78 | 64.59 | 36.78 |

The above table shows results after training the models for 500 epochs each

We observe that for the 13K parameter model, the RMSE reduces as the frame size is increased. This can be attributed to the larger context with which the model learns when frame size is large.

However, for the 4K model, as the number of parameters are too less, it is not able to learn larger context properly. Hence, it does not show much improvement as the frame size increases from 512 to 1024, but still it shows improvement when frame size is increased from 256 to 512.

Partially quantized models show comparable results compared to their float counterparts.

## Running on Device

It finally generates four files mod.a, tvmgen_default.h, test_vector.c, user_input_config.h 

mod.a, tvmgen_default.h can be found in
``` 
./tinyml-modelmaker/data/projects/<dataset_name>/run/{date-time}/{model_name}/compilation/artifacts/

``` 

test_vector.c and user_input_config.h can be found in 

``` 
./tinyml-modelmaker/data/projects/<dataset_name>/run/{date-time}/{model_name}/training/quantization/golden_vectors/

``` 

These four files can be used to run the model on the device as illustrated in running_model_on_device example