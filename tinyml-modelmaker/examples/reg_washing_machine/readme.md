
# Washing Machine Weight Loading
## Prediction of weight of clothes in Washing Machine based on Voltage, Current and Speed Features

### - Abhijeet Pal, Adithya Thonse, Tushar Sharma, Fasna S, Jashwanth Jadda

## Overview :
Prediction of weight of clothes in washing machine eliminates the use of weight sensor which is required for automatic water level adjustment, reducing manufacturing costs. We use machine learning models using features such as voltage, current and speed, ensuring water level adjustment without any mechanical failure. Provides a cost-effective and reliable sensor-less solution.

## About the Dataset :

The dataset consists of 6 current, voltage and speed based features. The dataset is obtained from TI Kilby Labs in Torque Control Mode. We have dataset of 100g precision from 0g to 1000g. Two different models are proposed and we obtain best RMSE of 33.90g.

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
├── {data_dir}/
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

* The dataset can be directly downloaded from https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/washing_machine_load_dataset.zip, it's directly downloaded in the example

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
target_module : timeseries
task_type: 'generic_timeseries_regression'
target_device: 'F28P55'
``` 

In the dataset section
``` 
data_dir: 'files'
dataset_name: 'washing_machine'
input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/01_02_00/datasets/washing_machine_load_dataset.zip
``` 

* The downloaded dataset, has data_dir, i.e. location of the files as {files} folder

In the data_processing_feature_extraction section

``` 
feat_ext_transform: ['SimpleWindow']
frame_size: 256
feature_size_per_frame: 256
variables: 6
``` 

* In this regression task, feature extraction such as FFT, Binning, ABS, other features extraction methods have not shown to improve the RMSE of prediction. Hence only a SimpleWindow with appropriate frame size is used

* variables is 6 is the count of number of input features for prediction


In the training section

``` 
model_name: 'Reg_Washing_Machine_13K_t'
batch_size: 128
training_epochs: 500
num_gpus: 1
quantization: 0
``` 

* 'Reg_Washing_Machine_13K_t' is the model_name which consists of 12477 trainable parameters, Reg_Washing_Machine_4K_t can also be used, it contains 4093 trainable parameters 
* Batch size can be tuned, 128 is used in this example. 
* Quantization is set to 0, only float models are used for good accuracy
* The Loss used for Training is Mean Squared Error (MSE) and metric used is RMSE

## Results
RMSE values obtained for both the models and different frame sizes
| Frame Size | Reg_Washing_Machine_4k_t | Reg_Washing_Machine_13k_t |
|----------|-------------|-------------|
| 256 | 50.54 | 33.90 |
| 512 | 40.51 | 35.70 |
| 1024 | 43.19 | 37.87 |


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