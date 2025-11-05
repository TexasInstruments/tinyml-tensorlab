# Foreseeing your model performance on F28P55x
### -Tushar Sharma, Adithya Thonse, Fasna S, Jaswanth Jadda, Abhijeet Pal
<hr>

## Overview

Tiny ML Modelmaker is a development tool that contains dataset handling, model training and compilation. Modelmaker will take the dataset, run the training scripts and compile the model using ti-mcu-nnc (TI MCU Neural Network Compiler). The compiled model can be used to work on TI based MCUs. Inference from models requires computation on device which amounts to cycles. Developers might want to reduce the time taken by model to as low as possible (reduce the cycles). But, reduction of computation can also result in lower accuracy from the model. 

This example talks about how user can foresee the cycles used by model in their application to speed up ML inference on MCUs. We will walkthrough the example by understanding CCS Studio, Modelmaker and using both to integrate the model on device.

In this example we will use the following:
- Device: LAUNCHXL-F28P55X
- C2000Ware 6.00.00.00
- CCS Studio 20.2.0


## CCS Studio

Code Composer Studio (CCS) is a free integrated development environment (IDE) provided by Texas Instruments (TI) for developing and debugging applications for TI's micro-controllers and processors. It offers various application examples for users to get started with their problem statement. This application example is not present as a part of CCS offering. We will see how to import the application **ex_model_performance_f28p55x** in CCS. We will use this example to run model on device.

### How to load sample application on CCS and flash on device

We will load the model performance example for f28p55x device using Code Composer Studio. The method will be different from example *running_model_on_device* as this example is not present in Resource Explorer. We will have to copy the application from this example to C2000Ware AI examples. Follow the steps below:

1. Open Code Composer Studio
2. Copy the application **ex_model_performance_f28p55x** given in this example 
3. Paste it at the location of C2000Ware AI examples[ti\C2000Ware_6_00_00_00\libraries\ai\feature_extract\c28\examples]
4. Go to File tab -> Select Import Projects(s)
![Import Project](assets/import_projects.png)
5. Find the folder **ex_model_performance_f28p55x** and click Select Folder
![Browse](assets/browse.png)
6. Click Finish to import the project

### Run the sample example

We will build the project and flash the program in device. The project has a 'C' file application_main.c, which contains the code for model inference and counting cycles. We will use debug mode to see the cycles of model inference. Based on your target device, you will have to set the target configuration as active. Here we are using LAUNCHXL-F28P55X.

7. Now we will build the project. Go to Project Tab -> Select Build Project(s)
![Build Project](assets/build.png)
8. Switch the active target device from **TMS320F28P550SJ9.ccxml** to **TMS320F28P550SJ9_LaunchPad.ccxml**.
![Active Target Configuration](assets/target_config.png)
9. Connect launchpad f28p55x to your system.
10. Flash the built project in device. Go to Run tab -> Select Flash Project
![Flash Application](assets/flash.png)
11. (Optional) If the update error comes, select 'Update'.
12. After the application is flashed, debug screen will appear. Select the debug icon.
![Debug Screen](assets/debug.png)
13. Let the example run by clicking the Continue button in Debug Window
![Example Run](assets/run.png)
14. You can see the cycles taken by the model in GEL Output window.
![Cycle Result](assets/result.png)

## ModelMaker

Modelmaker is configured using a [CONFIG_YAML](config.yaml) file. The config file controls the data processing, feature extraction, model configuration and model selection. In this example we will experiment with different input size (frame size) and deep learning models. To keep things simple we won't use any feature extraction transforms. 

In this example we will simply load dataset and pass it to the deep learning model. The focus is not to get the best accuracy but to understand the tradeoff between accuracy and model performance (cycles). We will sweep the frame size from [128, 1024] and use different TimeSeries_Generic_Models i.e. [TimeSeries_Generic_1k_t, TimeSeries_Generic_4k_t, TimeSeries_Generic_6k_t, TimeSeries_Generic_13k_t].

These can be configured in yaml in the following section
```yaml

data_processing_feature_extraction:
    feature_extraction_name: 'Custom_Default'
    data_proc_transforms: ['SimpleWindow']
    frame_size: 128   # Configure the frame size (input to the model (NCHW) -> (1,1,128,1))

training:
    enable: True #False
    model_name: 'TimeSeries_Generic_1k_t'    # Select the model
    
```

### Run model performance example

To run the modelmaker, go to tinyml-modelmaker folder. Open terminal and run the following commands

```sh
run_tinyml_modelmaker.sh docs/foreseeing_model_performance/config.yaml
```
This will run the default selected configuration of frame_size 128 and model TimeSeries_Generic_1k_t. To get a model with different input size (say 256), we will change the frame size to 256 and run the modelmaker with the same command as above.

You can see the output from running modelmaker in the following folder **tinyml-modelmaker/data/projects/model_performance/run/model_performance**.
Follow the below steps to deploy compiled model to CCS project and measure the cycles by running the application on device.

## Required files from Modelmaker to CCS Studio

The CCS example *ex_model_performance_f28p55x* requires 3 files from modelmaker. We will copy the files from modelmaker run to the CCS example project.

### Compiled model files

- mod.a: The compiled model is present in this file. 
  - Path Modelmaker: *tinyml-modelmaker/data/projects/model_performance/run/model_performance/compilation/artifacts/mod.a*
  - Path CCS Project: *ex_model_performance_f28p55x/artifacts/mod.a*
- tvmgen_default.h: Header file to access the model inference APIs from mod.a 
  - Path Modelmaker: *tinyml-modelmaker/data/projects/model_performance/run/model_performance/compilation/artifacts/tvmgen_default.h*
  - Path CCS Project: *ex_model_performance_f28p55x/artifacts/tvmgen_default.h*

### Model Description for device verification

- user_input_config.h: Configuration of model input size and output size. 
  - Path Modelmaker: *tinyml-modelmaker/data/projects/model_performance/run/model_performance/training/quantization/golden_vectors/user_input_config.h*
  - Path CCS Project: *ex_model_performance_f28p55x/user_input_config.h*

## Final Run

Run the modelmaker with command line. After the run is finished. Copy the 3 files (path present above) from Modelmaker to CCS Project. Build the CCS Project again, flash the program and start debugging the application. Check for the cycle count in GEL Output for different sets of frame size and models.

## Results

Verify the results from the table below. The input size is (NCHW) -> (1,1,frame size,1) and the output size is 2 for the arc fault dataset.

| Cycles Consumed (Accuracy) |         128        |         256         |         512       |         1024        |
|----------------------------|--------------------|---------------------|-------------------|---------------------|
| Timeseries_Generic_1k_t    |   103882 (80.32%)  |   188397 (85.83%)   |  372242 (87.33%)  |    692220 (94.43%)  |
| Timeseries_Generic_4k_t    |    71595 (86.45%)  |   109534 (89.29%)   |  184981 (91.79%)  |    326782 (94.85%)  |
| Timeseries_Generic_6k_t    |   107982 (89.97%)  |   164676 (90.05%)   |  261792 (92.62%)  |    462680 (95.68%)  |
| Timeseries_Generic_13k_t   |   199691 (91.83%)  |   312406 (92.60%)   |  535437 (93.21%)  |    985529 (95.54%)  |

<hr>
Update history:
[28th Aug 2025]: Compatible with v1.1 of Tiny ML Modelmaker