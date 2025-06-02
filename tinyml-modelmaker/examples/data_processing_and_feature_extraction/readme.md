# Motor Fault Classification Time Series Dataset : Focus on Data Processing and Feature Extraction 
### -Fasna Sharaf, Adithya Thonse, Tushar Sharma
<hr>

## Overview

The Motor Fault Classification Dataset is a multivariate time series dataset designed for bearing fault detection and classification in industrial motors based on extracted features from the dataset. This has been collected on an internal TI Testbed.

The dataset consists of measurements from a vibration sensor along _3 axes_ domains and is categorized into **six** fault classes. Each class represents a specific bearing condition, ranging from normal operation to various fault types. The classes are:

1. **Normal**<br>
2. **Contaminated**<br>
3. **Erosion**<br>
4. **Flaking**<br>
5. **No Lubrication**<br>
6. **Localized Fault**<br>

This dataset is already formatted according to TinyML Modelmaker needs. The link to the dataset can be found [here](http://software-dl.ti.com/C2000/esd/mcu_ai/01_00_00/datasets/motor_fault_classification_dsk.zip).

## Usage in TinyML ModelMaker

This dataset is designed to work with TinyML ModelMaker, an end-to-end model development tool that provides dataset handling, model training, and compilation.

```bash
./run_tinyml_modelmaker.sh F28P55 examples/data_processing_and_feature_extraction/config_timeseries_classification_motor_fault_dsk.yaml
```

Users can configure the model pipeline using a YAML configuration file (like shown in the command above), where different stages (dataset loading, data processing and feature extraction, training, testing, and compilation) can be enabled or disabled based on requirements.

## Understanding the YAML Configuration File
The YAML configuration file sets up the model training process in TinyML ModelMaker. It has several sections:

- **Common Section**: Defines general settings like the module type, task type, target device, and run name.
- **Dataset Section**: Provides details about the dataset, including whether to load it, the dataset name, and the path to the data file.
- **Data Processing and Feature Extraction Section**: Involves preparing data for the model by applying techniques such as downsampling, windowing, and feature extraction methods like FFT (Fast Fourier Transform).
- **Training Section**: Configures the model training parameters, including the model name, batch size, epochs, and learning rate.
- **Testing Section**: Indicates if testing should be enabled.
- **Compilation Section**: Sets options for compiling the trained model.

<hr>

## In this example, we'll dive deeper into the **Data Processing and Feature Extraction** section to explain the specifics.
<hr>

## Data Processing

In this section, we'll cover how to configure data processing transforms available in the YAML file.

**Available Transforms:**

Whatever transform you want to use, cascade it in the `data_proc_transforms` list and set the parameters for that transform. Here are the available transforms:-

1. **Simple Window**: Creates windows of specified size from the dataset.
2. **Downsample**: Reduces the sampling rate of the dataset.
 
The `variables` parameter specify the number of input channels (or variables) in the dataset.

### Using Simple Window alone

To use the Simple Window transform, you'll need to understand two key parameters: **frame_size** and **stride_size**.

- **frame_size**: This parameter defines the size of each window/frame. It determines the number of data points included in each segment of the dataset.
- **stride_size**: This parameter defines the amount of overlap between consecutive windows. It is calculated as a fraction of the sequence window size.

Here is an example of how to use this:-

```yaml
data_processing_feature_extraction:
    data_proc_transforms: ['SimpleWindow']
    frame_size: 256
    stride_size: 0.01
    variables: 3
```

### Using Downsample alone

The Downsample transform reduces the sampling rate of the dataset. Here are the key parameters:

- **sampling_rate**: The original sampling rate of the dataset.
- **new_sr**: The new sampling rate after downsampling.

Here is an example of how to use this:-

```yaml
data_processing_feature_extraction:
    data_proc_transforms: ['DownSample']
    sampling_rate: 313000
    new_sr: 3130
    variables: 3
```


### Using Downsample and Simple Window Together:

Here is an example of how to use this:-

```yaml
data_processing_feature_extraction:
    data_proc_transforms: ['DownSample', 'SimpleWindow']
    # Downsample
    sampling_rate: 313000
    new_sr: 3130
    # SimpleWindow
    frame_size: 256
    stride_size: 0.01
    variables: 3
```

By following these configurations, you can customize the data processing steps in your model training pipeline. 

## Feature Extraction

In this section, we'll focus on the feature extraction configurations available in the YAML file. These configurations allow you to extract relevant features from the dataset for model training. Subpoint number 12 dominates the other configurations. But to get to that, let us first explore all of the other configurations available.

### Available Parameters:

1. **feature_extraction_name**: Name of the feature extraction configuration (If you want to avoid the complexity of specifying each parameter that we'll discuss below, you only have to set just this one parameter to one of the four predefined presets for motor fault dataset. We'll see more about these predefined presets later in this section. Default value: None).

2. **dc_remove**: Whether to remove the DC component from the signal (It can take True or False values. Default value: True).

3. **min_bin**: Minimum bin size for binning transformation.(Default value: 1)

4. **normalize_bin**: Whether to normalize the binned data (It takes 0 or 1 value. Default value: 0).

5. **frame_size**: Size of each frame to be processed.(Default value: 3009)

6. **feature_size_per_frame**: Number of features to extract from each frame.(Default value: None)

7. **num_frame_concat**: Number of frames to concatenate for each sample.(Default value: 1)

8. **frame_skip**: Number of frames to skip between processed frames.(Default value: 1)

9. **log_base**: Base of the logarithm for log scaling (Default value: None).

10. **log_mul**: Multiplicator for log scaling (Default value: None).

11. **log_threshold**: Threshold for log scaling to prevent negative infinity values (Default value: None).

12. **feat_ext_transform**: List of transformations to apply to the data (Default value: []).
      
      Available transforms are:-

      (i) **WINDOWING**: Splits the dataset into smaller segments or windows of size `frame_size`.

      (ii) **FFT_FE**: Fast Fourier Transform Feature Extraction (Converts time domain data to frequency domain data).

      (iii) **NORMALIZE**: Scales the data to a range relative to frame size.

      (iv) **DC_REMOVE**: Removes the DC component from the signal.
      
      (v) **BINNING**: Reduces the data size by grouping adjacent data points into bins and using the average value of each bin. It takes into account the `min_bin` and `normalize_bin` parameters.

      (vi) **RAW_FE**: Takes the mean of the wave within each frame. If `dc_remove` parameter is set to True, it also removes the DC component from the signal.

      (vii) **FFT_POS_HALF**: Selects DC component and the `min_bin` samples of the FFT.

      (viii) **ABS**: Takes the absolute value of the signal.

      (ix) **LOG_DB**: Converts the amplitude to a logarithmic scale in decibels. It takes into account `log_base`, `log_mul` and `log_threshold` parameters.

      (x) **CONCAT**: Concatenates the extracted features into a single vector. It takes into account the `num_frame_concat` parameter.

13. **store_feat_ext_data**: Whether to store the extracted features (It can take True or False values. (Default value: False)).

14. **feat_ext_store_dir**: Directory to store the extracted features (Default value: None).

15. **dont_train_just_feat_ext**: Whether to only extract features without training (It can take True or False values. (Default value: False)).

16. **analysis_bandwidth**: Bandwidth of the signal to analyze (Default value: 1).

17. **stacking**: Refers to the way the extracted features are organized. It can take two values: `1D` and `2D1`.

      - In `1D` stacking, the features are concatenated into a single sequence.
       Example, if you have features `[a1, a2, a3]` from channel 1, `[b1, b2, b3]` from channel 2, and `[c1, c2, c3]` from channel 3, the 1D stacked feature will be:
     ```[a1, a2, a3, b1, b2, b3, c1, c2, c3]```

      - In `2D1` stacking,features are arranged in a matrix.
        For example, using the same features `[a1, a2, a3]` from channel 1, `[b1, b2, b3]` from channel 2, and `[c1, c2, c3]` from channel 3, the 2D1 stacked feature will be:
      ```
      [
         [a1, a2, a3],
         [b1, b2, b3],
         [c1, c2, c3]
      ]
      ```
      (Default value: '2D1')

18. **offset**: Controls the overlap between consecutive frames of data (Default value: 0).

    - Without offset : Frames are created consecutively,with no overlap (step size of 1).

    - With offset : Frames overlap by adding a fractional step size 1/n where n is the offset parameter. For example, if offset is set to 2, the offset for each frame is 1/2, so frames overlap by 50%.

19. **scale**: Adjusts the amplitude of the extracted features to a specific range (Default value: None).

20. **variables**: Specify the number of input channels (or variables) in the dataset.

<br>
Here is an example of how to custom specify the parameters for feature extraction:

<br>

```yaml
data_processing_feature_extraction:
   feature_extraction_name: Custom_MotorFault
   frame_size: 512
   feature_size_per_frame: 64
   num_frame_concat: 2
   normalize_bin: 1
   stacking: 2D1
   feat_ext_transform: ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT']
   offset: 0
   scale: 1
   frame_skip: 1
   log_mul: 20
   log_base: 10
   log_threshold: 1e-100
   store_feat_ext_data: False
   variables: 3
```

<br>

Here is an example of how to apply both data processing and feature extraction:-

```yaml
data_processing_feature_extraction:

   # Data processing
   data_proc_transforms: ['SimpleWindow']
   frame_size: 512
   stride_size: 0.01
   # Feature extraction
   feature_extraction_name: Custom_MotorFault
   feature_size_per_frame: 64
   num_frame_concat: 2
   normalize_bin: 1
   stacking: 2D1
   feat_ext_transform: ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT']
   offset: 0
   scale: 1
   frame_skip: 1
   log_mul: 20
   log_base: 10
   log_threshold: 1e-100
   store_feat_ext_data: False
   variables: 3
```

## Predefined Feature Extraction Presets:

Instead of custom specifying these parameters above, you can select from one of the predefined feature extraction presets available to simplify the process. These presets are defined to move directly to feature extraction without applying data processing transforms like Simple Window and Downsampling.  Here are the available options for the motor fault dataset (training results for each preset are provided under each preset):

1. **MotorFault_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_1D**:

    Definition for this preset:-

    - **data_proc_transforms**: []
    - **feat_ext_transform**: ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB','CONCAT']
    - **frame_size**: 256
    - **feature_size_per_frame**: 16
    - **num_frame_concat**: 8
    - **normalize_bin**: True
    - **dc_remove**: True
    - **offset**: 0
    - **scale**: 1
    - **stacking**: '1D'
    - **frame_skip**: 1
    - **log_mul**: 20
    - **log_base**: 10
    - **log_threshold**: 1e-100
    - **sampling_rate**: 1
    - **variables**: 3

   Usage:-

   ```yaml
   data_processing_feature_extraction:
    feature_extraction_name: MotorFault_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_1D
    store_feat_ext_data: False
   ```
   &emsp;(An additional parameter along with this preset is set not to store feature extraction data.)

   **Training results:**

   &emsp; Best Epoch: 1

   &emsp; Accuracy: **99.995**

   &emsp; F1-Score: 1.000

   &emsp; AUC ROC Score: 1.000
 
<br>

2. **MotorFault_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1**:

   Definition for this preset:-

    - **data_proc_transforms**: []
    - **feat_ext_transform**: ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT']
    - **frame_size**: 256
    - **feature_size_per_frame**: 16
    - **num_frame_concat**: 8
    - **normalize_bin**: True
    - **dc_remove**: True
    - **offset**: 0
    - **scale**: 1
    - **stacking**: '2D1'
    - **frame_skip**: 1
    - **log_mul**: 20
    - **log_base**: 10
    - **log_threshold**: 1e-100
    - **sampling_rate**: 1
    - **variables**: 3

   Usage:-

   ```yaml
   data_processing_feature_extraction:
    feature_extraction_name: MotorFault_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1
    store_feat_ext_data: False
   ```

   **Training results:**

   &emsp;   Best Epoch: 0

   &emsp;   Accuracy: **100.000**

   &emsp;   F1-Score: 1.000
      
   &emsp;   AUC ROC Score:1.000

3. **MotorFault_256Input_FFT_128Feature_1Frame_3InputChannel_removeDC_2D1**:

   Definition for this preset:-

    - **data_proc_transforms**: []
    - **feat_ext_transform**: ['FFT_FE', 'FFT_POS_HALF', 'ABS', 'DC_REMOVE', 'LOG_DB', 'CONCAT']
    - **frame_size**: 256
    - **feature_size_per_frame**: 128
    - **num_frame_concat**: 1
    - **normalize_bin**: True
    - **dc_remove**: True
    - **offset**: 0
    - **scale**: 1
    - **stacking**: '2D1'
    - **frame_skip**: 1
    - **log_mul**: 20
    - **log_base**: 10
    - **log_threshold**: 1e-100
    - **sampling_rate**: 1
    - **variables**: 3

   Usage:-

   ```yaml
   data_processing_feature_extraction:
    feature_extraction_name: MotorFault_256Input_FFT_128Feature_1Frame_3InputChannel_removeDC_2D1
    store_feat_ext_data: False
   ```

   **Training results**

   &emsp;   Best Epoch: 9

   &emsp;   Accuracy: **97.939**

   &emsp;   F1-Score: 0.979

   &emsp;   AUC ROC Score: 0.979

4. **MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_2D1**:

   Definition for this preset:-

    - **data_proc_transforms**: []
    - **feat_ext_transform**: ['RAW_FE', 'CONCAT']
    - **frame_size**: 128
    - **feature_size_per_frame**: 128
    - **num_frame_concat**: 1
    - **normalize_bin**: True
    - **dc_remove**: True
    - **offset**: 0
    - **scale**: 1
    - **stacking**: '2D1'
    - **frame_skip**: 1
    - **sampling_rate**: 1
    - **variables**: 3

   Usage:-

   ```yaml
   data_processing_feature_extraction:
    feature_extraction_name: MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_2D1
    store_feat_ext_data: False
   ```

   **Training results:**

   &emsp;   Best Epoch: 6

   &emsp;   Accuracy: **92.299**

   &emsp;   F1-Score: 0.922

   &emsp;   AUC ROC Score: 0.922

<br>     

For this example, only training is enabled, while testing and compilation are set to False in the YAML file. By comparing the training results of each preset, we can see that the second preset,`MotorFault_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1`, gives the best accuracy. Hence, it is a very good feature extraction preset for this example.

<hr>
Update history:
[28th Feb 2025]: Compatible with v1.0 of Tiny ML Modelmaker
