##  The Config File

This guide showcases the commonly used options under each section. This file should more or less be enough to guide the user to feeding options.
However, if you want to delve into the depth of options for each section, we suggest you check out: [All possible options](../tinyml_modelmaker/ai_modules/timeseries/params.py)

The [config_dsk.yaml](../examples/dc_arc_fault/config_dsk.yaml) or [config_dsi.yaml](../examples/dc_arc_fault/config_dsi.yaml) are examples that have combined sections of the following structure with a subset of each of the options.

### Section1: Common
These options are about the overall run
```python
common:
    target_module: 'timeseries'             # timeseries :Currently supported, more modules to come soon. 
    task_type: 'arc_fault'                  # arc_fault/motor_fault
    target_device: 'F28P55'                 # Optional, will be overwritten by whatever is given on the commandline during run_tinyml_modelmaker.sh <target_device> config.yaml
    run_name: '{date-time}/{model_name}'    # Run directory name is based on this, you can provide your own name, recommended to keep as is
```

### Section 2: Dataset
```python
dataset:                                    # Enable/disable dataset loading
    enable: True                            # True to enable dataset loading, else will directly start from training step
    dataset_name: dc_arc_fault_example_dsk     # You can give any name you want, for folder naming purpose
    input_data_path: 'http://uda0484689.dhcp.ti.com:8100/arc_fault_classification_dsk.zip'  # Can be a url/local folder location to a .zip file or a normal directory 
```

### Section 3: Data Processing and Feature Extraction
```python
data_processing_feature_extraction:
    data_proc_transforms: [ Downsample, SimpleWindow ]  # transforms: 'DownSample SimpleWindow'
    sampling_rate: 1                                    # Original Sample Rate [1]
    new_sr: 1                                           # New Sample Rate, New Sample Rate can be given [1]
    stride_size: 1                                      # Stride size i.e move by 'frame_size*stride_size' samples for the next window [1]
    frame_size: 512                                     # Window frame length [512]

    feat_ext_transform: []                # Use this as '[]' for Time Domain processing without any of the below options
    feature_extraction_name: FFT1024      # Presets supported: FFT1024 (Other presets will be supported soon)
    frame_size: 512                       # Total number of features extracted (256/384/512) [256]
    num_frame_concat: 1                   # Number of frames concatenated for feature extraction (1/4/6/16) [1]
    min_bin: 1                            # Minimum FFT bin number used for feature extraction (0-256) [1]
    feature_size_per_frame: 1             # Number of features extracted for each frame (1-8) [1]
```
### Section 5: Training
```python
training:  
    enable: True                            # Enable/disable training
    model_name: 'TimeSeries_Generic_13k_t'  # Check config_timeseries*.yaml files for more models
                                            # Extra models for devices other than F28P55: ArcFault_cnn_largest/ArcFault_cnn_200/ArcFault_cnn_300, ArcFault_cnn_700, MotorFault_base1, MotorFault_base2
    model_config: ''                        # Method to tweak the parameters of the chosen model. See ../scripts/tcresnet.yaml
    batch_size: 16384                       # Batch size to train
    training_epochs: 10                     # Number of epochs to run training (QAT training_epochs is run on a different logic)
    run_quant_train_only: True              # Directly run Quant Training instead of Float+QAT Training
    num_gpus: 1                             # 0- use CPU training. 1- Use GPU training (System needs to have GPU)
    variables: 1                            # Number of data columns (excluding time column) in the dataset 
```
### Section 6: Compilation
```python
compilation:  # To enable/disable compilation
    enable: True                            # Enable/disable compilation of onnx model-> device runnable libraries 
    preset_name: default_preset             # (least_memory_preset, best_performance_preset, default_preset) [default_preset]
```