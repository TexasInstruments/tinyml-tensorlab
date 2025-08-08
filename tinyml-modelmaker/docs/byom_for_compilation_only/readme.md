# Bring Your Own Model (BYOM) for Compilation

If you have your own AI Model Training Framework but you want to compile your .onnx/.tflite model for TI MCUs, then you can use Tiny ML Modelmaker for this purpose.

The only change is that the config.yaml that you use will have to look something like below:
```python
common:  # The common section can be plainly copied as it is
    target_module: 'timeseries'
    task_type: 'generic_timeseries_classification'     
    target_device: 'F28P55'
    run_name: '{date-time}/{model_name}'
dataset:
    enable: False  # Please note the 'False'. This is important to disable data loading which is not important
    dataset_name: abc  # Can be anything, used for directory name 
feature_extraction:  # Retain this information
    feature_extraction_name: None
training:
    enable: False # Please note the 'False'. This is important to disable data loading which is not important
    model_name: 'a'   # Can be anything, used for directory name 
compilation:
    enable: True
    model_path: "/x/y/z/model.onnx"  # Most important: Path of the model to be compiled
```
The last line of the above file is where you specify the model to be compiled.

Then, you can run the following using [config.yaml](config.yaml)
```
run_tinyml_modelmaker.sh  <target_device> <config_file>

Example:
run_tinyml_modelmaker.sh F28P55 config.yaml
```
