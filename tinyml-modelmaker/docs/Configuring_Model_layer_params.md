# Configure the model layers

* This is a guide on configuring the model's input arguments.
* Called the model_config, examples can be found inside the [misc](../misc) folder.
* To use this, you need to provide the path of the local `model_config` in the `config_*_*.yaml` under the `training` subsection.
* Example:
  * In [config_timeseries_classification_dsi.yaml](../config_timeseries_classification_dsi.yaml)
  * Under training section:
```yaml
training:
    enable: True
    model_name: 'TimeSeries_Generic_13k_t'
    model_config: '/home/a/b/tinyml-modelmaker/misc/TimeSeries_Generic_x_t.yaml'
```

* The contents of the yaml file needs to have the parameters that the model accepts as an input argument
* For example, the class definition of CNN_TS_GEN_BASE_13K (which is referred to by TimeSeries_Generic_13k_t) contains the following input arguments:
  * input_features, variables, num_classes, with_input_batchnorm 
```python
class CNN_TS_GEN_BASE_13K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=1, num_classes=2, with_input_batchnorm=True):
```

* So we can have (none or upto) the following arguments in `/home/a/b/tinyml-modelmaker/misc/TimeSeries_Generic_13k_t.yaml`
```yaml
input_features: 512
variables: 2
num_classes: 3
with_input_batchnorm: False
``` 

* This will edit the model properties accordingly
* Kindly see more examples in [misc](../misc) folder.
