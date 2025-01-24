## Easy to use model surgery utilities using torch.fx

This can be used to replace modules in a model that are derived from torch.nn.Modules. Most of the common classes used a model such as torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU are derived from torch.nn.Modules, so they can be easily replaced. If a model has used torch.relu operator directory or a x.relu (x being a tensor), those can also be replaced by this surgery utility. 
## APIs

### convert_to_lite_fx
This is the main model surgery function.
#### Usage: 
```
## insert this line into your model training script
model = convert_to_lite_fx(model)
### get_replacement_dict_default
This can be used to get the default replacement dict. This replacement dict can then be modified and passed to the convert api. 

#### Usage:
```
replacement_dict = copy.deepcopy(edgeai_torchmodelopt.xmodelopt.surgery.v2.get_replacement_dict_default())
replacement_dict.update({torch.nn.GELU: torch.nn.ReLU})
```

Now apply the conversion using the updated replacement_dict
```
model = convert_to_lite_fx(model, replacement_dict=replacement_dict)
```

The value of the replacement entry can also be a function name:
```
replacement_dict = get_replacement_dict_default()
replacement_dict.update({'my_layer':my_replacement_function})
```
