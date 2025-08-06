import torch
from ..utils import py_utils
from .generic_model_spec import GenericModelWithSpec

class FC_CNN_TS_GEN_BASE_13K(GenericModelWithSpec):
    def __init__(self, config, input_features=1, variables=1, num_classes=1, with_input_batchnorm=True):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes)
        # Generate and initialize the model specification
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec,
                                   variables=self.variables,
                                   input_features=self.input_features,
                                   num_classes=self.num_classes,
                                   with_input_batchnorm=self.with_input_batchnorm)
    
    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict(type='IdentityLayer')}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=32, kernel_size=(3,1), padding=(1,0), stride=(1,1))}
        layers += {'2':dict(type='ConvBNReLULayer', in_channels=32, out_channels=64, kernel_size=(3,1), padding=(1,0), stride=(1,1))}
        layers += {'3':dict(type='ReshapeLayer', ndim=2)}
        layers += {'4':dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}

        model_spec = dict(model_spec=layers)
        return model_spec