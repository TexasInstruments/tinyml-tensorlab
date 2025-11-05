import torch

from ..utils import py_utils
from .generic_model_spec import GenericModelWithSpec

class CNN_LENET5(GenericModelWithSpec):
    def __init__(self, config, input_features=(28,28), variables=1, num_classes=10, with_input_batchnorm=True):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes,
                                   with_input_batchnorm=self.with_input_batchnorm)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0':dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict(type='IdentityLayer')}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(0,0))}
        layers += {'2':dict(type='MaxPoolLayer', kernel_size=(2,2), stride=(2,2), padding=(0,0))}
        layers += {'3':dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(0,0))}
        layers += {'4':dict(type='MaxPoolLayer', kernel_size=(2,2), stride=(2,2), padding=(0,0))}
        layers += {'5':dict(type='ReshapeLayer', ndim=2)}
        layers += {'6' :dict(type='LinearLayer', in_features=400, out_features=self.num_classes *12)}
        layers += {'7' :dict(type='ReluLayer')}
        layers += {'8' :dict(type='LinearLayer', in_features=self.num_classes*12, out_features=84)}
        layers += {'9' :dict(type='ReluLayer')}
        layers += {'10':dict(type='LinearLayer', in_features=84, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec

