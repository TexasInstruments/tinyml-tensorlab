from ..utils import py_utils
from .generic_model_spec import GenericModelWithSpec


class REG_TS_GEN_BASE_3K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=1, num_classes=2, with_input_batchnorm=True):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes,
                                   with_input_batchnorm=self.with_input_batchnorm)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0' :dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict(type='IdentityLayer')}
        layers += {'1' :dict(type='ReshapeLayer', ndim=2)}
        # layers += {'1p':dict(type='MaxPoolLayer', kernel_size=(3,1), stride=(2,1))}
        layers += {'2' :dict(type='LinearLayer', in_features=None, out_features=self.num_classes *32)}
        layers += {'3' :dict(type='ReluLayer')}
        layers += {'4' :dict(type='LinearLayer', in_features=None, out_features=self.num_classes *16)}
        layers += {'5' :dict(type='ReluLayer')}
        layers += {'6' :dict(type='LinearLayer', in_features=None, out_features=self.num_classes *4)}
        layers += {'7' :dict(type='ReluLayer')}
        layers += {'8' :dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec

class REG_TS_GEN_BASE_10K(GenericModelWithSpec):
    def __init__(self, config, input_features=16, variables=4, num_classes=2, with_input_batchnorm=True, kernel=3):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes,
                         kernel=kernel)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes,
                                   with_input_batchnorm=self.with_input_batchnorm)
        # define the method to feed the inputs to the model
    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0' :dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict(type='IdentityLayer')}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'2':dict(type='MaxPoolLayer', kernel_size=(1,1), stride=(1,1))}
        layers += {'3':dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'4':dict(type='MaxPoolLayer', kernel_size=(1,1), stride=(1,1))}
        layers += {'5':dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'6':dict(type='AdaptiveAvgPoolLayer', output_size=(4,1))}
        layers += {'7' :dict(type='ReshapeLayer', ndim=2)}
        layers += {'8' :dict(type='LinearLayer', in_features=None, out_features=self.num_classes * 64)}
        layers += {'9' :dict(type='ReluLayer')}
        layers += {'910' :dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec
    
