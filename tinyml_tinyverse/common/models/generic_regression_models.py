from ..utils import py_utils
from .generic_model_spec import GenericModelWithSpec


class REG_TS_GEN_BASE_3K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=1, num_outputs=1, with_input_batchnorm=True):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_outputs=num_outputs)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_outputs,
                                   with_input_batchnorm=self.with_input_batchnorm)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0' :dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict(type='IdentityLayer')}
        layers += {'1' :dict(type='ReshapeLayer', ndim=2)}
        layers += {'2' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs *32)}
        layers += {'3' :dict(type='ReluLayer')}
        layers += {'4' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs *16)}
        layers += {'5' :dict(type='ReluLayer')}
        layers += {'6' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs *4)}
        layers += {'7' :dict(type='ReluLayer')}
        layers += {'8' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs)}
        model_spec = dict(model_spec=layers)
        return model_spec

class REG_TS_GEN_BASE_10K(GenericModelWithSpec):
    def __init__(self, config, input_features=16, variables=4, num_outputs=1, with_input_batchnorm=True, kernel=3):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_outputs=num_outputs,
                         kernel=kernel)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_outputs,
                                   with_input_batchnorm=self.with_input_batchnorm)
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
        layers += {'8' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs * 64)}
        layers += {'9' :dict(type='ReluLayer')}
        layers += {'10' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs)}
        model_spec = dict(model_spec=layers)
        return model_spec
    
class REG_TS_CNN_13K(GenericModelWithSpec):
    def __init__(self, config, input_features=6, kernel=3, num_outputs=1):
        super().__init__(config, input_features=input_features, kernel=kernel, num_outputs=num_outputs)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                    input_features=self.input_features, num_classes=self.num_outputs,
                                    with_input_batchnorm=False)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0' :dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=16, kernel_size=(self.kernel,1), stride=(2,1), padding = (1,0))}
        layers += {'2':dict(type='MaxPoolLayer', kernel_size=(3,1))}
        layers += {'3':dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(self.kernel,1), stride=(2,1), padding = (1,0))}
        layers += {'4':dict(type='MaxPoolLayer',kernel_size=(3,1))}
        layers += {'5':dict(type='ConvBNReLULayer', in_channels=32, out_channels=64, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'6':dict(type='AdaptiveAvgPoolLayer', output_size=(2,1))}
        layers += {'7' :dict(type='ReshapeLayer', ndim=2)}
        layers += {'8' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs * 32)}
        layers += {'9' :dict(type='ReluLayer')}
        layers += {'10' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs)}
        model_spec = dict(model_spec=layers)
        return model_spec
    
class REG_TS_CNN_4K(GenericModelWithSpec):
    def __init__(self, config, input_features=6, kernel=3, num_outputs=1):
        super().__init__(config, input_features=input_features, kernel=kernel, num_outputs=num_outputs)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                    input_features=self.input_features, num_classes=self.num_outputs,
                                    with_input_batchnorm=False)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0' :dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=16, kernel_size=(self.kernel,1), stride=(2,1), padding = (1,0))}
        layers += {'2':dict(type='MaxPoolLayer', kernel_size=(3,1))}
        layers += {'3':dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(self.kernel,1), stride=(2,1), padding = (1,0))}
        layers += {'4':dict(type='AdaptiveAvgPoolLayer', output_size=(2,1))}
        layers += {'5' :dict(type='ReshapeLayer', ndim=2)}
        layers += {'6' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs * 32)}
        layers += {'7' :dict(type='ReluLayer')}
        layers += {'8' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs)}
        model_spec = dict(model_spec=layers)
        return model_spec
    