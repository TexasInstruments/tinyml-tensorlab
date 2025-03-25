import torch

from ..utils import py_utils
from .generic_model_spec import GenericModelWithSpec

def get_conv_bn_relu(in_channels: int, out_channels: int, kernel_size, padding=None, stride=1):
        # calculate the padding according to kernel if not provided
        padding = padding or (kernel_size[0]//2, kernel_size[1]//2)
        layers = []
        # perform conv, bn and relu on the input with in_channels and output of out_channels
        layers += [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, padding=padding, stride=stride)]
        layers += [torch.nn.BatchNorm2d(num_features=out_channels)]
        layers += [torch.nn.ReLU()]
        return layers

class CNN_TS_GEN_BASE_1K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=1, num_classes=2, with_input_batchnorm=True):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes,
                                   with_input_batchnorm=self.with_input_batchnorm)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0':dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict(type='IdentityLayer')}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(5,1), stride=(1,1))}
        layers += {'1a':dict(type='ConvBNReLULayer', in_channels=8, out_channels=8, kernel_size=(5,1), stride=(1,1))}
        layers += {'2': dict(type='MaxPoolLayer', kernel_size=(3, 1), stride=(2, 1))}
        layers += {'3':dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(3,1), stride=(1,1))}
        layers += {'5':dict(type='AdaptiveAvgPoolLayer', output_size=(4,1))}
        layers += {'6':dict(type='ReshapeLayer', ndim=2)}
        layers += {'7':dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec

class CNN_TS_GEN_BASE_2K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=1, num_classes=2, with_input_batchnorm=True):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes,
                                   with_input_batchnorm=self.with_input_batchnorm)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0':dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict(type='IdentityLayer')}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=16, kernel_size=(5,1), stride=(1,1))}
        layers += {'2':dict(type='ConvBNReLULayer', in_channels=16, out_channels=16, kernel_size=(3,1), stride=(1,1))}
        layers += {'3': dict(type='MaxPoolLayer', kernel_size=(3, 1), stride=(2, 1))}
        layers += {'4':dict(type='ConvBNReLULayer', in_channels=16, out_channels=16, kernel_size=(5,1), stride=(2,1))}
        layers += {'5':dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(5,1), stride=(2,1))}
        layers += {'6':dict(type='AdaptiveAvgPoolLayer', output_size=(4,1))}
        layers += {'7':dict(type='ReshapeLayer', ndim=2)}
        layers += {'8':dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec

class CNN_TS_GEN_BASE_4K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=1, num_classes=2, with_input_batchnorm=True):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes,
                                   with_input_batchnorm=self.with_input_batchnorm)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0':dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict(type='IdentityLayer')}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(7,1), stride=(2,1))}
        layers += {'1p':dict(type='MaxPoolLayer', kernel_size=(3,1), stride=(2,1))}
        layers += {'2':dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(5,1), stride=(2,1))}
        layers += {'3':dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(5,1), stride=(2,1))}
        layers += {'4':dict(type='AdaptiveAvgPoolLayer', output_size=(4,1))}
        layers += {'5':dict(type='ReshapeLayer', ndim=2)}
        layers += {'6':dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec


class CNN_TS_GEN_BASE_6K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=1, num_classes=2, with_input_batchnorm=True):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes,
                                   with_input_batchnorm=self.with_input_batchnorm)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0':dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict(type='IdentityLayer')}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=16, kernel_size=(5,1), stride=(1,1))}
        layers += {'2':dict(type='ConvBNReLULayer', in_channels=16, out_channels=16, kernel_size=(3,1), stride=(1,1))}
        layers += {'3': dict(type='MaxPoolLayer', kernel_size=(3, 1), stride=(2, 1))}
        layers += {'4':dict(type='ConvBNReLULayer', in_channels=16, out_channels=16, kernel_size=(5,1), stride=(2,1))}
        layers += {'5':dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(5,1), stride=(2,1))}
        layers += {'6':dict(type='AdaptiveAvgPoolLayer', output_size=(4,1))}
        layers += {'7':dict(type='ReshapeLayer', ndim=2)}
        layers += {'8':dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec


class CNN_TS_GEN_BASE_13K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=1, num_classes=2, with_input_batchnorm=True):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes,
                                   with_input_batchnorm=self.with_input_batchnorm)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0':dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict(type='IdentityLayer')}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(7,1), stride=(2,1))}
        #layers += {'1p':dict(type='MaxPoolLayer', kernel_size=(3,1), stride=(2,1))}
        layers += {'2':dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(3,1), stride=(2,1))}
        layers += {'3':dict(type='ConvBNReLULayer', in_channels=16, out_channels=16, kernel_size=(3,1), stride=(1,1))}
        layers += {'4':dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(3,1), stride=(2,1))}
        layers += {'5':dict(type='ConvBNReLULayer', in_channels=32, out_channels=32, kernel_size=(3,1), stride=(1,1))}
        layers += {'6':dict(type='ConvBNReLULayer', in_channels=32, out_channels=64, kernel_size=(3,1), stride=(2,1))}
        layers += {'7':dict(type='AdaptiveAvgPoolLayer', output_size=(4,1))}
        layers += {'8':dict(type='ReshapeLayer', ndim=2)}
        layers += {'9':dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec


class RES_CNN_TS_GEN_BASE_3K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=3, num_classes=4, with_input_batchnorm=True):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes)

        top_layer = torch.nn.BatchNorm2d(num_features=self.variables) if self.with_input_batchnorm else torch.nn.Identity()
        self.top_layer = top_layer

        layer1 = []
        layer1 += [torch.nn.Conv2d(in_channels=self.variables, out_channels=self.out_channel_layer1, kernel_size=(3, 1), stride=(2, 1))]
        layer1 += [torch.nn.BatchNorm2d(self.out_channel_layer1)]
        layer1 += [torch.nn.ReLU()]

        layer1 += [torch.nn.Conv2d(in_channels=self.out_channel_layer1, out_channels=self.out_channel_layer2, kernel_size=(3, 1), stride=(2, 1))]
        layer1 += [torch.nn.BatchNorm2d(self.out_channel_layer2)]
        layer1 += [torch.nn.ReLU()]

        layer1 += [torch.nn.Conv2d(in_channels=self.out_channel_layer2, out_channels=self.out_channel_layer3, kernel_size=(3, 1), stride=(2, 1))]
        layer1 += [torch.nn.BatchNorm2d(self.out_channel_layer3)]
        layer1 += [torch.nn.ReLU()]
        self.layer1 = torch.nn.Sequential(*layer1)

        layer2 = []
        layer2 += [torch.nn.Conv2d(in_channels=self.variables, out_channels=self.out_channel_layer3, kernel_size=(3, 1), stride=(2, 1))]
        layer2 += [torch.nn.BatchNorm2d(self.out_channel_layer3)]
        layer2 += [torch.nn.ReLU()]
        self.layer2 = torch.nn.Sequential(*layer2)

        bottom_layers = []
        bottom_layers += [torch.nn.AdaptiveAvgPool2d((3, 1))]
        bottom_layers += [torch.nn.Flatten()]
        bottom_layers += [torch.nn.Linear(in_features=self.out_channel_layer3*3, out_features=self.num_classes)]
        self.bottom_layers = torch.nn.Sequential(*bottom_layers)

    def forward(self, x):
        x = self.top_layer(x)
        res = x
        for layer in self.layer1:
            x = layer(x)
        for layer in self.layer2:
            res = layer(res)
        x = torch.cat((x, res), dim=2)
        for layer in self.bottom_layers:
            x = layer(x)
        return x


class RES_SLICE_CNN_TS_GEN_BASE_3K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=6, num_classes=2, with_input_batchnorm=True, slice_ratio=0.5):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes, 
                         slice_ratio=0.5)
        
        hidden_channels = [8, 16]

        layer1 = [torch.nn.BatchNorm2d(num_features=self.variables) if self.with_input_batchnorm else torch.nn.Identity()]
        layer1 += get_conv_bn_relu(self.variables, hidden_channels[0], (3, 1), stride=(2, 1))
        layer1 += get_conv_bn_relu(hidden_channels[0], hidden_channels[1], (3, 1), stride=(2, 1))
        layer1 += [torch.nn.AdaptiveAvgPool2d((self.variables, 1))]
        self.layer1 = torch.nn.Sequential(*layer1)

        layer2 = [torch.nn.BatchNorm2d(num_features=self.variables) if self.with_input_batchnorm else torch.nn.Identity()]
        layer2 += get_conv_bn_relu(self.variables, hidden_channels[0], (3, 1), stride=(2, 1))
        layer2 += get_conv_bn_relu(hidden_channels[0], hidden_channels[1], (3, 1), stride=(2, 1))
        layer2 += [torch.nn.AdaptiveAvgPool2d((self.variables, 1))]
        self.layer2 = torch.nn.Sequential(*layer2)

        bottom_layers = []
        bottom_layers += [torch.nn.Flatten()]
        bottom_layers += [torch.nn.Linear(in_features=hidden_channels[1]*self.variables, out_features=self.num_classes)]
        self.bottom_layers = torch.nn.Sequential(*bottom_layers)

    def forward(self, x):
        x1 = x[:,:,:int(self.slice_ratio*self.input_features),:]
        x2 = x[:,:,int(self.slice_ratio*self.input_features):,:]
        for layer in self.layer1:
            x1 = layer(x1)
        for layer in self.layer2:
            x2 = layer(x2)
        x = x1 + x2
        for layer in self.bottom_layers:
            x = layer(x)
        return x