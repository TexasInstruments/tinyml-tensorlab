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

class CNN_TS_GEN_BASE_100(GenericModelWithSpec):
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
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=4, kernel_size=(1,1), stride=(1,1))}
        layers += {'2':dict(type='ConvBNReLULayer', in_channels=4, out_channels=4, kernel_size=(3,1), stride=(1,1))}
        layers += {'3':dict(type='AdaptiveAvgPoolLayer', output_size=(1,1))}
        layers += {'4':dict(type='ReshapeLayer', ndim=2)}
        layers += {'5':dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec


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
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict(type='IdentityLayer')}
        # Early aggressive downsampling to reduce feature map size quickly
        layers += {'1': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=16, kernel_size=(3, 1), stride=(4, 1))}
        layers += {'2': dict(type='MaxPoolLayer', kernel_size=(3, 1), stride=(2, 1))}
        # Moderate channel expansion with depthwise separable convolution
        layers += {'3a': dict(type='ConvBNReLULayer', in_channels=16, out_channels=16, kernel_size=(3, 1), stride=(1, 1), groups=16)}
        layers += {'3b': dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(1, 1), stride=(1, 1))}
        # Additional downsampling
        layers += {'4': dict(type='MaxPoolLayer', kernel_size=(3, 1), stride=(2, 1))}
        # Feature extraction with more channels
        layers += {'5a': dict(type='ConvBNReLULayer', in_channels=32, out_channels=32, kernel_size=(3, 1), stride=(1, 1), groups=32)}
        layers += {'5b': dict(type='ConvBNReLULayer', in_channels=32, out_channels=48, kernel_size=(1, 1), stride=(1, 1))}
        # Final feature extraction
        layers += {'6': dict(type='ConvBNReLULayer', in_channels=48, out_channels=16, kernel_size=(5, 1), stride=(1, 1))}
        # Global pooling and classification
        layers += {'7': dict(type='AdaptiveAvgPoolLayer', output_size=(4, 1))}
        layers += {'8': dict(type='ReshapeLayer', ndim=2)}
        layers += {'9': dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}

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


class RES_CAT_CNN_TS_GEN_BASE_3K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=3, num_classes=4, with_input_batchnorm=True, out_channel_layer1=4, out_channel_layer2=8, out_channel_layer3=16):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes,
                         out_channel_layer1=out_channel_layer1, out_channel_layer2=out_channel_layer2, out_channel_layer3=out_channel_layer3)

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
        bottom_layers += [torch.nn.AdaptiveAvgPool2d((1, 1))]
        bottom_layers += [torch.nn.Flatten()]
        bottom_layers += [torch.nn.Linear(in_features=self.out_channel_layer3, out_features=self.num_classes)]
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


class RES_ADD_CNN_TS_GEN_BASE_3K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=3, num_classes=4, with_input_batchnorm=True, out_channel_layer1=4, out_channel_layer2=8, out_channel_layer3=16):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes,
                         out_channel_layer1=out_channel_layer1, out_channel_layer2=out_channel_layer2, out_channel_layer3=out_channel_layer3)

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
        layer1 += [torch.nn.AdaptiveAvgPool2d((1, 1))]
        self.layer1 = torch.nn.Sequential(*layer1)

        layer2 = []
        layer2 += [torch.nn.Conv2d(in_channels=self.variables, out_channels=self.out_channel_layer3, kernel_size=(3, 1), stride=(2, 1))]
        layer2 += [torch.nn.BatchNorm2d(self.out_channel_layer3)]
        layer2 += [torch.nn.ReLU()]
        layer2 += [torch.nn.AdaptiveAvgPool2d((1, 1))]
        self.layer2 = torch.nn.Sequential(*layer2)

        bottom_layers = []
        bottom_layers += [torch.nn.Flatten()]
        bottom_layers += [torch.nn.Linear(in_features=self.out_channel_layer3, out_features=self.num_classes)]
        self.bottom_layers = torch.nn.Sequential(*bottom_layers)

    def forward(self, x):
        x = self.top_layer(x)
        res = x
        for layer in self.layer1:
            x = layer(x)
        for layer in self.layer2:
            res = layer(res)
        x = x + res
        for layer in self.bottom_layers:
            x = layer(x)
        return x


class HAR_TINIE_CNN_2K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=3, num_classes=4, with_input_batchnorm=True):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes)
        
        layers = []
        in_ch = self.variables
        out_ch = 16
        
        bn1 = torch.nn.BatchNorm2d(num_features=self.variables) if self.with_input_batchnorm else torch.nn.Identity()
        
        cn1 = torch.nn.Conv2d(in_ch, out_ch, kernel_size=(5, 1))
        bn2 = torch.nn.BatchNorm2d(out_ch)
        relu = torch.nn.ReLU()

        cn2 = torch.nn.Conv2d(out_ch, out_ch, kernel_size=(5, 1))
        bn3 = torch.nn.BatchNorm2d(out_ch)
        relu2 = torch.nn.ReLU()
        
        ap = torch.nn.AdaptiveAvgPool2d((1, 1))
        fl = torch.nn.Flatten()
        ln = torch.nn.Linear(out_ch, self.num_classes)

        layers = [bn1, cn1, bn2, relu, cn2, bn3, relu2, ap, fl, ln]
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class YOLO_Classifier_8K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=3, num_classes=4, with_input_batchnorm=True, depth=3):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes,
                         depth=3)
        
        input_channels = self.variables
        output_channels = 16

        # Define your layers here
        def conv_bn_relu(in_ch, out_ch, with_pool=True):
            layers = []
            layers += [torch.nn.Conv2d(in_ch, out_ch, kernel_size=(3, 1), stride=(1, 1), padding=(1, 1))]
            layers += [torch.nn.BatchNorm2d(out_ch)]
            layers += [torch.nn.ReLU()]
            if with_pool:
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            return layers
        self.layers = [torch.nn.BatchNorm2d(num_features=self.variables) if self.with_input_batchnorm else torch.nn.Identity()]
        for _ in range(self.depth-2):
            layer = conv_bn_relu(input_channels, output_channels)
            self.layers += layer
            input_channels = output_channels
            output_channels = output_channels * 2
        
        l1 = conv_bn_relu(input_channels, output_channels, False)
        l2 = conv_bn_relu(output_channels, input_channels*4, False)
        gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        flat = torch.nn.Flatten()
        fc = torch.nn.Linear(input_channels*4, self.num_classes)

        self.layers += l1 + l2 + [gap, flat, fc]
        self.layers = torch.nn.Sequential(*self.layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

