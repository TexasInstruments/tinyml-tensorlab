import torch
import numpy as np
from ..utils import py_utils
from .generic_model_spec import GenericModelWithSpec


class FEModel1(GenericModelWithSpec):
    def __init__(self, variables, out_features, output_channels=32, with_input_batchnorm=True):
        super(FEModel1, self).__init__(variables=variables, out_features=out_features, output_channels=output_channels, with_input_batchnorm=with_input_batchnorm)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   out_features=self.out_features, output_channels=self.output_channels,
                                   with_input_batchnorm=self.with_input_batchnorm)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0' :dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict
            (type='IdentityLayer')}
        layers += {'1' :dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=self.output_channels, kernel_size=(5 ,1), stride=(1 ,1))}
        layers += \
            {'1a': dict(type='ConvBNReLULayer', in_channels=self.output_channels, out_channels=self.output_channels,
                       kernel_size=(5, 1), stride=(1, 1))}
        layers += {'2': dict(type='MaxPoolLayer', kernel_size=(3, 1), stride=(2, 1))}
        layers += {'3': dict(type='ConvBNReLULayer', in_channels=self.output_channels, out_channels=self.variables,
                             kernel_size=(3, 1), stride=(1, 1))}
        layers += {'5': dict(type='AdaptiveAvgPoolLayer', output_size=(self.out_features, 1))}
        model_spec = dict(model_spec=layers)
        return model_spec


class FEModel2(torch.nn.Module):
    def __init__(self, variables, out_features, output_channels=32):
        super(FEModel2, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=variables, out_channels=output_channels, kernel_size=(3, 1), padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=(3, 1),
                                     padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=output_channels, out_channels=variables, kernel_size=(3, 1), padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((out_features, 1))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x[..., None])
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
    # def __init__(self, input_features=512, variables=1, out_features=32, with_input_batchnorm=True):
    #     super(FEModel, self).__init__()
    #     self.conv1 = nn.Conv2d(in_channels=variables, out_channels=16, kernel_size=(3,1), stride=(1,1), padding=1)
    #     self.conv2 = nn.Conv2d(in_channels=16, out_channels=variables, kernel_size=(3,1), stride=(1,1), padding=1)
    #     self.relu = nn.ReLU()
    #
    # def forward(self, x):
    #     x = self.relu(self.conv1(x[..., None]))
    #     x = self.relu(self.conv2(x))
    #     return x


def get_conv_bn_relu(in_channels: int, out_channels: int, kernel_size: tuple[int], padding=None, stride=1,
                     with_relu=True):
    # calculate the padding according to kernel if not provided
    padding = padding or (kernel_size[0] // 2, kernel_size[1] // 2)
    layers = []
    # perform conv, bn and relu on the input with in_channels and output of out_channels
    layers += [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)]
    if with_relu:
        layers += [torch.nn.BatchNorm2d(num_features=out_channels)]
        layers += [torch.nn.ReLU()]
    return layers


class FEModel(torch.nn.Module):
    def __init__(self, variables, out_features, output_channels=32, with_input_batchnorm=True):
        super(FEModel, self).__init__()
        self.variables = variables
        self.out_features = out_features
        layers = []
        layers += [torch.nn.BatchNorm2d(num_features=self.variables) if with_input_batchnorm else torch.nn.Identity()]
        layers += get_conv_bn_relu(variables, 32, kernel_size=(5, 1), padding=None, stride=(2, 1))
        layers += get_conv_bn_relu(32, 64, kernel_size=(5, 1), padding=None, stride=(2, 1))
        layers += get_conv_bn_relu(64, 128, kernel_size=(5, 1), padding=None, stride=(2, 1))
        layers += get_conv_bn_relu(128, out_features, kernel_size=(5, 1), padding=None, stride=(2, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x[..., None])
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = x.view((-1, self.variables, self.out_features, 1))
        return x


class CombinedModel(torch.nn.Module):
    def __init__(self, model1, model2):
        super(CombinedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return x


class FEModelLinear(torch.nn.Module):
    def __init__(self, variables, in_features, out_features, output_channels=32, with_input_batchnorm=False):
        super(FEModelLinear, self).__init__()
        self.variables = variables
        self.out_features = out_features
        self.in_features = in_features
        self.adaptive_avg_pool2D = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        layers = []
        start_channel = 16
        iterations = int(np.log2(out_features // start_channel))

        layers += get_conv_bn_relu(self.variables, start_channel * self.variables, kernel_size=(5, 1), padding=None,
                                   stride=(2, 1))
        in_ch = start_channel

        for iteration in range(iterations):
            out_ch = start_channel * (2 ** (iteration + 1))
            layers += get_conv_bn_relu(in_ch * self.variables, out_ch * self.variables, kernel_size=(5, 1),
                                       padding=None, stride=(2, 1))
            in_ch = out_ch

        layers += get_conv_bn_relu(out_ch * self.variables, out_ch * self.variables, kernel_size=(5, 1), padding=None,
                                   stride=(2, 1), with_relu=False)
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x[..., None]
        batch_size = x.shape[0]
        for layer in self.layers:
            x = layer(x)
        x = self.adaptive_avg_pool2D(x)
        x = x.view((batch_size, self.variables, self.out_features, 1))
        return x


class NeuralNetworkWithPreprocess(torch.nn.Module):
    def __init__(self, preprocess, model):
        super().__init__()
        self.preprocess = preprocess
        self.model = model

    def forward(self, x):
        if self.preprocess:
            x = self.preprocess(x)
            if self.model:
                x = x.detach()
                for p in self.preprocess.parameters():
                    p.requires_grad_ = False
                #
                for m in self.preprocess.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()
                    #
                #
            #
        #
        if self.model:
            x = self.model(x)
        #
        return x

