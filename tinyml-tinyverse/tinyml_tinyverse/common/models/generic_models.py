#################################################################################
# Copyright (c) 2023-2024, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#################################################################################

import numpy as np
import yaml
import torch

from ..utils import py_utils
from .generic_model_spec import GenericModelWithSpec

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
        layers += {'0':dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict(type='IdentityLayer')}
        layers += {'1':dict(type='ReshapeLayer', ndim=2)}
        #layers += {'1p':dict(type='MaxPoolLayer', kernel_size=(3,1), stride=(2,1))}
        layers += {'2':dict(type='LinearLayer', in_features=None, out_features=self.num_classes*32)}
        layers += {'3':dict(type='ReluLayer')}
        layers += {'4':dict(type='LinearLayer', in_features=None, out_features=self.num_classes*16)}
        layers += {'5':dict(type='ReluLayer')}
        layers += {'6':dict(type='LinearLayer', in_features=None, out_features=self.num_classes*4)}
        layers += {'7':dict(type='ReluLayer')}
        layers += {'8':dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec
    

class RES_CNN_TS_GEN_BASE_3K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=3, num_classes=4, with_input_batchnorm=True):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes)

        features = 8
        top_layer = torch.nn.BatchNorm2d(num_features=variables) if with_input_batchnorm else torch.nn.Identity()
        self.top_layer = top_layer

        layer1 = []
        layer1 += [torch.nn.Conv2d(in_channels=variables, out_channels=features, kernel_size=(3, 1), stride=(2, 1))]
        layer1 += [torch.nn.BatchNorm2d(features)]
        layer1 += [torch.nn.ReLU()]

        layer1 += [torch.nn.Conv2d(in_channels=features, out_channels=features*2, kernel_size=(3, 1), stride=(2, 1))]
        layer1 += [torch.nn.BatchNorm2d(features*2)]
        layer1 += [torch.nn.ReLU()]

        layer1 += [torch.nn.Conv2d(in_channels=features*2, out_channels=features*4, kernel_size=(3, 1), stride=(2, 1))]
        layer1 += [torch.nn.BatchNorm2d(features*4)]
        layer1 += [torch.nn.ReLU()]
        self.layer1 = torch.nn.Sequential(*layer1)

        layer2 = []
        layer2 += [torch.nn.Conv2d(in_channels=variables, out_channels=features*4, kernel_size=(3, 1), stride=(2, 1))]
        layer2 += [torch.nn.BatchNorm2d(features*4)]
        layer2 += [torch.nn.ReLU()]
        self.layer2 = torch.nn.Sequential(*layer2)

        bottom_layers = []
        bottom_layers += [torch.nn.AdaptiveAvgPool2d((3, 1))]
        bottom_layers += [torch.nn.Flatten()]
        bottom_layers += [torch.nn.Linear(in_features=32*3, out_features=num_classes)]
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


class FEModel1(GenericModelWithSpec):
    def __init__(self, variables, out_features, output_channels=32, with_input_batchnorm=True):
        super(FEModel1, self).__init__(variables=variables, out_features=out_features, output_channels=output_channels, with_input_batchnorm=with_input_batchnorm)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   out_features=self.out_features, output_channels=self.output_channels,
                                   with_input_batchnorm=self.with_input_batchnorm)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0':dict(type='BatchNormLayer', num_features=self.variables) if self.with_input_batchnorm else dict(type='IdentityLayer')}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=self.output_channels, kernel_size=(5,1), stride=(1,1))}
        layers += {'1a':dict(type='ConvBNReLULayer', in_channels=self.output_channels, out_channels=self.output_channels, kernel_size=(5,1), stride=(1,1))}
        layers += {'2': dict(type='MaxPoolLayer', kernel_size=(3, 1), stride=(2, 1))}
        layers += {'3':dict(type='ConvBNReLULayer', in_channels=self.output_channels, out_channels=self.variables, kernel_size=(3,1), stride=(1,1))}
        layers += {'5':dict(type='AdaptiveAvgPoolLayer', output_size=(self.out_features,1))}
        model_spec = dict(model_spec=layers)
        return model_spec


class FEModel2(torch.nn.Module):
    def __init__(self, variables, out_features, output_channels=32):
        super(FEModel2, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=variables, out_channels=output_channels, kernel_size=(3, 1), padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=(3, 1), padding=1)
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
def get_conv_bn_relu(in_channels: int, out_channels: int, kernel_size: tuple[int], padding=None, stride=1, with_relu=True):
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
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1,1))
        x = x.view((-1,self.variables,self.out_features,1))
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
        iterations = int(np.log2(out_features//start_channel))
        
        layers += get_conv_bn_relu(self.variables, start_channel*self.variables, kernel_size=(5, 1), padding=None, stride=(2, 1))
        in_ch = start_channel

        for iteration in range(iterations):
            out_ch = start_channel*(2**(iteration+1))
            layers += get_conv_bn_relu(in_ch*self.variables, out_ch*self.variables, kernel_size=(5, 1), padding=None, stride=(2, 1))
            in_ch = out_ch

        layers += get_conv_bn_relu(out_ch*self.variables, out_ch*self.variables, kernel_size=(5, 1), padding=None, stride=(2, 1), with_relu=False)
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

if __name__ == '__main__':
    yaml.Dumper.ignore_aliases = lambda *data: True
    filename = 'CNN_TS_GEN_BASE_13K.yaml'
    with open(filename, 'w') as fp:
        yaml.dump(CNN_TS_GEN_BASE_13K(dict()).model_spec, fp, default_flow_style=False, sort_keys=False)

    filename = 'CNN_TS_GEN_BASE_6K.yaml'
    with open(filename, 'w') as fp:
        yaml.dump(CNN_TS_GEN_BASE_6K(dict()).model_spec, fp, default_flow_style=False, sort_keys=False)

    filename = 'CNN_TS_GEN_BASE_4K.yaml'
    with open(filename, 'w') as fp:
        yaml.dump(CNN_TS_GEN_BASE_4K(dict()).model_spec, fp, default_flow_style=False, sort_keys=False)

    filename = 'CNN_TS_GEN_BASE_1K.yaml'
    with open(filename, 'w') as fp:
        yaml.dump(CNN_TS_GEN_BASE_1K(dict()).model_spec, fp, default_flow_style=False, sort_keys=False)
