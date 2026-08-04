#################################################################################
# Copyright (c) 2023-2026, Texas Instruments
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

"""Feature extraction models for time series data."""

import torch
import numpy as np

from ..utils import py_utils
from .base import GenericModelWithSpec


class FEModel1(GenericModelWithSpec):
    def __init__(self, variables, out_features, output_channels=32):
        super(FEModel1, self).__init__(variables=variables, out_features=out_features, output_channels=output_channels)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   out_features=self.out_features, output_channels=self.output_channels)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0' :dict(type='BatchNormLayer', num_features=self.variables)}
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


def get_conv_bn_relu(in_channels: int, out_channels: int, kernel_size: tuple, padding=None, stride=1,
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
    def __init__(self, variables, out_features, output_channels=32):
        super(FEModel, self).__init__()
        self.variables = variables
        self.out_features = out_features
        layers = []
        layers += [torch.nn.BatchNorm2d(num_features=self.variables)]
        layers += get_conv_bn_relu(variables, 32, kernel_size=(5, 1), padding=None, stride=(2, 1))
        layers += get_conv_bn_relu(32, 64, kernel_size=(5, 1), padding=None, stride=(2, 1))
        layers += get_conv_bn_relu(64, 128, kernel_size=(5, 1), padding=None, stride=(2, 1))
        layers += get_conv_bn_relu(128, out_features, kernel_size=(5, 1), padding=None, stride=(2, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x[..., None])
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1,1))
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
    def __init__(self, variables, in_features, out_features, output_channels=32):
        super(FEModelLinear, self).__init__()
        self.variables = variables
        self.out_features = out_features
        self.in_features = in_features
        self.adaptive_avg_pool2D = torch.nn.AdaptiveAvgPool2d(output_size=(1,1))

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
        # Only freeze preprocess when it's genuinely acting as a frozen feature
        # extractor ahead of a trainable model -- matches the original
        # conditional (the old code only detached/eval'd preprocess when
        # self.model was also set).
        self._freeze_preprocess = self.preprocess is not None and self.model is not None
        if self._freeze_preprocess:
            for p in self.preprocess.parameters():
                p.requires_grad = False  # was `p.requires_grad_ = False`, which
                # assigns over the bound method instead of calling it -- a no-op
                # that never actually disabled gradients.

    def train(self, mode=True):
        # preprocess must stay in eval mode permanently once frozen, regardless
        # of how many times the outer container's train()/eval() is toggled.
        # Without this override, the standard per-epoch model.train() call
        # recursively resets preprocess back to train mode too (nn.Module.train()
        # propagates to all child modules by default) -- silently corrupting its
        # BatchNorm running stats on the very next forward pass, since the old
        # code only called m.eval() *after* running preprocess once, and only
        # within forward(), so the fix couldn't survive the next train() call.
        super().train(mode)
        if self._freeze_preprocess:
            self.preprocess.eval()
        return self

    def forward(self, x):
        if self.preprocess:
            if self._freeze_preprocess:
                with torch.no_grad():
                    x = self.preprocess(x)
            else:
                x = self.preprocess(x)
        if self.model:
            x = self.model(x)
        return x


# Export all feature extraction models
# None of these classes are meant to be selected by name through the public
# model registry (get_model()) -- they're composition/helper classes with
# positional-arg constructors (e.g. FEModel1(variables, out_features, ...),
# CombinedModel(model1, model2)) that don't accept get_model()'s
# config=<dict> calling convention, unlike real registry models. The
# registry's auto-discovery loop (_register_models_from_module in
# models/__init__.py) previously registered every name in __all__
# regardless, so selecting one of these by --model/model_name (they appeared
# as legitimate entries in list_models()/model_dict.keys()) raised a raw
# TypeError: __init__() got an unexpected keyword argument 'config' instead
# of the intended "model not found" error. Deliberately empty: keeps this
# module's own import (and the registry loader's __all__-presence check)
# working, without the fallback "discover every model-shaped class" path
# re-registering these classes anyway. They remain directly importable
# (e.g. `from tinyml_modelzoo.models.feature_extraction import FEModel1`,
# which real callers already use) -- __all__ only affects `from module
# import *` and this registry's own discovery loop.
__all__ = []
