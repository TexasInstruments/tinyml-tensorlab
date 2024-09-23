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

import collections
import copy

import torch.nn as nn
import yaml

from . import tinynn


class GenericModelWithSpec(nn.Module):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config
        self.model_spec = None
        self.features = None
        self._init_args(config=config, **kwargs)

    def init_model_from_file(self, filename, variables=1, num_classes=2, **kwargs):
        self._read_model_spec(filename)
        self._init_model_from_spec(variables=variables, num_classes=num_classes, **kwargs)
        return self

    def write_model_to_file(self, filename):
        assert self.model_spec is not None, 'model_spec is invalid'
        self._write_model_spec(filename)
        return self

    def get_model_spec(self):
        assert self.model_spec is not None, 'model_spec is invalid'
        return self.model_spec

    def _init_args(self, config=None, **kwargs):
        if len(kwargs) > 0:
            config = config or self.config or dict()
            for key, value in kwargs.items():
                value_wtih_default = config.get(key, value)
                setattr(self, key, value_wtih_default)

    def _init_model_from_spec(self, model_spec=None, variables=None, input_features=None,
                              num_classes=None, with_input_batchnorm=None):
        assert input_features is not None, f'argument input_features must be provided with a valid value. got {input_features}'
        model_spec = model_spec or self.model_spec
        model_spec = copy.deepcopy(model_spec)
        layers_spec = model_spec['model_spec'] if isinstance(model_spec, dict) and 'model_spec' in model_spec else model_spec
        if isinstance(layers_spec, list):
            layers_spec = {str(k): v for k, v in enumerate(layers_spec)}
        #

        # update model_spec based on input/output channels
        first_key = list(layers_spec.keys())[0]
        first_layer_spec = layers_spec[first_key]
        if first_layer_spec['type'] == 'BatchNormLayer' or with_input_batchnorm is True:
            layers_spec[first_key] = dict(type='BatchNormLayer', num_features=variables)
        elif first_layer_spec['type'] == 'IdentityLayer' or with_input_batchnorm is False:
            layers_spec[first_key] = dict(type='IdentityLayer')

        second_key = list(layers_spec.keys())[1]
        seocond_layer_spec = layers_spec[second_key]
        if seocond_layer_spec['type'] in ('ConvLayer', 'ConvBNReLULayer'):
            seocond_layer_spec['in_channels'] = variables

        last_key = list(layers_spec.keys())[-1]
        last_layer_spec = layers_spec[last_key]
        if last_layer_spec['type'] == 'LinearLayer':
            last_layer_spec['out_features'] = num_classes

        # instantiate the model
        layers = collections.OrderedDict()
        input_tensor_size = [1, variables, input_features, 1]
        for layer_id, layer_spec in layers_spec.items():
            layer_type_str = layer_spec.pop('type')
            layer_type_str = layer_type_str.split('.')[-1]
            # handle special case
            if layer_type_str == 'LinearLayer' and layer_spec['in_features'] is None:
                layer_spec['in_features'] = input_tensor_size[-1]
            #
            layer_type = getattr(tinynn, layer_type_str)
            layer, output_tensor_size = layer_type(**layer_spec, input_tensor_size=input_tensor_size)
            layers[layer_id] = layer
            input_tensor_size = output_tensor_size

        self.features = nn.Sequential(layers)
        return self

    def _read_model_spec(self, filename):
        with open(filename) as fp:
            self.model_spec = yaml.safe_load(fp)
        #
        return self.model_spec

    def _write_model_spec(self, filename, model_spec=None):
        model_spec = model_spec or self.model_spec
        with open(filename, 'w') as fp:
            yaml.safe_dump(model_spec, fp)
        #
        return True

    def forward(self, x):
        x = self.features(x)
        return x


# Usage:
# GenericModelWithSpec().init_model_from_file('model.yaml')
