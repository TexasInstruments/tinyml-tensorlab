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

import yaml

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
