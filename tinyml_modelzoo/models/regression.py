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

"""Regression models for time series data."""

from ..utils import py_utils
from .base import GenericModelWithSpec


class REG_TS_GEN_BASE_3K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=1, num_outputs=1):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_outputs=num_outputs)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_outputs)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0' :dict(type='BatchNormLayer', num_features=self.variables)}
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
    def __init__(self, config, input_features=16, variables=4, num_outputs=1, kernel=3):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_outputs=num_outputs,
                         kernel=kernel)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_outputs)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0' :dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'2':dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'3':dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'4':dict(type='AdaptiveAvgPoolLayer', output_size=(4,1))}
        layers += {'5' :dict(type='ReshapeLayer', ndim=2)}
        layers += {'6' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs * 64)}
        layers += {'7' :dict(type='ReluLayer')}
        layers += {'8' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs)}
        model_spec = dict(model_spec=layers)
        return model_spec


class REG_TS_GEN_BASE_1K(GenericModelWithSpec):
    def __init__(self, config, input_features=16, variables=4, num_outputs=1, kernel=3):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_outputs=num_outputs,
                         kernel=kernel)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_outputs)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0' :dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'2':dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'3':dict(type='AdaptiveAvgPoolLayer', output_size=(1,1))}
        layers += {'4' :dict(type='ReshapeLayer', ndim=2)}
        layers += {'5' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs * 8)}
        layers += {'6' :dict(type='ReluLayer')}
        layers += {'7' :dict(type='LinearLayer', in_features=None, out_features=self.num_outputs)}
        model_spec = dict(model_spec=layers)
        return model_spec


class REG_TS_GEN_BASE_13K(GenericModelWithSpec):
    def __init__(self, config, input_features=6, variables=4, kernel=3, num_outputs=1):
        super().__init__(config, input_features=input_features, variables=4, kernel=kernel, num_outputs=num_outputs)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                    input_features=self.input_features, num_classes=self.num_outputs)

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


class REG_TS_GEN_BASE_4K(GenericModelWithSpec):
    def __init__(self, config, input_features=6, variables=4, kernel=3, num_outputs=1):
        super().__init__(config, input_features=input_features, variables=4, kernel=kernel, num_outputs=num_outputs)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                    input_features=self.input_features, num_classes=self.num_outputs)

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


# NPU-Optimized Regression Models
# These models follow TI NPU constraints:
# - All channels are multiples of 4 (m4)
# - GCONV kernel heights <= 7
# - MaxPool kernels <= 4
# - FC input features >= 16

class REG_TS_GEN_BASE_500_NPU(GenericModelWithSpec):
    """NPU-optimized ~500 param regression model.

    Architecture: BN -> Conv(4ch) -> Conv(8ch) -> AvgPool -> FC
    NPU Compliance: m4 channels, kH<=7, FC input>=16
    """
    def __init__(self, config, input_features=16, variables=4, num_outputs=1, kernel=3):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_outputs=num_outputs, kernel=kernel)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_outputs)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=4, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'2': dict(type='ConvBNReLULayer', in_channels=4, out_channels=8, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'3': dict(type='AdaptiveAvgPoolLayer', output_size=(2,1))}
        layers += {'4': dict(type='ReshapeLayer', ndim=2)}
        layers += {'5': dict(type='LinearLayer', in_features=None, out_features=self.num_outputs)}
        model_spec = dict(model_spec=layers)
        return model_spec


class REG_TS_GEN_BASE_2K_NPU(GenericModelWithSpec):
    """NPU-optimized ~2K param regression model.

    Architecture: BN -> Conv(8ch) -> Conv(16ch) -> Conv(16ch) -> AvgPool -> FC
    NPU Compliance: m4 channels, kH<=7, FC input>=16
    """
    def __init__(self, config, input_features=16, variables=4, num_outputs=1, kernel=3):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_outputs=num_outputs, kernel=kernel)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_outputs)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'2': dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'3': dict(type='ConvBNReLULayer', in_channels=16, out_channels=16, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'4': dict(type='AdaptiveAvgPoolLayer', output_size=(2,1))}
        layers += {'5': dict(type='ReshapeLayer', ndim=2)}
        layers += {'6': dict(type='LinearLayer', in_features=None, out_features=self.num_outputs * 8)}
        layers += {'7': dict(type='ReluLayer')}
        layers += {'8': dict(type='LinearLayer', in_features=None, out_features=self.num_outputs)}
        model_spec = dict(model_spec=layers)
        return model_spec


class REG_TS_GEN_BASE_6K_NPU(GenericModelWithSpec):
    """NPU-optimized ~6K param regression model using depthwise separable convolutions.

    Architecture: BN -> Conv -> (DWCONV+PWCONV)x2 -> AvgPool -> FC
    NPU Compliance: m4 channels, DWCONV kW<=7, FC input>=16
    """
    def __init__(self, config, input_features=16, variables=4, num_outputs=1, kernel=5):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_outputs=num_outputs, kernel=kernel)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_outputs)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        # Initial conv to get to m4 channels
        layers += {'1': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(self.kernel,1), stride=(2,1), padding=(2,0))}
        # Depthwise separable block 1
        layers += {'2': dict(type='ConvBNReLULayer', in_channels=8, out_channels=8, kernel_size=(self.kernel,1), stride=(2,1), padding=(2,0), groups=8)}
        layers += {'3': dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0))}
        # Depthwise separable block 2
        layers += {'4': dict(type='ConvBNReLULayer', in_channels=16, out_channels=16, kernel_size=(self.kernel,1), stride=(2,1), padding=(2,0), groups=16)}
        layers += {'5': dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0))}
        layers += {'6': dict(type='AdaptiveAvgPoolLayer', output_size=(2,1))}
        layers += {'7': dict(type='ReshapeLayer', ndim=2)}
        layers += {'8': dict(type='LinearLayer', in_features=None, out_features=self.num_outputs * 32)}
        layers += {'9': dict(type='ReluLayer')}
        layers += {'10': dict(type='LinearLayer', in_features=None, out_features=self.num_outputs)}
        model_spec = dict(model_spec=layers)
        return model_spec


class REG_TS_GEN_BASE_8K_NPU(GenericModelWithSpec):
    """NPU-optimized ~8K param regression model.

    Architecture: BN -> Conv(8ch) -> Conv(16ch) -> Conv(32ch) -> Conv(32ch) -> AvgPool -> FC
    NPU Compliance: m4 channels, kH<=7, FC input>=16
    """
    def __init__(self, config, input_features=16, variables=4, num_outputs=1, kernel=3):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_outputs=num_outputs, kernel=kernel)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_outputs)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'2': dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'3': dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'4': dict(type='ConvBNReLULayer', in_channels=32, out_channels=32, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'5': dict(type='AdaptiveAvgPoolLayer', output_size=(2,1))}
        layers += {'6': dict(type='ReshapeLayer', ndim=2)}
        layers += {'7': dict(type='LinearLayer', in_features=None, out_features=self.num_outputs * 32)}
        layers += {'8': dict(type='ReluLayer')}
        layers += {'9': dict(type='LinearLayer', in_features=None, out_features=self.num_outputs)}
        model_spec = dict(model_spec=layers)
        return model_spec


class REG_TS_GEN_BASE_20K_NPU(GenericModelWithSpec):
    """NPU-optimized ~20K param regression model.

    Architecture: BN -> Conv(16ch) -> MaxPool -> Conv(32ch) -> MaxPool -> Conv(64ch) -> Conv(64ch) -> AvgPool -> FC
    NPU Compliance: m4 channels, kH<=7, MaxPool<=4, FC input>=16
    """
    def __init__(self, config, input_features=16, variables=4, num_outputs=1, kernel=3):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_outputs=num_outputs, kernel=kernel)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_outputs)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=16, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'2': dict(type='MaxPoolLayer', kernel_size=(3,1))}
        layers += {'3': dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'4': dict(type='MaxPoolLayer', kernel_size=(3,1))}
        layers += {'5': dict(type='ConvBNReLULayer', in_channels=32, out_channels=64, kernel_size=(self.kernel,1), stride=(2,1), padding=(1,0))}
        layers += {'6': dict(type='ConvBNReLULayer', in_channels=64, out_channels=32, kernel_size=(self.kernel,1), stride=(1,1), padding=(1,0))}
        layers += {'7': dict(type='AdaptiveAvgPoolLayer', output_size=(2,1))}
        layers += {'8': dict(type='ReshapeLayer', ndim=2)}
        layers += {'9': dict(type='LinearLayer', in_features=None, out_features=self.num_outputs * 64)}
        layers += {'10': dict(type='ReluLayer')}
        layers += {'11': dict(type='LinearLayer', in_features=None, out_features=self.num_outputs)}
        model_spec = dict(model_spec=layers)
        return model_spec


# Export all regression models
__all__ = [
    # Existing models (already NPU compliant)
    'REG_TS_GEN_BASE_1K',
    'REG_TS_GEN_BASE_3K',
    'REG_TS_GEN_BASE_10K',
    'REG_TS_GEN_BASE_4K',
    'REG_TS_GEN_BASE_13K',
    # NPU-Optimized gap-filling models
    'REG_TS_GEN_BASE_500_NPU',
    'REG_TS_GEN_BASE_2K_NPU',
    'REG_TS_GEN_BASE_6K_NPU',
    'REG_TS_GEN_BASE_8K_NPU',
    'REG_TS_GEN_BASE_20K_NPU',
]
