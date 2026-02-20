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

"""Classification models for time series data."""

import torch

from ..utils import py_utils
from .base import GenericModelWithSpec


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


class CNN_TS_GEN_BASE_1K_NPU(GenericModelWithSpec):
    """
    NPU-Compliant 1K-parameter model.

    This model is already fully NPU compliant:
    - All channels m4 (8, 8, 16)
    - All kernel sizes <= 7 (5, 5, 3, 3)
    - MaxPool kernel <= 4 (3)
    - FC input >= 16 (64 features)

    Architecture: BatchNorm -> Conv5x1 -> Conv5x1 -> MaxPool -> Conv3x1 -> AdaptiveAvgPool -> FC
    ~1K parameters
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=2):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0':dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(5,1), stride=(1,1))}
        layers += {'1a':dict(type='ConvBNReLULayer', in_channels=8, out_channels=8, kernel_size=(5,1), stride=(1,1))}
        layers += {'2': dict(type='MaxPoolLayer', kernel_size=(3, 1), stride=(2, 1))}
        layers += {'3':dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(3,1), stride=(1,1))}
        layers += {'5':dict(type='AdaptiveAvgPoolLayer', output_size=(4,1))}
        layers += {'6':dict(type='ReshapeLayer', ndim=2)}
        layers += {'7':dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec


class CNN_TS_GEN_BASE_2K_NPU(GenericModelWithSpec):
    """
    NPU-Compliant 2K-parameter model.

    This model is already fully NPU compliant:
    - All channels m4 (16, 16, 16, 32)
    - All kernel sizes <= 7 (5, 3, 5, 5)
    - MaxPool kernel <= 4 (3)
    - FC input >= 16 (128 features)

    Architecture: BatchNorm -> Conv5x1 -> Conv3x1 -> MaxPool -> Conv5x1 -> Conv5x1 -> AdaptiveAvgPool -> FC
    ~2K parameters
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=2):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0':dict(type='BatchNormLayer', num_features=self.variables)}
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


class CNN_TS_GEN_BASE_6K_NPU(GenericModelWithSpec):
    """
    NPU-Compliant 6K-parameter model with depthwise separable convolutions.

    This model is already fully NPU compliant:
    - All channels m4 (16, 16, 32, 32, 48, 16)
    - All kernel sizes <= 7 (3, 3, 1, 3, 1, 5)
    - MaxPool kernels <= 4 (3, 3)
    - Uses depthwise separable convolutions (DWCONV + PWCONV pattern)
    - FC input >= 16 (64 features)

    Architecture: BatchNorm -> Conv -> MaxPool -> [DWCONV+PWCONV] x 2 -> Conv -> AdaptiveAvgPool -> FC
    ~6K parameters
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=2):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
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


class RES_CAT_CNN_TS_GEN_BASE_3K(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=3, num_classes=4, out_channel_layer1=4, out_channel_layer2=8, out_channel_layer3=16):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes,
                         out_channel_layer1=out_channel_layer1, out_channel_layer2=out_channel_layer2, out_channel_layer3=out_channel_layer3)

        top_layer = torch.nn.BatchNorm2d(num_features=self.variables)
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
    def __init__(self, config, input_features=512, variables=3, num_classes=4, out_channel_layer1=4, out_channel_layer2=8, out_channel_layer3=16):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes,
                         out_channel_layer1=out_channel_layer1, out_channel_layer2=out_channel_layer2, out_channel_layer3=out_channel_layer3)

        top_layer = torch.nn.BatchNorm2d(num_features=self.variables)
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
    def __init__(self, config, input_features=512, variables=3, num_classes=4):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)

        layers = []
        in_ch = self.variables
        out_ch = 16

        bn1 = torch.nn.BatchNorm2d(num_features=self.variables)

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
    def __init__(self, config, input_features=512, variables=3, num_classes=4, depth=3):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes,
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
        self.layers = [torch.nn.BatchNorm2d(num_features=self.variables)]
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


class CNN_TS_PIR2D_BASE(GenericModelWithSpec):
    def __init__(self, config, input_features=(25,25), variables=1, num_classes=3, use_bn: bool = False):
        super().__init__(config, input_features=input_features, variables=variables, num_classes=num_classes)
        # Define the layers of the CNN
        C_in = variables
        H, W = input_features
        self.input_scale = torch.nn.Parameter(torch.tensor(1/1024.0), requires_grad=False)
        self.input_clamp = torch.nn.Hardtanh(min_val=-1.0, max_val=1.0)
        self.bn0   = torch.nn.BatchNorm2d(num_features=1) if use_bn else torch.nn.Identity()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn1   = torch.nn.BatchNorm2d(num_features=8) if use_bn else torch.nn.Identity()
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn2   = torch.nn.BatchNorm2d(num_features=16) if use_bn else torch.nn.Identity()
        self.relu2 = torch.nn.ReLU()
        self.pool  = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(0,0))
        # ---- analytic output-size math (no helper calls) ----
        # conv1 params
        k1h, k1w = self.conv1.kernel_size
        s1h, s1w = self.conv1.stride
        p1h, p1w = self.conv1.padding
        d1h, d1w = self.conv1.dilation

        # pool params
        kph, kpw = self.pool.kernel_size
        sph, spw = self.pool.stride
        pph, ppw = self.pool.padding

        # conv2 params
        k2h, k2w = self.conv2.kernel_size
        s2h, s2w = self.conv2.stride
        p2h, p2w = self.conv2.padding
        d2h, d2w = self.conv2.dilation

        # after conv1
        H1 = (H + 2*p1h - d1h*(k1h - 1) - 1) // s1h + 1
        W1 = (W + 2*p1w - d1w*(k1w - 1) - 1) // s1w + 1
        # after pool1
        H2 = (H1 + 2*pph - (kph - 1) - 1) // sph + 1
        W2 = (W1 + 2*ppw - (kpw - 1) - 1) // spw + 1
        # after conv2
        H3 = (H2 + 2*p2h - d2h*(k2h - 1) - 1) // s2h + 1
        W3 = (W2 + 2*p2w - d2w*(k2w - 1) - 1) // s2w + 1
        # after pool2
        H4 = (H3 + 2*pph - (kph - 1) - 1) // sph + 1
        W4 = (W3 + 2*ppw - (kpw - 1) - 1) // spw + 1

        fc_input_features = H4 * W4 * self.conv2.out_channels
        self.flatten = torch.nn.Flatten()
        self.fc1     = torch.nn.Linear(in_features=fc_input_features, out_features=128)
        self.relu3 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=self.num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.flatten(x)
        if self.fc1 is None:
            self.fc1 = torch.nn.Linear(x.shape[1],128).to(x.device)

        x = self.fc1(x)
        fe = self.relu3(x)
        x = self.dropout(fe)
        x = self.fc2(x)
        return x


# =============================================================================
# NPU-OPTIMIZED MODELS
# These models are designed to fully comply with TI NPU constraints for
# optimal hardware acceleration on F28P55, F28P65, and similar devices.
# See docs/NPU_CONFIGURATION_GUIDELINES.md for detailed constraints.
# =============================================================================


class CNN_TS_GEN_BASE_100_NPU(GenericModelWithSpec):
    """
    NPU-Optimized 100-parameter model.

    Fixes from original CNN_TS_GEN_BASE_100:
    - Increased output channels to 8 (was 4) to ensure FC input >= 8 features
    - All channels are m4 compliant

    Architecture: BatchNorm -> Conv1x1 -> Conv3x1 -> AdaptiveAvgPool -> FC
    ~120 parameters
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=2):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        # Use 8 output channels (m4) to ensure FC gets minimum 8 features
        layers += {'1': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(1, 1), stride=(1, 1))}
        layers += {'2': dict(type='ConvBNReLULayer', in_channels=8, out_channels=8, kernel_size=(3, 1), stride=(1, 1))}
        layers += {'3': dict(type='AdaptiveAvgPoolLayer', output_size=(1, 1))}
        layers += {'4': dict(type='ReshapeLayer', ndim=2)}
        # FC input = 8 features (meets NPU minimum)
        layers += {'5': dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec


class CNN_TS_GEN_BASE_500_NPU(GenericModelWithSpec):
    """
    NPU-Optimized ~500-parameter model.

    Fills gap between 100 and 1K parameter models.
    All layers NPU compliant:
    - Channels: 8, 8, 16 (all m4)
    - Kernel sizes: 5, 3 (all <= 7)
    - FC input: 16 features (meets minimum)

    Architecture: BatchNorm -> Conv5x1 -> Conv3x1 -> AdaptiveAvgPool -> FC
    ~500 parameters
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=2):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(5, 1), stride=(1, 1))}
        layers += {'2': dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(3, 1), stride=(1, 1))}
        layers += {'3': dict(type='AdaptiveAvgPoolLayer', output_size=(1, 1))}
        layers += {'4': dict(type='ReshapeLayer', ndim=2)}
        layers += {'5': dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec


class CNN_TS_GEN_BASE_4K_NPU(GenericModelWithSpec):
    """
    NPU-Optimized 4K-parameter model.

    Fixes from original CNN_TS_GEN_BASE_4K:
    - Changed kernel_size=(7,1) to (5,1) to stay well within kH<=7 limit
    - All other configurations already NPU compliant

    Architecture: BatchNorm -> Conv5x1/s2 -> MaxPool -> Conv5x1/s2 -> Conv5x1/s2 -> AdaptiveAvgPool -> FC
    ~4K parameters
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=2):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        # Changed from (7,1) to (5,1) for better NPU compatibility
        layers += {'1': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(5, 1), stride=(2, 1))}
        layers += {'1a': dict(type='ConvBNReLULayer', in_channels=8, out_channels=8, kernel_size=(3, 1), stride=(1, 1))}
        layers += {'1p': dict(type='MaxPoolLayer', kernel_size=(3, 1), stride=(2, 1))}
        layers += {'2': dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(5, 1), stride=(2, 1))}
        layers += {'3': dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(5, 1), stride=(2, 1))}
        layers += {'4': dict(type='AdaptiveAvgPoolLayer', output_size=(4, 1))}
        layers += {'5': dict(type='ReshapeLayer', ndim=2)}
        layers += {'6': dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec


class CNN_TS_GEN_BASE_8K_NPU(GenericModelWithSpec):
    """
    NPU-Optimized ~8K-parameter model.

    Fills gap between 6K and 13K parameter models.
    Uses depthwise separable convolutions for efficiency.
    All layers NPU compliant:
    - All channels m4 (16, 32, 48, 64)
    - All kernel sizes <= 7
    - Depthwise separable pattern (DWCONV + PWCONV)

    Architecture: BatchNorm -> Conv -> [DWCONV+PWCONV] x 2 -> Conv -> AdaptiveAvgPool -> FC
    ~8K parameters
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=2):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        # Initial conv with downsampling
        layers += {'1': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=16, kernel_size=(5, 1), stride=(2, 1))}
        layers += {'2': dict(type='MaxPoolLayer', kernel_size=(3, 1), stride=(2, 1))}
        # Depthwise separable block 1
        layers += {'3a': dict(type='ConvBNReLULayer', in_channels=16, out_channels=16, kernel_size=(5, 1), stride=(1, 1), groups=16)}
        layers += {'3b': dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(1, 1), stride=(1, 1))}
        layers += {'4': dict(type='MaxPoolLayer', kernel_size=(3, 1), stride=(2, 1))}
        # Depthwise separable block 2
        layers += {'5a': dict(type='ConvBNReLULayer', in_channels=32, out_channels=32, kernel_size=(5, 1), stride=(1, 1), groups=32)}
        layers += {'5b': dict(type='ConvBNReLULayer', in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1))}
        # Final conv
        layers += {'6': dict(type='ConvBNReLULayer', in_channels=64, out_channels=32, kernel_size=(3, 1), stride=(1, 1))}
        layers += {'7': dict(type='AdaptiveAvgPoolLayer', output_size=(4, 1))}
        layers += {'8': dict(type='ReshapeLayer', ndim=2)}
        layers += {'9': dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec


class CNN_TS_GEN_BASE_13K_NPU(GenericModelWithSpec):
    """
    NPU-Optimized 13K-parameter model.

    Fixes from original CNN_TS_GEN_BASE_13K:
    - Changed first kernel_size=(7,1) to (5,1) for safer NPU margin
    - All other configurations already NPU compliant

    Architecture: BatchNorm -> 6x Conv layers with progressive channel expansion -> AdaptiveAvgPool -> FC
    ~13K parameters
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=2):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        # Changed from (7,1) to (5,1) for better NPU compatibility
        layers += {'1': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(5, 1), stride=(2, 1))}
        layers += {'2': dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(3, 1), stride=(2, 1))}
        layers += {'3': dict(type='ConvBNReLULayer', in_channels=16, out_channels=16, kernel_size=(3, 1), stride=(1, 1))}
        layers += {'4': dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(3, 1), stride=(2, 1))}
        layers += {'5': dict(type='ConvBNReLULayer', in_channels=32, out_channels=32, kernel_size=(3, 1), stride=(1, 1))}
        layers += {'6': dict(type='ConvBNReLULayer', in_channels=32, out_channels=64, kernel_size=(3, 1), stride=(2, 1))}
        layers += {'7': dict(type='AdaptiveAvgPoolLayer', output_size=(4, 1))}
        layers += {'8': dict(type='ReshapeLayer', ndim=2)}
        layers += {'9': dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec


class CNN_TS_GEN_BASE_20K_NPU(GenericModelWithSpec):
    """
    NPU-Optimized ~20K-parameter model.

    Fills gap between 13K and 55K parameter models.
    All layers NPU compliant:
    - All channels m4 (8, 16, 32, 64, 128)
    - All kernel sizes <= 7
    - Mix of regular and depthwise separable convolutions

    Architecture: BatchNorm -> Conv -> Conv -> [DWCONV+PWCONV] -> Conv -> Conv -> AdaptiveAvgPool -> FC
    ~20K parameters
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=2):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        # Stage 1: Initial feature extraction
        layers += {'1': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=16, kernel_size=(5, 1), stride=(2, 1))}
        layers += {'2': dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(3, 1), stride=(2, 1))}
        layers += {'3': dict(type='MaxPoolLayer', kernel_size=(3, 1), stride=(2, 1))}
        # Stage 2: Depthwise separable block
        layers += {'4a': dict(type='ConvBNReLULayer', in_channels=32, out_channels=32, kernel_size=(5, 1), stride=(1, 1), groups=32)}
        layers += {'4b': dict(type='ConvBNReLULayer', in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1))}
        # Stage 3: Feature refinement
        layers += {'5': dict(type='ConvBNReLULayer', in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(2, 1))}
        layers += {'6': dict(type='ConvBNReLULayer', in_channels=64, out_channels=128, kernel_size=(3, 1), stride=(1, 1))}
        # Stage 4: Final conv and pooling
        layers += {'7': dict(type='ConvBNReLULayer', in_channels=128, out_channels=64, kernel_size=(3, 1), stride=(1, 1))}
        layers += {'8': dict(type='AdaptiveAvgPoolLayer', output_size=(4, 1))}
        layers += {'9': dict(type='ReshapeLayer', ndim=2)}
        layers += {'10': dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec


class CNN_TS_GEN_BASE_55K_NPU(GenericModelWithSpec):
    """
    NPU-Optimized 55K-parameter model for large input sequences.

    Fixes from original CNN_TS_GEN_BASE_55K:
    - Replaced kernel_size=(16,1) with two (5,1) convolutions
    - Replaced kernel_size=(12,1) with (5,1) + (5,1)
    - Replaced kernel_size=(9,1) with (5,1) + (3,1)
    - Replaced MaxPool kernel=(8,1) with (4,1)
    - All channels now m4 compliant (removed 12)

    Architecture: Multiple stages of small convolutions instead of large kernels
    ~55K parameters
    """
    def __init__(self, config, input_features=2500, variables=1, num_classes=4):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}

        # Stage 1: Replace (16,1) kernel with two (5,1) convolutions + aggressive stride
        layers += {'1a': dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(5, 1), stride=(2, 1))}
        layers += {'1b': dict(type='ConvBNReLULayer', in_channels=8, out_channels=8, kernel_size=(5, 1), stride=(2, 1))}
        layers += {'2': dict(type='MaxPoolLayer', kernel_size=(4, 1), stride=(4, 1))}  # Changed from (8,1)

        # Stage 2: Replace (12,1) kernel with two smaller convolutions
        layers += {'3a': dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(5, 1), stride=(1, 1))}
        layers += {'3b': dict(type='ConvBNReLULayer', in_channels=16, out_channels=16, kernel_size=(5, 1), stride=(2, 1))}
        layers += {'4': dict(type='MaxPoolLayer', kernel_size=(4, 1), stride=(2, 1))}

        # Stage 3: Replace (9,1) kernel with (5,1) + (3,1)
        layers += {'5a': dict(type='ConvBNReLULayer', in_channels=16, out_channels=32, kernel_size=(5, 1), stride=(1, 1))}
        layers += {'5b': dict(type='ConvBNReLULayer', in_channels=32, out_channels=32, kernel_size=(3, 1), stride=(1, 1))}
        layers += {'6': dict(type='MaxPoolLayer', kernel_size=(4, 1), stride=(2, 1))}  # Changed from (5,1)

        # Stage 4: Feature extraction with compliant kernels
        layers += {'7': dict(type='ConvBNReLULayer', in_channels=32, out_channels=64, kernel_size=(5, 1), stride=(1, 1))}
        layers += {'8': dict(type='MaxPoolLayer', kernel_size=(4, 1), stride=(2, 1))}

        # Stage 5: Final feature extraction
        layers += {'9': dict(type='ConvBNReLULayer', in_channels=64, out_channels=64, kernel_size=(5, 1), stride=(1, 1))}
        layers += {'10': dict(type='MaxPoolLayer', kernel_size=(2, 1), stride=(2, 1))}

        layers += {'11': dict(type='ConvBNReLULayer', in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1))}

        layers += {'12': dict(type='ReshapeLayer', ndim=2)}
        layers += {'13': dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}

        model_spec = dict(model_spec=layers)
        return model_spec


# Export all classification models
__all__ = [
    # Non-NPU models (residual networks, specialized architectures)
    'RES_CAT_CNN_TS_GEN_BASE_3K',
    'RES_ADD_CNN_TS_GEN_BASE_3K',
    'HAR_TINIE_CNN_2K',
    'YOLO_Classifier_8K',
    'CNN_TS_PIR2D_BASE',
    # NPU-Optimized models (fully NPU compliant)
    'CNN_TS_GEN_BASE_100_NPU',
    'CNN_TS_GEN_BASE_500_NPU',
    'CNN_TS_GEN_BASE_1K_NPU',
    'CNN_TS_GEN_BASE_2K_NPU',
    'CNN_TS_GEN_BASE_4K_NPU',
    'CNN_TS_GEN_BASE_6K_NPU',
    'CNN_TS_GEN_BASE_8K_NPU',
    'CNN_TS_GEN_BASE_13K_NPU',
    'CNN_TS_GEN_BASE_20K_NPU',
    'CNN_TS_GEN_BASE_55K_NPU',
]
