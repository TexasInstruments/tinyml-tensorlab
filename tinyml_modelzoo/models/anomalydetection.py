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

"""Autoencoder and anomaly detection models for time series data."""

import torch


class AE_CNN_TS_GEN_BASE_4K(torch.nn.Module):
    def __init__(self, config, input_features=512, variables=1, num_classes=4,
                 features=8):
        super(AE_CNN_TS_GEN_BASE_4K, self).__init__()

        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=variables, out_channels=features, kernel_size=(3, 1), stride=(2, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features * 2, kernel_size=(3, 1), stride=(2, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 4, kernel_size=(3, 1), stride=(2, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 4),
            torch.nn.ReLU()
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 2, kernel_size=(3, 1), stride=(1, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=(3, 1), stride=(1, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=variables, kernel_size=(3, 1), stride=(1, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(variables),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        return x


class AE_CNN_TS_GEN_BASE_16K(torch.nn.Module):
    def __init__(self, config, input_features=512, variables=1, num_classes=4, features=8):
        super(AE_CNN_TS_GEN_BASE_16K, self).__init__()

        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=variables, out_channels=features, kernel_size=(3, 1), stride=(2, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features * 2, kernel_size=(3, 1), stride=(2, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 4, kernel_size=(3, 1), stride=(2, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 8, kernel_size=(3, 1), stride=(2, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 8),
            torch.nn.ReLU()
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 8, out_channels=features * 4, kernel_size=(3, 1), stride=(1, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 2, kernel_size=(3, 1), stride=(1, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=(3, 1), stride=(1, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=variables, kernel_size=(3, 1), stride=(1, 1),
                            padding=(1, 0)),
            torch.nn.BatchNorm2d(variables),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = self.decoder(x)
        return x


class AE_CNN_TS_GEN_BASE_1K(torch.nn.Module):
    def __init__(self, config, input_features=512, variables=1, num_classes=4, features=16):
        super(AE_CNN_TS_GEN_BASE_1K, self).__init__()
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=variables, out_channels=features, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features // 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features // 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features // 2, out_channels=features // 4, kernel_size=(3, 1), stride=(2, 1),
                      padding=(1, 0)),
            torch.nn.BatchNorm2d(features // 4),
            torch.nn.ReLU()
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features // 4, out_channels=features // 2, kernel_size=(3, 1), stride=(1, 1),
                      padding=(1, 0)),
            torch.nn.BatchNorm2d(features // 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features // 2, out_channels=features, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=variables, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(variables),
            torch.nn.ReLU()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = self.decoder(x)
        return x


class AD_CNN_TS_17K(torch.nn.Module):
    def __init__(self, config):
        super(AD_CNN_TS_17K, self).__init__()

        output_channels = 8

        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(config['variables'], output_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU()
        )
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(output_channels, output_channels*2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels*2),
            torch.nn.ReLU()
        )
        self.enc3 = torch.nn.Sequential(
            torch.nn.Conv2d(output_channels*2, output_channels*4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels*4),
            torch.nn.ReLU()
        )
        self.enc4 = torch.nn.Sequential(
            torch.nn.Conv2d(output_channels*4, output_channels*8, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels*8),
            torch.nn.ReLU()
        )

        # Decoder layers
        self.dec4 = torch.nn.Sequential(
            torch.nn.Conv2d(output_channels*8, output_channels*4, kernel_size=(3, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels*4),
            torch.nn.ReLU()
        )
        self.dec3 = torch.nn.Sequential(
            torch.nn.Conv2d(output_channels*4, output_channels*2, kernel_size=(3, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels*2),
            torch.nn.ReLU()
        )
        self.dec2 = torch.nn.Sequential(
            torch.nn.Conv2d(output_channels*2, output_channels, kernel_size=(3, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU()
        )
        self.dec1 = torch.nn.Conv2d(output_channels, config['variables'], kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x):
        # Store intermediate sizes
        x0_size = x.shape[2:]

        # Encoder
        x1 = self.enc1(x)
        x1_size = x1.shape[2:]

        x2 = self.enc2(x1)
        x2_size = x2.shape[2:]

        x3 = self.enc3(x2)
        x3_size = x3.shape[2:]

        x4 = self.enc4(x3)

        # Decoder with exact size matching
        d4 = torch.nn.functional.interpolate(x4, size=x3_size, mode='bilinear')
        d4 = self.dec4(d4)

        d3 = torch.nn.functional.interpolate(d4, size=x2_size, mode='bilinear')
        d3 = self.dec3(d3)

        d2 = torch.nn.functional.interpolate(d3, size=x1_size, mode='bilinear')
        d2 = self.dec2(d2)

        d1 = torch.nn.functional.interpolate(d2, size=x0_size, mode='bilinear')
        output = self.dec1(d1)

        return output


class AD_3_LAYER_DEEP_LINEAR_MODEL_TS(torch.nn.Module):
    def __init__(self, config):
        super(AD_3_LAYER_DEEP_LINEAR_MODEL_TS, self).__init__()

        # Define dimensions for encoder
        input_size = config['input_features']*config['variables']
        hidden_size1 = input_size // 4
        hidden_size2 = hidden_size1 // 2
        bottleneck_size = hidden_size2 // 2

        # Encoder layers
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, hidden_size1),
            torch.nn.BatchNorm1d(hidden_size1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size1, hidden_size2),
            torch.nn.BatchNorm1d(hidden_size2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size2, bottleneck_size),
            torch.nn.BatchNorm1d(bottleneck_size),
            torch.nn.ReLU(),
        )

        # Decoder layers
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_size, hidden_size2),
            torch.nn.BatchNorm1d(hidden_size2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size2, hidden_size1),
            torch.nn.BatchNorm1d(hidden_size1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size1, input_size),
        )

        # Store original shape information for reshaping
        self.input_shape = None

    def forward(self, x):

        self.input_shape = x.shape
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        # Reshape back to original dimensions
        decoded = decoded.view(self.input_shape)
        return decoded


class AD_3_LAYER_DEEP_ONDEVICE_TRAINABLE_MODEL_TS(torch.nn.Module):
    def __init__(self,config):
        super(AD_3_LAYER_DEEP_ONDEVICE_TRAINABLE_MODEL_TS, self).__init__()
        input_size = config['input_features']*config['variables']
        hidden_size1 = input_size // 2
        hidden_size2 = hidden_size1 // 2
        bottleneck_size = hidden_size2 // 2

        # Encoder layers
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, hidden_size1),
            torch.nn.BatchNorm1d(hidden_size1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size1, hidden_size2),
            torch.nn.BatchNorm1d(hidden_size2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size2, bottleneck_size),
            torch.nn.BatchNorm1d(bottleneck_size),
            torch.nn.ReLU(),
        )

        # Decoder layers
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_size, hidden_size2),
            torch.nn.BatchNorm1d(hidden_size2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size2, hidden_size1),
            torch.nn.BatchNorm1d(hidden_size1),
            torch.nn.ReLU()
        )

        self.frozen_model = torch.nn.Sequential(
            self.encoder,
            self.decoder
        )

        self.trainable_last_layer = torch.nn.Linear(hidden_size1, input_size)

    def forward(self, x):
        self.input_dim = x.shape
        frozen_model_output = self.frozen_model(x)
        output = self.trainable_last_layer(frozen_model_output)
        output = output.view(self.input_dim)
        return output


# NPU-Optimized Anomaly Detection Models
# These models follow TI NPU constraints for the encoder portion:
# - All channels are multiples of 4 (m4)
# - GCONV kernel heights <= 7
# Note: Decoder upsampling (interpolate) falls back to CPU

class AE_CNN_TS_GEN_BASE_500_NPU(torch.nn.Module):
    """NPU-optimized ~500 param autoencoder.

    Architecture: Conv(4ch) -> Conv(8ch) encoder with symmetric decoder.
    NPU Compliance: m4 channels, kH<=7 for encoder convolutions.
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=4, features=4):
        super(AE_CNN_TS_GEN_BASE_500_NPU, self).__init__()

        # Encoder - NPU optimized with m4 channels
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=variables, out_channels=features, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features * 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU()
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=variables, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(variables),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        return x


class AE_CNN_TS_GEN_BASE_2K_NPU(torch.nn.Module):
    """NPU-optimized ~2K param autoencoder.

    Architecture: Conv(8ch) -> Conv(16ch) -> Conv(16ch) encoder with symmetric decoder.
    NPU Compliance: m4 channels, kH<=7 for encoder convolutions.
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=4, features=8):
        super(AE_CNN_TS_GEN_BASE_2K_NPU, self).__init__()

        # Encoder - NPU optimized with m4 channels
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=variables, out_channels=features, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features * 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU()
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=variables, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(variables),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        return x


class AE_CNN_TS_GEN_BASE_6K_NPU(torch.nn.Module):
    """NPU-optimized ~6K param autoencoder with depthwise separable convolutions.

    Architecture: Conv -> (DWCONV+PWCONV)x2 encoder with symmetric decoder.
    NPU Compliance: m4 channels, DWCONV kW<=7 for encoder.
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=4, features=8):
        super(AE_CNN_TS_GEN_BASE_6K_NPU, self).__init__()

        # Encoder - NPU optimized with depthwise separable convolutions
        self.encoder = torch.nn.Sequential(
            # Initial conv to get to m4 channels
            torch.nn.Conv2d(in_channels=variables, out_channels=features, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            # Depthwise separable block 1
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0), groups=features),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features * 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            # Depthwise separable block 2
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 2, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0), groups=features * 2),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            torch.nn.BatchNorm2d(features * 4),
            torch.nn.ReLU()
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=variables, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(variables),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        return x


class AE_CNN_TS_GEN_BASE_8K_NPU(torch.nn.Module):
    """NPU-optimized ~8K param autoencoder.

    Architecture: Conv(8ch) -> Conv(16ch) -> Conv(32ch) -> Conv(32ch) encoder with symmetric decoder.
    NPU Compliance: m4 channels, kH<=7 for encoder convolutions.
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=4, features=8):
        super(AE_CNN_TS_GEN_BASE_8K_NPU, self).__init__()

        # Encoder - NPU optimized with m4 channels
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=variables, out_channels=features, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features * 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 4),
            torch.nn.ReLU()
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 4, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=variables, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(variables),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        return x


class AE_CNN_TS_GEN_BASE_10K_NPU(torch.nn.Module):
    """NPU-optimized ~10K param autoencoder.

    Architecture: Conv(16ch) -> Conv(32ch) -> Conv(32ch) encoder with symmetric decoder.
    NPU Compliance: m4 channels, kH<=7 for encoder convolutions.
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=4, features=16):
        super(AE_CNN_TS_GEN_BASE_10K_NPU, self).__init__()

        # Encoder - NPU optimized with m4 channels
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=variables, out_channels=features, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features * 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU()
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=variables, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(variables),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        return x


class AE_CNN_TS_GEN_BASE_20K_NPU(torch.nn.Module):
    """NPU-optimized ~20K param autoencoder.

    Architecture: Conv(16ch) -> Conv(32ch) -> Conv(64ch) -> Conv(64ch) encoder with symmetric decoder.
    NPU Compliance: m4 channels, kH<=7 for encoder convolutions.
    """
    def __init__(self, config, input_features=512, variables=1, num_classes=4, features=16):
        super(AE_CNN_TS_GEN_BASE_20K_NPU, self).__init__()

        # Encoder - NPU optimized with m4 channels
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=variables, out_channels=features, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features * 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 4),
            torch.nn.ReLU()
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 4, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=variables, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(variables),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        return x


# Export all anomaly detection models
__all__ = [
    # Existing models
    'AE_CNN_TS_GEN_BASE_1K',
    'AE_CNN_TS_GEN_BASE_4K',
    'AE_CNN_TS_GEN_BASE_16K',
    'AD_CNN_TS_17K',
    'AD_3_LAYER_DEEP_LINEAR_MODEL_TS',
    'AD_3_LAYER_DEEP_ONDEVICE_TRAINABLE_MODEL_TS',
    # NPU-Optimized gap-filling models
    'AE_CNN_TS_GEN_BASE_500_NPU',
    'AE_CNN_TS_GEN_BASE_2K_NPU',
    'AE_CNN_TS_GEN_BASE_6K_NPU',
    'AE_CNN_TS_GEN_BASE_8K_NPU',
    'AE_CNN_TS_GEN_BASE_10K_NPU',
    'AE_CNN_TS_GEN_BASE_20K_NPU',
]
