import torch
from ..utils import py_utils
from .generic_model_spec import GenericModelWithSpec


class AE_CNN_TS_GEN_BASE_4K1(GenericModelWithSpec):
    def __init__(self, config, input_features=512, variables=3, num_classes=4, with_input_batchnorm=True):
        super().__init__(config, input_features=input_features, variables=variables,
                         with_input_batchnorm=with_input_batchnorm, num_classes=num_classes)

        features = 8
        top_layer = torch.nn.BatchNorm2d(num_features=variables)
        self.top_layer = top_layer

        encoder = []
        encoder += [torch.nn.Conv2d(in_channels=variables, out_channels=features, kernel_size=(3, 1), stride=(2, 1))]
        encoder += [torch.nn.BatchNorm2d(features)]
        encoder += [torch.nn.ReLU()]

        encoder += [torch.nn.Conv2d(in_channels=features, out_channels=features * 2, kernel_size=(3, 1), stride=(2, 1))]
        encoder += [torch.nn.BatchNorm2d(features * 2)]
        encoder += [torch.nn.ReLU()]

        encoder += [
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 4, kernel_size=(3, 1), stride=(2, 1))]
        encoder += [torch.nn.BatchNorm2d(features * 4)]
        encoder += [torch.nn.ReLU()]
        self.encoder = torch.nn.Sequential(*encoder)

        decoder = []
        decoder += [
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 2, kernel_size=(3, 1), stride=(2, 1))]
        decoder += [torch.nn.BatchNorm2d(features * 4)]
        decoder += [torch.nn.ReLU()]

        decoder += [torch.nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=(3, 1), stride=(2, 1))]
        decoder += [torch.nn.BatchNorm2d(features * 4)]
        decoder += [torch.nn.ReLU()]

        decoder += [torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=(3, 1), stride=(2, 1))]
        decoder += [torch.nn.BatchNorm2d(features * 4)]
        decoder += [torch.nn.ReLU()]
        self.decoder = torch.nn.Sequential(*decoder)

    def forward(self, x):
        x = self.top_layer(x)
        for layer in self.encoder:
            x = layer(x)
        for layer in self.decoder:
            x = layer(x)
        return x


class AE_CNN_TS_GEN_BASE_4K(torch.nn.Module):
    def __init__(self, config, input_features=512, variables=1, num_classes=4, with_input_batchnorm=True,
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
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')  # Upsample
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')  # Upsample
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')  # Upsample
        return x


class AE_CNN_TS_GEN_BASE_16K(torch.nn.Module):
    def __init__(self, config, input_features=512, variables=1, num_classes=4, with_input_batchnorm=True, features=8):
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
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')  # Upsample
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')  # Upsample
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')  # Upsample
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')  # Upsample
        x = self.decoder(x)
        return x


class AE_CNN_TS_GEN_BASE_1K(torch.nn.Module):
    def __init__(self, config, input_features=512, variables=1, num_classes=4, with_input_batchnorm=True, features=16):
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
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')  # Upsample
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')  # Upsample
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')  # Upsample
        x = self.decoder(x)
        return x
    

class AD_CNN_TS_17K(torch.nn.Module):
    def __init__(self, config):
        super(AD_CNN_TS_17K, self).__init__()

        output_channels = 8
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(config['variables'], output_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels, output_channels*2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels*2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels*2, output_channels*4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels*4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels*4, output_channels*8, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels*8),
            torch.nn.ReLU(),
        )

        ### Decode here
        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=(2,1), mode='bilinear'),
            torch.nn.Conv2d(output_channels*8, output_channels*4, kernel_size=(3, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels*4),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=(2,1), mode='bilinear'),
            torch.nn.Conv2d(output_channels*4, output_channels*2, kernel_size=(3, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels*2),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=(2,1), mode='bilinear'),
            torch.nn.Conv2d(output_channels*2, output_channels, kernel_size=(3, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=(2,1), mode='bilinear'),
            torch.nn.Conv2d(output_channels, config['variables'], kernel_size=(3, 1), padding=(1, 0)),
        )

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        return decoded