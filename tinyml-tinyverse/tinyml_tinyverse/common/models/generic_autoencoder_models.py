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
        x0_size = x.shape[2:]  # Original size
        
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