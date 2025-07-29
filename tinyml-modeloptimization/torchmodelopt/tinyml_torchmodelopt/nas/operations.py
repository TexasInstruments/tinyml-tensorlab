import torch.nn as nn

# OPS is a dictionary mapping operation names to functions that create the corresponding nn.Module.
# Each function takes (C, stride, affine) as arguments and returns a layer/module.
OPS = {
    'none' : lambda C, stride, affine: Zero(stride),  # No operation (returns zeros)
    # 'adaptive_avg_pool_6x1' : lambda C, stride, affine: nn.AdaptiveAvgPool2d((6, 1)),  # Not used
    'avg_pool_3x1' : lambda C, stride, affine: nn.AvgPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),  # 3x1 average pooling
    # 'adaptive_avg_pool_3x1' : lambda C, stride, affine: nn.AdaptiveAvgPool2d((3, 1)),  # Not used
    'max_pool_3x1' : lambda C, stride, affine: nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),  # 3x1 max pooling
    'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),  # Identity or downsample
    'conv_bn_relu_3x1' : lambda C, stride, affine: ConvBNReLU(C, C, (3, 1), (stride, 1), (1, 0), affine=affine),  # 3x1 conv + BN + ReLU
    'conv_bn_relu_5x1' : lambda C, stride, affine: ConvBNReLU(C, C, (5, 1), (stride, 1), (2, 0), affine=affine),  # 5x1 conv + BN + ReLU
    'conv_bn_relu_7x1' : lambda C, stride, affine: ConvBNReLU(C, C, (7, 1), (stride, 1), (3, 0), affine=affine),  # 7x1 conv + BN + ReLU
    'sep_conv_3x1' : lambda C, stride, affine: SepConv(C, C, (3, 1), (stride, 1), (1, 0), affine=affine),         # 3x1 separable conv
    'sep_conv_5x1' : lambda C, stride, affine: SepConv(C, C, (5, 1), (stride, 1), (2, 0), affine=affine),         # 5x1 separable conv
    'sep_conv_7x1' : lambda C, stride, affine: SepConv(C, C, (7, 1), (stride, 1), (3, 0), affine=affine),         # 7x1 separable conv
    'dil_conv_3x1' : lambda C, stride, affine: DilConv(C, C, (3, 1), (stride, 1), (2, 0), (2, 1), affine=affine), # 3x1 dilated conv
    'dil_conv_5x1' : lambda C, stride, affine: DilConv(C, C, (5, 1), (stride, 1), (4, 0), (2, 1), affine=affine), # 5x1 dilated conv
}

class DilConv(nn.Module):
    """
    Dilated depthwise separable convolution followed by pointwise convolution,
    batch normalization, and ReLU activation.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            # Depthwise convolution with dilation
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            # Pointwise convolution
            nn.Conv2d(C_in, C_out, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    """
    Separable convolution: depthwise conv, pointwise conv, repeated twice,
    each followed by batch normalization and ReLU activation.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            # First depthwise conv
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            # First pointwise conv
            nn.Conv2d(C_in, C_in, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            # Second depthwise conv
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            # Second pointwise conv
            nn.Conv2d(C_in, C_out, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class ConvBNReLU(nn.Module):
    """
    Standard convolution followed by batch normalization and ReLU activation.
    """
    def __init__(self, C_in, C_out, kernel, stride, padding, affine=True):
        super(ConvBNReLU, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel, stride, padding),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False)
        )
        
    def forward(self, x):
        return self.op(x)
    
class Identity(nn.Module):
    """
    Identity operation (returns input as is).
    """
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

class Zero(nn.Module):
    """
    Zero operation (returns zeros of the same shape as input, possibly downsampled).
    """
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        # Downsample by stride along the height dimension, then zero
        return x[:, :, ::self.stride, :].mul(0.)
    
class FactorizedReduce(nn.Module):
    """
    Reduces spatial dimension (height) by a factor of 2 and increases channels.
    """
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0  # Output channels must be even
        # 1x1 convolution with stride 2 for downsampling
        self.conv_1 = nn.Conv2d(C_in, C_out, 1, stride=(2, 1), padding=0)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv_1(x)  # Downsample and increase channels
        out = self.bn(out)    # Batch normalization
        out = self.relu(out)  # ReLU activation
        return out